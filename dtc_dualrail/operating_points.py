"""Locate maximally hybridized operating points of tunable dual-rail circuits."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr
from pathos.pools import ProcessPool
from scipy.optimize import linear_sum_assignment, minimize_scalar

from dtc_dualrail.constants import DATA_DIR
from dtc_dualrail.io import save_dataset

OPERATING_POINT_SCHEMA = "avoided-crossing-local-minimum-v3"


def _circuit_operator(circuit, name: str):
    operator = getattr(circuit, name, None)
    if callable(operator):
        return operator()

    operators_by_name = getattr(circuit, "operators_by_name", {})
    operator = operators_by_name.get(name)
    if callable(operator):
        return operator()
    raise AttributeError(f"circuit does not provide {name}")


def evaluate_rail_hybridization(
    circuit,
    phi_1: float,
    phi_2: float,
    *,
    flux_params: tuple[str, str] = ("Φ1", "Φ2"),
    evals_count: int = 6,
):
    """Evaluate rail mixing at one pair of external fluxes.

    The two rail-like one-excitation states are assigned from their ground-state
    transition strengths under the physical ``n1`` and ``n2`` charge operators.
    For each rail, its strength is normalized over those two dressed states.  At
    maximal hybridization both normalized rows are ``(1/2, 1/2)``.

    Returns a dictionary containing a signed ``polarization`` (zero at maximal
    mixing), a ``hybridization`` score in ``[0, 1]``, the mean rail transition
    frequency, the avoided-crossing splitting, and the two dressed level indices.
    """
    if evals_count < 3:
        raise ValueError("evals_count must be at least 3")
    if len(flux_params) != 2 or flux_params[0] == flux_params[1]:
        raise ValueError("flux_params must contain two distinct names")

    setattr(circuit, flux_params[0], float(phi_1))
    setattr(circuit, flux_params[1], float(phi_2))
    energies, eigenstates = circuit.eigensys(evals_count=evals_count)
    energies = np.asarray(energies, dtype=float)
    eigenstates = np.asarray(eigenstates)
    if energies.shape != (evals_count,):
        raise ValueError("circuit returned an unexpected number of eigenenergies")
    if eigenstates.ndim != 2 or eigenstates.shape[1] != evals_count:
        raise ValueError("circuit eigenstates must be columns of a 2D array")

    n1_operator = _circuit_operator(circuit, "n1_operator")
    n2_operator = _circuit_operator(circuit, "n2_operator")
    candidates = np.arange(1, evals_count)
    ground = eigenstates[:, 0]
    excited = eigenstates[:, candidates]
    strengths = np.vstack(
        (
            np.abs(ground.conj() @ (n1_operator @ excited)) ** 2,
            np.abs(ground.conj() @ (n2_operator @ excited)) ** 2,
        )
    )

    maxima = np.max(strengths, axis=1)
    if np.any(~np.isfinite(maxima)) or np.any(maxima <= 0):
        raise ValueError(
            "rail charge operators have no finite ground-state transitions"
        )

    # Assign one distinct dressed state to each physical rail.  The two selected
    # levels span the rail-like one-excitation manifold even at exact resonance,
    # where either individual assignment is otherwise ambiguous.
    normalized_strengths = strengths / maxima[:, None]
    rail_rows, candidate_cols = linear_sum_assignment(-normalized_strengths)
    if rail_rows.size != 2:
        raise ValueError("could not identify two distinct rail-like states")
    levels = np.sort(candidates[candidate_cols])
    selected_cols = np.searchsorted(candidates, levels)
    selected = strengths[:, selected_cols]
    totals = np.sum(selected, axis=1)
    if np.any(totals <= 0):
        raise ValueError("selected rail manifold has zero transition strength")

    participation = selected / totals[:, None]
    polarization = float(participation[0, 0] - participation[1, 0])
    hybridization = float(
        np.clip(1.0 - 2.0 * np.mean(np.abs(participation - 0.5)), 0.0, 1.0)
    )
    transitions = energies[levels] - energies[0]

    charge_matrix_element = np.sqrt(selected)
    two_excitation_frequency_ghz = None
    remaining = np.setdiff1d(candidates, levels)
    if remaining.size >= 3:
        parent_states = eigenstates[:, levels]
        candidate_states = eigenstates[:, remaining]
        two_excitation_score = np.zeros(remaining.size)
        for operator in (n1_operator, n2_operator):
            two_excitation_score += np.sum(
                np.abs(parent_states.conj().T @ operator @ candidate_states) ** 2,
                axis=0,
            )
        selected_two = remaining[np.argsort(two_excitation_score)[-3:]]
        selected_two.sort()
        two_excitation_frequency_ghz = energies[selected_two] - energies[0]

    return {
        "polarization": polarization,
        "hybridization": hybridization,
        "operating_frequency_ghz": float(np.mean(transitions)),
        "splitting_ghz": float(np.abs(np.diff(transitions))[0]),
        "rail_levels": levels,
        "rail_participation": participation,
        "charge_matrix_element": charge_matrix_element,
        "two_excitation_frequency_ghz": two_excitation_frequency_ghz,
    }


def _flux_values(values, name: str, *, minimum_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < minimum_size:
        raise ValueError(
            f"{name} must be a one-dimensional array with at least {minimum_size} values"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _find_operating_points_chunk(job):
    (
        circuit,
        phi_1_values,
        phi_2_values,
        flux_params,
        evals_count,
        min_hybridization,
        minimize_xtol,
        evaluator,
    ) = job
    original_flux = {
        name: getattr(circuit, name) for name in flux_params if hasattr(circuit, name)
    }
    cache = {}

    def evaluate(phi_1, phi_2):
        key = (float(phi_1), float(phi_2))
        if key not in cache:
            result = evaluator(
                circuit,
                key[0],
                key[1],
                flux_params=flux_params,
                evals_count=evals_count,
            )
            for required in (
                "hybridization",
                "operating_frequency_ghz",
                "splitting_ghz",
                "rail_levels",
            ):
                if required not in result:
                    raise ValueError(f"evaluator result is missing {required}")
            cache[key] = result
        return cache[key]

    found = []
    try:
        for phi_2 in phi_2_values:
            splitting = np.asarray(
                [evaluate(phi_1, phi_2)["splitting_ghz"] for phi_1 in phi_1_values],
                dtype=float,
            )
            if not np.all(np.isfinite(splitting)):
                raise ValueError("evaluator returned non-finite rail splitting")

            center = splitting[1:-1]
            local_minimum_indices = (
                np.flatnonzero(
                    (center <= splitting[:-2])
                    & (center <= splitting[2:])
                    & ((center < splitting[:-2]) | (center < splitting[2:]))
                )
                + 1
            )
            minima = []
            for idx in local_minimum_indices:
                left = float(phi_1_values[idx - 1])
                right = float(phi_1_values[idx + 1])
                optimization = minimize_scalar(
                    lambda phi_1, fixed_phi_2=phi_2: evaluate(phi_1, fixed_phi_2)[
                        "splitting_ghz"
                    ],
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": minimize_xtol},
                )
                if not optimization.success or not np.isfinite(optimization.fun):
                    raise RuntimeError(
                        "failed to refine rail-splitting minimum for "
                        f"{flux_params[1]}={float(phi_2)}"
                    )
                minima.append((float(optimization.x), int(optimization.nfev)))

            unique_minima = []
            for minimum in sorted(minima):
                if (
                    not unique_minima
                    or abs(minimum[0] - unique_minima[-1][0]) > 10 * minimize_xtol
                ):
                    unique_minima.append(minimum)

            for phi_1, minimizer_nfev in unique_minima:
                result = evaluate(phi_1, phi_2)
                hybridization = float(result["hybridization"])
                if not np.isfinite(hybridization) or hybridization < min_hybridization:
                    continue
                levels = np.asarray(result["rail_levels"], dtype=int)
                if levels.shape != (2,):
                    raise ValueError("evaluator rail_levels must contain two indices")
                optional_arrays = {}
                for name, shape in (
                    ("rail_participation", (2, 2)),
                    ("charge_matrix_element", (2, 2)),
                    ("two_excitation_frequency_ghz", (3,)),
                ):
                    value = result.get(name)
                    if value is None:
                        continue
                    value = np.asarray(value, dtype=float)
                    if value.shape != shape or not np.all(np.isfinite(value)):
                        raise ValueError(
                            f"evaluator {name} must be a finite array with shape {shape}"
                        )
                    optional_arrays[name] = value
                found.append(
                    {
                        "phi_1": phi_1,
                        "phi_2": float(phi_2),
                        "polarization": float(result.get("polarization", np.nan)),
                        "hybridization": hybridization,
                        "operating_frequency_ghz": float(
                            result["operating_frequency_ghz"]
                        ),
                        "splitting_ghz": float(result["splitting_ghz"]),
                        "rail_levels": levels,
                        "minimizer_nfev": minimizer_nfev,
                        **optional_arrays,
                    }
                )
    finally:
        for name, value in original_flux.items():
            setattr(circuit, name, value)
    return found


def find_operating_points(
    circuit,
    phi_1_values,
    phi_2_values,
    *,
    flux_params: tuple[str, str] = ("Φ1", "Φ2"),
    evals_count: int = 6,
    min_hybridization: float = 0.95,
    minimize_xtol: float = 1e-6,
    cpus: int = 1,
    evaluator: Callable = evaluate_rail_hybridization,
) -> xr.Dataset:
    """Find every maximally hybridized flux pair on selected ``phi_2`` slices.

    On every fixed ``phi_2`` slice, a coarse, strictly increasing
    ``phi_1_values`` grid identifies all interior local minima of the energy
    splitting between the two rail-like dressed states.  Each minimum is then
    refined by bounded scalar minimization.  Multiple minima on one slice are
    retained, including the symmetry-related avoided crossings of a SQUID
    tuning curve.  The polarization is reported only as a diagnostic and does
    not determine the operating point.

    With ``cpus > 1``, contiguous groups of ``phi_2`` slices are evaluated in
    separate ``pathos`` processes.  Each worker receives its own circuit copy,
    so the flux mutations performed by the evaluator remain process-local.

    The returned dataset has one row per accepted flux pair.  Points whose
    normalized rail mixing is below ``min_hybridization`` are rejected.
    """
    phi_1_values = _flux_values(phi_1_values, "phi_1_values", minimum_size=3)
    phi_2_values = _flux_values(phi_2_values, "phi_2_values", minimum_size=1)
    if len(flux_params) != 2 or flux_params[0] == flux_params[1]:
        raise ValueError("flux_params must contain two distinct names")
    if evals_count < 3:
        raise ValueError("evals_count must be at least 3")
    if not 0 <= min_hybridization <= 1:
        raise ValueError("min_hybridization must lie in [0, 1]")
    if not np.isfinite(minimize_xtol) or minimize_xtol <= 0:
        raise ValueError("minimize_xtol must be positive and finite")
    if not isinstance(cpus, (int, np.integer)) or isinstance(cpus, bool) or cpus < 1:
        raise ValueError("cpus must be an integer of at least 1")

    missing_flux_params = [name for name in flux_params if not hasattr(circuit, name)]
    if evaluator is evaluate_rail_hybridization and missing_flux_params:
        raise AttributeError(
            f"circuit does not provide flux parameters {missing_flux_params}"
        )

    worker_count = min(int(cpus), phi_2_values.size)
    phi_2_chunks = np.array_split(phi_2_values, worker_count)
    jobs = [
        (
            circuit,
            phi_1_values,
            chunk,
            flux_params,
            evals_count,
            min_hybridization,
            minimize_xtol,
            evaluator,
        )
        for chunk in phi_2_chunks
    ]
    if worker_count == 1:
        chunk_results = list(map(_find_operating_points_chunk, jobs))
    else:
        pool = ProcessPool(nodes=worker_count)
        try:
            chunk_results = pool.map(_find_operating_points_chunk, jobs)
        finally:
            pool.close()
            pool.join()
            pool.clear()
    found = [point for chunk in chunk_results for point in chunk]

    point_count = len(found)
    rail_levels = np.empty((point_count, 2), dtype=int)
    if found:
        rail_levels[:] = [point["rail_levels"] for point in found]

    data_vars = {
        "phi_1": (
            ("operating_point",),
            [point["phi_1"] for point in found],
        ),
        "phi_2": (
            ("operating_point",),
            [point["phi_2"] for point in found],
        ),
        "polarization": (
            ("operating_point",),
            [point["polarization"] for point in found],
        ),
        "hybridization": (
            ("operating_point",),
            [point["hybridization"] for point in found],
        ),
        "operating_frequency_ghz": (
            ("operating_point",),
            [point["operating_frequency_ghz"] for point in found],
        ),
        "splitting_ghz": (
            ("operating_point",),
            [point["splitting_ghz"] for point in found],
        ),
        "minimizer_nfev": (
            ("operating_point",),
            [point["minimizer_nfev"] for point in found],
        ),
        "rail_level": (("operating_point", "rail_state"), rail_levels),
    }
    optional_variables = {
        "rail_participation": (
            ("operating_point", "physical_rail", "rail_state"),
            (2, 2),
        ),
        "charge_matrix_element": (
            ("operating_point", "physical_rail", "rail_state"),
            (2, 2),
        ),
        "two_excitation_frequency_ghz": (
            ("operating_point", "two_excitation_state"),
            (3,),
        ),
    }
    for name, (dimensions, shape) in optional_variables.items():
        present = [name in point for point in found]
        if any(present) and not all(present):
            raise ValueError(
                f"evaluator returned {name} for only some operating points"
            )
        if all(present) and present:
            values = np.asarray([point[name] for point in found], dtype=float)
            if values.shape != (point_count, *shape):
                raise ValueError(f"evaluator returned inconsistent {name} shapes")
            data_vars[name] = (dimensions, values)

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "operating_point": np.arange(point_count),
            "rail_state": ["lower", "upper"],
            "physical_rail": ["rail_1", "rail_2"],
            "two_excitation_state": ["20", "11", "02"],
        },
        attrs={
            "description": "Maximally hybridized dual-rail operating points.",
            "method": "local minima of dressed rail energy splitting",
            "operating_point_schema": OPERATING_POINT_SCHEMA,
            "flux_param_1": flux_params[0],
            "flux_param_2": flux_params[1],
            "phi_2_slice_count": phi_2_values.size,
            "cpus": int(cpus),
            "worker_count": worker_count,
            "evals_count": evals_count,
            "min_hybridization": min_hybridization,
            "minimize_xtol": minimize_xtol,
        },
    )
    if found and all("charge_matrix_element" in point for point in found):
        dataset["charge_matrix_element"] = (
            ("operating_point", "physical_rail", "rail_state"),
            np.asarray([point["charge_matrix_element"] for point in found]),
        )
        dataset = dataset.assign_coords(physical_rail=["rail_1", "rail_2"])
    if found and all(
        point.get("two_excitation_frequency_ghz") is not None for point in found
    ):
        dataset["two_excitation_frequency_ghz"] = (
            ("operating_point", "two_excitation_state"),
            np.asarray([point["two_excitation_frequency_ghz"] for point in found]),
        )
        dataset = dataset.assign_coords(two_excitation_state=["low", "middle", "high"])
    dataset["phi_1"].attrs["units"] = "flux quantum"
    dataset["phi_2"].attrs["units"] = "flux quantum"
    dataset["operating_frequency_ghz"].attrs["units"] = "GHz"
    dataset["splitting_ghz"].attrs["units"] = "GHz"
    if "two_excitation_frequency_ghz" in dataset:
        dataset["two_excitation_frequency_ghz"].attrs["units"] = "GHz"
    return dataset


def load_or_compute_operating_points(
    circuit=None,
    phi_1_values=None,
    phi_2_values=None,
    *,
    path=DATA_DIR / "operating_points.nc",
    force: bool = False,
    input_fingerprint: str | None = None,
    **find_kwargs,
):
    """Load a validated operating-point dataset or compute and save it."""
    path = Path(path)
    if path.exists() and not force:
        dataset = xr.load_dataset(path)
        required = {
            "phi_1",
            "phi_2",
            "hybridization",
            "operating_frequency_ghz",
            "splitting_ghz",
            "rail_level",
        }
        missing = sorted(required - set(dataset.data_vars))
        if missing:
            dataset.close()
            raise ValueError(
                f"stale operating-point cache {path} is missing {missing}; "
                "rerun with force=True"
            )
        if dataset.attrs.get("operating_point_schema") != OPERATING_POINT_SCHEMA:
            dataset.close()
            raise ValueError(
                f"stale operating-point cache {path} uses an older schema; "
                "rerun with force=True"
            )
        if (
            input_fingerprint is not None
            and dataset.attrs.get("input_fingerprint") != input_fingerprint
        ):
            dataset.close()
            raise ValueError(
                f"operating-point cache {path} has different circuit or grid "
                "inputs; rerun with force=True"
            )
        return dataset, path

    if circuit is None or phi_1_values is None or phi_2_values is None:
        raise ValueError(
            "circuit, phi_1_values, and phi_2_values are required to compute "
            "operating points"
        )
    dataset = find_operating_points(
        circuit,
        phi_1_values,
        phi_2_values,
        **find_kwargs,
    )
    if input_fingerprint is not None:
        dataset.attrs["input_fingerprint"] = input_fingerprint
    return dataset, save_dataset(dataset, path)


__all__ = [
    "OPERATING_POINT_SCHEMA",
    "evaluate_rail_hybridization",
    "find_operating_points",
    "load_or_compute_operating_points",
]
