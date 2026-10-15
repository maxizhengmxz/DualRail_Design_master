"""Dispersive-shift evaluation at dual-rail operating points."""

from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path

import numpy as np
import xarray as xr
from pathos.pools import ProcessPool

from dtc_dualrail.circuits import (
    _circuit_factory_name,
    _validate_circuit_factory_params,
    gen_tdt,
)
from dtc_dualrail.constants import (
    BRANCH_INDEX,
    BRANCH_LABELS,
    DATA_DIR,
    TRACKING_SCHEMA,
)
from dtc_dualrail.io import save_dataset
from dtc_dualrail.operating_points import OPERATING_POINT_SCHEMA
from dtc_dualrail.tracking import track_levels_by_overlap

OPERATING_POINT_DISPERSIVE_SCHEMA = "operating-point-dispersive-v1"


def _abs_delta_chi_over_chi_p(chi_p: float, chi_m: float) -> float:
    if not np.isfinite(chi_p) or not np.isfinite(chi_m) or chi_p == 0.0:
        return np.inf
    return float(np.abs((chi_p - chi_m) / chi_p))


def _validate_operating_points(operating_points: xr.Dataset) -> int:
    if not isinstance(operating_points, xr.Dataset):
        raise TypeError("operating_points must be an xarray.Dataset")
    required = {"phi_1", "phi_2"}
    missing = sorted(required - set(operating_points.data_vars))
    if missing:
        raise ValueError(f"operating_points is missing {missing}")
    if operating_points.attrs.get("operating_point_schema") != OPERATING_POINT_SCHEMA:
        raise ValueError("operating_points uses an unsupported operating-point schema")
    if "operating_point" not in operating_points.sizes:
        raise ValueError("operating_points must contain an operating_point dimension")

    point_count = operating_points.sizes["operating_point"]
    if point_count < 1:
        raise ValueError("operating_points must contain at least one point")
    for name in ("phi_1", "phi_2"):
        values = operating_points[name]
        if values.dims != ("operating_point",):
            raise ValueError(f"operating_points {name} must depend on operating_point")
        if not np.all(np.isfinite(values.values)):
            raise ValueError(f"operating_points {name} must contain finite values")
    return point_count


def _operating_point_fingerprint(operating_points: xr.Dataset) -> str:
    digest = sha256()
    for name in ("phi_1", "phi_2"):
        values = np.ascontiguousarray(operating_points[name].values, dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _numeric_params(params: Mapping) -> dict:
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    numeric = {}
    for name, value in params.items():
        numeric[name] = float(value)
        if not np.isfinite(numeric[name]):
            raise ValueError(f"parameter {name} must be finite")
    return numeric


def _param_fingerprint(params: Mapping) -> str:
    digest = sha256()
    for name in sorted(params):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(float(params[name]), dtype="<f8").tobytes())
    return digest.hexdigest()


def evaluate_dispersive_shifts_at_flux_pair(
    circuit,
    phi_1: float,
    phi_2: float,
    *,
    flux_params: tuple[str, str] = ("Φ1", "Φ2"),
    evals_count: int = 24,
):
    """Evaluate dressed energies and conditional resonator shifts at one pair."""
    if evals_count < len(BRANCH_LABELS):
        raise ValueError(f"evals_count must be at least {len(BRANCH_LABELS)}")
    if len(flux_params) != 2 or flux_params[0] == flux_params[1]:
        raise ValueError("flux_params must contain two distinct names")

    setattr(circuit, flux_params[0], float(phi_1))
    setattr(circuit, flux_params[1], float(phi_2))
    energies, states = circuit.eigensys(evals_count=evals_count)
    tracked_level, assignment_score, tracked_energy = track_levels_by_overlap(
        states,
        energies,
        q3_operator=circuit.Q3_operator(),
        n1_operator=circuit.n1_operator(),
        n2_operator=circuit.n2_operator(),
    )
    tracked_energy = tracked_energy[0]
    tracked_level = tracked_level[0]
    assignment_score = assignment_score[0]

    ground = tracked_energy[BRANCH_INDEX["g"]]
    plus = tracked_energy[BRANCH_INDEX["p"]]
    minus = tracked_energy[BRANCH_INDEX["m"]]
    resonator = tracked_energy[BRANCH_INDEX["r"]]
    plus_resonator = tracked_energy[BRANCH_INDEX["pr"]]
    minus_resonator = tracked_energy[BRANCH_INDEX["mr"]]
    chi_p = plus_resonator - resonator - plus + ground
    chi_m = minus_resonator - resonator - minus + ground

    return {
        "energy": np.asarray(energies),
        "tracked_energy": tracked_energy,
        "tracked_level": tracked_level,
        "assignment_score": assignment_score,
        "chi_p": float(chi_p),
        "chi_m": float(chi_m),
        "delta_chi": float(chi_p - chi_m),
        "abs_delta_chi_over_chi_p": _abs_delta_chi_over_chi_p(chi_p, chi_m),
    }


def _dispersive_chunk(job):
    (
        point_indices,
        phi_1,
        phi_2,
        params,
        circuit_factory,
        flux_params,
        evals_count,
        evaluator,
    ) = job
    circuit = circuit_factory(**params)
    results = []
    for point_idx in point_indices:
        result = evaluator(
            circuit,
            phi_1[point_idx],
            phi_2[point_idx],
            flux_params=flux_params,
            evals_count=evals_count,
        )
        required = {
            "energy",
            "tracked_energy",
            "tracked_level",
            "assignment_score",
            "chi_p",
            "chi_m",
            "delta_chi",
            "abs_delta_chi_over_chi_p",
        }
        missing = sorted(required - set(result))
        if missing:
            raise ValueError(f"dispersive evaluator result is missing {missing}")
        results.append({"point_idx": int(point_idx), **result})
    return results


def run_operating_point_dispersive_sweep(
    operating_points: xr.Dataset,
    params: Mapping,
    *,
    circuit_factory=gen_tdt,
    flux_params: tuple[str, str] = ("Φ1", "Φ2"),
    evals_count: int = 24,
    cpus: int = 1,
    evaluator: Callable = evaluate_dispersive_shifts_at_flux_pair,
) -> xr.Dataset:
    """Evaluate dispersive shifts at every supplied operating-point flux pair."""
    point_count = _validate_operating_points(operating_points)
    params = _numeric_params(params)
    _validate_circuit_factory_params(circuit_factory, [params])
    if len(flux_params) != 2 or flux_params[0] == flux_params[1]:
        raise ValueError("flux_params must contain two distinct names")
    if evals_count < len(BRANCH_LABELS):
        raise ValueError(f"evals_count must be at least {len(BRANCH_LABELS)}")
    if not isinstance(cpus, (int, np.integer)) or isinstance(cpus, bool) or cpus < 1:
        raise ValueError("cpus must be an integer of at least 1")

    reserved = {
        "energy",
        "tracked_energy",
        "tracked_level",
        "assignment_score",
        "chi_p",
        "chi_m",
        "delta_chi",
        "abs_delta_chi_mhz",
        "abs_delta_chi_over_chi_p",
        "param_value",
    }
    conflicts = sorted(reserved & set(operating_points.data_vars))
    if conflicts:
        raise ValueError(
            f"operating_points already contains computed variables {conflicts}"
        )

    phi_1 = np.asarray(operating_points["phi_1"].values, dtype=float)
    phi_2 = np.asarray(operating_points["phi_2"].values, dtype=float)
    worker_count = min(int(cpus), point_count)
    point_chunks = np.array_split(np.arange(point_count), worker_count)
    jobs = [
        (
            chunk,
            phi_1,
            phi_2,
            params,
            circuit_factory,
            flux_params,
            evals_count,
            evaluator,
        )
        for chunk in point_chunks
    ]
    if worker_count == 1:
        chunk_results = list(map(_dispersive_chunk, jobs))
    else:
        pool = ProcessPool(nodes=worker_count)
        try:
            chunk_results = pool.map(_dispersive_chunk, jobs)
        finally:
            pool.close()
            pool.join()
            pool.clear()
    results = sorted(
        (result for chunk in chunk_results for result in chunk),
        key=lambda result: result["point_idx"],
    )

    energy = np.asarray([result["energy"] for result in results], dtype=float)
    tracked_energy = np.asarray(
        [result["tracked_energy"] for result in results], dtype=float
    )
    tracked_level = np.asarray(
        [result["tracked_level"] for result in results], dtype=int
    )
    assignment_score = np.asarray(
        [result["assignment_score"] for result in results], dtype=float
    )
    expected_branch_shape = (point_count, len(BRANCH_LABELS))
    if energy.shape != (point_count, evals_count):
        raise ValueError("dispersive evaluator energy shape does not match evals_count")
    for name, values in (
        ("tracked_energy", tracked_energy),
        ("tracked_level", tracked_level),
        ("assignment_score", assignment_score),
    ):
        if values.shape != expected_branch_shape:
            raise ValueError(
                f"dispersive evaluator {name} shape does not match tracked branches"
            )

    chi_p = np.asarray([result["chi_p"] for result in results], dtype=float)
    chi_m = np.asarray([result["chi_m"] for result in results], dtype=float)
    delta_chi = np.asarray([result["delta_chi"] for result in results], dtype=float)
    objective = np.asarray(
        [result["abs_delta_chi_over_chi_p"] for result in results], dtype=float
    )

    dataset = operating_points.copy(deep=True)
    dataset = dataset.assign_coords(
        level=np.arange(evals_count),
        branch=BRANCH_LABELS,
        param=list(params),
    )
    dataset["energy"] = (("operating_point", "level"), energy)
    dataset["tracked_energy"] = (
        ("operating_point", "branch"),
        tracked_energy,
    )
    dataset["tracked_level"] = (("operating_point", "branch"), tracked_level)
    dataset["assignment_score"] = (
        ("operating_point", "branch"),
        assignment_score,
    )
    dataset["chi_p"] = (("operating_point",), chi_p)
    dataset["chi_m"] = (("operating_point",), chi_m)
    dataset["delta_chi"] = (("operating_point",), delta_chi)
    dataset["abs_delta_chi_mhz"] = (
        ("operating_point",),
        1e3 * np.abs(delta_chi),
    )
    dataset["abs_delta_chi_over_chi_p"] = (("operating_point",), objective)
    dataset["param_value"] = (("param",), [params[name] for name in params])
    dataset.attrs.update(
        {
            "description": "Dispersive shifts at dual-rail operating points.",
            "operating_point_dispersive_schema": OPERATING_POINT_DISPERSIVE_SCHEMA,
            "source_operating_point_schema": OPERATING_POINT_SCHEMA,
            "operating_point_fingerprint": _operating_point_fingerprint(
                operating_points
            ),
            "param_fingerprint": _param_fingerprint(params),
            "circuit_factory": _circuit_factory_name(circuit_factory),
            "flux_param_1": flux_params[0],
            "flux_param_2": flux_params[1],
            "evals_count": evals_count,
            "cpus": int(cpus),
            "worker_count": worker_count,
            "tracking_schema": TRACKING_SCHEMA,
        }
    )
    for name in (
        "energy",
        "tracked_energy",
        "chi_p",
        "chi_m",
        "delta_chi",
    ):
        dataset[name].attrs["units"] = "GHz"
    dataset["abs_delta_chi_mhz"].attrs["units"] = "MHz"
    dataset["abs_delta_chi_over_chi_p"].attrs["units"] = "dimensionless"
    return dataset


def load_or_compute_operating_point_dispersive_sweep(
    operating_points: xr.Dataset | None = None,
    params: Mapping | None = None,
    *,
    path=DATA_DIR / "operating_point_dispersive_sweep.nc",
    force: bool = False,
    circuit_factory=gen_tdt,
    flux_params: tuple[str, str] = ("Φ1", "Φ2"),
    evals_count: int = 24,
    cpus: int = 1,
    evaluator: Callable = evaluate_dispersive_shifts_at_flux_pair,
):
    """Load a validated dispersive sweep or compute and save it."""
    path = Path(path)
    if path.exists() and not force:
        dataset = xr.load_dataset(path)
        required = {
            "energy",
            "tracked_energy",
            "tracked_level",
            "assignment_score",
            "chi_p",
            "chi_m",
            "delta_chi",
            "abs_delta_chi_mhz",
            "abs_delta_chi_over_chi_p",
            "param_value",
        }
        missing = sorted(required - set(dataset.data_vars))
        if missing:
            dataset.close()
            raise ValueError(
                f"stale operating-point dispersive cache {path} is missing "
                f"{missing}; rerun with force=True"
            )
        if (
            dataset.attrs.get("operating_point_dispersive_schema")
            != OPERATING_POINT_DISPERSIVE_SCHEMA
            or dataset.attrs.get("tracking_schema") != TRACKING_SCHEMA
        ):
            dataset.close()
            raise ValueError(
                f"stale operating-point dispersive cache {path} uses an older "
                "schema; rerun with force=True"
            )
        if operating_points is not None:
            _validate_operating_points(operating_points)
            expected = _operating_point_fingerprint(operating_points)
            if dataset.attrs.get("operating_point_fingerprint") != expected:
                dataset.close()
                raise ValueError(
                    f"operating-point dispersive cache {path} was computed for "
                    "different flux pairs; rerun with force=True"
                )
        if params is not None:
            params = _numeric_params(params)
            if dataset.attrs.get("param_fingerprint") != _param_fingerprint(params):
                dataset.close()
                raise ValueError(
                    f"operating-point dispersive cache {path} was computed for "
                    "different circuit parameters; rerun with force=True"
                )
            expected_factory = _circuit_factory_name(circuit_factory)
            if dataset.attrs.get("circuit_factory") != expected_factory:
                dataset.close()
                raise ValueError(
                    f"operating-point dispersive cache {path} used a different "
                    "circuit factory; rerun with force=True"
                )
        return dataset, path

    if operating_points is None or params is None:
        raise ValueError(
            "operating_points and params are required to compute dispersive shifts"
        )
    dataset = run_operating_point_dispersive_sweep(
        operating_points,
        params,
        circuit_factory=circuit_factory,
        flux_params=flux_params,
        evals_count=evals_count,
        cpus=cpus,
        evaluator=evaluator,
    )
    return dataset, save_dataset(dataset, path)


__all__ = [
    "OPERATING_POINT_DISPERSIVE_SCHEMA",
    "evaluate_dispersive_shifts_at_flux_pair",
    "load_or_compute_operating_point_dispersive_sweep",
    "run_operating_point_dispersive_sweep",
]
