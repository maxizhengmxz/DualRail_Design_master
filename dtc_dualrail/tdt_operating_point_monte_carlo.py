"""Monte Carlo operating-point and dispersive analysis for TDT circuits."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr

from dtc_dualrail.circuits import (
    _circuit_factory_name,
    _validate_circuit_factory_params,
    gen_tdt,
)
from dtc_dualrail.constants import (
    BRANCH_LABELS,
    DATA_DIR,
    DEFAULT_PARAM_REL_SIGMA,
    TRACKING_SCHEMA,
)
from dtc_dualrail.io import save_dataset
from dtc_dualrail.operating_point_dispersive import (
    OPERATING_POINT_DISPERSIVE_SCHEMA,
    evaluate_dispersive_shifts_at_flux_pair,
    run_operating_point_dispersive_sweep,
)
from dtc_dualrail.operating_points import (
    OPERATING_POINT_SCHEMA,
    evaluate_rail_hybridization,
    find_operating_points,
)
from dtc_dualrail.sampling import (
    _correlation_groups_attr,
    _normalize_correlation_groups,
    _sample_parameters,
)

TDT_OPERATING_POINT_MONTE_CARLO_SCHEMA = "tdt-operating-point-monte-carlo-v1"


def _numeric_params(params: Mapping) -> dict[str, float]:
    if not isinstance(params, Mapping) or not params:
        raise ValueError("params must be a non-empty mapping")

    numeric = {name: float(value) for name, value in params.items()}
    nonfinite = [name for name, value in numeric.items() if not np.isfinite(value)]
    if nonfinite:
        raise ValueError(f"parameters must be finite: {nonfinite}")
    return numeric


def _empty_operating_point_result(dispersive_evals_count: int) -> xr.Dataset:
    """Create the result schema when no sample has an accepted operating point."""
    point_count = 0
    branch_count = len(BRANCH_LABELS)
    dataset = xr.Dataset(
        data_vars={
            "phi_1": ("operating_point", np.empty(point_count)),
            "phi_2": ("operating_point", np.empty(point_count)),
            "polarization": ("operating_point", np.empty(point_count)),
            "hybridization": ("operating_point", np.empty(point_count)),
            "operating_frequency_ghz": (
                "operating_point",
                np.empty(point_count),
            ),
            "splitting_ghz": ("operating_point", np.empty(point_count)),
            "minimizer_nfev": (
                "operating_point",
                np.empty(point_count, dtype=int),
            ),
            "rail_level": (
                ("operating_point", "rail_state"),
                np.empty((point_count, 2), dtype=int),
            ),
            "energy": (
                ("operating_point", "level"),
                np.empty((point_count, dispersive_evals_count)),
            ),
            "tracked_energy": (
                ("operating_point", "branch"),
                np.empty((point_count, branch_count)),
            ),
            "tracked_level": (
                ("operating_point", "branch"),
                np.empty((point_count, branch_count), dtype=int),
            ),
            "assignment_score": (
                ("operating_point", "branch"),
                np.empty((point_count, branch_count)),
            ),
            "chi_p": ("operating_point", np.empty(point_count)),
            "chi_m": ("operating_point", np.empty(point_count)),
            "delta_chi": ("operating_point", np.empty(point_count)),
            "abs_delta_chi_mhz": ("operating_point", np.empty(point_count)),
            "abs_delta_chi_over_chi_p": (
                "operating_point",
                np.empty(point_count),
            ),
        },
        coords={
            "operating_point": np.arange(point_count),
            "rail_state": ["lower", "upper"],
            "level": np.arange(dispersive_evals_count),
            "branch": BRANCH_LABELS,
        },
    )
    return dataset


def _set_result_units(dataset: xr.Dataset) -> None:
    dataset["phi_1"].attrs["units"] = "flux quantum"
    dataset["phi_2"].attrs["units"] = "flux quantum"
    for name in (
        "operating_frequency_ghz",
        "splitting_ghz",
        "energy",
        "tracked_energy",
        "chi_p",
        "chi_m",
        "delta_chi",
    ):
        dataset[name].attrs["units"] = "GHz"
    dataset["abs_delta_chi_mhz"].attrs["units"] = "MHz"
    dataset["abs_delta_chi_over_chi_p"].attrs["units"] = "dimensionless"


def run_tdt_operating_point_monte_carlo(
    params: Mapping,
    phi_1_values,
    phi_2_values,
    *,
    samples: int = 200,
    seed: int = 20260528,
    param_sigma=None,
    param_rel_sigma=None,
    param_correlation_groups=None,
    param_rel_min=0.01,
    circuit_factory=gen_tdt,
    flux_params: tuple[str, str] = ("Φ1", "Φ2"),
    operating_point_evals_count: int = 6,
    dispersive_evals_count: int = 24,
    min_hybridization: float = 0.95,
    minimize_xtol: float = 1e-6,
    cpus: int = 1,
    operating_point_evaluator=evaluate_rail_hybridization,
    dispersive_evaluator=evaluate_dispersive_shifts_at_flux_pair,
) -> xr.Dataset:
    """Sample TDT devices and evaluate every accepted dual-rail operating point.

    Each Monte Carlo sample receives a newly sampled device parameter set.
    ``find_operating_points`` locates the maximally hybridized ``(phi_1, phi_2)``
    pairs for that device, and ``run_operating_point_dispersive_sweep`` computes
    the spectrum, tracked levels, conditional dispersive shifts, and their
    difference at every accepted pair.

    Different samples can produce different numbers of operating points. The
    result therefore uses a flat ``operating_point`` dimension. Its
    ``sample_index`` coordinate identifies the originating Monte Carlo sample,
    while ``sample_operating_point`` is the point's local index within that
    sample.

    ``cpus`` is passed to each operating-point search and each dispersive sweep.
    Samples themselves run sequentially to avoid nested process pools.
    """
    params = _numeric_params(params)
    if not isinstance(samples, (int, np.integer)) or isinstance(samples, bool):
        raise ValueError("samples must be a positive integer")
    if samples < 1:
        raise ValueError("samples must be a positive integer")

    phi_1_values = np.asarray(phi_1_values, dtype=float)
    phi_2_values = np.asarray(phi_2_values, dtype=float)
    param_rel_sigma = (
        DEFAULT_PARAM_REL_SIGMA if param_rel_sigma is None else param_rel_sigma
    )
    _validate_circuit_factory_params(circuit_factory, [params])
    _normalize_correlation_groups(param_correlation_groups, params)

    param_names = list(params)
    param_values = np.full((samples, len(param_names)), np.nan)
    param_sigmas = np.full_like(param_values, np.nan)
    operating_point_count = np.zeros(samples, dtype=int)
    sample_results = []
    point_offset = 0

    child_seeds = np.random.SeedSequence(seed).spawn(samples)
    for sample_idx, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed.generate_state(1)[0])
        sampled_params, resolved_param_sigma = _sample_parameters(
            params,
            rng,
            param_sigma=param_sigma,
            param_rel_sigma=param_rel_sigma,
            param_correlation_groups=param_correlation_groups,
            param_rel_min=param_rel_min,
        )
        param_values[sample_idx] = [
            sampled_params[param_name] for param_name in param_names
        ]
        param_sigmas[sample_idx] = [
            resolved_param_sigma[param_name] for param_name in param_names
        ]

        circuit = circuit_factory(**sampled_params)
        operating_points = find_operating_points(
            circuit,
            phi_1_values,
            phi_2_values,
            flux_params=flux_params,
            evals_count=operating_point_evals_count,
            min_hybridization=min_hybridization,
            minimize_xtol=minimize_xtol,
            cpus=cpus,
            evaluator=operating_point_evaluator,
        )
        point_count = operating_points.sizes["operating_point"]
        operating_point_count[sample_idx] = point_count
        if point_count == 0:
            continue

        dispersive = run_operating_point_dispersive_sweep(
            operating_points,
            sampled_params,
            circuit_factory=circuit_factory,
            flux_params=flux_params,
            evals_count=dispersive_evals_count,
            cpus=cpus,
            evaluator=dispersive_evaluator,
        )
        dispersive = dispersive.drop_vars("param_value")
        dispersive = dispersive.assign_coords(
            operating_point=np.arange(point_offset, point_offset + point_count),
            sample_index=(
                "operating_point",
                np.full(point_count, sample_idx, dtype=int),
            ),
            sample_operating_point=(
                "operating_point",
                np.arange(point_count),
            ),
        )
        sample_results.append(dispersive)
        point_offset += point_count

    if sample_results:
        dataset = xr.concat(
            sample_results,
            dim="operating_point",
            data_vars="all",
            coords="minimal",
            compat="equals",
            combine_attrs="drop",
        )
    else:
        dataset = _empty_operating_point_result(dispersive_evals_count)
        dataset = dataset.assign_coords(
            sample_index=("operating_point", np.empty(0, dtype=int)),
            sample_operating_point=("operating_point", np.empty(0, dtype=int)),
        )

    dataset = dataset.assign_coords(
        sample=np.arange(samples),
        param=param_names,
        phi_1_search=phi_1_values,
        phi_2_search=phi_2_values,
    )
    dataset["param_value"] = (("sample", "param"), param_values)
    dataset["param_sigma"] = (("sample", "param"), param_sigmas)
    dataset["param_center"] = (("param",), [params[name] for name in param_names])
    dataset["operating_point_count"] = (("sample",), operating_point_count)
    dataset.attrs = {
        "description": (
            "Monte Carlo TDT device samples with independently located dual-rail "
            "operating points and dispersive shifts."
        ),
        "tdt_operating_point_monte_carlo_schema": (
            TDT_OPERATING_POINT_MONTE_CARLO_SCHEMA
        ),
        "operating_point_schema": OPERATING_POINT_SCHEMA,
        "operating_point_dispersive_schema": OPERATING_POINT_DISPERSIVE_SCHEMA,
        "tracking_schema": TRACKING_SCHEMA,
        "samples": int(samples),
        "seed": int(seed),
        "sigma_mode": (
            "absolute_with_relative_fallback" if param_sigma is not None else "relative"
        ),
        "param_correlation_groups": _correlation_groups_attr(
            param_correlation_groups,
            param_names,
        ),
        "param_rel_min": str(param_rel_min),
        "circuit_factory": _circuit_factory_name(circuit_factory),
        "flux_param_1": flux_params[0],
        "flux_param_2": flux_params[1],
        "operating_point_evals_count": int(operating_point_evals_count),
        "dispersive_evals_count": int(dispersive_evals_count),
        "min_hybridization": float(min_hybridization),
        "minimize_xtol": float(minimize_xtol),
        "cpus": int(cpus),
    }
    _set_result_units(dataset)
    return dataset


def load_or_compute_tdt_operating_point_monte_carlo(
    params: Mapping | None = None,
    phi_1_values=None,
    phi_2_values=None,
    *,
    path=DATA_DIR / "tdt_operating_point_monte_carlo.nc",
    force: bool = False,
    **run_kwargs,
):
    """Load a validated combined TDT Monte Carlo result or compute and save it."""
    path = Path(path)
    if path.exists() and not force:
        dataset = xr.load_dataset(path)
        required = {
            "phi_1",
            "phi_2",
            "energy",
            "tracked_energy",
            "chi_p",
            "chi_m",
            "delta_chi",
            "abs_delta_chi_mhz",
            "abs_delta_chi_over_chi_p",
            "param_value",
            "param_sigma",
            "operating_point_count",
        }
        missing = sorted(required - set(dataset.data_vars))
        if missing:
            dataset.close()
            raise ValueError(
                f"stale TDT operating-point Monte Carlo cache {path} is missing "
                f"{missing}; rerun with force=True"
            )
        if (
            dataset.attrs.get("tdt_operating_point_monte_carlo_schema")
            != TDT_OPERATING_POINT_MONTE_CARLO_SCHEMA
            or dataset.attrs.get("operating_point_schema") != OPERATING_POINT_SCHEMA
            or dataset.attrs.get("operating_point_dispersive_schema")
            != OPERATING_POINT_DISPERSIVE_SCHEMA
            or dataset.attrs.get("tracking_schema") != TRACKING_SCHEMA
        ):
            dataset.close()
            raise ValueError(
                f"stale TDT operating-point Monte Carlo cache {path} uses an "
                "older schema; rerun with force=True"
            )
        return dataset, path

    if params is None or phi_1_values is None or phi_2_values is None:
        raise ValueError(
            "params, phi_1_values, and phi_2_values are required to compute "
            "the TDT operating-point Monte Carlo result"
        )
    dataset = run_tdt_operating_point_monte_carlo(
        params,
        phi_1_values,
        phi_2_values,
        **run_kwargs,
    )
    return dataset, save_dataset(dataset, path)


__all__ = [
    "TDT_OPERATING_POINT_MONTE_CARLO_SCHEMA",
    "load_or_compute_tdt_operating_point_monte_carlo",
    "run_tdt_operating_point_monte_carlo",
]
