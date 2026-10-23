"""Tunable resonator evaluation, optimization, and Monte Carlo helpers."""

from pathlib import Path

import numpy as np
import xarray as xr
from pathos.pools import ProcessPool
from scipy.optimize import minimize_scalar

from core.circuits import (
    _circuit_factory_name,
    _validate_circuit_factory_params,
    gen_dtc,
)
from core.constants import (
    BRANCH_INDEX,
    BRANCH_LABELS,
    DATA_DIR,
    DEFAULT_PARAM_REL_SIGMA,
    FLUX_PARAM,
    TRACKING_SCHEMA,
)
from core.io import save_dataset
from legacy.analysis.sampling import (
    _correlation_groups_attr,
    _normalize_correlation_groups,
    _sample_parameters,
)
from core.tracking import track_levels_by_overlap


def _abs_delta_chi_over_chi_p(chi_p, chi_m):
    if not np.isfinite(chi_p) or not np.isfinite(chi_m) or chi_p == 0.0:
        return np.inf
    return float(np.abs((chi_p - chi_m) / chi_p))


def evaluate_dispersive_shifts(
    params,
    *,
    circuit_factory=gen_dtc,
    flux=None,
    flux_param=FLUX_PARAM,
    evals_count=24,
):
    if evals_count < len(BRANCH_LABELS):
        raise ValueError(f"evals_count must be at least {len(BRANCH_LABELS)}")

    circuit = circuit_factory(**params)
    if flux_param is not None:
        if flux is None:
            raise ValueError("flux is required when flux_param is set")
        setattr(circuit, flux_param, flux)

    energies, states = circuit.eigensys(evals_count=evals_count)
    tracked_level, assignment_score, tracked_energy = track_levels_by_overlap(
        states,
        energies,
        q3_operator=circuit.Q3_operator(),
        n1_operator=circuit.n1_operator(),
        n2_operator=circuit.n2_operator(),
    )
    ground = tracked_energy[0, BRANCH_INDEX["g"]]
    plus = tracked_energy[0, BRANCH_INDEX["p"]]
    minus = tracked_energy[0, BRANCH_INDEX["m"]]
    resonator = tracked_energy[0, BRANCH_INDEX["r"]]
    plus_resonator = tracked_energy[0, BRANCH_INDEX["pr"]]
    minus_resonator = tracked_energy[0, BRANCH_INDEX["mr"]]
    chi_p = plus_resonator - resonator - plus + ground
    chi_m = minus_resonator - resonator - minus + ground

    return {
        "energy": np.asarray(energies),
        "tracked_energy": tracked_energy[0],
        "tracked_level": tracked_level[0],
        "assignment_score": assignment_score[0],
        "chi_p": float(chi_p),
        "chi_m": float(chi_m),
        "delta_chi": float(chi_p - chi_m),
        "objective": _abs_delta_chi_over_chi_p(chi_p, chi_m),
    }


def optimize_resonator_inductance(
    params,
    *,
    circuit_factory=gen_dtc,
    flux=None,
    flux_param=FLUX_PARAM,
    evals_count=24,
    lr_bounds=(0.8, 2.5),
    grid_size=21,
    xatol=1e-5,
    evaluator=evaluate_dispersive_shifts,
):
    lower, upper = map(float, lr_bounds)
    if not lower < upper:
        raise ValueError("lr_bounds must be strictly increasing")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    if "Cr" not in params or "Lr" not in params:
        raise ValueError("params must contain Cr and Lr")

    evaluations = []

    def evaluate(lr):
        trial_params = dict(params)
        trial_params["Lr"] = float(lr)
        result = evaluator(
            trial_params,
            circuit_factory=circuit_factory,
            flux=flux,
            flux_param=flux_param,
            evals_count=evals_count,
        )
        objective = float(result.get("objective", np.inf))
        evaluations.append((float(lr), result, objective))
        return objective

    grid = np.linspace(lower, upper, grid_size)
    grid_objective = np.asarray([evaluate(lr) for lr in grid])
    finite = np.isfinite(grid_objective)
    if not finite.any():
        return {
            "success": False,
            "optimal_lr": np.nan,
            "boundary_hit": False,
            "nfev": len(evaluations),
        }

    best_idx = int(np.nanargmin(np.where(finite, grid_objective, np.nan)))
    left = grid[max(0, best_idx - 1)]
    right = grid[min(grid_size - 1, best_idx + 1)]
    if left < right:
        try:
            minimize_scalar(
                evaluate,
                bounds=(left, right),
                method="bounded",
                options={"xatol": xatol},
            )
        except (ArithmeticError, ValueError):
            pass

    finite_results = [item for item in evaluations if np.isfinite(item[2])]
    optimal_lr, best, _ = min(finite_results, key=lambda item: item[2])
    return {
        **best,
        "success": True,
        "optimal_lr": optimal_lr,
        "boundary_hit": bool(
            np.isclose(optimal_lr, lower, atol=xatol, rtol=0.0)
            or np.isclose(optimal_lr, upper, atol=xatol, rtol=0.0)
        ),
        "nfev": len(evaluations),
    }


def _tunable_resonator_sweep_case(job):
    (
        sampled_params,
        lr,
        circuit_factory,
        flux,
        flux_param,
        evals_count,
        evaluator,
    ) = job
    sweep_params = dict(sampled_params)
    sweep_params["Lr"] = float(lr)
    return evaluator(
        sweep_params,
        circuit_factory=circuit_factory,
        flux=flux,
        flux_param=flux_param,
        evals_count=evals_count,
    )


def run_tunable_resonator_sample_sweep(
    params,
    lr_values,
    *,
    seed=20260528,
    evals_count=24,
    cpus=8,
    param_sigma=None,
    param_rel_sigma=None,
    param_correlation_groups=None,
    param_rel_min=0.01,
    circuit_factory=gen_dtc,
    flux=None,
    flux_param=FLUX_PARAM,
    evaluator=evaluate_dispersive_shifts,
):
    lr_values = np.asarray(lr_values, dtype=float)
    if lr_values.ndim != 1 or lr_values.size == 0:
        raise ValueError("lr_values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(lr_values)) or np.any(lr_values <= 0):
        raise ValueError("lr_values must contain only positive finite values")
    if "Cr" not in params or "Lr" not in params:
        raise ValueError("params must contain Cr and Lr")
    if cpus < 1:
        raise ValueError("cpus must be at least 1")

    param_names = list(params)
    param_rel_sigma = (
        DEFAULT_PARAM_REL_SIGMA if param_rel_sigma is None else param_rel_sigma
    )
    _validate_circuit_factory_params(circuit_factory, [params])
    _normalize_correlation_groups(param_correlation_groups, param_names)
    sampled_params, resolved_param_sigma = _sample_parameters(
        params,
        np.random.default_rng(seed),
        param_sigma=param_sigma,
        param_rel_sigma=param_rel_sigma,
        param_correlation_groups=param_correlation_groups,
        param_rel_min=param_rel_min,
        frozen_params={"Cr", "Lr"},
    )
    sampled_cr = float(sampled_params["Cr"])
    if not np.isfinite(sampled_cr) or sampled_cr <= 0:
        raise ValueError("sampled Cr must be positive and finite")

    jobs = [
        (
            sampled_params,
            lr,
            circuit_factory,
            flux,
            flux_param,
            evals_count,
            evaluator,
        )
        for lr in lr_values
    ]
    if cpus == 1:
        results = list(map(_tunable_resonator_sweep_case, jobs))
    else:
        pool = ProcessPool(nodes=cpus)
        try:
            results = pool.map(_tunable_resonator_sweep_case, jobs)
        finally:
            pool.close()
            pool.join()
            pool.clear()

    energy = np.asarray([result["energy"] for result in results])
    tracked_energy = np.asarray([result["tracked_energy"] for result in results])
    tracked_level = np.asarray([result["tracked_level"] for result in results])
    assignment_score = np.asarray([result["assignment_score"] for result in results])
    chi_p = np.asarray([result["chi_p"] for result in results])
    chi_m = np.asarray([result["chi_m"] for result in results])
    delta_chi = np.asarray([result["delta_chi"] for result in results])
    resonator_frequency_ghz = (
        1 / np.sqrt(lr_values * 1e-9 * sampled_cr * 1e-15) / (2 * np.pi) / 1e9
    )

    ds = xr.Dataset(
        data_vars={
            "resonator_frequency_ghz": (("lr",), resonator_frequency_ghz),
            "chi_p": (("lr",), chi_p),
            "chi_m": (("lr",), chi_m),
            "delta_chi": (("lr",), delta_chi),
            "abs_delta_chi_mhz": (("lr",), 1e3 * np.abs(delta_chi)),
            "energy": (("lr", "level"), energy),
            "tracked_energy": (("lr", "branch"), tracked_energy),
            "tracked_level": (("lr", "branch"), tracked_level),
            "assignment_score": (("lr", "branch"), assignment_score),
            "sampled_param": (
                ("param",),
                [sampled_params[name] for name in param_names],
            ),
            "param_sigma": (
                ("param",),
                [resolved_param_sigma[name] for name in param_names],
            ),
        },
        coords={
            "lr": lr_values,
            "level": np.arange(energy.shape[1]),
            "branch": BRANCH_LABELS[: tracked_energy.shape[1]],
            "param": param_names,
        },
        attrs={
            "description": "One Monte Carlo sample swept over readout inductance.",
            "seed": seed,
            "cpus": cpus,
            "circuit_factory": _circuit_factory_name(circuit_factory),
            "frozen_params": "Cr,Lr",
            "flux_param": "" if flux_param is None else flux_param,
            "flux_value": np.nan if flux is None else flux,
            "param_correlation_groups": _correlation_groups_attr(
                param_correlation_groups, param_names
            ),
            "param_rel_min": str(param_rel_min),
            "tracking_schema": TRACKING_SCHEMA,
        },
    )
    ds["lr"].attrs["units"] = "nH"
    for name in (
        "resonator_frequency_ghz",
        "chi_p",
        "chi_m",
        "delta_chi",
        "energy",
        "tracked_energy",
    ):
        ds[name].attrs["units"] = "GHz"
    ds["abs_delta_chi_mhz"].attrs["units"] = "MHz"
    return ds


def _tunable_resonator_case(job):
    (
        poi_idx,
        sample_idx,
        poi_name,
        flux,
        params,
        seed,
        param_sigma,
        param_rel_sigma,
        param_correlation_groups,
        param_rel_min,
        circuit_factory,
        flux_param,
        evals_count,
        lr_bounds,
        grid_size,
        xatol,
        optimizer,
        evaluator,
    ) = job
    sampled_params, resolved_param_sigma = _sample_parameters(
        params,
        np.random.default_rng(seed),
        param_sigma=param_sigma,
        param_rel_sigma=param_rel_sigma,
        param_correlation_groups=param_correlation_groups,
        param_rel_min=param_rel_min,
        frozen_params={"Cr", "Lr"},
    )
    original = evaluator(
        sampled_params,
        circuit_factory=circuit_factory,
        flux=flux,
        flux_param=flux_param,
        evals_count=evals_count,
    )
    optimized = optimizer(
        sampled_params,
        circuit_factory=circuit_factory,
        flux=flux,
        flux_param=flux_param,
        evals_count=evals_count,
        lr_bounds=lr_bounds,
        grid_size=grid_size,
        xatol=xatol,
    )
    return {
        "poi_idx": poi_idx,
        "sample_idx": sample_idx,
        "poi_name": poi_name,
        "params": sampled_params,
        "param_sigma": resolved_param_sigma,
        "original_chi_p": original["chi_p"],
        "original_chi_m": original["chi_m"],
        "original_delta_chi": original["delta_chi"],
        **optimized,
    }


def run_tunable_resonator_monte_carlo(
    poi_list,
    *,
    samples=200,
    cpus=8,
    seed=20260528,
    evals_count=24,
    param_sigma=None,
    param_rel_sigma=None,
    param_correlation_groups=None,
    param_rel_min=0.01,
    circuit_factory=gen_dtc,
    flux_param=FLUX_PARAM,
    lr_bounds=(0.8, 2.5),
    grid_size=21,
    xatol=1e-5,
    optimizer=optimize_resonator_inductance,
    evaluator=evaluate_dispersive_shifts,
):
    if not poi_list:
        raise ValueError("poi_list must contain at least one point of interest")
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if cpus < 1:
        raise ValueError("cpus must be at least 1")

    param_names = list(poi_list[0]["params"])
    if "Cr" not in param_names or "Lr" not in param_names:
        raise ValueError("point-of-interest params must contain Cr and Lr")
    for poi in poi_list:
        if list(poi["params"]) != param_names:
            raise ValueError(
                "all points of interest must use identical parameter names"
            )

    param_rel_sigma = (
        DEFAULT_PARAM_REL_SIGMA if param_rel_sigma is None else param_rel_sigma
    )
    _validate_circuit_factory_params(
        circuit_factory, [poi["params"] for poi in poi_list]
    )
    _normalize_correlation_groups(param_correlation_groups, param_names)

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(poi_list) * samples)
    jobs = []
    seed_idx = 0
    for poi_idx, poi in enumerate(poi_list):
        for sample_idx in range(samples):
            jobs.append(
                (
                    poi_idx,
                    sample_idx,
                    poi["name"],
                    poi.get("flux"),
                    poi["params"],
                    child_seeds[seed_idx].generate_state(1)[0],
                    param_sigma,
                    param_rel_sigma,
                    param_correlation_groups,
                    param_rel_min,
                    circuit_factory,
                    flux_param,
                    evals_count,
                    lr_bounds,
                    grid_size,
                    xatol,
                    optimizer,
                    evaluator,
                )
            )
            seed_idx += 1

    if cpus == 1:
        results = list(map(_tunable_resonator_case, jobs))
    else:
        with ProcessPool(nodes=cpus) as pool:
            results = pool.map(_tunable_resonator_case, jobs)

    poi_count = len(poi_list)
    shape = (poi_count, samples)
    param_values = np.full(shape + (len(param_names),), np.nan)
    param_sigmas = np.full_like(param_values, np.nan)
    energy = np.full(shape + (evals_count,), np.nan)
    branch_count = len(BRANCH_LABELS)
    tracked_energy = np.full(shape + (branch_count,), np.nan)
    tracked_level = np.full(shape + (branch_count,), -1, dtype=int)
    assignment_score = np.full(shape + (branch_count,), np.nan)
    optimal_lr = np.full(shape, np.nan)
    original_chi_p = np.full(shape, np.nan)
    original_chi_m = np.full(shape, np.nan)
    original_delta_chi = np.full(shape, np.nan)
    chi_p = np.full(shape, np.nan)
    chi_m = np.full(shape, np.nan)
    delta_chi = np.full(shape, np.nan)
    objective = np.full(shape, np.nan)
    success = np.zeros(shape, dtype=bool)
    nfev = np.zeros(shape, dtype=int)
    boundary_hit = np.zeros(shape, dtype=bool)

    for result in results:
        idx = (result["poi_idx"], result["sample_idx"])
        param_values[idx] = [result["params"][name] for name in param_names]
        param_sigmas[idx] = [result["param_sigma"][name] for name in param_names]
        success[idx] = bool(result.get("success", False))
        nfev[idx] = int(result.get("nfev", 0))
        boundary_hit[idx] = bool(result.get("boundary_hit", False))
        optimal_lr[idx] = result.get("optimal_lr", np.nan)
        original_chi_p[idx] = result["original_chi_p"]
        original_chi_m[idx] = result["original_chi_m"]
        original_delta_chi[idx] = result["original_delta_chi"]
        if not success[idx]:
            continue
        energy[idx] = result["energy"]
        tracked_energy[idx] = result["tracked_energy"]
        tracked_level[idx] = result["tracked_level"]
        assignment_score[idx] = result["assignment_score"]
        chi_p[idx] = result["chi_p"]
        chi_m[idx] = result["chi_m"]
        delta_chi[idx] = result["delta_chi"]
        objective[idx] = result["objective"]

    sampled_cr = param_values[:, :, param_names.index("Cr")]
    resonator_frequency_ghz = np.full(shape, np.nan)
    valid_frequency = (
        success
        & np.isfinite(optimal_lr)
        & (optimal_lr > 0)
        & np.isfinite(sampled_cr)
        & (sampled_cr > 0)
    )
    resonator_frequency_ghz[valid_frequency] = (
        1
        / np.sqrt(
            optimal_lr[valid_frequency] * 1e-9 * sampled_cr[valid_frequency] * 1e-15
        )
        / (2 * np.pi)
        / 1e9
    )

    ds = xr.Dataset(
        data_vars={
            "flux": (("poi",), [poi.get("flux", np.nan) for poi in poi_list]),
            "param_value": (("poi", "sample", "param"), param_values),
            "param_sigma": (("poi", "sample", "param"), param_sigmas),
            "energy": (("poi", "sample", "level"), energy),
            "tracked_energy": (("poi", "sample", "branch"), tracked_energy),
            "tracked_level": (("poi", "sample", "branch"), tracked_level),
            "assignment_score": (("poi", "sample", "branch"), assignment_score),
            "optimal_lr": (("poi", "sample"), optimal_lr),
            "original_chi_p": (("poi", "sample"), original_chi_p),
            "original_chi_m": (("poi", "sample"), original_chi_m),
            "original_delta_chi": (("poi", "sample"), original_delta_chi),
            "resonator_frequency_ghz": (
                ("poi", "sample"),
                resonator_frequency_ghz,
            ),
            "chi_p": (("poi", "sample"), chi_p),
            "chi_m": (("poi", "sample"), chi_m),
            "delta_chi": (("poi", "sample"), delta_chi),
            "abs_delta_chi_mhz": (("poi", "sample"), 1e3 * np.abs(delta_chi)),
            "abs_delta_chi_over_chi_p": (("poi", "sample"), objective),
            "optimizer_success": (("poi", "sample"), success),
            "optimizer_nfev": (("poi", "sample"), nfev),
            "optimizer_boundary_hit": (("poi", "sample"), boundary_hit),
        },
        coords={
            "poi": [poi["name"] for poi in poi_list],
            "sample": np.arange(samples),
            "param": param_names,
            "level": np.arange(evals_count),
            "branch": BRANCH_LABELS,
        },
        attrs={
            "description": "Monte Carlo samples with optimized readout inductance.",
            "circuit_factory": _circuit_factory_name(circuit_factory),
            "frozen_params": "Cr,Lr",
            "flux_param": "" if flux_param is None else flux_param,
            "seed": seed,
            "samples_per_poi": samples,
            "evals_count": evals_count,
            "lr_lower_nh": float(lr_bounds[0]),
            "lr_upper_nh": float(lr_bounds[1]),
            "lr_grid_size": grid_size,
            "lr_xatol_nh": xatol,
            "param_correlation_groups": _correlation_groups_attr(
                param_correlation_groups, param_names
            ),
            "param_rel_min": str(param_rel_min),
            "tracking_schema": TRACKING_SCHEMA,
        },
    )
    for name in (
        "energy",
        "tracked_energy",
        "original_chi_p",
        "original_chi_m",
        "original_delta_chi",
        "resonator_frequency_ghz",
        "chi_p",
        "chi_m",
        "delta_chi",
    ):
        ds[name].attrs["units"] = "GHz"
    ds["abs_delta_chi_mhz"].attrs["units"] = "MHz"
    ds["optimal_lr"].attrs["units"] = "nH"
    return ds


def load_or_compute_tunable_resonator_monte_carlo(
    poi_list,
    *,
    path=DATA_DIR / "monte_carlo_tunable_resonator.nc",
    force=False,
    samples=200,
    cpus=8,
    seed=20260528,
    evals_count=24,
    param_sigma=None,
    param_rel_sigma=None,
    param_correlation_groups=None,
    param_rel_min=0.01,
    circuit_factory=gen_dtc,
    flux_param=FLUX_PARAM,
    lr_bounds=(0.8, 2.5),
    grid_size=21,
    xatol=1e-5,
    optimizer=optimize_resonator_inductance,
    evaluator=evaluate_dispersive_shifts,
):
    path = Path(path)
    if path.exists() and not force:
        ds = xr.load_dataset(path)
        required = {
            "assignment_score",
            "original_chi_p",
            "original_chi_m",
            "original_delta_chi",
            "resonator_frequency_ghz",
        }
        missing = sorted(required - set(ds.data_vars))
        if missing:
            ds.close()
            raise ValueError(
                f"stale tunable-resonator cache {path} is missing {missing}; "
                "rerun with force=True"
            )
        if ds.attrs.get("tracking_schema") != TRACKING_SCHEMA:
            ds.close()
            raise ValueError(
                f"stale tunable-resonator cache {path} uses an older tracking "
                "schema; rerun with force=True"
            )
        return ds, path

    ds = run_tunable_resonator_monte_carlo(
        poi_list,
        samples=samples,
        cpus=cpus,
        seed=seed,
        evals_count=evals_count,
        param_sigma=param_sigma,
        param_rel_sigma=param_rel_sigma,
        param_correlation_groups=param_correlation_groups,
        param_rel_min=param_rel_min,
        circuit_factory=circuit_factory,
        flux_param=flux_param,
        lr_bounds=lr_bounds,
        grid_size=grid_size,
        xatol=xatol,
        optimizer=optimizer,
        evaluator=evaluator,
    )
    return ds, save_dataset(ds, path)


__all__ = [
    "_abs_delta_chi_over_chi_p",
    "_tunable_resonator_case",
    "evaluate_dispersive_shifts",
    "load_or_compute_tunable_resonator_monte_carlo",
    "optimize_resonator_inductance",
    "run_tunable_resonator_monte_carlo",
    "run_tunable_resonator_sample_sweep",
]
