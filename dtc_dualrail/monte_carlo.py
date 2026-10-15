"""Monte Carlo dataset generation and caching."""

from pathlib import Path

import numpy as np
import xarray as xr
from pathos.pools import ProcessPool

from dtc_dualrail.circuits import (
    _circuit_factory_name,
    _validate_circuit_factory_params,
    gen_dtc,
)
from dtc_dualrail.constants import (
    BRANCH_INDEX,
    BRANCH_LABELS,
    DATA_DIR,
    DEFAULT_PARAM_REL_SIGMA,
    FLUX_PARAM,
    TRACKING_SCHEMA,
)
from dtc_dualrail.io import save_dataset
from dtc_dualrail.sampling import (
    _correlation_groups_attr,
    _normalize_correlation_groups,
    _sample_parameters,
)
from dtc_dualrail.tracking import track_levels_by_overlap


def _monte_carlo_case(job):
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
        evals_count,
    ) = job
    rng = np.random.default_rng(seed)
    sampled_params, resolved_param_sigma = _sample_parameters(
        params,
        rng,
        param_sigma=param_sigma,
        param_rel_sigma=param_rel_sigma,
        param_correlation_groups=param_correlation_groups,
        param_rel_min=param_rel_min,
    )

    circuit = circuit_factory(**sampled_params)
    try:
        setattr(circuit, FLUX_PARAM, flux)
    except Exception:
        pass
    energies, states = circuit.eigensys(evals_count=evals_count)
    energy = np.asarray(energies)[None, :]
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
        "poi_idx": poi_idx,
        "sample_idx": sample_idx,
        "poi_name": poi_name,
        "flux": flux,
        "params": sampled_params,
        "param_sigma": resolved_param_sigma,
        "energy": energy[0],
        "tracked_energy": tracked_energy[0],
        "tracked_level": tracked_level[0],
        "assignment_score": assignment_score[0],
        "chi_p": chi_p,
        "chi_m": chi_m,
        "delta_chi": chi_p - chi_m,
    }


def run_monte_carlo(
    poi_list,
    *,
    samples: int = 200,
    cpus: int = 8,
    seed: int = 20260528,
    evals_count: int = 24,
    param_sigma=None,
    param_rel_sigma=None,
    param_correlation_groups=None,
    param_rel_min=0.01,
    circuit_factory=gen_dtc,
) -> xr.Dataset:
    param_rel_sigma = (
        DEFAULT_PARAM_REL_SIGMA if param_rel_sigma is None else param_rel_sigma
    )
    _validate_circuit_factory_params(
        circuit_factory,
        [poi["params"] for poi in poi_list],
    )
    _normalize_correlation_groups(param_correlation_groups, poi_list[0]["params"])
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
                    poi["flux"],
                    poi["params"],
                    child_seeds[seed_idx].generate_state(1)[0],
                    param_sigma,
                    param_rel_sigma,
                    param_correlation_groups,
                    param_rel_min,
                    circuit_factory,
                    evals_count,
                )
            )
            seed_idx += 1

    with ProcessPool(nodes=cpus) as pool:
        results = pool.map(_monte_carlo_case, jobs)

    poi_count = len(poi_list)
    param_names = list(poi_list[0]["params"])
    level_count = len(results[0]["energy"])
    branch_count = len(results[0]["tracked_energy"])

    param_values = np.full((poi_count, samples, len(param_names)), np.nan)
    param_sigmas = np.full_like(param_values, np.nan)
    energy = np.full((poi_count, samples, level_count), np.nan)
    tracked_energy = np.full((poi_count, samples, branch_count), np.nan)
    tracked_level = np.full((poi_count, samples, branch_count), -1, dtype=int)
    assignment_score = np.full((poi_count, samples, branch_count), np.nan)
    chi_p = np.full((poi_count, samples), np.nan)
    chi_m = np.full_like(chi_p, np.nan)
    delta_chi = np.full_like(chi_p, np.nan)

    for result in results:
        poi_idx = result["poi_idx"]
        sample_idx = result["sample_idx"]
        param_values[poi_idx, sample_idx] = [
            result["params"][param_name] for param_name in param_names
        ]
        param_sigmas[poi_idx, sample_idx] = [
            result["param_sigma"][param_name] for param_name in param_names
        ]
        energy[poi_idx, sample_idx] = result["energy"]
        tracked_energy[poi_idx, sample_idx] = result["tracked_energy"]
        tracked_level[poi_idx, sample_idx] = result["tracked_level"]
        assignment_score[poi_idx, sample_idx] = result["assignment_score"]
        chi_p[poi_idx, sample_idx] = result["chi_p"]
        chi_m[poi_idx, sample_idx] = result["chi_m"]
        delta_chi[poi_idx, sample_idx] = result["delta_chi"]

    ds = xr.Dataset(
        data_vars={
            "flux": (("poi",), [poi["flux"] for poi in poi_list]),
            "param_value": (("poi", "sample", "param"), param_values),
            "param_sigma": (("poi", "sample", "param"), param_sigmas),
            "energy": (("poi", "sample", "level"), energy),
            "tracked_energy": (("poi", "sample", "branch"), tracked_energy),
            "tracked_level": (("poi", "sample", "branch"), tracked_level),
            "assignment_score": (("poi", "sample", "branch"), assignment_score),
            "chi_p": (("poi", "sample"), chi_p),
            "chi_m": (("poi", "sample"), chi_m),
            "delta_chi": (("poi", "sample"), delta_chi),
            "abs_delta_chi_mhz": (("poi", "sample"), 1e3 * np.abs(delta_chi)),
        },
        coords={
            "poi": [poi["name"] for poi in poi_list],
            "sample": np.arange(samples),
            "param": param_names,
            "level": np.arange(level_count),
            "branch": BRANCH_LABELS[:branch_count],
        },
        attrs={
            "description": "Monte Carlo robustness samples around DTC points of interest.",
            "flux_param": FLUX_PARAM,
            "energy_units": "GHz",
            "chi_units": "GHz",
            "abs_delta_chi_mhz_units": "MHz",
            "samples_per_poi": samples,
            "seed": seed,
            "evals_count": evals_count,
            "cpus": cpus,
            "sigma_mode": (
                "absolute_with_relative_fallback"
                if param_sigma is not None
                else "relative"
            ),
            "param_correlation_groups": _correlation_groups_attr(
                param_correlation_groups,
                param_names,
            ),
            "param_rel_min": str(param_rel_min),
            "circuit_factory": _circuit_factory_name(circuit_factory),
            "tracking_schema": TRACKING_SCHEMA,
        },
    )
    return ds


def load_or_compute_monte_carlo(
    poi_list,
    *,
    path=DATA_DIR / "monte_carlo_chi.nc",
    force: bool = False,
    samples: int = 200,
    cpus: int = 8,
    seed: int = 20260528,
    evals_count: int = 24,
    param_sigma=None,
    param_rel_sigma=None,
    param_correlation_groups=None,
    param_rel_min=0.01,
    circuit_factory=gen_dtc,
):
    path = Path(path)
    if path.exists() and not force:
        ds = xr.load_dataset(path)
        if ds.attrs.get("tracking_schema") != TRACKING_SCHEMA:
            ds.close()
            raise ValueError(
                f"stale Monte Carlo cache {path} uses an older tracking schema; "
                "rerun with force=True"
            )
        return ds, path

    ds = run_monte_carlo(
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
    )
    return ds, save_dataset(ds, path)


__all__ = [
    "_monte_carlo_case",
    "load_or_compute_monte_carlo",
    "run_monte_carlo",
]
