"""Reusable sweep helpers."""

from pathlib import Path

import numpy as np
import xarray as xr
from pathos.pools import ProcessPool

from dtc_dualrail.circuits import gen_dtc as default_gen_dtc
from dtc_dualrail.constants import (
    BRANCH_INDEX,
    BRANCH_LABELS,
    DATA_DIR,
    FLUX_PARAM,
    TRACKING_SCHEMA,
)
from dtc_dualrail.io import save_dataset
from dtc_dualrail.tracking import track_levels_by_overlap


def _target_scales(param_name, param_targets):
    if param_targets is None:
        return {param_name: 1.0}
    if isinstance(param_targets, dict):
        if len(param_targets) == 0:
            raise ValueError("param targets must not be empty")
        return dict(param_targets)
    param_targets = tuple(param_targets)
    if len(param_targets) == 0:
        raise ValueError("param targets must not be empty")
    return {target: 1.0 for target in param_targets}


def _two_param_case(job):
    (
        param1_idx,
        param2_idx,
        param1_name,
        param1_value,
        param1_target_scales,
        param2_name,
        param2_value,
        param2_target_scales,
        fixed_flux,
        base_params,
        param_update,
        circuit_factory,
        evals_count,
        flux_param,
        get_eigenstates,
    ) = job

    params = dict(base_params)
    for target, scale in param1_target_scales.items():
        params[target] = scale * param1_value
    for target, scale in param2_target_scales.items():
        params[target] = scale * param2_value
    if param_update is not None:
        params.update(
            param_update(param1_name, param1_value, param2_name, param2_value, params)
        )

    circuit = circuit_factory(**params)
    sweep = circuit.get_spectrum_vs_paramvals(
        flux_param,
        np.asarray([fixed_flux]),
        evals_count=evals_count,
        get_eigenstates=get_eigenstates,
        num_cpus=1,
    )

    energy = np.asarray(sweep.energy_table)[0]
    result = {
        "param1_idx": param1_idx,
        "param2_idx": param2_idx,
        "energy": energy,
        "params": params,
    }

    if get_eigenstates:
        setattr(circuit, flux_param, fixed_flux)
        tracked_level, assignment_score, tracked_energy = track_levels_by_overlap(
            sweep.state_table,
            np.asarray(sweep.energy_table),
            q3_operator=circuit.Q3_operator(),
            n1_operator=circuit.n1_operator(),
            n2_operator=circuit.n2_operator(),
        )
        result.update(
            {
                "tracked_energy": tracked_energy[0],
                "tracked_level": tracked_level[0],
                "assignment_score": assignment_score[0],
            }
        )

    return result


def run_two_param_sweep(
    gen_dtc=default_gen_dtc,
    param1_name: str = "C2r",
    param1_values=None,
    param2_name: str = "C2",
    param2_values=None,
    *,
    param1_targets=None,
    param2_targets=None,
    fixed_flux: float = 0.0,
    base_params=None,
    param_update=None,
    evals_count: int = 24,
    flux_param: str = FLUX_PARAM,
    get_eigenstates: bool = False,
    cpus: int = 8,
) -> xr.Dataset:
    if param1_values is None or param2_values is None:
        raise ValueError("param1_values and param2_values are required")
    if param1_name == param2_name:
        raise ValueError("param1_name and param2_name must differ")

    param1_values = np.asarray(param1_values)
    param2_values = np.asarray(param2_values)
    param1_target_scales = _target_scales(param1_name, param1_targets)
    param2_target_scales = _target_scales(param2_name, param2_targets)
    param1_targets = tuple(param1_target_scales)
    param2_targets = tuple(param2_target_scales)
    shared_targets = set(param1_targets) & set(param2_targets)
    if shared_targets:
        raise ValueError(
            f"param targets cannot be shared across axes: {shared_targets}"
        )

    base_params = {} if base_params is None else dict(base_params)

    jobs = [
        (
            param1_idx,
            param2_idx,
            param1_name,
            param1_value,
            param1_target_scales,
            param2_name,
            param2_value,
            param2_target_scales,
            fixed_flux,
            base_params,
            param_update,
            gen_dtc,
            evals_count,
            flux_param,
            get_eigenstates,
        )
        for param1_idx, param1_value in enumerate(param1_values)
        for param2_idx, param2_value in enumerate(param2_values)
    ]

    if cpus == 1:
        results = list(map(_two_param_case, jobs))
    else:
        with ProcessPool(nodes=cpus) as pool:
            results = pool.map(_two_param_case, jobs)

    level_count = len(results[0]["energy"])
    energy = np.full((len(param1_values), len(param2_values), level_count), np.nan)

    if get_eigenstates:
        branch_count = len(results[0]["tracked_energy"])
        if branch_count != len(BRANCH_LABELS):
            raise ValueError("tracked state count does not match BRANCH_LABELS")
        tracked_energy = np.full(
            (len(param1_values), len(param2_values), branch_count), np.nan
        )
        tracked_level = np.full(
            (len(param1_values), len(param2_values), branch_count), -1, dtype=int
        )
        assignment_score = np.full_like(tracked_energy, np.nan)

    for result in results:
        i = result["param1_idx"]
        j = result["param2_idx"]
        energy[i, j] = result["energy"]
        if get_eigenstates:
            tracked_energy[i, j] = result["tracked_energy"]
            tracked_level[i, j] = result["tracked_level"]
            assignment_score[i, j] = result["assignment_score"]

    data_vars = {
        "energy": ((param1_name, param2_name, "level"), energy),
    }
    coords = {
        param1_name: param1_values,
        param2_name: param2_values,
        "level": np.arange(level_count),
        "flux": fixed_flux,
    }
    for target in param1_targets:
        if target != param1_name and target not in coords:
            coords[target] = (param1_name, param1_target_scales[target] * param1_values)
    for target in param2_targets:
        if target != param2_name and target not in coords:
            coords[target] = (param2_name, param2_target_scales[target] * param2_values)

    if get_eigenstates:
        branch_coord = BRANCH_LABELS[:branch_count]
        dims = (param1_name, param2_name, "branch")
        ground = tracked_energy[..., BRANCH_INDEX["g"]]
        plus = tracked_energy[..., BRANCH_INDEX["p"]]
        minus = tracked_energy[..., BRANCH_INDEX["m"]]
        resonator = tracked_energy[..., BRANCH_INDEX["r"]]
        plus_resonator = tracked_energy[..., BRANCH_INDEX["pr"]]
        minus_resonator = tracked_energy[..., BRANCH_INDEX["mr"]]
        chi_p = plus_resonator - resonator - plus + ground
        chi_m = minus_resonator - resonator - minus + ground
        data_vars.update(
            {
                "tracked_energy": (dims, tracked_energy),
                "tracked_level": (dims, tracked_level),
                "assignment_score": (dims, assignment_score),
                "chi_p": ((param1_name, param2_name), chi_p),
                "chi_m": ((param1_name, param2_name), chi_m),
                "delta_chi": ((param1_name, param2_name), chi_p - chi_m),
            }
        )
        coords["branch"] = branch_coord

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "description": "DTC two-parameter spectrum sweep at fixed external flux.",
            "flux_param": flux_param,
            "fixed_flux": fixed_flux,
            "energy_units": "GHz",
            "sweep_param_1": param1_name,
            "sweep_param_2": param2_name,
            "sweep_param_1_targets": ", ".join(param1_targets),
            "sweep_param_2_targets": ", ".join(param2_targets),
            "sweep_param_1_target_scales": ", ".join(
                f"{target}:{scale}" for target, scale in param1_target_scales.items()
            ),
            "sweep_param_2_target_scales": ", ".join(
                f"{target}:{scale}" for target, scale in param2_target_scales.items()
            ),
            "evals_count": evals_count,
            "cpus": cpus,
        },
    )
    if get_eigenstates:
        ds.attrs["tracking_schema"] = TRACKING_SCHEMA
        ds["chi_p"].attrs["units"] = "GHz"
        ds["chi_m"].attrs["units"] = "GHz"
        ds["delta_chi"].attrs["units"] = "GHz"
    return ds


def load_or_compute_two_param_sweep(
    path=DATA_DIR / "two_param_sweep.nc",
    *,
    force: bool = False,
    **sweep_kwargs,
):
    path = Path(path)
    if path.exists() and not force:
        return xr.load_dataset(path), path

    ds = run_two_param_sweep(**sweep_kwargs)
    return ds, save_dataset(ds, path)


__all__ = [
    "_target_scales",
    "_two_param_case",
    "load_or_compute_two_param_sweep",
    "run_two_param_sweep",
]
