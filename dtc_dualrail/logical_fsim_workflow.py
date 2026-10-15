"""Cache and dataset helpers for the logical-fSim marimo workflow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from dtc_dualrail.dualrail_fsim import (
    LOGICAL_STATE_LABELS,
    SUBSYSTEM_LABELS,
    DualRailPulseResult,
)
from dtc_dualrail.io import save_dataset

LOGICAL_FSIM_REFERENCE_SCHEMA = "logical-fsim-configured-reference-v1"
LOGICAL_FSIM_PULSE_SWEEP_SCHEMA = "logical-fsim-pulse-sweep-v2"


def load_or_create_configured_reference(
    factory: Callable[[], xr.Dataset],
    *,
    path: str | Path,
    fingerprint: str,
    force: bool = False,
) -> tuple[xr.Dataset, Path]:
    """Load or create deterministic configured contour inputs for smoke runs."""

    path = Path(path)
    if path.exists() and not force:
        dataset = xr.load_dataset(path)
        compatible = (
            dataset.attrs.get("configured_reference_schema")
            == LOGICAL_FSIM_REFERENCE_SCHEMA
            and dataset.attrs.get("fingerprint") == fingerprint
        )
        if compatible:
            return dataset, path
        dataset.close()
        raise ValueError(
            f"configured reference cache {path} has different inputs; "
            "rerun with force=True"
        )
    dataset = factory()
    dataset.attrs.update(
        configured_reference_schema=LOGICAL_FSIM_REFERENCE_SCHEMA,
        fingerprint=fingerprint,
    )
    return dataset, save_dataset(dataset, path)


def pulse_results_to_dataset(
    results: Sequence[DualRailPulseResult],
    rise_hold_pairs: Sequence[Sequence[float]],
    *,
    gate_type: str,
    fingerprint: str,
    gate_point: dict[str, float],
) -> xr.Dataset:
    """Persist every field of a pulse result for a candidate sweep."""

    if not results:
        raise ValueError("pulse sweep must contain at least one result")
    if len(results) != len(rise_hold_pairs):
        raise ValueError("results and rise/hold pairs must have the same length")
    sample_counts = np.asarray([result.time_ns.size for result in results], dtype=int)
    sample_count = int(np.max(sample_counts))

    def stack(name: str) -> np.ndarray:
        return np.asarray([getattr(result, name) for result in results])

    def stack_ragged(name: str) -> np.ndarray:
        arrays = [np.asarray(getattr(result, name)) for result in results]
        output = np.full(
            (len(arrays), sample_count, *arrays[0].shape[1:]),
            np.nan,
            dtype=float,
        )
        for index, values in enumerate(arrays):
            output[index, : values.shape[0]] = values
        return output

    scalar_fields = (
        "theta_rad",
        "phi_rad",
        "swap_probability",
        "average_leakage",
        "parasitic_amplitude",
        "unitarity_error",
        "coherent_fsim_error",
        "iswap_coherent_fidelity",
        "iswap_fidelity_with_leakage",
        "cphase_coherent_fidelity",
        "cphase_fidelity_with_leakage",
        "target_cphase_rad",
    )
    input_fields = (
        "survival_by_state",
        "leakage_by_state",
        "bare_code_probability_by_state",
        "coupler_excited_probability_by_state",
        "pair_a_invalid_probability_by_state",
        "pair_b_invalid_probability_by_state",
        "maximum_transient_leakage_by_state",
        "maximum_transient_coupler_excited_probability_by_state",
        "maximum_transient_pair_a_invalid_probability_by_state",
        "maximum_transient_pair_b_invalid_probability_by_state",
        "dynamic_phase_rad",
        "logical_assignment",
        "logical_assignment_probability",
    )
    pairs = np.asarray(rise_hold_pairs, dtype=float)
    data_vars: dict = {
        "rise_time_ns": ("candidate", pairs[:, 0]),
        "hold_time_ns": ("candidate", pairs[:, 1]),
        "duration_ns": ("candidate", 2 * pairs[:, 0] + pairs[:, 1]),
        "sample_count": ("candidate", sample_counts),
        "time_ns": (("candidate", "sample"), stack_ragged("time_ns")),
        "logical_frequency_a_mhz": (
            ("candidate", "sample"),
            stack_ragged("logical_frequency_a_mhz"),
        ),
        "logical_frequency_b_mhz": (
            ("candidate", "sample"),
            stack_ragged("logical_frequency_b_mhz"),
        ),
        "subsystem_flux": (
            ("candidate", "sample", "subsystem"),
            stack_ragged("subsystem_flux"),
        ),
        "max_flux_slew_per_ns": (
            ("candidate", "subsystem"),
            stack("max_flux_slew_per_ns"),
        ),
        "projected_unitary_real": (
            ("candidate", "logical_output", "logical_input"),
            stack("projected_unitary").real,
        ),
        "projected_unitary_imag": (
            ("candidate", "logical_output", "logical_input"),
            stack("projected_unitary").imag,
        ),
        "closest_unitary_real": (
            ("candidate", "logical_output", "logical_input"),
            stack("closest_unitary").real,
        ),
        "closest_unitary_imag": (
            ("candidate", "logical_output", "logical_input"),
            stack("closest_unitary").imag,
        ),
        "iswap_virtual_z_rad": (
            ("candidate", "local_z_parameter"),
            stack("iswap_virtual_z_rad"),
        ),
        "cphase_virtual_z_rad": (
            ("candidate", "local_z_parameter"),
            stack("cphase_virtual_z_rad"),
        ),
        "solver_method": (
            "candidate",
            np.asarray([result.solver_method for result in results]),
        ),
        "solver_status": (
            "candidate",
            np.asarray([result.solver_status for result in results]),
        ),
        "result_fingerprint": (
            "candidate",
            np.asarray([result.fingerprint for result in results]),
        ),
    }
    data_vars.update(
        {name: ("candidate", stack(name)) for name in scalar_fields}
    )
    data_vars.update(
        {
            name.removesuffix("_by_state"): (
                ("candidate", "logical_input"),
                stack(name),
            )
            for name in input_fields
        }
    )
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "candidate": np.arange(len(results)),
            "sample": np.arange(sample_count),
            "subsystem": list(SUBSYSTEM_LABELS),
            "logical_input": list(LOGICAL_STATE_LABELS),
            "logical_output": list(LOGICAL_STATE_LABELS),
            "local_z_parameter": ["before_A", "before_B", "after_A", "after_B"],
        },
        attrs={
            "description": f"{gate_type} pulse characterization sweep",
            "logical_fsim_pulse_sweep_schema": LOGICAL_FSIM_PULSE_SWEEP_SCHEMA,
            "gate_type": gate_type,
            "fingerprint": fingerprint,
            **{f"gate_{key}": value for key, value in gate_point.items()},
        },
    )
    dataset.time_ns.attrs["units"] = "ns"
    dataset.logical_frequency_a_mhz.attrs["units"] = "MHz"
    dataset.logical_frequency_b_mhz.attrs["units"] = "MHz"
    dataset.subsystem_flux.attrs["units"] = "flux quantum"
    dataset.max_flux_slew_per_ns.attrs["units"] = "flux quantum / ns"
    for name in (
        "theta_rad",
        "phi_rad",
        "dynamic_phase_rad",
        "iswap_virtual_z_rad",
        "cphase_virtual_z_rad",
    ):
        dataset[name].attrs["units"] = "rad"
    return dataset


def load_compatible_pulse_sweep(
    path: str | Path, fingerprint: str
) -> xr.Dataset | None:
    """Return a complete compatible sweep, or ``None`` for a stale cache."""

    path = Path(path)
    if not path.exists():
        return None
    dataset = xr.load_dataset(path)
    compatible = (
        dataset.attrs.get("logical_fsim_pulse_sweep_schema")
        == LOGICAL_FSIM_PULSE_SWEEP_SCHEMA
        and dataset.attrs.get("fingerprint") == fingerprint
    )
    if compatible:
        return dataset
    dataset.close()
    return None


__all__ = [
    "LOGICAL_FSIM_PULSE_SWEEP_SCHEMA",
    "LOGICAL_FSIM_REFERENCE_SCHEMA",
    "load_compatible_pulse_sweep",
    "load_or_create_configured_reference",
    "pulse_results_to_dataset",
]
