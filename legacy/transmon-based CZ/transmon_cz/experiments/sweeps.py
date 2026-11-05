#!/usr/bin/env python3
"""Sweep logical CZ performance versus charging-energy mismatch and EJ,A2."""
# %% Imports and configuration
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import qutip as qt
from scipy.optimize import brentq


from ..simulation import (
    LABELS,
    TWOPI,
    BasisSpec,
    EvolutionSpec,
    FourTransmonSystem,
    TransmonSpec,
    build_local_model,
    cz_collision_splittings,
    first_odd_pi_index,
    odd_pi_index,
    simulate_logical_basis,
    transition_table,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PACKAGE_ROOT / "output"
CONFIG_PATH = PACKAGE_ROOT / "configs" / "parameter_sweep_config.json"


# %% Model helpers
def _axis(values: list[float] | dict) -> np.ndarray:
    if isinstance(values, dict):
        return np.linspace(values["start"], values["stop"], values["count"])
    return np.asarray(values, dtype=float)


def _system_with_ec(raw, ec_low, ec_high, ej_a1, ej_a2, ej_b1, ej_b2):
    transmons = {label: dict(raw["transmons"][label]) for label in LABELS}
    for label in ("A1", "B1"):
        transmons[label]["EC_GHz"] = ec_low + (raw["transmons"]["B1"]["EC_GHz"] - raw["transmons"]["A1"]["EC_GHz"] if label == "B1" else 0)
    for label in ("A2", "B2"):
        transmons[label]["EC_GHz"] = ec_high + (raw["transmons"]["B2"]["EC_GHz"] - raw["transmons"]["A2"]["EC_GHz"] if label == "B2" else 0)
    transmons["A1"]["EJ_GHz"] = ej_a1
    transmons["A2"]["EJ_GHz"] = ej_a2
    transmons["B1"]["EJ_GHz"] = ej_b1
    transmons["B2"]["EJ_GHz"] = ej_b2
    specs = {label: TransmonSpec(**transmons[label]) for label in LABELS}
    return FourTransmonSystem(
        specs, raw["couplings_GHz"], BasisSpec(**raw["basis"])
    )


def _system(raw, delta_ec, ej_a1, ej_a2, ej_b1, ej_b2) -> FourTransmonSystem:
    # Original sweep convention: Delta_EC = EC_(A2,B2) - EC_(A1,B1).
    ec_low = float(raw["sweep"]["EC_A1_B1_GHz"])
    return _system_with_ec(
        raw, ec_low, ec_low + delta_ec, ej_a1, ej_a2, ej_b1, ej_b2
    )


def _f01(raw: dict, label: str, ec: float, ej: float) -> float:
    if label in ("B1", "B2"):
        counterpart = "A1" if label == "B1" else "A2"
        ec += raw["transmons"][label]["EC_GHz"] - raw["transmons"][counterpart]["EC_GHz"]
    spec = TransmonSpec(EC_GHz=ec, EJ_GHz=ej, ng=raw["transmons"][label]["ng"])
    local = build_local_model(label, spec, BasisSpec(**raw["basis"]))
    return float(local.levels_GHz[1] - local.levels_GHz[0])


def _qobj_scalar(value) -> complex:
    """Extract a scalar from either QuTiP 4 scalar products or QuTiP 5 Qobjs."""
    if isinstance(value, qt.Qobj):
        if value.shape != (1, 1):
            raise ValueError(f"Expected a scalar Qobj, got shape {value.shape}")
        value = value.full()[0, 0]
    return complex(value)


def _collision_detuning(system: FourTransmonSystem, channel: str) -> float:
    collision_states = {
        "0110_to_0200": ((0, 1, 1, 0), (0, 2, 0, 0)),
        "0110_to_0020": ((0, 1, 1, 0), (0, 0, 2, 0)),
    }
    try:
        first_levels, second_levels = collision_states[channel]
    except KeyError as error:
        raise ValueError(f"Unknown collision channel {channel!r}") from error
    hamiltonian_ghz = system.hamiltonian() / TWOPI
    first = system.local_product_ket(first_levels)
    second = system.local_product_ket(second_levels)

    def energy(state) -> float:
        value = state.dag() * hamiltonian_ghz * state
        return float(np.real(_qobj_scalar(value)))

    # This is exactly the bare_detuning_GHz convention used by
    # static_two_state_splitting: second-state energy minus first-state energy.
    return energy(second) - energy(first)


def _effective_exchange(system: FourTransmonSystem, left: str, right: str) -> float:
    """One-excitation exchange rate from the configured charge coupling."""
    coupling = system.couplings_GHz[f"{left}_{right}"]

    def n01(label: str) -> float:
        local = system.local[label]
        value = local.level_kets[0].dag() * local.n_op * local.level_kets[1]
        return abs(_qobj_scalar(value))

    return float(abs(coupling) * n01(left) * n01(right))


def _first_odd_pi_index(phases_rad: np.ndarray, tolerance_rad: float) -> int | None:
    """Compatibility wrapper around the shared gate-analysis helper."""
    return first_odd_pi_index(phases_rad, tolerance_rad)


def _nth_odd_pi_index(phases_rad: np.ndarray, order: int) -> int | None:
    """Compatibility wrapper around the shared gate-analysis helper."""
    return odd_pi_index(phases_rad, order=order)


def _store_first_pi_metrics(
    arrays: dict[str, np.ndarray],
    index: tuple[int, int],
    result: dict,
    tolerance_rad: float,
) -> None:
    """Store first-odd-pi metrics calculated by the shared simulator."""
    if result.get("first_odd_pi_index") is None:
        return
    arrays["first_pi_conditional_time_ns"][index] = result["first_odd_pi_time_ns"]
    arrays["conditional_cz_fidelity_at_first_pi"][index] = result[
        "conditional_cz_fidelity_at_first_odd_pi"
    ]
    arrays["acceptance_at_first_pi"][index] = result["acceptance_at_first_odd_pi"]
    arrays["conditional_phase_error_at_first_pi_rad"][index] = result[
        "conditional_phase_error_at_first_odd_pi_rad"
    ]


def make_colormaps(data: dict[str, np.ndarray], output: Path) -> None:
    x = data["delta_EC_GHz"] * 1000.0
    y = data["EJ_A2_GHz"]
    panels = (
        ("best_average_cz_fidelity", "Best logical CZ fidelity", "viridis"),
        ("leakage_at_best", "Nonlogical leakage at best fidelity", "magma"),
        ("solved_EJ_A1_GHz", "Solved $E_{J,A1}/h$ (GHz)", "viridis"),
        ("solved_EJ_B1_GHz", "Solved $E_{J,B1}/h$ (GHz)", "viridis"),
        ("solved_EJ_B2_GHz", "Solved $E_{J,B2}/h$ (GHz)", "viridis"),
        (
            "best_conditional_cz_fidelity",
            "Best conditional logical CZ fidelity",
            "viridis",
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (key, title, cmap) in zip(axes.flat, panels):
        mesh = axis.pcolormesh(x, y, data[key], shading="auto", cmap=cmap)
        axis.set_xlabel(r"$\Delta E_C/h$ (MHz)")
        axis.set_ylabel(r"$E_{J,A2}/h$ (GHz)")
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_conditional_gate_speed_colormap(
    data: dict[str, np.ndarray],
    output: Path,
    maximum_duration_ns: float,
    y_key: str,
    y_scale: float,
    y_label: str,
    fidelity_threshold: float = 0.85,
    boundary_gate_time_ns: float = 90.0,
) -> None:
    """Plot the conditional gate-speed proxy with prioritized status colors."""
    x = data["delta_EC_GHz"] * 1000.0
    y = np.atleast_1d(data[y_key]) * y_scale
    conditional_time = data["best_conditional_time_ns"]
    conditional_fidelity = data["best_conditional_cz_fidelity"]
    end_slope = data.get("conditional_fidelity_end_slope_per_ns")
    valid = np.isfinite(conditional_time) & np.isfinite(conditional_fidelity)
    positive_time = valid & (conditional_time > 0.0)
    if end_slope is None:
        boundary_limited = np.zeros_like(valid, dtype=bool)
        print(
            "Boundary colors omitted: saved data do not contain "
            "conditional_fidelity_end_slope_per_ns. Rerun the 100 ns sweep."
        )
    else:
        boundary_limited = (
            valid
            & np.isfinite(end_slope)
            & (conditional_time > boundary_gate_time_ns)
            & (end_slope > 0.0)
        )
    low_fidelity = valid & (conditional_fidelity < fidelity_threshold)

    # For a first odd-pi accumulation, |zeta_eff| ~= 1/(2 t_gate).
    speed_mhz = np.full_like(conditional_time, np.nan, dtype=float)
    np.divide(500.0, conditional_time, out=speed_mhz, where=positive_time)

    figure, axis = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    normal_points = np.ma.masked_where(
        ~positive_time | low_fidelity | boundary_limited, speed_mhz
    )
    mesh = axis.pcolormesh(x, y, normal_points, shading="auto", cmap=cmap)
    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label(r"Conditional gate-speed proxy $500/t_{\rm cond}$ (MHz)")

    # Draw low fidelity first, then boundary-limited points on top. This gives
    # the maximum-duration check priority when both conditions are true.
    red_points = np.ma.masked_where(~low_fidelity | boundary_limited, np.ones_like(speed_mhz))
    axis.pcolormesh(x, y, red_points, shading="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    boundary_points = np.ma.masked_where(~boundary_limited, np.ones_like(speed_mhz))
    axis.pcolormesh(
        x, y, boundary_points, shading="auto",
        cmap=plt.matplotlib.colors.ListedColormap(["black"]),
        vmin=0.0, vmax=1.0,
    )

    axis.set_xlabel(r"$\Delta E_C/h$ (MHz)")
    axis.set_ylabel(y_label)
    axis.set_title("Conditional gate-speed proxy")
    axis.legend(
        handles=[
            Patch(
                facecolor="black",
                label=(
                    rf"$t_{{\rm cond}}>{boundary_gate_time_ns:g}$ ns and "
                    "end trend rising"
                ),
            ),
            Patch(
                facecolor=plt.get_cmap("Reds")(1.0),
                label=rf"$F_{{\rm CZ,cond}}<{100*fidelity_threshold:.0f}\%$",
            ),
            Patch(facecolor="lightgray", label="Invalid / NaN"),
        ],
        loc="best",
        framealpha=0.95,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_best_conditional_gate_time_colormap(
    data: dict[str, np.ndarray],
    output: Path,
    y_key: str,
    y_scale: float,
    y_label: str,
    fidelity_threshold: float = 0.85,
    boundary_gate_time_ns: float = 490.0,
) -> None:
    """Plot globally optimized conditional gate time with status overlays."""
    x = data["delta_EC_GHz"] * 1000.0
    y = np.atleast_1d(data[y_key]) * y_scale
    gate_time = data["best_conditional_time_ns"]
    fidelity = data["best_conditional_cz_fidelity"]
    end_slope = data["conditional_fidelity_end_slope_per_ns"]
    valid = np.isfinite(gate_time) & np.isfinite(fidelity)
    boundary = (
        valid
        & (gate_time > boundary_gate_time_ns)
        & np.isfinite(end_slope)
        & (end_slope > 0.0)
    )
    low_fidelity = valid & (fidelity < fidelity_threshold) & ~boundary

    figure, axis = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    ordinary = np.ma.masked_where(~valid | low_fidelity | boundary, gate_time)
    mesh = axis.pcolormesh(x, y, ordinary, shading="auto", cmap=cmap)
    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label("Best conditional gate time (ns)")

    red = np.ma.masked_where(~low_fidelity, np.ones_like(gate_time))
    axis.pcolormesh(x, y, red, shading="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    black = np.ma.masked_where(~boundary, np.ones_like(gate_time))
    axis.pcolormesh(
        x, y, black, shading="auto",
        cmap=plt.matplotlib.colors.ListedColormap(["black"]),
        vmin=0.0, vmax=1.0,
    )
    axis.set_xlabel(r"$\Delta E_C/h$ (MHz)")
    axis.set_ylabel(y_label)
    axis.set_title("Best conditional gate time")
    axis.legend(
        handles=[
            Patch(
                facecolor="black",
                label=(
                    rf"$t_{{\rm cond}}>{boundary_gate_time_ns:g}$ ns and "
                    "end trend rising"
                ),
            ),
            Patch(
                facecolor=plt.get_cmap("Reds")(1.0),
                label=rf"$F_{{\rm CZ,cond}}<{100*fidelity_threshold:.0f}\%$",
            ),
            Patch(facecolor="lightgray", label="Invalid / NaN"),
        ],
        loc="best",
        framealpha=0.95,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_first_pi_gate_time_colormap(
    data: dict[str, np.ndarray],
    output: Path,
    y_key: str,
    y_scale: float,
    y_label: str,
    fidelity_threshold: float = 0.85,
    boundary_gate_time_ns: float = 490.0,
) -> None:
    """Plot earliest odd-pi arrival, with boundary and low-fidelity overlays."""
    x = data["delta_EC_GHz"] * 1000.0
    y = np.atleast_1d(data[y_key]) * y_scale
    gate_time = data["first_pi_conditional_time_ns"]
    fidelity = data["conditional_cz_fidelity_at_first_pi"]
    end_slope = data["conditional_fidelity_end_slope_per_ns"]
    valid = np.isfinite(gate_time) & np.isfinite(fidelity)
    boundary = (
        valid
        & (gate_time > boundary_gate_time_ns)
        & np.isfinite(end_slope)
        & (end_slope > 0.0)
    )
    low_fidelity = valid & (fidelity < fidelity_threshold) & ~boundary

    figure, axis = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    ordinary = np.ma.masked_where(~valid | low_fidelity | boundary, gate_time)
    mesh = axis.pcolormesh(x, y, ordinary, shading="auto", cmap=cmap)
    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label("First odd-$\\pi$ conditional gate time (ns)")

    red = np.ma.masked_where(~low_fidelity, np.ones_like(gate_time))
    axis.pcolormesh(x, y, red, shading="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    black = np.ma.masked_where(~boundary, np.ones_like(gate_time))
    axis.pcolormesh(
        x, y, black, shading="auto",
        cmap=plt.matplotlib.colors.ListedColormap(["black"]),
        vmin=0.0, vmax=1.0,
    )
    axis.set_xlabel(r"$\Delta E_C/h$ (MHz)")
    axis.set_ylabel(y_label)
    axis.set_title(r"Earliest odd-$\pi$ conditional gate time")
    axis.legend(
        handles=[
            Patch(
                facecolor="black",
                label=(
                    rf"$t_{{\rm first\,\pi}}>{boundary_gate_time_ns:g}$ ns and "
                    "end trend rising"
                ),
            ),
            Patch(
                facecolor=plt.get_cmap("Reds")(1.0),
                label=rf"$F_{{\rm CZ,cond}}<{100*fidelity_threshold:.0f}\%$ at first $\pi$",
            ),
            Patch(facecolor="lightgray", label=r"No odd-$\pi$ arrival / invalid"),
        ],
        loc="best",
        framealpha=0.95,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_parameter_sweep(
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, np.ndarray]:
    """Run the sweep, save its outputs, and return the numerical arrays."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    if raw.get("noise") is not None:
        raise ValueError("The logical fidelity sweep currently requires noise=null")

    sweep = raw["sweep"]
    deltas = _axis(sweep["delta_EC_GHz"])
    ej_a2_values = _axis(sweep["EJ_A2_GHz"])
    channel = sweep.get("collision_channel", "0110_to_0200")
    bracket = tuple(float(value) for value in sweep["EJ_B1_bracket_GHz"])
    baseline_delta = (
        raw["transmons"]["A2"]["EC_GHz"] - raw["transmons"]["A1"]["EC_GHz"]
    )
    baseline = _system(
        raw,
        baseline_delta,
        raw["transmons"]["A1"]["EJ_GHz"],
        raw["transmons"]["A2"]["EJ_GHz"],
        raw["transmons"]["B1"]["EJ_GHz"],
        raw["transmons"]["B2"]["EJ_GHz"],
    )
    target = sweep.get("target_bare_detuning_GHz")
    if target is None:
        target = _collision_detuning(baseline, channel)
    target = float(target)
    target_a = sweep.get("target_A1_minus_A2_detuning_GHz")
    if target_a is None:
        target_a = (
            baseline.local["A1"].levels_GHz[1]
            - baseline.local["A2"].levels_GHz[1]
        )
    target_b = sweep.get("target_B1_minus_B2_detuning_GHz")
    if target_b is None:
        target_b = (
            baseline.local["B1"].levels_GHz[1]
            - baseline.local["B2"].levels_GHz[1]
        )
    target_a, target_b = float(target_a), float(target_b)
    rail_bracket = tuple(
        float(value) for value in sweep.get("rail_EJ_bracket_GHz", bracket)
    )

    shape = (ej_a2_values.size, deltas.size)
    arrays = {
        "delta_EC_GHz": deltas,
        "EJ_A2_GHz": ej_a2_values,
        "solved_EJ_A1_GHz": np.full(shape, np.nan),
        "solved_EJ_B1_GHz": np.full(shape, np.nan),
        "solved_EJ_B2_GHz": np.full(shape, np.nan),
        "achieved_bare_detuning_GHz": np.full(shape, np.nan),
        "achieved_A1_minus_A2_detuning_GHz": np.full(shape, np.nan),
        "achieved_B1_minus_B2_detuning_GHz": np.full(shape, np.nan),
        "minimum_EJ_over_EC": np.full(shape, np.nan),
        "A_detuning_over_exchange": np.full(shape, np.nan),
        "B_detuning_over_exchange": np.full(shape, np.nan),
        "best_average_cz_fidelity": np.full(shape, np.nan),
        "best_time_ns": np.full(shape, np.nan),
        "best_conditional_cz_fidelity": np.full(shape, np.nan),
        "best_conditional_time_ns": np.full(shape, np.nan),
        "first_pi_conditional_time_ns": np.full(shape, np.nan),
        "conditional_cz_fidelity_at_first_pi": np.full(shape, np.nan),
        "acceptance_at_first_pi": np.full(shape, np.nan),
        "conditional_phase_error_at_first_pi_rad": np.full(shape, np.nan),
        "conditional_fidelity_end_slope_per_ns": np.full(shape, np.nan),
        "acceptance_at_best_unconditional": np.full(shape, np.nan),
        "acceptance_at_best_conditional": np.full(shape, np.nan),
        "leakage_at_best": np.full(shape, np.nan),
        "erasure_detectable_leakage_at_best": np.full(shape, np.nan),
        "erasure_undetectable_leakage_at_best": np.full(shape, np.nan),
    }
    evolution = EvolutionSpec(**raw["evolution"])
    phase_tolerance = float(sweep.get("conditional_phase_tolerance_rad", 0.05))
    trace_shape = shape + (evolution.samples,)
    trace_arrays = {
        "conditional_cz_fidelity_vs_time": np.full(trace_shape, np.nan),
        "conditional_phase_vs_time_rad": np.full(trace_shape, np.nan),
        "acceptance_probability_vs_time": np.full(trace_shape, np.nan),
        "logical_survival_vs_time": np.full(trace_shape, np.nan),
    }
    for iy, ej_a2 in enumerate(ej_a2_values):
        for ix, delta_ec in enumerate(deltas):
            ec_low = float(sweep["EC_A1_B1_GHz"])
            ec_high = ec_low + delta_ec

            def residual_a1(ej_a1_value: float) -> float:
                return (
                    _f01(raw, "A1", ec_low, ej_a1_value)
                    - _f01(raw, "A2", ec_high, ej_a2)
                    - target_a
                )

            def solve_b2(ej_b1_value: float) -> float:
                def residual_b2(ej_b2_value: float) -> float:
                    return (
                        _f01(raw, "B1", ec_low, ej_b1_value)
                        - _f01(raw, "B2", ec_high, ej_b2_value)
                        - target_b
                    )

                return brentq(residual_b2, *rail_bracket, xtol=1e-10)

            def residual(ej_b1: float) -> float:
                ej_b2_value = solve_b2(ej_b1)
                return _collision_detuning(
                    _system(raw, delta_ec, ej_a1, ej_a2, ej_b1, ej_b2_value),
                    channel,
                ) - target

            try:
                ej_a1 = brentq(residual_a1, *rail_bracket, xtol=1e-10)
                ej_b1 = brentq(residual, *bracket, xtol=1e-10)
                ej_b2 = solve_b2(ej_b1)
            except ValueError as error:
                print(f"Skipping Delta_EC={delta_ec:g}, EJ_A2={ej_a2:g}: {error}")
                continue
            system = _system(raw, delta_ec, ej_a1, ej_a2, ej_b1, ej_b2)
            achieved = _collision_detuning(system, channel)
            minimum_ratio = min(
                spec.EJ_GHz / spec.EC_GHz for spec in system.specs.values()
            )
            dispersive_a = abs(target_a) / _effective_exchange(system, "A1", "A2")
            dispersive_b = abs(target_b) / _effective_exchange(system, "B1", "B2")
            minimum_required_ratio = float(sweep.get("minimum_EJ_over_EC", 0.0))
            if minimum_ratio < minimum_required_ratio:
                print(
                    f"Skipping Delta_EC={delta_ec:g}, EJ_A2={ej_a2:g}: "
                    f"EJ/EC min={minimum_ratio:.3f}"
                )
                continue
            result = simulate_logical_basis(
                system,
                system.logical_basis(),
                evolution,
                odd_pi_tolerance_rad=phase_tolerance,
            )
            best = result["best_index"]
            arrays["solved_EJ_A1_GHz"][iy, ix] = ej_a1
            arrays["solved_EJ_B1_GHz"][iy, ix] = ej_b1
            arrays["solved_EJ_B2_GHz"][iy, ix] = ej_b2
            arrays["achieved_bare_detuning_GHz"][iy, ix] = achieved
            arrays["achieved_A1_minus_A2_detuning_GHz"][iy, ix] = (
                system.local["A1"].levels_GHz[1]
                - system.local["A2"].levels_GHz[1]
            )
            arrays["achieved_B1_minus_B2_detuning_GHz"][iy, ix] = (
                system.local["B1"].levels_GHz[1]
                - system.local["B2"].levels_GHz[1]
            )
            arrays["minimum_EJ_over_EC"][iy, ix] = minimum_ratio
            arrays["A_detuning_over_exchange"][iy, ix] = dispersive_a
            arrays["B_detuning_over_exchange"][iy, ix] = dispersive_b
            arrays["best_average_cz_fidelity"][iy, ix] = result[
                "best_average_cz_fidelity"
            ]
            arrays["best_time_ns"][iy, ix] = result["best_time_ns"]
            arrays["best_conditional_cz_fidelity"][iy, ix] = result[
                "best_conditional_cz_fidelity"
            ]
            arrays["best_conditional_time_ns"][iy, ix] = result[
                "best_conditional_time_ns"
            ]
            arrays["conditional_fidelity_end_slope_per_ns"][iy, ix] = result[
                "conditional_fidelity_end_slope_per_ns"
            ]
            _store_first_pi_metrics(
                arrays, (iy, ix), result, phase_tolerance
            )
            trace_arrays["conditional_cz_fidelity_vs_time"][iy, ix] = result[
                "conditional_cz_fidelity_vs_time"
            ]
            trace_arrays["conditional_phase_vs_time_rad"][iy, ix] = result[
                "conditional_phase_vs_time_rad"
            ]
            trace_arrays["acceptance_probability_vs_time"][iy, ix] = result[
                "acceptance_probability_vs_time"
            ]
            trace_arrays["logical_survival_vs_time"][iy, ix] = result[
                "logical_survival_vs_time"
            ]
            arrays["acceptance_at_best_unconditional"][iy, ix] = result[
                "acceptance_at_best_unconditional"
            ]
            arrays["acceptance_at_best_conditional"][iy, ix] = result[
                "acceptance_at_best_conditional"
            ]
            arrays["leakage_at_best"][iy, ix] = 1.0 - result[
                "logical_survival_vs_time"
            ][best]
            arrays["erasure_detectable_leakage_at_best"][iy, ix] = result[
                "average_erasure_detectable_vs_time"
            ][best]
            arrays["erasure_undetectable_leakage_at_best"][iy, ix] = result[
                "average_erasure_undetectable_vs_time"
            ][best]
            print(
                f"Delta_EC={delta_ec:g}, EJ_A1={ej_a1:g}, EJ_A2={ej_a2:g}, "
                f"EJ_B1={ej_b1:g}, EJ_B2={ej_b2:g}, "
                f"F_uncond={result['best_average_cz_fidelity']:.6f}, "
                f"F_cond={result['best_conditional_cz_fidelity']:.6f}"
            )

    output = Path(raw.get("output_directory", "../output/parameter_sweep_output"))
    if not output.is_absolute():
        output = config_path.parent / output
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "parameter_sweep_data.npz", **arrays)
    np.savez_compressed(
        output / "parameter_sweep_time_traces.npz",
        delta_EC_GHz=deltas,
        EJ_A2_GHz=ej_a2_values,
        times_ns=np.linspace(0.0, evolution.duration_ns, evolution.samples),
        **trace_arrays,
    )
    metadata = {
        "delta_EC_convention": "EC_A2 = EC_B2 = EC_A1_B1 + delta_EC; EC_A1 = EC_B1",
        "collision_channel": channel,
        "target_bare_detuning_GHz": target,
        "target_A1_minus_A2_detuning_GHz": target_a,
        "target_B1_minus_B2_detuning_GHz": target_b,
        "EJ_B1_bracket_GHz": list(bracket),
        "rail_EJ_bracket_GHz": list(rail_bracket),
        "minimum_EJ_over_EC": sweep.get("minimum_EJ_over_EC"),
        "fidelity_definitions": {
            "best_average_cz_fidelity": (
                "Maximum unconditional logical CZ fidelity over sampled times"
            ),
            "best_conditional_cz_fidelity": (
                "Maximum fidelity after conditioning on no detectable erasure; "
                "optimized independently over sampled times"
            ),
        },
        "config": str(config_path),
    }
    (output / "parameter_sweep_report.json").write_text(json.dumps(metadata, indent=2))
    make_colormaps(arrays, output / "parameter_sweep_colormaps.png")
    make_conditional_gate_speed_colormap(
        arrays,
        output / "parameter_sweep_conditional_gate_speed.png",
        evolution.duration_ns,
        "EJ_A2_GHz",
        1.0,
        r"$E_{J,A2}/h$ (GHz)",
        boundary_gate_time_ns=490.0,
    )
    make_best_conditional_gate_time_colormap(
        arrays,
        output / "parameter_sweep_best_conditional_gate_time.png",
        "EJ_A2_GHz",
        1.0,
        r"$E_{J,A2}/h$ (GHz)",
    )
    make_first_pi_gate_time_colormap(
        arrays,
        output / "parameter_sweep_first_pi_conditional_gate_time.png",
        "EJ_A2_GHz",
        1.0,
        r"$E_{J,A2}/h$ (GHz)",
    )
    print(f"Saved sweep results to {output}")
    return arrays


# %% Fixed-EJ_A2 sweep versus Delta_EC and EC_A2
def make_ec_a2_colormaps(data: dict[str, np.ndarray], output: Path) -> None:
    x = data["delta_EC_GHz"] * 1000.0
    y = data["EC_A2_GHz"] * 1000.0
    panels = (
        ("best_average_cz_fidelity", "Best logical CZ fidelity", "viridis"),
        ("leakage_at_best", "Nonlogical leakage at best fidelity", "magma"),
        ("solved_EJ_A1_GHz", "Solved $E_{J,A1}/h$ (GHz)", "viridis"),
        ("solved_EJ_B1_GHz", "Solved $E_{J,B1}/h$ (GHz)", "viridis"),
        ("solved_EJ_B2_GHz", "Solved $E_{J,B2}/h$ (GHz)", "viridis"),
        (
            "best_conditional_cz_fidelity",
            "Best conditional logical CZ fidelity",
            "viridis",
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (key, title, cmap_name) in zip(axes.flat, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")
        mesh = axis.pcolormesh(
            x, y, np.ma.masked_invalid(data[key]), shading="auto", cmap=cmap
        )
        axis.set_xlabel(r"$\Delta E_C/h$ (MHz)")
        axis.set_ylabel(r"$E_{C,A2}/h=E_{C,B2}/h$ (MHz)")
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_ec_a2_sweep(
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, np.ndarray]:
    """Sweep Delta_EC and EC_A2 with EJ_A2 fixed by the configuration."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    if raw.get("noise") is not None:
        raise ValueError("The logical fidelity sweep currently requires noise=null")

    sweep = raw["ec_a2_sweep"]
    old_sweep = raw["sweep"]
    deltas = _axis(sweep["delta_EC_GHz"])
    ec_a2_values = _axis(sweep["EC_A2_GHz"])
    ej_a2 = float(sweep.get("EJ_A2_GHz", 17.0))
    channel = sweep.get(
        "collision_channel", old_sweep.get("collision_channel", "0110_to_0200")
    )
    target = float(
        sweep.get("target_bare_detuning_GHz", old_sweep["target_bare_detuning_GHz"])
    )
    target_a = float(
        sweep.get(
            "target_A1_minus_A2_detuning_GHz",
            old_sweep["target_A1_minus_A2_detuning_GHz"],
        )
    )
    target_b = float(
        sweep.get(
            "target_B1_minus_B2_detuning_GHz",
            old_sweep["target_B1_minus_B2_detuning_GHz"],
        )
    )
    bracket = tuple(
        float(value)
        for value in sweep.get("EJ_B1_bracket_GHz", old_sweep["EJ_B1_bracket_GHz"])
    )
    rail_bracket = tuple(
        float(value)
        for value in sweep.get("rail_EJ_bracket_GHz", old_sweep["rail_EJ_bracket_GHz"])
    )
    minimum_required_ratio = float(
        sweep.get("minimum_EJ_over_EC", old_sweep.get("minimum_EJ_over_EC", 0.0))
    )
    shape = (ec_a2_values.size, deltas.size)
    names = (
        "solved_EJ_A1_GHz", "solved_EJ_B1_GHz", "solved_EJ_B2_GHz",
        "achieved_bare_detuning_GHz", "achieved_A1_minus_A2_detuning_GHz",
        "achieved_B1_minus_B2_detuning_GHz", "minimum_EJ_over_EC",
        "A_detuning_over_exchange", "B_detuning_over_exchange",
        "best_average_cz_fidelity", "best_time_ns",
        "best_conditional_cz_fidelity", "best_conditional_time_ns",
        "first_pi_conditional_time_ns", "conditional_cz_fidelity_at_first_pi",
        "acceptance_at_first_pi", "conditional_phase_error_at_first_pi_rad",
        "conditional_fidelity_end_slope_per_ns",
        "acceptance_at_best_unconditional", "acceptance_at_best_conditional",
        "leakage_at_best", "erasure_detectable_leakage_at_best",
        "erasure_undetectable_leakage_at_best",
    )
    arrays = {name: np.full(shape, np.nan) for name in names}
    arrays.update(
        {"delta_EC_GHz": deltas, "EC_A2_GHz": ec_a2_values, "EJ_A2_GHz": ej_a2}
    )
    evolution = EvolutionSpec(**raw["evolution"])
    phase_tolerance = float(sweep.get("conditional_phase_tolerance_rad", 0.05))
    trace_shape = shape + (evolution.samples,)
    trace_arrays = {
        "conditional_cz_fidelity_vs_time": np.full(trace_shape, np.nan),
        "conditional_phase_vs_time_rad": np.full(trace_shape, np.nan),
        "acceptance_probability_vs_time": np.full(trace_shape, np.nan),
        "logical_survival_vs_time": np.full(trace_shape, np.nan),
    }

    for iy, ec_high in enumerate(ec_a2_values):
        for ix, delta_ec in enumerate(deltas):
            ec_low = ec_high - delta_ec
            if ec_low <= 0.0:
                print(f"Skipping Delta_EC={delta_ec:g}, EC_A2={ec_high:g}: EC_A1 <= 0")
                continue

            def residual_a1(ej_a1_value: float) -> float:
                return (
                    _f01(raw, "A1", ec_low, ej_a1_value)
                    - _f01(raw, "A2", ec_high, ej_a2)
                    - target_a
                )

            def solve_b2(ej_b1_value: float) -> float:
                def residual_b2(ej_b2_value: float) -> float:
                    return (
                        _f01(raw, "B1", ec_low, ej_b1_value)
                        - _f01(raw, "B2", ec_high, ej_b2_value)
                        - target_b
                    )
                return brentq(residual_b2, *rail_bracket, xtol=1e-10)

            try:
                ej_a1 = brentq(residual_a1, *rail_bracket, xtol=1e-10)

                def collision_residual(ej_b1_value: float) -> float:
                    ej_b2_value = solve_b2(ej_b1_value)
                    system = _system_with_ec(
                        raw, ec_low, ec_high, ej_a1, ej_a2,
                        ej_b1_value, ej_b2_value,
                    )
                    return _collision_detuning(system, channel) - target

                ej_b1 = brentq(collision_residual, *bracket, xtol=1e-10)
                ej_b2 = solve_b2(ej_b1)
            except ValueError as error:
                print(f"Skipping Delta_EC={delta_ec:g}, EC_A2={ec_high:g}: {error}")
                continue

            system = _system_with_ec(
                raw, ec_low, ec_high, ej_a1, ej_a2, ej_b1, ej_b2
            )
            minimum_ratio = min(
                spec.EJ_GHz / spec.EC_GHz for spec in system.specs.values()
            )
            if minimum_ratio < minimum_required_ratio:
                print(
                    f"Skipping Delta_EC={delta_ec:g}, EC_A2={ec_high:g}: "
                    f"EJ/EC min={minimum_ratio:.3f}"
                )
                continue

            result = simulate_logical_basis(
                system,
                system.logical_basis(),
                evolution,
                odd_pi_tolerance_rad=phase_tolerance,
            )
            best = result["best_index"]
            arrays["solved_EJ_A1_GHz"][iy, ix] = ej_a1
            arrays["solved_EJ_B1_GHz"][iy, ix] = ej_b1
            arrays["solved_EJ_B2_GHz"][iy, ix] = ej_b2
            arrays["achieved_bare_detuning_GHz"][iy, ix] = _collision_detuning(system, channel)
            arrays["achieved_A1_minus_A2_detuning_GHz"][iy, ix] = system.local["A1"].levels_GHz[1] - system.local["A2"].levels_GHz[1]
            arrays["achieved_B1_minus_B2_detuning_GHz"][iy, ix] = system.local["B1"].levels_GHz[1] - system.local["B2"].levels_GHz[1]
            arrays["minimum_EJ_over_EC"][iy, ix] = minimum_ratio
            arrays["A_detuning_over_exchange"][iy, ix] = abs(target_a) / _effective_exchange(system, "A1", "A2")
            arrays["B_detuning_over_exchange"][iy, ix] = abs(target_b) / _effective_exchange(system, "B1", "B2")
            arrays["best_average_cz_fidelity"][iy, ix] = result["best_average_cz_fidelity"]
            arrays["best_time_ns"][iy, ix] = result["best_time_ns"]
            arrays["best_conditional_cz_fidelity"][iy, ix] = result["best_conditional_cz_fidelity"]
            arrays["best_conditional_time_ns"][iy, ix] = result["best_conditional_time_ns"]
            arrays["conditional_fidelity_end_slope_per_ns"][iy, ix] = result[
                "conditional_fidelity_end_slope_per_ns"
            ]
            _store_first_pi_metrics(
                arrays, (iy, ix), result, phase_tolerance
            )
            trace_arrays["conditional_cz_fidelity_vs_time"][iy, ix] = result[
                "conditional_cz_fidelity_vs_time"
            ]
            trace_arrays["conditional_phase_vs_time_rad"][iy, ix] = result[
                "conditional_phase_vs_time_rad"
            ]
            trace_arrays["acceptance_probability_vs_time"][iy, ix] = result[
                "acceptance_probability_vs_time"
            ]
            trace_arrays["logical_survival_vs_time"][iy, ix] = result[
                "logical_survival_vs_time"
            ]
            arrays["acceptance_at_best_unconditional"][iy, ix] = result["acceptance_at_best_unconditional"]
            arrays["acceptance_at_best_conditional"][iy, ix] = result["acceptance_at_best_conditional"]
            arrays["leakage_at_best"][iy, ix] = 1.0 - result["logical_survival_vs_time"][best]
            arrays["erasure_detectable_leakage_at_best"][iy, ix] = result["average_erasure_detectable_vs_time"][best]
            arrays["erasure_undetectable_leakage_at_best"][iy, ix] = result["average_erasure_undetectable_vs_time"][best]
            print(
                f"Delta_EC={delta_ec:g}, EC_A2={ec_high:g}, EJ_A2={ej_a2:g}, "
                f"F={result['best_average_cz_fidelity']:.6f}"
            )

    output = Path(sweep.get("output_directory", "../output/ec_a2_sweep_output"))
    if not output.is_absolute():
        output = config_path.parent / output
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "ec_a2_sweep_data.npz", **arrays)
    np.savez_compressed(
        output / "ec_a2_sweep_time_traces.npz",
        delta_EC_GHz=deltas,
        EC_A2_GHz=ec_a2_values,
        EJ_A2_GHz=np.asarray(ej_a2),
        times_ns=np.linspace(0.0, evolution.duration_ns, evolution.samples),
        **trace_arrays,
    )
    metadata = {
        "delta_EC_convention": "EC_A2 = EC_B2; EC_A1 = EC_B1 = EC_A2 - delta_EC",
        "fixed_EJ_A2_GHz": ej_a2,
        "collision_channel": channel,
        "target_bare_detuning_GHz": target,
        "target_A1_minus_A2_detuning_GHz": target_a,
        "target_B1_minus_B2_detuning_GHz": target_b,
        "minimum_EJ_over_EC": minimum_required_ratio,
        "config": str(config_path),
    }
    (output / "ec_a2_sweep_report.json").write_text(json.dumps(metadata, indent=2))
    make_ec_a2_colormaps(arrays, output / "ec_a2_sweep_colormaps.png")
    make_conditional_gate_speed_colormap(
        arrays,
        output / "ec_a2_sweep_conditional_gate_speed.png",
        evolution.duration_ns,
        "EC_A2_GHz",
        1000.0,
        r"$E_{C,A2}/h=E_{C,B2}/h$ (MHz)",
        boundary_gate_time_ns=490.0,
    )
    make_best_conditional_gate_time_colormap(
        arrays,
        output / "ec_a2_sweep_best_conditional_gate_time.png",
        "EC_A2_GHz",
        1000.0,
        r"$E_{C,A2}/h=E_{C,B2}/h$ (MHz)",
    )
    make_first_pi_gate_time_colormap(
        arrays,
        output / "ec_a2_sweep_first_pi_conditional_gate_time.png",
        "EC_A2_GHz",
        1000.0,
        r"$E_{C,A2}/h=E_{C,B2}/h$ (MHz)",
    )
    print(f"Saved fixed-EJ_A2 sweep results to {output}")
    return arrays


# %% Dark-region case-study helpers
def _solve_ej_sweep_point(raw: dict, delta_ec: float, ej_a2: float):
    """Solve the three constrained EJ values for one point of the first sweep."""
    sweep = raw["sweep"]
    ec_low = float(sweep["EC_A1_B1_GHz"])
    ec_high = ec_low + delta_ec
    target_a = float(sweep["target_A1_minus_A2_detuning_GHz"])
    target_b = float(sweep["target_B1_minus_B2_detuning_GHz"])
    target_cz = float(sweep["target_bare_detuning_GHz"])
    channel = sweep.get("collision_channel", "0110_to_0200")
    bracket = tuple(float(value) for value in sweep["EJ_B1_bracket_GHz"])
    rail_bracket = tuple(float(value) for value in sweep["rail_EJ_bracket_GHz"])

    def residual_a1(ej_a1_value: float) -> float:
        return (
            _f01(raw, "A1", ec_low, ej_a1_value)
            - _f01(raw, "A2", ec_high, ej_a2)
            - target_a
        )

    ej_a1 = brentq(residual_a1, *rail_bracket, xtol=1e-10)

    def solve_b2(ej_b1_value: float) -> float:
        def residual_b2(ej_b2_value: float) -> float:
            return (
                _f01(raw, "B1", ec_low, ej_b1_value)
                - _f01(raw, "B2", ec_high, ej_b2_value)
                - target_b
            )

        return brentq(residual_b2, *rail_bracket, xtol=1e-10)

    def residual_b1(ej_b1_value: float) -> float:
        ej_b2_value = solve_b2(ej_b1_value)
        system = _system(
            raw, delta_ec, ej_a1, ej_a2, ej_b1_value, ej_b2_value
        )
        return _collision_detuning(system, channel) - target_cz

    ej_b1 = brentq(residual_b1, *bracket, xtol=1e-10)
    ej_b2 = solve_b2(ej_b1)
    return _system(raw, delta_ec, ej_a1, ej_a2, ej_b1, ej_b2)


def _make_case_study_plot(result: dict, title: str, output: Path) -> None:
    times = result["times_ns"]
    total_leakage = 1.0 - result["logical_survival_vs_time"]
    detectable = result["average_erasure_detectable_vs_time"]
    undetectable = result["average_erasure_undetectable_vs_time"]
    sector_names = {
        "logical": "logical manifold",
        "sector_11": "one excitation per rail",
        "sector_20": "two in A, zero in B",
        "sector_02": "zero in A, two in B",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes[0, 0].plot(times, result["average_cz_fidelity_vs_time"], label="unconditional")
    axes[0, 0].plot(times, result["conditional_cz_fidelity_vs_time"], label="conditional")
    axes[0, 0].set_ylabel("CZ fidelity")
    axes[0, 0].legend()

    axes[0, 1].plot(times, total_leakage, label="total nonlogical leakage")
    axes[0, 1].plot(times, detectable, label="erasure-detectable")
    axes[0, 1].plot(times, undetectable, label="undetectable nonlogical")
    axes[0, 1].set_ylabel("Average probability")
    axes[0, 1].legend()

    axes[1, 0].plot(times, result["conditional_phase_vs_time_rad"])
    axes[1, 0].axhline(np.pi, color="black", ls=":", label=r"$\pi$")
    axes[1, 0].axhline(-np.pi, color="black", ls=":")
    axes[1, 0].set_ylabel("Conditional phase (rad)")
    axes[1, 0].set_xlabel("Time (ns)")
    axes[1, 0].legend()

    for key, label in sector_names.items():
        values = np.mean([run[key] for run in result["runs"]], axis=0)
        axes[1, 1].plot(times, values, label=label)
    axes[1, 1].set_ylabel("Mean population")
    axes[1, 1].set_xlabel("Time (ns)")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.axvline(result["best_time_ns"], color="C0", ls="--", alpha=0.7)
        axis.axvline(
            result["best_conditional_time_ns"], color="C1", ls="-.", alpha=0.7
        )
        axis.grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _pair_dressed_manifolds(system: FourTransmonSystem, pair: str) -> dict:
    """Return dressed pair states classified by dominant local excitation sector."""
    if pair == "A":
        left, right, coupling_key = "A1", "A2", "A1_A2"
    else:
        left, right, coupling_key = "B1", "B2", "B1_B2"
    lm, rm = system.local[left], system.local[right]
    identity = qt.qeye(system.dim)
    h_pair = TWOPI * (
        4.0 * lm.spec.EC_GHz * qt.tensor(lm.n2_op, identity)
        - lm.spec.EJ_GHz * qt.tensor(lm.cos_phi_op, identity)
        + 4.0 * rm.spec.EC_GHz * qt.tensor(identity, rm.n2_op)
        - rm.spec.EJ_GHz * qt.tensor(identity, rm.cos_phi_op)
        + system.couplings_GHz[coupling_key] * qt.tensor(lm.n_op, rm.n_op)
    )
    energies, states = h_pair.eigenstates()
    energies = np.asarray(energies) / TWOPI

    def bare(left_level: int, right_level: int):
        return qt.tensor(lm.level_kets[left_level], rm.level_kets[right_level])

    sector_states = {
        excitation: [
            bare(n_left, excitation - n_left)
            for n_left in range(excitation + 1)
            if n_left < system.dim and excitation - n_left < system.dim
        ]
        for excitation in (0, 1, 2)
    }
    weights = {
        excitation: np.asarray(
            [
                sum(abs(reference.overlap(state)) ** 2 for reference in references)
                for state in states
            ]
        )
        for excitation, references in sector_states.items()
    }
    indices = {
        0: [int(np.argmax(weights[0]))],
        1: sorted(np.argsort(weights[1])[-2:], key=lambda i: energies[i]),
        2: sorted(np.argsort(weights[2])[-3:], key=lambda i: energies[i]),
    }
    two_excitation = []
    for branch, state_index in enumerate(indices[2]):
        components = {
            "20": float(abs(bare(2, 0).overlap(states[state_index])) ** 2),
            "11": float(abs(bare(1, 1).overlap(states[state_index])) ** 2),
            "02": float(abs(bare(0, 2).overlap(states[state_index])) ** 2),
        }
        two_excitation.append(
            {
                "branch": branch,
                "state": states[state_index],
                "energy_GHz": float(energies[state_index]),
                "components": components,
            }
        )
    return {
        "ground": states[indices[0][0]],
        "ground_energy_GHz": float(energies[indices[0][0]]),
        "logical": [states[index] for index in indices[1]],
        "logical_energies_GHz": [float(energies[index]) for index in indices[1]],
        "two_excitation": two_excitation,
    }


def dressed_basis_diagnostic(system: FourTransmonSystem, output: Path) -> dict:
    """Compare pre-CZ dressed rail products with full-system eigenstates."""
    rail_a = _pair_dressed_manifolds(system, "A")
    rail_b = _pair_dressed_manifolds(system, "B")
    row_states, row_labels, row_energies = [], [], []
    logical_rows = []
    for ia, state_a in enumerate(rail_a["logical"]):
        for ib, state_b in enumerate(rail_b["logical"]):
            logical_rows.append(len(row_states))
            row_states.append(qt.tensor(state_a, state_b))
            row_labels.append(f"|{ia}{ib}>L")
            row_energies.append(
                rail_a["logical_energies_GHz"][ia]
                + rail_b["logical_energies_GHz"][ib]
            )
    a_leak_rows = []
    for branch in rail_a["two_excitation"]:
        a_leak_rows.append(len(row_states))
        row_states.append(qt.tensor(branch["state"], rail_b["ground"]))
        row_labels.append(f"A20[{branch['branch']}] x B0")
        row_energies.append(branch["energy_GHz"] + rail_b["ground_energy_GHz"])
    for branch in rail_b["two_excitation"]:
        row_states.append(qt.tensor(rail_a["ground"], branch["state"]))
        row_labels.append(f"A0 x B20[{branch['branch']}]")
        row_energies.append(rail_a["ground_energy_GHz"] + branch["energy_GHz"])

    target_a_branch = int(
        np.argmax([branch["components"]["02"] for branch in rail_a["two_excitation"]])
    )
    target_x_row = a_leak_rows[target_a_branch]
    h_cz = (
        system.couplings_GHz["A2_B1"]
        * system.n_full["A2"]
        * system.n_full["B1"]
    )
    logical_candidates = []
    for row in logical_rows:
        coupling = row_states[row].dag() * h_cz * row_states[target_x_row]
        if hasattr(coupling, "full"):
            coupling = coupling.full()[0, 0]
        detuning = row_energies[target_x_row] - row_energies[row]
        gap = np.sqrt(detuning**2 + 4.0 * abs(coupling) ** 2)
        logical_candidates.append(
            {
                "logical_label": row_labels[row],
                "detuning_GHz": float(detuning),
                "coupling_GHz": float(abs(coupling)),
                "two_state_mixing": float(2.0 * abs(coupling) / gap) if gap else 1.0,
            }
        )
    target_l = max(logical_candidates, key=lambda item: item["two_state_mixing"])

    full_energies_rad, full_vectors = system.spectral_decomposition()
    full_energies = full_energies_rad / TWOPI
    overlaps_all = np.asarray(
        [
            [
                abs(state.overlap(qt.Qobj(full_vectors[:, k, None], dims=state.dims))) ** 2
                for k in range(full_vectors.shape[1])
            ]
            for state in row_states
        ]
    )
    column_count = min(14, full_vectors.shape[1])
    columns = np.argsort(np.sum(overlaps_all, axis=0))[-column_count:]
    columns = columns[np.argsort(full_energies[columns])]
    overlaps = overlaps_all[:, columns]

    figure, axis = plt.subplots(figsize=(14, 7))
    image = axis.imshow(overlaps, aspect="auto", origin="upper", vmin=0.0, vmax=1.0)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_xticks(
        np.arange(column_count),
        [f"E{k}\n{full_energies[k]:.3f}" for k in columns],
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("Full-system eigenstate and energy (GHz)")
    axis.set_title("Pre-CZ dressed-state overlap matrix")
    figure.colorbar(image, ax=axis, label=r"$|\langle\psi_j^{(0)}|E_k\rangle|^2$")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)

    return {
        "row_labels": row_labels,
        "row_energies_GHz": [float(value) for value in row_energies],
        "selected_full_indices": columns.tolist(),
        "selected_full_energies_GHz": full_energies[columns].tolist(),
        "overlap_matrix": overlaps.tolist(),
        "target_leakage_label": row_labels[target_x_row],
        "target_A_two_excitation_branch": target_a_branch,
        "A_two_excitation_branches": [
            {
                "branch": branch["branch"],
                "energy_GHz": branch["energy_GHz"],
                "components": branch["components"],
            }
            for branch in rail_a["two_excitation"]
        ],
        "logical_to_target_leakage": logical_candidates,
        "most_resonant_logical_candidate": target_l,
    }


def run_dark_region_case_studies(config_path: str | Path = CONFIG_PATH) -> list[dict]:
    """Diagnose selected dark points using a longer time-domain simulation."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    case_config = raw["dark_region_case_study"]
    evolution = EvolutionSpec(**case_config["evolution"])
    output = Path(case_config.get("output_directory", "../output/dark_region_case_study"))
    if not output.is_absolute():
        output = config_path.parent / output
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    comparison_runs = []

    for index, point in enumerate(case_config["points"], start=1):
        delta_ec = float(point["delta_EC_GHz"])
        ej_a2 = float(point["EJ_A2_GHz"])
        case_label = str(point.get("label", f"point_{index}"))
        system = _solve_ej_sweep_point(raw, delta_ec, ej_a2)
        result = simulate_logical_basis(system, system.logical_basis(), evolution)
        collision_data = cz_collision_splittings(system)
        stem = (
            f"point_{index}_{case_label}_dEC_{1000*delta_ec:.1f}MHz_"
            f"EJA2_{ej_a2:.1f}GHz"
        )
        dressed_diagnostic = dressed_basis_diagnostic(
            system, output / f"{stem}_dressed_overlap.png"
        )
        best = result["best_index"]
        summary = {
            "index": index,
            "label": case_label,
            "delta_EC_GHz": delta_ec,
            "EJ_A2_GHz": ej_a2,
            "solved_EJ_GHz": {
                label: system.specs[label].EJ_GHz for label in LABELS
            },
            "transition_table": transition_table(system),
            "CZ_collision_diagnostics": {
                name: {
                    "bare_detuning_GHz": data["bare_detuning_GHz"],
                    "resonant_two_level_gap_GHz": data["resonant_two_level_gap_GHz"],
                    "full_dressed_splitting_GHz": data["full_dressed_splitting_GHz"],
                    "dressed_subspace_weights": data["dressed_subspace_weights"].tolist(),
                }
                for name, data in collision_data.items()
            },
            "dressed_basis_diagnostic": dressed_diagnostic,
            "best_time_ns": result["best_time_ns"],
            "best_average_cz_fidelity": result["best_average_cz_fidelity"],
            "nonlogical_leakage_at_best": float(
                1.0 - result["logical_survival_vs_time"][best]
            ),
            # Retained for compatibility with reports generated by older code.
            "logical_leakage_at_best": float(
                1.0 - result["logical_survival_vs_time"][best]
            ),
            "erasure_detectable_leakage_at_best": float(
                result["average_erasure_detectable_vs_time"][best]
            ),
            "erasure_undetectable_leakage_at_best": float(
                result["average_erasure_undetectable_vs_time"][best]
            ),
            "best_conditional_time_ns": result["best_conditional_time_ns"],
            "best_conditional_cz_fidelity": result["best_conditional_cz_fidelity"],
            "acceptance_at_best_conditional": result["acceptance_at_best_conditional"],
        }
        summaries.append(summary)
        comparison_runs.append((case_label, delta_ec, ej_a2, result))
        np.savez(
            output / f"{stem}.npz",
            times_ns=result["times_ns"],
            unconditional_fidelity=result["average_cz_fidelity_vs_time"],
            conditional_fidelity=result["conditional_cz_fidelity_vs_time"],
            conditional_phase_rad=result["conditional_phase_vs_time_rad"],
            nonlogical_leakage=1.0 - result["logical_survival_vs_time"],
            logical_leakage=1.0 - result["logical_survival_vs_time"],
            detectable_leakage=result["average_erasure_detectable_vs_time"],
            undetectable_leakage=result["average_erasure_undetectable_vs_time"],
        )
        _make_case_study_plot(
            result,
            rf"Point {index} ({case_label}): "
            rf"$\Delta E_C/h={1000*delta_ec:.1f}$ MHz, "
            rf"$E_{{J,A2}}/h={ej_a2:.1f}$ GHz",
            output / f"{stem}.png",
        )
        print(
            f"Point {index} ({case_label}): "
            f"F={result['best_average_cz_fidelity']:.6f} "
            f"at {result['best_time_ns']:.3f} ns"
        )

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for index, (case_label, delta_ec, ej_a2, result) in enumerate(comparison_runs, start=1):
        label = (
            f"{index} {case_label}: {1000*delta_ec:.1f} MHz, "
            f"EJ,A2={ej_a2:.1f} GHz"
        )
        times = result["times_ns"]
        axes[0, 0].plot(times, result["average_cz_fidelity_vs_time"], label=label)
        axes[0, 1].plot(times, result["conditional_cz_fidelity_vs_time"], label=label)
        axes[1, 0].plot(times, 1.0 - result["logical_survival_vs_time"], label=label)
        axes[1, 1].plot(times, result["conditional_phase_vs_time_rad"], label=label)
    axes[0, 0].set_title("Unconditional logical CZ fidelity")
    axes[0, 1].set_title("Conditional logical CZ fidelity")
    axes[1, 0].set_title("Nonlogical leakage")
    axes[1, 1].set_title("Conditional phase")
    axes[1, 1].axhline(np.pi, color="black", ls=":")
    axes[1, 1].axhline(-np.pi, color="black", ls=":")
    for axis in axes.flat:
        axis.set_xlabel("Time (ns)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "case_study_comparison.png", dpi=180)
    plt.close(figure)
    (output / "case_study_report.json").write_text(json.dumps(summaries, indent=2))
    return summaries


# %% Standard full-output case studies
def run_standard_case_studies(config_path: str | Path = CONFIG_PATH) -> list[Path]:
    """Run each selected point through the ordinary run_simulation.py workflow."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    case_config = raw["dark_region_case_study"]
    base_output = config_path.parent / case_config.get(
        "output_directory", "../output/dark_region_case_study"
    )
    configuration_directory = PACKAGE_ROOT / "configs" / "case_studies"
    configuration_directory.mkdir(parents=True, exist_ok=True)
    generated_configs = []

    for index, point in enumerate(case_config["points"], start=1):
        delta_ec = float(point["delta_EC_GHz"])
        ej_a2 = float(point["EJ_A2_GHz"])
        case_label = str(point.get("label", f"point_{index}"))
        system = _solve_ej_sweep_point(raw, delta_ec, ej_a2)
        stem = (
            f"point_{index}_{case_label}_dEC_{1000*delta_ec:.1f}MHz_"
            f"EJA2_{ej_a2:.1f}GHz"
        )
        point_output = base_output / stem
        point_config = {
            "transmons": {
                label: {
                    "EC_GHz": system.specs[label].EC_GHz,
                    "EJ_GHz": system.specs[label].EJ_GHz,
                    "ng": system.specs[label].ng,
                }
                for label in LABELS
            },
            "couplings_GHz": raw["couplings_GHz"],
            "basis": raw["basis"],
            "evolution": case_config["evolution"],
            "noise": raw.get("noise"),
            "uncoupled_spectrum": raw.get(
                "uncoupled_spectrum", {"collision_window_GHz": 0.150}
            ),
            "output_directory": str(point_output),
        }
        point_config_path = configuration_directory / f"{stem}.json"
        point_config_path.write_text(json.dumps(point_config, indent=2))
        generated_configs.append(point_config_path)
        print(f"Running standard simulation for {stem}")
        subprocess.run(
            [sys.executable, str(PACKAGE_ROOT / "run.py"), "simulate", str(point_config_path)],
            cwd=PACKAGE_ROOT.parent,
            check=True,
        )

    print(f"Saved five standard simulation cases to {base_output}")
    return generated_configs


# %% Gate-time selection diagnostics along fixed-EJ_A2 slices
def run_gate_time_slice_diagnostics(
    config_path: str | Path = CONFIG_PATH,
) -> dict[float, dict[str, np.ndarray]]:
    """Save complete time traces along selected fixed-EJ_A2 sweep slices."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    settings = raw.get(
        "gate_time_slice_diagnostics",
        {
            "EJ_A2_GHz": [20.0, 12.5],
            "conditional_phase_tolerance_rad": 0.05,
            "output_directory": "../output/parameter_sweep_output/gate_time_slice_diagnostics",
        },
    )
    deltas = _axis(raw["sweep"]["delta_EC_GHz"])
    ej_values = np.asarray(settings["EJ_A2_GHz"], dtype=float)
    phase_tolerance = float(settings.get("conditional_phase_tolerance_rad", 0.05))
    evolution = EvolutionSpec(**raw["evolution"])
    output = Path(settings["output_directory"])
    if not output.is_absolute():
        output = config_path.parent / output
    output.mkdir(parents=True, exist_ok=True)
    collected = {}

    for ej_a2 in ej_values:
        times = np.linspace(0.0, evolution.duration_ns, evolution.samples)
        shape = (deltas.size, times.size)
        conditional_fidelity = np.full(shape, np.nan)
        conditional_phase = np.full(shape, np.nan)
        acceptance = np.full(shape, np.nan)
        logical_survival = np.full(shape, np.nan)
        solved_ej = np.full((deltas.size, 3), np.nan)
        best_time = np.full(deltas.size, np.nan)
        best_fidelity = np.full(deltas.size, np.nan)
        first_pi_time = np.full(deltas.size, np.nan)
        fidelity_at_first_pi = np.full(deltas.size, np.nan)
        acceptance_at_first_pi = np.full(deltas.size, np.nan)

        for index, delta_ec in enumerate(deltas):
            try:
                system = _solve_ej_sweep_point(raw, float(delta_ec), float(ej_a2))
            except ValueError as error:
                print(
                    f"Skipping slice EJ_A2={ej_a2:g}, Delta_EC={delta_ec:g}: "
                    f"{error}"
                )
                continue
            minimum_ratio = min(
                spec.EJ_GHz / spec.EC_GHz for spec in system.specs.values()
            )
            minimum_required_ratio = float(
                raw["sweep"].get("minimum_EJ_over_EC", 0.0)
            )
            if minimum_ratio < minimum_required_ratio:
                print(
                    f"Skipping slice EJ_A2={ej_a2:g}, Delta_EC={delta_ec:g}: "
                    f"EJ/EC min={minimum_ratio:.3f}"
                )
                continue

            result = simulate_logical_basis(
                system,
                system.logical_basis(),
                evolution,
                odd_pi_tolerance_rad=phase_tolerance,
            )
            conditional_fidelity[index] = result["conditional_cz_fidelity_vs_time"]
            conditional_phase[index] = result["conditional_phase_vs_time_rad"]
            acceptance[index] = result["acceptance_probability_vs_time"]
            logical_survival[index] = result["logical_survival_vs_time"]
            solved_ej[index] = [
                system.specs["A1"].EJ_GHz,
                system.specs["B1"].EJ_GHz,
                system.specs["B2"].EJ_GHz,
            ]
            best_time[index] = result["best_conditional_time_ns"]
            best_fidelity[index] = result["best_conditional_cz_fidelity"]
            first_pi = _first_odd_pi_index(
                result["conditional_phase_vs_time_rad"], phase_tolerance
            )
            if first_pi is not None:
                first_pi_time[index] = result["times_ns"][first_pi]
                fidelity_at_first_pi[index] = result[
                    "conditional_cz_fidelity_vs_time"
                ][first_pi]
                acceptance_at_first_pi[index] = result[
                    "acceptance_probability_vs_time"
                ][first_pi]
            print(
                f"Slice EJ_A2={ej_a2:g}, Delta_EC={1000*delta_ec:.2f} MHz: "
                f"t_best={best_time[index]:.3f} ns, "
                f"t_first_pi={first_pi_time[index]:.3f} ns"
            )

        slice_data = {
            "delta_EC_GHz": deltas,
            "EJ_A2_GHz": np.asarray(ej_a2),
            "times_ns": times,
            "conditional_cz_fidelity_vs_time": conditional_fidelity,
            "conditional_phase_vs_time_rad": conditional_phase,
            "acceptance_probability_vs_time": acceptance,
            "logical_survival_vs_time": logical_survival,
            "solved_EJ_A1_B1_B2_GHz": solved_ej,
            "best_conditional_time_ns": best_time,
            "best_conditional_cz_fidelity": best_fidelity,
            "first_pi_conditional_time_ns": first_pi_time,
            "conditional_cz_fidelity_at_first_pi": fidelity_at_first_pi,
            "acceptance_at_first_pi": acceptance_at_first_pi,
            "conditional_phase_tolerance_rad": np.asarray(phase_tolerance),
        }
        stem = f"EJA2_{ej_a2:g}GHz".replace(".", "p")
        np.savez(output / f"{stem}_time_traces.npz", **slice_data)

        colors = plt.get_cmap("viridis")(
            np.linspace(0.0, 1.0, deltas.size)
        )
        figure, axes = plt.subplots(
            2, 1, figsize=(11, 9), sharex=True, constrained_layout=True
        )
        for index, (delta_ec, color) in enumerate(zip(deltas, colors)):
            label = f"{1000*delta_ec:.1f} MHz"
            axes[0].plot(
                times, conditional_fidelity[index], color=color, lw=1.0, label=label
            )
            axes[1].plot(
                times, conditional_phase[index] / np.pi, color=color, lw=1.0
            )
        axes[0].axhline(0.85, color="red", ls="--", lw=1.0)
        axes[0].set_ylabel("Conditional CZ fidelity")
        axes[0].set_ylim(0.35, 1.01)
        axes[0].set_title(rf"$E_{{J,A2}}/h={ej_a2:g}$ GHz")
        axes[1].set_xlabel("Time (ns)")
        axes[1].set_ylabel(r"Accumulated conditional phase / $\pi$")
        axes[1].axhline(1.0, color="black", ls=":", lw=1.0)
        axes[1].axhline(-1.0, color="black", ls=":", lw=1.0)
        for axis in axes:
            axis.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, title=r"$\Delta E_C/h$", loc="outside right center",
            ncol=1, fontsize=7,
        )
        figure.savefig(output / f"{stem}_time_traces.png", dpi=180)
        plt.close(figure)
        collected[float(ej_a2)] = slice_data

    figure, axes = plt.subplots(
        len(ej_values), 1, figsize=(9, 4.2 * len(ej_values)),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, ej_a2 in zip(axes, ej_values):
        data = collected[float(ej_a2)]
        x = data["delta_EC_GHz"] * 1000.0
        axis.plot(
            x, data["best_conditional_time_ns"], "o-", ms=4,
            label="Global best conditional fidelity",
        )
        axis.plot(
            x, data["first_pi_conditional_time_ns"], "s-", ms=4,
            label=rf"First odd-$\pi$ arrival ($\pm${phase_tolerance:g} rad)",
        )
        axis.set_ylabel("Gate time (ns)")
        axis.set_title(rf"$E_{{J,A2}}/h={ej_a2:g}$ GHz")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel(r"$\Delta E_C/h$ (MHz)")
    figure.savefig(output / "gate_time_selection_comparison.png", dpi=180)
    plt.close(figure)
    return collected


# %% Exact-Delta_EC vertical slices versus EJ_A2
def run_vertical_slice_diagnostics(
    config_path: str | Path = CONFIG_PATH,
) -> dict[float, dict[str, np.ndarray]]:
    """Save full time traces and coupling diagnostics at fixed Delta_EC."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    settings = raw["vertical_slice_diagnostics"]
    delta_values = np.asarray(settings["delta_EC_GHz"], dtype=float)
    ej_values = _axis(raw["sweep"]["EJ_A2_GHz"])
    phase_tolerance = float(settings.get("conditional_phase_tolerance_rad", 0.05))
    evolution = EvolutionSpec(**raw["evolution"])
    output = Path(settings["output_directory"])
    if not output.is_absolute():
        output = config_path.parent / output
    output.mkdir(parents=True, exist_ok=True)
    collected = {}

    for delta_ec in delta_values:
        times = np.linspace(0.0, evolution.duration_ns, evolution.samples)
        trace_shape = (ej_values.size, times.size)
        fidelity_traces = np.full(trace_shape, np.nan)
        phase_traces = np.full(trace_shape, np.nan)
        acceptance_traces = np.full(trace_shape, np.nan)
        survival_traces = np.full(trace_shape, np.nan)
        solved_ej = np.full((ej_values.size, 3), np.nan)
        bare_coupling = np.full(ej_values.size, np.nan)
        fourth_root_proxy = np.full(ej_values.size, np.nan)
        best_time = np.full(ej_values.size, np.nan)
        best_fidelity = np.full(ej_values.size, np.nan)
        first_pi_time = np.full(ej_values.size, np.nan)
        fidelity_at_first_pi = np.full(ej_values.size, np.nan)
        acceptance_at_first_pi = np.full(ej_values.size, np.nan)

        for index, ej_a2 in enumerate(ej_values):
            try:
                system = _solve_ej_sweep_point(raw, float(delta_ec), float(ej_a2))
            except ValueError as error:
                print(
                    f"Skipping Delta_EC={1000*delta_ec:g} MHz, "
                    f"EJ_A2={ej_a2:g}: {error}"
                )
                continue
            if min(
                spec.EJ_GHz / spec.EC_GHz for spec in system.specs.values()
            ) < float(raw["sweep"].get("minimum_EJ_over_EC", 0.0)):
                continue

            result = simulate_logical_basis(system, system.logical_basis(), evolution)
            fidelity_traces[index] = result["conditional_cz_fidelity_vs_time"]
            phase_traces[index] = result["conditional_phase_vs_time_rad"]
            acceptance_traces[index] = result["acceptance_probability_vs_time"]
            survival_traces[index] = result["logical_survival_vs_time"]
            solved_ej[index] = [
                system.specs["A1"].EJ_GHz,
                system.specs["B1"].EJ_GHz,
                system.specs["B2"].EJ_GHz,
            ]
            collision = cz_collision_splittings(system)["0110_to_0200"]
            bare_coupling[index] = abs(complex(collision["direct_coupling_GHz"]))
            fourth_root_proxy[index] = (
                (system.specs["A2"].EJ_GHz / system.specs["A2"].EC_GHz) ** 0.25
                * (system.specs["B1"].EJ_GHz / system.specs["B1"].EC_GHz) ** 0.25
            )
            best_time[index] = result["best_conditional_time_ns"]
            best_fidelity[index] = result["best_conditional_cz_fidelity"]
            first_pi = _first_odd_pi_index(
                result["conditional_phase_vs_time_rad"], phase_tolerance
            )
            if first_pi is not None:
                first_pi_time[index] = result["times_ns"][first_pi]
                fidelity_at_first_pi[index] = fidelity_traces[index, first_pi]
                acceptance_at_first_pi[index] = acceptance_traces[index, first_pi]
            print(
                f"Delta_EC={1000*delta_ec:g} MHz, EJ_A2={ej_a2:g}: "
                f"|V_bare|={1000*bare_coupling[index]:.3f} MHz, "
                f"t_first_pi={first_pi_time[index]:.3f} ns"
            )

        data = {
            "delta_EC_GHz": np.asarray(delta_ec),
            "EJ_A2_GHz": ej_values,
            "times_ns": times,
            "conditional_cz_fidelity_vs_time": fidelity_traces,
            "conditional_phase_vs_time_rad": phase_traces,
            "acceptance_probability_vs_time": acceptance_traces,
            "logical_survival_vs_time": survival_traces,
            "solved_EJ_A1_B1_B2_GHz": solved_ej,
            "bare_0110_to_0200_coupling_GHz": bare_coupling,
            "fourth_root_charge_matrix_proxy": fourth_root_proxy,
            "best_conditional_time_ns": best_time,
            "best_conditional_cz_fidelity": best_fidelity,
            "first_pi_conditional_time_ns": first_pi_time,
            "conditional_cz_fidelity_at_first_pi": fidelity_at_first_pi,
            "acceptance_at_first_pi": acceptance_at_first_pi,
            "conditional_phase_tolerance_rad": np.asarray(phase_tolerance),
        }
        stem = f"deltaEC_{1000*delta_ec:g}MHz".replace(".", "p")
        np.savez(output / f"{stem}_vertical_slice.npz", **data)
        collected[float(delta_ec)] = data

        # Match the horizontal-slice time-domain plot: fidelity above, phase below.
        colors = plt.get_cmap("plasma")(np.linspace(0.0, 1.0, ej_values.size))
        figure, axes = plt.subplots(
            2, 1, figsize=(11, 9), sharex=True, constrained_layout=True
        )
        for index, (ej_a2, color) in enumerate(zip(ej_values, colors)):
            axes[0].plot(
                times, fidelity_traces[index], color=color, lw=0.9,
                label=f"{ej_a2:g} GHz",
            )
            axes[1].plot(times, phase_traces[index] / np.pi, color=color, lw=0.9)
        axes[0].axhline(0.85, color="red", ls="--", lw=1.0)
        axes[0].set_ylabel("Conditional CZ fidelity")
        axes[0].set_ylim(0.35, 1.01)
        axes[0].set_title(rf"$\Delta E_C/h={1000*delta_ec:g}$ MHz")
        axes[1].set_xlabel("Time (ns)")
        axes[1].set_ylabel(r"Accumulated conditional phase / $\pi$")
        axes[1].axhline(1.0, color="black", ls=":", lw=1.0)
        axes[1].axhline(-1.0, color="black", ls=":", lw=1.0)
        for axis in axes:
            axis.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, title=r"$E_{J,A2}/h$", loc="outside right center",
            fontsize=7,
        )
        figure.savefig(output / f"{stem}_time_traces.png", dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True, constrained_layout=True)
    for delta_ec, color in zip(delta_values, ("C0", "C1")):
        data = collected[float(delta_ec)]
        label = rf"$\Delta E_C/h={1000*delta_ec:g}$ MHz"
        actual = 1000.0 * data["bare_0110_to_0200_coupling_GHz"]
        proxy = data["fourth_root_charge_matrix_proxy"]
        finite = np.isfinite(actual) & np.isfinite(proxy)
        scaled_proxy = np.full_like(proxy, np.nan)
        if np.any(finite):
            scale = np.sum(actual[finite] * proxy[finite]) / np.sum(proxy[finite] ** 2)
            scaled_proxy[finite] = scale * proxy[finite]
        axes[0].plot(ej_values, actual, "o-", color=color, ms=4, label=label)
        axes[0].plot(
            ej_values, scaled_proxy, "--", color=color, alpha=0.8,
            label=label + " fourth-root proxy",
        )
        axes[1].plot(
            ej_values, data["best_conditional_time_ns"], "o--", color=color,
            ms=4, alpha=0.65,
        )
        axes[1].plot(
            ej_values, data["first_pi_conditional_time_ns"], "s-", color=color,
            ms=4, label=label,
        )
        axes[2].plot(
            ej_values, data["conditional_cz_fidelity_at_first_pi"], "s-",
            color=color, ms=4, label=label,
        )
    axes[0].set_ylabel(r"$|V_{\rm bare}|/h$ (MHz)")
    axes[0].set_title("Bare collision matrix element and fourth-root proxy")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("Gate time (ns)")
    axes[1].set_title(r"Dashed circles: global optimum; squares: first odd-$\pi$")
    axes[1].legend(fontsize=8)
    axes[2].axhline(0.85, color="red", ls="--", lw=1.0)
    axes[2].set_ylabel(r"Conditional fidelity at first odd-$\pi$")
    axes[2].set_xlabel(r"$E_{J,A2}/h$ (GHz)")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output / "vertical_slice_scaling_comparison.png", dpi=180)
    plt.close(figure)
    return collected


# %% Fixed-Delta_EC, fixed-EJ_A2 sweep versus rail and CZ detuning
def make_detuning_sweep_colormaps(data: dict[str, np.ndarray], output: Path) -> None:
    x = data["delta_CZ_GHz"] * 1000.0
    y = data["delta_rail_GHz"] * 1000.0
    panels = (
        ("conditional_cz_fidelity_at_first_pi", r"Conditional fidelity at first odd-$\pi$", "viridis"),
        ("unconditional_cz_fidelity_at_first_pi", r"Unconditional fidelity at first odd-$\pi$", "viridis"),
        ("first_pi_conditional_time_ns", r"First odd-$\pi$ time (ns)", "viridis_r"),
        ("acceptance_at_first_pi", r"Acceptance at first odd-$\pi$", "viridis"),
        ("best_conditional_cz_fidelity", "Best conditional CZ fidelity", "viridis"),
        ("best_conditional_time_ns", "Time of best conditional fidelity (ns)", "viridis_r"),
        ("best_average_cz_fidelity", "Best unconditional logical CZ fidelity", "viridis"),
        ("conditional_phase_error_at_first_pi_rad", r"Phase error at first odd-$\pi$ (rad)", "magma"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for axis, (key, title, cmap_name) in zip(axes.flat, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")
        mesh = axis.pcolormesh(x, y, np.ma.masked_invalid(data[key]), shading="auto", cmap=cmap)
        axis.set_xlabel(r"$\Delta_{CZ}/h$ (MHz): $E_{0200}-E_{0110}$")
        axis.set_ylabel(r"$\delta_{rail}/h$ (MHz)")
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_detuning_sweep(config_path: str | Path = CONFIG_PATH) -> dict[str, np.ndarray]:
    """Sweep signed intra-rail magnitude and bare CZ collision detuning."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    if raw.get("noise") is not None:
        raise ValueError("The logical fidelity sweep currently requires noise=null")
    sweep = raw["detuning_sweep"]
    delta_ec = float(sweep["delta_EC_GHz"])
    ej_a2 = float(sweep["EJ_A2_GHz"])
    rail_values = _axis(sweep["delta_rail_GHz"])
    cz_values = _axis(sweep["delta_CZ_GHz"])
    channel = sweep.get("collision_channel", "0110_to_0200")
    bracket = tuple(float(v) for v in sweep["EJ_B1_bracket_GHz"])
    rail_bracket = tuple(float(v) for v in sweep.get("rail_EJ_bracket_GHz", bracket))
    ec_low = float(raw["sweep"]["EC_A1_B1_GHz"])
    ec_high = ec_low + delta_ec
    evolution = EvolutionSpec(**raw["evolution"])
    tolerance = float(sweep.get("conditional_phase_tolerance_rad", 0.05))
    shape = (rail_values.size, cz_values.size)
    names = (
        "solved_EJ_A1_GHz", "solved_EJ_B1_GHz", "solved_EJ_B2_GHz",
        "achieved_bare_detuning_GHz", "achieved_A1_minus_A2_detuning_GHz",
        "achieved_B1_minus_B2_detuning_GHz", "minimum_EJ_over_EC",
        "bare_collision_matrix_element_GHz", "best_average_cz_fidelity", "best_time_ns",
        "best_conditional_cz_fidelity", "best_conditional_time_ns",
        "first_pi_conditional_time_ns", "conditional_cz_fidelity_at_first_pi",
        "unconditional_cz_fidelity_at_first_pi",
        "acceptance_at_first_pi", "conditional_phase_error_at_first_pi_rad",
        "wrong_logical_population_at_first_pi",
        "erasure_detectable_leakage_at_first_pi",
        "erasure_undetectable_leakage_at_first_pi",
        "nonlogical_leakage_at_first_pi",
        "acceptance_at_best_unconditional", "acceptance_at_best_conditional", "leakage_at_best",
    )
    arrays = {name: np.full(shape, np.nan) for name in names}
    arrays.update(delta_rail_GHz=rail_values, delta_CZ_GHz=cz_values,
                  delta_EC_GHz=np.asarray(delta_ec), EJ_A2_GHz=np.asarray(ej_a2))
    trace_shape = shape + (evolution.samples,)
    trace_arrays = {name: np.full(trace_shape, np.nan) for name in (
        "average_cz_fidelity_vs_time", "conditional_cz_fidelity_vs_time", "conditional_phase_vs_time_rad",
        "acceptance_probability_vs_time", "logical_survival_vs_time",
        "wrong_logical_population_vs_time", "average_erasure_detectable_vs_time",
        "average_erasure_undetectable_vs_time")}

    for iy, delta_rail in enumerate(rail_values):
        target_a, target_b = -float(delta_rail), float(delta_rail)

        def residual_a1(ej: float) -> float:
            return _f01(raw, "A1", ec_low, ej) - _f01(raw, "A2", ec_high, ej_a2) - target_a

        try:
            ej_a1 = brentq(residual_a1, *rail_bracket, xtol=1e-10)
        except ValueError as error:
            print(f"Skipping delta_rail={delta_rail:g}: A1 solve failed: {error}")
            continue

        for ix, delta_cz in enumerate(cz_values):
            def solve_b2(ej_b1: float) -> float:
                def residual_b2(ej_b2: float) -> float:
                    return _f01(raw, "B1", ec_low, ej_b1) - _f01(raw, "B2", ec_high, ej_b2) - target_b
                return brentq(residual_b2, *rail_bracket, xtol=1e-10)

            def residual_b1(ej_b1: float) -> float:
                ej_b2 = solve_b2(ej_b1)
                system = _system(raw, delta_ec, ej_a1, ej_a2, ej_b1, ej_b2)
                return _collision_detuning(system, channel) - float(delta_cz)

            try:
                ej_b1 = brentq(residual_b1, *bracket, xtol=1e-10)
                ej_b2 = solve_b2(ej_b1)
            except ValueError as error:
                print(f"Skipping rail={delta_rail:g}, CZ={delta_cz:g}: {error}")
                continue
            system = _system(raw, delta_ec, ej_a1, ej_a2, ej_b1, ej_b2)
            minimum_ratio = min(spec.EJ_GHz / spec.EC_GHz for spec in system.specs.values())
            if minimum_ratio < float(sweep.get("minimum_EJ_over_EC", 0.0)):
                print(f"Skipping rail={delta_rail:g}, CZ={delta_cz:g}: EJ/EC min={minimum_ratio:.3f}")
                continue
            result = simulate_logical_basis(system, system.logical_basis(), evolution)
            index = (iy, ix)
            best = result["best_index"]
            arrays["solved_EJ_A1_GHz"][index] = ej_a1
            arrays["solved_EJ_B1_GHz"][index] = ej_b1
            arrays["solved_EJ_B2_GHz"][index] = ej_b2
            arrays["achieved_bare_detuning_GHz"][index] = _collision_detuning(system, channel)
            arrays["achieved_A1_minus_A2_detuning_GHz"][index] = system.local["A1"].levels_GHz[1] - system.local["A2"].levels_GHz[1]
            arrays["achieved_B1_minus_B2_detuning_GHz"][index] = system.local["B1"].levels_GHz[1] - system.local["B2"].levels_GHz[1]
            arrays["minimum_EJ_over_EC"][index] = minimum_ratio
            coupling = system.couplings_GHz["A2_B1"]
            n21 = system.local["A2"].level_kets[2].dag() * system.local["A2"].n_op * system.local["A2"].level_kets[1]
            n01 = system.local["B1"].level_kets[0].dag() * system.local["B1"].n_op * system.local["B1"].level_kets[1]
            arrays["bare_collision_matrix_element_GHz"][index] = abs(
                coupling * _qobj_scalar(n21) * _qobj_scalar(n01)
            )
            for key in ("best_average_cz_fidelity", "best_time_ns", "best_conditional_cz_fidelity", "best_conditional_time_ns", "acceptance_at_best_unconditional", "acceptance_at_best_conditional"):
                arrays[key][index] = result[key]
            arrays["leakage_at_best"][index] = 1.0 - result["logical_survival_vs_time"][best]
            _store_first_pi_metrics(arrays, index, result, tolerance)
            first_pi = _first_odd_pi_index(result["conditional_phase_vs_time_rad"], tolerance)
            if first_pi is not None:
                arrays["unconditional_cz_fidelity_at_first_pi"][index] = result[
                    "average_cz_fidelity_vs_time"
                ][first_pi]
                logical_populations = result["logical_populations_vs_time"]
                correct_population = np.trace(
                    logical_populations, axis1=1, axis2=2
                ) / 4.0
                wrong_population = (
                    np.sum(logical_populations, axis=(1, 2)) / 4.0
                    - correct_population
                )
                arrays["wrong_logical_population_at_first_pi"][index] = wrong_population[first_pi]
                arrays["erasure_detectable_leakage_at_first_pi"][index] = result[
                    "average_erasure_detectable_vs_time"
                ][first_pi]
                arrays["erasure_undetectable_leakage_at_first_pi"][index] = result[
                    "average_erasure_undetectable_vs_time"
                ][first_pi]
                arrays["nonlogical_leakage_at_first_pi"][index] = 1.0 - result[
                    "logical_survival_vs_time"
                ][first_pi]
            trace_arrays["wrong_logical_population_vs_time"][index] = (
                result["average_wrong_logical_population_vs_time"]
            )
            trace_arrays["average_erasure_detectable_vs_time"][index] = result[
                "average_erasure_detectable_vs_time"
            ]
            trace_arrays["average_erasure_undetectable_vs_time"][index] = result[
                "average_erasure_undetectable_vs_time"
            ]
            for key in trace_arrays:
                if key in result:
                    trace_arrays[key][index] = result[key]
            print(f"rail={delta_rail*1000:.1f} MHz, CZ={delta_cz*1000:.1f} MHz, F_cond={result['best_conditional_cz_fidelity']:.6f}, t_pi={arrays['first_pi_conditional_time_ns'][index]:.3f} ns")

    output = Path(sweep.get("output_directory", "../output/detuning_sweep_output"))
    if not output.is_absolute():
        output = config_path.parent / output
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "detuning_sweep_data.npz", **arrays)
    np.savez_compressed(output / "detuning_sweep_time_traces.npz",
                        delta_rail_GHz=rail_values, delta_CZ_GHz=cz_values,
                        times_ns=np.linspace(0.0, evolution.duration_ns, evolution.samples),
                        **trace_arrays)
    metadata = {
        "delta_EC_convention": "EC_A2 = EC_B2 = EC_A1 + delta_EC; EC_A1 = EC_B1",
        "rail_detuning_convention": "f01_A1 - f01_A2 = -delta_rail; f01_B1 - f01_B2 = +delta_rail",
        "CZ_detuning_convention": "delta_CZ = E_0200 - E_0110",
        "fixed_delta_EC_GHz": delta_ec, "fixed_EJ_A2_GHz": ej_a2,
        "evolution": raw["evolution"], "config": str(config_path),
    }
    (output / "detuning_sweep_report.json").write_text(json.dumps(metadata, indent=2))
    make_detuning_sweep_colormaps(arrays, output / "detuning_sweep_colormaps.png")
    print(f"Saved detuning sweep results to {output}")
    return arrays


# %% Fixed-Delta_EC, fixed-delta_rail sweep versus EJ_A2 and CZ detuning
def make_ej_a2_detuning_sweep_colormaps(
    data: dict[str, np.ndarray], output: Path
) -> None:
    """Plot the same metrics as the rail-detuning map with EJ_A2 on y."""
    x = data["delta_CZ_GHz"] * 1000.0
    y = data["EJ_A2_GHz"]
    panels = (
        ("conditional_cz_fidelity_at_first_pi", r"Conditional fidelity at first odd-$\pi$", "viridis", None, 0.999),
        ("unconditional_cz_fidelity_at_first_pi", r"Unconditional fidelity at first odd-$\pi$", "viridis", None, None),
        ("first_pi_conditional_time_ns", r"First odd-$\pi$ time (ns)", "viridis_r", None, None),
        ("acceptance_at_first_pi", r"Acceptance at first odd-$\pi$", "viridis", None, None),
        ("best_conditional_cz_fidelity", "Best conditional CZ fidelity", "viridis", None, 0.999),
        ("best_conditional_time_ns", "Time of best conditional fidelity (ns)", "viridis_r", None, None),
        ("best_average_cz_fidelity", "Best unconditional logical CZ fidelity", "viridis", None, None),
        ("conditional_phase_error_at_first_pi_rad", r"Phase error at first odd-$\pi$ (rad)", "magma", None, None),
    )
    figure, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for axis, (key, title, cmap_name, vmin, vmax) in zip(axes.flat, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")
        mesh = axis.pcolormesh(
            x, y, np.ma.masked_invalid(data[key]), shading="auto", cmap=cmap,
            vmin=vmin, vmax=vmax,
        )
        axis.set_xlabel(r"$\Delta_{CZ}/h$ (MHz): $E_{0200}-E_{0110}$")
        axis.set_ylabel(r"$E_{J,A2}/h$ (GHz)")
        axis.set_title(title)
        if vmin is not None:
            extend = "min"
        elif vmax is not None:
            extend = "max"
        else:
            extend = "neither"
        figure.colorbar(mesh, ax=axis, extend=extend)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_ej_a2_detuning_sweep(
    config_path: str | Path = CONFIG_PATH,
    config_key: str = "ej_a2_detuning_sweep",
) -> dict[str, np.ndarray]:
    """Sweep EJ_A2 and bare CZ detuning at a fixed signed rail detuning."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    if raw.get("noise") is not None:
        raise ValueError("The logical fidelity sweep currently requires noise=null")
    sweep = raw[config_key]
    delta_ec = float(sweep["delta_EC_GHz"])
    delta_rail = float(sweep.get("delta_rail_GHz", 0.0))
    ej_a2_values = _axis(sweep["EJ_A2_GHz"])
    cz_values = _axis(sweep["delta_CZ_GHz"])
    channel = sweep.get("collision_channel", "0110_to_0200")
    bracket = tuple(float(v) for v in sweep["EJ_B1_bracket_GHz"])
    rail_bracket = tuple(float(v) for v in sweep.get("rail_EJ_bracket_GHz", bracket))
    ec_low = float(raw["sweep"]["EC_A1_B1_GHz"])
    ec_high = ec_low + delta_ec
    evolution = EvolutionSpec(**raw["evolution"])
    tolerance = float(sweep.get("conditional_phase_tolerance_rad", 0.05))
    minimum_allowed_ratio = float(sweep.get("minimum_EJ_over_EC", 0.0))
    shape = (ej_a2_values.size, cz_values.size)
    names = (
        "solved_EJ_A1_GHz", "solved_EJ_B1_GHz", "solved_EJ_B2_GHz",
        "achieved_bare_detuning_GHz", "achieved_A1_minus_A2_detuning_GHz",
        "achieved_B1_minus_B2_detuning_GHz", "minimum_EJ_over_EC",
        "bare_collision_matrix_element_GHz", "best_average_cz_fidelity", "best_time_ns",
        "best_conditional_cz_fidelity", "best_conditional_time_ns",
        "first_pi_conditional_time_ns", "conditional_cz_fidelity_at_first_pi",
        "unconditional_cz_fidelity_at_first_pi", "acceptance_at_first_pi",
        "conditional_phase_error_at_first_pi_rad", "wrong_logical_population_at_first_pi",
        "erasure_detectable_leakage_at_first_pi", "erasure_undetectable_leakage_at_first_pi",
        "nonlogical_leakage_at_first_pi", "acceptance_at_best_unconditional",
        "acceptance_at_best_conditional", "leakage_at_best",
    )
    arrays = {name: np.full(shape, np.nan) for name in names}
    arrays.update(
        EJ_A2_GHz=ej_a2_values,
        delta_CZ_GHz=cz_values,
        delta_EC_GHz=np.asarray(delta_ec),
        delta_rail_GHz=np.asarray(delta_rail),
    )
    trace_shape = shape + (evolution.samples,)
    trace_arrays = {name: np.full(trace_shape, np.nan) for name in (
        "average_cz_fidelity_vs_time", "conditional_cz_fidelity_vs_time",
        "conditional_phase_vs_time_rad", "acceptance_probability_vs_time",
        "logical_survival_vs_time", "wrong_logical_population_vs_time",
        "average_erasure_detectable_vs_time", "average_erasure_undetectable_vs_time",
    )}
    target_a, target_b = -delta_rail, delta_rail

    for iy, ej_a2 in enumerate(ej_a2_values):
        def residual_a1(ej: float) -> float:
            return _f01(raw, "A1", ec_low, ej) - _f01(raw, "A2", ec_high, ej_a2) - target_a

        try:
            ej_a1 = brentq(residual_a1, *rail_bracket, xtol=1e-10)
        except ValueError as error:
            print(f"Skipping EJ_A2={ej_a2:g}: A1 solve failed: {error}")
            continue

        for ix, delta_cz in enumerate(cz_values):
            def solve_b2(ej_b1: float) -> float:
                def residual_b2(ej_b2: float) -> float:
                    return _f01(raw, "B1", ec_low, ej_b1) - _f01(raw, "B2", ec_high, ej_b2) - target_b
                return brentq(residual_b2, *rail_bracket, xtol=1e-10)

            def residual_b1(ej_b1: float) -> float:
                ej_b2 = solve_b2(ej_b1)
                system = _system(raw, delta_ec, ej_a1, ej_a2, ej_b1, ej_b2)
                return _collision_detuning(system, channel) - float(delta_cz)

            try:
                ej_b1 = brentq(residual_b1, *bracket, xtol=1e-10)
                ej_b2 = solve_b2(ej_b1)
            except ValueError as error:
                print(f"Skipping EJ_A2={ej_a2:g}, CZ={delta_cz:g}: {error}")
                continue

            system = _system(raw, delta_ec, ej_a1, ej_a2, ej_b1, ej_b2)
            minimum_ratio = min(
                spec.EJ_GHz / spec.EC_GHz for spec in system.specs.values()
            )
            if minimum_ratio < minimum_allowed_ratio:
                print(
                    f"Skipping EJ_A2={ej_a2:g}, CZ={delta_cz:g}: "
                    f"EJ/EC min={minimum_ratio:.3f}"
                )
                continue

            result = simulate_logical_basis(system, system.logical_basis(), evolution)
            index = (iy, ix)
            best = result["best_index"]
            arrays["solved_EJ_A1_GHz"][index] = ej_a1
            arrays["solved_EJ_B1_GHz"][index] = ej_b1
            arrays["solved_EJ_B2_GHz"][index] = ej_b2
            arrays["achieved_bare_detuning_GHz"][index] = _collision_detuning(system, channel)
            arrays["achieved_A1_minus_A2_detuning_GHz"][index] = (
                system.local["A1"].levels_GHz[1] - system.local["A2"].levels_GHz[1]
            )
            arrays["achieved_B1_minus_B2_detuning_GHz"][index] = (
                system.local["B1"].levels_GHz[1] - system.local["B2"].levels_GHz[1]
            )
            arrays["minimum_EJ_over_EC"][index] = minimum_ratio
            coupling = system.couplings_GHz["A2_B1"]
            n21 = system.local["A2"].level_kets[2].dag() * system.local["A2"].n_op * system.local["A2"].level_kets[1]
            n01 = system.local["B1"].level_kets[0].dag() * system.local["B1"].n_op * system.local["B1"].level_kets[1]
            arrays["bare_collision_matrix_element_GHz"][index] = abs(
                coupling * _qobj_scalar(n21) * _qobj_scalar(n01)
            )
            for key in (
                "best_average_cz_fidelity", "best_time_ns",
                "best_conditional_cz_fidelity", "best_conditional_time_ns",
                "acceptance_at_best_unconditional", "acceptance_at_best_conditional",
            ):
                arrays[key][index] = result[key]
            arrays["leakage_at_best"][index] = 1.0 - result["logical_survival_vs_time"][best]
            _store_first_pi_metrics(arrays, index, result, tolerance)
            first_pi = _first_odd_pi_index(result["conditional_phase_vs_time_rad"], tolerance)
            if first_pi is not None:
                arrays["unconditional_cz_fidelity_at_first_pi"][index] = result[
                    "average_cz_fidelity_vs_time"
                ][first_pi]
                logical_populations = result["logical_populations_vs_time"]
                correct_population = np.trace(logical_populations, axis1=1, axis2=2) / 4.0
                wrong_population = np.sum(logical_populations, axis=(1, 2)) / 4.0 - correct_population
                arrays["wrong_logical_population_at_first_pi"][index] = wrong_population[first_pi]
                arrays["erasure_detectable_leakage_at_first_pi"][index] = result[
                    "average_erasure_detectable_vs_time"
                ][first_pi]
                arrays["erasure_undetectable_leakage_at_first_pi"][index] = result[
                    "average_erasure_undetectable_vs_time"
                ][first_pi]
                arrays["nonlogical_leakage_at_first_pi"][index] = 1.0 - result[
                    "logical_survival_vs_time"
                ][first_pi]
            trace_arrays["wrong_logical_population_vs_time"][index] = result[
                "average_wrong_logical_population_vs_time"
            ]
            trace_arrays["average_erasure_detectable_vs_time"][index] = result[
                "average_erasure_detectable_vs_time"
            ]
            trace_arrays["average_erasure_undetectable_vs_time"][index] = result[
                "average_erasure_undetectable_vs_time"
            ]
            for key in trace_arrays:
                if key in result:
                    trace_arrays[key][index] = result[key]
            print(
                f"EJ_A2={ej_a2:g} GHz, CZ={delta_cz*1000:.1f} MHz, "
                f"F_cond={result['best_conditional_cz_fidelity']:.6f}, "
                f"t_pi={arrays['first_pi_conditional_time_ns'][index]:.3f} ns"
            )

    output = Path(sweep.get("output_directory", f"../output/{config_key}_output"))
    if not output.is_absolute():
        output = config_path.parent / output
    output.mkdir(parents=True, exist_ok=True)
    file_stem = str(sweep.get("file_stem", config_key))
    np.savez(output / f"{file_stem}_data.npz", **arrays)
    np.savez_compressed(
        output / f"{file_stem}_time_traces.npz",
        EJ_A2_GHz=ej_a2_values,
        delta_CZ_GHz=cz_values,
        times_ns=np.linspace(0.0, evolution.duration_ns, evolution.samples),
        **trace_arrays,
    )
    metadata = {
        "delta_EC_convention": "EC_A2 = EC_B2 = EC_A1 + delta_EC; EC_A1 = EC_B1",
        "rail_detuning_convention": "f01_A1 - f01_A2 = -delta_rail; f01_B1 - f01_B2 = +delta_rail",
        "CZ_detuning_convention": "delta_CZ = E_0200 - E_0110",
        "fixed_delta_EC_GHz": delta_ec,
        "fixed_delta_rail_GHz": delta_rail,
        "config_key": config_key,
        "evolution": raw["evolution"],
        "config": str(config_path),
    }
    (output / f"{file_stem}_report.json").write_text(
        json.dumps(metadata, indent=2)
    )
    make_ej_a2_detuning_sweep_colormaps(
        arrays, output / f"{file_stem}_colormaps.png"
    )
    print(f"Saved EJ_A2/CZ-detuning sweep results to {output}")
    return arrays


# %% Fixed-operating-point sweep versus intra-rail and inter-rail coupling
def make_coupling_sweep_colormaps(
    data: dict[str, np.ndarray], output: Path
) -> None:
    """Plot gate performance versus configured charge-coupling coefficients."""
    x = data["intra_rail_coupling_GHz"] * 1000.0
    y = data["inter_rail_coupling_GHz"] * 1000.0
    panels = (
        ("conditional_cz_fidelity_at_first_pi", r"Conditional fidelity at first odd-$\pi$", "viridis", None, 0.999),
        ("unconditional_cz_fidelity_at_first_pi", r"Unconditional fidelity at first odd-$\pi$", "viridis", None, None),
        ("first_pi_conditional_time_ns", r"First odd-$\pi$ time (ns)", "viridis_r", None, None),
        ("acceptance_at_first_pi", r"Acceptance at first odd-$\pi$", "viridis", None, None),
        ("wrong_logical_population_at_first_pi", r"Wrong-logical population at first odd-$\pi$", "magma", None, None),
        ("nonlogical_leakage_at_first_pi", r"Nonlogical leakage at first odd-$\pi$", "magma", None, None),
        ("inter_detuning_over_exchange", r"$|\omega_{A2}-\omega_{B1}|/g_{A2B1}$", "viridis", None, None),
        ("best_conditional_cz_fidelity", "Best conditional CZ fidelity", "viridis", None, 0.999),
    )
    figure, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for axis, (key, title, cmap_name, vmin, vmax) in zip(axes.flat, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")
        mesh = axis.pcolormesh(
            x, y, np.ma.masked_invalid(data[key]), shading="auto", cmap=cmap,
            vmin=vmin, vmax=vmax,
        )
        axis.set_xlabel(r"Common intra-rail charge coupling / $h$ (MHz)")
        axis.set_ylabel(r"A2--B1 charge coupling / $h$ (MHz)")
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis, extend="max" if vmax is not None else "neither")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_coupling_sweep_conditional_fidelity_comparison(
    data: dict[str, np.ndarray], output: Path
) -> None:
    """Compare first-odd-pi and globally best conditional fidelities."""
    x = data["intra_rail_coupling_GHz"] * 1000.0
    y = data["inter_rail_coupling_GHz"] * 1000.0
    first_pi = data["conditional_cz_fidelity_at_first_pi"]
    best = data["best_conditional_cz_fidelity"]
    difference = best - first_pi
    panels = (
        (first_pi, r"Conditional fidelity at first odd-$\pi$", "viridis", None, 0.999, "max"),
        (best, "Best conditional CZ fidelity", "viridis", None, 0.999, "max"),
        (difference, r"$F_{\mathrm{cond,best}}-F_{\mathrm{cond,first-}\pi}$", "magma", 0.0, None, "neither"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for axis, (values, title, cmap_name, vmin, vmax, extend) in zip(axes, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")
        mesh = axis.pcolormesh(
            x, y, np.ma.masked_invalid(values), shading="auto", cmap=cmap,
            vmin=vmin, vmax=vmax,
        )
        axis.set_xlabel(r"Common intra-rail charge coupling / $h$ (MHz)")
        axis.set_ylabel(r"A2--B1 charge coupling / $h$ (MHz)")
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis, extend=extend)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_coupling_sweep(
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, np.ndarray]:
    """Sweep common intra-rail and A2--B1 coupling at one fixed operating point."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    if raw.get("noise") is not None:
        raise ValueError("The logical fidelity sweep currently requires noise=null")
    sweep = raw["coupling_sweep"]
    delta_ec = float(sweep["delta_EC_GHz"])
    ej_a2 = float(sweep["EJ_A2_GHz"])
    delta_rail = float(sweep.get("delta_rail_GHz", 0.0))
    delta_cz = float(sweep["delta_CZ_GHz"])
    intra_values = _axis(sweep["intra_rail_coupling_GHz"])
    inter_values = _axis(sweep["inter_rail_coupling_GHz"])
    channel = sweep.get("collision_channel", "0110_to_0200")
    bracket = tuple(float(v) for v in sweep["EJ_B1_bracket_GHz"])
    rail_bracket = tuple(float(v) for v in sweep.get("rail_EJ_bracket_GHz", bracket))
    minimum_ratio = float(sweep.get("minimum_EJ_over_EC", 0.0))
    maximum_intra_ratio = float(
        sweep.get("maximum_intra_detuning_over_exchange", np.inf)
    )
    minimum_inter_ratio = float(
        sweep.get("minimum_inter_detuning_over_exchange", 0.0)
    )
    tolerance = float(sweep.get("conditional_phase_tolerance_rad", 0.05))
    evolution = EvolutionSpec(**raw["evolution"])
    ec_low = float(raw["sweep"]["EC_A1_B1_GHz"])
    ec_high = ec_low + delta_ec
    target_a, target_b = -delta_rail, delta_rail

    def residual_a1(ej: float) -> float:
        return _f01(raw, "A1", ec_low, ej) - _f01(raw, "A2", ec_high, ej_a2) - target_a

    ej_a1 = brentq(residual_a1, *rail_bracket, xtol=1e-10)

    def solve_b2(ej_b1: float) -> float:
        def residual_b2(ej_b2: float) -> float:
            return _f01(raw, "B1", ec_low, ej_b1) - _f01(raw, "B2", ec_high, ej_b2) - target_b
        return brentq(residual_b2, *rail_bracket, xtol=1e-10)

    def residual_b1(ej_b1: float) -> float:
        ej_b2 = solve_b2(ej_b1)
        system = _system(raw, delta_ec, ej_a1, ej_a2, ej_b1, ej_b2)
        return _collision_detuning(system, channel) - delta_cz

    ej_b1 = brentq(residual_b1, *bracket, xtol=1e-10)
    ej_b2 = solve_b2(ej_b1)

    shape = (inter_values.size, intra_values.size)
    names = (
        "best_average_cz_fidelity", "best_time_ns", "best_conditional_cz_fidelity",
        "best_conditional_time_ns", "first_pi_conditional_time_ns",
        "conditional_cz_fidelity_at_first_pi", "unconditional_cz_fidelity_at_first_pi",
        "acceptance_at_first_pi", "conditional_phase_error_at_first_pi_rad",
        "wrong_logical_population_at_first_pi", "erasure_detectable_leakage_at_first_pi",
        "erasure_undetectable_leakage_at_first_pi", "nonlogical_leakage_at_first_pi",
        "acceptance_at_best_unconditional", "acceptance_at_best_conditional", "leakage_at_best",
        "effective_exchange_A_GHz", "effective_exchange_B_GHz",
        "effective_exchange_A2_B1_GHz", "A_detuning_over_exchange",
        "B_detuning_over_exchange", "inter_detuning_over_exchange",
        "bare_collision_matrix_element_GHz", "collision_detuning_over_matrix_element",
        "achieved_bare_detuning_GHz", "achieved_A1_minus_A2_detuning_GHz",
        "achieved_B1_minus_B2_detuning_GHz",
    )
    arrays = {name: np.full(shape, np.nan) for name in names}
    arrays.update(
        intra_rail_coupling_GHz=intra_values,
        inter_rail_coupling_GHz=inter_values,
        delta_EC_GHz=np.asarray(delta_ec), EJ_A2_GHz=np.asarray(ej_a2),
        delta_rail_GHz=np.asarray(delta_rail), delta_CZ_GHz=np.asarray(delta_cz),
        solved_EJ_A1_GHz=np.asarray(ej_a1), solved_EJ_B1_GHz=np.asarray(ej_b1),
        solved_EJ_B2_GHz=np.asarray(ej_b2),
    )
    trace_shape = shape + (evolution.samples,)
    trace_arrays = {name: np.full(trace_shape, np.nan) for name in (
        "average_cz_fidelity_vs_time", "conditional_cz_fidelity_vs_time",
        "conditional_phase_vs_time_rad", "acceptance_probability_vs_time",
        "logical_survival_vs_time", "wrong_logical_population_vs_time",
        "average_erasure_detectable_vs_time", "average_erasure_undetectable_vs_time",
    )}

    for iy, inter_coupling in enumerate(inter_values):
        for ix, intra_coupling in enumerate(intra_values):
            point_raw = dict(raw)
            point_raw["couplings_GHz"] = {
                "A1_A2": float(intra_coupling),
                "B1_B2": float(intra_coupling),
                "A2_B1": float(inter_coupling),
            }
            system = _system(point_raw, delta_ec, ej_a1, ej_a2, ej_b1, ej_b2)
            transmon_ratio = min(
                spec.EJ_GHz / spec.EC_GHz for spec in system.specs.values()
            )
            if transmon_ratio < minimum_ratio:
                continue
            g_a = _effective_exchange(system, "A1", "A2")
            g_b = _effective_exchange(system, "B1", "B2")
            g_inter = _effective_exchange(system, "A2", "B1")
            detuning_a = abs(
                system.local["A1"].levels_GHz[1] - system.local["A2"].levels_GHz[1]
            )
            detuning_b = abs(
                system.local["B1"].levels_GHz[1] - system.local["B2"].levels_GHz[1]
            )
            detuning_inter = abs(
                system.local["A2"].levels_GHz[1] - system.local["B1"].levels_GHz[1]
            )
            ratio_a = detuning_a / g_a
            ratio_b = detuning_b / g_b
            ratio_inter = detuning_inter / g_inter
            index = (iy, ix)
            arrays["effective_exchange_A_GHz"][index] = g_a
            arrays["effective_exchange_B_GHz"][index] = g_b
            arrays["effective_exchange_A2_B1_GHz"][index] = g_inter
            arrays["A_detuning_over_exchange"][index] = ratio_a
            arrays["B_detuning_over_exchange"][index] = ratio_b
            arrays["inter_detuning_over_exchange"][index] = ratio_inter
            arrays["achieved_bare_detuning_GHz"][index] = _collision_detuning(
                system, channel
            )
            arrays["achieved_A1_minus_A2_detuning_GHz"][index] = (
                system.local["A1"].levels_GHz[1] - system.local["A2"].levels_GHz[1]
            )
            arrays["achieved_B1_minus_B2_detuning_GHz"][index] = (
                system.local["B1"].levels_GHz[1] - system.local["B2"].levels_GHz[1]
            )
            if max(ratio_a, ratio_b) > maximum_intra_ratio or ratio_inter < minimum_inter_ratio:
                print(
                    f"Skipping Jintra={intra_coupling*1000:.1f} MHz, "
                    f"Jinter={inter_coupling*1000:.1f} MHz: "
                    f"rail ratios=({ratio_a:.3g}, {ratio_b:.3g}), "
                    f"inter ratio={ratio_inter:.3g}"
                )
                continue

            n21 = (
                system.local["A2"].level_kets[2].dag()
                * system.local["A2"].n_op
                * system.local["A2"].level_kets[1]
            )
            n01 = (
                system.local["B1"].level_kets[0].dag()
                * system.local["B1"].n_op
                * system.local["B1"].level_kets[1]
            )
            collision_matrix_element = abs(
                system.couplings_GHz["A2_B1"]
                * _qobj_scalar(n21)
                * _qobj_scalar(n01)
            )
            arrays["bare_collision_matrix_element_GHz"][index] = collision_matrix_element
            arrays["collision_detuning_over_matrix_element"][index] = (
                abs(delta_cz) / collision_matrix_element
            )
            result = simulate_logical_basis(system, system.logical_basis(), evolution)
            best = result["best_index"]
            for key in (
                "best_average_cz_fidelity", "best_time_ns", "best_conditional_cz_fidelity",
                "best_conditional_time_ns", "acceptance_at_best_unconditional",
                "acceptance_at_best_conditional",
            ):
                arrays[key][index] = result[key]
            arrays["leakage_at_best"][index] = 1.0 - result["logical_survival_vs_time"][best]
            _store_first_pi_metrics(arrays, index, result, tolerance)
            first_pi = _first_odd_pi_index(result["conditional_phase_vs_time_rad"], tolerance)
            if first_pi is not None:
                arrays["unconditional_cz_fidelity_at_first_pi"][index] = result[
                    "average_cz_fidelity_vs_time"
                ][first_pi]
                arrays["wrong_logical_population_at_first_pi"][index] = result[
                    "average_wrong_logical_population_vs_time"
                ][first_pi]
                arrays["erasure_detectable_leakage_at_first_pi"][index] = result[
                    "average_erasure_detectable_vs_time"
                ][first_pi]
                arrays["erasure_undetectable_leakage_at_first_pi"][index] = result[
                    "average_erasure_undetectable_vs_time"
                ][first_pi]
                arrays["nonlogical_leakage_at_first_pi"][index] = 1.0 - result[
                    "logical_survival_vs_time"
                ][first_pi]
            for key in trace_arrays:
                if key in result:
                    trace_arrays[key][index] = result[key]
            print(
                f"Jintra={intra_coupling*1000:.1f} MHz, "
                f"Jinter={inter_coupling*1000:.1f} MHz, "
                f"Fcond={result['best_conditional_cz_fidelity']:.6f}, "
                f"tpi={arrays['first_pi_conditional_time_ns'][index]:.3f} ns"
            )

    output = Path(sweep.get("output_directory", "../output/coupling_sweep_output"))
    if not output.is_absolute():
        output = config_path.parent / output
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "coupling_sweep_data.npz", **arrays)
    np.savez_compressed(
        output / "coupling_sweep_time_traces.npz",
        intra_rail_coupling_GHz=intra_values,
        inter_rail_coupling_GHz=inter_values,
        times_ns=np.linspace(0.0, evolution.duration_ns, evolution.samples),
        **trace_arrays,
    )
    metadata = {
        "fixed_delta_EC_GHz": delta_ec, "fixed_EJ_A2_GHz": ej_a2,
        "fixed_delta_rail_GHz": delta_rail, "fixed_delta_CZ_GHz": delta_cz,
        "fixed_EJ_A1_GHz": ej_a1, "fixed_EJ_B1_GHz": ej_b1,
        "fixed_EJ_B2_GHz": ej_b2,
        "intra_rail_definition": "A1_A2 = B1_B2 = swept common charge-coupling coefficient",
        "inter_rail_definition": "A2_B1 swept charge-coupling coefficient",
        "intra_regime_guard": "max(|delta_A|/g_A, |delta_B|/g_B) <= configured maximum",
        "inter_regime_guard": "|f01_A2-f01_B1|/g_A2B1 >= configured minimum",
        "evolution": raw["evolution"], "config": str(config_path),
    }
    (output / "coupling_sweep_report.json").write_text(json.dumps(metadata, indent=2))
    make_coupling_sweep_colormaps(arrays, output / "coupling_sweep_colormaps.png")
    make_coupling_sweep_conditional_fidelity_comparison(
        arrays, output / "coupling_sweep_conditional_fidelity_comparison.png"
    )
    print(f"Saved coupling sweep results to {output}")
    return arrays


def plot_detuning_time_trace_linecut(
    output_directory: str | Path,
    delta_rail_MHz: float = 0.0,
) -> tuple[Path, Path]:
    """Plot saved raw traces and timing metrics at one delta_rail linecut."""
    output = Path(output_directory)
    summary_file = output / "detuning_sweep_data.npz"
    traces_file = output / "detuning_sweep_time_traces.npz"
    summary = np.load(summary_file)
    traces = np.load(traces_file)
    rail_MHz = summary["delta_rail_GHz"] * 1000.0
    cz_MHz = summary["delta_CZ_GHz"] * 1000.0
    iy = int(np.argmin(np.abs(rail_MHz - delta_rail_MHz)))
    times = traces["times_ns"]
    fidelity = traces["conditional_cz_fidelity_vs_time"][iy]
    phase = traces["conditional_phase_vs_time_rad"][iy]
    first_pi_time = summary["first_pi_conditional_time_ns"][iy]
    best_time = summary["best_conditional_time_ns"][iy]
    best_fidelity = summary["best_conditional_cz_fidelity"][iy]
    first_pi_fidelity = summary["conditional_cz_fidelity_at_first_pi"][iy]

    colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, cz_MHz.size))
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for ix, (delta_cz, color) in enumerate(zip(cz_MHz, colors)):
        axes[0].plot(times, fidelity[ix], color=color, lw=0.9, alpha=0.85)
        axes[1].plot(times, np.unwrap(phase[ix]) / np.pi, color=color, lw=0.9, alpha=0.85)
        for gate_time, marker in ((first_pi_time[ix], "o"), (best_time[ix], "x")):
            it = int(np.argmin(np.abs(times - gate_time)))
            axes[0].plot(times[it], fidelity[ix, it], marker=marker, color=color,
                         ms=4.5, mew=1.0, ls="none")
            axes[1].plot(times[it], np.unwrap(phase[ix])[it] / np.pi,
                         marker=marker, color=color, ms=4.5, mew=1.0, ls="none")
    axes[0].set_ylabel("Conditional CZ fidelity")
    axes[0].set_title(
        rf"Raw time traces at $\delta_{{rail}}/h={rail_MHz[iy]:g}$ MHz; "
        r"circles: first odd-$\pi$, crosses: best conditional fidelity"
    )
    axes[1].set_xlabel("Evolution time (ns)")
    axes[1].set_ylabel(r"Unwrapped conditional phase / $\pi$")
    for axis in axes:
        axis.grid(alpha=0.2)
    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(cz_MHz.min(), cz_MHz.max()), cmap="viridis"
    )
    figure.colorbar(scalar, ax=axes, label=r"$\Delta_{CZ}/h$ (MHz)", pad=0.02)
    traces_output = output / f"detuning_time_traces_delta_rail_{rail_MHz[iy]:g}MHz.png"
    figure.savefig(traces_output, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    axes[0].plot(cz_MHz, first_pi_time, "o-", label=r"First odd-$\pi$ time")
    axes[0].plot(cz_MHz, best_time, "x--", label="Best conditional-fidelity time")
    axes[0].set_ylabel("Gate time (ns)")
    axes[0].legend()
    axes[1].plot(cz_MHz, first_pi_fidelity, "o-", label=r"Fidelity at first odd-$\pi$")
    axes[1].plot(cz_MHz, best_fidelity, "x--", label="Best conditional fidelity")
    axes[1].set_xlabel(r"$\Delta_{CZ}/h$ (MHz)")
    axes[1].set_ylabel("Conditional CZ fidelity")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(rf"Timing comparison at $\delta_{{rail}}/h={rail_MHz[iy]:g}$ MHz")
    comparison_output = output / f"detuning_gate_time_comparison_delta_rail_{rail_MHz[iy]:g}MHz.png"
    figure.savefig(comparison_output, dpi=180)
    plt.close(figure)
    return traces_output, comparison_output


def refresh_first_pi_metrics_from_saved_traces(
    summary_file: str | Path,
    traces_file: str | Path,
    tolerance_rad: float = 0.05,
) -> dict[str, np.ndarray]:
    """Recompute first-odd-pi summary fields without rerunning evolution."""
    summary_file, traces_file = Path(summary_file), Path(traces_file)
    with np.load(summary_file) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    with np.load(traces_file) as saved:
        times = saved["times_ns"]
        phases = saved["conditional_phase_vs_time_rad"]
        fidelities = saved["conditional_cz_fidelity_vs_time"]
        acceptances = saved["acceptance_probability_vs_time"]
        wrong_population = saved.get("wrong_logical_population_vs_time")
        detectable = saved.get("average_erasure_detectable_vs_time")
        undetectable = saved.get("average_erasure_undetectable_vs_time")
        logical_survival = saved["logical_survival_vs_time"]
    shape = phases.shape[:-1]
    for key in (
        "first_pi_conditional_time_ns",
        "conditional_cz_fidelity_at_first_pi",
        "unconditional_cz_fidelity_at_first_pi",
        "acceptance_at_first_pi",
        "conditional_phase_error_at_first_pi_rad",
        "wrong_logical_population_at_first_pi",
        "erasure_detectable_leakage_at_first_pi",
        "erasure_undetectable_leakage_at_first_pi",
        "nonlogical_leakage_at_first_pi",
    ):
        arrays[key] = np.full(shape, np.nan)
    for index in np.ndindex(shape):
        first_pi = _first_odd_pi_index(phases[index], tolerance_rad)
        if first_pi is None:
            continue
        arrays["first_pi_conditional_time_ns"][index] = times[first_pi]
        arrays["conditional_cz_fidelity_at_first_pi"][index] = fidelities[index][first_pi]
        arrays["acceptance_at_first_pi"][index] = acceptances[index][first_pi]
        arrays["unconditional_cz_fidelity_at_first_pi"][index] = (
            fidelities[index][first_pi] * acceptances[index][first_pi]
        )
        arrays["conditional_phase_error_at_first_pi_rad"][index] = abs(
            np.angle(np.exp(1j * (phases[index][first_pi] - np.pi)))
        )
        arrays["nonlogical_leakage_at_first_pi"][index] = 1.0 - logical_survival[index][first_pi]
        if wrong_population is not None:
            arrays["wrong_logical_population_at_first_pi"][index] = wrong_population[index][first_pi]
        if detectable is not None:
            arrays["erasure_detectable_leakage_at_first_pi"][index] = detectable[index][first_pi]
        if undetectable is not None:
            arrays["erasure_undetectable_leakage_at_first_pi"][index] = undetectable[index][first_pi]
    np.savez(summary_file, **arrays)
    return arrays


def add_second_odd_pi_metrics_from_saved_traces(
    summary_file: str | Path,
    traces_file: str | Path,
) -> dict[str, np.ndarray]:
    """Add metrics at the first 3-pi crossing using existing saved traces."""
    summary_file, traces_file = Path(summary_file), Path(traces_file)
    with np.load(summary_file) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    with np.load(traces_file) as saved:
        times = saved["times_ns"]
        phases = saved["conditional_phase_vs_time_rad"]
        fidelities = saved["conditional_cz_fidelity_vs_time"]
        acceptances = saved["acceptance_probability_vs_time"]
        wrong_population = saved.get("wrong_logical_population_vs_time")
        detectable = saved.get("average_erasure_detectable_vs_time")
        undetectable = saved.get("average_erasure_undetectable_vs_time")
        logical_survival = saved["logical_survival_vs_time"]
    shape = phases.shape[:-1]
    fields = (
        "second_odd_pi_time_ns",
        "conditional_cz_fidelity_at_second_odd_pi",
        "unconditional_cz_fidelity_at_second_odd_pi",
        "acceptance_at_second_odd_pi",
        "conditional_phase_error_at_second_odd_pi_rad",
        "wrong_logical_population_at_second_odd_pi",
        "erasure_detectable_leakage_at_second_odd_pi",
        "erasure_undetectable_leakage_at_second_odd_pi",
        "nonlogical_leakage_at_second_odd_pi",
    )
    for key in fields:
        arrays[key] = np.full(shape, np.nan)
    for index in np.ndindex(shape):
        selected = _nth_odd_pi_index(phases[index], order=2)
        if selected is None:
            continue
        arrays["second_odd_pi_time_ns"][index] = times[selected]
        arrays["conditional_cz_fidelity_at_second_odd_pi"][index] = fidelities[index][selected]
        arrays["acceptance_at_second_odd_pi"][index] = acceptances[index][selected]
        arrays["unconditional_cz_fidelity_at_second_odd_pi"][index] = (
            fidelities[index][selected] * acceptances[index][selected]
        )
        phase = np.unwrap(phases[index])[selected]
        arrays["conditional_phase_error_at_second_odd_pi_rad"][index] = abs(
            phase - np.sign(phase) * 3.0 * np.pi
        )
        arrays["nonlogical_leakage_at_second_odd_pi"][index] = 1.0 - logical_survival[index][selected]
        if wrong_population is not None:
            arrays["wrong_logical_population_at_second_odd_pi"][index] = wrong_population[index][selected]
        if detectable is not None:
            arrays["erasure_detectable_leakage_at_second_odd_pi"][index] = detectable[index][selected]
        if undetectable is not None:
            arrays["erasure_undetectable_leakage_at_second_odd_pi"][index] = undetectable[index][selected]
    np.savez(summary_file, **arrays)
    return arrays


def make_second_odd_pi_colormaps(data: dict[str, np.ndarray], output: Path) -> None:
    """Plot gate metrics at the second odd-pi (3-pi) crossing."""
    x = data["delta_CZ_GHz"] * 1000.0
    y = data["delta_rail_GHz"] * 1000.0
    interval = data["second_odd_pi_time_ns"] - data["first_pi_conditional_time_ns"]
    panels = (
        (data["conditional_cz_fidelity_at_second_odd_pi"], r"Conditional fidelity at second odd-$\pi$", "viridis"),
        (data["unconditional_cz_fidelity_at_second_odd_pi"], r"Unconditional fidelity at second odd-$\pi$", "viridis"),
        (data["second_odd_pi_time_ns"], r"Second odd-$\pi$ time (ns)", "viridis_r"),
        (data["acceptance_at_second_odd_pi"], r"Acceptance at second odd-$\pi$", "viridis"),
        (data["conditional_phase_error_at_second_odd_pi_rad"], r"Phase error at second odd-$\pi$ (rad)", "magma"),
        (interval, r"Time from first to second odd-$\pi$ (ns)", "viridis_r"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (values, title, cmap_name) in zip(axes.flat, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")
        mesh = axis.pcolormesh(x, y, np.ma.masked_invalid(values), shading="auto", cmap=cmap)
        axis.set_xlabel(r"$\Delta_{CZ}/h$ (MHz): $E_{0200}-E_{0110}$")
        axis.set_ylabel(r"$\delta_{rail}/h$ (MHz)")
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_population_error_colormaps(
    data: dict[str, np.ndarray],
    output: Path,
    odd_pi_order: int,
) -> None:
    """Plot in-logical transfer and the two nonlogical leakage classes."""
    if odd_pi_order == 1:
        suffix, label = "first_pi", r"first odd-$\pi$"
    elif odd_pi_order == 2:
        suffix, label = "second_odd_pi", r"second odd-$\pi$"
    else:
        raise ValueError("Only first and second odd-pi summaries are available")
    x = data["delta_CZ_GHz"] * 1000.0
    y = data["delta_rail_GHz"] * 1000.0
    wrong = data[f"wrong_logical_population_at_{suffix}"]
    detectable = data[f"erasure_detectable_leakage_at_{suffix}"]
    undetectable = data[f"erasure_undetectable_leakage_at_{suffix}"]
    nonlogical = data[f"nonlogical_leakage_at_{suffix}"]
    closure_error = nonlogical - detectable - undetectable
    panels = (
        (wrong, f"Wrong-logical-state population at {label}", "magma"),
        (detectable, f"Erasure-detectable leakage at {label}", "magma"),
        (undetectable, f"Erasure-undetectable leakage at {label}", "magma"),
        (nonlogical, f"Total nonlogical leakage at {label}", "magma"),
        (wrong + nonlogical, f"Total population error at {label}", "magma"),
        (np.abs(closure_error), r"$|L_{nonlogical}-L_{detectable}-L_{undetectable}|$", "viridis"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for axis, (values, title, cmap_name) in zip(axes.flat, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")
        mesh = axis.pcolormesh(x, y, np.ma.masked_invalid(values), shading="auto", cmap=cmap)
        axis.set_xlabel(r"$\Delta_{CZ}/h$ (MHz): $E_{0200}-E_{0110}$")
        axis.set_ylabel(r"$\delta_{rail}/h$ (MHz)")
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_wrong_logical_distribution_linecut(
    config_path: str | Path = CONFIG_PATH,
    delta_rail_MHz: float = 0.0,
) -> dict[str, np.ndarray]:
    """Resolve all off-diagonal logical transitions on one saved sweep row."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    sweep = raw["detuning_sweep"]
    output = Path(sweep.get("output_directory", "../output/detuning_sweep_output"))
    if not output.is_absolute():
        output = config_path.parent / output
    with np.load(output / "detuning_sweep_data.npz") as saved:
        summary = {key: saved[key].copy() for key in saved.files}
    rail_MHz = summary["delta_rail_GHz"] * 1000.0
    cz_MHz = summary["delta_CZ_GHz"] * 1000.0
    iy = int(np.argmin(np.abs(rail_MHz - delta_rail_MHz)))
    delta_ec = float(summary["delta_EC_GHz"])
    ej_a2 = float(summary["EJ_A2_GHz"])
    evolution = EvolutionSpec(**raw["evolution"])
    matrices = {
        "first_odd_pi": np.full((cz_MHz.size, 4, 4), np.nan),
        "second_odd_pi": np.full((cz_MHz.size, 4, 4), np.nan),
    }
    selected_times = {
        "first_odd_pi": np.full(cz_MHz.size, np.nan),
        "second_odd_pi": np.full(cz_MHz.size, np.nan),
    }
    for ix, delta_cz in enumerate(cz_MHz):
        system = _system(
            raw,
            delta_ec,
            float(summary["solved_EJ_A1_GHz"][iy, ix]),
            ej_a2,
            float(summary["solved_EJ_B1_GHz"][iy, ix]),
            float(summary["solved_EJ_B2_GHz"][iy, ix]),
        )
        result = simulate_logical_basis(system, system.logical_basis(), evolution)
        for order, key in ((1, "first_odd_pi"), (2, "second_odd_pi")):
            selected = odd_pi_index(
                result["conditional_phase_vs_time_rad"],
                order=order,
                tolerance_rad=float(sweep.get("conditional_phase_tolerance_rad", 0.05)),
            )
            if selected is None:
                continue
            matrices[key][ix] = result["logical_populations_vs_time"][selected]
            selected_times[key][ix] = result["times_ns"][selected]
        print(f"Resolved logical transfer at rail={rail_MHz[iy]:g} MHz, CZ={delta_cz:g} MHz")

    labels = ("00", "01", "10", "11")
    channels = [(out, inp) for inp in range(4) for out in range(4) if out != inp]
    channel_labels = [rf"${labels[inp]}_L\to {labels[out]}_L$" for out, inp in channels]
    channel_values = {
        key: np.asarray([[matrix[out, inp] for out, inp in channels] for matrix in values]).T
        for key, values in matrices.items()
    }
    vmax = max(np.nanmax(values) for values in channel_values.values())
    figure, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    for axis, key, title in zip(
        axes,
        ("first_odd_pi", "second_odd_pi"),
        (r"First odd-$\pi$", r"Second odd-$\pi$ ($3\pi$)"),
    ):
        mesh = axis.pcolormesh(
            cz_MHz,
            np.arange(len(channels)),
            channel_values[key],
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        axis.set_yticks(np.arange(len(channels)), channel_labels)
        axis.set_ylabel("Wrong logical transition")
        axis.set_title(title + rf" at $\delta_{{rail}}/h={rail_MHz[iy]:g}$ MHz")
        figure.colorbar(mesh, ax=axis, label="Population")
    axes[-1].set_xlabel(r"$\Delta_{CZ}/h$ (MHz)")
    figure_output = output / f"wrong_logical_distribution_delta_rail_{rail_MHz[iy]:g}MHz.png"
    figure.savefig(figure_output, dpi=180)
    plt.close(figure)

    data_output = output / f"wrong_logical_distribution_delta_rail_{rail_MHz[iy]:g}MHz.npz"
    np.savez(
        data_output,
        delta_rail_MHz=rail_MHz[iy],
        delta_CZ_MHz=cz_MHz,
        channel_output_indices=np.asarray([out for out, _ in channels]),
        channel_input_indices=np.asarray([inp for _, inp in channels]),
        channel_labels=np.asarray(channel_labels),
        first_odd_pi_population_matrices=matrices["first_odd_pi"],
        second_odd_pi_population_matrices=matrices["second_odd_pi"],
        first_odd_pi_times_ns=selected_times["first_odd_pi"],
        second_odd_pi_times_ns=selected_times["second_odd_pi"],
    )
    return {
        "delta_CZ_MHz": cz_MHz,
        "channel_labels": np.asarray(channel_labels),
        "first_odd_pi_channel_populations": channel_values["first_odd_pi"],
        "second_odd_pi_channel_populations": channel_values["second_odd_pi"],
        "figure": np.asarray(str(figure_output)),
        "data": np.asarray(str(data_output)),
    }


def make_logical_intermediate_spectrum(
    config_path: str | Path = CONFIG_PATH,
    delta_rail_MHz: float = 0.0,
    delta_CZ_MHz: float = 5.0,
) -> Path:
    """Visualize pre-CZ logical levels and coupled (2,0)/(0,2) branches."""
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text())
    output = config_path.parent / raw["detuning_sweep"].get(
        "output_directory", "../output/detuning_sweep_output"
    )
    with np.load(output / "detuning_sweep_data.npz") as saved:
        summary = {key: saved[key].copy() for key in saved.files}
    rail_MHz = summary["delta_rail_GHz"] * 1000.0
    cz_MHz = summary["delta_CZ_GHz"] * 1000.0
    iy = int(np.argmin(np.abs(rail_MHz - delta_rail_MHz)))
    ix = int(np.argmin(np.abs(cz_MHz - delta_CZ_MHz)))
    system = _system(
        raw,
        float(summary["delta_EC_GHz"]),
        float(summary["solved_EJ_A1_GHz"][iy, ix]),
        float(summary["EJ_A2_GHz"]),
        float(summary["solved_EJ_B1_GHz"][iy, ix]),
        float(summary["solved_EJ_B2_GHz"][iy, ix]),
    )
    interaction = (
        system.couplings_GHz["A2_B1"]
        * system.n_full["A2"]
        * system.n_full["B1"]
    )
    h0 = system.hamiltonian() / TWOPI - interaction
    energies, eigenstates = h0.eigenstates()
    energies = np.asarray(energies, dtype=float)
    logical = system.logical_basis()
    logical_labels = np.asarray(logical.labels)
    logical_energies = np.asarray(
        [float(np.real(_qobj_scalar(state.dag() * h0 * state)))
         for state in logical.states]
    )
    sector_names = ("(1,1)", "(2,0)", "(0,2)", "(0,0)", "(2,2)")
    sector_pairs = ((1, 1), (2, 0), (0, 2), (0, 0), (2, 2))
    projectors = [system.sector_projector(*pair) for pair in sector_pairs]
    weights = np.asarray(
        [[float(qt.expect(projector, state).real) for projector in projectors]
         for state in eigenstates]
    )
    dominant = np.argmax(weights, axis=1)
    coupling = np.asarray(
        [[abs(_qobj_scalar(state.dag() * interaction * logical_state))
          for logical_state in logical.states] for state in eigenstates]
    )
    reference = float(np.mean(logical_energies))
    window = (
        energies >= np.min(logical_energies) - 0.8
    ) & (
        energies <= np.max(logical_energies) + 0.8
    )
    colors = {
        "(1,1)": "C0", "(2,0)": "C3", "(0,2)": "C2",
        "(0,0)": "0.65", "(2,2)": "0.45",
    }
    figure, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for sector_index, sector_name in enumerate(sector_names):
        selected = np.flatnonzero(window & (dominant == sector_index))
        axes[0].scatter(
            np.full(selected.size, sector_index),
            (energies[selected] - reference) * 1000.0,
            s=25 + 45 * weights[selected, sector_index],
            color=colors[sector_name],
            alpha=0.75,
            label=sector_name,
        )
    for label, energy in zip(logical_labels, logical_energies):
        axes[0].scatter(0, (energy - reference) * 1000.0, marker="*", s=180,
                        color="gold", edgecolor="black", zorder=5)
        axes[0].annotate(rf"$|{label[:2]}\rangle_L$", (0, (energy-reference)*1000.0),
                         xytext=(8, 0), textcoords="offset points", va="center")
    axes[0].set_xticks(np.arange(len(sector_names)), sector_names)
    axes[0].set_xlabel(r"Dominant pre-CZ excitation sector $(N_A,N_B)$")
    axes[0].set_ylabel("Energy relative to logical-manifold mean (MHz)")
    axes[0].set_title("Pre-CZ dressed spectrum (A2-B1 interaction removed)")
    axes[0].grid(axis="y", alpha=0.2)

    marker_scale = 350.0 / max(np.max(coupling[:, :]), 1e-15)
    for logical_index, (label, logical_energy) in enumerate(
        zip(logical_labels, logical_energies)
    ):
        for sector_index, offset in ((1, -0.12), (2, 0.12)):
            selected = np.flatnonzero(
                (dominant == sector_index) & (coupling[:, logical_index] > 1e-8)
                & (np.abs(energies - logical_energy) < 0.8)
            )
            axes[1].scatter(
                np.full(selected.size, logical_index + offset),
                (energies[selected] - logical_energy) * 1000.0,
                s=15 + marker_scale * coupling[selected, logical_index],
                color=colors[sector_names[sector_index]],
                alpha=0.7,
                label=sector_names[sector_index] if logical_index == 0 else None,
            )
    axes[1].axhline(0.0, color="black", lw=1.0, ls="--")
    axes[1].set_xticks(np.arange(4), [rf"$|{label[:2]}\rangle_L$" for label in logical_labels])
    axes[1].set_xlabel("Initial logical state")
    axes[1].set_ylabel(r"Intermediate-state denominator $E_m-E_i$ (MHz)")
    axes[1].set_title("Accessible intermediate branches; marker size = direct coupling")
    axes[1].legend(title=r"$(N_A,N_B)$")
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        rf"$\delta_{{rail}}/h={rail_MHz[iy]:g}$ MHz, "
        rf"$\Delta_{{CZ}}/h={cz_MHz[ix]:g}$ MHz"
    )
    figure_output = output / (
        f"logical_intermediate_spectrum_rail_{rail_MHz[iy]:g}MHz_"
        f"CZ_{cz_MHz[ix]:g}MHz.png"
    )
    figure.savefig(figure_output, dpi=180)
    plt.close(figure)
    np.savez(
        figure_output.with_suffix(".npz"),
        eigenenergies_GHz=energies,
        logical_energies_GHz=logical_energies,
        logical_labels=logical_labels,
        sector_names=np.asarray(sector_names),
        sector_weights=weights,
        dominant_sector_index=dominant,
        coupling_to_logical_GHz=coupling,
    )
    return figure_output


