"""Search effective-EJ gate operating points for two coupled dual rails.

The search reuses the four-transmon model and logical-gate evaluator from
``dualrail_sim``.  One Josephson energy is scanned; the other three are solved
from the two signed intra-rail detunings and one selected CZ-collision
detuning.  This reduces the nominal four-dimensional search to one dimension.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.optimize import brentq

from .simulation import (
    LABELS,
    TWOPI,
    BasisSpec,
    EvolutionSpec,
    FourTransmonSystem,
    TransmonSpec,
    build_local_model,
    cz_collision_splittings,
    simulate_logical_basis,
    logical_diagnostics,
)


GATE_OPERATING_POINT_SCHEMA = "independent-ec-cz-gate-points-v1"
_COLLISION_STATES = {
    "0110_to_0200": ((0, 1, 1, 0), (0, 2, 0, 0)),
    "0110_to_0020": ((0, 1, 1, 0), (0, 0, 2, 0)),
}


def _ej_axis(values: Sequence[float]) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("ej_list must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(axis)) or np.any(axis <= 0.0):
        raise ValueError("ej_list must contain finite positive values")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("ej_list must be strictly increasing")
    return axis


def _f01(raw: Mapping, label: str, ec: float, ej: float) -> float:
    spec = TransmonSpec(EC_GHz=ec, EJ_GHz=ej, ng=raw["transmons"][label].get("ng", 0.0))
    local = build_local_model(label, spec, BasisSpec(**raw["basis"]))
    return float(local.levels_GHz[1] - local.levels_GHz[0])


def _system(
    raw: Mapping,
    ej_a1: float,
    ej_a2: float,
    ej_b1: float,
    ej_b2: float,
) -> FourTransmonSystem:
    transmons = {label: dict(raw["transmons"][label]) for label in LABELS}
    # Four independent effective EC inputs come from raw["transmons"].
    for label, ej in zip(LABELS, (ej_a1, ej_a2, ej_b1, ej_b2), strict=True):
        transmons[label].update(EJ_GHz=ej)
    specs = {label: TransmonSpec(**transmons[label]) for label in LABELS}
    return FourTransmonSystem(specs, raw["couplings_GHz"], BasisSpec(**raw["basis"]))


def _collision_detuning(system: FourTransmonSystem, channel: str) -> float:
    try:
        first_levels, second_levels = _COLLISION_STATES[channel]
    except KeyError as error:
        raise ValueError(f"Unknown collision channel {channel!r}") from error
    hamiltonian_ghz = system.hamiltonian() / TWOPI

    def energy(levels) -> float:
        state = system.local_product_ket(levels)
        value = state.dag() * hamiltonian_ghz * state
        if hasattr(value, "full"):
            value = value.full()[0, 0]
        return float(np.real(value))

    return energy(second_levels) - energy(first_levels)


def _finite_sign_change_brackets(function, grid: np.ndarray) -> list[tuple[float, float]]:
    """Return grid intervals that bracket roots, skipping invalid samples."""
    values = np.full(grid.size, np.nan)
    for index, value in enumerate(grid):
        try:
            values[index] = function(float(value))
        except ValueError:
            pass
    brackets = []
    for index in range(grid.size - 1):
        left_value, right_value = values[index : index + 2]
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0:
            brackets.append((float(grid[index]), float(grid[index])))
        elif left_value * right_value < 0.0:
            brackets.append((float(grid[index]), float(grid[index + 1])))
    if np.isfinite(values[-1]) and values[-1] == 0.0:
        brackets.append((float(grid[-1]), float(grid[-1])))
    return brackets


def _solve_brackets(function, brackets, xtol: float) -> list[float]:
    roots = []
    for left, right in brackets:
        root = left if left == right else float(brentq(function, left, right, xtol=xtol))
        if not roots or abs(root - roots[-1]) > 10.0 * xtol:
            roots.append(root)
    return roots


def find_gate_operating_points(
    raw: Mapping,
    ej_list: Sequence[float],
    *,
    ec_a1_b1_GHz: float | None = None,
    delta_ec_GHz: float = 0.150,
    target_a_detuning_GHz: float = 0.0,
    target_b_detuning_GHz: float = 0.0,
    target_cz_detuning_GHz: float = 0.0,
    collision_channel: str = "0110_to_0200",
    minimum_EJ_over_EC: float = 25.0,
    minimum_other_collision_detuning_GHz: float = 0.0,
    evolution: EvolutionSpec | Mapping | None = None,
    minimum_first_odd_pi_conditional_fidelity: float | None = None,
    minimum_first_odd_pi_unconditional_fidelity: float | None = None,
    minimum_first_odd_pi_acceptance: float | None = None,
    maximum_first_odd_pi_time_ns: float | None = None,
    conditional_phase_tolerance_rad: float = 0.05,
    root_xtol: float = 1e-10,
) -> xr.Dataset:
    """Find accepted gate points for four independent effective EC inputs.

    ECs come from raw['transmons']; only an explicit ec_a1_b1_GHz keyword
    requests the original symmetric shorthand. Scan EJ_A1 over ej_list.

    ``EJ_A2`` is solved to hold the A-rail detuning, ``EJ_B2`` is solved from
    ``EJ_B1`` to hold the B-rail detuning, and ``EJ_B1`` is solved to hit the
    requested bare CZ detuning.  All roots are restricted to the common range
    ``[min(ej_list), max(ej_list)]``.  Logical dynamics are evaluated only when
    at least one first-odd-pi acceptance threshold is supplied. Global-best
    fidelity metrics are retained as diagnostics but are not used to accept a
    gate point.
    """
    ej_axis = _ej_axis(ej_list)
    if collision_channel not in _COLLISION_STATES:
        raise ValueError(f"Unknown collision channel {collision_channel!r}")
    for name, value in (
        (
            "minimum_first_odd_pi_conditional_fidelity",
            minimum_first_odd_pi_conditional_fidelity,
        ),
        (
            "minimum_first_odd_pi_unconditional_fidelity",
            minimum_first_odd_pi_unconditional_fidelity,
        ),
        ("minimum_first_odd_pi_acceptance", minimum_first_odd_pi_acceptance),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if maximum_first_odd_pi_time_ns is not None and maximum_first_odd_pi_time_ns <= 0:
        raise ValueError("maximum_first_odd_pi_time_ns must be positive")
    run_dynamics = any(
        value is not None
        for value in (
            minimum_first_odd_pi_conditional_fidelity,
            minimum_first_odd_pi_unconditional_fidelity,
            minimum_first_odd_pi_acceptance,
            maximum_first_odd_pi_time_ns,
        )
    )
    if run_dynamics and evolution is None:
        raise ValueError("evolution is required when applying a fidelity filter")
    if isinstance(evolution, Mapping):
        evolution = EvolutionSpec(**evolution)

    import copy
    raw = copy.deepcopy(raw)
    if ec_a1_b1_GHz is not None:  # Explicit legacy symmetric shorthand.
        for label in ("A1", "B1"):
            raw["transmons"][label]["EC_GHz"] = float(ec_a1_b1_GHz)
        for label in ("A2", "B2"):
            raw["transmons"][label]["EC_GHz"] = float(ec_a1_b1_GHz) + delta_ec_GHz
    ecs = {label: float(raw["transmons"][label]["EC_GHz"]) for label in LABELS}
    if not all(np.isfinite(v) and v > 0 for v in ecs.values()):
        raise ValueError("Four finite positive effective EC inputs are required")
    bracket = (float(ej_axis[0]), float(ej_axis[-1]))
    other_channel = next(name for name in _COLLISION_STATES if name != collision_channel)
    accepted: list[dict[str, float]] = []

    for ej_a1 in ej_axis:
        def residual_a2(ej_a2: float) -> float:
            return (
                _f01(raw, "A1", ecs["A1"], ej_a1)
                - _f01(raw, "A2", ecs["A2"], ej_a2)
                - target_a_detuning_GHz
            )

        def solve_b2(ej_b1: float) -> float:
            def residual_b2(ej_b2: float) -> float:
                return (
                    _f01(raw, "B1", ecs["B1"], ej_b1)
                    - _f01(raw, "B2", ecs["B2"], ej_b2)
                    - target_b_detuning_GHz
                )

            return float(brentq(residual_b2, *bracket, xtol=root_xtol))

        a2_roots = _solve_brackets(
            residual_a2,
            _finite_sign_change_brackets(residual_a2, ej_axis),
            root_xtol,
        )
        if not a2_roots:
            continue
        for ej_a2 in a2_roots:
            def residual_b1(ej_b1: float) -> float:
                ej_b2 = solve_b2(ej_b1)
                system = _system(raw, ej_a1, ej_a2, ej_b1, ej_b2)
                return _collision_detuning(system, collision_channel) - target_cz_detuning_GHz

            b1_roots = _solve_brackets(
                residual_b1,
                _finite_sign_change_brackets(residual_b1, ej_axis),
                root_xtol,
            )
            for ej_b1 in b1_roots:
                ej_b2 = solve_b2(ej_b1)
                system = _system(raw, ej_a1, ej_a2, ej_b1, ej_b2)
                minimum_ratio = min(
                    spec.EJ_GHz / spec.EC_GHz for spec in system.specs.values()
                )
                if minimum_ratio < minimum_EJ_over_EC:
                    continue
                collisions = cz_collision_splittings(system)
                other_detuning = float(collisions[other_channel]["bare_detuning_GHz"])
                if abs(other_detuning) < minimum_other_collision_detuning_GHz:
                    continue

                row = {
                    **logical_diagnostics(system),
                    "EJ_A1_GHz": float(ej_a1),
                    "EJ_A2_GHz": ej_a2,
                    "EJ_B1_GHz": ej_b1,
                    "EJ_B2_GHz": ej_b2,
                    "achieved_A_detuning_GHz": (
                        system.local["A1"].levels_GHz[1]
                        - system.local["A2"].levels_GHz[1]
                    ),
                    "achieved_B_detuning_GHz": (
                        system.local["B1"].levels_GHz[1]
                        - system.local["B2"].levels_GHz[1]
                    ),
                    "achieved_CZ_detuning_GHz": _collision_detuning(
                        system, collision_channel
                    ),
                    "other_collision_detuning_GHz": other_detuning,
                    "minimum_EJ_over_EC": minimum_ratio,
                    "best_conditional_cz_fidelity": np.nan,
                    "best_unconditional_cz_fidelity": np.nan,
                    "best_conditional_time_ns": np.nan,
                    "best_unconditional_time_ns": np.nan,
                    "first_odd_pi_conditional_cz_fidelity": np.nan,
                    "first_odd_pi_unconditional_cz_fidelity": np.nan,
                    "first_odd_pi_time_ns": np.nan,
                    "first_odd_pi_acceptance": np.nan,
                    "first_odd_pi_phase_error_rad": np.nan,
                    "conditional_fidelity_end_slope_per_ns": np.nan,
                }
                if run_dynamics:
                    result = simulate_logical_basis(
                        system,
                        system.logical_basis(),
                        evolution,
                        odd_pi_tolerance_rad=conditional_phase_tolerance_rad,
                    )
                    row.update(
                        best_conditional_cz_fidelity=float(
                            result["best_conditional_cz_fidelity"]
                        ),
                        best_unconditional_cz_fidelity=float(
                            result["best_average_cz_fidelity"]
                        ),
                        best_conditional_time_ns=float(
                            result["best_conditional_time_ns"]
                        ),
                        best_unconditional_time_ns=float(result["best_time_ns"]),
                        first_odd_pi_conditional_cz_fidelity=float(
                            result["conditional_cz_fidelity_at_first_odd_pi"]
                        ),
                        first_odd_pi_unconditional_cz_fidelity=float(
                            result["unconditional_cz_fidelity_at_first_odd_pi"]
                        ),
                        first_odd_pi_time_ns=float(result["first_odd_pi_time_ns"]),
                        first_odd_pi_acceptance=float(
                            result["acceptance_at_first_odd_pi"]
                        ),
                        first_odd_pi_phase_error_rad=float(
                            result["conditional_phase_error_at_first_odd_pi_rad"]
                        ),
                        conditional_fidelity_end_slope_per_ns=float(
                            result["conditional_fidelity_end_slope_per_ns"]
                        ),
                    )
                    first_pi_checks = (
                        (
                            minimum_first_odd_pi_conditional_fidelity,
                            row["first_odd_pi_conditional_cz_fidelity"],
                            "minimum",
                        ),
                        (
                            minimum_first_odd_pi_unconditional_fidelity,
                            row["first_odd_pi_unconditional_cz_fidelity"],
                            "minimum",
                        ),
                        (
                            minimum_first_odd_pi_acceptance,
                            row["first_odd_pi_acceptance"],
                            "minimum",
                        ),
                        (
                            maximum_first_odd_pi_time_ns,
                            row["first_odd_pi_time_ns"],
                            "maximum",
                        ),
                    )
                    if any(
                        threshold is not None
                        and (
                            not np.isfinite(value)
                            or (mode == "minimum" and value < threshold)
                            or (mode == "maximum" and value > threshold)
                        )
                        for threshold, value, mode in first_pi_checks
                    ):
                        continue
                accepted.append(row)

    names = (
        "H_A", "H_B", "logical_frequency_A_GHz", "logical_frequency_B_GHz", "logical_detuning_GHz",
        "EJ_A1_GHz", "EJ_A2_GHz", "EJ_B1_GHz", "EJ_B2_GHz",
        "achieved_A_detuning_GHz", "achieved_B_detuning_GHz",
        "achieved_CZ_detuning_GHz", "other_collision_detuning_GHz",
        "minimum_EJ_over_EC", "best_conditional_cz_fidelity",
        "best_unconditional_cz_fidelity", "best_conditional_time_ns",
        "best_unconditional_time_ns", "first_odd_pi_conditional_cz_fidelity",
        "first_odd_pi_unconditional_cz_fidelity", "first_odd_pi_time_ns",
        "first_odd_pi_acceptance", "first_odd_pi_phase_error_rad",
        "conditional_fidelity_end_slope_per_ns",
    )
    dataset = xr.Dataset(
        {name: (("gate_operating_point",), [row[name] for row in accepted]) for name in names},
        coords={"gate_operating_point": np.arange(len(accepted))},
        attrs={
            "gate_operating_point_schema": GATE_OPERATING_POINT_SCHEMA,
            "description": "Accepted effective-EJ four-transmon CZ gate points.",
            **{f"EC_{label}_GHz": value for label, value in ecs.items()},
            "delta_EC_A_GHz": ecs["A2"]-ecs["A1"],
            "delta_EC_B_GHz": ecs["B2"]-ecs["B1"],
            "collision_channel": collision_channel,
            "target_A_detuning_GHz": float(target_a_detuning_GHz),
            "target_B_detuning_GHz": float(target_b_detuning_GHz),
            "target_CZ_detuning_GHz": float(target_cz_detuning_GHz),
            "minimum_EJ_over_EC": float(minimum_EJ_over_EC),
            "minimum_other_collision_detuning_GHz": float(minimum_other_collision_detuning_GHz),
            "conditional_phase_tolerance_rad": float(conditional_phase_tolerance_rad),
            "EJ_range_GHz": bracket,
            "EJ_A1_scan_count": int(ej_axis.size),
        },
    )
    for name in names:
        if name.endswith("_GHz"):
            dataset[name].attrs["units"] = "GHz"
        elif name.endswith("_ns"):
            dataset[name].attrs["units"] = "ns"
    return dataset


def save_gate_operating_points(dataset: xr.Dataset, path: str | Path) -> Path:
    """Save a gate-point dataset to NetCDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    return path


