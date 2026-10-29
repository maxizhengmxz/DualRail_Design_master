"""One EC point: minimum-S1 selection followed by exact-pi calibration.

Pair occupations are explicit: 0020 selects B bare20; 0200 selects A bare02.
Numerical calibration re-solves dependent couplings; it is not a hardware pulse.
"""

import hashlib
import json
import time
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .channel_policy import (
    LOGICAL_ORDER,
    S1Candidate,
    off_target_definition,
    select_channel,
)
from .config import validate_payload
from .model import (
    collision_residual,
    frequency_limits,
    layout,
    pair_model,
    target_overlap,
)


def evaluate_ec(payload: dict) -> dict:
    """Select one feasible S1 minimum and calibrate its total CZ phase.

    All frequencies are cyclic GHz. The output is JSON-compatible; normal
    infeasibility is returned as a status rather than an execution exception.
    """
    validate_payload(payload)
    started = time.perf_counter()
    budget = payload.get("point_budget_seconds", 180)

    def guard():
        if time.perf_counter() - started > budget:
            raise TimeoutError("EC point exceeded its finite calculation budget")

    channel = payload.get("resonance_channel", "0020")
    if channel not in ("0020", "0200"):
        raise ValueError("resonance_channel must be 0020 or 0200")
    bounds = payload.get(
        "bare_f01_bounds_GHz", {name: [4.0, 5.0] for name in ("A1", "A2", "B1", "B2")}
    )
    a_low, a_high = (
        max(bounds[k][0] for k in ("A1", "A2")),
        min(bounds[k][1] for k in ("A1", "A2")),
    )
    b_bound_low, b_bound_high = (
        max(bounds[k][0] for k in ("B1", "B2")),
        min(bounds[k][1] for k in ("B1", "B2")),
    )
    if a_low > a_high or b_bound_low > b_bound_high:
        raise ValueError("Empty within-rail bare frequency intersection")
    target_index_a = 8 if channel == "0020" else 2

    def a_pair(frequency):
        return pair_model(ec[0], ec[1], frequency, split_a, target_index_a)

    def b_pair(frequency):
        if not b_bound_low - 1e-10 <= frequency <= b_bound_high + 1e-10:
            raise ValueError("B frequency outside configured transmon bounds")
        return pair_model(ec[2], ec[3], frequency, split_b)

    ec = tuple(payload["ec_GHz"])
    if len(ec) != 4 or not all(0 < value <= 0.300 for value in ec):
        raise ValueError("Four EC values in (0,0.300] GHz required")
    lower, upper, step = payload.get("frequency_GHz", [a_low, a_high, 0.05])
    if lower < a_low or upper > a_high or step <= 0 or lower > upper:
        raise ValueError("A frequency scan must lie within both transmon bounds")
    g = payload.get("g_GHz", 0.005)
    if not np.isfinite(g) or g <= 0:
        raise ValueError("g_GHz must be positive and finite")
    time_low, time_high = payload.get("duration_ns", [0.7 / (2 * g), 1.3 / (2 * g)])
    if not 0 < time_low < time_high:
        raise ValueError("Invalid gate duration window")
    split_a, split_b = payload.get("logical_splittings_GHz", [0.15, 0.09])
    logical_sources = payload.get("logical_sources", list(LOGICAL_ORDER))
    comparison_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    records, valid = [], []
    failures = {}
    b_limits = [frequency_limits(value) for value in ec[2:]]
    b_low, b_high = (
        max(b_bound_low, *(v[0] for v in b_limits)),
        min(b_bound_high, *(v[1] for v in b_limits)),
    )
    if b_low >= b_high:
        return {"status": "no_feasible_B_frequency_range", "resonance_channel": channel}

    def rank(candidates):
        if not candidates:
            return None
        selected = select_channel(
            [
                S1Candidate(
                    row["logical_source"],
                    "B20" if channel == "0020" else "A02",
                    row["fB_GHz"],
                    row["S1"],
                    g * 1000,
                    row["maximum_logical_ratio"],
                    "xm-full-cosine-4^4",
                    comparison_id,
                    stationary_frequency_ghz=row["fA_GHz"],
                    tracking_verified=True,
                )
                for row in candidates
            ],
            moved_pair="B",
            auxiliary="B20" if channel == "0020" else "A02",
            target_g_mhz=g * 1000,
        )
        best = next(
            row
            for row in candidates
            if row["logical_source"] == selected.source_logical
            and row["fB_GHz"] == selected.gate_frequency_ghz
            and row["fA_GHz"] == selected.stationary_frequency_ghz
        )
        return dict(best, channel_selection=selected.to_dict())

    def screening(f_a):
        return rank(
            [
                row
                for source in logical_sources
                if (row := screening_source(f_a, source)) is not None
            ]
        )

    def screening_source(f_a, logical_source):
        guard()
        a = a_pair(float(f_a))
        candidates = []
        previous = None
        for f_b in np.linspace(b_low + 1e-7, b_high - 1e-7, 14):
            guard()
            try:
                b = b_pair(float(f_b))
                residual = collision_residual(
                    a, b, channel, logical_source=logical_source
                )
                if (
                    previous
                    and previous[1] * residual <= 0
                    and (channel == "0200" or previous[2] == b["target"])
                ):
                    branch = b["target"]

                    def equation(frequency, branch=branch):
                        model = b_pair(float(frequency))
                        if channel == "0020" and model["target"] != branch:
                            raise ValueError("collision_branch_change")
                        return collision_residual(
                            a, model, channel, logical_source=logical_source
                        )

                    root = brentq(equation, previous[0], float(f_b), xtol=1e-9)
                    at_root = b_pair(root)
                    *_, score = layout(
                        a, at_root, g, channel, logical_source=logical_source
                    )
                    if score["maximum_logical_ratio"] < 0.2:
                        candidates.append(
                            {
                                "fA_GHz": float(f_a),
                                "fB_GHz": root,
                                "logical_source": logical_source,
                                **score,
                            }
                        )
                previous = float(f_b), residual, b["target"]
            except ValueError:
                previous = None
        return rank(candidates)

    frequencies = np.linspace(lower, upper, round((upper - lower) / step) + 1)
    for frequency in frequencies:
        try:
            result = screening(float(frequency))
            records.append(
                result or {"fA_GHz": float(frequency), "status": "no_feasible_root"}
            )
            if result:
                valid.append(result)
        except ValueError as error:
            failures[str(error)] = failures.get(str(error), 0) + 1
    if not valid:
        return {
            "status": "no_feasible_S1_position",
            "screening": records,
            "failures": failures,
        }
    best = rank(valid)
    boundary = abs(best["fA_GHz"] - lower) < 1e-7 or abs(best["fA_GHz"] - upper) < 1e-7
    if not boundary:
        center = best["fA_GHz"]
        refinement = []

        def objective(frequency):
            try:
                result = screening(float(frequency))
                if result:
                    refinement.append(result)
                    return result["S1"]
            except ValueError:
                pass
            return 1e6

        minimize_scalar(
            objective,
            bounds=(max(lower, center - step), min(upper, center + step)),
            method="bounded",
            options={"xatol": 0.0005, "maxiter": 16},
        )
        if refinement:
            best = rank([best] + refinement)
    logical_source = best["logical_source"]
    selected_channel = best["channel_selection"]
    f_a, base_b = best["fA_GHz"], best["fB_GHz"]
    a = a_pair(f_a)
    reference_b = b_pair(base_b)

    @lru_cache(maxsize=64)
    def build(offset):
        guard()
        b = b_pair(base_b + offset * 0.001)
        if channel == "0020" and (
            b["target"] != reference_b["target"] or target_overlap(reference_b, b) < 0.9
        ):
            raise ValueError("auxiliary_branch_change")
        energies, unit, indices, target, coupling, scores = layout(
            a, b, g, channel, logical_source=logical_source
        )
        eigenvalues, vectors = np.linalg.eigh(np.diag(energies) + coupling * unit)
        coefficients = vectors[indices].T
        return eigenvalues, coefficients, b, coupling, scores

    def at(offset, duration):
        es, coefficients, b, coupling, scores = build(float(offset))
        matrix = coefficients.T @ (
            np.exp(-2j * np.pi * es * duration)[:, None] * coefficients
        )
        if min(abs(np.diag(matrix))) < 0.1:
            raise ValueError("small_return_amplitude")
        q = matrix[3, 3] * matrix[0, 0] * np.conj(matrix[2, 2] * matrix[1, 1])
        return matrix, float(np.angle(-q)), b, coupling, scores

    def calibrate_time(duration):
        guard()
        offsets = [-12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 12.0]
        values = []
        for offset in offsets:
            try:
                values.append(at(offset, duration)[1])
            except ValueError:
                values.append(float("nan"))
        options = []
        for j in range(len(offsets) - 1):
            if (
                not np.isfinite(values[j] + values[j + 1])
                or values[j] * values[j + 1] > 0
                or abs(values[j] - values[j + 1]) > np.pi
            ):
                continue
            try:
                root = brentq(
                    lambda offset: at(offset, duration)[1],
                    offsets[j],
                    offsets[j + 1],
                    xtol=2e-5,
                )
                matrix, error, b, coupling, scores = at(root, duration)
                if abs(error) > 0.01 or scores["maximum_logical_ratio"] >= 0.2:
                    continue
                acceptance = float(np.sum(abs(matrix) ** 2) / 4)
                # With total phase pi, local Z corrections align all diagonal terms.
                fidelity = float(
                    (4 * acceptance + np.sum(abs(np.diag(matrix))) ** 2) / 20
                )
                options.append(
                    {
                        "time_ns": float(duration),
                        "channel_selection": selected_channel,
                        "B_frequency_offset_MHz": root,
                        "fB_GHz": base_b + 0.001 * root,
                        "raw_fidelity": fidelity,
                        "postselected_fidelity": fidelity / acceptance,
                        "acceptance": acceptance,
                        "leakage_by_input": list(
                            map(float, 1 - np.sum(abs(matrix) ** 2, axis=0))
                        ),
                        "return_by_input": list(map(float, abs(np.diag(matrix)) ** 2)),
                        "phase_error_rad": error,
                        "EJ_GHz": a["EJ"] + b["EJ"],
                        "couplings_GHz": {
                            "A1_A2": a["J"],
                            "B1_B2": b["J"],
                            "A2_B1": coupling,
                        },
                        "auxiliary_bare_weight": b["target_weight"]
                        if channel == "0020"
                        else a["target_weight"],
                        **scores,
                    }
                )
            except ValueError:
                pass
        return max(options, key=lambda v: v["raw_fidelity"]) if options else None

    times = np.linspace(time_low, time_high, 21)
    time_step = (time_high - time_low) / 20
    calibrated = []
    for duration in times:
        result = calibrate_time(float(duration))
        if result:
            calibrated.append(result)
    if not calibrated:
        return {
            "status": "no_exact_pi_calibration",
            "S1_position": best,
            "frequency_boundary": boundary,
            "screening": records,
        }
    # Refine several TIME maxima at the same single S1 operating position.
    starts = sorted(calibrated, key=lambda v: v["raw_fidelity"], reverse=True)[:3]
    for candidate in starts:
        center = candidate["time_ns"]

        def objective(duration):
            value = calibrate_time(float(duration))
            if value:
                calibrated.append(value)
                return -value["raw_fidelity"]
            return 1.0

        minimize_scalar(
            objective,
            bounds=(
                max(time_low, center - time_step),
                min(time_high, center + time_step),
            ),
            method="bounded",
            options={"xatol": 0.03, "maxiter": 20},
        )
    final = max(calibrated, key=lambda v: v["raw_fidelity"])
    return {
        "status": "calibrated",
        "coupler_handoff_off_target": off_target_definition("five"),
        "resonance_channel": channel,
        "logical_source": logical_source,
        "channel_selection": selected_channel,
        "duration_window_ns": [time_low, time_high],
        "ec_GHz": ec,
        "S1_position": best,
        "frequency_boundary": boundary,
        "calibration_time_boundary": min(
            abs(final["time_ns"] - time_low), abs(final["time_ns"] - time_high)
        )
        < 0.05,
        "calibration_detuning_boundary": abs(final["B_frequency_offset_MHz"]) > 11.95,
        "fA_GHz": f_a,
        "g_GHz": g,
        "logical_splittings_GHz": [split_a, split_b],
        "final": final,
        "screening": records,
        "calculation_seconds": time.perf_counter() - started,
    }
