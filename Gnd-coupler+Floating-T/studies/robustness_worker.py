"""Bounded additional rescue; hard physics screens unchanged; no active source edits."""

import os

for k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[k] = "1"
import sys, pathlib, time, copy
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORKFLOW = HERE.parent
sys.path[:0] = [str(WORKFLOW), str(WORKFLOW.parent)]
import floating_coupler.fabrication as fab


def score_ro(result, ro):
    from readout_workflow.recovery import candidate_rank, recovery_tiers
    from standalone_pad_readout import policy

    policy(ro["case"])
    point = result.get("point") or result.get("diagnostic_point")
    if point is None:
        return (-1.0, -1e30, -1e30)
    tier = recovery_tiers(ro["case"], 45.0)[-1]
    return candidate_rank(
        point,
        tier,
        float(
            ro.get("nominal", {}).get(
                "operating_frequency_ghz", ro["case"]["bare_frequency_guide_ghz"]
            )
        ),
    )


def missing_gate_diagnostic(model, seed, off):
    from floating_coupler.calibrate_on import PhaseCalibration

    upper = min(10.999, off.get("fc_ghz", 11.0) - 0.001)
    rows = []
    errors = []
    if upper <= 7.0001:
        return dict(
            found=False,
            approximate=False,
            status="No allowed ON interval; no finite gate diagnostic",
        )
    for fc in np.unique(
        np.clip(
            [seed["fc_ghz"], seed["fc_ghz"] + 0.15, seed["fc_ghz"] - 0.15],
            7.0001,
            upper,
        )
    ):
        cal = PhaseCalibration(
            model,
            float(fc),
            fab.SEARCH,
            [seed["fb_ghz"] - 0.2, seed["fb_ghz"] + 0.2],
            phase_samples=25,
        )
        for fb in seed["fb_ghz"] + np.array([0.0, -0.02, 0.02, -0.08, 0.08, -0.2, 0.2]):
            for t in [70.0, 100.0, 130.0]:
                try:
                    row = cal.at(float(fb), float(t))
                    ph = abs(float(np.angle(np.exp(1j * row["phase_error_rad"]))))
                    if not np.isfinite(ph + row["postselected_cz_fidelity"]):
                        continue
                    controls = all(45 * model.ec[i] < model.ej[i] < 25 for i in [0, 1])
                    from floating_coupler.design import ej_for_frequency

                    controls = controls and all(
                        45 * model.ec[i] < ej_for_frequency(fb, model.ec[i], 35) < 25
                        for i in [3, 4]
                    )
                    valid = (
                        controls
                        and row["minimum_dressed_reference_overlap"] >= 0.8
                        and row["actual_bias_score_diagnostics"][
                            "maximum_logical_ratio"
                        ]
                        < 0.2
                    )
                    rows.append((valid, ph, row))
                except (ValueError, RuntimeError, KeyError) as e:
                    errors.append(str(e))
    if not rows:
        return dict(
            found=False,
            approximate=False,
            status="No finite diagnostic after bounded fallback",
            errors=errors[:3],
        )
    legal = [r for r in rows if r[0]]
    pool = legal or rows
    phase = min(q[1] for q in pool)
    ties = [q for q in pool if q[1] <= phase + 1e-7]
    valid, ph, best = max(ties, key=lambda q: q[2]["postselected_cz_fidelity"])
    return dict(
        found=False,
        approximate=True,
        phase_pi_met=False,
        best=best,
        constraints_valid=bool(valid),
        status="Finite diagnostic only; not phase-pi accepted"
        + ("" if valid else "; hardware/overlap/interaction screens violated"),
        objective="Closest wrapped conditional phase, then selected fidelity; legal controls preferred",
        converged=False,
    )


def worker(task):
    started = time.perf_counter()
    base = task["base"]
    prior = task.pop("prior_result", None)
    kind = task["kind"]
    result = fab.sample(task)
    notes = []
    # Additional starts for a retained failure; use the same ordered rescue ladder and limits.
    if prior is not None and result.get("readout"):
        retained = copy.deepcopy(result["readout"])
        changed = False
        for pair in ["A", "B"]:
            if retained[pair].get("passed"):
                continue
            ro = base["readouts"][pair]
            point = retained[pair].get("diagnostic_point")
            if point is None:
                continue
            best = retained[pair]
            for offset in [-0.03, 0.03]:
                warm = copy.deepcopy(ro)
                warm["coordinates"][:2] = [
                    point["bare_f1_ghz"] + offset,
                    point["bare_f2_ghz"] + offset,
                ]
                cand = fab.readout_fit(warm, task["errors"], True)
                if (cand.get("passed") and not best.get("passed")) or (
                    cand.get("passed") == best.get("passed")
                    and score_ro(cand, ro) > score_ro(best, ro)
                ):
                    best = cand
                if best.get("passed"):
                    break
            if best is not retained[pair]:
                retained[pair] = best
                changed = True
        if changed:
            rerun = dict(task, retained_readouts=retained)
            result = fab.sample(rerun)
            notes.append(
                "Additional readout seed; downstream recomputed even if still rejected"
            )
    if result.get("idle_shifts_ghz"):
        model, _ = fab.five_model(base, task["errors"], result["idle_shifts_ghz"])
        seed = copy.deepcopy(base["gate_seed"])
        seed["fb_ghz"] += (
            sum(result["idle_shifts_ghz"].get(n, 0) for n in ["B1", "B2"]) / 2
        )
        gate = result.get("gate", {})
        if (
            not gate.get("found")
            or gate.get("best", {}).get("postselected_cz_fidelity", 0) < 0.999
        ):
            tries = [gate]
            for offset in [-0.15, 0.15]:
                alt = dict(seed)
                alt["fc_ghz"] = float(
                    np.clip(
                        alt["fc_ghz"] + offset,
                        7.0001,
                        min(10.999, result.get("off", {}).get("fc_ghz", 11) - 0.001),
                    )
                )
                try:
                    tries.append(fab.gate_fit(model, alt, result["off"], True))
                except (ValueError, RuntimeError) as e:
                    notes.append("Gate restart error " + repr(e))
                accepted = [q for q in tries if q.get("found")]
                if (
                    accepted
                    and max(q["best"]["postselected_cz_fidelity"] for q in accepted)
                    >= 0.999
                ):
                    break
            accepted = [q for q in tries if q.get("found")]
            if accepted:
                result["gate"] = max(
                    accepted, key=lambda q: q["best"]["postselected_cz_fidelity"]
                )
            else:
                candidates = [q for q in tries if q.get("best")]
                if candidates:
                    result["gate"] = min(
                        candidates,
                        key=lambda q: abs(
                            np.angle(np.exp(1j * q["best"]["phase_error_rad"]))
                        ),
                    )
            notes.append(
                "Bounded extra gate starts; selected-fidelity objective; no fidelity failure threshold"
            )
        if not result.get("gate", {}).get("best"):
            result["gate"] = missing_gate_diagnostic(model, seed, result.get("off", {}))
            notes.append(
                "Missing accepted gate retained as finite diagnostic when available"
            )
    result["followup"] = dict(
        notes=notes,
        prior_id=prior.get("id") if prior else None,
        hard_constraints_unchanged=True,
        readout_mean_chi_floor_mhz=1,
        seconds=time.perf_counter() - started,
    )
    result["seconds"] = time.perf_counter() - started
    result["kind"] = kind
    return result
