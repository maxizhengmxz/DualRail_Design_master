# ruff: noqa: E402
# BLAS thread limits must be set before importing numerical libraries.
"""Fixed-hardware resonator/operation-frequency map with maximum mean chi.

The physical Cr and Lr co-vary at 50 ohms; all other physical capacitors stay
fixed. Each pixel fixes the dressed logical mean, leaving EJ differential
calibration to maximize abs((chi_minus+chi_plus)/2). Constraints, assignment and
sign validation are inherited from the saved standalone coupler readout case.
This is a bounded numerical maximum, not a global optimality certificate.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, root
from threadpoolctl import threadpool_limits

from . import calculation as study
from .resonator_frequency_tolerance import device_at_frequency, policy
from .storage import atomic_json, utc
from .validation import INTERMEDIATE, audit_point


def signed_ratio(p):
    return (p["chi_p_ghz"] - p["chi_m_ghz"]) / max(
        abs((p["chi_p_ghz"] + p["chi_m_ghz"]) / 2), 1e-12
    )


def constraints(p):
    """Scaled signed slacks; chi strength is the objective, not a constraint."""
    c = study.CONDITIONS
    return np.array(
        [
            (c.mismatch_max - abs(signed_ratio(p))) / 0.01,
            (p["hybridization"] - c.hybridization_min) / 0.05,
            (c.dispersive_max - p["lambda_max"]) / 0.1,
            (p["min_assignment_overlap"] - c.reference_overlap_min) / 0.05,
            (p["splitting_ghz"] - c.logical_frequency_ghz[0]) / 0.02,
            (c.logical_frequency_ghz[1] - p["splitting_ghz"]) / 0.02,
            *[(c.ej_upper_ghz - p[f"ej{i}_ghz"]) / 10 for i in (1, 2)],
            *[(p[f"ej{i}_over_ec{i}"] - c.ej_over_ec_min) / 50 for i in (1, 2)],
            *[(p[f"loaded_ec{i}_mhz"] - c.loaded_ec_mhz[0]) / 100 for i in (1, 2)],
            *[(c.loaded_ec_mhz[1] - p[f"loaded_ec{i}_mhz"]) / 100 for i in (1, 2)],
            p["operating_frequency_ghz"] - c.operation_frequency_ghz[0],
            c.operation_frequency_ghz[1] - p["operating_frequency_ghz"],
            p["dressed_r_ghz"] - p["operating_frequency_ghz"],
        ]
    )


def solve_pixel(task):
    ro, fr, target = task["readout"], task["fr"], task["operation"]
    policy(ro["case"])
    started = time.monotonic()
    dev, _ = device_at_frequency(ro, fr)
    model = study.ReadoutModel(dev, INTERMEDIATE)
    cache = {}
    trials = []

    def evaluate(x):
        key = tuple(map(float, x))
        if key not in cache:
            fs = [x[0] + x[1] * 0.0005, x[0] - x[1] * 0.0005]
            ej = [
                study.inverse_frequency(float(f), float(ec), INTERMEDIATE)
                for f, ec in zip(fs, model.ec_ghz[:2], strict=True)
            ]
            cache[key] = study.annotate(model, *ej)
        return cache[key]

    def equal(x):
        return (evaluate(x)["operating_frequency_ghz"] - target) / 0.01

    def admissible(p, tol=2e-6):
        return (
            abs(p["operating_frequency_ghz"] - target) <= 1e-6
            and min(constraints(p)) >= -tol
        )

    with threadpool_limits(limits=1):
        start = np.array([target + 0.003, 0.0])
        for rho in (0.0, -0.00998, 0.00998):

            def equations(x, rho=rho):
                if not (3.0 < x[0] < 7.0 and abs(x[1]) < 250):
                    return np.array([100.0 + abs(x[0]), 100.0 + abs(x[1])])
                p = evaluate(x)
                return [equal(x), signed_ratio(p) - rho]

            fit = root(
                equations,
                start,
                method="hybr",
                options={"xtol": 1e-8, "maxfev": 35, "eps": 1e-8},
            )
            if not (3.0 < fit.x[0] < 7.0 and abs(fit.x[1]) < 250):
                continue
            p = evaluate(fit.x)
            trials.append(
                dict(
                    kind="matching_band",
                    rho=rho,
                    x=fit.x.tolist(),
                    success=bool(fit.success),
                    point=p,
                )
            )
            if rho == 0.0:
                start = fit.x
        if not trials:
            return dict(
                bare_resonator_frequency_ghz=fr,
                operation_target_ghz=target,
                status="no_solution_found",
                point=None,
                seconds=time.monotonic() - started,
                trials=trials,
            )
        best = min(
            trials,
            key=lambda r: (
                max(0, -min(constraints(r["point"]))),
                -study.absolute_mean_chi_mhz(r["point"]),
            ),
        )
        # Matched roots outside immutable EJ/EC/operation bounds cannot be repaired
        # by sub-MHz differential calibration. Keep their diagnostic, no blind fit.
        hard_miss = any(min(constraints(r["point"])[6:10]) < -0.005 for r in trials)
        if not hard_miss:
            fit = minimize(
                lambda x: -study.absolute_mean_chi_mhz(evaluate(x)),
                best["x"],
                method="SLSQP",
                bounds=[(3.8, 5.8), (-40.0, 40.0)],
                constraints=[
                    {"type": "eq", "fun": equal},
                    {"type": "ineq", "fun": lambda x: constraints(evaluate(x)) - 1e-7},
                ],
                options={"maxiter": 18, "ftol": 2e-7, "eps": 1e-5},
            )
            p = evaluate(fit.x)
            trials.append(
                dict(
                    kind="maximum_mean_chi",
                    x=fit.x.tolist(),
                    success=bool(fit.success),
                    message=str(fit.message),
                    point=p,
                )
            )
        candidates = sorted(
            [p for p in cache.values() if admissible(p)],
            key=study.absolute_mean_chi_mhz,
            reverse=True,
        )
        chosen = None
        audits = []
        # Validate the strongest candidate. If edge rounding rejects it, try a
        # genuinely different inward candidate rather than the same cached point.
        used = []
        # Guarded matching roots must be tried explicitly: a coarse EJ
        # deduplication can otherwise discard the inward root together with
        # a numerically rejected optimizer endpoint near the 1% threshold.
        guarded = sorted(
            [
                r["point"]
                for r in trials
                if r["kind"] == "matching_band" and admissible(r["point"])
            ],
            key=study.absolute_mean_chi_mhz,
            reverse=True,
        )
        ordered = candidates[:1] + guarded + candidates[1:]
        for p in ordered:
            if len(used) >= 6:
                break
            if any(
                abs(p["ej1_ghz"] - q["ej1_ghz"]) + abs(p["ej2_ghz"] - q["ej2_ghz"])
                < 1e-8
                for q in used
            ):
                continue
            if study.failures(p):
                continue
            used.append(p)
            q = audit_point(p, representative=False)
            f = study.failures(q, require_sign_check=True)
            if abs(q["operating_frequency_ghz"] - target) > 1e-6:
                f.append("fixed_operation_frequency")
            audits.append(dict(failures=f, point=q))
            if not f:
                chosen = q
                break
        diagnostic = (
            max(candidates, key=study.absolute_mean_chi_mhz)
            if candidates
            else min(
                (r["point"] for r in trials),
                key=lambda p: np.linalg.norm(np.minimum(constraints(p), 0)),
            )
        )
    return dict(
        bare_resonator_frequency_ghz=fr,
        operation_target_ghz=target,
        status="validated_pass" if chosen else "no_validated_solution_found",
        point=chosen,
        best_screening=diagnostic,
        screening_maximum_mean_chi_mhz=study.absolute_mean_chi_mhz(diagnostic),
        other_constraint_slacks=constraints(diagnostic).tolist(),
        failures=study.failures(diagnostic),
        audits=audits,
        trials=trials,
        seconds=time.monotonic() - started,
        completed_utc=utc(),
        maximum_scope="Matched near-resonant branch, bounded differential-EJ search; strongest validated witness, no global certificate",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--resonator-range-ghz", type=float, nargs=2, default=(5.3, 6.9))
    p.add_argument("--operation-range-ghz", type=float, nargs=2, default=(4.4, 5.2))
    a = p.parse_args()
    if not a.run:
        print("Use --run to calculate.")
        return
    ro = json.loads(a.input.read_text(encoding="utf-8-sig"))
    policy(ro["case"])
    a.output.mkdir(exist_ok=True, parents=True)
    frs = sorted(
        set(
            [
                float(round(x, 9))
                for x in np.arange(
                    a.resonator_range_ghz[0], a.resonator_range_ghz[1] + 1e-8, 0.05
                )
            ]
            + [float(ro["coordinates"][4])]
        )
    )
    ops = [
        float(round(x, 9))
        for x in np.arange(
            a.operation_range_ghz[0], a.operation_range_ghz[1] + 1e-8, 0.01
        )
    ]
    tasks = [dict(readout=ro, fr=f, operation=o) for f in frs for o in ops]
    if a.pilot:
        fr0 = float(ro["coordinates"][4])
        op0 = float(ro["nominal"]["operating_frequency_ghz"])
        tasks = [
            dict(readout=ro, fr=f, operation=o)
            for f, o in [
                (fr0, op0),
                (fr0, op0 - 0.1),
                (fr0, op0 + 0.05),
                (fr0 - 0.3, op0 - 0.3),
                (fr0 + 0.2, op0),
                (fr0 + 0.2, op0 + 0.1),
            ]
        ]
    settings = dict(
        input=str(a.input.resolve()),
        input_sha256=sha256(a.input.read_bytes()).hexdigest(),
        conditions=asdict(study.CONDITIONS),
        basis=asdict(INTERMEDIATE),
        resonator_frequencies_ghz=frs,
        operation_frequencies_ghz=ops,
        pilot=a.pilot,
        scope="Cr/Lr vary at fixed 50 ohms; all other actual pad capacitors fixed. Fixed dressed operation frequency per pixel. Differential EJ calibrated to maximize absolute signed-mean chi. Original nominal constraints and sign validation retained.",
        source_hashes={
            str(q): sha256(q.read_bytes()).hexdigest()
            for q in [Path(__file__), Path(study.__file__)]
        },
    )
    saved = a.output / "settings.json"
    if saved.exists() and json.loads(saved.read_text()) != json.loads(
        json.dumps(settings)
    ):
        raise ValueError("Settings differ; choose new output")
    atomic_json(saved, settings)
    rows = []
    pending = []

    def path(t):
        return a.output / f"fr_{t['fr']:.9f}_op_{t['operation']:.6f}.json"

    for t in tasks:
        if path(t).exists():
            rows.append(json.loads(path(t).read_text()))
        else:
            pending.append(t)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(solve_pixel, t): t for t in pending}
        for future in as_completed(futures):
            r = future.result()
            rows.append(r)
            atomic_json(path(futures[future]), r)
            atomic_json(
                a.output / "progress.json",
                dict(
                    completed=len(rows),
                    total=len(tasks),
                    passed=sum(r["point"] is not None for r in rows),
                    elapsed_seconds=time.monotonic() - started,
                    updated_utc=utc(),
                ),
            )
            print(
                json.dumps(
                    dict(
                        done=len(rows),
                        total=len(tasks),
                        status=r["status"],
                        fr=r["bare_resonator_frequency_ghz"],
                        operation=r["operation_target_ghz"],
                        seconds=r["seconds"],
                    )
                ),
                flush=True,
            )
    rows.sort(
        key=lambda r: (r["bare_resonator_frequency_ghz"], r["operation_target_ghz"])
    )
    atomic_json(a.output / "results.json", rows)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
