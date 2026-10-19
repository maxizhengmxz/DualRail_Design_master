"""Explain unsuccessful pixels using nearest found 1%-matched candidates.

Minimize the L2 norm of normalized constraint deficits at fixed C1/C2.
All failed conditions and source seed IDs are retained. A bounded search miss
is not proof of physical infeasibility; the dominant label depends on SCALES.
"""

import time
from datetime import UTC, datetime

import numpy as np
from scipy.optimize import least_squares, minimize
from threadpoolctl import threadpool_limits

from . import calculation as study

SCALES = dict(
    hybridization=0.05,
    dispersive=0.1,
    assignment_overlap=0.05,
    logical_frequency=0.01,
    operation_frequency=0.5,
    operation_below_readout=0.5,
    EC_range=50.0,
    EJ_limit=5.0,
    EJ_over_EC=10.0,
    absolute_mean_chi=1.0,
)

DIAGNOSTIC_BUDGET = dict(starts=5, fit_nfev=30, polish_iterations=35, seconds=70)

LOWER = np.array([2.5, -1.0, 0.005 / 2, 0.1 / 5])

UPPER = np.array([9.5, 1.0, 15.0 / 2, 25.0 / 5])


def utc():
    return datetime.now(UTC).isoformat()


def deficits(p):
    c = study.CONDITIONS
    op = p["operating_frequency_ghz"]
    ec = [p[f"loaded_ec{i}_mhz"] for i in (1, 2)]
    values = dict(
        hybridization=max(0, c.hybridization_min - p["hybridization"]),
        dispersive=max(0, p["lambda_max"] - c.dispersive_max),
        assignment_overlap=max(
            0, c.reference_overlap_min - p["min_assignment_overlap"]
        ),
        logical_frequency=max(
            0,
            c.logical_frequency_ghz[0] - p["splitting_ghz"],
            p["splitting_ghz"] - c.logical_frequency_ghz[1],
        ),
        operation_frequency=max(
            0, c.operation_frequency_ghz[0] - op, op - c.operation_frequency_ghz[1]
        ),
        operation_below_readout=max(0, op - p["dressed_r_ghz"]),
        EC_range=max(0, c.loaded_ec_mhz[0] - min(ec), max(ec) - c.loaded_ec_mhz[1]),
        EJ_limit=max(0, max(p["ej1_ghz"], p["ej2_ghz"]) - c.ej_upper_ghz),
        EJ_over_EC=max(0, c.ej_over_ec_min - min(p["ej1_over_ec1"], p["ej2_over_ec2"])),
        absolute_mean_chi=max(
            0, c.abs_mean_chi_min_mhz - study.absolute_mean_chi_mhz(p)
        ),
    )
    return {k: float(v / SCALES[k]) for k, v in values.items()}


def diagnostic_coordinates(p):
    x = study.x_from_point(p)
    return np.clip([x[0], x[1] * 0.01, x[2] / 2, x[3] / 5], LOWER + 1e-7, UPPER - 1e-7)


def diagnose(task):
    pair = task["pair"]
    started = time.monotonic()
    cache = {}
    attempts = []

    def point(y):
        key = tuple(map(float, y))
        if key not in cache:
            cache[key] = study.point_from_x(
                *pair, [y[0], y[1] / 0.01, y[2] * 2, y[3] * 5]
            )
        return cache[key]

    def score(p):
        return float(np.linalg.norm(list(deficits(p).values())))

    starts = []
    for p in task["warm"]:
        y = diagnostic_coordinates(p)
        if not any(np.linalg.norm(y - z) < 1e-5 for z in starts):
            starts.append(y)
    if not starts:
        starts = [np.array([4.4, 0, 1, 1])]
    for frequency in (4.4, 5.2, 7.5):
        y = starts[0].copy()
        y[0] = frequency
        starts.append(y)
    with threadpool_limits(limits=1):
        for index, seed in enumerate(starts[: DIAGNOSTIC_BUDGET["starts"]]):
            if index >= 2 and time.monotonic() - started > DIAGNOSTIC_BUDGET["seconds"]:
                break
            try:
                fit = least_squares(
                    lambda y: np.r_[
                        max(0, point(y)["ratio"] - 0.00995) / 0.01 * 100,
                        list(deficits(point(y)).values()),
                    ],
                    seed,
                    bounds=(LOWER, UPPER),
                    max_nfev=DIAGNOSTIC_BUDGET["fit_nfev"],
                    diff_step=2e-5,
                    ftol=2e-6,
                    xtol=2e-6,
                    gtol=2e-6,
                )
                opt = minimize(
                    lambda y: score(point(y)) ** 2,
                    fit.x,
                    method="SLSQP",
                    bounds=list(zip(LOWER, UPPER, strict=True)),
                    constraints=[
                        dict(
                            type="ineq",
                            fun=lambda y: (0.009999 - point(y)["ratio"]) / 0.01,
                        )
                    ],
                    options=dict(
                        maxiter=DIAGNOSTIC_BUDGET["polish_iterations"],
                        ftol=2e-7,
                        eps=1e-5,
                    ),
                )
                p = point(opt.x)
                attempts.append(
                    dict(
                        seed=seed.tolist(),
                        fit_nfev=int(fit.nfev),
                        optimizer_success=bool(opt.success),
                        message=str(opt.message),
                        endpoint=p,
                        normalized_deficits=deficits(p),
                    )
                )
                if p["ratio"] <= 0.01 and score(p) < 1e-6:
                    break
            except (ValueError, FloatingPointError) as error:
                attempts.append(dict(seed=seed.tolist(), error=repr(error)))
        matched = [p for p in cache.values() if p["ratio"] <= 0.01 + 1e-8]
        reliable = [p for p in matched if p["min_assignment_overlap"] >= 0.95 - 1e-8]
        ranked = sorted(reliable or matched, key=score)
        chosen_seeds = []
        for p in ranked:
            if not any(
                np.linalg.norm(diagnostic_coordinates(p) - diagnostic_coordinates(q))
                < 1e-4
                for q in chosen_seeds
            ):
                chosen_seeds.append(p)
            if len(chosen_seeds) >= 4:
                break
        validated = []
        for p in chosen_seeds:
            model = study.ReadoutModel(
                study.Device(**p["device"]), study.VALIDATION_BASIS
            )
            q = study.annotate(model, p["ej1_ghz"], p["ej2_ghz"])
            q["x"] = study.x_from_point(q)
            validated.append(q)
        reliable = [
            p
            for p in validated
            if p["ratio"] <= 0.01 + 1e-8 and p["min_assignment_overlap"] >= 0.95 - 1e-8
        ]
        if reliable:
            p = min(reliable, key=score)
            p = study.check_chi_sign_stability(p)
            # A sign-unstable nearest candidate must not hide another fully
            # acceptable witness discovered during this same diagnostic job.
            if study.failures(p, require_sign_check=True):
                for alternative in sorted(reliable, key=score):
                    if study.failures(alternative):
                        continue
                    alternative = study.check_chi_sign_stability(alternative)
                    if not study.failures(alternative, require_sign_check=True):
                        p = alternative
                        break
            failed = study.failures(p, require_sign_check=True)
            status = "recovered_viable" if not failed else "matched_reliable"
            d = deficits(p)
            ranked_failures = [name for name in failed if name in d]
            dominant = (
                max(ranked_failures, key=lambda name: d[name])
                if ranked_failures
                else ("chi_sign_stability" if failed else "recovered_viable")
            )
        else:
            p = (
                min(validated, key=lambda q: (max(0, q["ratio"] - 0.01), score(q)))
                if validated
                else (
                    min(
                        cache.values(),
                        key=lambda q: (max(0, q["ratio"] - 0.01), score(q)),
                    )
                    if cache
                    else None
                )
            )
            failed = (
                study.failures(p, require_sign_check=True) if p else ["no_evaluation"]
            )
            status = "no_reliable_chi_match"
            dominant = status
            d = deficits(p) if p else {}
    return dict(
        c1=pair[0],
        c2=pair[1],
        status=status,
        point=p,
        failures=failed,
        dominant=dominant,
        normalized_deficits=d,
        score=score(p) if p else None,
        attempts=attempts,
        validation_candidates=validated,
        evaluations=len(cache),
        seconds=time.monotonic() - started,
        completed_utc=utc(),
        seed_source_ids=task.get("seed_source_ids", []),
    )
