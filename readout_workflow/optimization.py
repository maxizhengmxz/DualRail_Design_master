"""Maximize readout strength at any fixed capacitance pair, then continue it.

First maximize |(chi+ + chi-)/2| with matching <=1% and all existing screens.
Then continue a nearby design while retaining >=99.5% of the best validated
strength found at each pixel. Both phases audit the physical inputs at the
large basis and on the independent EJ sign grid. This is a best-found search,
not a global optimum or fabrication robustness certificate.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from readout_workflow.validation import metrics

from . import calculation as study
from core.transmon import Basis, transmon_spectrum
from core.electrostatics import exchange_j01_ghz

AUDIT = Basis(25, 12, 10, 60)
SCALES = np.array([0.05, 1.0, 0.25, 1.0])
RETAIN = 0.995


def solve(task):
    """Multistart maximum-strength or constrained nearest-neighbor solve."""
    started = time.monotonic()
    c1 = task["c1"]
    c2 = task["c2"]
    mode = task["mode"]
    lower, upper = map(np.array, study.nominal_search_bounds())
    bounds = list(zip(lower, upper, strict=True))
    tier = study.BASELINE
    cache, attempts = {}, []
    target = np.array(task.get("target", [0, 0, 0, 0]))
    floor = task.get("strength_floor", 1.000002)

    def evaluate(x, basis):
        key = (basis, tuple(map(float, x)))
        if key not in cache:
            cache[key] = study.point_from_x(c1, c2, x, basis)
        return cache[key]

    def slacks(x, basis):
        p = evaluate(x, basis)
        return np.r_[
            study.slacks(p, tier, include_H=True, margin=1),
            (p["loaded_ec1_mhz"] - 150.0001) / 50,
            (p["loaded_ec2_mhz"] - 150.0001) / 50,
            (399.9999 - p["loaded_ec1_mhz"]) / 50,
            (399.9999 - p["loaded_ec2_mhz"]) / 50,
            (p["hybridization"] - 0.950002) / 0.05,
            study.absolute_mean_chi_mhz(p) - max(floor, 1.000002),
        ]

    def objective(x, basis):
        if mode == "smooth":
            return float(np.sum(((np.array(x) - target) / SCALES) ** 2))
        return -study.absolute_mean_chi_mhz(evaluate(x, basis))

    seeds = [np.array(study.x_from_point(p)) for p in task["warm"]]
    seeds += [np.array(x) for x in task.get("starts", [])]
    unique = []
    for x in seeds:
        x = np.clip(x, lower + 1e-6, upper - 1e-6)
        if not any(np.linalg.norm(x - old) < 1e-5 for old in unique):
            unique.append(x)
    with threadpool_limits(limits=1):
        for seed in unique:
            try:
                fit = minimize(
                    lambda x: objective(x, study.SCREENING_BASIS),
                    seed,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=[
                        dict(
                            type="ineq", fun=lambda x: slacks(x, study.SCREENING_BASIS)
                        )
                    ],
                    options=dict(maxiter=45, ftol=1e-9, eps=2e-6),
                )
                polish = minimize(
                    lambda x: objective(x, study.VALIDATION_BASIS),
                    fit.x,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=[
                        dict(
                            type="ineq", fun=lambda x: slacks(x, study.VALIDATION_BASIS)
                        )
                    ],
                    options=dict(maxiter=40, ftol=2e-10, eps=1e-6),
                )
                p = evaluate(polish.x, study.VALIDATION_BASIS)
                attempts.append(
                    dict(
                        seed=seed.tolist(),
                        screen_success=bool(fit.success),
                        polish_success=bool(polish.success),
                        message=str(polish.message),
                        iterations=int(polish.nit),
                        endpoint=p,
                        failures=study.failures(p),
                    )
                )
            except (ValueError, FloatingPointError) as error:
                attempts.append(dict(seed=seed.tolist(), error=repr(error)))

        candidates = [
            p
            for (b, _), p in cache.items()
            if b == study.VALIDATION_BASIS
            and not study.failures(p)
            and study.absolute_mean_chi_mhz(p) >= floor - 1e-7
        ]
        # Validated input seeds are also candidates. Recompute all physical
        # inputs at the audit basis; never accept another pixel's diagnostics.
        candidates += [
            p
            for p in task["warm"]
            if p["device"]["C1_shunt_fF"] == c1
            and p["device"]["C2_shunt_fF"] == c2
            and study.absolute_mean_chi_mhz(p) >= floor - 1e-7
        ]

        def score(p):
            if mode == "smooth":
                return float(
                    np.sum(((np.array(study.x_from_point(p)) - target) / SCALES) ** 2)
                )
            return -study.absolute_mean_chi_mhz(p)

        seen, audited = [], []
        for p in sorted(candidates, key=score):
            x = np.array(study.x_from_point(p))
            if any(np.linalg.norm((x - old) / SCALES) < 0.005 for old in seen):
                continue
            seen.append(x)
            model = study.ReadoutModel(study.Device(**p["device"]), AUDIT)
            q = study.annotate(model, p["ej1_ghz"], p["ej2_ghz"])
            q["x"] = study.x_from_point(q)
            if not study.failures(q) and study.absolute_mean_chi_mhz(q) >= floor - 1e-7:
                q = study.check_chi_sign_stability(q, model)
                if not study.failures(q, require_sign_check=True):
                    n1 = transmon_spectrum(q["ej1_ghz"], model.ec_ghz[0], AUDIT)[1]
                    n2 = transmon_spectrum(q["ej2_ghz"], model.ec_ghz[1], AUDIT)[1]
                    q["J12_ghz"] = float(model.J_ghz[0, 1])
                    q["J01_12_mhz"] = 1000 * exchange_j01_ghz(
                        model.J_ghz[0, 1], n1[0, 1], n2[0, 1]
                    )
                    q["g12_over_2pi_mhz"] = q["J01_12_mhz"]  # Legacy alias.
                    audited.append((q, p))
                    break  # Candidates sorted by the objective, then audited.
            if len(seen) >= 5:
                break
        result = dict(
            c1=c1,
            c2=c2,
            mode=mode,
            attempts=attempts,
            seconds=time.monotonic() - started,
            evaluations=len(cache),
            target=target.tolist() if mode == "smooth" else None,
            strength_floor=floor,
        )
        if audited:
            q, p = audited[0]
            result.update(
                status="validated_pass",
                point=q,
                screening_point=p,
                metrics=metrics(q),
                basis_ratio_difference=q["ratio"] - p["ratio"],
                failures=study.failures(q, require_sign_check=True),
            )
        else:
            result.update(status="no_validated_optimum")
        return result
