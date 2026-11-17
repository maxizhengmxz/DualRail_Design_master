"""Bounded standalone chi maximization with an audited H-margin fallback.

This operates on the explicit-pad designer's coordinates and constraints. It
does not retune a coupler, change loaded ECs, or modify global readout policy.
"""

import time

import numpy as np
from scipy.optimize import minimize

H_FLOORS = (0.96, 0.95)
H_SEARCH_MARGIN = 1e-4


def maximize_chi(
    initial,
    starts,
    bounds,
    evaluate,
    equalities,
    inequalities,
    audit,
    failures,
    strength,
    *,
    maxiter=60,
    max_audits=3,
    objective=None,
    objective_name="maximize_absolute_signed_mean_chi",
    scope="nominal_frequency_initialization_only",
    h_floors=H_FLOORS,
    audit_with_floor=False,
):
    """Maximize absolute signed-mean chi, preferring H>96% over any H>95% point.

    Callbacks keep all topology-specific calibration constraints in the caller.
    The screening H buffer is numerical; acceptance uses the requested strict H
    floor at the audited basis. Search misses are not infeasibility certificates.
    Input seeds remain candidates, so a failed optimizer cannot erase a good seed.
    """
    begun = time.monotonic()
    lower, upper = np.asarray(bounds, dtype=float)
    seeds = []
    for start in [initial, *starts]:
        x = np.clip(np.asarray(start, dtype=float), lower, upper)
        if not any(np.linalg.norm(x - old) < 1e-8 for old in seeds):
            seeds.append(x)
    seeds = seeds[:4]
    tiers = []
    for floor in h_floors:
        candidates, attempts = [], []

        def consider(x, candidates=candidates, floor=floor):
            x = np.asarray(x, dtype=float)
            if (
                np.any(x < lower)
                or np.any(x > upper)
                or any(np.linalg.norm(x - old) < 1e-8 for old in candidates)
            ):
                return
            point = evaluate(x)
            if (
                np.all(np.isfinite(equalities(x)))
                and np.max(abs(np.asarray(equalities(x))), initial=0.0) <= 1e-6
                and min(inequalities(x, floor)) >= -1e-7
                and point["hybridization"] > floor
            ):
                candidates.append(x.copy())

        for seed in seeds:
            try:
                consider(seed)
                fit = minimize(
                    objective
                    if objective is not None
                    else lambda x: -strength(evaluate(x)),
                    seed,
                    method="SLSQP",
                    bounds=list(zip(lower, upper, strict=True)),
                    constraints=[
                        dict(type="eq", fun=equalities),
                        dict(
                            type="ineq",
                            fun=lambda x, floor=floor: inequalities(x, floor),
                        ),
                    ],
                    options=dict(maxiter=maxiter, ftol=1e-8, eps=1e-6),
                )
                attempts.append(
                    dict(
                        x=fit.x.tolist(),
                        success=bool(fit.success),
                        message=str(fit.message),
                    )
                )
                consider(fit.x)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                attempts.append(dict(error=str(error)))

        audited, accepted = [], []
        candidates.sort(key=lambda x: strength(evaluate(x)), reverse=True)
        for x in candidates[:max_audits]:
            try:
                result = audit(x, floor) if audit_with_floor else audit(x)
                point = result["point"]
                rejected = list(failures(point))
                if not point["hybridization"] > floor:
                    rejected.append("requested_strict_H_floor")
                audited.append(
                    dict(
                        x=x.tolist(),
                        failures=rejected,
                        H=point["hybridization"],
                        absolute_mean_chi_mhz=strength(point),
                    )
                )
                if not rejected:
                    accepted.append((strength(point), x, result))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                audited.append(dict(x=x.tolist(), error=str(error)))
        tiers.append(dict(H_floor=floor, attempts=attempts, audits=audited))
        if accepted:
            _, x, result = max(accepted, key=lambda row: row[0])
            fallback = floor != H_FLOORS[0]
            warning = (
                "No audited solution found with H > 96%; using the H > 95% "
                "fallback. The requested extra H margin was not achieved."
                if fallback
                else None
            )
            return dict(
                status="validated_pass",
                coordinates=x.tolist(),
                selected=result,
                selected_H_floor=floor,
                fallback_used=fallback,
                warning=warning,
                tiers=tiers,
                objective=objective_name,
                scope=scope,
                global_optimum_certified=False,
                seconds=time.monotonic() - begun,
            )
    return dict(
        status="no_validated_solution_found",
        coordinates=None,
        selected=None,
        selected_H_floor=None,
        fallback_used=False,
        warning="No audited readout solution found at H > 96% or H > 95%; "
        "do not hand off an unoptimized point as an accepted design.",
        tiers=tiers,
        objective=objective_name,
        scope=scope,
        global_optimum_certified=False,
        seconds=time.monotonic() - begun,
    )


def readout_handoff(readouts):
    """Reject failed new designs and preserve the H fallback through integration.

    Historical saved designs remain readable without pretending they underwent
    the new optimization. They are explicitly labelled as legacy records.
    """
    summaries = {}
    for pair, result in readouts.items():
        optimization = result.get("readout_strength_optimization")
        if result.get("workflow_failures") or (
            optimization and optimization["status"] != "validated_pass"
        ):
            raise ValueError(
                f"{pair}: rejected standalone readout design; "
                "resolve its workflow/optimization failures before integration"
            )
        if optimization and (
            optimization.get("scope") == "nominal_frequency_initialization_only"
            or optimization.get("objective") == "maximize_absolute_signed_mean_chi"
        ):
            raise ValueError(
                f"{pair}: nominal-frequency initialization is not prescribed-range "
                "validation. Run prescribed-range optimization before handing "
                "this new result to the seven-mode workflow."
            )
        if optimization and optimization.get("scope") == "prescribed_range_validated":
            validation = optimization.get("range_validation", {})
            if not validation.get("passed"):
                raise ValueError(
                    f"{pair}: prescribed range has no passing validation record"
                )
        summaries[pair] = dict(
            optimization_status=(
                optimization["status"]
                if optimization
                else "legacy_no_optimization_record"
            ),
            selected_H_floor=(
                optimization.get("selected_H_floor") if optimization else None
            ),
            fallback_used=bool(optimization and optimization.get("fallback_used")),
            warning=optimization.get("warning") if optimization else None,
            range_ghz=optimization.get("range_ghz") if optimization else None,
            worst_H=optimization.get("range_validation", {}).get("worst_H")
            if optimization
            else None,
            worst_mean_chi_mhz=optimization.get("range_validation", {}).get(
                "worst_mean_chi_mhz"
            )
            if optimization
            else None,
        )
    return summaries
