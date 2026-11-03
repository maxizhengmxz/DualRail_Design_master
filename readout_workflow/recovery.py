"""EJ-only fabrication recovery with the established readout rescue ladder.

Order restored from b0964c3:readout_workflow/reordered_robustness.py.
Current case windows and approved EJ/EC baseline replace historical defaults.
The final user-approved rescue permits 5% chi mismatch. Capacitances, EJ
ceiling, chi strength and dispersive/overlap screens are never relaxed. A bounded miss is not proof of physical infeasibility.
"""

import time
from copy import deepcopy
from dataclasses import asdict

import numpy as np
from scipy.optimize import least_squares, minimize

from core.transmon import Basis, transmon_spectrum

from . import calculation as study


def recovery_tiers(case, ej_ec_min=45.0):
    """Default OPT-IN fabrication rescue; not nominal readout acceptance.

    Start at baseline and stop on the first feasible tier. Later entries are
    cumulative, including 5% mismatch only at the final tier. Preserve this
    case's supplied logical windows; report the tier rather than treating a
    rescued point as a baseline pass.
    """
    window = list(case["dressed_split_window_ghz"])
    first = dict(
        name="baseline",
        ec_min=150.0,
        H=0.95,
        logical_frequency=window,
        ej_ec=ej_ec_min,
        fixed=True,
        mismatch_max=study.CONDITIONS.mismatch_max,
    )
    stages = [first]
    for name, edits in [
        ("EC120", dict(ec_min=120.0)),
        ("free_operation_frequency", dict(fixed=False)),
        (
            "wider_logical_frequency",
            dict(logical_frequency=[max(0, window[0] - 0.01), window[1] + 0.01]),
        ),
        ("H90", dict(H=0.90)),
        ("EJ_EC40", dict(ej_ec=40.0)),
        ("chi_matching_5percent", dict(mismatch_max=0.05)),
    ]:
        stages.append(dict(deepcopy(stages[-1]), name=name, **edits))
    return stages


def candidate_rank(point, tier, target):
    """Feasible first; healthy matched diagnostics outrank unphysical endpoints."""
    failures = study.reasons(point, tier, target)
    slacks = study.slacks(point, tier, include_H=True)
    if not np.isfinite(slacks).all():
        return (-1.0, -float("inf"), -float("inf"))
    error = np.linalg.norm(np.minimum(slacks, 0.0))
    if tier["fixed"]:
        error = np.hypot(error, (point["operating_frequency_ghz"] - target) / 0.02)
    meaningful = not (
        set(failures) - {"absolute_mean_chi", "fixed_operation_frequency"}
    )
    return (
        3.0 if not failures else 2.0 if meaningful else 1.0,
        -float(error),
        float(point["hybridization"]),
    )


class _Found(Exception):
    pass


class _Budget(Exception):
    pass


class Calibration:
    def __init__(self, device, basis, target, reference):
        self.model = study.ReadoutModel(device, basis)
        self.basis = basis
        self.target = target
        self.reference = reference
        self.best = None
        self.current_tier = None
        self.calls = 0
        self.cache = {}
        self.started = 0.0
        self.seconds = 12.0
        ec = self.model.ec_ghz[:2]
        low = [transmon_spectrum(40.0001 * e, e, basis)[0][1] for e in ec]
        high = [
            transmon_spectrum(study.CONDITIONS.ej_upper_ghz - 0.0001, e, basis)[0][1]
            for e in ec
        ]
        # Difference bounds cover every allowed 70-170 MHz rail splitting;
        # they are a finite near-resonant search domain, not a new acceptance cut.
        self.bounds = (np.array([sum(low) / 2, -20.0]), np.array([sum(high) / 2, 20.0]))

    def evaluate(self, mean, difference):
        key = (float(mean), float(difference))
        if key in self.cache:
            return self.cache[key]
        ec = self.model.ec_ghz[:2]
        fs = [mean + 0.005 * difference, mean - 0.005 * difference]
        ej = [
            study.inverse_frequency(f, e, self.basis)
            for f, e in zip(fs, ec, strict=True)
        ]
        point = self.at_ej(*ej)
        if len(self.cache) >= 512:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = point
        return point

    def at_ej(self, ej1, ej2):
        return study.annotate(self.model, float(ej1), float(ej2))

    def point(self, x):
        if time.monotonic() - self.started > self.seconds:
            raise _Budget()
        p = self.evaluate(*map(float, x))
        self.calls += 1
        if self.best is None or candidate_rank(
            p, self.current_tier, self.target
        ) > candidate_rank(self.best, self.current_tier, self.target):
            self.best = p
        if not study.reasons(p, self.current_tier, self.target):
            raise _Found()
        return p

    @staticmethod
    def coordinates(p):
        return np.array(
            [
                (p["bare_f1_ghz"] + p["bare_f2_ghz"]) / 2,
                (p["bare_f1_ghz"] - p["bare_f2_ghz"]) / 0.01,
            ]
        )

    def solve(self, tier, warm=None, seconds=12.0):
        self.current_tier = tier
        self.best = None
        self.started = time.monotonic()
        self.seconds = seconds
        before = self.calls
        attempts = []
        original = np.array(
            [sum(self.reference) / 2, (self.reference[0] - self.reference[1]) / 0.01]
        )
        seeds = [
            self.coordinates(warm) if warm is not None else original,
            original + [0, 0.6],
            original - [0, 0.6],
        ]
        if not tier["fixed"]:
            upper = min(
                transmon_spectrum(study.CONDITIONS.ej_upper_ghz - 0.01, e, self.basis)[
                    0
                ][1]
                for e in self.model.ec_ghz[:2]
            )
            lower = max(
                transmon_spectrum((tier["ej_ec"] + 0.01) * e, e, self.basis)[0][1]
                for e in self.model.ec_ghz[:2]
            )
            seeds += [
                np.array([upper - 0.005, original[1]]),
                original + [0.1, 0.0],
                original - [0.1, 0.0],
                np.array([(lower + upper) / 2, original[1]]),
            ]
            seeds = [seeds[0], seeds[3], *seeds[1:3], *seeds[4:]]

        def residual(x):
            p = self.point(x)
            v = np.minimum(study.slacks(p, tier, include_H=True, margin=1), 0.0)
            return (
                np.r_[v, (p["operating_frequency_ghz"] - self.target) / 0.02]
                if tier["fixed"]
                else v
            )

        try:
            for seed in seeds[:4]:
                seed = np.clip(seed, self.bounds[0] + 1e-6, self.bounds[1] - 1e-6)
                self.point(seed)
                fit = least_squares(
                    residual,
                    seed,
                    bounds=self.bounds,
                    max_nfev=24,
                    diff_step=2e-5,
                    xtol=1e-7,
                    ftol=1e-6,
                    gtol=1e-6,
                )
                constraints = [
                    dict(
                        type="ineq",
                        fun=lambda x: study.slacks(
                            self.point(x), tier, include_H=True, margin=1
                        ),
                    )
                ]
                if tier["fixed"]:
                    constraints.append(
                        dict(
                            type="eq",
                            fun=lambda x: (
                                (self.point(x)["operating_frequency_ghz"] - self.target)
                                / 0.02
                            ),
                        )
                    )
                opt = minimize(
                    lambda x: -self.point(x)["hybridization"],
                    fit.x,
                    method="SLSQP",
                    bounds=list(zip(*self.bounds, strict=True)),
                    constraints=constraints,
                    options=dict(maxiter=24, ftol=1e-6, eps=1e-5),
                )
                self.point(opt.x)
                attempts.append(
                    dict(success=bool(opt.success), message=str(opt.message))
                )
        except _Found:
            reason = "first feasible candidate"
        except _Budget:
            reason = "finite time budget"
        else:
            reason = "finite starts exhausted"
        return dict(
            point=self.best,
            failures=study.reasons(self.best, tier, self.target)
            if self.best
            else ["no_reliable_candidate"],
            attempts=attempts,
            stop_reason=reason,
            evaluations=self.calls - before,
            seconds=time.monotonic() - self.started,
        )


def recover(device, reference, case, representative=False, bases=None):
    bases = bases or [Basis(20, 6, 5, 30), Basis(25, 8, 6, 40), Basis(25, 12, 10, 60)]
    target = float(
        case.get("recovery_target_dressed_ghz", case["bare_frequency_guide_ghz"])
    )
    tiers = recovery_tiers(case, study.CONDITIONS.ej_over_ec_min)
    calibrations = [Calibration(device, b, target, reference) for b in bases]
    fast = calibrations[0]
    stages = []
    warm = None
    accepted = None
    chosen_tier = None
    last_set = None
    for tier in tiers:
        ec = fast.model.ec_ghz[:2] * 1000
        if (
            min(ec) < tier["ec_min"] - 1e-7
            or max(ec) >= study.CONDITIONS.loaded_ec_mhz[1]
        ):
            stages.append(
                dict(
                    tier=tier,
                    passed=False,
                    failures=["EC_range"],
                    source="immutable EC screen",
                )
            )
            continue
        feasible_set = {k: v for k, v in tier.items() if k not in ("name", "ec_min")}
        if feasible_set == last_set:
            stages.append(
                dict(
                    tier=tier,
                    passed=False,
                    failures=stages[-1]["failures"],
                    source="inactive relaxation; identical feasible set",
                )
            )
            continue
        last_set = feasible_set
        result = fast.solve(tier, warm)
        p = result["point"]
        audits = []
        if p is not None:
            warm = p
        if p is not None and not result["failures"]:
            for cal in calibrations[1:]:
                q = cal.at_ej(p["ej1_ghz"], p["ej2_ghz"])
                failures = study.reasons(q, tier, target)
                audit = dict(basis=asdict(cal.basis), fixed_control_failures=failures)
                if failures:
                    repair = cal.solve(tier, q)
                    audit["repair"] = repair
                    q = repair["point"]
                    failures = repair["failures"]
                audits.append(audit)
                p = q
                if p is None or failures:
                    break
                near = (
                    p["ratio"] > 0.008
                    or abs(p["hybridization"] - tier["H"]) < 0.001
                    or abs(study.absolute_mean_chi_mhz(p) - 1) < 0.01
                )
                if cal is calibrations[1] and not (near or representative):
                    break
            if p is not None and not study.reasons(p, tier, target):
                p = study.check_chi_sign_stability(p, early_exit=True)
                failures = study.reasons(p, tier, target, require_sign_check=True)
                if not failures:
                    accepted = p
                    chosen_tier = tier
                result["failures"] = failures
            else:
                result["failures"] = (
                    study.reasons(p, tier, target) if p else ["no_reliable_candidate"]
                )
            if p is not None:
                warm = p
        result["tier"] = tier
        result["audits"] = audits
        result["passed"] = accepted is not None
        stages.append(result)
        if accepted is not None:
            break
    # A failed result has NO idle point. Its best diagnostic is explicitly separate.
    if accepted is None:
        return dict(
            point=None,
            passed=False,
            baseline_passed=False,
            relaxed=False,
            tier=None,
            ej_ghz=None,
            bare_frequencies_ghz=None,
            nominal_failures=stages[-1].get("failures", ["unrecovered"]),
            sign_passed=None,
            diagnostic_point=warm,
            history=stages,
            status="unrecovered under finite rescue ladder",
        )
    accepted = dict(
        accepted, robustness_mean_chi_abs_mhz=study.absolute_mean_chi_mhz(accepted)
    )
    baseline_failures = study.reasons(
        accepted, tiers[0], target, require_sign_check=True
    )
    return dict(
        point=accepted,
        passed=True,
        baseline_passed=not baseline_failures,
        baseline_failures=baseline_failures,
        relaxed=chosen_tier["name"] != "baseline",
        tier=chosen_tier["name"],
        tier_constraints=chosen_tier,
        ej_ghz=[accepted["ej1_ghz"], accepted["ej2_ghz"]],
        bare_frequencies_ghz=[accepted["bare_f1_ghz"], accepted["bare_f2_ghz"]],
        nominal_failures=[],
        sign_passed=True,
        history=stages,
        status="recovered at " + chosen_tier["name"],
    )
