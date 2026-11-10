"""Prescribed-range objective on ONE nominal physical-pad readout design.

Capacitors are synthesized for the specified nominal loaded ECs, once per
hardware candidate. Scenario evaluation changes physical resonator C/L and
rail EJs only. No coupler, spectator rail, capacitor re-fit or 6 GHz assumption
enters a scenario. The returned nominal record feeds existing seven-mode code.
"""

import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from readout_workflow import calculation as study
from readout_workflow.resonator_robust_design import frequency_grid
from readout_workflow.validation import audit_point

from .readout_optimization import H_SEARCH_MARGIN, maximize_chi


def required_range(case):
    """Ranges have no numerical default; the agent asks when design starts."""
    values = case.get("bare_resonator_frequency_range_ghz")
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 2
        or any(value is None for value in values)
    ):
        raise ValueError(
            "Supply bare_resonator_frequency_range_ghz = [LOW, HIGH] "
            "in GHz for this rail before starting an actual design."
        )
    low, high = map(float, values)
    if not all(np.isfinite([low, high])) or not 0 < low < high:
        raise ValueError("Resonator range requires positive increasing GHz bounds")
    return low, high


def case_for_pair(request, pair):
    """Accept four independent request bounds; never infer the other rail."""
    cases = deepcopy(request["pairs"])
    settings = request.get("standalone_readout_optimization", {})
    if settings.get("mode", "prescribed_resonator_range") not in (
        "prescribed",
        "prescribed_resonator_range",
    ):
        raise ValueError(
            "The explicit-pad design entry point uses prescribed ranges; "
            "run exploratory tolerance mapping separately in the readout workflow."
        )
    supplied = settings.get("bare_resonator_frequency_range_ghz", {})
    for name in ("A", "B"):
        if name not in cases:
            raise ValueError("Provide cases and resonator bounds for both A and B")
        bounds = supplied.get(name)
        existing = cases[name].get("bare_resonator_frequency_range_ghz")
        if bounds is not None and all(v is not None for v in bounds):
            if existing is not None and list(existing) != list(bounds):
                raise ValueError(f"Conflicting {name} resonator range inputs")
            cases[name]["bare_resonator_frequency_range_ghz"] = bounds
        required_range(cases[name])
        cases[name]["readout_range_options"] = dict(
            settings.get("numerical_defaults", {}),
            **cases[name].get("readout_range_options", {}),
        )
    return cases[pair]


def sample_slacks(point, floor, chi_floor_mhz):
    """Existing acceptance plus strict H and epigraph; EC may drift off-reference."""
    c = study.CONDITIONS
    return np.r_[
        study.slacks(point, dict(study.BASELINE, H=floor), include_H=True, margin=0.05),
        point["hybridization"] - floor - H_SEARCH_MARGIN,
        study.absolute_mean_chi_mhz(point) - chi_floor_mhz,
        *[(point[f"loaded_ec{i}_mhz"] - c.loaded_ec_mhz[0]) / 100 for i in (1, 2)],
        *[
            (c.loaded_ec_mhz[1] - point[f"loaded_ec{i}_mhz"] - 1e-6) / 100
            for i in (1, 2)
        ],
    ]


def checked_sample(point, floor, chi_floor_mhz):
    failures = list(study.failures(point, require_sign_check=True))
    if not point["hybridization"] > floor:
        failures.append("requested_strict_H_floor")
    if study.absolute_mean_chi_mhz(point) < chi_floor_mhz:
        failures.append("worst_case_chi_target")
    return failures


class PadRangeObjective:
    """Caches physical hardware separately from per-frequency EJ evaluations."""

    def __init__(
        self,
        case,
        seed,
        bounds,
        nominal_evaluate,
        nominal_equalities,
        nominal_inequalities,
        nominal_audit,
        builder,
        evaluator,
        basis,
        extra_frequencies=(),
    ):
        self.case = case
        self.low, self.high = required_range(case)
        self.options = case.get("readout_range_options", {})
        self.step = float(self.options.get("frequency_step_ghz", 0.05))
        self.seed = np.asarray(seed, float)
        lower, upper = np.asarray(bounds, float).copy()
        lower[4], upper[4] = max(lower[4], self.low), min(upper[4], self.high)
        if lower[4] >= upper[4]:
            raise ValueError(
                "Requested range does not overlap the allowed nominal LC reference bounds"
            )
        reference = float(np.clip(self.seed[4], lower[4], upper[4]))
        self.frequencies = frequency_grid(self.low, self.high, self.step, reference)
        self.frequencies = np.array(
            sorted(set(self.frequencies.tolist() + list(extra_frequencies)))
        )
        self.failed_frequencies = set()
        self.builder, self.evaluator, self.basis = builder, evaluator, basis
        self.nominal_evaluate = nominal_evaluate
        self.nominal_equalities = nominal_equalities
        self.nominal_inequalities = nominal_inequalities
        self.nominal_audit = nominal_audit
        self.hardware_cache, self.point_cache = OrderedDict(), OrderedDict()
        self.scenario_models = OrderedDict()
        self.hardware_builds = self.scenario_evaluations = 0
        # These are numerical EJ coordinates, not extra physical acceptance bounds.
        c = study.CONDITIONS
        self.bias_bounds = np.asarray(
            case.get(
                "readout_recalibration_bare_bounds_ghz",
                [
                    [
                        c.operation_frequency_ghz[0] - 0.2,
                        c.operation_frequency_ghz[1] + 0.2,
                    ]
                ]
                * 2,
            ),
            float,
        )
        if (
            self.bias_bounds.shape != (2, 2)
            or not np.isfinite(self.bias_bounds).all()
            or np.any(self.bias_bounds[:, 0] <= 0)
            or np.any(self.bias_bounds[:, 0] >= self.bias_bounds[:, 1])
        ):
            raise ValueError(
                "Readout recalibration bounds must be two positive increasing GHz intervals"
            )
        n = len(self.frequencies)
        self.bounds = (
            np.r_[lower, np.tile(self.bias_bounds[:, 0], n), c.abs_mean_chi_min_mhz],
            np.r_[upper, np.tile(self.bias_bounds[:, 1], n), 100.0],
        )
        sample = self.nominal_evaluate(self.seed)
        self.constraint_count = len(self.nominal_inequalities(self.seed)) + (
            n + 1
        ) * len(sample_slacks(sample, 0.96, c.abs_mean_chi_min_mhz))

    def hardware(self, coordinates):
        # Bare qubit frequencies do not enter physical capacitor synthesis.
        key = tuple(map(float, coordinates[2:7]))
        if key not in self.hardware_cache:
            self.hardware_cache[key] = self.builder(self.case, coordinates[:7])
            self.hardware_builds += 1
            if len(self.hardware_cache) > 256:
                self.hardware_cache.popitem(last=False)
        self.hardware_cache.move_to_end(key)
        return self.hardware_cache[key]

    def point(self, hardware, frequency, bare):
        key = (
            tuple(map(float, hardware[2:7])),
            float(frequency),
            tuple(map(float, bare)),
        )
        if key not in self.point_cache:
            from readout_workflow.resonator_frequency_tolerance import (
                device_at_frequency,
            )

            model_key = (key[0], key[1])
            if model_key not in self.scenario_models:
                _, physical = self.hardware(hardware)
                device, _ = device_at_frequency(
                    dict(case=self.case, physical=physical), frequency
                )
                self.scenario_models[model_key] = study.ReadoutModel(device, self.basis)
                if len(self.scenario_models) > 128:
                    self.scenario_models.popitem(last=False)
            self.scenario_models.move_to_end(model_key)
            local = np.array(hardware[:7], float)
            local[:2] = bare
            self.point_cache[key] = self.evaluator(
                self.case, local, self.basis, model=self.scenario_models[model_key]
            )[0]
            self.scenario_evaluations += 1
            if len(self.point_cache) > 8192:
                self.point_cache.popitem(last=False)
        self.point_cache.move_to_end(key)
        return self.point_cache[key]

    def evaluate(self, x):
        nominal = self.nominal_evaluate(x[:7])
        points = [
            self.point(x[:7], float(f), x[7 + 2 * j : 9 + 2 * j])
            for j, f in enumerate(self.frequencies)
        ]
        return dict(
            hybridization=min(p["hybridization"] for p in [nominal, *points]),
            worst_mean_chi_mhz=min(
                map(study.absolute_mean_chi_mhz, [nominal, *points])
            ),
            points=points,
            nominal=nominal,
        )

    def equalities(self, x):
        try:
            return self.nominal_equalities(x[:7])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return np.full(3, 1e4)

    def inequalities(self, x, floor):
        try:
            result = self.evaluate(x)
            return np.concatenate(
                [
                    np.atleast_1d(self.nominal_inequalities(x[:7])),
                    *[
                        sample_slacks(p, floor, x[-1])
                        for p in [result["nominal"], *result["points"]]
                    ],
                ]
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return np.full(self.constraint_count, -1e4)

    def start(self, hardware):
        hardware = np.asarray(hardware, float).copy()
        hardware = np.clip(hardware, self.bounds[0][:7], self.bounds[1][:7])
        guesses = []
        for frequency in self.frequencies:
            guesses.extend(
                np.clip(
                    hardware[:2] + 0.6 * (frequency - hardware[4]),
                    self.bias_bounds[:, 0],
                    self.bias_bounds[:, 1],
                )
            )
        return np.r_[hardware, guesses, study.CONDITIONS.abs_mean_chi_min_mhz]

    def validate_frequency(self, hardware, frequency, guesses, floor, chi_floor):
        """EJ-only repair at fixed hardware; no nominal EC compensation here."""
        attempts = []
        for guess in guesses[:2]:
            guess = np.clip(guess, self.bias_bounds[:, 0], self.bias_bounds[:, 1])
            proposals = [guess]
            # First-feasible recourse avoids an optimizer for already good biases.
            for optimize in (False, True):
                try:
                    if optimize:
                        fit = minimize(
                            lambda v: (
                                -study.absolute_mean_chi_mhz(
                                    self.point(hardware, frequency, v)
                                )
                            ),
                            guess,
                            method="SLSQP",
                            bounds=list(map(tuple, self.bias_bounds)),
                            constraints=[
                                dict(
                                    type="ineq",
                                    fun=lambda v: sample_slacks(
                                        self.point(hardware, frequency, v),
                                        floor,
                                        chi_floor,
                                    ),
                                )
                            ],
                            options=dict(maxiter=25, ftol=1e-8, eps=1e-6),
                        )
                        proposals = [fit.x]
                    for bare in proposals:
                        point = self.point(hardware, frequency, bare)
                        if min(sample_slacks(point, floor, chi_floor)) < -1e-7:
                            continue
                        audited = audit_point(point, representative=False)
                        failed = checked_sample(audited, floor, chi_floor)
                        attempts.append(
                            dict(bare_f01_ghz=list(map(float, bare)), failures=failed)
                        )
                        if not failed:
                            return dict(
                                frequency_ghz=float(frequency),
                                point=audited,
                                failures=[],
                                attempts=attempts,
                            )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    attempts.append(dict(error=str(error)))
        return dict(
            frequency_ghz=float(frequency),
            point=None,
            failures=["no_audited_EJ_recalibration_found"],
            attempts=attempts,
        )

    def audit(self, x, floor):
        nominal = self.nominal_audit(x[:7])
        # Same 1 kHz allowance as the readout range-design workflow. It is a
        # numerical validation allowance, not a relaxation of its 1 MHz floor.
        chi_floor = max(study.CONDITIONS.abs_mean_chi_min_mhz, float(x[-1]) - 0.001)
        failed = checked_sample(nominal["point"], floor, chi_floor)
        if np.max(abs(np.asarray(self.nominal_equalities(x[:7]))), initial=0.0) > 1e-6:
            failed.append("nominal_calibration_equalities")
        # Include endpoints, nominal reference, half steps and optimized-sample midpoints.
        grid = sorted(
            set(
                frequency_grid(self.low, self.high, self.step / 2, float(x[4])).tolist()
                + [
                    round(float((a + b) / 2), 10)
                    for a, b in zip(
                        self.frequencies[:-1], self.frequencies[1:], strict=True
                    )
                ]
            )
        )
        witnesses = []
        recourse = np.asarray(x[7:-1]).reshape(-1, 2)
        for frequency in grid:
            interpolated = [
                np.interp(frequency, self.frequencies, recourse[:, i]) for i in (0, 1)
            ]
            row = self.validate_frequency(
                x[:7], frequency, [interpolated, x[:2]], floor, chi_floor
            )
            witnesses.append(row)
            if row["failures"]:
                failed.append(f"range_validation:{frequency:.10f}")
                self.failed_frequencies.add(float(frequency))
        points = [
            nominal["point"],
            *[r["point"] for r in witnesses if r["point"] is not None],
        ]
        aggregate = dict(
            hybridization=min(p["hybridization"] for p in points),
            worst_mean_chi_mhz=min(map(study.absolute_mean_chi_mhz, points)),
            failures=failed,
        )
        return dict(
            point=aggregate,
            nominal=nominal["point"],
            physical=nominal["physical"],
            range_validation=dict(
                passed=not failed,
                range_ghz=[self.low, self.high],
                nominal_reference_ghz=float(x[4]),
                samples=witnesses,
                optimized_frequencies_ghz=self.frequencies.tolist(),
                validation_step_ghz=self.step / 2,
                chi_validation_floor_mhz=chi_floor,
                chi_validation_allowance_mhz=0.001,
                worst_mean_chi_mhz=aggregate["worst_mean_chi_mhz"],
                worst_H=aggregate["hybridization"],
                H_strict_floor=floor,
                nominal_loaded_ec_ghz=list(self.case["loaded_ec_ghz"]),
                capacitor_policy="one fixed nominal physical pad layout; no scenario EC compensation",
                finite_grid_only=True,
            ),
        )


def optimize_pad_range(*args, **kwargs):
    """Optimize one rail per process with one BLAS thread, preserving the caller's policy."""
    with threadpool_limits(limits=1):
        return _optimize_pad_range(*args, **kwargs)


def _optimize_pad_range(
    case,
    seed,
    starts,
    bounds,
    nominal_evaluate,
    equalities,
    inequalities,
    nominal_audit,
    builder,
    evaluator,
    basis,
):
    begun = time.monotonic()
    options = case.get("readout_range_options", {})
    maxiter = int(options.get("maxiter", 60))
    max_audits = int(options.get("max_audits", 2))
    refinement_rounds = int(options.get("refinement_rounds", 2))
    if maxiter < 1 or max_audits < 1 or refinement_rounds < 1:
        raise ValueError(
            "Positive readout maxiter, max_audits and refinement_rounds required"
        )
    from .readout_optimization import H_FLOORS

    all_tiers = []
    extra = set()
    total_seconds = 0.0
    total_hardware = total_evaluations = 0
    for floor in H_FLOORS:
        for refinement in range(refinement_rounds):
            model = PadRangeObjective(
                case,
                seed,
                bounds,
                nominal_evaluate,
                equalities,
                inequalities,
                nominal_audit,
                builder,
                evaluator,
                basis,
                extra_frequencies=extra,
            )
            result = maximize_chi(
                model.start(seed),
                [model.start(x) for x in starts],
                model.bounds,
                model.evaluate,
                model.equalities,
                model.inequalities,
                model.audit,
                lambda aggregate: aggregate["failures"],
                lambda aggregate: aggregate["worst_mean_chi_mhz"],
                maxiter=maxiter,
                max_audits=max_audits,
                objective=lambda x: -x[-1],
                objective_name="maximize_worst_absolute_signed_mean_chi_over_prescribed_range",
                scope="prescribed_range_validated",
                h_floors=(floor,),
                audit_with_floor=True,
            )
            total_seconds += result["seconds"]
            total_hardware += model.hardware_builds
            total_evaluations += model.scenario_evaluations
            all_tiers.extend(
                dict(tier, refinement_round=refinement) for tier in result["tiers"]
            )
            if result["selected"] is not None:
                break
            new = model.failed_frequencies - set(model.frequencies)
            if not new:
                break
            audited = result["tiers"][0]["audits"]
            if audited:
                # Restart hardware refinement from the strongest screened layout
                # rather than discarding its progress after an interior failure.
                seed = np.asarray(audited[0]["x"][:7])
            # A failed interior sample becomes a constrained optimization scenario
            # before relaxing H. Never shorten the prescribed interval.
            extra.update(new)
        if result["selected"] is not None:
            break
    result["tiers"] = all_tiers
    result["optimizer_seconds"] = total_seconds
    result["seconds"] = time.monotonic() - begun
    result.update(
        mode="prescribed_resonator_range",
        range_ghz=[model.low, model.high],
        scenario_basis=asdict(basis),
        frequency_step_ghz=model.step,
        optimizer_budget=dict(
            maxiter=maxiter, max_audits=max_audits, refinement_rounds=refinement_rounds
        ),
        hardware_builds=total_hardware,
        scenario_evaluations=total_evaluations,
    )
    if result["selected"] is not None:
        result["coordinates"] = result["coordinates"][:7]
        result["range_validation"] = result["selected"]["range_validation"]
        if result["warning"]:
            result["warning"] += " The full requested resonator interval was retained."
    return result
