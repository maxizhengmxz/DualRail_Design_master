"""Cavity tolerance anchored to one selected EC-tuple operating witness.

Unlike ``resonator_robust_design --mode tolerance``, this workflow does not
accept an unrelated existential EJ witness at each cavity frequency.  The
nominal cavity must use the supplied bare transmon frequencies exactly, and
all other samples are reached by outward continuation within individual bare
frequency budgets.  Fixed-witness audit and optional capacitor redesign are
reported separately.  No calculation starts without ``--run``.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from . import calculation as study
from .direct_ec_feasibility import capacitances
from .ec_resonator_tolerance import at_frequency, set_policy
from .storage import atomic_json, utc
from .validation import INTERMEDIATE, audit_point


def synthesize_device(ec_mhz, c12_ff, readout_ff, reference_ghz):
    """Preserve the requested loaded EC tuple at the supplied cavity reference."""
    old = study.BARE_RESONATOR_GHZ
    try:
        study.BARE_RESONATOR_GHZ = float(reference_ghz)
        c1, c2 = capacitances(*ec_mhz, float(c12_ff), float(readout_ff))
    finally:
        study.BARE_RESONATOR_GHZ = old
    omega = 2 * np.pi * float(reference_ghz) * 1e9
    return study.Device(
        C1_shunt_fF=c1,
        C2_shunt_fF=c2,
        C12_fF=float(c12_ff),
        C1r_fF=float(readout_ff),
        C2r_fF=float(readout_ff),
        Cr_fF=1e15 / (omega * 50.0),
        Lr_nH=1e9 * 50.0 / omega,
    )


def evaluate(device, cavity_ghz, bare_f01_ghz, basis=INTERMEDIATE):
    model = study.ReadoutModel(at_frequency(asdict(device), float(cavity_ghz)), basis)
    ej = [
        study.inverse_frequency(float(f), float(ec), basis)
        for f, ec in zip(bare_f01_ghz, model.ec_ghz[:2], strict=True)
    ]
    return study.annotate(model, *ej)


def frequency_bounds(targets, halfwidth_ghz):
    targets = np.asarray(targets, float)
    width = float(halfwidth_ghz)
    if targets.shape != (2,) or not np.isfinite(targets).all() or width <= 0:
        raise ValueError("Two finite targets and a positive bare-frequency halfwidth required")
    return targets - width, targets + width


def screen_failures(point):
    return list(study.failures(point))


def solve_frequency(device, cavity, targets, halfwidth, warm, exact=False, max_nfev=70):
    """Find the next point on one bounded continuation branch."""
    low, high = frequency_bounds(targets, halfwidth)
    cache = {}

    def point(x):
        key = tuple(map(float, x))
        if key not in cache:
            cache[key] = evaluate(device, cavity, x)
        return cache[key]

    def residual(x):
        p = point(x)
        return np.r_[
            study.violations(p, margin=0.015),
            max(0.0, study.CONDITIONS.abs_mean_chi_min_mhz + 0.00005
                - study.absolute_mean_chi_mhz(p)),
        ]

    if exact:
        proposals = [np.asarray(targets, float)]
        fits = []
    else:
        targets = np.asarray(targets, float)
        seeds = [np.asarray(warm, float), targets]
        for fraction in (-1.0, -0.5, 0.5, 1.0):
            seeds.append(targets + fraction * halfwidth)
        for fraction in (-0.5, -0.25, 0.25, 0.5):
            seeds.append(targets + fraction * halfwidth * np.array([1.0, -1.0]))
        # Wider budgets must retain the useful starts from narrower budgets;
        # otherwise numerical discovery can perversely worsen when a physical
        # bound is relaxed.
        for shift in (-0.1, -0.05, -0.025, 0.025, 0.05, 0.1):
            if abs(shift) <= halfwidth + 1e-12:
                seeds.append(targets + shift)
                seeds.append(targets + shift * np.array([1.0, -1.0]))
        proposals, fits = [], []
        with threadpool_limits(limits=1):
            for seed in seeds:
                seed = np.clip(seed, low, high)
                proposals.append(seed)
                # Most continuation points are already feasible at a bounded
                # common/differential seed. Preserve that witness and avoid
                # spending a local solve on later, unrelated starts.
                if np.linalg.norm(residual(seed)) <= 1e-9:
                    break
                fit = least_squares(
                    residual,
                    seed,
                    bounds=(low, high),
                    x_scale=[max(halfwidth, 0.01)] * 2,
                    diff_step=2e-5,
                    max_nfev=max_nfev,
                    ftol=1e-9,
                    xtol=1e-10,
                    gtol=1e-9,
                )
                fits.append(dict(success=bool(fit.success), cost=float(fit.cost),
                                 nfev=int(fit.nfev), x=fit.x.tolist()))
                proposals.append(fit.x)
    unique = []
    for x in proposals:
        if not any(np.linalg.norm(x - prior) < 1e-9 for prior in unique):
            unique.append(x)
    attempts = []
    for x in sorted(unique, key=lambda q: np.linalg.norm(residual(q))):
        p = point(x)
        failures = screen_failures(p)
        if failures:
            attempts.append(dict(bare_f01_ghz=x.tolist(), failures=failures))
            continue
        audited = audit_point(p, representative=False)
        failures = study.failures(audited, require_sign_check=True)
        attempts.append(dict(bare_f01_ghz=x.tolist(), failures=failures))
        if not failures:
            return dict(status="validated_pass", frequency_ghz=float(cavity),
                        bare_f01_ghz=x.tolist(), point=audited, attempts=attempts,
                        fits=fits, evaluations=len(cache))
    closest = min(cache.values(), key=lambda p: np.linalg.norm(study.violations(p)))
    return dict(status="no_validated_solution_found", frequency_ghz=float(cavity),
                bare_f01_ghz=None, point=None, failures=screen_failures(closest),
                closest=closest, attempts=attempts, fits=fits, evaluations=len(cache))


def continuation(device, reference, targets, halfwidth, low, high, step, max_nfev=70):
    """Validate center first, then walk outward; never bridge a failed sample."""
    if not low <= reference <= high or step <= 0:
        raise ValueError("Cavity interval must contain its reference and use positive step")
    center = solve_frequency(device, reference, targets, halfwidth, targets,
                             exact=True, max_nfev=max_nfev)
    rows = {round(float(reference), 10): center}
    if center["status"] != "validated_pass":
        return dict(status="nominal_anchor_failed", passed=False, samples=[center])
    for direction in (-1, 1):
        warm = np.asarray(targets, float)
        f = float(reference)
        boundary = float(low if direction < 0 else high)
        while True:
            nxt = f + direction * step
            if (direction < 0 and nxt < boundary) or (direction > 0 and nxt > boundary):
                nxt = boundary
            if abs(f - boundary) < 1e-10:
                break
            key = round(float(nxt), 10)
            row = solve_frequency(device, key, targets, halfwidth, warm,
                                  max_nfev=max_nfev)
            rows[key] = row
            if row["status"] != "validated_pass":
                break
            warm = np.asarray(row["bare_f01_ghz"], float)
            f = key
            if abs(f - (low if direction < 0 else high)) < 1e-10:
                break
    ordered = [rows[k] for k in sorted(rows)]
    passed = all(row["status"] == "validated_pass" for row in ordered)
    points = [row["point"] for row in ordered if row["point"]]
    jumps = []
    passing_rows = [row for row in ordered if row["point"]]
    for a, b in zip(passing_rows[:-1], passing_rows[1:], strict=True):
        jumps.append(max(abs(np.asarray(b["bare_f01_ghz"]) - a["bare_f01_ghz"])))
    return dict(
        status="validated_connected_range" if passed else "connected_range_failed",
        passed=passed,
        range_ghz=[float(low), float(high)],
        reference_ghz=float(reference),
        samples=ordered,
        worst_H=min((p["hybridization"] for p in points), default=None),
        worst_mean_chi_mhz=min((study.absolute_mean_chi_mhz(p) for p in points), default=None),
        maximum_adjacent_bare_frequency_jump_mhz=(max(jumps) * 1000 if jumps else 0.0),
        finite_grid_only=True,
    )


def redesign(case, halfwidth, low, high, step, max_nfev=160):
    """Feasibility redesign of C12/Cg, certified by connected continuation."""
    ec = case["ec_mhz"]
    reference = case["nominal_cavity_ghz"]
    targets = np.asarray(case["nominal_bare_f01_ghz"], float)
    frequencies = np.array(sorted(set(
        [reference, low, high]
        + list(np.arange(reference, low - 1e-10, -step))
        + list(np.arange(reference, high + 1e-10, step))
    )))
    frequencies = frequencies[(frequencies >= low - 1e-10) & (frequencies <= high + 1e-10)]
    nonref = [float(f) for f in frequencies if abs(f - reference) > 1e-10]
    low_f, high_f = frequency_bounds(targets, halfwidth)
    cap_bounds = np.array([study.CONDITIONS.c12_bounds_ff,
                           study.CONDITIONS.readout_cap_bounds_ff], float)
    cache = {}

    def unpack(x):
        values = {round(reference, 10): targets}
        for j, f in enumerate(nonref):
            values[round(f, 10)] = np.asarray(x[2 + 2*j:4 + 2*j])
        return values

    def points(x):
        key = tuple(map(float, x))
        if key not in cache:
            device = synthesize_device(ec, x[0], x[1], reference)
            values = unpack(x)
            cache[key] = (device, [evaluate(device, f, values[round(float(f), 10)])
                                   for f in frequencies])
        return cache[key]

    def residual(x):
        _, pts = points(x)
        return np.concatenate([
            np.r_[study.violations(p, margin=0.015),
                  max(0.0, study.CONDITIONS.abs_mean_chi_min_mhz + 0.00005
                      - study.absolute_mean_chi_mhz(p))]
            for p in pts
        ])

    starts = [case["fixed_capacitors_ff"], *case.get("capacitor_starts_ff", [])]
    attempts = []
    best = None
    for caps in starts:
        hardware = synthesize_device(ec, *caps, reference)
        values = []
        for f in nonref:
            delta = 0.6 * (f - reference)
            values.extend(np.clip(targets + delta, low_f, high_f))
        x0 = np.r_[caps, values]
        lower = np.r_[cap_bounds[:, 0], np.tile(low_f, len(nonref))]
        upper = np.r_[cap_bounds[:, 1], np.tile(high_f, len(nonref))]
        with threadpool_limits(limits=1):
            fit = least_squares(residual, np.clip(x0, lower, upper), bounds=(lower, upper),
                                x_scale=np.r_[[5.0, 5.0], np.full(2*len(nonref), max(halfwidth, .01))],
                                diff_step=2e-5, max_nfev=max_nfev,
                                ftol=1e-9, xtol=1e-10, gtol=1e-9)
        score = float(np.linalg.norm(residual(fit.x)))
        attempts.append(dict(caps_ff=fit.x[:2].tolist(), cost=float(fit.cost),
                             residual_norm=score, nfev=int(fit.nfev), success=bool(fit.success)))
        candidate = continuation(synthesize_device(ec, *fit.x[:2], reference), reference,
                                 targets, halfwidth, low, high, step)
        if candidate["passed"]:
            best = dict(capacitors_ff=fit.x[:2].tolist(), validation=candidate)
            break
    return dict(status="validated_redesign" if best else "no_validated_redesign_found",
                selected=best, attempts=attempts)


def symmetric_range_search(case, halfwidth, fixed_device, redesign_hardware, step):
    """Bracket connected symmetric coverage without crossing a failed width."""
    reference = float(case["nominal_cavity_ghz"])
    ceiling = float(case.get("maximum_cavity_halfwidth_ghz", 0.25))
    resolution = float(case.get("cavity_boundary_resolution_ghz", 0.0125))
    if ceiling <= 0 or resolution <= 0:
        raise ValueError("Positive cavity ceiling and boundary resolution required")
    good = 0.0
    selected = None
    trials = []
    warm_caps = []
    width = min(0.05, ceiling)
    bad = None
    while True:
        low, high = reference - width, reference + width
        if redesign_hardware:
            local = dict(case)
            local["capacitor_starts_ff"] = [*warm_caps, *case.get("capacitor_starts_ff", [])]
            result = redesign(local, halfwidth, low, high, step)
            passed = result["selected"] is not None
            if passed:
                warm_caps = [result["selected"]["capacitors_ff"]]
                selected = result["selected"]
        else:
            result = continuation(fixed_device, reference, case["nominal_bare_f01_ghz"],
                                  halfwidth, low, high, step)
            passed = result["passed"]
            if passed:
                selected = dict(capacitors_ff=case["fixed_capacitors_ff"], validation=result)
        trials.append(dict(halfwidth_ghz=width, passed=passed, result=result))
        if not passed:
            bad = width
            break
        good = width
        if width >= ceiling - 1e-10:
            break
        width = min(width + 0.05, ceiling)
    while bad is not None and bad - good > resolution + 1e-10:
        width = (good + bad) / 2
        low, high = reference - width, reference + width
        if redesign_hardware:
            local = dict(case)
            local["capacitor_starts_ff"] = [*warm_caps, *case.get("capacitor_starts_ff", [])]
            result = redesign(local, halfwidth, low, high, step)
            passed = result["selected"] is not None
            if passed:
                warm_caps = [result["selected"]["capacitors_ff"]]
                selected = result["selected"]
        else:
            result = continuation(fixed_device, reference, case["nominal_bare_f01_ghz"],
                                  halfwidth, low, high, step)
            passed = result["passed"]
            if passed:
                selected = dict(capacitors_ff=case["fixed_capacitors_ff"], validation=result)
        trials.append(dict(halfwidth_ghz=width, passed=passed, result=result))
        if passed:
            good = width
        else:
            bad = width
    return dict(
        status="validated_sampled_range" if selected else "nominal_only",
        validated_halfwidth_ghz=good,
        range_ghz=[reference-good, reference+good],
        failed_halfwidth_ghz=bad,
        censored_by_ceiling=bad is None and good >= ceiling-1e-10,
        boundary_resolution_ghz=resolution,
        selected=selected,
        trials=trials,
    )


def run_case(case, output):
    set_policy(case["conditions"])
    reference = float(case["nominal_cavity_ghz"])
    targets = case["nominal_bare_f01_ghz"]
    interval = case["requested_cavity_range_ghz"]
    step = float(case.get("frequency_step_ghz", 0.05))
    results = []
    prior_redesign_maximum = None
    fixed = synthesize_device(case["ec_mhz"], *case["fixed_capacitors_ff"], reference)
    for budget_mhz in case["bare_frequency_halfwidths_mhz"]:
        width = float(budget_mhz) / 1000
        audit = continuation(fixed, reference, targets, width, *interval, step)
        redesigned = redesign(case, width, *interval, step)
        fixed_maximum = symmetric_range_search(case, width, fixed, False, step)
        if prior_redesign_maximum and prior_redesign_maximum["censored_by_ceiling"]:
            redesign_maximum = deepcopy(prior_redesign_maximum)
            redesign_maximum["inherited_from_smaller_frequency_budget"] = True
        else:
            redesign_maximum = symmetric_range_search(case, width, fixed, True, step)
        if (prior_redesign_maximum
                and prior_redesign_maximum["validated_halfwidth_ghz"]
                > redesign_maximum["validated_halfwidth_ghz"]):
            retained = deepcopy(prior_redesign_maximum)
            retained["inherited_as_monotonic_lower_bound"] = True
            retained["wider_budget_search_diagnostic"] = redesign_maximum
            redesign_maximum = retained
        if redesign_maximum.get("selected") is not None:
            prior_redesign_maximum = redesign_maximum
        row = dict(bare_frequency_halfwidth_mhz=float(budget_mhz),
                   fixed_witness=audit, redesigned=redesigned,
                   fixed_maximum_symmetric_range=fixed_maximum,
                   redesigned_maximum_symmetric_range=redesign_maximum)
        results.append(row)
        atomic_json(Path(output) / f"budget_{budget_mhz:g}_mhz.json", row)
    result = dict(case=case, results=results, completed_utc=utc())
    atomic_json(Path(output) / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--rail", choices=["A", "B"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not a.run:
        print("Use --run explicitly. No calculation started.")
        return
    request = json.loads(a.config.read_text())
    case = request["rails"][a.rail]
    a.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__)
    settings = dict(rail=a.rail, case=case, source_sha256=sha256(source.read_bytes()).hexdigest())
    path = a.output / "settings.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("Resume identity changed; use a fresh output directory")
    atomic_json(path, settings)
    started = time.monotonic()
    result = run_case(case, a.output)
    atomic_json(a.output / "completion.json", dict(status="complete",
                elapsed_seconds=time.monotonic()-started, completed_utc=utc()))
    print(json.dumps({"rail": a.rail, "results": [
        {"budget_mhz": r["bare_frequency_halfwidth_mhz"],
         "fixed_passed": r["fixed_witness"]["passed"],
         "redesign_status": r["redesigned"]["status"]} for r in result["results"]]}))


if __name__ == "__main__":
    main()
