# ruff: noqa: E402
# BLAS thread limits must be set before importing numerical libraries.
"""Optimize fixed capacitor designs against resonator-frequency variation.

Two explicit modes (never run on import):
* prescribed: maximize the WORST absolute signed-mean chi across a supplied
  resonator range, at selectable minimum H targets (a chi/H tradeoff curve).
* tolerance: maximize a symmetric range around 6 GHz at the acceptance H floor;
  stop after validating that range; chi and H remain acceptance constraints.

Loaded EC targets apply at 6 GHz / 50 Ohm only. A design shares C1, C2, C12,
C1r=C2r across all frequencies; only Cr/Lr (50 Ohm) and both EJs change.
Finite sampled local optimization is not a global or continuum certificate.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
import time
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from . import calculation as study
from .direct_ec_feasibility import capacitances
from .ec_design_improvement import slacks as nominal_slacks
from .ec_resonator_tolerance import at_frequency, set_policy
from .storage import atomic_json, utc
from .validation import INTERMEDIATE, audit_point

REFERENCE_GHZ = 6.0


def frequency_grid(low, high, step, reference_ghz=REFERENCE_GHZ):
    """Include both endpoints and the reference; no overshoot or missing center."""
    if not (
        all(np.isfinite([low, high, step, reference_ghz]))
        and 0 < low <= reference_ghz <= high
        and step > 0
    ):
        raise ValueError(
            f"Range must contain {reference_ghz:g} GHz, with positive frequency/step"
        )
    points = list(np.arange(low, high + 1e-10, step)) + [low, high, reference_ghz]
    return np.array(
        sorted(
            set(round(float(x), 10) for x in points if low - 1e-9 <= x <= high + 1e-9)
        )
    )


def design_device(ec_mhz, c12, cg):
    """Synthesize nominal ECs once; never re-fit C1/C2 at swept frequencies."""
    c1, c2 = capacitances(*ec_mhz, float(c12), float(cg))
    return study.Device(
        C1_shunt_fF=c1, C2_shunt_fF=c2, C12_fF=c12, C1r_fF=cg, C2r_fF=cg
    )


def point_slacks(p, hfloor):
    """Original acceptance plus a user H target; EC varies with resonator loading."""
    c = study.CONDITIONS
    return np.r_[
        nominal_slacks(p, hfloor),
        *[(p[f"loaded_ec{i}_mhz"] - c.loaded_ec_mhz[0]) / 100 for i in (1, 2)],
        *[(c.loaded_ec_mhz[1] - p[f"loaded_ec{i}_mhz"]) / 100 for i in (1, 2)],
    ]


def point_from_seed(device, frequency, nominal, shift=0.0):
    model = study.ReadoutModel(at_frequency(asdict(device), frequency), INTERMEDIATE)
    delta = shift + 0.6 * (frequency - REFERENCE_GHZ)
    ej = []
    for i, ec in enumerate(model.ec_ghz[:2], 1):
        ej.append(
            np.clip(
                study.inverse_frequency(
                    nominal[f"bare_f{i}_ghz"] + delta, float(ec), INTERMEDIATE
                ),
                0.1,
                study.CONDITIONS.ej_upper_ghz - 0.001,
            )
        )
    return ej


def optimize_scenarios(
    ec_mhz,
    nominal,
    frequencies,
    hfloor,
    folder,
    *,
    starts=3,
    maxiter=70,
    warm=None,
    feasibility_only=False,
):
    """Shared hardware with independent EJ recourse at every frequency.

    An epigraph variable t enforces chi(f)>=t. Maximizing t optimizes the
    weakest readout in the range, not a peak that hides weak frequencies.
    """
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    completed = folder / "optimization.json"
    if completed.exists():
        return json.loads(completed.read_text())
    freqs = np.array(frequencies, float)
    n = len(freqs)
    models = {}
    cache = {}
    records = []

    def evaluate(x):
        capkey = tuple(map(float, x[:2]))
        points = []
        if capkey not in models:
            dev = design_device(ec_mhz, *capkey)
            models[capkey] = [
                study.ReadoutModel(at_frequency(asdict(dev), float(f)), INTERMEDIATE)
                for f in freqs
            ]
        for j, m in enumerate(models[capkey]):
            ej = tuple(map(float, x[2 + 2 * j : 4 + 2 * j]))
            key = (capkey, j, ej)
            if key not in cache:
                cache[key] = study.annotate(m, *ej)
            points.append(cache[key])
        return points

    def constraints(x):
        result = []
        for p in evaluate(x):
            v = point_slacks(p, hfloor)
            guard = np.full_like(v, 2e-5)
            # EC's inclusive lower bound must remain inclusive. A numerical
            # interior guard here incorrectly rejects every nominal EC=150 design.
            guard[-4:-2] = 0.0
            result.append(
                np.r_[v - guard, study.absolute_mean_chi_mhz(p) - x[-1] - 2e-5]
            )
        return np.concatenate(result)

    c = study.CONDITIONS
    bounds = (
        [c.c12_bounds_ff, c.readout_cap_bounds_ff]
        + [(0.1, c.ej_upper_ghz - 1e-4)] * (2 * n)
        + [
            (
                c.abs_mean_chi_min_mhz,
                c.abs_mean_chi_min_mhz if feasibility_only else 100.0,
            )
        ]
    )
    seeds = []
    if warm:
        d = warm["points"][0]["device"]
        seed = [d["C12_fF"], d["C1r_fF"]]
        for f in freqs:
            prior = min(
                zip(warm["frequencies_ghz"], warm["points"], strict=True),
                key=lambda z: abs(z[0] - f),
            )[1]
            seed += [prior["ej1_ghz"], prior["ej2_ghz"]]
        seeds.append(seed + [max(1.0, warm["worst_mean_chi_mhz"] * 0.95)])
    for shift, scale in [(0.0, 1.0), (-0.3, 1.1), (0.2, 1.15), (-0.6, 1.2)]:
        d = nominal["device"]
        c12 = np.clip(d["C12_fF"] * scale, *c.c12_bounds_ff)
        cg = np.clip(d["C1r_fF"] * scale, *c.readout_cap_bounds_ff)
        dev = design_device(ec_mhz, c12, cg)
        x = [c12, cg]
        for f in freqs:
            x += point_from_seed(dev, float(f), nominal, shift)
        seeds.append(x + [1.0])
    with threadpool_limits(limits=1):
        for i, x in enumerate(seeds[:starts]):
            if feasibility_only:
                x[-1] = c.abs_mean_chi_min_mhz
            fit = minimize(
                (lambda v: 0.0) if feasibility_only else (lambda v: -v[-1]),
                x,
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "ineq", "fun": constraints}],
                options={"maxiter": maxiter, "ftol": 1e-7, "eps": 1e-5},
            )
            pts = evaluate(fit.x)
            record = dict(
                start=i,
                x=fit.x.tolist(),
                success=bool(fit.success),
                message=str(fit.message),
                minimum_slack=float(min(constraints(fit.x))),
                points=pts,
                frequencies_ghz=freqs.tolist(),
                worst_mean_chi_mhz=min(map(study.absolute_mean_chi_mhz, pts)),
                worst_H=min(p["hybridization"] for p in pts),
            )
            records.append(record)
            atomic_json(folder / f"start_{i}.json", record)
            print(
                json.dumps(
                    dict(
                        stage=str(folder),
                        start=i,
                        worst_chi=record["worst_mean_chi_mhz"],
                        worst_H=record["worst_H"],
                        slack=record["minimum_slack"],
                    )
                ),
                flush=True,
            )
            if feasibility_only and record["minimum_slack"] >= -1e-7:
                break
    valid = [r for r in records if r["minimum_slack"] >= -1e-7]
    best = max(valid, key=lambda r: r["worst_mean_chi_mhz"]) if valid else None
    result = dict(
        best=best, trials=records, frequencies_ghz=freqs.tolist(), H_floor=hfloor
    )
    atomic_json(completed, result)
    return result


def validate_range(
    solution, ec_mhz, hfloor, low, high, step, folder, *, optimize_chi=True
):
    """Audit optimized points and independently recalibrate half-step samples.

    Interior failures are returned to the scenario optimizer. Do not claim
    range coverage from passing endpoints alone. This remains a finite grid.
    """
    from .ec_resonator_tolerance import solve_frequency

    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    base = np.array(solution["frequencies_ghz"], float)
    freqs = sorted(
        set(
            frequency_grid(low, high, step / 2).tolist()
            + [
                round(float((a + b) / 2), 10)
                for a, b in zip(base[:-1], base[1:], strict=True)
            ]
        )
    )
    known = dict(zip(solution["frequencies_ghz"], solution["points"], strict=True))
    # Validate the worst-case objective, not merely the original 1 MHz floor.
    chi_floor = validation_chi_floor(solution, optimize_chi)
    center = min(
        solution["points"],
        key=lambda p: abs(
            1
            / (
                2
                * np.pi
                * np.sqrt(p["device"]["Cr_fF"] * 1e-15 * p["device"]["Lr_nH"] * 1e-9)
            )
            / 1e9
            - REFERENCE_GHZ
        ),
    )
    attempts = []
    bad = []
    passed = []
    # The chosen capacitor design is unchanged. Nominal reference stores its
    # 6 GHz device even if the nearest optimization seed came from another f.
    center = dict(center, device=asdict(at_frequency(center["device"], REFERENCE_GHZ)))
    for f in sorted(map(float, freqs), key=lambda x: abs(x - REFERENCE_GHZ)):
        name = folder / f"frequency_{f:.10f}.json"
        if name.exists():
            r = json.loads(name.read_text())
        else:
            original = known.get(round(f, 10))
            q = audit_point(original, representative=False) if original else None
            failures = (
                study.failures(q, require_sign_check=True)
                if q
                else ["needs_recalibration"]
            )
            if q and q["hybridization"] < hfloor:
                failures.append("H_target")
            if q and study.absolute_mean_chi_mhz(q) < chi_floor:
                failures.append("worst_case_chi_target")
            if failures:
                # H target is enforced during rescue too, not only checked afterward.
                original_c = study.CONDITIONS
                original_b = study.BASELINE.copy()
                try:
                    study.CONDITIONS = replace(
                        original_c,
                        hybridization_min=hfloor,
                        abs_mean_chi_min_mhz=chi_floor,
                    )
                    study.BASELINE = dict(original_b, H=hfloor)
                    warm = (
                        min(passed, key=lambda p: abs(p["frequency_ghz"] - f))["point"]
                        if passed
                        else original or center
                    )
                    rescued = solve_frequency(
                        center,
                        f,
                        "flexible",
                        warm,
                        dict(starts=3, max_nfev=35),
                        [original] if original else (),
                    )
                    q = rescued["point"]
                    failures = [] if q else rescued["failures"]
                finally:
                    study.CONDITIONS, study.BASELINE = original_c, original_b
            r = dict(frequency_ghz=f, point=q, failures=failures)
            atomic_json(name, r)
        attempts.append(r)
        if r["point"] and not r["failures"]:
            passed.append(r)
        else:
            bad.append(f)
    return dict(
        passed=not bad,
        failed_frequencies_ghz=bad,
        samples=attempts,
        validation_step_ghz=step / 2,
        chi_validation_floor_mhz=chi_floor,
        chi_validation_allowance_mhz=0.001 if optimize_chi else 0.0,
        strength_optimized=optimize_chi,
    )


def run_range(
    ec,
    nominal,
    low,
    high,
    hfloor,
    folder,
    starts,
    maxiter,
    step,
    rounds=3,
    *,
    warm_seed=None,
    feasibility_only=False,
):
    """Add failed interior samples, then refit shared hardware and all biases."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    if (folder / "result.json").exists():
        return json.loads((folder / "result.json").read_text())
    freqs = frequency_grid(low, high, step)
    warm = warm_seed
    result = None
    for k in range(rounds):
        fit = optimize_scenarios(
            ec,
            nominal,
            freqs,
            hfloor,
            folder / f"round_{k}",
            starts=starts,
            maxiter=maxiter,
            warm=warm,
            feasibility_only=feasibility_only,
        )
        if fit["best"] is None:
            result = dict(status="no_validated_design_found", optimization=fit)
            break
        validation = validate_range(
            fit["best"],
            ec,
            hfloor,
            low,
            high,
            step,
            folder / f"round_{k}" / "validation",
            optimize_chi=not feasibility_only,
        )
        result = dict(
            status="validated_sampled_range"
            if validation["passed"]
            else "interior_validation_failed",
            optimization=fit,
            validation=validation,
        )
        if validation["passed"]:
            pts = [r["point"] for r in validation["samples"]]
            result.update(
                worst_mean_chi_mhz=min(map(study.absolute_mean_chi_mhz, pts)),
                worst_H=min(p["hybridization"] for p in pts),
                range_ghz=[low, high],
                H_target=hfloor,
            )
            break
        freqs = np.array(
            sorted(set(freqs.tolist() + validation["failed_frequencies_ghz"]))
        )
        warm = fit["best"]
    atomic_json(folder / "result.json", result)
    return result


def quick_tolerance(ec, nominal, folder, starts, maxiter, step, resolution, ceiling):
    """Cheap range bracketing; spend dense validation only on final candidates.

    Screening uses endpoints plus 6 GHz, reuses previous hardware/biases and
    stops at first feasible start. Only the final densely audited interval is
    reported as validated. Unvalidated provisional boundaries are never output
    as physical tolerance. Every failure still gets the configured starts.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    h = study.CONDITIONS.hybridization_min
    warm = None
    good = 0.0
    bad = None

    def attempt(w):
        nonlocal warm
        fit = optimize_scenarios(
            ec,
            nominal,
            frequency_grid(6 - w, 6 + w, max(step, 2 * w)),
            h,
            folder / f"screen_{w:.8f}",
            starts=starts,
            maxiter=maxiter,
            warm=warm,
            feasibility_only=True,
        )
        if fit["best"]:
            warm = fit["best"]
        return fit["best"]

    best = attempt(0.0)
    if best is None:
        return dict(status="nominal_search_failed", range_ghz=None)
    w = min(0.05, ceiling)
    while w > 0:
        result = attempt(w)
        if result is None:
            bad = w
            break
        good = w
        best = result
        if w >= ceiling:
            break
        w = min(2 * w, ceiling)
    while bad is not None and bad - good > resolution + 1e-10:
        w = (good + bad) / 2
        result = attempt(w)
        if result is None:
            bad = w
        else:
            good = w
            best = result
    # A failed dense check triggers full-grid recourse/refinement. If that
    # fails in budget, shrink conservatively rather than certify endpoints.
    while good >= 0:
        r = run_range(
            ec,
            nominal,
            6 - good,
            6 + good,
            h,
            folder / f"validated_{good:.8f}",
            starts,
            maxiter,
            step,
            warm_seed=best,
            feasibility_only=True,
        )
        if r["status"] == "validated_sampled_range":
            return dict(
                status="validated_sampled_range",
                range_ghz=[6 - good, 6 + good],
                boundary=dict(
                    validated_halfwidth_ghz=good,
                    failed_search_halfwidth_ghz=bad,
                    censored_by_search_ceiling=bad is None and good >= ceiling,
                ),
                validated_result=r,
            )
        bad = good
        if good == 0:
            break
        good = max(0, good - resolution)
    return dict(status="dense_validation_failed", range_ghz=None)


def selected_h_targets(mode, supplied, acceptance_floor):
    """Exploration keeps the original H requirement; prescribed ranges expose tradeoffs."""
    if mode == "tolerance":
        if supplied is not None and any(
            abs(h - acceptance_floor) > 1e-10 for h in supplied
        ):
            raise ValueError(
                "Exploration uses the acceptance H floor only. Use prescribed mode for chi/H tradeoffs."
            )
        return [acceptance_floor]
    return sorted(
        set(
            supplied
            if supplied is not None
            else [h for h in [acceptance_floor, 0.97, 0.98] if h >= acceptance_floor]
        )
    )


def validation_chi_floor(solution, optimize_chi):
    """Do not impose an incidental seed chi as a hidden exploration target."""
    base = study.CONDITIONS.abs_mean_chi_min_mhz
    return max(base, solution["worst_mean_chi_mhz"] - 0.001) if optimize_chi else base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Saved nominal point, or JSON with point/nominal key",
    )
    p.add_argument(
        "--conditions",
        type=Path,
        required=True,
        help="Original settings JSON with conditions",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ec", type=float, nargs=2, required=True)
    p.add_argument("--mode", choices=["prescribed", "tolerance"], required=True)
    p.add_argument("--resonator-range-ghz", type=float, nargs=2)
    p.add_argument(
        "--h-targets",
        type=float,
        nargs="+",
        help="Prescribed mode: chi/H tradeoff targets. Tolerance mode: acceptance H floor only.",
    )
    p.add_argument("--frequency-step-ghz", type=float, default=0.05)
    p.add_argument("--max-halfwidth-ghz", type=float, default=1.0)
    p.add_argument("--boundary-resolution-ghz", type=float, default=0.00625)
    p.add_argument(
        "--quick-tolerance",
        action="store_true",
        help="Compatibility flag: tolerance mode now always uses first-feasible warm-started search",
    )
    p.add_argument(
        "--diagnose-tolerance",
        action="store_true",
        help="After each validated design, recheck fixed/flexible intervals under original acceptance",
    )
    p.add_argument("--starts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=70)
    a = p.parse_args()
    if not a.run:
        print("Use --run explicitly. No optimization started.")
        return
    c = json.loads(a.conditions.read_text(encoding="utf-8-sig"))
    set_policy(c["conditions"])
    a.h_targets = selected_h_targets(
        a.mode, a.h_targets, study.CONDITIONS.hybridization_min
    )
    if a.mode == "tolerance":
        a.quick_tolerance = True
    if any(h < study.CONDITIONS.hybridization_min or h >= 1 for h in a.h_targets):
        raise ValueError("H targets must be >= acceptance floor and < 1")
    if a.starts < 1 or a.maxiter < 1:
        raise ValueError("Positive starts/maxiter required")
    if (
        a.frequency_step_ghz <= 0
        or a.boundary_resolution_ghz <= 0
        or a.max_halfwidth_ghz < 0
        or a.max_halfwidth_ghz >= 6
    ):
        raise ValueError("Invalid step/halfwidth")
    if a.mode == "prescribed" and a.resonator_range_ghz is None:
        raise ValueError("Prescribed mode requires --resonator-range-ghz LOW HIGH")
    if a.resonator_range_ghz:
        frequency_grid(*a.resonator_range_ghz, a.frequency_step_ghz)
    data = json.loads(a.input.read_text(encoding="utf-8-sig"))
    nominal = data.get("point") or data.get("nominal") or data
    if any(abs(nominal[f"loaded_ec{i}_mhz"] - a.ec[i - 1]) > 1e-4 for i in (1, 2)):
        raise ValueError("Seed loaded EC tuple differs from --ec")
    if (
        abs(
            1
            / (
                2
                * np.pi
                * np.sqrt(
                    nominal["device"]["Cr_fF"]
                    * 1e-15
                    * nominal["device"]["Lr_nH"]
                    * 1e-9
                )
            )
            / 1e9
            - 6
        )
        > 1e-6
    ):
        raise ValueError("Nominal seed must use bare 6 GHz resonator")
    if (
        abs(
            np.sqrt(
                nominal["device"]["Lr_nH"] * 1e-9 / (nominal["device"]["Cr_fF"] * 1e-15)
            )
            - 50
        )
        > 1e-6
    ):
        raise ValueError("Nominal seed must use 50 Ohm resonator impedance")
    a.output.mkdir(exist_ok=True, parents=True)
    files = [
        Path(__file__),
        Path("readout_workflow/ec_design_improvement.py"),
        Path("readout_workflow/direct_ec_feasibility.py"),
        Path("readout_workflow/ec_resonator_tolerance.py"),
        Path("readout_workflow/calculation.py"),
        Path("readout_workflow/model.py"),
        Path("readout_workflow/validation.py"),
        Path("core/transmon.py"),
        Path("core/electrostatics.py"),
        Path("core/operating_point_dispersive.py"),
        Path("core/circuits.py"),
        Path("core/constants.py"),
        Path("core/runtime.py"),
        Path("core/tracking.py"),
    ]
    settings = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        conditions=asdict(study.CONDITIONS),
        source_hashes={str(f): sha256(f.read_bytes()).hexdigest() for f in files},
        input_sha256=sha256(a.input.read_bytes()).hexdigest(),
        condition_input_sha256=sha256(a.conditions.read_bytes()).hexdigest(),
        scope=(
            "Maximum symmetric resonator tolerance; chi and H are acceptance constraints only; no secondary chi maximization."
            if a.mode == "tolerance"
            else "Prescribed resonator range; maximize worst mean-chi at each minimum H target."
        )
        + " Shared fixed capacitors, independent EJ recourse, no capacitor fabrication-error study.",
    )
    path = a.output / "settings.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("Resume identity changed; use a fresh directory")
    atomic_json(path, settings)
    atomic_json(a.output / "nominal_input.json", nominal)
    snap = a.output / "source_snapshot"
    for file in files:
        dest = snap / file.resolve().relative_to(Path.cwd())
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(file.read_bytes())
    started = time.monotonic()
    range_ = a.resonator_range_ghz
    boundary = None
    if a.mode == "tolerance" and a.quick_tolerance:
        quick = quick_tolerance(
            a.ec,
            nominal,
            a.output / "quick_range",
            a.starts,
            a.maxiter,
            a.frequency_step_ghz,
            a.boundary_resolution_ghz,
            a.max_halfwidth_ghz,
        )
        atomic_json(a.output / "quick_range_result.json", quick)
        if quick["status"] != "validated_sampled_range":
            atomic_json(
                a.output / "completion.json",
                dict(
                    phase="no_validated_design_found",
                    elapsed_seconds=time.monotonic() - started,
                    completed_utc=utc(),
                ),
            )
            return
        range_ = quick["range_ghz"]
        boundary = quick["boundary"]
    results = []
    for h in sorted(set(a.h_targets)):
        if a.mode == "tolerance":
            # Exploration: the certified robust design is the final answer.
            # Do not add a strength objective after the range search.
            r = dict(quick["validated_result"], strength_optimized=False)
            (a.output / f"H_{h:.6f}").mkdir(parents=True, exist_ok=True)
            atomic_json(a.output / f"H_{h:.6f}" / "result.json", r)
        else:
            r = run_range(
                a.ec,
                nominal,
                *range_,
                h,
                a.output / f"H_{h:.6f}",
                a.starts,
                a.maxiter,
                a.frequency_step_ghz,
            )
            r = dict(r, strength_optimized=True)
        entry = dict(H_target=h, result=r)
        if r["status"] == "validated_sampled_range":
            center = next(
                q["point"]
                for q in r["validation"]["samples"]
                if abs(q["frequency_ghz"] - 6) < 1e-9
            )
            atomic_json(
                a.output / f"H_{h:.6f}" / "nominal.json",
                dict(point=center, ec_mhz=a.ec, conditions=asdict(study.CONDITIONS)),
            )
            if a.diagnose_tolerance:
                from .ec_resonator_tolerance import run_tuple

                policy = dict(
                    conditions=asdict(study.CONDITIONS),
                    frequency_range_ghz=[4.5, 12.0],
                    resonator_step_ghz=a.frequency_step_ghz,
                    boundary_resolution_ghz=a.boundary_resolution_ghz,
                    starts=3,
                    max_nfev=35,
                )
                diagnostic = run_tuple(
                    dict(
                        row=dict(
                            ec1_target_mhz=a.ec[0], ec2_target_mhz=a.ec[1], point=center
                        ),
                        settings=policy,
                        folder=str(a.output / f"H_{h:.6f}" / "tolerance_diagnostic"),
                    )
                )
                entry["tolerance_diagnostic"] = diagnostic
        results.append(entry)
        atomic_json(a.output / "results.json", results)
    atomic_json(
        a.output / "completion.json",
        dict(
            phase="complete",
            range_ghz=range_,
            boundary=boundary,
            elapsed_seconds=time.monotonic() - started,
            completed_utc=utc(),
        ),
    )


if __name__ == "__main__":
    main()
