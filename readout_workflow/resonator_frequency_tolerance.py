# ruff: noqa: E402
# BLAS thread limits must be set before importing numerical libraries.
"""Fixed-50-ohm resonator tolerance of one saved standalone readout triple.

Only Cr, Lr and the two transmon EJs vary. All actual pad capacitors and links
stay fixed; loaded ECs are recalculated without compensation. Uses the saved
coupler-design readout policy (including its EJ/EC >45 exception), not a
fabrication rescue ladder. Run explicitly with --run; importing does no work.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from readout_workflow import calculation as study
from readout_workflow.storage import atomic_json, utc
from readout_workflow.validation import FAST, INTERMEDIATE, audit_point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Gnd-coupler+Floating-T"))
from floating_coupler.fabrication import readout_device
from standalone_pad_readout import policy


def device_at_frequency(readout, frequency_ghz, impedance_ohm=50.0):
    """Rebuild the physical pad circuit; effective modal Cr is not physical Cr."""
    varied = deepcopy(readout)
    omega = 2 * np.pi * float(frequency_ghz) * 1e9
    varied["physical"]["resonator_self_capacitance_ff"] = 1e15 / (omega * impedance_ohm)
    varied["physical"]["resonator_inductance_nh"] = 1e9 * impedance_ohm / omega
    device = readout_device(varied, {})
    varied["physical"]["effective_device_for_exact_equivalent_H"] = asdict(device)
    return device, varied["physical"]


def solve(readout, frequency_ghz, warm=()):
    policy(readout["case"])
    device, physical = device_at_frequency(readout, frequency_ghz)
    started = time.monotonic()
    attempts = []
    closest = None
    chosen = None
    nominal = readout["nominal"]
    nominal_fr = readout["coordinates"][4]
    starts = []
    for p in warm:
        starts.extend(
            [
                [
                    0.5 * (p["bare_f1_ghz"] + p["bare_f2_ghz"]),
                    p["bare_f1_ghz"] - p["bare_f2_ghz"],
                ],
                [
                    0.5 * (p["bare_f1_ghz"] + p["bare_f2_ghz"])
                    + 0.3 * (frequency_ghz - nominal_fr),
                    p["bare_f1_ghz"] - p["bare_f2_ghz"],
                ],
            ]
        )
    f0 = 0.5 * (nominal["bare_f1_ghz"] + nominal["bare_f2_ghz"])
    delta = nominal["bare_f1_ghz"] - nominal["bare_f2_ghz"]
    starts.extend(
        [
            [f0 + frequency_ghz - nominal_fr, delta],
            [f0, delta],
            [frequency_ghz - 1.4, 0.0],
            [frequency_ghz - 1.7, 0.004],
        ]
    )
    lo = np.array([3.8, -0.25])
    hi = np.array([5.8, 0.25])
    unique = []
    for x in starts:
        x = np.clip(x, lo + 1e-6, hi - 1e-6)
        if not any(np.linalg.norm(x - y) < 1e-5 for y in unique):
            unique.append(x)
    with threadpool_limits(limits=1):
        for seed in unique[:5]:
            x = seed
            for basis in (FAST, INTERMEDIATE):
                model = study.ReadoutModel(device, basis)
                cache = {}

                def evaluate(x, cache=cache, model=model, basis=basis):
                    key = tuple(map(float, x))
                    if key not in cache:
                        fs = [x[0] + x[1] / 2, x[0] - x[1] / 2]
                        ej = [
                            study.inverse_frequency(float(f), float(ec), basis)
                            for f, ec in zip(fs, model.ec_ghz[:2], strict=True)
                        ]
                        cache[key] = study.annotate(model, *ej)
                    return cache[key]

                def residual(x):
                    p = evaluate(x)
                    v = study.violations(p, margin=0.15)
                    # Small strength guard helps prevent audit rounding at 1 MHz.
                    return np.r_[v, max(0, 1.003 - study.absolute_mean_chi_mhz(p))]

                fit = least_squares(
                    residual,
                    x,
                    bounds=(lo, hi),
                    x_scale=[0.3, 0.01],
                    diff_step=2e-5,
                    max_nfev=55,
                    ftol=2e-9,
                    xtol=2e-9,
                    gtol=2e-9,
                )
                x = fit.x
                p = evaluate(x)
                attempt = dict(
                    seed=seed.tolist(),
                    basis=asdict(basis),
                    x=x.tolist(),
                    nfev=int(fit.nfev),
                    cost=float(fit.cost),
                    failures=study.failures(p),
                    endpoint=p,
                )
                attempts.append(attempt)
                candidates = sorted(
                    cache.values(), key=lambda q: np.linalg.norm(study.violations(q))
                )
                best = candidates[0]
                if closest is None or np.linalg.norm(
                    study.violations(best)
                ) < np.linalg.norm(study.violations(closest)):
                    closest = best
                if study.failures(best):
                    break
                checked = audit_point(best, representative=False)
                rejected = study.failures(checked, require_sign_check=True)
                attempt["audit_failures"] = rejected
                attempt["audited_point"] = checked
                if not rejected:
                    chosen = checked
                    break
                # Refine in the intermediate basis if the screen missed a boundary.
            if chosen is not None:
                break
    return dict(
        bare_resonator_frequency_ghz=float(frequency_ghz),
        status="validated_pass" if chosen else "no_validated_solution_found",
        point=chosen,
        closest=closest,
        attempts=attempts,
        physical_resonator_capacitance_ff=physical["resonator_self_capacitance_ff"],
        physical_resonator_inductance_nh=physical["resonator_inductance_nh"],
        seconds=time.monotonic() - started,
        completed_utc=utc(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--range-ghz", type=float, nargs=2, default=(5.0, 7.5))
    parser.add_argument("--step-ghz", type=float, default=0.05)
    parser.add_argument("--boundary-resolution-ghz", type=float, default=0.005)
    args = parser.parse_args()
    if not args.run:
        print("Use --run to calculate.")
        return
    source = args.input.resolve()
    readout = json.loads(source.read_text(encoding="utf-8-sig"))
    policy(readout["case"])
    args.output.mkdir(parents=True, exist_ok=True)
    nominal_fr = float(readout["coordinates"][4])
    rows = []
    config = dict(
        source=str(source),
        source_sha256=sha256(source.read_bytes()).hexdigest(),
        conditions=asdict(study.CONDITIONS),
        frequency_range_ghz=args.range_ghz,
        step_ghz=args.step_ghz,
        boundary_resolution_ghz=args.boundary_resolution_ghz,
        impedance_ohm=50,
        scope="Standalone rail pair and resonator only. Cr and Lr vary at fixed 50 ohms; all other physical caps fixed, loaded ECs uncompensated. Only rail EJs recalibrate. Flexible operation frequency; no fabrication rescue.",
        nominal_physical=readout["physical"],
    )
    settings_path = args.output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != json.loads(
        json.dumps(config)
    ):
        raise ValueError("Saved settings differ; use a fresh output directory.")
    atomic_json(settings_path, config)

    def at(f):
        prior = next(
            (r for r in rows if abs(r["bare_resonator_frequency_ghz"] - f) < 1e-10),
            None,
        )
        if prior:
            return prior
        path = args.output / f"frequency_{f:.9f}.json"
        if path.exists():
            row = json.loads(path.read_text())
        else:
            good = sorted(
                [r for r in rows if r["point"]],
                key=lambda r: abs(r["bare_resonator_frequency_ghz"] - f),
            )
            row = solve(readout, f, [r["point"] for r in good[:1]])
            atomic_json(path, row)
        rows.append(row)
        atomic_json(
            args.output / "progress.json",
            dict(
                completed=len(rows),
                passed=sum(r["point"] is not None for r in rows),
                last_frequency_ghz=f,
                updated_utc=utc(),
            ),
        )
        print(
            json.dumps(dict(frequency=f, status=row["status"], seconds=row["seconds"])),
            flush=True,
        )
        return row

    t = time.monotonic()
    at(nominal_fr)
    grid = np.arange(args.range_ghz[0], args.range_ghz[1] + 1e-9, args.step_ghz)
    for f in sorted(grid, key=lambda x: abs(x - nominal_fr)):
        at(float(round(f, 9)))
    ordered = sorted(rows, key=lambda r: r["bare_resonator_frequency_ghz"])
    brackets = []
    for a, b in zip(ordered[:-1], ordered[1:], strict=True):
        if (a["point"] is None) == (b["point"] is None):
            continue
        left, right = a, b
        while (
            right["bare_resonator_frequency_ghz"] - left["bare_resonator_frequency_ghz"]
            > args.boundary_resolution_ghz
        ):
            mid = at(
                (
                    left["bare_resonator_frequency_ghz"]
                    + right["bare_resonator_frequency_ghz"]
                )
                / 2
            )
            if (mid["point"] is None) == (left["point"] is None):
                left = mid
            else:
                right = mid
        brackets.append(
            dict(
                lower_frequency_ghz=left["bare_resonator_frequency_ghz"],
                upper_frequency_ghz=right["bare_resonator_frequency_ghz"],
                lower_pass=left["point"] is not None,
                upper_pass=right["point"] is not None,
            )
        )
    rows.sort(key=lambda r: r["bare_resonator_frequency_ghz"])
    atomic_json(args.output / "results.json", rows)
    atomic_json(
        args.output / "summary.json",
        dict(
            samples=len(rows),
            passed=sum(r["point"] is not None for r in rows),
            boundary_brackets=brackets,
            elapsed_seconds=time.monotonic() - t,
            nominal_bare_frequency_ghz=nominal_fr,
        ),
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
