# ruff: noqa: E402
# BLAS thread limits must be set before importing numerical libraries.
"""Resonator tolerance of saved EC-map implementations, with EJ-only rescue.

Run explicitly; imports perform no calculation. Each EC tuple owns its saved
three-node capacitors. Cr/Lr co-vary at 50 ohm; other capacitors are immutable.
Fixed operation means the saved dressed mean, with both EJs recalibratable.
Flexible operation can move within the original conditions. Search stops at
first validated witness; failure means no witness within the recorded budget.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
import json
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from . import calculation as study
from .storage import atomic_json, utc
from .validation import INTERMEDIATE, audit_point


def set_policy(conditions):
    fields = dict(conditions)
    for k in (
        "logical_frequency_ghz",
        "loaded_ec_mhz",
        "operation_frequency_ghz",
        "c12_bounds_ff",
        "readout_cap_bounds_ff",
    ):
        fields[k] = tuple(fields[k])
    study.CONDITIONS = study.Conditions(**fields)
    c = study.CONDITIONS
    study.BASELINE = dict(
        ec_min=c.loaded_ec_mhz[0],
        H=c.hybridization_min,
        logical_frequency=c.logical_frequency_ghz,
        ej_ec=c.ej_over_ec_min,
        fixed=False,
    )


def at_frequency(device, frequency):
    """No hidden compensation: rebuild inverse C after changing only Cr and Lr."""
    omega = 2 * np.pi * frequency * 1e9
    return replace(
        study.Device(**device), Cr_fF=1e15 / (omega * 50.0), Lr_nH=1e9 * 50.0 / omega
    )


def solve_frequency(nominal, frequency, mode, warm, settings, extra=()):
    started = time.monotonic()
    dev = at_frequency(nominal["device"], frequency)
    target = nominal["operating_frequency_ghz"] if mode == "fixed" else None
    model = study.ReadoutModel(dev, INTERMEDIATE)
    cache = {}
    attempts = []
    audits = []

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

    def coords(p):
        return np.array(
            [
                0.5 * (p["bare_f1_ghz"] + p["bare_f2_ghz"]),
                1000 * (p["bare_f1_ghz"] - p["bare_f2_ghz"]),
            ]
        )

    def residual(x):
        p = evaluate(x)
        v = study.violations(p, margin=0.015)
        # Interior guards avoid numerical rejection without changing acceptance.
        v = np.r_[v, max(0, 1.00005 - study.absolute_mean_chi_mhz(p))]
        if target is not None:
            v = np.r_[v, (p["operating_frequency_ghz"] - target) / 0.001]
        return v

    def check(p):
        if study.failures(p):
            return None
        if (
            target is not None
            and abs(p["operating_frequency_ghz"] - target)
            > study.CONDITIONS.fixed_frequency_tolerance_ghz
        ):
            return None
        q = audit_point(p, representative=False)
        failed = study.failures(q, require_sign_check=True)
        if (
            target is not None
            and abs(q["operating_frequency_ghz"] - target)
            > study.CONDITIONS.fixed_frequency_tolerance_ghz
        ):
            failed.append("fixed_operation_frequency")
        audits.append(dict(failures=failed, point=q))
        return q if not failed else None

    seeds = [coords(p) for p in [warm, *extra, nominal] if p]
    if target is None:
        a = coords(nominal)
        a[0] += frequency - 6.0
        seeds.insert(1, a)
    unique = []
    low = np.array([3.5, -250.0])
    high = np.array([8.1, 250.0])
    for seed in seeds:
        seed = np.clip(seed, low + 1e-6, high - 1e-6)
        if not any(np.linalg.norm(seed - q) < 1e-6 for q in unique):
            unique.append(seed)
    chosen = None
    with threadpool_limits(limits=1):
        for seed in unique[: settings["starts"]]:
            p = evaluate(seed)
            chosen = check(p)
            if chosen:
                break
            fit = least_squares(
                residual,
                seed,
                bounds=(low, high),
                x_scale=[0.3, 10.0],
                diff_step=2e-5,
                max_nfev=settings["max_nfev"],
                ftol=1e-9,
                xtol=1e-10,
                gtol=1e-9,
            )
            p = evaluate(fit.x)
            attempts.append(
                dict(
                    seed=seed.tolist(),
                    x=fit.x.tolist(),
                    nfev=int(fit.nfev),
                    failures=study.failures(p),
                    cost=float(fit.cost),
                )
            )
            proposals = sorted(
                cache.values(),
                key=lambda p: (
                    np.linalg.norm(study.violations(p))
                    + (
                        abs(p["operating_frequency_ghz"] - target) / 0.001
                        if target is not None
                        else 0
                    )
                ),
            )[:3]
            for p in proposals:
                chosen = check(p)
                if chosen:
                    break
            if chosen:
                break
    closest = min(
        cache.values(),
        key=lambda p: (
            np.linalg.norm(study.violations(p))
            + (
                abs(p["operating_frequency_ghz"] - target) / 0.001
                if target is not None
                else 0
            )
        ),
    )
    return dict(
        frequency_ghz=frequency,
        mode=mode,
        status="validated_pass" if chosen else "no_validated_solution_found",
        point=chosen,
        closest=closest,
        failures=study.failures(closest),
        fixed_operation_error_ghz=(
            closest["operating_frequency_ghz"] - target if target is not None else None
        ),
        attempts=attempts,
        audits=audits,
        evaluations=len(cache),
        seconds=time.monotonic() - started,
        completed_utc=utc(),
    )


def run_tuple(task):
    """Four boundary searches per tuple; every frequency call is resumable."""
    settings = task["settings"]
    set_policy(settings["conditions"])
    row = task["row"]
    nominal = row["point"]
    folder = Path(task["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / "result.json"
    if saved.exists():
        return json.loads(saved.read_text())
    started = time.monotonic()
    history = {}
    bounds = {}

    def at(mode, frequency, warm, extra=()):
        key = (mode, round(float(frequency), 9))
        path = folder / f"{mode}_{key[1]:.9f}.json"
        if key in history:
            return history[key]
        if path.exists():
            r = json.loads(path.read_text())
        else:
            r = solve_frequency(nominal, key[1], mode, warm, settings, extra)
            atomic_json(path, r)
        history[key] = r
        atomic_json(
            folder / "progress.json",
            dict(
                samples=len(history),
                last_mode=mode,
                last_frequency_ghz=frequency,
                updated_utc=utc(),
            ),
        )
        return r

    fixed_passes = {}
    for mode in ("fixed", "flexible"):
        center = at(mode, 6.0, nominal)
        if not center["point"]:
            bounds[mode] = dict(status="nominal_revalidation_failed", center=center)
            continue
        ends = {}
        for direction, label in [(-1, "lower"), (1, "upper")]:
            good = center
            bad = None
            f = 6.0
            while True:
                next_f = round(f + direction * settings["resonator_step_ghz"], 9)
                if (
                    next_f < settings["frequency_range_ghz"][0] - 1e-8
                    or next_f > settings["frequency_range_ghz"][1] + 1e-8
                ):
                    break
                extra = [fixed_passes[next_f]] if next_f in fixed_passes else []
                r = at(mode, next_f, good["point"], extra)
                if r["point"] is None:
                    bad = r
                    break
                good = r
                f = next_f
                if mode == "fixed":
                    fixed_passes[next_f] = r["point"]
            if bad:
                while (
                    abs(bad["frequency_ghz"] - good["frequency_ghz"])
                    > settings["boundary_resolution_ghz"] + 1e-10
                ):
                    mid = round((bad["frequency_ghz"] + good["frequency_ghz"]) / 2, 9)
                    r = at(
                        mode,
                        mid,
                        good["point"],
                        [fixed_passes[mid]] if mid in fixed_passes else [],
                    )
                    if r["point"] is None:
                        bad = r
                    else:
                        good = r
                        if mode == "fixed":
                            fixed_passes[mid] = r["point"]
            ends[label] = dict(
                validated_frequency_ghz=good["frequency_ghz"],
                tolerance_mhz=abs(good["frequency_ghz"] - 6) * 1000,
                failed_frequency_ghz=bad["frequency_ghz"] if bad else None,
                failed_checks=bad["failures"] if bad else [],
                censored_by_scan_range=bad is None,
                endpoint_point=good["point"],
            )
        bounds[mode] = dict(
            status="complete",
            nominal_operation_ghz=nominal["operating_frequency_ghz"],
            **ends,
        )
    # Fixed witnesses MUST be accepted by flexible policy. Retain explicit checks
    # instead of widening a flexible interval across an unchecked gap.
    consistency = []
    if all(
        bounds.get(m, {}).get("status") == "complete" for m in ("fixed", "flexible")
    ):
        for side in ("lower", "upper"):
            a = bounds["fixed"][side]["validated_frequency_ghz"]
            b = bounds["flexible"][side]["validated_frequency_ghz"]
            if (side == "lower" and b > a + 1e-9) or (side == "upper" and b < a - 1e-9):
                consistency.append(side)
    result = dict(
        ec1_target_mhz=row["ec1_target_mhz"],
        ec2_target_mhz=row["ec2_target_mhz"],
        nominal=nominal,
        bounds=bounds,
        consistency_warnings=consistency,
        samples=len(history),
        seconds=time.monotonic() - started,
        completed_utc=utc(),
    )
    atomic_json(saved, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ec-step-mhz", type=int, default=2)
    p.add_argument("--wall-hours", type=float, default=12.0)
    p.add_argument("--frequency-range-ghz", nargs=2, type=float, default=(4.5, 12.0))
    p.add_argument("--pilot", action="store_true")
    p.add_argument(
        "--source",
        type=Path,
        default=Path("Ec_tuple search workflow/readout/direct_ec/rail_b_step2"),
    )
    a = p.parse_args()
    if not a.run:
        print("Use --run explicitly.")
        return
    if a.ec_step_mhz < 2 or a.ec_step_mhz % 2:
        raise ValueError("EC step must be a positive multiple of 2 MHz")
    source = a.source / "results.json"
    allrows = json.loads(source.read_text())
    policy = json.loads((a.source / "settings.json").read_text())["conditions"]
    rows = [
        r
        for r in allrows
        if r.get("point")
        and (r["ec1_target_mhz"] - 150) % a.ec_step_mhz == 0
        and (r["ec2_target_mhz"] - 150) % a.ec_step_mhz == 0
    ]
    # First cover a 4 MHz sublattice, then fill its remaining 2 MHz phases.
    # Ordered saved implementations remain separate; no symmetry cloning.
    rows.sort(
        key=lambda r: (
            ((r["ec1_target_mhz"] - 150) % (2 * a.ec_step_mhz)) // a.ec_step_mhz
            + 2 * ((r["ec2_target_mhz"] - 150) % (2 * a.ec_step_mhz)) // a.ec_step_mhz,
            r["ec1_target_mhz"],
            r["ec2_target_mhz"],
        )
    )
    if a.pilot:
        # Deterministic spatially distributed points, not a random cluster.
        targets = [
            (160, 260),
            (180, 220),
            (200, 200),
            (250, 170),
            (154, 180),
            (280, 160),
            (220, 250),
            (180, 154),
        ]
        pool = [r for r in allrows if r.get("point")]
        selected = []
        for x, y in targets:
            r = min(
                pool,
                key=lambda r: (
                    (r["ec1_target_mhz"] - x) ** 2 + (r["ec2_target_mhz"] - y) ** 2
                ),
            )
            if (r["ec1_target_mhz"], r["ec2_target_mhz"]) not in [
                (q["ec1_target_mhz"], q["ec2_target_mhz"]) for q in selected
            ]:
                selected.append(r)
        rows = selected
    a.output.mkdir(parents=True, exist_ok=True)
    files = [
        Path(__file__).resolve(),
        Path(study.__file__).resolve(),
        *[
            Path(q).resolve()
            for q in (
                "readout_workflow/model.py",
                "readout_workflow/validation.py",
                "core/transmon.py",
                "core/electrostatics.py",
                "core/operating_point_dispersive.py",
                "core/circuits.py",
                "core/constants.py",
                "core/runtime.py",
                "core/tracking.py",
            )
        ],
    ]
    settings = dict(
        schema="ec-resonator-tolerance-v1",
        conditions=policy,
        impedance_ohm=50,
        nominal_resonator_ghz=6.0,
        frequency_range_ghz=list(a.frequency_range_ghz),
        resonator_step_ghz=0.05,
        boundary_resolution_ghz=0.00625,
        starts=3,
        max_nfev=35,
        workers=a.workers,
        wall_hours=a.wall_hours,
        ec_step_mhz=a.ec_step_mhz,
        lattice_origin_mhz=150,
        pilot=a.pilot,
        targets=[[r["ec1_target_mhz"], r["ec2_target_mhz"]] for r in rows],
        input_sha256=sha256(source.read_bytes()).hexdigest(),
        source_hashes={str(x): sha256(x.read_bytes()).hexdigest() for x in files},
        scope="Each saved three-node implementation fixed except Cr/Lr at 50 Ohm and both EJs. First feasible, no strength optimization. Stop at first failed coarse sample, refine bracket. Connectedness is grid-resolved, not a continuum proof.",
    )
    sp = a.output / "settings.json"
    if sp.exists() and json.loads(sp.read_text()) != settings:
        raise ValueError("Configuration/source mismatch: use fresh output")
    atomic_json(sp, settings)
    # Complete source witnesses and source snapshots make the run transferable.
    if not (a.output / "input_witnesses.json").exists():
        atomic_json(a.output / "input_witnesses.json", rows)
    snap = a.output / "source_snapshot"
    snap.mkdir(exist_ok=True)
    for x in files:
        dest = snap / x.relative_to(Path.cwd())
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(x.read_bytes())
    started = time.monotonic()
    results = []
    pending = []
    for row in rows:
        folder = a.output / f"ec_{row['ec1_target_mhz']}_{row['ec2_target_mhz']}"
        if (folder / "result.json").exists():
            results.append(json.loads((folder / "result.json").read_text()))
        else:
            pending.append(dict(row=row, settings=settings, folder=str(folder)))
    # Keep only one outstanding tuple per worker. The 12-hour budget stops
    # scheduling new tuples; active tuples finish and retain full checkpoints.
    remaining = iter(pending)
    done_new = 0
    budget = a.wall_hours * 3600
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {}

        def schedule():
            if time.monotonic() - started >= budget:
                return False
            try:
                task = next(remaining)
            except StopIteration:
                return False
            futures[pool.submit(run_tuple, task)] = task
            return True

        for _ in range(a.workers):
            schedule()
        while futures:
            completed, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            if not completed:
                atomic_json(
                    a.output / "progress.json",
                    dict(
                        phase="running",
                        completed=len(results),
                        total=len(rows),
                        active=len(futures),
                        elapsed_seconds=time.monotonic() - started,
                        updated_utc=utc(),
                    ),
                )
            for fut in completed:
                task = futures.pop(fut)
                try:
                    r = fut.result()
                    results.append(r)
                    done_new += 1
                except Exception:
                    atomic_json(
                        Path(task["folder"]) / "exception.json",
                        dict(traceback=traceback.format_exc(), updated_utc=utc()),
                    )
                    raise
                elapsed = time.monotonic() - started
                progress = dict(
                    phase="running",
                    completed=len(results),
                    total=len(rows),
                    workers=a.workers,
                    elapsed_seconds=elapsed,
                    estimated_remaining_hours=(elapsed / max(1, done_new))
                    * (len(rows) - len(results))
                    / 3600,
                    last_tuple=[r["ec1_target_mhz"], r["ec2_target_mhz"]],
                    updated_utc=utc(),
                )
                atomic_json(a.output / "progress.json", progress)
                atomic_json(a.output / "results.json", results)
                print(json.dumps(progress), flush=True)
                schedule()
    phase = "complete" if len(results) == len(rows) else "time_budget_reached"
    atomic_json(
        a.output / "progress.json",
        dict(
            phase=phase,
            completed=len(results),
            total=len(rows),
            elapsed_seconds=time.monotonic() - started,
            updated_utc=utc(),
        ),
    )
    print(phase.upper(), flush=True)


if __name__ == "__main__":
    main()
