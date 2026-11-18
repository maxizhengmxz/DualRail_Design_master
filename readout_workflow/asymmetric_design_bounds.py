# ruff: noqa: E402
# Configure numerical threads before importing NumPy.
"""Asymmetric resonator limits of saved optimized capacitor designs.

Four limits per tuple: down/up, fixed/flexible dressed operation frequency.
No capacitor redesign or chi optimization. Reuse the existing certified
symmetric interval and saved frequency witnesses. Every new point uses the
original adaptive acceptance and sign checks; checkpoints are resumable.
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

from . import calculation as study
from .ec_resonator_tolerance import at_frequency, set_policy, solve_frequency
from .storage import atomic_json, utc
from .validation import FAST


def saved_endpoint(task, frequency, nominal):
    """Locate a previously certified endpoint of THIS physical capacitor design."""
    roots = [Path(task["source_point"])] + [
        Path(p) for p in task.get("prior_points", [])
    ]
    for root in roots:
        for path in root.glob(f"**/validation/frequency_{frequency:.10f}.json"):
            r = json.loads(path.read_text())
            p = r.get("point")
            if p is None or r.get("failures"):
                continue
            if any(
                abs(p["device"][k] - nominal["device"][k]) > 1e-7 for k in study.CAPS
            ):
                continue
            expected = asdict(at_frequency(nominal["device"], frequency))
            if any(
                abs(p["device"][k] - expected[k]) > 1e-7 for k in ("Cr_fF", "Lr_nH")
            ):
                continue
            if not study.failures(p, require_sign_check=True):
                return p, str(path)
    raise ValueError(
        f"No matching saved endpoint certificate at {frequency} for {task['ec']}"
    )


def run_design(task):
    config = task["config"]
    set_policy(config["conditions"])
    out = Path(task["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    source = Path(task["source_point"])
    nominal = json.loads((source / "H_0.950000" / "nominal.json").read_text())["point"]
    if study.failures(nominal, require_sign_check=True):
        raise ValueError("Source nominal certificate is not valid")
    start = time.monotonic()
    cache = {}
    bounds = {}
    history = []

    def at(mode, f, warm, extra=()):
        f = round(float(f), 9)
        key = (mode, f)
        path = out / f"{mode}_{f:.9f}.json"
        if key in cache:
            return cache[key]
        if path.exists():
            r = json.loads(path.read_text())
        elif abs(f - 6) < 1e-9:
            r = dict(
                frequency_ghz=f,
                mode=mode,
                point=nominal,
                status="validated_pass",
                reused_source=str(source / "H_0.950000" / "nominal.json"),
                failures=[],
                seconds=0.0,
            )
        else:
            # A fixed-operation witness is also a valid flexible witness.
            prior = cache.get(("fixed", f)) if mode == "flexible" else None
            if prior and prior["point"]:
                r = dict(prior, mode="flexible", reused_fixed_witness=True)
            else:
                fast = dict(config, screening_basis=asdict(FAST))
                r = solve_frequency(nominal, f, mode, warm, fast, extra)
                if r["point"] is None:
                    medium = solve_frequency(
                        nominal,
                        f,
                        mode,
                        r.get("closest") or warm,
                        config,
                        [warm, *extra],
                    )
                    medium["fast_screen_seconds"] = r["seconds"]
                    r = medium
                if (
                    r["point"] is None
                    and mode == "fixed"
                    and abs(
                        r["closest"]["operating_frequency_ghz"]
                        - nominal["operating_frequency_ghz"]
                    )
                    > study.CONDITIONS.fixed_frequency_tolerance_ghz
                ):
                    r["failures"] = list(
                        set(r["failures"] + ["fixed_operation_frequency"])
                    )
        cache[key] = r
        history.append(dict(mode=mode, frequency_ghz=f, status=r["status"]))
        atomic_json(path, r)
        atomic_json(
            out / "progress.json",
            dict(
                samples=len(cache),
                last_mode=mode,
                last_frequency_ghz=f,
                updated_utc=utc(),
            ),
        )
        return r

    known = {}
    for sign, side in [(-1, "lower"), (1, "upper")]:
        f = round(6 + sign * task["radius_ghz"], 9)
        if abs(f - 6) < 1e-9:
            known[side] = dict(frequency_ghz=6.0, point=nominal)
        else:
            p, lineage = saved_endpoint(task, f, nominal)
            known[side] = dict(frequency_ghz=f, point=p, reused_source=lineage)
    for mode in ("fixed", "flexible"):
        center = at(mode, 6.0, nominal)
        ends = {}
        for direction, side in [(-1, "lower"), (1, "upper")]:
            good = center
            bad = None
            if mode == "flexible":
                # Union of two certified intervals containing 6 GHz is connected.
                options = [
                    known[side],
                    dict(
                        frequency_ghz=bounds["fixed"][side]["validated_frequency_ghz"],
                        point=bounds["fixed"][side]["endpoint_point"],
                    ),
                ]
                good = (
                    min(options, key=lambda r: r["frequency_ghz"])
                    if direction < 0
                    else max(options, key=lambda r: r["frequency_ghz"])
                )
            while True:
                f = round(
                    good["frequency_ghz"] + direction * config["resonator_step_ghz"], 9
                )
                limit = config["frequency_range_ghz"][0 if direction < 0 else 1]
                f = max(f, limit) if direction < 0 else min(f, limit)
                if abs(f - good["frequency_ghz"]) < 1e-10:
                    break
                r = at(mode, f, good["point"])
                if r["point"] is None:
                    bad = r
                    break
                good = r
            if bad:
                while (
                    abs(good["frequency_ghz"] - bad["frequency_ghz"])
                    > config["boundary_resolution_ghz"] + 1e-10
                ):
                    f = (good["frequency_ghz"] + bad["frequency_ghz"]) / 2
                    r = at(mode, f, good["point"])
                    if r["point"] is None:
                        bad = r
                    else:
                        good = r
            ends[side] = dict(
                validated_frequency_ghz=good["frequency_ghz"],
                tolerance_mhz=abs(good["frequency_ghz"] - 6) * 1000,
                failed_frequency_ghz=bad["frequency_ghz"] if bad else None,
                failed_checks=bad.get("failures", []) if bad else [],
                censored_by_scan_range=bad is None,
                endpoint_point=good["point"],
            )
        bounds[mode] = ends
    result = dict(
        pair=task["pair"],
        ec_mhz=task["ec"],
        nominal=nominal,
        bounds=bounds,
        source_symmetric_radius_ghz=task["radius_ghz"],
        nominal_operation_ghz=nominal["operating_frequency_ghz"],
        seconds=time.monotonic() - start,
        samples=len(cache),
        history=history,
        completed_utc=utc(),
    )
    atomic_json(out / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    if not a.run:
        print("Use --run explicitly")
        return
    plan = json.loads((a.source / "sequence_plan.json").read_text())
    recorded = {k.replace(chr(92), "/"): v for k, v in plan["source_hashes"].items()}
    for relative in (
        "readout_workflow/calculation.py",
        "readout_workflow/model.py",
        "readout_workflow/validation.py",
        "core/transmon.py",
        "core/electrostatics.py",
    ):
        if sha256(Path(relative).read_bytes()).hexdigest() != recorded[relative]:
            raise ValueError(f"Source physics changed: {relative}")
    compact = json.loads((a.source / "report_compact.json").read_text())
    tasks = []
    for pair, stage in [("B", "B_fine"), ("A", "A_refined")]:
        config = dict(
            conditions=json.loads(
                (a.source / "input_snapshot" / pair / "settings.json").read_text()
            )["conditions"],
            resonator_step_ghz=0.05,
            boundary_resolution_ghz=0.00625,
            frequency_range_ghz=[4.0, 16.0],
            starts=3,
            max_nfev=35,
        )
        for r in compact[stage]:
            x, y = r["ec"]
            point = a.source / stage / f"ec_{x}_{y}"
            tasks.append(
                dict(
                    pair=pair,
                    ec=r["ec"],
                    radius_ghz=r["radius_mhz"] / 1000,
                    source_point=str(point),
                    prior_points=[str(a.source / "A_coarse" / f"ec_{x}_{y}")]
                    if pair == "A"
                    else [],
                    config=config,
                    output=str(a.output / pair / f"ec_{x}_{y}"),
                )
            )
    if a.pilot:
        wanted = {("B", 176, 212), ("B", 208, 242), ("A", 152, 218), ("A", 164, 230)}
        tasks = [t for t in tasks if (t["pair"], *t["ec"]) in wanted]
    a.output.mkdir(parents=True, exist_ok=True)
    files = [
        Path(__file__),
        Path("readout_workflow/ec_resonator_tolerance.py"),
        Path("readout_workflow/calculation.py"),
        Path("readout_workflow/model.py"),
        Path("readout_workflow/validation.py"),
        Path("core/transmon.py"),
        Path("core/electrostatics.py"),
    ]
    settings = dict(
        workers=a.workers,
        pilot=a.pilot,
        source=str(a.source.resolve()),
        source_compact_sha256=sha256(
            (a.source / "report_compact.json").read_bytes()
        ).hexdigest(),
        tasks=[dict(pair=t["pair"], ec=t["ec"], config=t["config"]) for t in tasks],
        source_hashes={str(f): sha256(f.read_bytes()).hexdigest() for f in files},
        scope="Asymmetric limits of existing optimized designs. Both EJs recalibrate; fixed mode preserves nominal dressed mean, flexible mode may move it. No capacitor redesign, no chi optimization.",
    )
    path = a.output / "settings.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("Resume identity changed; use new output")
    atomic_json(path, settings)
    for f in files:
        dest = a.output / "source_snapshot" / f.resolve().relative_to(Path.cwd())
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(f.read_bytes())
    pending = []
    results = []
    for t in tasks:
        path = Path(t["output"]) / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            pending.append(t)
    tic = time.monotonic()
    errors = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run_design, t): t for t in pending}
        for future in as_completed(futures):
            t = futures[future]
            try:
                r = future.result()
            except Exception as exc:
                errors.append(dict(task=t, error=repr(exc), updated_utc=utc()))
                atomic_json(a.output / "errors.json", errors)
                continue
            results.append(r)
            if len(results) % 25 == 0:
                atomic_json(a.output / "results.json", results)
            state = dict(
                phase="running",
                completed=len(results),
                total=len(tasks),
                elapsed_seconds=time.monotonic() - tic,
                updated_utc=utc(),
            )
            atomic_json(a.output / "progress.json", state)
            print(json.dumps(state), flush=True)
    atomic_json(a.output / "results.json", results)
    atomic_json(
        a.output / "progress.json",
        dict(
            phase="complete_with_errors" if errors else "complete",
            completed=len(results),
            total=len(tasks),
            errors=len(errors),
            elapsed_seconds=time.monotonic() - tic,
            updated_utc=utc(),
        ),
    )
    if errors:
        raise RuntimeError(f"{len(errors)} asymmetric-bound tasks require review")


if __name__ == "__main__":
    main()
