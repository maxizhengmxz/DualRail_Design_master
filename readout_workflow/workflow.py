"""Map-only pipeline: feasibility -> diagnosis -> strength -> continuation.

Workers calculate; only the parent commits events. Every failed pixel is
diagnosed at the same 1 fF spacing. Accepted map points maximize absolute mean
chi, with the agreed 0.5% continuity allowance. No robustness or report renderer
is imported or launched. Results are numerical JSON/CSV, not presentation.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import calculation as study
from . import diagnostics as diagnosis
from . import optimization
from .storage import Ledger, atomic_json, key, utc
from .validation import audit_point, metrics


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    atomic_json(path, value)


def compact(record):
    """Small live index; full attempts and validation candidates stay in SQLite."""
    return {
        k: record[k]
        for k in (
            "c1",
            "c2",
            "status",
            "point",
            "failures",
            "dominant",
            "normalized_deficits",
            "score",
            "sample_id",
            "phase",
            "metrics",
            "baseline_eligible",
            "baseline_H",
            "baseline_failures",
            "design",
            "kind",
            "id",
            "errors",
            "first_pass",
            "seconds",
            "error",
        )
        if k in record
    }


def nearest(pair, records, count=3):
    candidates = [r for r in records if r.get("point")]
    return sorted(
        candidates,
        key=lambda r: (
            (r["point"]["device"]["C1_shunt_fF"] - pair[0]) ** 2
            + (r["point"]["device"]["C2_shunt_fF"] - pair[1]) ** 2
        ),
    )[:count]


def run_jobs(pool, tasks, worker, completed, workers):
    """Keep a bounded queue full; tasks can use newly completed neighbors."""
    iterator = iter(tasks)
    pending = {}

    def submit():
        try:
            task = next(iterator)
        except StopIteration:
            return False
        pending[pool.submit(worker, task)] = task
        return True

    for _ in range(workers * 2):
        if not submit():
            break
    while pending:
        finished, _ = wait(pending, timeout=45, return_when=FIRST_COMPLETED)
        if not finished:
            print(
                json.dumps(dict(waiting=True, utc=utc(), active=len(pending))),
                flush=True,
            )
        for future in finished:
            task = pending.pop(future)
            # Worker exceptions stop this phase rather than becoming false
            # "infeasible" labels. Finished records survive a safe restart.
            completed(task, future.result())
            submit()


def continue_row(task):
    """Ascending-C1 continuation within one C2 row; never interpolate pixels."""
    previous = task.get("previous")
    records = []
    with threadpool_limits(limits=1):
        for baseline in task["points"]:
            c1, c2 = baseline["c1"], baseline["c2"]
            point = baseline["point"]
            floor = 0.995 * study.absolute_mean_chi_mhz(point)
            if previous is None or c1 != previous["device"]["C1_shunt_fF"] + 1:
                result = dict(
                    c1=c1,
                    c2=c2,
                    mode="smooth",
                    status="validated_pass",
                    point=point,
                    metrics=metrics(point),
                    failures=[],
                    strength_floor=floor,
                    reused_row_start=True,
                )
            else:
                result = optimization.solve(
                    dict(
                        c1=c1,
                        c2=c2,
                        mode="smooth",
                        warm=[point, previous],
                        target=study.x_from_point(previous),
                        strength_floor=floor,
                    )
                )
                if result["status"] != "validated_pass":
                    # Explicit, audited fallback retains full strength. It is
                    # recorded, not disguised as a successful local projection.
                    result.update(
                        status="validated_pass",
                        point=point,
                        metrics=metrics(point),
                        failures=[],
                        continuation_fallback=True,
                    )
            previous = result["point"]
            records.append(result)
    return records


def configuration(pairs, archive_path=None):
    """Fingerprint equations, physical rules and seeds independently of old IDs."""
    package = Path(__file__).resolve().parent
    sources = (
        "calculation.py",
        "diagnostics.py",
        "optimization.py",
        "validation.py",
        "workflow.py",
        "storage.py",
    )
    return dict(
        schema="readout-map-only-v3",
        conditions=asdict(study.CONDITIONS),
        grid=dict(min_ff=36, max_ff=128, step_ff=1, triangle="C1>C2"),
        pair_count=len(pairs),
        selected_pairs=pairs if len(pairs) != 4278 else None,
        diagnostic_step_ff=1,
        objective="maximize abs(chi_minus+chi_plus)/2",
        retained_strength_fraction=0.995,
        scope="map_only",
        seeds_sha256=sha256(
            archive_path.read_bytes() if archive_path is not None else b"[]"
        ).hexdigest(),
        numerical_source_sha256={
            name: sha256((package / name).read_bytes()).hexdigest() for name in sources
        },
    )


def run(args):
    """Resume committed nominal events; never run fabrication perturbations."""
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = [(c1, c2) for c1 in range(36, 129) for c2 in range(36, c1)]
    pairs.sort(key=lambda p: (p[0] % 2 or p[1] % 2, p[1], p[0]))
    if args.pair:
        pairs = [tuple(args.pair)]
        if not 36 <= pairs[0][1] < pairs[0][0] <= 128:
            raise ValueError("Require 36 <= C2 < C1 <= 128 fF")
    if args.limit:
        pairs = pairs[: args.limit]
    policy = configuration(pairs)
    ledger = Ledger(output, policy)
    save_json(output / "settings.json", policy)
    # A fresh checkout starts from algorithmic seeds and newly found neighbors.
    # Historical seed/result archives are not prerequisites for this workflow.
    old = {}
    active = {}
    for phase in ("feasibility_normal", "feasibility_deep", "diagnostic"):
        for record in ledger.rows(phase):
            if phase != "diagnostic" or record["status"] == "recovered_viable":
                active[key((record["c1"], record["c2"]))] = compact(record)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for level in ("normal", "deep"):
                phase = "feasibility_" + level
                ledger.phase = phase
                ledger.progress()

                def tasks(level=level, phase=phase):
                    for pair in pairs:
                        prior = active.get(key(pair))
                        if ledger.done(phase, pair) or (
                            prior and prior["status"] == "validated_pass"
                        ):
                            continue
                        seeds = nearest(
                            pair,
                            [
                                r
                                for r in active.values()
                                if r["status"] == "validated_pass"
                            ],
                            2,
                        )
                        if old.get(key(pair), {}).get("point"):
                            seeds = [old[key(pair)]] + seeds
                        if prior and prior.get("point"):
                            seeds = [prior] + seeds
                        yield dict(
                            pair=pair,
                            level=level,
                            warm=[r["point"] for r in seeds],
                        )

                def accept(task, result, phase=phase):
                    pair = task["pair"]
                    record = ledger.save(phase, pair, result)
                    active[key(pair)] = compact(record)
                    print(
                        json.dumps(
                            dict(phase=phase, pair=pair, status=result["status"])
                        ),
                        flush=True,
                    )

                run_jobs(pool, tasks(), study.search_and_validate, accept, args.workers)
            save_json(output / "feasibility_results.json", list(active.values()))
            ledger.phase = "diagnostic"
            failed = [
                pair
                for pair in pairs
                if active[key(pair)]["status"] != "validated_pass"
            ]
            batch_path = output / "diagnostic_batch.json"
            if batch_path.exists():
                # Preserve the initial batch even after it discovers recoveries.
                failed = [tuple(pair) for pair in load_json(batch_path)["targets"]]
            else:
                save_json(
                    batch_path,
                    dict(step_ff=1, targets=failed, policy_hash=ledger.policy_hash),
                )

            def diagnostic_tasks():
                for pair in failed:
                    if ledger.done("diagnostic", pair):
                        continue
                    seeds = nearest(
                        pair,
                        [r for r in active.values() if r["status"] == "validated_pass"],
                        3,
                    )
                    prior = active[key(pair)]
                    if prior.get("point"):
                        seeds = [prior] + seeds
                    yield dict(
                        pair=pair,
                        warm=[r["point"] for r in seeds],
                        seed_source_ids=[
                            r.get("sample_id", "archive:" + key(pair)) for r in seeds
                        ],
                    )

            def accept_diagnostic(task, result):
                record = ledger.save("diagnostic", task["pair"], result)
                if result["status"] == "recovered_viable":
                    active[key(task["pair"])] = compact(record)
                print(
                    json.dumps(
                        dict(
                            phase="diagnostic",
                            pair=task["pair"],
                            status=result["status"],
                            dominant=result["dominant"],
                        )
                    ),
                    flush=True,
                )

            run_jobs(
                pool,
                diagnostic_tasks(),
                diagnosis.diagnose,
                accept_diagnostic,
                args.workers,
            )
            diagnostics = [compact(r) for r in ledger.rows("diagnostic")]
            save_json(output / "diagnostic_samples.json", diagnostics)

            best = {}
            for pass_number in (1, 2):
                phase = f"strength_{pass_number}"
                for r in ledger.rows(phase):
                    if r["status"] == "validated_pass":
                        k = key((r["c1"], r["c2"]))
                        if k not in best or study.absolute_mean_chi_mhz(
                            r["point"]
                        ) > study.absolute_mean_chi_mhz(best[k]["point"]):
                            best[k] = compact(r)
                ledger.phase = phase

                def strength_tasks(phase=phase, pass_number=pass_number):
                    for pair in pairs:
                        baseline = active[key(pair)]
                        if baseline["status"] not in (
                            "validated_pass",
                            "recovered_viable",
                        ) or ledger.done(phase, pair):
                            continue
                        neighbors = nearest(pair, list(best.values()), 2)
                        warm = [baseline["point"]] + [r["point"] for r in neighbors]
                        starts = []
                        if pass_number == 1:
                            x = np.array(study.x_from_point(baseline["point"]))
                            for shift in (-0.25, 0.25):
                                trial = x.copy()
                                trial[0] += shift
                                starts.append(trial.tolist())
                        yield dict(
                            c1=pair[0],
                            c2=pair[1],
                            mode="maximum",
                            warm=warm,
                            starts=starts,
                        )

                def accept_strength(task, result, phase=phase):
                    pair = (task["c1"], task["c2"])
                    if result["status"] != "validated_pass":
                        q = audit_point(active[key(pair)]["point"])
                        if study.failures(q, require_sign_check=True):
                            raise RuntimeError(f"No audited nominal witness at {pair}")
                        result.update(
                            status="validated_pass",
                            point=q,
                            metrics=metrics(q),
                            failures=[],
                            maximum_search_fallback=True,
                        )
                    record = ledger.save(phase, pair, result)
                    k = key(pair)
                    if k not in best or study.absolute_mean_chi_mhz(
                        record["point"]
                    ) > study.absolute_mean_chi_mhz(best[k]["point"]):
                        best[k] = compact(record)
                    print(
                        json.dumps(
                            dict(
                                phase=phase,
                                pair=pair,
                                strength=study.absolute_mean_chi_mhz(record["point"]),
                            )
                        ),
                        flush=True,
                    )

                run_jobs(
                    pool,
                    strength_tasks(),
                    optimization.solve,
                    accept_strength,
                    args.workers,
                )
                save_json(output / "maximum_results.json", list(best.values()))

            ledger.phase = "continuation"
            selected = {
                key((r["c1"], r["c2"])): compact(r) for r in ledger.rows("continuation")
            }
            rows = defaultdict(list)
            for r in best.values():
                rows[r["c2"]].append(r)
            row_tasks = []
            for c2, records in sorted(rows.items()):
                records.sort(key=lambda r: r["c1"])
                pending = [
                    r for r in records if not ledger.done("continuation", (r["c1"], c2))
                ]
                if pending:
                    preceding = selected.get(key((pending[0]["c1"] - 1, c2)))
                    row_tasks.append(
                        dict(
                            points=pending,
                            previous=preceding["point"] if preceding else None,
                        )
                    )

            # Parallelize independent rows, but submit only one dependent pixel
            # from a row at a time. Every completed pixel is checkpointed.
            row_iterator = iter(row_tasks)
            pending_rows = {}

            def submit_pixel(row):
                task = dict(points=row["points"][:1], previous=row["previous"])
                pending_rows[pool.submit(continue_row, task)] = row

            for _ in range(args.workers):
                row = next(row_iterator, None)
                if row is not None:
                    submit_pixel(row)
            while pending_rows:
                finished, _ = wait(
                    pending_rows, timeout=45, return_when=FIRST_COMPLETED
                )
                if not finished:
                    print(
                        json.dumps(dict(phase="continuation", waiting=True, utc=utc())),
                        flush=True,
                    )
                for future in finished:
                    row = pending_rows.pop(future)
                    result = future.result()[0]
                    pair = (result["c1"], result["c2"])
                    record = ledger.save("continuation", pair, result)
                    selected[key(pair)] = compact(record)
                    print(json.dumps(dict(phase="continuation", pair=pair)), flush=True)
                    row["previous"] = result["point"]
                    row["points"] = row["points"][1:]
                    if row["points"]:
                        submit_pixel(row)
                    else:
                        following = next(row_iterator, None)
                        if following is not None:
                            submit_pixel(following)
            final = []
            for pair in pairs:
                final.append(selected.get(key(pair), active[key(pair)]))
            save_json(output / "map_results.json", final)
            audit = audit_map(final, diagnostics, best, len(pairs))
            save_json(output / "audit.json", audit)
        export_points(output, final)
        ledger.phase = "map_complete"
        ledger.progress(audit_passed=True, completed_utc=utc())
        print(
            json.dumps(
                dict(
                    map_complete=True, viable=audit["viable_pairs"], output=str(output)
                )
            ),
            flush=True,
        )
    except Exception as error:
        ledger.phase = "stopped_with_error"
        ledger.progress(error=repr(error))
        raise
    finally:
        ledger.close()


def audit_map(rows, diagnostics, best, expected):
    """Verify coverage and every selected witness without changing its inputs."""
    assert len(rows) == expected
    passed = [row for row in rows if row["status"] == "validated_pass"]
    for row in passed:
        point = row["point"]
        assert not study.failures(point, require_sign_check=True)
        assert point["basis"] == asdict(optimization.AUDIT)
        assert point["device"]["C1_shunt_fF"] == row["c1"]
        assert point["device"]["C2_shunt_fF"] == row["c2"]
        assert point["device"]["C1r_fF"] == point["device"]["C2r_fF"]
        maximum = best[key((row["c1"], row["c2"]))]["point"]
        assert study.absolute_mean_chi_mhz(point) >= (
            0.995 * study.absolute_mean_chi_mhz(maximum) - 1e-7
        )
    diagnosed = {key((row["c1"], row["c2"])) for row in diagnostics}
    assert all(
        key((row["c1"], row["c2"])) in diagnosed
        for row in rows
        if row["status"] != "validated_pass"
    )
    return dict(
        passed=True,
        map_pairs=len(rows),
        viable_pairs=len(passed),
        diagnostic_samples=len(diagnostics),
        robustness_samples=0,
        scope="map_only",
    )


def export_points(output, rows):
    """Flat numerical data for inspection; no plotting/report dependencies."""
    columns = [
        "C1_fF",
        "C2_fF",
        "status",
        "C12_fF",
        "C1r_fF",
        "C2r_fF",
        "EJ1_GHz",
        "EJ2_GHz",
        "EC1_MHz",
        "EC2_MHz",
        "operation_GHz",
        "logical_MHz",
        "chi_minus_MHz",
        "chi_plus_MHz",
        "chi_mean_abs_MHz",
        "chi_ratio",
        "H",
        "multilevel_dispersive",
    ]
    with (output / "points.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in sorted(rows, key=lambda r: (r["c1"], r["c2"])):
            p = row.get("point")
            values = [row["c1"], row["c2"], row["status"]]
            if p:
                d = p["device"]
                values += [
                    d["C12_fF"],
                    d["C1r_fF"],
                    d["C2r_fF"],
                    p["ej1_ghz"],
                    p["ej2_ghz"],
                    p["loaded_ec1_mhz"],
                    p["loaded_ec2_mhz"],
                    p["operating_frequency_ghz"],
                    p["splitting_ghz"] * 1000,
                    p["chi_m_ghz"] * 1000,
                    p["chi_p_ghz"] * 1000,
                    study.absolute_mean_chi_mhz(p),
                    p["ratio"],
                    p["hybridization"],
                    p["lambda_max"],
                ]
            writer.writerow(values)
