"""Bounded, resumable process runner for independent numerical EC tasks.

This module imports only the standard library until a worker starts. It does
not launch calculations on import. A handler receives one JSON task payload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve()
_HANDLER = None
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Windows indexers may briefly hold the destination; preserve atomic writes.
    for attempt in range(101):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 100:
                raise
            time.sleep(0.05)


def axis(spec):
    start, stop, step = (Decimal(str(spec[k])) for k in ("start", "stop", "step"))
    if step <= 0 or stop < start:
        raise ValueError("Invalid axis bounds or step")
    values = []
    while start <= stop:
        values.append(start)
        start += step
    if values[-1] != stop:
        values.append(stop)
    return values


def make_tasks(config):
    defaults = config.get("task_defaults", {})
    tasks = []
    if "tasks" in config and "grid" in config:
        raise ValueError("Specify explicit tasks OR a grid")
    if "tasks" in config:
        payloads = [{**defaults, **item} for item in config["tasks"]]
    elif "grid" in config:
        grid = config["grid"]
        b_values = axis(grid["B_MHz"])
        upper_spec = dict(grid["B_MHz"])
        upper_spec["step"] = str(
            Decimal(str(upper_spec["step"]))
            * Decimal(str(grid.get("upper_step_multiplier", 1)))
        )
        upper = axis(upper_spec)
        pairs = [(b1, b2) for b1 in b_values for b2 in b_values if b2 <= b1]
        pairs += [(b1, b2) for b1 in upper for b2 in upper if b2 > b1]
        logical_axis_name = next(
            (name for name in ("logical_A_GHz", "logical_B_GHz") if name in grid),
            None,
        )
        if logical_axis_name:
            logical_values = axis(grid[logical_axis_name])
            fixed_a1 = Decimal(str(grid["A1_fixed_MHz"])) / 1000
            fixed_a2 = Decimal(str(grid["A2_fixed_MHz"])) / 1000
            base_splits = defaults.get("logical_splittings_GHz", [0.150, 0.090])
            split_index = 0 if logical_axis_name == "logical_A_GHz" else 1
            payloads = [
                {
                    **defaults,
                    "logical_splittings_GHz": [
                        float(logical_value)
                        if split_index == 0
                        else float(base_splits[0]),
                        float(logical_value)
                        if split_index == 1
                        else float(base_splits[1]),
                    ],
                    "ec_GHz": [
                        float(fixed_a1),
                        float(fixed_a2),
                        float(b1 / 1000),
                        float(b2 / 1000),
                    ],
                }
                for logical_value in logical_values
                for b1, b2 in pairs
            ]
        else:
            a_values = axis(grid["A1_MHz"])
            payloads = [
                {
                    **defaults,
                    "ec_GHz": [
                        float(a / 1000),
                        float(Decimal(str(grid["A2_fixed_MHz"])) / 1000),
                        float(b1 / 1000),
                        float(b2 / 1000),
                    ],
                }
                for a in a_values
                for b1, b2 in pairs
            ]
    else:
        raise ValueError("Configuration requires tasks or grid")
    for payload in payloads:
        tasks.append({"id": fingerprint(payload), "payload": payload})
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task payloads")
    return tasks


def initialize_worker(handler):
    global _HANDLER
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    sys.path.insert(0, str(HERE.parents[2]))
    sys.path.insert(0, str(HERE.parents[1]))
    sys.path.insert(0, str(HERE.parent))  # Retained standalone experiment helpers.
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    module, function = handler.split(":", 1)
    _HANDLER = getattr(importlib.import_module(module), function)


def execute_task(task):
    started = time.perf_counter()
    record = {
        "task_id": task["id"],
        "payload": task["payload"],
        "worker_pid": os.getpid(),
    }
    try:
        record.update(status="ok", result=_HANDLER(task["payload"]))
        canonical(record)  # Reject NaN or unserializable values inside the worker.
    except Exception as error:
        record.pop("result", None)
        record.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(limit=12),
        )
    from threadpoolctl import threadpool_info

    record["blas_threads"] = [
        p["num_threads"] for p in threadpool_info() if p.get("user_api") == "blas"
    ]
    record["duration_seconds"] = time.perf_counter() - started
    return record


def load_checkpoints(directory, tasks):
    existing = {}
    for task in tasks:
        path = directory / f"{task['id']}.json"
        if not path.exists():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Unreadable checkpoint: {path}; preserve it and use another output directory"
            ) from error
        if (
            record.get("task_id") != task["id"]
            or record.get("payload") != task["payload"]
        ):
            raise ValueError(f"Checkpoint identity mismatch: {path}")
        if record.get("status") not in ("ok", "failed"):
            raise ValueError(f"Invalid checkpoint status: {path}")
        existing[task["id"]] = record
    return existing


def run(config_path, *, output=None, workers=None, retry_failed=False, dry_run=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported schema_version")
    count = workers if workers is not None else config.get("workers", 4)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("workers must be a positive integer")
    handler = config["handler"]
    if handler.count(":") != 1:
        raise ValueError("handler must have the form module:function")
    tasks = make_tasks(config)
    directory = (
        Path(output) if output else config_path.parent / config["output_directory"]
    )
    directory = directory.resolve()
    # Worker count and output location are execution choices, not physics identity.
    source_paths = sorted(
        list(HERE.parent.rglob("*.py")) + list((HERE.parents[2] / "core").rglob("*.py"))
    )
    source_hash = fingerprint(
        [
            (
                str(p.relative_to(HERE.parents[2])),
                hashlib.sha256(p.read_bytes()).hexdigest(),
            )
            for p in source_paths
        ]
    )
    identity = fingerprint(
        {
            "schema_version": 1,
            "handler": handler,
            "tasks": tasks,
            "source_hash": source_hash,
            "handler_revision": config.get("handler_revision", 1),
        }
    )
    if dry_run:
        print(
            json.dumps(
                {
                    "tasks": len(tasks),
                    "workers": count,
                    "handler": handler,
                    "output_directory": str(directory),
                    "run_hash": identity,
                }
            )
        )
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    points = directory / "points"
    points.mkdir(exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_hash") != identity:
            expected = {t["id"] for t in tasks}
            old_ids = {p.stem for p in points.glob("*.json")}
            extension = (
                config.get("extend_completed_run", False)
                and previous.get("status") == "complete"
                and previous.get("source_hash") == source_hash
                and previous.get("handler") == handler
                and previous.get("handler_revision", 1)
                == config.get("handler_revision", 1)
                and old_ids <= expected
                and len(old_ids) == previous.get("total")
            )
            if not extension:
                raise ValueError(
                    "Output belongs to a different configuration; choose a new output directory"
                )
    elif any(points.glob("*.json")):
        raise ValueError(
            "Checkpoints exist without a manifest; choose a new output directory"
        )
    records = load_checkpoints(points, tasks)
    pending = [
        task
        for task in tasks
        if task["id"] not in records
        or (retry_failed and records[task["id"]]["status"] == "failed")
    ]
    if retry_failed:
        for task in pending:
            records.pop(task["id"], None)
    manifest = {
        "run_hash": identity,
        "source_hash": source_hash,
        "handler": handler,
        "handler_revision": config.get("handler_revision", 1),
        "workers": count,
        "total": len(tasks),
        "resumed": len(records),
        "status": "running",
    }
    started = time.perf_counter()
    counts = {
        status: sum(r["status"] == status for r in records.values())
        for status in ("ok", "failed")
    }

    def checkpoint(status="running"):
        manifest.update(
            status=status,
            completed=len(records),
            succeeded=counts["ok"],
            failed=counts["failed"],
            elapsed_session_seconds=time.perf_counter() - started,
        )
        atomic_json(manifest_path, manifest)

    checkpoint()
    print(
        f"{len(tasks)} tasks; {len(records)} cached; {len(pending)} pending; {count} workers",
        flush=True,
    )
    if not pending:
        checkpoint("complete")
        return 1 if manifest["failed"] else 0
    for name in THREAD_VARIABLES:
        os.environ[name] = (
            "1"  # Set before spawned interpreters import numerical packages.
        )
    # Resolve handler errors once instead of repeatedly crashing pool initializers.
    initialize_worker(handler)
    if not callable(_HANDLER):
        raise ValueError("Configured handler is not callable")
    context = mp.get_context("spawn")
    pool = context.Pool(count, initializer=initialize_worker, initargs=(handler,))
    results = queue.Queue()
    iterator = iter(pending)
    active = {}
    interval = float(config.get("progress_interval_seconds", 5))
    timeout = float(config.get("task_timeout_seconds", 300))
    last_progress = time.perf_counter()
    max_in_flight = max(1, count * 2)

    def fill():
        while len(active) < max_in_flight:
            task = next(iterator, None)
            if task is None:
                break
            active[task["id"]] = time.perf_counter()
            pool.apply_async(
                execute_task, (task,), callback=lambda result: results.put(result)
            )

    try:
        fill()
        while active:
            try:
                record = results.get(timeout=0.25)
                active.pop(record["task_id"])
                atomic_json(points / f"{record['task_id']}.json", record)
                records[record["task_id"]] = record
                counts[record["status"]] += 1
                checkpoint()
                fill()
            except queue.Empty:
                pass
            now = time.perf_counter()
            if any(now - sent > timeout for sent in active.values()):
                raise TimeoutError(
                    "An in-flight task exceeded task_timeout_seconds; resume to retry unfinished tasks"
                )
            if now - last_progress >= interval:
                done = len(records) - manifest["resumed"]
                rate = done / max(now - started, 1e-9)
                eta = (len(tasks) - len(records)) / rate if rate else None
                print(
                    f"{len(records)}/{len(tasks)} completed; {manifest['failed']} failed; "
                    f"ETA {eta:.1f}s"
                    if eta is not None
                    else "Waiting for first result",
                    flush=True,
                )
                last_progress = now
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        checkpoint("interrupted")
        print(
            "Interrupted; completed tasks are checkpointed. Re-run the same command to resume.",
            flush=True,
        )
        return 130
    except BaseException:
        pool.terminate()
        pool.join()
        checkpoint("interrupted")
        raise
    checkpoint("complete")
    return 1 if manifest["failed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(
            args.config,
            output=args.output,
            workers=args.workers,
            retry_failed=args.retry_failed,
            dry_run=args.dry_run,
        )
    except (ValueError, OSError, TimeoutError, ImportError, AttributeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
