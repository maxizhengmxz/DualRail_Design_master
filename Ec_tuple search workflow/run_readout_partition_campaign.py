#!/usr/bin/env python3
"""Run one ordered half-plane of the fixed-rail-A EC-tuple workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
A_RESULTS = HERE / "readout/direct_ec/rail_a_step2/results.json"
REPETITION = HERE / "run_sampled_anchor_repetition.py"


def utc():
    return datetime.now(UTC).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition(side, a_results=A_RESULTS):
    rows = json.loads(a_results.read_text())
    selected = []
    for row in rows:
        if row.get("status") != "validated_pass":
            continue
        ec1 = int(row["ec1_target_mhz"])
        ec2 = int(row["ec2_target_mhz"])
        keep = ec1 > ec2 if side == "gt" else ec1 < ec2
        if keep:
            selected.append((ec1, ec2))
    selected.sort()
    if len(selected) != len(set(selected)):
        raise RuntimeError("Duplicate ordered rail-A EC pairs in readout results")
    if not selected:
        raise RuntimeError(f"No validated rail-A points on side {side}")
    return [
        {
            "selection_id": f"A_{side.upper()}_{index:04d}",
            "ec_a_mhz": [ec1, ec2],
        }
        for index, (ec1, ec2) in enumerate(selected, 1)
    ]


def completed(directory):
    return (directory / "gate_output/compaction_summary.json").exists()


def terminal_anchor_screening_failure(directory):
    """Retain deterministic pre-gate screening outcomes across resumes.

    Interrupted repetitions have a partial log but do not contain either of
    these terminal messages, so they remain pending and can reuse checkpoints.
    Operational/subprocess failures are likewise retried rather than frozen.
    """
    log = directory / "repetition.log"
    if not log.exists():
        return False
    text = log.read_text(errors="replace")
    return any(
        message in text
        for message in (
            "No connected readout-frequency component",
            "Largest component has fewer than 5 distinct validated samples",
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("gt", "lt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rail-a-results", type=Path, default=A_RESULTS)
    parser.add_argument("--rail-b-results", type=Path, default=HERE/"readout/direct_ec/rail_b_step2/results.json")
    parser.add_argument("--rail-a-bare-resonator-ghz", type=float, default=6.0)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.parallel < 1 or args.workers < 1:
        raise ValueError("parallel and workers must be positive")

    output = args.output.resolve()
    args.rail_a_results=args.rail_a_results.resolve();args.rail_b_results=args.rail_b_results.resolve()
    selections = partition(args.side,args.rail_a_results)
    digest = source_sha256(args.rail_a_results)
    manifest_value = {
        "schema_version": 1,
        "created_utc": utc(),
        "side": args.side,
        "rule": "ec_a1_mhz > ec_a2_mhz"
        if args.side == "gt"
        else "ec_a1_mhz < ec_a2_mhz",
        "source_results": str(args.rail_a_results),
        "rail_b_results": str(args.rail_b_results),
        "rail_a_bare_resonator_ghz": args.rail_a_bare_resonator_ghz,
        "source_sha256": digest,
        "selection_count": len(selections),
        "selections": selections,
    }

    if args.dry_run:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "side": args.side,
                    "selection_count": len(selections),
                    "first": selections[0],
                    "last": selections[-1],
                    "source_sha256": digest,
                    "parallel": args.parallel,
                    "workers": args.workers,
                },
                indent=2,
            )
        )
        return 0

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "partition_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        stable_keys = ("schema_version", "side", "rule", "source_sha256", "selections")
        if any(previous.get(key) != manifest_value.get(key) for key in stable_keys):
            raise RuntimeError(
                "Existing partition manifest differs; use a fresh output directory"
            )
    else:
        atomic_json(manifest, manifest_value)

    launch = output / "launch.json"
    if args.background:
        if launch.exists():
            previous = json.loads(launch.read_text())
            try:
                os.kill(previous["pid"], 0)
                raise RuntimeError("Partition supervisor is already running")
            except ProcessLookupError:
                pass
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--side",
            args.side,
            "--output",
            str(output),
            "--parallel",
            str(args.parallel),
            "--workers",
            str(args.workers),
            "--rail-a-results",str(args.rail_a_results),
            "--rail-b-results",str(args.rail_b_results),
            "--rail-a-bare-resonator-ghz",str(args.rail_a_bare_resonator_ghz),
        ]
        with (output / "supervisor.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        atomic_json(
            launch,
            {
                "pid": process.pid,
                "started_utc": utc(),
                "command": command,
                "side": args.side,
                "selection_count": len(selections),
            },
        )
        print(
            json.dumps(
                {
                    "pid": process.pid,
                    "campaign": str(output),
                    "selection_count": len(selections),
                    "parallel": args.parallel,
                    "workers_per_repetition": args.workers,
                },
                indent=2,
            )
        )
        return 0

    jobs = []
    for selection in selections:
        ec = selection["ec_a_mhz"]
        directory = output / (
            f"{selection['selection_id']}_ecA_{ec[0]}_{ec[1]}"
        )
        state = (
            "complete" if completed(directory)
            else "failed" if terminal_anchor_screening_failure(directory)
            else "pending"
        )
        jobs.append(
            {
                "selection": selection,
                "directory": directory,
                "state": state,
            }
        )

    active = {}
    started = time.monotonic()
    status_path = output / "campaign_status.json"

    def snapshot():
        counts = {
            state: sum(job["state"] == state for job in jobs)
            for state in ("pending", "running", "complete", "failed")
        }
        atomic_json(
            status_path,
            {
                "updated_utc": utc(),
                "side": args.side,
                "source_sha256": digest,
                "parallel_repetitions": args.parallel,
                "workers_per_repetition": args.workers,
                "elapsed_session_seconds": time.monotonic() - started,
                "counts": counts,
                "jobs": [
                    {
                        "selection_id": job["selection"]["selection_id"],
                        "ec_a_mhz": job["selection"]["ec_a_mhz"],
                        "state": job["state"],
                        "directory": str(job["directory"]),
                    }
                    for job in jobs
                ],
            },
        )

    snapshot()
    while any(job["state"] not in ("complete", "failed") for job in jobs):
        for job in jobs:
            if len(active) >= args.parallel:
                break
            if job["state"] != "pending":
                continue
            selection = job["selection"]
            ec = selection["ec_a_mhz"]
            job["directory"].mkdir(parents=True, exist_ok=True)
            log = (job["directory"] / "repetition.log").open("a")
            command = [
                sys.executable,
                str(REPETITION),
                "--selection-id",
                selection["selection_id"],
                "--ec-a-mhz",
                *map(str, ec),
                "--output",
                str(job["directory"]),
                "--workers",
                str(args.workers),
                "--rail-a-results",str(args.rail_a_results),
                "--rail-b-results",str(args.rail_b_results),
                "--rail-a-bare-resonator-ghz",str(args.rail_a_bare_resonator_ghz),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            job.update(state="running", process=process, log=log)
            active[process.pid] = job
            snapshot()

        time.sleep(5)
        for pid, job in list(active.items()):
            code = job["process"].poll()
            if code is None:
                continue
            job["log"].close()
            active.pop(pid)
            job["state"] = "complete" if code == 0 and completed(job["directory"]) else "failed"
            snapshot()

    snapshot()
    return 1 if any(job["state"] == "failed" for job in jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
