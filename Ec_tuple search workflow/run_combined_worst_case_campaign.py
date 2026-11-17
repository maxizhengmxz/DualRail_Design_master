#!/usr/bin/env python3
"""Run maximum-S1 anchor diagnostics for successful combined rail-A points."""
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
REPETITION = HERE / "run_worst_case_anchor_repetition.py"
DEFAULT_SELECTIONS = (
    HERE
    / "combined_overlay_selection_2026-11-12"
    / "combined_overlay_best_configurations.json"
)
B_RESULTS = {
    "original_6p00_6p00": (
        HERE
        / "cavity_split_campaign_2026-11-09/readout/direct_ec"
        / "rail_b_step2_fr5p95/results.json"
    ),
    "split_6p05_5p95": (
        HERE
        / "cavity_split_campaign_2026-11-09/readout/direct_ec"
        / "rail_b_step2_fr5p95/results.json"
    ),
}
A_RESULTS = (
    HERE
    / "cavity_split_campaign_2026-11-09/readout/direct_ec"
    / "rail_a_step2_fr6p05/results.json"
)


def utc() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def derived_seed(master: int, campaign: str, selection_id: str, ec: list[int]) -> int:
    text = f"{master}:{campaign}:{selection_id}:{ec[0]}:{ec[1]}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, default=20261112)
    parser.add_argument(
        "--anchor-sampling",
        choices=("fractional", "random"),
        default="fractional",
    )
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.parallel < 1 or args.workers < 1:
        raise ValueError("parallel and workers must be positive")
    source = json.loads(args.selections.resolve().read_text())
    jobs = []
    for row in source["selections"]:
        campaign = row["campaign"]
        ec = [int(value) for value in row["ec_a_mhz"]]
        seed = derived_seed(args.master_seed, campaign, row["selection_id"], ec)
        directory = (
            args.output.resolve()
            / campaign
            / f"{row['selection_id']}_ecA_{ec[0]}_{ec[1]}"
        )
        jobs.append(
            {
                "campaign": campaign,
                "selection_id": row["selection_id"],
                "ec_a_mhz": ec,
                "source_repetition": row["source_repetition"],
                "rail_a_results": str(A_RESULTS.resolve()),
                "rail_b_results": str(B_RESULTS[campaign].resolve()),
                "random_seed": seed,
                "directory": str(directory),
            }
        )
    manifest_value = {
        "schema_version": 1,
        "created_utc": utc(),
        "source_selections": str(args.selections.resolve()),
        "master_seed": args.master_seed,
        "anchor_sampling": args.anchor_sampling,
        "anchor_sampling_fractions": [0.1, 0.3, 0.5, 0.7, 0.9]
        if args.anchor_sampling == "fractional"
        else None,
        "anchor_ranking": "maximum_S1_worst_case",
        "within_anchor_channel_policy": "minimum_aggregate_S1_at_common_intended_g",
        "fidelity_fallback_count": 1,
        "rail_a_bare_resonator_ghz": 6.05,
        "rail_b_bare_resonator_ghz": 5.95,
        "readout_policy": "rerun rail-A 4-8 GHz readout scan at every EC-A candidate",
        "pipeline_stages": [
            "rail_A_readout_frequency_scan",
            f"{args.anchor_sampling}_anchor_sampling",
            "maximum_S1_anchor_gate_grid",
            "independent_rail_B_readout_overlay",
            "best_candidate_selection_and_campaign_plots",
        ],
        "hardware_reconciliation": "excluded",
        "job_count": len(jobs),
        "jobs": jobs,
    }
    if args.dry_run:
        print(json.dumps(manifest_value, indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "campaign_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        for key in ("source_selections", "master_seed", "anchor_sampling", "anchor_ranking", "jobs"):
            if old.get(key) != manifest_value.get(key):
                raise RuntimeError("Existing campaign manifest differs; use a fresh output")
    else:
        atomic_json(manifest, manifest_value)
    if args.background:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--selections",
            str(args.selections.resolve()),
            "--output",
            str(output),
            "--master-seed",
            str(args.master_seed),
            "--anchor-sampling",
            args.anchor_sampling,
            "--parallel",
            str(args.parallel),
            "--workers",
            str(args.workers),
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
            output / "launch.json",
            {"pid": process.pid, "started_utc": utc(), "command": command},
        )
        print(json.dumps({"pid": process.pid, "output": str(output), "jobs": len(jobs)}, indent=2))
        return
    for job in jobs:
        directory = Path(job["directory"])
        compact = directory / "gate_output/compaction_summary.json"
        job["state"] = "complete" if compact.exists() else "pending"
    active = {}
    status_path = output / "campaign_status.json"

    def snapshot() -> None:
        counts = {
            state: sum(job["state"] == state for job in jobs)
            for state in ("pending", "running", "complete", "failed")
        }
        atomic_json(
            status_path,
            {
                "updated_utc": utc(),
                "parallel_repetitions": args.parallel,
                "workers_per_repetition": args.workers,
                "counts": counts,
                "jobs": [
                    {
                        key: job[key]
                        for key in (
                            "campaign",
                            "selection_id",
                            "ec_a_mhz",
                            "random_seed",
                            "directory",
                            "state",
                        )
                    }
                    for job in jobs
                ],
            },
        )

    snapshot()
    while any(job["state"] in {"pending", "running"} for job in jobs):
        for job in jobs:
            if len(active) >= args.parallel:
                break
            if job["state"] != "pending":
                continue
            directory = Path(job["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "repetition.log").open("a")
            command = [
                sys.executable,
                str(REPETITION),
                "--selection-id",
                job["selection_id"],
                "--ec-a-mhz",
                *map(str, job["ec_a_mhz"]),
                "--rail-a-results",
                job["rail_a_results"],
                "--rail-b-results",
                job["rail_b_results"],
                "--output",
                job["directory"],
                "--anchor-sampling",
                args.anchor_sampling,
                "--workers",
                str(args.workers),
            ]
            if args.anchor_sampling == "random":
                command.extend(["--random-seed", str(job["random_seed"])])
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            job.update(state="running", process=process, log=log)
            active[process.pid] = job
        snapshot()
        time.sleep(5)
        for pid, job in list(active.items()):
            code = job["process"].poll()
            if code is None:
                continue
            job["log"].close()
            job["state"] = "complete" if code == 0 else "failed"
            del active[pid]
            snapshot()
    snapshot()
    subprocess.run(
        [
            sys.executable,
            str(HERE / "postprocess_worst_case_campaign.py"),
            str(output),
            "--rail-a-results",
            str(A_RESULTS.resolve()),
            "--rail-b-results",
            str(B_RESULTS["split_6p05_5p95"].resolve()),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
