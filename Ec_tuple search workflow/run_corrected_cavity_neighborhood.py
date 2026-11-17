#!/usr/bin/env python3
"""Rerun the selected 5 MHz 4D neighborhood with corrected cavity provenance."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REPETITION = HERE / "run_sampled_anchor_repetition.py"
DEFAULT_A_RESULTS = (
    HERE
    / "cavity_split_campaign_2026-11-09/readout/direct_ec/rail_a_step2_fr6p05/results.json"
)
DEFAULT_B_RESULTS = (
    HERE
    / "cavity_split_campaign_2026-11-09/readout/direct_ec/rail_b_step2_fr5p95/results.json"
)


def utc() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def offsets(radius_mhz: float = 5.0, step_mhz: int = 2):
    limit = int(math.floor(radius_mhz / step_mhz)) * step_mhz
    values = range(-limit, limit + 1, step_mhz)
    return [
        (a1, a2, b1, b2)
        for a1 in values
        for a2 in values
        for b1 in values
        for b2 in values
        if math.sqrt(a1 * a1 + a2 * a2 + b1 * b1 + b2 * b2)
        <= radius_mhz + 1e-12
    ]


def validated_index(path: Path):
    rows = json.loads(path.read_text())
    return {
        (round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"])): row
        for row in rows
        if row.get("status") == "validated_pass"
    }


def complete(directory: Path) -> bool:
    return (directory / "gate_output/compaction_summary.json").exists()


def summarize(
    output: Path,
    expected: set[tuple[int, int, int, int]],
    center: tuple[int, int, int, int],
):
    rows = []
    for repetition in sorted((output / "repetitions").glob("A_*")):
        selection = json.loads((repetition / "selection.json").read_text())
        a1, a2 = selection["ec_a_mhz"]
        database = repetition / "gate_output/compact_anchor_results.sqlite"
        if not database.exists():
            continue
        connection = sqlite3.connect(database)
        records = connection.execute(
            "SELECT ec_b1_mhz,ec_b2_mhz,status,raw_fidelity,"
            "postselected_fidelity,selected_anchor_id,result_json FROM gate_results"
        ).fetchall()
        connection.close()
        for b1, b2, status, raw, post, anchor, payload in records:
            result = json.loads(payload)
            final = result.get("final") or {}
            channel = result.get("channel_selection") or final.get("channel_selection") or {}
            coupling = (final.get("couplings_GHz") or {}).get("A2_B1")
            key = (round(a1), round(a2), round(b1), round(b2))
            rows.append(
                {
                    "ec_tuple_mhz": list(key),
                    "status": status,
                    "raw_fidelity": raw,
                    "postselected_fidelity": post,
                    "selected_anchor_id": anchor,
                    "logical_source": result.get("logical_source"),
                    "gate_frequency_ghz": final.get("fB_GHz"),
                    "gate_time_ns": final.get("time_ns"),
                    "aggregate_s1": channel.get("s1", final.get("S1")),
                    "maximum_logical_ratio": channel.get(
                        "maximum_logical_ratio", final.get("maximum_logical_ratio")
                    ),
                    "required_interrail_coupling_mhz": (
                        None if coupling is None else 1000.0 * float(coupling)
                    ),
                    "fixed_hardware_pass": _fixed_hardware_pass(repetition, b1, b2),
                }
            )
    found = {tuple(row["ec_tuple_mhz"]) for row in rows}
    sources = Counter(row["logical_source"] for row in rows)
    anchors = Counter(row["selected_anchor_id"] for row in rows)
    hard_pass = [
        row
        for row in rows
        if row["status"] == "calibrated"
        and (row["raw_fidelity"] or 0) > 0.95
        and (row["postselected_fidelity"] or 0) > 0.999
    ]
    value = {
        "schema_version": 1,
        "created_utc": utc(),
        "center_ec_tuple_mhz": list(center),
        "radius_mhz": 5.0,
        "lattice_step_mhz": 2,
        "expected_tuple_count": len(expected),
        "recorded_tuple_count": len(rows),
        "missing_tuples": [list(key) for key in sorted(expected - found)],
        "logical_source_counts": dict(sorted(sources.items(), key=lambda item: str(item[0]))),
        "single_logical_source_throughout": len(sources) == 1 and len(rows) == len(expected),
        "selected_anchor_counts": dict(sorted(anchors.items(), key=lambda item: str(item[0]))),
        "hard_gate_pass_count": len(hard_pass),
        "fixed_hardware_pass_count": sum(row["fixed_hardware_pass"] is True for row in rows),
        "fixed_hardware_fail_count": sum(row["fixed_hardware_pass"] is False for row in rows),
        "rows": sorted(rows, key=lambda row: row["ec_tuple_mhz"]),
    }
    atomic_json(output / "corrected_neighborhood_summary.json", value)
    return value


def _fixed_hardware_pass(repetition: Path, b1: float, b2: float):
    database = repetition / "fixed_hardware_reconciliation/fixed_hardware_results.sqlite"
    if not database.exists():
        return None
    connection = sqlite3.connect(database)
    row = connection.execute(
        "SELECT reconciled_pass FROM reconciliation WHERE ec_b1_mhz=? AND ec_b2_mhz=?",
        (round(b1), round(b2)),
    ).fetchone()
    connection.close()
    return None if row is None else bool(row[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--center-ec-tuple-mhz", type=int, nargs=4,
        default=(154, 208, 216, 252),
        metavar=("EC_A1", "EC_A2", "EC_B1", "EC_B2"),
    )
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rail-a-results", type=Path, default=DEFAULT_A_RESULTS)
    parser.add_argument("--rail-b-results", type=Path, default=DEFAULT_B_RESULTS)
    parser.add_argument("--wait-for-status", type=Path)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.background:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output", str(output),
            "--center-ec-tuple-mhz", *map(str, args.center_ec_tuple_mhz),
            "--parallel", str(args.parallel),
            "--workers", str(args.workers),
            "--rail-a-results", str(args.rail_a_results.resolve()),
            "--rail-b-results", str(args.rail_b_results.resolve()),
        ]
        if args.wait_for_status:
            command.extend(["--wait-for-status", str(args.wait_for_status.resolve())])
        with (output / "supervisor.log").open("a") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        atomic_json(output / "launch.json", {"pid": process.pid, "started_utc": utc(), "command": command})
        print(json.dumps({"pid": process.pid, "output": str(output)}, indent=2))
        return

    if args.wait_for_status:
        while True:
            status = json.loads(args.wait_for_status.read_text())
            counts = status.get("counts", {})
            if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
                break
            atomic_json(output / "queued_status.json", {"updated_utc": utc(), "waiting_for": str(args.wait_for_status), "upstream_counts": counts})
            time.sleep(60)

    center = tuple(args.center_ec_tuple_mhz)
    neighborhood = {
        tuple(center[index] + delta[index] for index in range(4))
        for delta in offsets()
    }
    grouped = defaultdict(list)
    for a1, a2, b1, b2 in sorted(neighborhood):
        grouped[(a1, a2)].append((b1, b2))

    a_index = validated_index(args.rail_a_results.resolve())
    b_index = validated_index(args.rail_b_results.resolve())
    missing_a = sorted(set(grouped) - set(a_index))
    missing_b = sorted({b for values in grouped.values() for b in values} - set(b_index))
    if missing_a or missing_b:
        raise RuntimeError(f"Corrected readout map is missing targets: A={missing_a}, B={missing_b}")

    manifest = {
        "schema_version": 1,
        "created_utc": utc(),
        "center_ec_tuple_mhz": list(center),
        "radius_mhz": 5.0,
        "lattice_step_mhz": 2,
        "tuple_count": len(neighborhood),
        "rail_a_pair_count": len(grouped),
        "rail_a_bare_resonator_ghz": 6.05,
        "rail_b_bare_resonator_ghz": 5.95,
        "rail_a_results": str(args.rail_a_results.resolve()),
        "rail_b_results": str(args.rail_b_results.resolve()),
    }
    atomic_json(output / "campaign_manifest.json", manifest)
    repetitions = output / "repetitions"
    repetitions.mkdir(exist_ok=True)

    jobs = []
    for index, ((a1, a2), b_values) in enumerate(sorted(grouped.items()), 1):
        directory = repetitions / f"A_{index:02d}_ecA_{a1}_{a2}"
        directory.mkdir(exist_ok=True)
        selection = {"selection_id": f"A_{index:02d}", "ec_a_mhz": [a1, a2], "ec_b_mhz": [list(value) for value in b_values]}
        atomic_json(directory / "selection.json", selection)
        subset = [b_index[value] for value in b_values]
        atomic_json(directory / "rail_b_targets_fr5p95.json", subset)
        jobs.append((selection, directory))

    def run(job):
        selection, directory = job
        if complete(directory):
            return selection["selection_id"], "complete"
        command = [
            sys.executable, str(REPETITION),
            "--selection-id", selection["selection_id"],
            "--ec-a-mhz", *map(str, selection["ec_a_mhz"]),
            "--output", str(directory),
            "--workers", str(args.workers),
            "--rail-a-results", str(args.rail_a_results.resolve()),
            "--rail-b-results", str(directory / "rail_b_targets_fr5p95.json"),
            "--rail-a-bare-resonator-ghz", "6.05",
        ]
        with (directory / "repetition.log").open("a") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return selection["selection_id"], "complete"

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            selection, _ = futures[future]
            try:
                future.result()
            except Exception as error:
                atomic_json(output / "failure.json", {"updated_utc": utc(), "selection": selection, "error": repr(error)})
                raise
            counts = Counter("complete" if complete(directory) else "pending" for _, directory in jobs)
            atomic_json(output / "campaign_status.json", {"updated_utc": utc(), "counts": dict(counts)})

    summary = summarize(output, neighborhood, center)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
