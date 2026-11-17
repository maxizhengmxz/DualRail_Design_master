#!/usr/bin/env python3
"""Prepare and launch the corrected dual-auxiliary rail-B grid comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "corrected_cavity_strict_tuple_A154_208_B216_252_2026-11-13/config.json"
B_RESULTS = (
    HERE / "cavity_split_campaign_2026-11-09/readout/direct_ec"
    / "rail_b_step2_fr5p95/results.json"
)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def utc() -> str:
    return datetime.now(UTC).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference = json.loads(REFERENCE.read_text())
    anchors = reference["task_defaults"]["rail_A_anchors"]
    cavity_values = {round(anchor["bare_resonator_frequency_GHz"], 9) for anchor in anchors}
    if cavity_values != {6.05}:
        raise RuntimeError(f"Expected corrected 6.05 GHz rail-A anchors, found {cavity_values}")
    b_rows = json.loads(B_RESULTS.read_text())
    tasks = [
        {
            "ec_GHz": [
                0.154,
                0.208,
                float(row["ec1_target_mhz"]) / 1000.0,
                float(row["ec2_target_mhz"]) / 1000.0,
            ]
        }
        for row in b_rows
    ]
    defaults = dict(reference["task_defaults"])
    defaults.update(
        {
            "rail_A_anchors": anchors,
            "resonance_channels": ["0020", "0200"],
            "fidelity_fallback_count": 10,
            "maximum_S1": 1e99,
            "maximum_source_score": 1e99,
        }
    )
    config = {
        "schema_version": 1,
        "handler": "xm_cz.dual_auxiliary_evaluator:evaluate_dual_auxiliary_anchor_set",
        "handler_revision": 1,
        "workers": args.workers,
        "progress_interval_seconds": 10,
        "task_timeout_seconds": 900,
        "output_directory": "gate_output",
        "task_defaults": defaults,
        "tasks": tasks,
        "gate_domain_policy": "all ordered rail-B EC targets independent of readout",
        "selection_policy": (
            "screen both 0020 and 0200 for every corrected rail-A anchor; "
            "rank verified channel minima globally by aggregate S1 at common intended g; "
            "use calibrated-fidelity fallback across the combined candidate list"
        ),
        "readout_sources": {
            "rail_a_frequency_results": reference["readout_sources"]["rail_a_frequency_results"],
            "rail_b_direct_ec_results": str(B_RESULTS.resolve()),
        },
        "cavity_provenance_GHz": {"rail_A": 6.05, "rail_B": 5.95},
        "comparison_scope": {
            "ec_a_mhz": [154, 208],
            "rail_b_target_count": len(tasks),
            "auxiliaries": {"0020": "B20", "0200": "A02"},
        },
    }
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Existing configuration differs; use a fresh output directory")
    atomic_json(config_path, config)
    atomic_json(
        output / "campaign_manifest.json",
        {
            "schema_version": 1,
            "created_utc": utc(),
            "config": str(config_path),
            "reference_config": str(REFERENCE.resolve()),
            "rail_b_results": str(B_RESULTS.resolve()),
            "rail_b_target_count": len(tasks),
            "workers": args.workers,
            "cavity_provenance_GHz": {"rail_A": 6.05, "rail_B": 5.95},
        },
    )
    command = [sys.executable, "-m", "xm_cz.runner", str(config_path)]
    if args.background:
        with (output / "gate.log").open("a") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT / "2Q-gate_XM",
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        atomic_json(
            output / "launch.json",
            {"pid": process.pid, "started_utc": utc(), "command": command},
        )
        print(json.dumps({"pid": process.pid, "output": str(output), "tasks": len(tasks)}, indent=2))
    else:
        subprocess.run(command, cwd=ROOT / "2Q-gate_XM", check=True)


if __name__ == "__main__":
    main()
