#!/usr/bin/env python3
"""Run one maximum-S1 diagnostic rail-B gate grid."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dualrail_worst_case_matplotlib")
sys.path.insert(0, str(HERE))
from run_sampled_anchor_repetition import anchors
from readout_workflow.physical_frequency_bounds import require_bare_resonator


def command(arguments: list[str], log: Path, cwd: Path = ROOT) -> None:
    with log.open("a") as stream:
        subprocess.run(
            arguments,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--ec-a-mhz", nargs=2, type=float, required=True)
    parser.add_argument("--rail-a-results", type=Path, required=True)
    parser.add_argument("--rail-b-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--anchor-sampling",
        choices=("fractional", "random"),
        default="fractional",
    )
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-rail-a-resonator-ghz", type=float, default=6.05)
    parser.add_argument("--expected-rail-b-resonator-ghz", type=float, default=5.95)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scan = output / "readout_frequency_scan"
    frequency_results = scan / "results.json"
    if not frequency_results.exists():
        command(
            [
                sys.executable,
                "-m",
                "readout_workflow.physical_frequency_bounds",
                "--ec-mhz",
                *map(str, args.ec_a_mhz),
                "--frequency-ghz",
                "4",
                "8",
                "0.025",
                "--logical-frequency-mhz",
                "140",
                "160",
                "--bare-resonator-ghz",
                str(args.expected_rail_a_resonator_ghz),
                "--warm-results",
                str(args.rail_a_results.resolve()),
                "--output",
                str(scan),
            ],
            output / "readout_scan.log",
        )
    rows = json.loads(frequency_results.read_text())
    for row in rows:
        if row.get("point"):
            require_bare_resonator(
                row["point"], args.expected_rail_a_resonator_ghz
            )
    rail_b_rows = json.loads(args.rail_b_results.resolve().read_text())
    for row in rail_b_rows:
        if row.get("status") == "validated_pass" and row.get("point"):
            require_bare_resonator(
                row["point"], args.expected_rail_b_resonator_ghz
            )
    if args.anchor_sampling == "random" and args.random_seed is None:
        raise ValueError("Random anchor sampling requires --random-seed")
    frequencies, component, count = anchors(
        rows,
        sampling=args.anchor_sampling,
        random_seed=args.random_seed,
    )
    low, high = component
    selection = {
        "selection_id": args.selection_id,
        "ec_a_mhz": args.ec_a_mhz,
        "sampling_policy": args.anchor_sampling,
        "sampling_fractions": [0.1, 0.3, 0.5, 0.7, 0.9]
        if args.anchor_sampling == "fractional"
        else None,
        "random_seed": args.random_seed if args.anchor_sampling == "random" else None,
        "largest_sampled_component_ghz": component,
        "component_sample_count": count,
        "eligible_anchor_frequency_ghz": [low + 0.1 * (high - low), low + 0.9 * (high - low)]
        if args.anchor_sampling == "fractional"
        else [low, high],
        "selected_component_fractions": [
            (frequency - low) / (high - low) for frequency in frequencies
        ],
        "selected_validated_frequencies_ghz": frequencies,
        "anchor_ranking": "maximum_S1_worst_case",
        "within_anchor_channel_policy": "minimum_aggregate_S1_at_common_intended_g",
        "fidelity_fallback_count": 1,
        "source_frequency_results": str(frequency_results),
        "rail_a_warm_results": str(args.rail_a_results.resolve()),
        "rail_a_bare_resonator_ghz": args.expected_rail_a_resonator_ghz,
        "rail_b_bare_resonator_ghz": args.expected_rail_b_resonator_ghz,
    }
    selection_path = output / "anchor_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("Saved random-anchor selection differs; use a fresh output")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")

    config = output / "config.json"
    if not config.exists():
        command(
            [
                sys.executable,
                str(HERE / "prepare_fixed_a_anchor_campaign.py"),
                "--frequency-results",
                str(frequency_results),
                "--rail-b-results",
                str(args.rail_b_results.resolve()),
                "--frequencies-ghz",
                *map(str, frequencies),
                "--output",
                str(output),
                "--workers",
                str(args.workers),
                "--rail-b-bare-frequency-ghz",
                "4",
                "5.5",
                "--anchor-ranking",
                "maximum_S1_worst_case",
                "--fidelity-fallback-count",
                "1",
                "--fixed-a-bounds-from-anchors",
            ],
            output / "prepare.log",
        )
    manifest = output / "gate_output/manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("status") != "complete":
        command(
            [sys.executable, "-m", "xm_cz.runner", str(config)],
            output / "gate.log",
            ROOT / "2Q-gate_XM",
        )
    summary = output / "gate_output/anchor_campaign_summary.json"
    if not summary.exists():
        command(
            [sys.executable, str(HERE / "postprocess_anchor_campaign.py"), str(output)],
            output / "post.log",
        )
    compact = output / "gate_output/compaction_summary.json"
    if not compact.exists():
        command(
            [
                sys.executable,
                str(HERE / "compact_anchor_campaign.py"),
                str(output),
                "--prune",
            ],
            output / "post.log",
        )


if __name__ == "__main__":
    main()
