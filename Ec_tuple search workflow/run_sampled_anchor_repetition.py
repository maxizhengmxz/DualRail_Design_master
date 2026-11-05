#!/usr/bin/env python3
"""Run one EC-A repetition from readout frequency scan through compaction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DIRECT = HERE / "readout/direct_ec/rail_a_step2/results.json"


def command(args, log, cwd=ROOT):
    with log.open("a") as stream:
        subprocess.run(
            args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True
        )


def anchors(rows, step=0.025, fractions=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Choose readout-interior witnesses before aggregate-S1 gate selection.

    The percentages refer to the largest connected sampled feasible interval.
    Keep the actual selected samples within its inner 10-90 percent, even when
    nearest-grid rounding merges requested fractions. A sparse scan must be
    refined instead of adding endpoint anchors. This is a sampled design margin,
    not a certificate of fixed-hardware tuning or fabrication tolerance.
    """
    if not isfinite(step) or step <= 0:
        raise ValueError("Frequency scan step must be finite and positive")
    if (
        not fractions
        or len(set(fractions)) != len(fractions)
        or any(not isfinite(value) or not 0.1 <= value <= 0.9 for value in fractions)
    ):
        raise ValueError("Anchor fractions must be distinct and within 0.1-0.9")
    good = sorted(
        (r for r in rows if r["status"] == "validated_pass"),
        key=lambda r: r["target_frequency_ghz"],
    )
    groups = []
    group = []
    for row in good:
        if (
            group
            and row["target_frequency_ghz"] - group[-1]["target_frequency_ghz"]
            > step * 1.5
        ):
            groups.append(group)
            group = []
        group.append(row)
    if group:
        groups.append(group)
    if not groups:
        raise RuntimeError("No connected readout-frequency component")
    group = max(
        groups,
        key=lambda g: (
            g[-1]["target_frequency_ghz"] - g[0]["target_frequency_ghz"],
            len(g),
        ),
    )
    low, high = group[0]["target_frequency_ghz"], group[-1]["target_frequency_ghz"]
    width = high - low
    interior = {
        row["target_frequency_ghz"]
        for row in group
        if low + 0.1 * width - 1e-12
        <= row["target_frequency_ghz"]
        <= low + 0.9 * width + 1e-12
        and low < row["target_frequency_ghz"] < high
    }
    if len(interior) < len(fractions):
        raise RuntimeError(
            f"Largest component has fewer than {len(fractions)} distinct validated "
            "samples inside its 10-90% band. Refine the frequency scan."
        )
    chosen = []
    for fraction in fractions:
        target = low + fraction * width
        selected = min(
            interior, key=lambda frequency: (abs(frequency - target), frequency)
        )
        chosen.append(selected)
        interior.remove(selected)
    return sorted(chosen), (low, high), len(group)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection-id", required=True)
    p.add_argument("--ec-a-mhz", type=float, nargs=2, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scan = out / "readout_frequency_scan"
    results = scan / "results.json"
    if not results.exists():
        command(
            [
                sys.executable,
                "-m",
                "readout_workflow.physical_frequency_bounds",
                "--ec-mhz",
                *map(str, a.ec_a_mhz),
                "--frequency-ghz",
                "4",
                "8",
                "0.025",
                "--logical-frequency-mhz",
                "140",
                "160",
                "--warm-results",
                str(DIRECT),
                "--output",
                str(scan),
            ],
            out / "readout_scan.log",
        )
    frequencies, component, count = anchors(json.loads(results.read_text()))
    config = out / "config.json"
    if config.exists():
        saved = json.loads(config.read_text())["task_defaults"]["rail_A_anchors"]
        saved_frequencies = sorted(
            anchor["readout_operating_frequency_GHz"] for anchor in saved
        )
        if len(saved_frequencies) != len(frequencies) or any(
            abs(old - new) > 1e-7
            for old, new in zip(saved_frequencies, frequencies, strict=True)
        ):
            raise RuntimeError(
                "Cached anchors differ from the 10-90% selection. Use a fresh output directory."
            )
    low, high = component
    (out / "anchor_selection.json").write_text(
        json.dumps(
            {
                "selection_id": a.selection_id,
                "ec_a_mhz": a.ec_a_mhz,
                "largest_sampled_component_ghz": component,
                "component_sample_count": count,
                "sampling_fractions": [0.1, 0.3, 0.5, 0.7, 0.9],
                "allowed_anchor_frequency_ghz": [
                    low + 0.1 * (high - low),
                    low + 0.9 * (high - low),
                ],
                "selected_component_fractions": [
                    (frequency - low) / (high - low) for frequency in frequencies
                ],
                "selected_validated_frequencies_ghz": frequencies,
            },
            indent=2,
        )
        + "\n"
    )
    if not config.exists():
        command(
            [
                sys.executable,
                str(HERE / "prepare_fixed_a_anchor_campaign.py"),
                "--frequency-results",
                str(results),
                "--frequencies-ghz",
                *map(str, frequencies),
                "--rail-b-bare-frequency-ghz",
                "4",
                "5.5",
                "--output",
                str(out),
                "--workers",
                str(a.workers),
            ],
            out / "prepare.log",
        )
    manifest = out / "gate_output/manifest.json"
    if (
        not manifest.exists()
        or json.loads(manifest.read_text()).get("status") != "complete"
    ):
        command(
            [sys.executable, "-m", "xm_cz.runner", str(config)],
            out / "gate.log",
            ROOT / "2Q-gate_XM",
        )
    if not (out / "gate_output/compaction_summary.json").exists():
        command(
            [
                sys.executable,
                str(HERE / "run_anchor_post_pipeline.py"),
                str(out),
                "--workers",
                str(a.workers),
                "--compact-and-prune",
            ],
            out / "post.log",
        )


if __name__ == "__main__":
    main()
