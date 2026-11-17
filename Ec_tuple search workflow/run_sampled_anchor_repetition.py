#!/usr/bin/env python3
"""Run one EC-A repetition from readout frequency scan through compaction."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from readout_workflow.physical_frequency_bounds import require_bare_resonator
DIRECT = HERE / "readout/direct_ec/rail_a_step2/results.json"


def command(args, log, cwd=ROOT):
    with log.open("a") as stream:
        subprocess.run(
            args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True
        )


def anchors(
    rows,
    step=0.025,
    fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
    sampling="fractional",
    random_seed=None,
):
    """Choose readout-interior witnesses before aggregate-S1 gate selection.

    The percentages refer to the largest connected sampled feasible interval.
    Keep the actual selected samples within its inner 10-90 percent, even when
    nearest-grid rounding merges requested fractions. A sparse scan must be
    refined instead of adding endpoint anchors. This is a sampled design margin,
    not a certificate of fixed-hardware tuning or fabrication tolerance.
    """
    if not isfinite(step) or step <= 0:
        raise ValueError("Frequency scan step must be finite and positive")
    if sampling not in {"fractional", "random"}:
        raise ValueError("Anchor sampling must be fractional or random")
    if (
        sampling == "fractional"
        and (
        not fractions
        or len(set(fractions)) != len(fractions)
        or any(not isfinite(value) or not 0.1 <= value <= 0.9 for value in fractions)
        )
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
    if sampling == "random":
        count = len(fractions)
        if len(group) < count:
            raise RuntimeError(
                f"Largest component has fewer than {count} distinct validated samples. "
                "Refine the frequency scan."
            )
        frequencies = [row["target_frequency_ghz"] for row in group]
        return sorted(random.Random(random_seed).sample(frequencies, count)), (low, high), len(group)
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
    p.add_argument("--target-g-mhz", type=float, default=5.0)
    p.add_argument("--rail-a-results", type=Path, default=DIRECT)
    p.add_argument("--rail-b-results", type=Path, default=HERE/"readout/direct_ec/rail_b_step2/results.json")
    p.add_argument("--only-ec-b-mhz", type=float, nargs=2, metavar=("EC_B1", "EC_B2"))
    p.add_argument("--rail-a-bare-resonator-ghz", type=float, default=6.0)
    p.add_argument("--anchor-sampling", choices=("fractional", "random"), default="fractional")
    p.add_argument("--anchor-random-seed", type=int)
    p.add_argument(
        "--anchor-ranking",
        choices=("minimum_S1", "maximum_S1_worst_case"),
        default="minimum_S1",
    )
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
                "--bare-resonator-ghz",
                str(a.rail_a_bare_resonator_ghz),
                "--warm-results",
                str(a.rail_a_results),
                "--output",
                str(scan),
            ],
            out / "readout_scan.log",
        )
    frequency_rows = json.loads(results.read_text())
    for row in frequency_rows:
        if row.get("point"):
            require_bare_resonator(row["point"], a.rail_a_bare_resonator_ghz)
    if a.anchor_sampling == "random" and a.anchor_random_seed is None:
        raise ValueError("Random anchor sampling requires --anchor-random-seed")
    frequencies, component, count = anchors(
        frequency_rows,
        sampling=a.anchor_sampling,
        random_seed=a.anchor_random_seed,
    )
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
                "Cached anchors differ from the requested selection. Use a fresh output directory."
            )
    low, high = component
    (out / "anchor_selection.json").write_text(
        json.dumps(
            {
                "selection_id": a.selection_id,
                "ec_a_mhz": a.ec_a_mhz,
                "largest_sampled_component_ghz": component,
                "component_sample_count": count,
                "sampling_policy": a.anchor_sampling,
                "sampling_fractions": [0.1, 0.3, 0.5, 0.7, 0.9]
                if a.anchor_sampling == "fractional"
                else None,
                "random_seed": a.anchor_random_seed,
                "eligible_anchor_frequency_ghz": [low, high]
                if a.anchor_sampling == "random"
                else [low + 0.1 * (high - low), low + 0.9 * (high - low)],
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
        prepare_args = [
                sys.executable,
                str(HERE / "prepare_fixed_a_anchor_campaign.py"),
                "--frequency-results",
                str(results),
                "--frequencies-ghz",
                *map(str, frequencies),
                "--rail-b-bare-frequency-ghz",
                "4",
                "5.5",
                "--rail-b-results",
                str(a.rail_b_results),
                "--output",
                str(out),
                "--workers",
                str(a.workers),
                "--target-g-mhz",
                str(a.target_g_mhz),
                "--anchor-ranking",
                a.anchor_ranking,
                "--fidelity-fallback-count",
                "1" if a.anchor_ranking == "maximum_S1_worst_case" else "5",
            ]
        if a.only_ec_b_mhz is not None:
            prepare_args.extend(["--only-ec-b-mhz", *map(str, a.only_ec_b_mhz)])
        command(
            prepare_args,
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
