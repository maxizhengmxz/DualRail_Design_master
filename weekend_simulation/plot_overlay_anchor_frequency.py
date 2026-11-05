#!/usr/bin/env python3
"""Color a readout/gate overlay region by the selected rail-A anchor frequency."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
DEFAULT_B_READOUT = HERE / "readout" / "direct_ec" / "rail_b_step2" / "results.json"
RAW_MIN = 0.95
POST_MIN = 0.999


def run(campaign: Path, readout_results: Path) -> dict:
    readout = {
        (int(round(row["ec1_target_mhz"])), int(round(row["ec2_target_mhz"]))): row["status"] == "validated_pass"
        for row in json.loads(readout_results.read_text())
    }
    con = sqlite3.connect(campaign / "gate_output" / "compact_anchor_results.sqlite")
    rows = []
    for b1, b2, raw, post, result_text in con.execute(
        "SELECT ec_b1_mhz,ec_b2_mhz,raw_fidelity,postselected_fidelity,result_json FROM gate_results"
    ):
        ec = (int(round(b1)), int(round(b2)))
        result = json.loads(result_text)
        anchor = result.get("selected_anchor") or {}
        frequency = anchor.get("readout_operating_frequency_GHz")
        gate_pass = raw is not None and post is not None and raw > RAW_MIN and post > POST_MIN
        readout_pass = readout[ec]
        category = "overlay" if gate_pass and readout_pass else "gate_only" if gate_pass else "readout_only" if readout_pass else "neither"
        rows.append({"ec": ec, "category": category, "anchor_frequency_ghz": frequency})
    con.close()

    overlay = [row for row in rows if row["category"] == "overlay"]
    if not overlay or any(row["anchor_frequency_ghz"] is None for row in overlay):
        raise RuntimeError("Overlay points are missing selected-anchor frequencies")
    frequencies = sorted({round(row["anchor_frequency_ghz"], 9) for row in overlay})
    output = campaign / "gate_output" / "plots"
    output.mkdir(parents=True, exist_ok=True)

    base_colors = {"neither": "#dddddd", "readout_only": "#8ecae6", "gate_only": "#f4a261"}
    fig, ax = plt.subplots(figsize=(8.2, 7.0), layout="constrained")
    for category in ("neither", "readout_only", "gate_only"):
        subset = [row for row in rows if row["category"] == category]
        ax.scatter(
            [row["ec"][0] for row in subset], [row["ec"][1] for row in subset],
            c=base_colors[category], s=14, marker="s", linewidths=0, zorder=1,
        )
    image = ax.scatter(
        [row["ec"][0] for row in overlay], [row["ec"][1] for row in overlay],
        c=[row["anchor_frequency_ghz"] for row in overlay], cmap="viridis",
        vmin=min(frequencies), vmax=max(frequencies), s=15, marker="s", linewidths=0, zorder=2,
    )
    colorbar = fig.colorbar(image, ax=ax, label="Selected rail-A readout anchor frequency (GHz)", shrink=.88)
    colorbar.set_ticks(frequencies)
    colorbar.set_ticklabels([f"{value:.3f}" for value in frequencies])
    ax.legend(handles=[
        Patch(color=base_colors["neither"], label="Neither feasible"),
        Patch(color=base_colors["readout_only"], label="Readout only"),
        Patch(color=base_colors["gate_only"], label="Gate only"),
        Line2D([], [], marker="s", ls="", color=plt.get_cmap("viridis")(.65), label="Readout + gate overlay (colored by anchor)"),
    ], loc="best", fontsize=8)
    ax.set(
        title="A4: selected rail-A anchor frequency on the independent feasibility overlay",
        xlabel=r"$E_{C,B1}/h$ (MHz)", ylabel=r"$E_{C,B2}/h$ (MHz)",
    )
    ax.set_aspect("equal")
    png = output / "readout_gate_overlay_selected_anchor_frequency.png"
    pdf = output / "readout_gate_overlay_selected_anchor_frequency.pdf"
    fig.savefig(png, dpi=240); fig.savefig(pdf); plt.close(fig)

    summary = {
        "campaign": str(campaign),
        "raw_fidelity_threshold_strict": RAW_MIN,
        "postselected_fidelity_threshold_strict": POST_MIN,
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "overlay_anchor_frequency_counts": {
            f"{frequency:.3f}": sum(abs(row["anchor_frequency_ghz"] - frequency) < 1e-7 for row in overlay)
            for frequency in frequencies
        },
        "plots": [str(png), str(pdf)],
    }
    (output / "readout_gate_overlay_selected_anchor_frequency.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--readout-results", type=Path, default=DEFAULT_B_READOUT)
    args = parser.parse_args()
    print(json.dumps(run(args.campaign.resolve(), args.readout_results.resolve()), indent=2))
