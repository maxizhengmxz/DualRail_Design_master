#!/usr/bin/env python3
"""Label eight selected rail-A EC pairs and show their rail-B EC spread."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Circle, Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE / "combined_overlay_selection_2026-11-12" / "idle_above_all_interactions_200MHz"
SOURCE = ROOT / "reselected_configurations.json"
B_RESULTS = (
    HERE / "cavity_split_campaign_2026-11-09" / "readout" / "direct_ec"
    / "rail_b_step2_fr5p95" / "results.json"
)
OUTPUT = ROOT / "eight_selected_eca_to_ecb"
A1_VALUES = (156, 158, 160, 162)
A2_VALUES = (240, 242)


def background(ax) -> None:
    rows = json.loads(B_RESULTS.read_text())
    xs = sorted({round(r["ec1_target_mhz"]) for r in rows})
    ys = sorted({round(r["ec2_target_mhz"]) for r in rows})
    values = np.full((len(ys), len(xs)), np.nan)
    for row in rows:
        x, y = round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"])
        values[ys.index(y), xs.index(x)] = float(row["status"] == "validated_pass")
    ax.imshow(
        values, origin="lower", interpolation="nearest",
        extent=[min(xs)-1, max(xs)+1, min(ys)-1, max(ys)+1], aspect="equal",
        cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
        norm=BoundaryNorm([-.5, .5, 1.5], 2),
    )


def main() -> None:
    rows = [
        row for row in json.loads(SOURCE.read_text())["selections"]
        if row["ec_a_mhz"][0] in A1_VALUES and row["ec_a_mhz"][1] in A2_VALUES
    ]
    rows.sort(key=lambda r: (-r["ec_a_mhz"][1], r["ec_a_mhz"][0]))
    if len(rows) != 8:
        raise RuntimeError(f"Expected eight selected rail-A points, found {len(rows)}")
    for index, row in enumerate(rows, 1):
        row["display_label"] = f"A{index}"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    clearance = [r["neighborhood_minimum_idle_above_interactions_clearance_mhz"] for r in rows]
    norm = Normalize(min(clearance), max(clearance))
    cmap = plt.get_cmap("plasma")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), layout="constrained")

    image = axes[0].scatter(
        [r["ec_a_mhz"][0] for r in rows], [r["ec_a_mhz"][1] for r in rows],
        c=clearance, cmap=cmap, norm=norm, s=105,
        edgecolors="black", linewidths=.7, zorder=4,
    )
    for row in rows:
        axes[0].annotate(
            row["display_label"], row["ec_a_mhz"], xytext=(5, 5),
            textcoords="offset points", fontsize=9, fontweight="bold",
        )
    axes[0].set(
        title="Eight selected rail-A EC pairs",
        xlabel=r"$E_{C,A1}/h$ (MHz)", ylabel=r"$E_{C,A2}/h$ (MHz)",
        xlim=(154.5, 163.5), ylim=(239, 243),
        xticks=A1_VALUES, yticks=A2_VALUES,
    )
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=.2)

    background(axes[1])
    label_offsets: dict[tuple[int, int], int] = {}
    for row in rows:
        ec = tuple(row["ec_b_mhz"])
        radius = row["anchor_consistent_robustness_margin_mhz"]
        color = cmap(norm(row["neighborhood_minimum_idle_above_interactions_clearance_mhz"]))
        axes[1].add_patch(Circle(ec, radius, facecolor="none", edgecolor=color, lw=1.1, alpha=.35))
        axes[1].scatter(*ec, c=[color], s=88, edgecolors="black", linewidths=.7, zorder=4)
        offset_index = label_offsets.get(ec, 0)
        axes[1].annotate(
            row["display_label"], ec, xytext=(5, 5 + 11*offset_index),
            textcoords="offset points", fontsize=8.5, fontweight="bold", zorder=5,
        )
        label_offsets[ec] = offset_index + 1
    axes[1].legend(
        handles=[
            Patch(facecolor="#d8efe8", edgecolor=".5", label="Rail-B readout valid"),
            Patch(facecolor="#e2e6e9", edgecolor=".5", label="Rail-B readout invalid"),
            Patch(facecolor="white", edgecolor=".5", label="Outside sampled EC domain"),
        ],
        title=r"Independent rail-B readout ($f_{r,B}=5.95$ GHz)",
        loc="upper right", fontsize=8, title_fontsize=8,
    )
    axes[1].set(
        title="Selected rail-B EC spread\ncircle radius = clearance- and anchor-consistent margin",
        xlabel=r"$E_{C,B1}/h$ (MHz)", ylabel=r"$E_{C,B2}/h$ (MHz)",
        xlim=(245, 280), ylim=(167, 203),
    )
    axes[1].set_aspect("equal")
    fig.colorbar(
        image, ax=axes,
        label="Neighborhood minimum idle-above-all-interactions clearance (MHz)",
        shrink=.82,
    )
    fig.suptitle("Eight selected rail-A designs and their rail-B configurations", fontsize=13)
    fig.savefig(OUTPUT / "eight_selected_eca_to_ecb_spread.png", dpi=220)
    fig.savefig(OUTPUT / "eight_selected_eca_to_ecb_spread.pdf")
    plt.close(fig)

    records = [
        {
            "label": row["display_label"],
            "ec_a_mhz": row["ec_a_mhz"],
            "ec_b_mhz": row["ec_b_mhz"],
            "selected_anchor_id": row["selected_anchor_id"],
            "robustness_margin_mhz": row["anchor_consistent_robustness_margin_mhz"],
            "neighborhood_minimum_clearance_mhz": row["neighborhood_minimum_idle_above_interactions_clearance_mhz"],
            "idle_minus_selected_gate_frequency_mhz": row["idle_operating_minus_selected_gate_frequency_mhz"],
            "neighborhood_raw_fidelity_median": row["neighborhood_raw_fidelity_median"],
            "neighborhood_postselected_fidelity_median": row["neighborhood_postselected_fidelity_median"],
        }
        for row in rows
    ]
    (OUTPUT / "eight_selected_configurations.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "selections": records}, indent=2))


if __name__ == "__main__":
    main()
