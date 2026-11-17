#!/usr/bin/env python3
"""Map the upper-left rail-A cluster to its selected rail-B configurations."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.patches import Circle, Rectangle

HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "combined_overlay_selection_2026-11-12"
    / "combined_overlay_best_configurations.json"
)
A_RESULTS = (
    HERE
    / "cavity_split_campaign_2026-11-09/readout/direct_ec"
    / "rail_a_step2_fr6p05/results.json"
)
B_RESULTS = (
    HERE
    / "cavity_split_campaign_2026-11-09/readout/direct_ec"
    / "rail_b_step2_fr5p95/results.json"
)
OUTPUT = (
    HERE
    / "combined_overlay_selection_2026-11-12"
    / "upper_left_region_analysis"
)
BOUNDS = {"ec_a1_mhz": [150, 158], "ec_a2_mhz": [254, 258]}
MINIMUM_ANCHOR_MARGIN_MHZ = 20.0


def grid(path: Path):
    rows = json.loads(path.read_text())
    xs = sorted({round(row["ec1_target_mhz"]) for row in rows})
    ys = sorted({round(row["ec2_target_mhz"]) for row in rows})
    values = np.full((len(ys), len(xs)), np.nan)
    for row in rows:
        x, y = round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"])
        values[ys.index(y), xs.index(x)] = float(row["status"] == "validated_pass")
    return xs, ys, values


def background(ax, path: Path):
    xs, ys, values = grid(path)
    ax.imshow(
        values,
        origin="lower",
        interpolation="nearest",
        extent=[min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1],
        aspect="equal",
        cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
        norm=BoundaryNorm([-.5, .5, 1.5], 2),
    )
    return xs, ys


def main() -> None:
    all_rows = json.loads(SOURCE.read_text())["selections"]
    rows = [
        row
        for row in all_rows
        if BOUNDS["ec_a1_mhz"][0] <= row["ec_a_mhz"][0] <= BOUNDS["ec_a1_mhz"][1]
        and BOUNDS["ec_a2_mhz"][0] <= row["ec_a_mhz"][1] <= BOUNDS["ec_a2_mhz"][1]
        and row["anchor_consistent_robustness_margin_mhz"]
        >= MINIMUM_ANCHOR_MARGIN_MHZ
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shifts = [row["neighborhood_maximum_absolute_bare_frequency_shift_mhz"] for row in rows]
    norm = Normalize(min(shifts), max(shifts))
    cmap = plt.get_cmap("magma_r")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), layout="constrained")

    background(axes[0], A_RESULTS)
    image = axes[0].scatter(
        [row["ec_a_mhz"][0] for row in rows],
        [row["ec_a_mhz"][1] for row in rows],
        c=shifts,
        cmap=cmap,
        norm=norm,
        s=[24 + 3 * row["anchor_consistent_robustness_margin_mhz"] for row in rows],
        edgecolors="black",
        linewidths=.45,
        zorder=4,
    )
    axes[0].add_patch(
        Rectangle(
            (BOUNDS["ec_a1_mhz"][0] - 1, BOUNDS["ec_a2_mhz"][0] - 1),
            BOUNDS["ec_a1_mhz"][1] - BOUNDS["ec_a1_mhz"][0] + 2,
            BOUNDS["ec_a2_mhz"][1] - BOUNDS["ec_a2_mhz"][0] + 2,
            fill=False,
            ec="black",
            lw=1.3,
            zorder=5,
        )
    )
    axes[0].set(
        title=f"Rail-A upper-left candidate cluster ({len(rows)} points)\nmarker size = robustness radius",
        xlabel=r"$E_{C,A1}/h$ (MHz)",
        ylabel=r"$E_{C,A2}/h$ (MHz)",
        xlim=(147, 161),
        ylim=(251, 261),
    )
    axes[0].set_aspect("equal")

    background(axes[1], B_RESULTS)
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row["ec_b_mhz"])].append(row)
        center = row["ec_b_mhz"]
        radius = row["anchor_consistent_robustness_margin_mhz"]
        axes[1].add_patch(
            Circle(
                center,
                radius,
                facecolor="none",
                edgecolor=cmap(norm(row["neighborhood_maximum_absolute_bare_frequency_shift_mhz"])),
                lw=.75,
                alpha=.18,
                zorder=2,
            )
        )
    centers = sorted(grouped)
    center_shift = [
        float(np.median([r["neighborhood_maximum_absolute_bare_frequency_shift_mhz"] for r in grouped[ec]]))
        for ec in centers
    ]
    axes[1].scatter(
        [ec[0] for ec in centers],
        [ec[1] for ec in centers],
        c=center_shift,
        cmap=cmap,
        norm=norm,
        s=[40 + 18 * np.sqrt(len(grouped[ec])) for ec in centers],
        edgecolors="black",
        linewidths=.6,
        zorder=4,
    )
    for ec in centers:
        if len(grouped[ec]) > 1:
            axes[1].annotate(
                f"n={len(grouped[ec])}",
                ec,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                zorder=5,
            )
    axes[1].set(
        title="Selected rail-B EC configurations on the independent 5.95 GHz readout map\n"
        "circle radius = same-anchor robustness margin",
        xlabel=r"$E_{C,B1}/h$ (MHz)",
        ylabel=r"$E_{C,B2}/h$ (MHz)",
    )
    axes[1].set_aspect("equal")
    axes[1].legend(
        handles=[
            Patch(facecolor="#d8efe8", edgecolor="0.5", label="Rail-B readout valid"),
            Patch(facecolor="#e2e6e9", edgecolor="0.5", label="Rail-B readout invalid"),
            Patch(facecolor="white", edgecolor="0.5", label="Outside sampled EC domain"),
        ],
        title=r"Independent rail-B readout ($f_{r,B}=5.95$ GHz)",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
    )
    fig.colorbar(
        image,
        ax=axes,
        label=(
            "Worst-case rail-B bare-frequency tuning within the 5 MHz "
            "same-anchor neighborhood (MHz)"
        ),
        shrink=.82,
    )
    fig.suptitle(
        "High-robustness upper-left cluster: rail-A to selected rail-B mapping\n"
        "Selections predate the running worst-random-anchor campaign",
        fontsize=13,
    )
    fig.savefig(OUTPUT / "upper_left_rail_a_to_rail_b_with_robustness_radii.png", dpi=220)
    fig.savefig(OUTPUT / "upper_left_rail_a_to_rail_b_with_robustness_radii.pdf")
    plt.close(fig)

    fields = [
        "selection_id",
        "ec_a_mhz",
        "ec_b_mhz",
        "selected_anchor_id",
        "anchor_consistent_robustness_margin_mhz",
        "neighborhood_maximum_absolute_bare_frequency_shift_mhz",
        "neighborhood_raw_fidelity_median",
        "neighborhood_postselected_fidelity_median",
    ]
    with (OUTPUT / "upper_left_region_points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row[key], list) else row[key]
                    for key in fields
                }
            )
    summary = {
        "region_definition": BOUNDS,
        "minimum_anchor_consistent_robustness_margin_mhz": MINIMUM_ANCHOR_MARGIN_MHZ,
        "rail_A_point_count": len(rows),
        "unique_selected_rail_B_coordinates": len(grouped),
        "minimum_robustness_margin_mhz": min(
            row["anchor_consistent_robustness_margin_mhz"] for row in rows
        ),
        "median_robustness_margin_mhz": float(
            np.median([row["anchor_consistent_robustness_margin_mhz"] for row in rows])
        ),
        "maximum_robustness_margin_mhz": max(
            row["anchor_consistent_robustness_margin_mhz"] for row in rows
        ),
        "minimum_neighborhood_maximum_shift_mhz": min(shifts),
        "median_neighborhood_maximum_shift_mhz": float(np.median(shifts)),
        "maximum_neighborhood_maximum_shift_mhz": max(shifts),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2))


if __name__ == "__main__":
    main()
