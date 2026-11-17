#!/usr/bin/env python3
"""Map the circled high-robustness rail-A band to selected rail-B EC points."""
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
from matplotlib.patches import Circle, Patch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE / "combined_overlay_selection_2026-11-12" / "idle_above_all_interactions_200MHz"
SOURCE = ROOT / "reselected_configurations.json"
B_RESULTS = (
    HERE / "cavity_split_campaign_2026-11-09" / "readout" / "direct_ec"
    / "rail_b_step2_fr5p95" / "results.json"
)
OUTPUT = ROOT / "circled_high_robustness_ecb_spread"
A1_BOUNDS = (150, 170)
A2_BOUNDS = (238, 242)


def readout_background(ax):
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
    all_rows = json.loads(SOURCE.read_text())["selections"]
    rows = [
        row for row in all_rows
        if A1_BOUNDS[0] <= row["ec_a_mhz"][0] <= A1_BOUNDS[1]
        and A2_BOUNDS[0] <= row["ec_a_mhz"][1] <= A2_BOUNDS[1]
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row["ec_b_mhz"])].append(row)
    clearances = [r["neighborhood_minimum_idle_above_interactions_clearance_mhz"] for r in rows]
    norm = Normalize(min(clearances), max(clearances))
    cmap = plt.get_cmap("plasma")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), layout="constrained")

    left = axes[0].scatter(
        [r["ec_a_mhz"][0] for r in rows], [r["ec_a_mhz"][1] for r in rows],
        c=clearances, cmap=cmap, norm=norm,
        s=[28 + 5*r["anchor_consistent_robustness_margin_mhz"] for r in rows],
        edgecolors="black", linewidths=.45,
    )
    axes[0].add_patch(Rectangle(
        (A1_BOUNDS[0]-1, A2_BOUNDS[0]-1),
        A1_BOUNDS[1]-A1_BOUNDS[0]+2, A2_BOUNDS[1]-A2_BOUNDS[0]+2,
        fill=False, ec="#d62728", lw=2,
    ))
    axes[0].set(
        title=f"Circled rail-A band ({len(rows)} selected points)\nmarker size = robustness radius",
        xlabel=r"$E_{C,A1}/h$ (MHz)", ylabel=r"$E_{C,A2}/h$ (MHz)",
        xlim=(147, 173), ylim=(235, 245),
    )
    axes[0].set_aspect("equal")

    readout_background(axes[1])
    centers = sorted(grouped)
    for ec in centers:
        for row in grouped[ec]:
            axes[1].add_patch(Circle(
                ec, row["anchor_consistent_robustness_margin_mhz"],
                facecolor="none",
                edgecolor=cmap(norm(row["neighborhood_minimum_idle_above_interactions_clearance_mhz"])),
                lw=.9, alpha=.25,
            ))
    center_clearance = [
        float(np.median([r["neighborhood_minimum_idle_above_interactions_clearance_mhz"] for r in grouped[ec]]))
        for ec in centers
    ]
    axes[1].scatter(
        [ec[0] for ec in centers], [ec[1] for ec in centers],
        c=center_clearance, cmap=cmap, norm=norm,
        s=[50 + 22*np.sqrt(len(grouped[ec])) for ec in centers],
        edgecolors="black", linewidths=.65, zorder=4,
    )
    for ec in centers:
        axes[1].annotate(
            f"n={len(grouped[ec])}", ec, xytext=(4, 4), textcoords="offset points",
            fontsize=8, zorder=5,
        )
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
        title="Selected ordered rail-B EC configurations\ncircle radius = clearance- and anchor-consistent margin",
        xlabel=r"$E_{C,B1}/h$ (MHz)", ylabel=r"$E_{C,B2}/h$ (MHz)",
        xlim=(150, 285), ylim=(155, 210),
    )
    axes[1].set_aspect("equal")
    fig.colorbar(
        left, ax=axes,
        label="Neighborhood minimum idle-above-all-interactions clearance (MHz)",
        shrink=.82,
    )
    fig.suptitle("Rail-A circled high-robustness band to rail-B selection spread", fontsize=13)
    fig.savefig(OUTPUT / "rail_a_to_rail_b_spread.png", dpi=220)
    fig.savefig(OUTPUT / "rail_a_to_rail_b_spread.pdf")
    plt.close(fig)

    fields = [
        "selection_id", "ec_a_mhz", "ec_b_mhz", "selected_anchor_id",
        "anchor_consistent_robustness_margin_mhz",
        "neighborhood_minimum_idle_above_interactions_clearance_mhz",
        "idle_operating_minus_selected_gate_frequency_mhz",
        "neighborhood_raw_fidelity_median", "neighborhood_postselected_fidelity_median",
    ]
    with (OUTPUT / "circled_region_points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k]) if isinstance(row[k], list) else row[k] for k in fields})
    summary = {
        "rail_a_bounds_mhz": {"A1": A1_BOUNDS, "A2": A2_BOUNDS},
        "rail_a_point_count": len(rows),
        "unique_rail_b_point_count": len(centers),
        "rail_b_ec1_range_mhz": [min(x for x, _ in centers), max(x for x, _ in centers)],
        "rail_b_ec2_range_mhz": [min(y for _, y in centers), max(y for _, y in centers)],
        "rail_b_usage_counts": {f"{x},{y}": len(grouped[(x, y)]) for x, y in centers},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2))


if __name__ == "__main__":
    main()
