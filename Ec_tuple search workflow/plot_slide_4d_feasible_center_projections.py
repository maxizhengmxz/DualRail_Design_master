#!/usr/bin/env python3
"""Plot binary rail-plane projections of retained feasible 4D centers."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CENTERS = HERE / "relaxed_adaptive_4d_best_case_2026-11-17/relaxed_centers.csv"
OUTPUT = ROOT / "output/ec_tuple_selection_slides/four_dimensional_center_projections"
MAX_RELATIVE_J_SPREAD = 0.15
SELECTED = (158, 240, 158, 226)


def main() -> None:
    rows = []
    with CENTERS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["neighborhood_relative_j_req_spread"]) > MAX_RELATIVE_J_SPREAD:
                continue
            rows.append(tuple(json.loads(row["ec_tuple_mhz"])))

    rail_a_counts = Counter(item[:2] for item in rows)
    rail_b_counts = Counter(item[2:] for item in rows)
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 12.0), sharex=True, sharey=True)
    for axis, counts, selected, title in (
        (
            axes[0], rail_a_counts, SELECTED[:2],
            "Projection onto the rail-A EC plane",
        ),
        (
            axes[1], rail_b_counts, SELECTED[2:],
            "Projection onto the rail-B EC plane",
        ),
    ):
        coordinates = sorted(counts)
        axis.scatter(
            [item[0] for item in coordinates],
            [item[1] for item in coordinates],
            color="#2A9D8F", marker="s", s=22, linewidths=0,
            label="At least one feasible 4D center",
        )
        axis.scatter(
            [selected[0]], [selected[1]], marker="*", s=230,
            facecolor="#FFF176", edgecolor="#D62728", linewidth=1.6,
            zorder=5, label=f"Selected {selected} MHz",
        )
        axis.set(
            title=title,
            xlabel=r"$E_{C,1}/h$ (MHz)",
            ylabel=r"$E_{C,2}/h$ (MHz)",
            xlim=(145, 315),
            ylim=(145, 315),
        )
        axis.set_aspect("equal")
        axis.grid(False)
        axis.legend(loc="upper right", frameon=False, fontsize=9)
        for spine in axis.spines.values():
            spine.set_color("#374151")

    fig.suptitle(
        "Feasible 5 MHz-radius 4D neighborhood centers\n"
        r"All 233 sampled neighbors pass; relative $J_{\rm req}$ spread $\leq15\%$",
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.025,
        "Each projected point means that at least one hidden EC pair completes a feasible 4D tuple.",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.075, top=0.90, hspace=0.22)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = "feasible_4d_neighborhood_center_projections"
    fig.savefig(OUTPUT / f"{stem}.png", dpi=240, facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)

    summary = {
        "schema": "feasible-4d-center-projections-slide-v1",
        "source_centers": str(CENTERS.resolve()),
        "relative_j_req_spread_maximum": MAX_RELATIVE_J_SPREAD,
        "feasible_4d_centers": len(rows),
        "unique_rail_a_projection_points": len(rail_a_counts),
        "unique_rail_b_projection_points": len(rail_b_counts),
        "selected_ec_tuple_mhz": list(SELECTED),
        "projection_semantics": "a projected rail coordinate is shown when at least one hidden opposite-rail coordinate forms a retained feasible 4D center",
        "historical_provenance_warning": "retained center table combines historical best-case campaign halves; the promoted tuple itself has a separate corrected-cavity 233-point rerun",
        "figure_png": str((OUTPUT / f"{stem}.png").resolve()),
        "figure_pdf": str((OUTPUT / f"{stem}.pdf").resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
