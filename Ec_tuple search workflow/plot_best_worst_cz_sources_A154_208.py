#!/usr/bin/env python3
"""Compare selected CZ logical sources on the rail-B EC grid at A=(154,208)."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
BEST = (
    HERE
    / "cavity_split_campaign_2026-11-09/sampled_runs/rail_a_lt_partition"
    / "A_LT_0086_ecA_154_208/gate_output/compact_anchor_results.sqlite"
)
WORST = (
    HERE
    / "worst_case_fractional_anchor_corrected_cavities_2026-11-13"
    / "split_6p05_5p95/A_LT_0086_ecA_154_208"
    / "gate_output/compact_anchor_results.sqlite"
)
B_READOUT = (
    HERE
    / "cavity_split_campaign_2026-11-09/readout/direct_ec"
    / "rail_b_step2_fr5p95/results.json"
)
OUTPUT = (
    HERE
    / "worst_case_fractional_anchor_corrected_cavities_2026-11-13"
    / "campaign_summary/A154_208_best_vs_worst_cz_sources"
)

SOURCES = ("00", "01", "10", "11")
SELECTED_B = (216, 252)
COLORS = {
    "none": "#d8dde2",
    "00": "#e69f00",
    "01": "#377eb8",
    "10": "#2ca25f",
    "11": "#984ea3",
}


def load_readout():
    rows = json.loads(B_READOUT.read_text())
    xs = sorted({round(row["ec1_target_mhz"]) for row in rows})
    ys = sorted({round(row["ec2_target_mhz"]) for row in rows})
    feasible = np.zeros((len(ys), len(xs)), dtype=bool)
    for row in rows:
        if row.get("status") == "validated_pass":
            feasible[ys.index(round(row["ec2_target_mhz"])), xs.index(round(row["ec1_target_mhz"]))] = True
    return xs, ys, feasible


def load_gate(path: Path, xs: list[int], ys: list[int]):
    grid = np.zeros((len(ys), len(xs)), dtype=int)
    failed = []
    counts = Counter()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT ec_b1_mhz,ec_b2_mhz,status,raw_fidelity,"
        "postselected_fidelity,result_json FROM gate_results"
    ).fetchall()
    connection.close()
    for ec1, ec2, status, raw, post, text in rows:
        x, y = round(ec1), round(ec2)
        result = json.loads(text)
        source = result.get("logical_source")
        if source in SOURCES:
            grid[ys.index(y), xs.index(x)] = SOURCES.index(source) + 1
            counts[source] += 1
            if not (
                status == "calibrated"
                and (raw or 0.0) > 0.95
                and (post or 0.0) > 0.999
            ):
                failed.append((x, y))
        else:
            counts["no selected source"] += 1
            failed.append((x, y))
    return grid, failed, counts


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    xs, ys, feasible = load_readout()
    best_grid, best_failed, best_counts = load_gate(BEST, xs, ys)
    worst_grid, worst_failed, worst_counts = load_gate(WORST, xs, ys)
    cmap = ListedColormap([COLORS["none"], *(COLORS[source] for source in SOURCES)])
    norm = BoundaryNorm(np.arange(-0.5, 5.0, 1.0), cmap.N)
    extent = [min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), layout="constrained", sharex=True, sharey=True)
    panels = [
        (axes[0], best_grid, best_failed, best_counts, "Best case: minimum-S1 anchor with fidelity fallback"),
        (axes[1], worst_grid, worst_failed, worst_counts, "Worst case: maximum-S1 anchor, no fallback"),
    ]
    for ax, grid, failed, counts, title in panels:
        ax.imshow(grid, origin="lower", interpolation="nearest", extent=extent,
                  aspect="equal", cmap=cmap, norm=norm)
        # The black boundary is the independent rail-B idle-readout feasible map.
        ax.contour(xs, ys, feasible.astype(float), levels=[0.5], colors="black", linewidths=1.15)
        if failed:
            ax.scatter([x for x, _ in failed], [y for _, y in failed], marker="x",
                       s=13, linewidths=.55, color="#222222", zorder=3)
        ax.scatter(
            [SELECTED_B[0]], [SELECTED_B[1]], marker="*", s=235,
            facecolor="#fff176", edgecolor="#d62728", linewidth=1.8, zorder=5,
        )
        ax.annotate(
            "(216, 252)", SELECTED_B, xytext=(8, 8), textcoords="offset points",
            fontsize=9, fontweight="bold", color="#8b0000",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": .82,
                  "edgecolor": "#d62728", "linewidth": .7},
            zorder=6,
        )
        count_text = "  ".join(f"{source}: {counts[source]:,}" for source in SOURCES)
        if counts["no selected source"]:
            count_text += f"  none: {counts['no selected source']:,}"
        ax.set_title(title + "\n" + count_text, fontsize=10.5)
        ax.set_xlabel(r"$E_{C,B1}/h$ (MHz)")
        ax.set_ylabel(r"$E_{C,B2}/h$ (MHz)")
        ax.set_aspect("equal")

    handles = [
        Patch(facecolor=COLORS[source], edgecolor="none", label=rf"$|{source}\rangle\leftrightarrow|\widetilde{{0020}}\rangle$")
        for source in SOURCES
    ]
    handles.extend([
        Line2D([], [], color="black", lw=1.15, label="rail-B readout-feasible boundary"),
        Line2D([], [], color="#222222", marker="x", linestyle="none", markersize=5,
               label="no selected source or failed fidelity"),
        Line2D([], [], marker="*", linestyle="none", markersize=12,
               markerfacecolor="#fff176", markeredgecolor="#d62728",
               label=r"selected tuple: $(E_{C,B1},E_{C,B2})=(216,252)$ MHz"),
    ])
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        r"Selected CZ collision source on the rail-B grid at "
        r"$(E_{C,A1},E_{C,A2})=(154,208)$ MHz",
        fontsize=14,
    )
    figure_png = OUTPUT / "selected_cz_sources_best_vs_worst.png"
    figure_pdf = OUTPUT / "selected_cz_sources_best_vs_worst.pdf"
    fig.savefig(figure_png, dpi=220)
    fig.savefig(figure_pdf)
    plt.close(fig)

    summary = {
        "ec_a_mhz": [154, 208],
        "auxiliary": "B20 / |~0020>",
        "marked_ec_b_mhz": list(SELECTED_B),
        "source_color_map": {source: COLORS[source] for source in SOURCES},
        "rail_b_readout_results": str(B_READOUT.resolve()),
        "best_case_database": str(BEST.resolve()),
        "worst_case_database": str(WORST.resolve()),
        "best_case_source_counts": dict(best_counts),
        "worst_case_source_counts": dict(worst_counts),
        "best_case_failed_or_missing_source_count": len(best_failed),
        "worst_case_failed_or_missing_source_count": len(worst_failed),
        "figure_png": str(figure_png.resolve()),
        "figure_pdf": str(figure_pdf.resolve()),
        "provenance_note": (
            "The historical best-case split campaign predates the corrected-cavity "
            "downstream rerun; the worst-case campaign uses corrected 6.05/5.95 GHz provenance."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
