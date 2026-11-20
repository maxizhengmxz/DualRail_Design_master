#!/usr/bin/env python3
"""Plot gate metrics over the rail-B EC grid at fixed rail A=(158,240)."""

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
ROOT = HERE.parent
DATABASE = (
    HERE / "g5_corrected_cavity_A158_240_2026-11-17"
    / "gate_output" / "compact_anchor_results.sqlite"
)
OUTPUT = ROOT / "output/ec_tuple_selection_slides/rail_b_gate_maps_A158_240"
READOUT_RESULTS = (
    HERE / "cavity_split_campaign_2026-11-09" / "readout" / "direct_ec"
    / "rail_b_step2_fr5p95" / "results.json"
)
SELECTED_B = (158.0, 226.0)
SOURCE_COLORS = {
    "00": "#E69F00",
    "01": "#377EB8",
    "10": "#2CA25F",
    "11": "#984EA3",
}


def main() -> None:
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    saved = connection.execute(
        "SELECT ec_b1_mhz,ec_b2_mhz,status,raw_fidelity,"
        "postselected_fidelity,result_json FROM gate_results"
    ).fetchall()
    connection.close()

    rows = []
    failures = []
    counts = Counter()
    for ec_b1, ec_b2, status, raw, post, text in saved:
        result = json.loads(text)
        final = result.get("final") or {}
        source = result.get("logical_source") or final.get("logical_source")
        channel = final.get("channel_selection") or result.get("channel_selection") or {}
        auxiliary = channel.get("auxiliary")
        if (
            status != "calibrated"
            or raw is None
            or post is None
            or not final.get("couplings_GHz")
            or source not in SOURCE_COLORS
        ):
            failures.append((float(ec_b1), float(ec_b2)))
            continue
        partner = f"{source}-{auxiliary or 'B20'}"
        counts[partner] += 1
        rows.append(
            {
                "ec_b1": float(ec_b1),
                "ec_b2": float(ec_b2),
                "raw_percent": 100.0 * float(raw),
                "post_percent": 100.0 * float(post),
                "j_req_mhz": 1000.0 * float(final["couplings_GHz"]["A2_B1"]),
                "source": source,
                "partner": partner,
            }
        )

    x = np.asarray([row["ec_b1"] for row in rows])
    y = np.asarray([row["ec_b2"] for row in rows])
    raw = np.asarray([row["raw_percent"] for row in rows])
    post = np.asarray([row["post_percent"] for row in rows])
    j_req = np.asarray([row["j_req_mhz"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.0), sharex=True, sharey=True)
    panels = (
        (axes[0, 0], raw, "Raw CZ fidelity", "viridis", 95.0, 100.0, "%"),
        (axes[0, 1], post, "Post-selected CZ fidelity", "magma", 99.9, 100.0, "%"),
        (axes[1, 0], j_req, r"Required hardware coupling $J_{\rm req}/h$", "cividis", 4.0, 8.0, "MHz"),
    )
    for axis, values, title, cmap, low, high, unit in panels:
        image = axis.scatter(
            x, y, c=values, cmap=cmap, vmin=low, vmax=high,
            s=16, marker="s", linewidths=0, rasterized=True,
        )
        bar = fig.colorbar(image, ax=axis, fraction=0.047, pad=0.025)
        bar.set_label(unit)
        axis.set_title(title, fontsize=13)

    partner_axis = axes[1, 1]
    for source, color in SOURCE_COLORS.items():
        subset = [row for row in rows if row["source"] == source]
        if subset:
            partner_axis.scatter(
                [row["ec_b1"] for row in subset],
                [row["ec_b2"] for row in subset],
                color=color, s=16, marker="s", linewidths=0,
                rasterized=True,
            )
    if failures:
        partner_axis.scatter(
            [item[0] for item in failures], [item[1] for item in failures],
            color="#BFC5CA", marker="x", s=18, linewidths=0.7,
        )
    partner_axis.set_title("Selected CZ partner", fontsize=13)

    for axis in axes.flat:
        axis.scatter(
            [SELECTED_B[0]], [SELECTED_B[1]], marker="*", s=190,
            facecolor="#FFF176", edgecolor="#D62728", linewidth=1.4, zorder=8,
        )
        axis.set_aspect("equal")
        axis.grid(False)
        axis.set_xlabel(r"$E_{C,B1}/h$ (MHz)")
        axis.set_ylabel(r"$E_{C,B2}/h$ (MHz)")
        for spine in axis.spines.values():
            spine.set_color("#374151")

    legend = [
        Line2D(
            [], [], color=SOURCE_COLORS[source], marker="s", linestyle="none",
            markersize=7,
            label=rf"$|{source}\rangle\leftrightarrow|\widetilde{{0020}}\rangle$",
        )
        for source in ("00", "01", "10", "11")
    ]
    legend.extend(
        [
            Line2D([], [], color="#BFC5CA", marker="x", linestyle="none", label="No calibrated result"),
            Line2D(
                [], [], marker="*", linestyle="none", markersize=12,
                markerfacecolor="#FFF176", markeredgecolor="#D62728",
                label=r"Selected $(E_{C,B1},E_{C,B2})=(158,226)$ MHz",
            ),
        ]
    )
    partner_axis.legend(handles=legend, loc="upper left", fontsize=8.5, frameon=False)
    fig.suptitle(
        r"Gate maps on the rail-B EC grid at fixed $E_{C,A}/h=(158,240)$ MHz"
        "\n"
        r"pointwise minimum-S1 selection among five rail-A anchors, target $g/h=5$ MHz",
        fontsize=17,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.07, top=0.88, hspace=0.20, wspace=0.18)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = "rail_b_gate_metrics_and_selected_cz_partner_A158_240"
    fig.savefig(OUTPUT / f"{stem}.png", dpi=240, facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)

    readout = {
        (round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"])):
        row.get("status") == "validated_pass"
        for row in json.loads(READOUT_RESULTS.read_text())
    }
    xs = sorted({round(row[0]) for row in saved})
    ys = sorted({round(row[1]) for row in saved})
    x_index = {value: index for index, value in enumerate(xs)}
    y_index = {value: index for index, value in enumerate(ys)}
    overlay_grid = np.full((len(ys), len(xs)), np.nan)
    overlay_counts = Counter()
    for ec_b1, ec_b2, status, raw_value, post_value, _ in saved:
        coordinate = (round(ec_b1), round(ec_b2))
        gate_pass = (
            status == "calibrated"
            and raw_value is not None and raw_value > 0.95
            and post_value is not None and post_value > 0.999
        )
        readout_pass = bool(readout.get(coordinate, False))
        category = 3 if gate_pass and readout_pass else 2 if gate_pass else 1 if readout_pass else 0
        overlay_grid[y_index[coordinate[1]], x_index[coordinate[0]]] = category
        overlay_counts[("neither", "readout_only", "gate_only", "overlay")[category]] += 1

    overlay_cmap = ListedColormap(["#DEDEDE", "#8ECAE6", "#F4A261", "#2A9D8F"])
    overlay_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], overlay_cmap.N)
    overlay_fig, overlay_axis = plt.subplots(figsize=(8.2, 7.2))
    overlay_axis.imshow(
        np.ma.masked_invalid(overlay_grid),
        origin="lower",
        interpolation="nearest",
        extent=(min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1),
        cmap=overlay_cmap,
        norm=overlay_norm,
        aspect="equal",
    )
    overlay_axis.scatter(
        [SELECTED_B[0]], [SELECTED_B[1]], marker="*", s=210,
        facecolor="#FFF176", edgecolor="#D62728", linewidth=1.5, zorder=5,
    )
    overlay_axis.set(
        title=(
            "Independent rail-B readout and gate-feasibility overlay\n"
            r"$E_{C,A}/h=(158,240)$ MHz"
        ),
        xlabel=r"$E_{C,B1}/h$ (MHz)",
        ylabel=r"$E_{C,B2}/h$ (MHz)",
    )
    overlay_axis.grid(False)
    overlay_axis.legend(
        handles=[
            Patch(color=overlay_cmap(index), label=label)
            for index, label in enumerate(("Neither", "Readout only", "Gate only", "Readout + gate"))
        ] + [
            Line2D(
                [], [], marker="*", linestyle="none", markersize=12,
                markerfacecolor="#FFF176", markeredgecolor="#D62728",
                label=r"Selected $(158,226)$ MHz",
            )
        ],
        loc="upper right",
        frameon=False,
        fontsize=9,
    )
    overlay_fig.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.88)
    overlay_stem = "rail_b_readout_gate_overlay_no_grid"
    overlay_fig.savefig(OUTPUT / f"{overlay_stem}.png", dpi=240, facecolor="white")
    overlay_fig.savefig(OUTPUT / f"{overlay_stem}.pdf", facecolor="white")
    campaign_plots = DATABASE.parent / "plots"
    overlay_fig.savefig(campaign_plots / "readout_gate_overlay.png", dpi=240, facecolor="white")
    overlay_fig.savefig(campaign_plots / "readout_gate_overlay.pdf", facecolor="white")
    plt.close(overlay_fig)

    summary = {
        "schema": "rail-b-gate-maps-slide-v1",
        "fixed_ec_a_mhz": [158, 240],
        "selected_ec_b_mhz": list(SELECTED_B),
        "anchor_policy": "pointwise minimum aggregate S1 among five rail-A anchors with fidelity fallback",
        "marked_tuple_selected_anchor_frequency_ghz": 4.6,
        "target_g_mhz": 5.0,
        "calibrated_points": len(rows),
        "failed_points": len(failures),
        "selected_partner_counts": dict(counts),
        "raw_fidelity_percent_range": [float(raw.min()), float(raw.max())],
        "postselected_fidelity_percent_range": [float(post.min()), float(post.max())],
        "j_req_mhz_range": [float(j_req.min()), float(j_req.max())],
        "source_database": str(DATABASE.resolve()),
        "readout_results": str(READOUT_RESULTS.resolve()),
        "overlay_counts": dict(overlay_counts),
        "overlay_figure_png": str((OUTPUT / f"{overlay_stem}.png").resolve()),
        "overlay_figure_pdf": str((OUTPUT / f"{overlay_stem}.pdf").resolve()),
        "figure_png": str((OUTPUT / f"{stem}.png").resolve()),
        "figure_pdf": str((OUTPUT / f"{stem}.pdf").resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
