#!/usr/bin/env python3
"""Plot orthogonal best-case cross-sections through the promoted EC tuple."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SELECTIONS = HERE / "combined_overlay_selection_2026-11-12/combined_overlay_best_configurations.json"
READOUT_A = HERE / "cavity_split_campaign_2026-11-09/readout/direct_ec/rail_a_step2_fr6p05/results.json"
READOUT_B = HERE / "cavity_split_campaign_2026-11-09/readout/direct_ec/rail_b_step2_fr5p95/results.json"
OUTPUT = ROOT / "output/ec_tuple_selection_slides/selected_tuple_cross_sections"
SELECTED = (158, 240, 158, 226)
SOURCES = ("00", "01", "10", "11")
COLORS = ("#E69F00", "#377EB8", "#2CA25F", "#984EA3")


def readout_grid(path: Path):
    rows = json.loads(path.read_text())
    xs = sorted({round(row["ec1_target_mhz"]) for row in rows})
    ys = sorted({round(row["ec2_target_mhz"]) for row in rows})
    valid = np.zeros((len(ys), len(xs)), dtype=bool)
    for row in rows:
        if row.get("status") == "validated_pass":
            valid[ys.index(round(row["ec2_target_mhz"])), xs.index(round(row["ec1_target_mhz"]))] = True
    return xs, ys, valid


def campaign_directories():
    payload = json.loads(SELECTIONS.read_text())
    return {
        tuple(map(round, row["ec_a_mhz"])): Path(row["source_repetition"])
        for row in payload["selections"]
    }


def decode(ec1, ec2, status, raw, post, result_json):
    if result_json is None:
        return None
    result = json.loads(result_json)
    final = result.get("final") or {}
    source = result.get("logical_source") or final.get("logical_source")
    coupling = (final.get("couplings_GHz") or {}).get("A2_B1")
    raw = float(final.get("raw_fidelity", raw or float("nan")))
    post = float(final.get("postselected_fidelity", post or float("nan")))
    if status != "calibrated" or source not in SOURCES or coupling is None:
        return None
    if not raw > 0.95 or not post > 0.999:
        return None
    return {
        "coordinate": (round(ec1), round(ec2)),
        "source": source,
        "j_req_mhz": 1000.0 * float(coupling),
    }


def fixed_a_records(directory: Path):
    database = directory / "gate_output/compact_anchor_results.sqlite"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT ec_b1_mhz,ec_b2_mhz,status,raw_fidelity,postselected_fidelity,result_json FROM gate_results"
        ).fetchall()
    return [record for row in rows if (record := decode(*row)) is not None]


def fixed_b_records(directories, fixed_b):
    records = []
    for ec_a, directory in directories.items():
        database = directory / "gate_output/compact_anchor_results.sqlite"
        if not database.exists():
            continue
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT ?,?,status,raw_fidelity,postselected_fidelity,result_json FROM gate_results "
                "WHERE round(ec_b1_mhz)=? AND round(ec_b2_mhz)=?",
                (*ec_a, *fixed_b),
            ).fetchone()
        if row is not None and (record := decode(*row)) is not None:
            records.append(record)
    return records


def draw_boundary(axis, grid):
    xs, ys, valid = grid
    axis.contour(xs, ys, valid.astype(float), levels=[0.5], colors="black", linewidths=1.1, zorder=2)


def mark_selected(axis, coordinate):
    axis.add_patch(Circle(coordinate, 5.0, fill=False, edgecolor="#D62728", lw=1.4, ls="--", zorder=6))
    axis.scatter(*coordinate, marker="*", s=175, facecolor="#FFF176", edgecolor="#D62728", lw=1.4, zorder=7)


def main():
    directories = campaign_directories()
    a, b = SELECTED[:2], SELECTED[2:]
    b_records = fixed_a_records(directories[a])
    a_records = fixed_b_records(directories, b)
    all_j = [row["j_req_mhz"] for row in b_records + a_records]

    source_cmap = ListedColormap(COLORS)
    source_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), source_cmap.N)
    j_norm = Normalize(min(all_j), max(all_j))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 11.0), layout="constrained")

    for row_index, (records, grid, selected, rail) in enumerate(
        ((b_records, readout_grid(READOUT_B), b, "B"), (a_records, readout_grid(READOUT_A), a, "A"))
    ):
        coordinates = [record["coordinate"] for record in records]
        axes[row_index, 0].scatter(
            [p[0] for p in coordinates], [p[1] for p in coordinates],
            c=[SOURCES.index(record["source"]) for record in records],
            cmap=source_cmap, norm=source_norm, marker="s", s=24,
            linewidths=0, edgecolors="none", antialiased=False,
        )
        j_image = axes[row_index, 1].scatter(
            [p[0] for p in coordinates], [p[1] for p in coordinates],
            c=[record["j_req_mhz"] for record in records],
            cmap="viridis", norm=j_norm, marker="s", s=24,
            linewidths=0, edgecolors="none", antialiased=False,
        )
        for axis in axes[row_index]:
            draw_boundary(axis, grid)
            mark_selected(axis, selected)
            axis.set_aspect("equal")
            axis.grid(False)
            axis.set_xlabel(rf"$E_{{C,{rail}1}}/h$ (MHz)")
            axis.set_ylabel(rf"$E_{{C,{rail}2}}/h$ (MHz)")
        fixed_text = f"Fixed rail {'A' if rail == 'B' else 'B'} = {a if rail == 'B' else b} MHz"
        axes[row_index, 0].set_title(f"{fixed_text}: selected CZ partner")
        axes[row_index, 1].set_title(f"{fixed_text}: " + r"$J_{\rm req}/h$")

    fig.colorbar(j_image, ax=axes[:, 1], label=r"$J_{\rm req}/h$ (MHz)", shrink=0.82)
    legend = [
        Patch(facecolor=COLORS[i], label=rf"$|{source}\rangle\leftrightarrow|\widetilde{{0020}}\rangle$")
        for i, source in enumerate(SOURCES)
    ]
    legend += [
        Line2D([], [], color="black", lw=1.1, label="independent readout boundary"),
        Line2D([], [], marker="*", markerfacecolor="#FFF176", markeredgecolor="#D62728",
               linestyle="none", markersize=11, label="selected coordinate"),
        Line2D([], [], color="#D62728", ls="--", label="5 MHz rail-plane circle"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Orthogonal cross-sections through "
        + r"$(E_{C,A1},E_{C,A2},E_{C,B1},E_{C,B2})=(158,240,158,226)$ MHz",
        fontsize=15,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = "selected_tuple_fixed_a_fixed_b_cz_partner_jreq_cross_sections"
    png, pdf = OUTPUT / f"{stem}.png", OUTPUT / f"{stem}.pdf"
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    def center(records, coordinate):
        return next((row for row in records if row["coordinate"] == coordinate), None)

    summary = {
        "schema": "selected-tuple-orthogonal-cross-sections-slide-v1",
        "selected_ec_tuple_mhz": list(SELECTED),
        "fixed_a_valid_gate_points": len(b_records),
        "fixed_b_valid_gate_points": len(a_records),
        "selected_point_fixed_a_cross_section": center(b_records, b),
        "selected_point_fixed_b_cross_section": center(a_records, a),
        "shared_j_req_scale_mhz": [min(all_j), max(all_j)],
        "figure_png": str(png.resolve()),
        "figure_pdf": str(pdf.resolve()),
        "provenance_warning": "Historical retained best-case campaign records are plotted; the promoted center has a separate corrected-cavity rerun.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
