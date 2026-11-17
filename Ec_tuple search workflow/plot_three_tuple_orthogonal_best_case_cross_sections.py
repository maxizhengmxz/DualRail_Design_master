#!/usr/bin/env python3
"""Plot best-case A- and B-grid cross-sections for three selected EC tuples."""

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
SELECTION_FILE = HERE / "combined_overlay_selection_2026-11-12/combined_overlay_best_configurations.json"
ORIGINAL_A = HERE / "readout/direct_ec/rail_a_step2/results.json"
ORIGINAL_B = HERE / "readout/direct_ec/rail_b_step2/results.json"
SPLIT_A = HERE / "cavity_split_campaign_2026-11-09/readout/direct_ec/rail_a_step2_fr6p05/results.json"
SPLIT_B = HERE / "cavity_split_campaign_2026-11-09/readout/direct_ec/rail_b_step2_fr5p95/results.json"
OUTPUT = HERE / "relaxed_adaptive_4d_best_case_2026-11-17/three_tuple_orthogonal_cross_sections"

TUPLES = (
    ("Original", (154, 208, 216, 252)),
    ("Radius optimum", (234, 160, 240, 170)),
    ("Low-J candidate", (158, 240, 158, 226)),
)
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


def directories():
    payload = json.loads(SELECTION_FILE.read_text())
    return {
        tuple(map(round, row["ec_a_mhz"])): Path(row["source_repetition"])
        for row in payload["selections"]
    }


def decode_record(ec1, ec2, status, raw, post, text):
    if text is None:
        return None
    result = json.loads(text)
    final = result.get("final") or {}
    source = result.get("logical_source") or final.get("logical_source")
    coupling = (final.get("couplings_GHz") or {}).get("A2_B1")
    raw_value = float(final.get("raw_fidelity", raw or float("nan")))
    post_value = float(final.get("postselected_fidelity", post or float("nan")))
    if status != "calibrated" or source not in SOURCES or coupling is None:
        return None
    if not raw_value > 0.95 or not post_value > 0.999:
        return None
    return {
        "coordinate": (round(ec1), round(ec2)),
        "source": source,
        "j_req_mhz": 1000.0 * float(coupling),
    }


def fixed_a_map(directory: Path):
    database = directory / "gate_output/compact_anchor_results.sqlite"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    saved = connection.execute(
        "SELECT ec_b1_mhz,ec_b2_mhz,status,raw_fidelity,postselected_fidelity,result_json FROM gate_results"
    ).fetchall()
    connection.close()
    return [record for row in saved if (record := decode_record(*row)) is not None]


def fixed_b_map(all_directories, fixed_b):
    records = []
    for ec_a, directory in all_directories.items():
        database = directory / "gate_output/compact_anchor_results.sqlite"
        if not database.exists():
            continue
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        saved = connection.execute(
            "SELECT ?,?,status,raw_fidelity,postselected_fidelity,result_json FROM gate_results "
            "WHERE round(ec_b1_mhz)=? AND round(ec_b2_mhz)=?",
            (*ec_a, *fixed_b),
        ).fetchone()
        connection.close()
        if saved is not None and (record := decode_record(*saved)) is not None:
            records.append(record)
    return records


def split_provenance(directory: Path):
    return "cavity_split_campaign_2026-11-09" in str(directory)


def contour(ax, grid, color, linestyle="-"):
    xs, ys, valid = grid
    ax.contour(xs, ys, valid.astype(float), levels=[0.5], colors=color,
               linewidths=1.0, linestyles=linestyle, zorder=2)


def scatter_source(ax, records, cmap, norm):
    return ax.scatter(
        [row["coordinate"][0] for row in records],
        [row["coordinate"][1] for row in records],
        c=[SOURCES.index(row["source"]) for row in records],
        cmap=cmap, norm=norm, s=11, marker="s", linewidths=0,
    )


def scatter_j(ax, records, norm):
    return ax.scatter(
        [row["coordinate"][0] for row in records],
        [row["coordinate"][1] for row in records],
        c=[row["j_req_mhz"] for row in records],
        cmap="viridis", norm=norm, s=11, marker="s", linewidths=0,
    )


def mark(ax, coordinate):
    ax.add_patch(Circle(coordinate, radius=5.0, fill=False, edgecolor="#D62728",
                        linewidth=1.35, linestyle="--", zorder=7))
    ax.scatter(*coordinate, marker="*", s=155, facecolor="#FFF176",
               edgecolor="#D62728", linewidth=1.3, zorder=8)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_directories = directories()
    original_a, original_b = readout_grid(ORIGINAL_A), readout_grid(ORIGINAL_B)
    split_a, split_b = readout_grid(SPLIT_A), readout_grid(SPLIT_B)
    columns = []
    all_j = []
    for name, ec_tuple in TUPLES:
        a = ec_tuple[:2]
        b = ec_tuple[2:]
        directory = all_directories[a]
        b_records = fixed_a_map(directory)
        a_records = fixed_b_map(all_directories, b)
        all_j.extend(row["j_req_mhz"] for row in b_records)
        all_j.extend(row["j_req_mhz"] for row in a_records)
        columns.append({
            "name": name, "tuple": ec_tuple, "directory": directory,
            "split": split_provenance(directory),
            "b_records": b_records, "a_records": a_records,
        })

    source_cmap = ListedColormap(COLORS)
    source_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), source_cmap.N)
    j_norm = Normalize(min(all_j), max(all_j))
    fig, axes = plt.subplots(4, 3, figsize=(18.5, 22.0), layout="constrained")
    j_image = None
    summary = {"schema": "three-tuple-orthogonal-best-case-cross-sections-v1", "columns": []}
    for column, data in enumerate(columns):
        a, b = data["tuple"][:2], data["tuple"][2:]
        b_grid = split_b if data["split"] else original_b

        scatter_source(axes[0, column], data["b_records"], source_cmap, source_norm)
        contour(axes[0, column], b_grid, "black")
        mark(axes[0, column], b)
        axes[0, column].set_title(f"{data['name']}: fixed A={a}\nrail-B grid — CZ partner")

        j_image = scatter_j(axes[1, column], data["b_records"], j_norm)
        contour(axes[1, column], b_grid, "black")
        mark(axes[1, column], b)
        axes[1, column].set_title("rail-B grid — " + r"$J_{\rm req}$")

        scatter_source(axes[2, column], data["a_records"], source_cmap, source_norm)
        contour(axes[2, column], original_a, "#666666", "--")
        contour(axes[2, column], split_a, "black", "-")
        mark(axes[2, column], a)
        axes[2, column].set_title(f"fixed B={b}: rail-A grid — CZ partner")

        j_image = scatter_j(axes[3, column], data["a_records"], j_norm)
        contour(axes[3, column], original_a, "#666666", "--")
        contour(axes[3, column], split_a, "black", "-")
        mark(axes[3, column], a)
        axes[3, column].set_title("rail-A grid — " + r"$J_{\rm req}$")

        for row in (0, 1):
            axes[row, column].set_xlabel(r"$E_{C,B1}/h$ (MHz)")
            axes[row, column].set_ylabel(r"$E_{C,B2}/h$ (MHz)")
        for row in (2, 3):
            axes[row, column].set_xlabel(r"$E_{C,A1}/h$ (MHz)")
            axes[row, column].set_ylabel(r"$E_{C,A2}/h$ (MHz)")
        for row in range(4):
            axes[row, column].set_aspect("equal")

        center_b = next((row for row in data["b_records"] if row["coordinate"] == b), None)
        center_a = next((row for row in data["a_records"] if row["coordinate"] == a), None)
        summary["columns"].append({
            "name": data["name"], "ec_tuple_mhz": list(data["tuple"]),
            "source_repetition": str(data["directory"]),
            "fixed_a_b_grid_valid_point_count": len(data["b_records"]),
            "fixed_b_a_grid_valid_point_count": len(data["a_records"]),
            "selected_point_from_b_cross_section": center_b,
            "selected_point_from_a_cross_section": center_a,
        })

    fig.colorbar(j_image, ax=axes[[1, 3], :], label=r"$J_{\rm req}/h$ (MHz)", shrink=0.82)
    legend = [
        Patch(facecolor=COLORS[i], label=rf"$|{source}\rangle\leftrightarrow|\widetilde{{0020}}\rangle$")
        for i, source in enumerate(SOURCES)
    ]
    legend += [
        Line2D([], [], color="black", lw=1.0, label="split-cavity readout boundary"),
        Line2D([], [], color="#666666", lw=1.0, ls="--", label="original-cavity readout boundary"),
        Line2D([], [], marker="*", markerfacecolor="#FFF176", markeredgecolor="#D62728",
               linestyle="none", markersize=11, label="selected coordinate"),
        Line2D([], [], color="#D62728", ls="--", label="5 MHz rail-plane circle"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Best-case orthogonal cross-sections through three selected 4D EC tuples\n"
        "Rows 1–2 fix rail A; rows 3–4 fix rail B",
        fontsize=15,
    )
    png = OUTPUT / "three_tuple_best_case_A_and_B_cross_sections.png"
    pdf = OUTPUT / "three_tuple_best_case_A_and_B_cross_sections.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    summary.update({
        "shared_j_req_scale_mhz": [min(all_j), max(all_j)],
        "figure_png": str(png), "figure_pdf": str(pdf),
        "provenance_warning": (
            "Historical original/split best-case records are shown as retained. The split partition "
            "predates fully corrected downstream cavity propagation."
        ),
    })
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
