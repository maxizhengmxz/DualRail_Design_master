#!/usr/bin/env python3
"""Add selected-CZ-source and center J_req panels to relaxed 4D projections."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


HERE = Path(__file__).resolve().parent
RELAXED = HERE / "relaxed_adaptive_4d_best_case_2026-11-17"
BUILDER_PATH = HERE / "build_four_dimensional_feasibility.py"
CSV_PATH = RELAXED / "relaxed_centers.csv"
SUMMARY_PATH = RELAXED / "summary.json"
OUT_PNG = RELAXED / "relaxed_adaptive_4d_projection_cz_source_and_Jreq_original_tuple.png"
OUT_PDF = RELAXED / "relaxed_adaptive_4d_projection_cz_source_and_Jreq_original_tuple.pdf"
OUT_JSON = RELAXED / "projection_cz_source_and_Jreq_original_tuple_summary.json"
SELECTED_TUPLE = (154, 208, 216, 252)
REFERENCE_TUPLE = (234, 160, 240, 170)
LOW_J_TUPLE = (158, 240, 158, 226)

SOURCE_ORDER = ("00", "01", "10", "11")
SOURCE_COLORS = ("#E69F00", "#377EB8", "#2CA25F", "#984EA3")
MAX_RELATIVE_J_SPREAD = 0.15


def load_builder():
    spec = importlib.util.spec_from_file_location("strict_four_d", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rank_key(row):
    return (
        -row["four_dimensional_robustness_radius_mhz"],
        row["neighborhood_maximum_tuning_mhz"],
        row["neighborhood_relative_j_req_spread"],
        -row["neighborhood_minimum_clearance_mhz"],
        -row["neighborhood_minimum_raw_fidelity"],
        -row["neighborhood_minimum_postselected_fidelity"],
    )


def read_rows():
    numeric = {
        "four_dimensional_robustness_radius_mhz",
        "neighborhood_maximum_tuning_mhz",
        "neighborhood_relative_j_req_spread",
        "neighborhood_minimum_clearance_mhz",
        "neighborhood_minimum_raw_fidelity",
        "neighborhood_minimum_postselected_fidelity",
    }
    rows = []
    with CSV_PATH.open(newline="") as stream:
        for row in csv.DictReader(stream):
            row["ec_tuple_mhz"] = tuple(ast.literal_eval(row["ec_tuple_mhz"]))
            for key in numeric:
                row[key] = float(row[key])
            if row["neighborhood_relative_j_req_spread"] <= MAX_RELATIVE_J_SPREAD:
                rows.append(row)
    return rows


def projected_representatives(rows, indices):
    grouped = {}
    for row in rows:
        coordinate = tuple(row["ec_tuple_mhz"][i] for i in indices)
        current = grouped.get(coordinate)
        if current is None or rank_key(row) < rank_key(current):
            grouped[coordinate] = row
    return grouped


def main():
    rows = read_rows()
    summary = json.loads(SUMMARY_PATH.read_text())
    best_tuple = SELECTED_TUPLE
    projections = {
        "rail_a": projected_representatives(rows, (0, 1)),
        "rail_b": projected_representatives(rows, (2, 3)),
    }

    # Recover exact center-point J_req from the retained hard-valid tuple records.
    builder = load_builder()
    point_maps = {}
    for name, campaign, b_results in (
        ("original_6p00_6p00", builder.ORIGINAL, builder.ORIGINAL_B),
        ("split_campaign_saved_gate_provenance", builder.SPLIT, builder.SPLIT_B),
    ):
        point_maps[name], _ = builder.load_provenance(name, campaign, b_results)

    representatives = [row for group in projections.values() for row in group.values()]
    for row in representatives:
        point = point_maps[row["provenance"]][row["ec_tuple_mhz"]]
        row["center_j_req_mhz"] = float(point["required_interrail_coupling_mhz"])
        row["logical_source_at_center"] = str(point["logical_source"])

    all_j = [row["center_j_req_mhz"] for row in representatives]
    j_min, j_max = min(all_j), max(all_j)
    source_cmap = ListedColormap(SOURCE_COLORS)
    source_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], source_cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.2), layout="constrained")
    metadata = {}
    for column, (name, indices, title, x_label, y_label) in enumerate((
        ("rail_a", (0, 1), "Rail-A projection", r"$E_{C,A1}/h$ (MHz)", r"$E_{C,A2}/h$ (MHz)"),
        ("rail_b", (2, 3), "Rail-B projection", r"$E_{C,B1}/h$ (MHz)", r"$E_{C,B2}/h$ (MHz)"),
    )):
        grouped = projections[name]
        coordinates = list(grouped)
        selected = tuple(best_tuple[i] for i in indices)
        reference = tuple(REFERENCE_TUPLE[i] for i in indices)
        low_j = tuple(LOW_J_TUPLE[i] for i in indices)
        source_values = [SOURCE_ORDER.index(grouped[key]["logical_source_at_center"]) for key in coordinates]
        source_counts = {source: source_values.count(i) for i, source in enumerate(SOURCE_ORDER)}

        ax = axes[0, column]
        image = ax.scatter(
            [key[0] for key in coordinates], [key[1] for key in coordinates],
            c=source_values, cmap=source_cmap, norm=source_norm,
            s=25, edgecolors="black", linewidths=0.2,
        )
        colorbar = fig.colorbar(image, ax=ax, ticks=range(4), shrink=0.82)
        colorbar.ax.set_yticklabels([rf"$|{source}\rangle\leftrightarrow|\widetilde{{0020}}\rangle$" for source in SOURCE_ORDER])
        colorbar.set_label("Selected CZ logical source")
        ax.set_title(f"{title}: selected CZ partner")

        ax = axes[1, column]
        image = ax.scatter(
            [key[0] for key in coordinates], [key[1] for key in coordinates],
            c=[grouped[key]["center_j_req_mhz"] for key in coordinates],
            cmap="viridis", vmin=j_min, vmax=j_max,
            s=25, edgecolors="black", linewidths=0.2,
        )
        fig.colorbar(image, ax=ax, label=r"Center-point $J_{\rm req}/h$ (MHz)", shrink=0.82)
        ax.set_title(f"{title}: required inter-rail coupling")

        for ax in axes[:, column]:
            ax.scatter(*selected, marker="*", s=230, facecolor="#FFF176", edgecolor="#D62728", linewidth=1.7, zorder=5)
            ax.annotate(str(selected), selected, xytext=(7, 7), textcoords="offset points", fontsize=9,
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.82})
            ax.scatter(*reference, marker="X", s=90, facecolor="white", edgecolor="black", linewidth=1.1, zorder=4)
            ax.scatter(*low_j, marker="D", s=105, facecolor="#00E5FF", edgecolor="#004D60", linewidth=1.3, zorder=6)
            ax.annotate("low-J " + str(low_j), low_j, xytext=(7, -15), textcoords="offset points", fontsize=8.5,
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.82})
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_aspect("equal")

        metadata[name] = {
            "projected_coordinate_count": len(grouped),
            "source_counts": source_counts,
            "center_j_req_minimum_mhz": min(row["center_j_req_mhz"] for row in grouped.values()),
            "center_j_req_median_mhz": float(__import__("numpy").median([row["center_j_req_mhz"] for row in grouped.values()])),
            "center_j_req_maximum_mhz": max(row["center_j_req_mhz"] for row in grouped.values()),
        }

    fig.suptitle(
        "Relaxed calibration-adaptive 4D projections\n"
        "Representative at each projected coordinate uses the existing 4D ranking rule",
        fontsize=14,
    )
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    output = {
        "schema": "relaxed-4d-projection-cz-source-jreq-v1",
        "projection_semantics": "Best-ranked retained 4D center for each projected 2D coordinate.",
        "j_req_semantics": "Exact center-point required_interrail_coupling_mhz of that representative 4D tuple.",
        "relative_j_req_spread_limit": MAX_RELATIVE_J_SPREAD,
        "selected_tuple_mhz": list(best_tuple),
        "reference_relaxed_optimum_tuple_mhz": list(REFERENCE_TUPLE),
        "low_j_req_candidate_tuple_mhz": list(LOW_J_TUPLE),
        "shared_j_req_color_scale_mhz": [j_min, j_max],
        "projections": metadata,
        "provenance_warning": summary.get("historical_provenance_warning"),
        "figure_png": str(OUT_PNG),
        "figure_pdf": str(OUT_PDF),
    }
    OUT_JSON.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
