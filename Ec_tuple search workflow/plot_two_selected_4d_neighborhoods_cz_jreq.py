#!/usr/bin/env python3
"""Plot CZ source and J_req in two selected 5 MHz 4D neighborhoods."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
BUILDER_PATH = HERE / "build_four_dimensional_feasibility.py"
OUT = HERE / "relaxed_adaptive_4d_best_case_2026-11-17" / "two_selected_5MHz_neighborhoods"
STEP_MHZ = 2
RADIUS_MHZ = 5.0

SELECTIONS = (
    {
        "name": "Original selection",
        "tuple": (154, 208, 216, 252),
        "provenance": "split_campaign_saved_gate_provenance",
    },
    {
        "name": "Relaxed optimum",
        "tuple": (234, 160, 240, 170),
        "provenance": "original_6p00_6p00",
    },
)
SOURCE_ORDER = ("00", "01", "10", "11")
SOURCE_COLORS = ("#E69F00", "#377EB8", "#2CA25F", "#984EA3")


def load_builder():
    spec = importlib.util.spec_from_file_location("strict_four_d", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def offsets():
    values = range(-4, 5, STEP_MHZ)
    return [
        delta for delta in __import__("itertools").product(values, repeat=4)
        if sum(value * value for value in delta) <= RADIUS_MHZ * RADIUS_MHZ + 1e-9
    ]


def project(neighborhood, indices):
    grouped = defaultdict(list)
    for row in neighborhood:
        coordinate = tuple(row["ec_tuple_mhz"][i] for i in indices)
        grouped[coordinate].append(row)
    projected = {}
    for coordinate, rows in grouped.items():
        sources = sorted({str(row["logical_source"]) for row in rows})
        j_values = np.asarray([row["required_interrail_coupling_mhz"] for row in rows], dtype=float)
        projected[coordinate] = {
            "sources": sources,
            "source": sources[0] if len(sources) == 1 else "mixed",
            "hidden_tuple_count": len(rows),
            "j_req_minimum_mhz": float(j_values.min()),
            "j_req_median_mhz": float(np.median(j_values)),
            "j_req_maximum_mhz": float(j_values.max()),
        }
    return projected


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    builder = load_builder()
    maps = {}
    for name, campaign, b_results in (
        ("original_6p00_6p00", builder.ORIGINAL, builder.ORIGINAL_B),
        ("split_campaign_saved_gate_provenance", builder.SPLIT, builder.SPLIT_B),
    ):
        maps[name], _ = builder.load_provenance(name, campaign, b_results)

    required_offsets = offsets()
    results = []
    for selection in SELECTIONS:
        center = selection["tuple"]
        point_map = maps[selection["provenance"]]
        neighborhood = []
        missing = []
        for delta in required_offsets:
            key = tuple(center[i] + delta[i] for i in range(4))
            row = point_map.get(key)
            if row is None:
                missing.append(key)
            else:
                neighborhood.append(row)
        if missing:
            raise RuntimeError(f"{selection['name']} is missing {len(missing)} neighborhood points")
        results.append({
            **selection,
            "neighborhood": neighborhood,
            "rail_a": project(neighborhood, (0, 1)),
            "rail_b": project(neighborhood, (2, 3)),
        })

    all_j = [
        row["required_interrail_coupling_mhz"]
        for result in results for row in result["neighborhood"]
    ]
    j_min, j_max = min(all_j), max(all_j)
    source_cmap = ListedColormap(SOURCE_COLORS + ("#777777",))
    source_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], source_cmap.N)

    fig, axes = plt.subplots(4, 2, figsize=(13.8, 19.0), layout="constrained")
    summary = {
        "schema": "two-selected-5MHz-4d-neighborhood-projections-v1",
        "radius_mhz": RADIUS_MHZ,
        "grid_step_mhz": STEP_MHZ,
        "four_dimensional_offset_count": len(required_offsets),
        "projection_j_req_statistic": "median across hidden-coordinate points",
        "shared_j_req_scale_mhz": [j_min, j_max],
        "selections": [],
    }
    for selection_index, result in enumerate(results):
        selection_summary = {
            "name": result["name"],
            "ec_tuple_mhz": list(result["tuple"]),
            "provenance": result["provenance"],
            "logical_source_counts_in_full_4d_ball": dict(Counter(str(r["logical_source"]) for r in result["neighborhood"])),
            "j_req_minimum_mhz": min(r["required_interrail_coupling_mhz"] for r in result["neighborhood"]),
            "j_req_median_mhz": float(np.median([r["required_interrail_coupling_mhz"] for r in result["neighborhood"]])),
            "j_req_maximum_mhz": max(r["required_interrail_coupling_mhz"] for r in result["neighborhood"]),
            "projections": {},
        }
        for column, (rail, center_indices, labels) in enumerate((
            ("rail_a", (0, 1), (r"$E_{C,A1}/h$ (MHz)", r"$E_{C,A2}/h$ (MHz)")),
            ("rail_b", (2, 3), (r"$E_{C,B1}/h$ (MHz)", r"$E_{C,B2}/h$ (MHz)")),
        )):
            projected = result[rail]
            coordinates = sorted(projected)
            center_2d = tuple(result["tuple"][i] for i in center_indices)
            source_values = [
                SOURCE_ORDER.index(projected[key]["source"])
                if projected[key]["source"] in SOURCE_ORDER else 4
                for key in coordinates
            ]
            row0 = selection_index * 2
            ax = axes[row0, column]
            image = ax.scatter(
                [key[0] for key in coordinates], [key[1] for key in coordinates],
                c=source_values, cmap=source_cmap, norm=source_norm,
                s=88, edgecolors="black", linewidths=0.45,
            )
            colorbar = fig.colorbar(image, ax=ax, ticks=range(5), shrink=0.82)
            colorbar.ax.set_yticklabels(list(SOURCE_ORDER) + ["mixed"])
            colorbar.set_label("CZ logical source")
            ax.set_title(f"{result['name']} — {rail.replace('_', '-').title()} CZ partner")

            ax = axes[row0 + 1, column]
            image = ax.scatter(
                [key[0] for key in coordinates], [key[1] for key in coordinates],
                c=[projected[key]["j_req_median_mhz"] for key in coordinates],
                cmap="viridis", vmin=j_min, vmax=j_max,
                s=88, edgecolors="black", linewidths=0.45,
            )
            fig.colorbar(image, ax=ax, label=r"Median $J_{\rm req}/h$ over hidden coordinates (MHz)", shrink=0.82)
            ax.set_title(result["name"] + " — " + rail.replace("_", "-").title() + r" projected $J_{\rm req}$")

            for ax in axes[row0:row0 + 2, column]:
                ax.scatter(*center_2d, marker="*", s=260, facecolor="#FFF176", edgecolor="#D62728", linewidth=1.8, zorder=5)
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
                ax.set_aspect("equal")
                ax.grid(alpha=0.18)

            selection_summary["projections"][rail] = {
                "coordinate_count": len(projected),
                "mixed_source_coordinate_count": sum(v["source"] == "mixed" for v in projected.values()),
                "maximum_hidden_coordinate_j_req_span_mhz": max(
                    v["j_req_maximum_mhz"] - v["j_req_minimum_mhz"] for v in projected.values()
                ),
            }
        summary["selections"].append(selection_summary)

    fig.suptitle(
        "CZ partner and required coupling within selected 5 MHz-radius 4D neighborhoods\n"
        "Each rail-plane coordinate aggregates only the hidden coordinates inside the same 4D ball",
        fontsize=15,
    )
    png = OUT / "two_selected_neighborhoods_cz_source_and_Jreq.png"
    pdf = OUT / "two_selected_neighborhoods_cz_source_and_Jreq.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    summary["figure_png"] = str(png)
    summary["figure_pdf"] = str(pdf)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
