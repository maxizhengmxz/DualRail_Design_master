#!/usr/bin/env python3
"""Rank calibration-adaptive 4D neighborhoods from retained best-case data."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "build_four_dimensional_feasibility.py"
OUTPUT = HERE / "relaxed_adaptive_4d_best_case_2026-11-17"
RADIUS_MHZ = 5.0
STEP_MHZ = 2
MAX_REPORTED_RADIUS_MHZ = 20.0
MAX_RELATIVE_J_SPREAD = 0.15


def load_builder():
    spec = importlib.util.spec_from_file_location("strict_four_d", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def offsets():
    values = range(-4, 5, STEP_MHZ)
    return [
        (a1, a2, b1, b2)
        for a1 in values
        for a2 in values
        for b1 in values
        for b2 in values
        if a1*a1 + a2*a2 + b1*b1 + b2*b2 <= RADIUS_MHZ**2 + 1e-9
    ]


def analyze(name: str, points: dict):
    keys = np.asarray(list(points), dtype=int)
    lows = keys.min(axis=0) - STEP_MHZ
    highs = keys.max(axis=0) + STEP_MHZ
    shape = tuple(((highs - lows) // STEP_MHZ + 1).tolist())
    mask = np.zeros(shape, dtype=bool)
    indices = ((keys - lows) // STEP_MHZ).astype(int)
    mask[tuple(indices.T)] = True
    distance = distance_transform_edt(mask) * STEP_MHZ
    required = offsets()
    centers = []
    for number, key_array in enumerate(keys, 1):
        key = tuple(map(int, key_array))
        index = tuple(((key_array - lows) // STEP_MHZ).astype(int))
        radius = float(distance[index])
        if radius <= RADIUS_MHZ:
            continue
        neighborhood = [
            points[tuple(key[i] + delta[i] for i in range(4))]
            for delta in required
        ]
        tuning = np.asarray([row["idle_to_gate_tuning_mhz"] for row in neighborhood])
        clearance = np.asarray([row["idle_above_all_interactions_clearance_mhz"] for row in neighborhood])
        raw = np.asarray([row["raw_fidelity"] for row in neighborhood])
        post = np.asarray([row["postselected_fidelity"] for row in neighborhood])
        j_req = np.asarray([row["required_interrail_coupling_mhz"] for row in neighborhood])
        anchors = np.asarray([row["rail_a_anchor_frequency_ghz"] for row in neighborhood])
        j_midpoint = 0.5 * (j_req.max() + j_req.min())
        center = points[key]
        centers.append(
            {
                "provenance": name,
                "ec_tuple_mhz": list(key),
                "logical_source_at_center": center["logical_source"],
                "logical_sources_in_neighborhood": sorted({row["logical_source"] for row in neighborhood}),
                "anchor_ids_in_neighborhood": sorted({row["rail_a_anchor_id"] for row in neighborhood}),
                "four_dimensional_robustness_radius_mhz": min(radius, MAX_REPORTED_RADIUS_MHZ),
                "neighborhood_point_count": len(neighborhood),
                "neighborhood_maximum_tuning_mhz": float(tuning.max()),
                "neighborhood_median_tuning_mhz": float(np.median(tuning)),
                "neighborhood_minimum_clearance_mhz": float(clearance.min()),
                "neighborhood_minimum_raw_fidelity": float(raw.min()),
                "neighborhood_minimum_postselected_fidelity": float(post.min()),
                "neighborhood_minimum_j_req_mhz": float(j_req.min()),
                "neighborhood_median_j_req_mhz": float(np.median(j_req)),
                "neighborhood_maximum_j_req_mhz": float(j_req.max()),
                "neighborhood_j_req_span_mhz": float(j_req.max() - j_req.min()),
                "neighborhood_relative_j_req_spread": float((j_req.max() - j_req.min()) / j_midpoint),
                "neighborhood_anchor_frequency_min_ghz": float(anchors.min()),
                "neighborhood_anchor_frequency_max_ghz": float(anchors.max()),
                "neighborhood_anchor_frequency_span_mhz": float(1000.0 * (anchors.max() - anchors.min())),
            }
        )
        if number % 100000 == 0:
            print(f"[{name}] analyzed {number}/{len(keys)} points", flush=True)
    return centers


def rank_key(row):
    return (
        -row["four_dimensional_robustness_radius_mhz"],
        row["neighborhood_maximum_tuning_mhz"],
        row["neighborhood_relative_j_req_spread"],
        -row["neighborhood_minimum_clearance_mhz"],
        -row["neighborhood_minimum_raw_fidelity"],
        -row["neighborhood_minimum_postselected_fidelity"],
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    builder = load_builder()
    centers = []
    audits = {}
    for name, campaign, b_results in (
        ("original_6p00_6p00", builder.ORIGINAL, builder.ORIGINAL_B),
        ("split_campaign_saved_gate_provenance", builder.SPLIT, builder.SPLIT_B),
    ):
        points, audit = builder.load_provenance(name, campaign, b_results)
        audits[name] = audit
        centers.extend(analyze(name, points))
    if not centers:
        raise RuntimeError("No relaxed 5 MHz 4D neighborhoods")
    bounded = [row for row in centers if row["neighborhood_relative_j_req_spread"] <= MAX_RELATIVE_J_SPREAD]
    if not bounded:
        raise RuntimeError("No candidate satisfies the provisional J_req-spread bound")
    ranked = sorted(bounded, key=rank_key)
    best = ranked[0]

    fields = list(centers[0])
    with (OUTPUT / "relaxed_centers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in centers:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    summary = {
        "schema": "calibration-adaptive-4d-best-case-v1",
        "created_from": str(SOURCE.resolve()),
        "historical_provenance_warning": (
            "The retained best-case split partition predates corrected downstream cavity propagation; "
            "rankings are provisional until corrected-cavity reruns."
        ),
        "hard_requirements_retained": [
            "independent readout feasibility for both rails",
            "raw fidelity > 0.95",
            "postselected fidelity > 0.999",
            "successful non-boundary calibration",
            "downward rail-B tuning",
            "idle above every retained interaction by >200 MHz",
            "complete 233-point 5 MHz 4D neighborhood",
        ],
        "requirements_removed": [
            "same logical source throughout neighborhood",
            "nonempty joint rail-A connected interval",
            "stored anchor inside joint interval",
            "same anchor throughout neighborhood",
        ],
        "provisional_relative_j_req_spread_limit": MAX_RELATIVE_J_SPREAD,
        "ranking": [
            "largest 4D hard-valid radius",
            "smallest neighborhood maximum rail-B tuning",
            "smallest relative J_req spread",
            "largest neighborhood minimum clearance",
            "largest neighborhood minimum raw then postselected fidelity",
        ],
        "audit": audits,
        "relaxed_center_count": len(centers),
        "centers_with_relative_j_req_spread_at_most_15_percent": len(bounded),
        "best": best,
        "top_20": ranked[:20],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), layout="constrained")
    for ax, pair, title in (
        (axes[0], (0, 1), "Rail-A projection"),
        (axes[1], (2, 3), "Rail-B projection"),
    ):
        grouped = {}
        for row in bounded:
            coordinate = tuple(row["ec_tuple_mhz"][index] for index in pair)
            current = grouped.get(coordinate)
            if current is None or rank_key(row) < rank_key(current):
                grouped[coordinate] = row
        image = ax.scatter(
            [key[0] for key in grouped], [key[1] for key in grouped],
            c=[row["neighborhood_maximum_tuning_mhz"] for row in grouped.values()],
            cmap="magma_r", s=24, edgecolors="black", linewidths=.2,
        )
        selected = tuple(best["ec_tuple_mhz"][index] for index in pair)
        ax.scatter(*selected, marker="*", s=240, facecolor="#fff176", edgecolor="#d62728", linewidth=1.8)
        ax.annotate(str(selected), selected, xytext=(7, 7), textcoords="offset points", fontsize=9,
                    bbox={"boxstyle":"round,pad=0.2", "facecolor":"white", "alpha":.8})
        fig.colorbar(image, ax=ax, label="Neighborhood maximum rail-B tuning (MHz)", shrink=.82)
        ax.set(title=title, xlabel=r"$E_{C,1}/h$ (MHz)", ylabel=r"$E_{C,2}/h$ (MHz)")
        ax.set_aspect("equal")
    fig.suptitle("Relaxed calibration-adaptive 4D best-case ranking", fontsize=14)
    fig.savefig(OUTPUT / "relaxed_adaptive_4d_selection.png", dpi=220)
    fig.savefig(OUTPUT / "relaxed_adaptive_4d_selection.pdf")
    plt.close(fig)
    print(json.dumps({"output": str(OUTPUT), "centers": len(centers), "bounded": len(bounded), "best": best}, indent=2))


if __name__ == "__main__":
    main()
