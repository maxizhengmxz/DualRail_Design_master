#!/usr/bin/env python3
"""Find low-J_req robust 4D tuples in the upper-left rail-A island."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "relaxed_adaptive_4d_best_case_2026-11-17" / "relaxed_centers.csv"
OUTPUT = HERE / "relaxed_adaptive_4d_best_case_2026-11-17" / "upper_left_low_jreq_candidates.json"

NUMERIC = (
    "four_dimensional_robustness_radius_mhz",
    "neighborhood_maximum_tuning_mhz",
    "neighborhood_median_tuning_mhz",
    "neighborhood_minimum_clearance_mhz",
    "neighborhood_minimum_raw_fidelity",
    "neighborhood_minimum_postselected_fidelity",
    "neighborhood_minimum_j_req_mhz",
    "neighborhood_median_j_req_mhz",
    "neighborhood_maximum_j_req_mhz",
    "neighborhood_relative_j_req_spread",
)


def compact(row):
    return {
        "ec_tuple_mhz": list(row["ec_tuple_mhz"]),
        "provenance": row["provenance"],
        "logical_source": row["logical_source_at_center"],
        "robustness_radius_mhz": row["four_dimensional_robustness_radius_mhz"],
        "rail_a_projection_inward_margin_mhz": row.get("rail_a_projection_inward_margin_mhz"),
        "j_req_neighborhood_mhz": {
            "minimum": row["neighborhood_minimum_j_req_mhz"],
            "median": row["neighborhood_median_j_req_mhz"],
            "maximum": row["neighborhood_maximum_j_req_mhz"],
            "relative_spread": row["neighborhood_relative_j_req_spread"],
        },
        "maximum_rail_b_tuning_mhz": row["neighborhood_maximum_tuning_mhz"],
        "minimum_clearance_mhz": row["neighborhood_minimum_clearance_mhz"],
        "minimum_raw_fidelity": row["neighborhood_minimum_raw_fidelity"],
        "minimum_postselected_fidelity": row["neighborhood_minimum_postselected_fidelity"],
    }


def main():
    rows = []
    with SOURCE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            row["ec_tuple_mhz"] = tuple(ast.literal_eval(row["ec_tuple_mhz"]))
            for key in NUMERIC:
                row[key] = float(row[key])
            a1, a2, _, _ = row["ec_tuple_mhz"]
            if a1 <= 170 and a2 >= 235 and row["neighborhood_relative_j_req_spread"] <= 0.15:
                rows.append(row)

    # Measure distance to the boundary of the retained upper-left rail-A
    # projection on the 2 MHz lattice.  The raw EDT gives a boundary point
    # 2 MHz, so subtract one lattice step to report inward margin.
    a_coordinates = sorted({row["ec_tuple_mhz"][:2] for row in rows})
    a1_values = range(min(x for x, _ in a_coordinates), max(x for x, _ in a_coordinates) + 2, 2)
    a2_values = range(min(y for _, y in a_coordinates), max(y for _, y in a_coordinates) + 2, 2)
    a1_index = {value: index for index, value in enumerate(a1_values)}
    a2_index = {value: index for index, value in enumerate(a2_values)}
    mask = np.zeros((len(a2_index), len(a1_index)), dtype=bool)
    for a1, a2 in a_coordinates:
        mask[a2_index[a2], a1_index[a1]] = True
    padded = np.pad(mask, 1, constant_values=False)
    depth = distance_transform_edt(padded)[1:-1, 1:-1] * 2.0 - 2.0
    for row in rows:
        a1, a2 = row["ec_tuple_mhz"][:2]
        row["rail_a_projection_inward_margin_mhz"] = max(0.0, float(depth[a2_index[a2], a1_index[a1]]))

    robust = [row for row in rows if row["four_dimensional_robustness_radius_mhz"] >= 10.0]
    interior = [row for row in rows if row["rail_a_projection_inward_margin_mhz"] >= 6.0]
    by_low_j = sorted(
        robust,
        key=lambda row: (
            row["neighborhood_median_j_req_mhz"],
            row["neighborhood_maximum_j_req_mhz"],
            row["neighborhood_maximum_tuning_mhz"],
            -row["neighborhood_minimum_clearance_mhz"],
        ),
    )
    by_balanced = sorted(
        interior,
        key=lambda row: (
            row["neighborhood_maximum_j_req_mhz"],
            row["neighborhood_maximum_tuning_mhz"],
            -row["four_dimensional_robustness_radius_mhz"],
            -row["neighborhood_minimum_clearance_mhz"],
        ),
    )
    result = {
        "schema": "upper-left-low-jreq-candidates-v1",
        "filter": {
            "ec_a1_at_most_mhz": 170,
            "ec_a2_at_least_mhz": 235,
            "relative_j_req_spread_at_most": 0.15,
            "robustness_radius_at_least_mhz": 10.0,
        },
        "eligible_4d_center_count_before_radius": len(rows),
        "eligible_4d_center_count_after_radius": len(robust),
        "eligible_4d_center_count_after_6mhz_rail_a_inward_margin_without_10mhz_4d_radius_requirement": len(interior),
        "distinct_rail_a_pairs_after_radius": len({row["ec_tuple_mhz"][:2] for row in robust}),
        "lowest_median_j_req": [compact(row) for row in by_low_j[:20]],
        "lowest_neighborhood_maximum_j_req": [compact(row) for row in by_balanced[:20]],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
