#!/usr/bin/env python3
"""Build a read-only 4D EC feasibility map from retained campaign records."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "sampled_runs" / "rail_a_gt_partition_2026-11-06"
SPLIT = HERE / "cavity_split_campaign_2026-11-09" / "sampled_runs" / "rail_a_lt_partition"
ORIGINAL_B = HERE / "readout" / "direct_ec" / "rail_b_step2" / "results.json"
SPLIT_B = HERE / "cavity_split_campaign_2026-11-09" / "readout" / "direct_ec" / "rail_b_step2_fr5p95" / "results.json"
OUTPUT = HERE / "four_dimensional_feasibility_joint_anchor_range_5MHz_2026-11-12"
STEP_MHZ = 2
RAW_MIN = .95
POST_MIN = .999
CLEARANCE_MIN_MHZ = 200.0


def offsets(radius_mhz: float) -> list[tuple[float, tuple[int, int, int, int]]]:
    limit = math.ceil(radius_mhz / STEP_MHZ)
    values = []
    for a1 in range(-limit, limit + 1):
        for a2 in range(-limit, limit + 1):
            for b1 in range(-limit, limit + 1):
                for b2 in range(-limit, limit + 1):
                    distance = STEP_MHZ * math.sqrt(a1*a1 + a2*a2 + b1*b1 + b2*b2)
                    if distance <= radius_mhz + 1e-9:
                        values.append((distance, (a1*STEP_MHZ, a2*STEP_MHZ, b1*STEP_MHZ, b2*STEP_MHZ)))
    return sorted(values, key=lambda item: (item[0], item[1]))


def idle_lookup(path: Path) -> dict[tuple[int, int], dict]:
    return {
        (round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"])): row["point"]
        for row in json.loads(path.read_text())
        if row["status"] == "validated_pass"
    }


def retained_interactions(result: dict) -> list[float]:
    frequencies = []
    for screen in result.get("screening") or []:
        for candidate in screen.get("logical_source_candidates", []):
            if candidate.get("status") == "valid" and candidate.get("fB_GHz") is not None:
                frequencies.append(float(candidate["fB_GHz"]))
    return frequencies


def load_provenance(name: str, campaign: Path, b_results: Path) -> tuple[dict, dict]:
    idle = idle_lookup(b_results)
    status = json.loads((campaign / "campaign_status.json").read_text())
    points = {}
    audit = Counter()
    complete_jobs = [job for job in status["jobs"] if job["state"] == "complete"]
    for job_index, job in enumerate(complete_jobs, 1):
        directory = Path(job["directory"])
        anchor_selection_path = directory / "anchor_selection.json"
        if not anchor_selection_path.exists():
            audit["missing_anchor_selection"] += 1
            continue
        anchor_selection = json.loads(anchor_selection_path.read_text())
        common_range = anchor_selection.get("eligible_anchor_frequency_ghz")
        if common_range is None:
            common_range = anchor_selection.get("largest_sampled_component_ghz")
        if common_range is None or len(common_range) != 2:
            audit["missing_connected_anchor_range"] += 1
            continue
        range_low, range_high = map(float, common_range)
        database = directory / "gate_output" / "compact_anchor_results.sqlite"
        if not database.exists():
            audit["missing_gate_database"] += 1
            continue
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        saved = connection.execute(
            "SELECT ec_b1_mhz,ec_b2_mhz,status,raw_fidelity,postselected_fidelity,"
            "selected_anchor_id,result_json FROM gate_results"
        ).fetchall()
        connection.close()
        a1, a2 = map(round, job["ec_a_mhz"])
        for b1_value, b2_value, status_value, raw_column, post_column, anchor_column, text in saved:
            b = (round(b1_value), round(b2_value))
            idle_point = idle.get(b)
            if idle_point is None:
                audit["rail_B_readout_invalid"] += 1
                continue
            if status_value != "calibrated" or text is None:
                audit["gate_not_calibrated"] += 1
                continue
            result = json.loads(text)
            final = result.get("final") or {}
            raw = float(final.get("raw_fidelity", raw_column or float("nan")))
            post = float(final.get("postselected_fidelity", post_column or float("nan")))
            if not raw > RAW_MIN or not post > POST_MIN:
                audit["fidelity_failure"] += 1
                continue
            if any(bool(result.get(key)) for key in (
                "frequency_boundary", "calibration_time_boundary", "calibration_detuning_boundary"
            )):
                audit["calibration_boundary"] += 1
                continue
            interactions = retained_interactions(result)
            if not interactions:
                audit["missing_interaction_diagnostics"] += 1
                continue
            idle_bare = [float(idle_point["bare_f1_ghz"]), float(idle_point["bare_f2_ghz"])]
            clearance = 1000.0 * (min(idle_bare) - max(interactions))
            if not clearance > CLEARANCE_MIN_MHZ:
                audit["signed_clearance_failure"] += 1
                continue
            gate_frequency = float(final["fB_GHz"])
            tuning = 1000.0 * (float(idle_point["operating_frequency_ghz"]) - gate_frequency)
            if tuning <= 0:
                audit["non_downward_tuning"] += 1
                continue
            anchor = result.get("selected_anchor", {}).get("anchor_id", anchor_column)
            source = result.get("logical_source") or final.get("logical_source")
            selected_anchor = result.get("selected_anchor") or {}
            anchor_frequency = selected_anchor.get("readout_operating_frequency_GHz")
            if anchor is None or source is None or anchor_frequency is None:
                audit["missing_policy_identity"] += 1
                continue
            required_interrail_coupling = (final.get("couplings_GHz") or {}).get("A2_B1")
            if required_interrail_coupling is None:
                audit["missing_required_interrail_coupling"] += 1
                continue
            key = (a1, a2, b[0], b[1])
            points[key] = {
                "provenance": name,
                "ec_tuple_mhz": list(key),
                "selection_id": job["selection_id"],
                "rail_a_anchor_id": anchor,
                "rail_a_anchor_frequency_ghz": float(anchor_frequency),
                "rail_a_connected_operating_range_ghz": [range_low, range_high],
                "logical_source": source,
                "raw_fidelity": raw,
                "postselected_fidelity": post,
                "gate_frequency_ghz": gate_frequency,
                "idle_operating_frequency_ghz": float(idle_point["operating_frequency_ghz"]),
                "idle_to_gate_tuning_mhz": tuning,
                "idle_above_all_interactions_clearance_mhz": clearance,
                "maximum_interaction_frequency_ghz": max(interactions),
                "required_interrail_coupling_mhz": 1000.0 * float(required_interrail_coupling),
                "source_repetition": str(directory),
            }
            audit["hard_valid_points"] += 1
        if job_index % 25 == 0:
            print(f"[{name}] loaded {job_index}/{len(complete_jobs)} rail-A repetitions", flush=True)
    return points, dict(audit)


def ball_centers(
    points: dict, radius_mhz: float, maximum_gate_span_mhz: float
) -> tuple[list[dict], dict]:
    required_offsets = offsets(radius_mhz)
    centers = []
    audit = Counter()
    rail_b_stages = {
        "has_individually_hard_valid_tuple": set(),
        "has_complete_hard_valid_ball": set(),
        "has_nonempty_joint_anchor_range_ball": set(),
        "has_selected_anchors_inside_joint_range_ball": set(),
        "has_logical_source_consistent_ball": set(),
        "has_fully_valid_ball": set(),
    }
    rail_a_stages = {
        "has_individually_hard_valid_tuple": set(),
        "has_complete_hard_valid_ball": set(),
        "has_nonempty_joint_anchor_range_ball": set(),
        "has_selected_anchors_inside_joint_range_ball": set(),
        "has_logical_source_consistent_ball": set(),
        "has_fully_valid_ball": set(),
    }
    for index, (key, center) in enumerate(points.items(), 1):
        rail_b_coordinate = key[2:]
        rail_b_stages["has_individually_hard_valid_tuple"].add(rail_b_coordinate)
        rail_a = key[:2]
        rail_a_stages["has_individually_hard_valid_tuple"].add(rail_a)
        neighborhood = {}
        source = center["logical_source"]
        for _, delta in required_offsets:
            neighbor_key = tuple(key[i] + delta[i] for i in range(4))
            neighbor = points.get(neighbor_key)
            if neighbor is None:
                neighborhood = {}
                break
            neighborhood[neighbor_key] = neighbor
        if not neighborhood:
            audit["missing_hard_valid_neighbor"] += 1
            continue
        audit["complete_hard_valid_ball"] += 1
        rail_b_stages["has_complete_hard_valid_ball"].add(rail_b_coordinate)
        rail_a_stages["has_complete_hard_valid_ball"].add(rail_a)
        joint_low = max(
            neighbor["rail_a_connected_operating_range_ghz"][0]
            for neighbor in neighborhood.values()
        )
        joint_high = min(
            neighbor["rail_a_connected_operating_range_ghz"][1]
            for neighbor in neighborhood.values()
        )
        if joint_low > joint_high + 1e-9:
            audit["empty_joint_anchor_range_ball"] += 1
            continue
        audit["nonempty_joint_anchor_range_ball"] += 1
        rail_b_stages["has_nonempty_joint_anchor_range_ball"].add(rail_b_coordinate)
        rail_a_stages["has_nonempty_joint_anchor_range_ball"].add(rail_a)
        if any(
            not joint_low - 1e-9
            <= neighbor["rail_a_anchor_frequency_ghz"]
            <= joint_high + 1e-9
            for neighbor in neighborhood.values()
        ):
            audit["selected_anchor_outside_joint_range_ball"] += 1
            continue
        audit["selected_anchors_inside_joint_range_ball"] += 1
        rail_b_stages["has_selected_anchors_inside_joint_range_ball"].add(rail_b_coordinate)
        rail_a_stages["has_selected_anchors_inside_joint_range_ball"].add(rail_a)
        if any(neighbor["logical_source"] != source for neighbor in neighborhood.values()):
            audit["logical_source_inconsistent_ball"] += 1
            continue
        audit["logical_source_consistent_ball"] += 1
        rail_b_stages["has_logical_source_consistent_ball"].add(rail_b_coordinate)
        rail_a_stages["has_logical_source_consistent_ball"].add(rail_a)
        frequencies = [row["gate_frequency_ghz"] for row in neighborhood.values()]
        gate_span = 1000.0 * (max(frequencies) - min(frequencies))
        edge_jumps = []
        for neighbor_key, neighbor in neighborhood.items():
            for axis in range(4):
                adjacent_key = tuple(
                    neighbor_key[i] + (STEP_MHZ if i == axis else 0)
                    for i in range(4)
                )
                adjacent = neighborhood.get(adjacent_key)
                if adjacent is not None:
                    edge_jumps.append(
                        1000.0 * abs(
                            neighbor["gate_frequency_ghz"]
                            - adjacent["gate_frequency_ghz"]
                        )
                    )
        maximum_edge_jump = max(edge_jumps, default=0.0)
        if maximum_edge_jump > maximum_gate_span_mhz:
            audit["gate_frequency_discontinuous_ball"] += 1
            continue
        audit["fully_valid_ball"] += 1
        rail_b_stages["has_fully_valid_ball"].add(rail_b_coordinate)
        rail_a_stages["has_fully_valid_ball"].add(rail_a)
        row = dict(center)
        row.update(
            ball_radius_mhz=radius_mhz,
            ball_grid_point_count=len(neighborhood),
            joint_rail_a_operating_range_ghz=[joint_low, joint_high],
            joint_rail_a_operating_range_width_mhz=1000.0 * (joint_high - joint_low),
            neighborhood_gate_frequency_span_mhz=gate_span,
            neighborhood_maximum_adjacent_gate_frequency_jump_mhz=maximum_edge_jump,
            neighborhood_maximum_idle_to_gate_tuning_mhz=max(
                item["idle_to_gate_tuning_mhz"] for item in neighborhood.values()
            ),
            neighborhood_minimum_idle_to_gate_tuning_mhz=min(
                item["idle_to_gate_tuning_mhz"] for item in neighborhood.values()
            ),
            neighborhood_median_idle_to_gate_tuning_mhz=float(np.median(
                [item["idle_to_gate_tuning_mhz"] for item in neighborhood.values()]
            )),
            neighborhood_minimum_clearance_mhz=min(
                item["idle_above_all_interactions_clearance_mhz"] for item in neighborhood.values()
            ),
            neighborhood_minimum_raw_fidelity=min(item["raw_fidelity"] for item in neighborhood.values()),
            neighborhood_minimum_postselected_fidelity=min(
                item["postselected_fidelity"] for item in neighborhood.values()
            ),
            neighborhood_median_raw_fidelity=float(np.median(
                [item["raw_fidelity"] for item in neighborhood.values()]
            )),
            neighborhood_median_postselected_fidelity=float(np.median(
                [item["postselected_fidelity"] for item in neighborhood.values()]
            )),
        )
        centers.append(row)
        if index % 100000 == 0:
            print(f"screened {index}/{len(points)} 4D centers", flush=True)
    audit["rail_a_candidate_counts"] = {
        stage: len(coordinates) for stage, coordinates in rail_a_stages.items()
    }
    audit["rail_a_candidates_by_stage"] = {
        stage: [list(coordinate) for coordinate in sorted(coordinates)]
        for stage, coordinates in rail_a_stages.items()
    }
    audit["rail_b_candidate_counts"] = {
        stage: len(coordinates) for stage, coordinates in rail_b_stages.items()
    }
    audit["rail_b_candidates_by_stage"] = {
        stage: [list(coordinate) for coordinate in sorted(coordinates)]
        for stage, coordinates in rail_b_stages.items()
    }
    return centers, dict(audit)


def robustness_radius(
    center: dict, points: dict, maximum_gate_span_mhz: float, maximum_radius_mhz: float
) -> float:
    key = tuple(center["ec_tuple_mhz"])
    source = center["logical_source"]
    accumulated = {key: center}
    anchor_frequencies = [center["rail_a_anchor_frequency_ghz"]]
    joint_low, joint_high = center["rail_a_connected_operating_range_ghz"]
    for distance, delta in offsets(maximum_radius_mhz):
        neighbor_key = tuple(key[i] + delta[i] for i in range(4))
        neighbor = points.get(neighbor_key)
        if neighbor is None or neighbor["logical_source"] != source:
            return distance
        joint_low = max(joint_low, neighbor["rail_a_connected_operating_range_ghz"][0])
        joint_high = min(joint_high, neighbor["rail_a_connected_operating_range_ghz"][1])
        if joint_low > joint_high + 1e-9:
            return distance
        anchor_frequencies.append(neighbor["rail_a_anchor_frequency_ghz"])
        if any(not joint_low - 1e-9 <= frequency <= joint_high + 1e-9 for frequency in anchor_frequencies):
            return distance
        for axis in range(4):
            for sign in (-1, 1):
                adjacent_key = tuple(
                    neighbor_key[i] + (sign * STEP_MHZ if i == axis else 0)
                    for i in range(4)
                )
                adjacent = accumulated.get(adjacent_key)
                if adjacent is not None and 1000.0 * abs(
                    neighbor["gate_frequency_ghz"] - adjacent["gate_frequency_ghz"]
                ) > maximum_gate_span_mhz:
                    return distance
        accumulated[neighbor_key] = neighbor
    return maximum_radius_mhz


def projection_plot(centers: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), layout="constrained")
    for ax, indices, title, labels in (
        (axes[0], (0,1), "Rail-A projection of 4D-feasible centers", (r"$E_{C,A1}/h$ (MHz)",r"$E_{C,A2}/h$ (MHz)")),
        (axes[1], (2,3), "Rail-B projection of 4D-feasible centers", (r"$E_{C,B1}/h$ (MHz)",r"$E_{C,B2}/h$ (MHz)")),
    ):
        grouped = Counter((row["ec_tuple_mhz"][indices[0]],row["ec_tuple_mhz"][indices[1]]) for row in centers)
        values = list(grouped.values())
        image = ax.scatter([x for x,y in grouped],[y for x,y in grouped],c=values,cmap="viridis",s=28,
                           norm=Normalize(min(values),max(values)),edgecolors="black",linewidths=.2)
        fig.colorbar(image,ax=ax,label="Number of valid 4D centers in projection",shrink=.82)
        ax.set(title=title,xlabel=labels[0],ylabel=labels[1]);ax.set_aspect("equal")
    fig.suptitle("Strict 5 MHz-radius 4D EC feasibility map",fontsize=13)
    fig.savefig(output/"four_dimensional_feasibility_projections.png",dpi=220)
    fig.savefig(output/"four_dimensional_feasibility_projections.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-mhz", type=float, default=5.0)
    parser.add_argument("--maximum-gate-frequency-span-mhz", type=float, default=25.0)
    parser.add_argument("--maximum-reported-robustness-radius-mhz", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    configurations = [
        ("original_6p00_6p00", ORIGINAL, ORIGINAL_B),
        ("split_campaign_saved_gate_provenance", SPLIT, SPLIT_B),
    ]
    all_centers = []
    audits = {}
    point_counts = {}
    for name, campaign, b_results in configurations:
        points, audit = load_provenance(name, campaign, b_results)
        audits[name] = audit; point_counts[name] = len(points)
        centers, ball_audit = ball_centers(
            points, args.radius_mhz, args.maximum_gate_frequency_span_mhz
        )
        audits[name]["ball_audit"] = ball_audit
        for center in centers:
            center["four_dimensional_robustness_radius_mhz"] = robustness_radius(
                center, points, args.maximum_gate_frequency_span_mhz,
                args.maximum_reported_robustness_radius_mhz,
            )
        all_centers.extend(centers)
        print(f"[{name}] hard-valid={len(points)}, 5 MHz ball centers={len(centers)}", flush=True)
    if not all_centers:
        summary = {
            "schema":"four-dimensional-ec-feasibility-v1",
            "grid_step_mhz":STEP_MHZ,
            "ball_radius_mhz":args.radius_mhz,
            "maximum_gate_frequency_span_mhz":args.maximum_gate_frequency_span_mhz,
            "hard_valid_point_counts":point_counts,
            "audit_counts":audits,
            "five_mhz_ball_center_count":0,
            "result":"no strict 4D ball center",
        }
        (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps({"output":str(output),**summary},indent=2))
        return
    tuning_best = min(all_centers,key=lambda row:(row["neighborhood_maximum_idle_to_gate_tuning_mhz"],-row["four_dimensional_robustness_radius_mhz"]))
    robustness_best = max(all_centers,key=lambda row:(row["four_dimensional_robustness_radius_mhz"],-row["neighborhood_maximum_idle_to_gate_tuning_mhz"]))
    fields = [
        "provenance","ec_tuple_mhz","rail_a_anchor_id","rail_a_anchor_frequency_ghz",
        "joint_rail_a_operating_range_ghz","joint_rail_a_operating_range_width_mhz",
        "logical_source","ball_radius_mhz",
        "ball_grid_point_count","four_dimensional_robustness_radius_mhz",
        "neighborhood_gate_frequency_span_mhz","neighborhood_maximum_adjacent_gate_frequency_jump_mhz",
        "neighborhood_minimum_idle_to_gate_tuning_mhz",
        "neighborhood_maximum_idle_to_gate_tuning_mhz",
        "neighborhood_median_idle_to_gate_tuning_mhz","neighborhood_minimum_clearance_mhz",
        "neighborhood_minimum_raw_fidelity","neighborhood_minimum_postselected_fidelity",
        "neighborhood_median_raw_fidelity","neighborhood_median_postselected_fidelity",
    ]
    with (output/"four_dimensional_feasible_centers.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in all_centers:
            writer.writerow({key:json.dumps(row[key]) if isinstance(row[key],list) else row[key] for key in fields})
    summary = {
        "schema":"four-dimensional-ec-feasibility-v1",
        "grid_step_mhz":STEP_MHZ,
        "ball_radius_mhz":args.radius_mhz,
        "maximum_gate_frequency_span_mhz":args.maximum_gate_frequency_span_mhz,
        "hard_requirements":{
            "independent_readout_both_rails":True,"raw_fidelity_strictly_greater_than":RAW_MIN,
            "postselected_fidelity_strictly_greater_than":POST_MIN,
            "idle_above_all_retained_interactions_strictly_greater_than_mhz":CLEARANCE_MIN_MHZ,
            "nonempty_joint_rail_a_operating_range_throughout_ball":True,
            "every_selected_anchor_inside_joint_range":True,
            "same_logical_source_throughout_ball":True,
            "calibration_boundary_forbidden":True,"numerically_complete_calibrated_record_required":True,
        },
        "provenance_warning":"Historical split-cavity gate anchors retain the known downstream 6.00 GHz provenance error; this map is exploratory, not corrected-cavity production certification.",
        "hard_valid_point_counts":point_counts,"audit_counts":audits,
        "five_mhz_ball_center_count":len(all_centers),
        "tuning_optimal":tuning_best,"robustness_optimal":robustness_best,
    }
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    projection_plot(all_centers,output)
    print(json.dumps({"output":str(output),"centers":len(all_centers),
                      "tuning_optimal":{k:tuning_best[k] for k in ("ec_tuple_mhz","provenance","rail_a_anchor_id","logical_source","neighborhood_maximum_idle_to_gate_tuning_mhz","four_dimensional_robustness_radius_mhz")},
                      "robustness_optimal":{k:robustness_best[k] for k in ("ec_tuple_mhz","provenance","rail_a_anchor_id","logical_source","neighborhood_maximum_idle_to_gate_tuning_mhz","four_dimensional_robustness_radius_mhz")}},indent=2))


if __name__ == "__main__": main()
