#!/usr/bin/env python3
"""Select robust rail-B designs and plot a completed worst-anchor campaign."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt

RAW_MIN = 0.95
POST_MIN = 0.999
RADIUS_MHZ = 5.0
MIN_MARGIN_MHZ = 5.0
STEP_MHZ = 2
CLEARANCE_MIN_MHZ = 200.0


def readout(path: Path) -> tuple[dict[tuple[int, int], dict], list[tuple[int, int]]]:
    rows = json.loads(path.read_text())
    valid = {
        (round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"])): row["point"]
        for row in rows
        if row["status"] == "validated_pass"
    }
    domain = [
        (round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"]))
        for row in rows
    ]
    return valid, domain


def retained_interaction_frequencies(result: dict) -> list[float]:
    """Collect every valid logical-source collision retained by screening."""
    frequencies = []
    for screen in result.get("screening") or []:
        for candidate in screen.get("logical_source_candidates", []):
            if (
                candidate.get("status") == "valid"
                and candidate.get("fB_GHz") is not None
            ):
                frequencies.append(float(candidate["fB_GHz"]))
    return frequencies


def choose(directory: Path, idle: dict, domain: list[tuple[int, int]]) -> tuple[dict, dict]:
    database = directory / "gate_output/compact_anchor_results.sqlite"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    saved = connection.execute(
        """
        SELECT ec_b1_mhz,ec_b2_mhz,raw_fidelity,postselected_fidelity,
               selected_anchor_id,
               json_extract(result_json,'$.final.fB_GHz'),
               json_extract(result_json,'$.selected_anchor.readout_operating_frequency_GHz'),
               json_extract(result_json,'$.anchor_ranking'),result_json
        FROM gate_results
        WHERE raw_fidelity>? AND postselected_fidelity>?
        """,
        (RAW_MIN, POST_MIN),
    ).fetchall()
    connection.close()
    candidates = []
    rejection_counts = Counter()
    for ec1, ec2, raw, post, anchor, gate_frequency, anchor_frequency, ranking, result_text in saved:
        ec = (round(ec1), round(ec2))
        point = idle.get(ec)
        if point is None or anchor is None or gate_frequency is None:
            rejection_counts["missing_readout_or_gate_metadata"] += 1
            continue
        result = json.loads(result_text)
        interactions = retained_interaction_frequencies(result)
        if not interactions:
            rejection_counts["missing_interaction_diagnostics"] += 1
            continue
        idle_bare = [float(point["bare_f1_ghz"]), float(point["bare_f2_ghz"])]
        clearance = 1000.0 * (min(idle_bare) - max(interactions))
        if clearance <= CLEARANCE_MIN_MHZ:
            rejection_counts["signed_clearance_failure"] += 1
            continue
        idle_to_gate = 1000.0 * (
            float(point["operating_frequency_ghz"]) - float(gate_frequency)
        )
        if idle_to_gate <= 0:
            rejection_counts["non_downward_tuning"] += 1
            continue
        shifts = [
            1000 * (gate_frequency - point["bare_f1_ghz"]),
            1000 * (gate_frequency - point["bare_f2_ghz"]),
        ]
        candidates.append(
            {
                "ec_b_mhz": list(ec),
                "raw_fidelity": raw,
                "postselected_fidelity": post,
                "selected_anchor_id": anchor,
                "selected_anchor_readout_frequency_ghz": anchor_frequency,
                "anchor_ranking": ranking,
                "bare_frequency_shift_b1_mhz": shifts[0],
                "bare_frequency_shift_b2_mhz": shifts[1],
                "maximum_absolute_bare_frequency_shift_mhz": max(map(abs, shifts)),
                "idle_above_all_interactions_clearance_mhz": clearance,
                "maximum_interaction_frequency_ghz": max(interactions),
                "idle_to_gate_operating_frequency_tuning_mhz": idle_to_gate,
                "logical_source": result.get("logical_source"),
                "idle_J_B_GHz": point["J12_ghz"],
                "idle_logical_splitting_GHz": point["splitting_ghz"],
            }
        )
    if not candidates:
        raise RuntimeError("no fidelity-passing readout/gate overlay")
    xs = np.arange(min(x for x, _ in domain), max(x for x, _ in domain) + STEP_MHZ, STEP_MHZ)
    ys = np.arange(min(y for _, y in domain), max(y for _, y in domain) + STEP_MHZ, STEP_MHZ)
    xi = {round(value): index for index, value in enumerate(xs)}
    yi = {round(value): index for index, value in enumerate(ys)}
    by_ec = {tuple(row["ec_b_mhz"]): row for row in candidates}
    margins = {}
    for anchor in {row["selected_anchor_id"] for row in candidates}:
        mask = np.zeros((len(ys), len(xs)), bool)
        for row in candidates:
            if row["selected_anchor_id"] == anchor:
                x, y = row["ec_b_mhz"]
                mask[yi[y], xi[x]] = True
        margins[anchor] = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * STEP_MHZ
    offsets = [
        (dx, dy)
        for dx in range(-4, 5, STEP_MHZ)
        for dy in range(-4, 5, STEP_MHZ)
        if dx * dx + dy * dy <= RADIUS_MHZ**2 + 1e-9
    ]
    for row in candidates:
        x, y = row["ec_b_mhz"]
        anchor = row["selected_anchor_id"]
        row["anchor_consistent_robustness_margin_mhz"] = float(
            margins[anchor][yi[y], xi[x]]
        )
        neighborhood = [
            other
            for dx, dy in offsets
            if (other := by_ec.get((x + dx, y + dy))) is not None
            and other["selected_anchor_id"] == anchor
        ]
        raw = [other["raw_fidelity"] for other in neighborhood]
        post = [other["postselected_fidelity"] for other in neighborhood]
        shifts = [other["maximum_absolute_bare_frequency_shift_mhz"] for other in neighborhood]
        row.update(
            neighborhood_point_count=len(neighborhood),
            neighborhood_raw_fidelity_median=float(np.median(raw)),
            neighborhood_postselected_fidelity_median=float(np.median(post)),
            neighborhood_maximum_absolute_bare_frequency_shift_mhz=float(max(shifts)),
        )
    robust = [
        row
        for row in candidates
        if row["anchor_consistent_robustness_margin_mhz"] >= MIN_MARGIN_MHZ
    ]
    selected = min(
        robust or candidates,
        key=lambda row: (
            row["neighborhood_maximum_absolute_bare_frequency_shift_mhz"],
            -row["neighborhood_raw_fidelity_median"],
            -row["neighborhood_postselected_fidelity_median"],
            -row["anchor_consistent_robustness_margin_mhz"],
        ),
    )
    return selected, {
        "overlay_points": len(candidates),
        "points_meeting_same_anchor_margin": len(robust),
        "margin_fallback_used": not bool(robust),
        "rejection_counts": dict(rejection_counts),
    }


def background(path: Path) -> tuple[list[int], list[int], np.ndarray]:
    rows = json.loads(path.read_text())
    xs = sorted({round(row["ec1_target_mhz"]) for row in rows})
    ys = sorted({round(row["ec2_target_mhz"]) for row in rows})
    grid = np.full((len(ys), len(xs)), np.nan)
    for row in rows:
        x, y = round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"])
        grid[ys.index(y), xs.index(x)] = float(row["status"] == "validated_pass")
    return xs, ys, grid


def plot_best(rows: list[dict], a_results: Path, output: Path) -> None:
    xs, ys, grid = background(a_results)
    panels = [
        ("neighborhood_raw_fidelity_median", "Median raw CZ fidelity", 100, "viridis", None, "%"),
        ("neighborhood_postselected_fidelity_median", "Median post-selected CZ fidelity", 100, "viridis", (99.9, 100), "%"),
        ("neighborhood_maximum_absolute_bare_frequency_shift_mhz", "Neighborhood maximum |rail-B bare-frequency shift|", 1, "magma_r", None, "MHz"),
        ("anchor_consistent_robustness_margin_mhz", "Worst-anchor-consistent robustness margin", 1, "viridis", None, "MHz"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), layout="constrained")
    extent = [min(xs)-1, max(xs)+1, min(ys)-1, max(ys)+1]
    for ax, (key, title, scale, cmap, limits, unit) in zip(axes.flat, panels):
        ax.imshow(grid, origin="lower", interpolation="nearest", extent=extent,
                  aspect="equal", cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
                  norm=BoundaryNorm([-.5, .5, 1.5], 2))
        values = [row[key] * scale for row in rows]
        norm = Normalize(*(limits or (min(values), max(values))))
        image = ax.scatter([row["ec_a_mhz"][0] for row in rows],
                           [row["ec_a_mhz"][1] for row in rows],
                           c=values, cmap=cmap, norm=norm, s=34,
                           edgecolors="black", linewidths=.35)
        fig.colorbar(image, ax=ax, label=unit, shrink=.82)
        ax.plot([min(xs), max(xs)], [min(xs), max(xs)], "k--", lw=.7, alpha=.5)
        ax.set(title=title, xlabel=r"$E_{C,A1}/h$ (MHz)", ylabel=r"$E_{C,A2}/h$ (MHz)")
        ax.set_aspect("equal")
    fig.suptitle("Worst-S1 fractional-anchor campaign: best rail-B configuration per rail-A point\n"
                 "Rail-A cavity 6.05 GHz; rail-B cavity 5.95 GHz", fontsize=13)
    fig.savefig(output / "best_configurations_on_rail_a_readout_map.png", dpi=220)
    fig.savefig(output / "best_configurations_on_rail_a_readout_map.pdf")
    plt.close(fig)


def plot_status(jobs: list[dict], a_results: Path, output: Path) -> None:
    xs, ys, grid = background(a_results)
    styles = {"complete with selection": ("#16865c", "o"),
              "complete without selection": ("#e69f00", "s"),
              "failed": ("#cc3311", "x")}
    fig, ax = plt.subplots(figsize=(10, 8.5), layout="constrained")
    ax.imshow(grid, origin="lower", interpolation="nearest",
              extent=[min(xs)-1,max(xs)+1,min(ys)-1,max(ys)+1], aspect="equal",
              cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
              norm=BoundaryNorm([-.5,.5,1.5],2))
    handles=[]
    for category,(color,marker) in styles.items():
        subset=[row for row in jobs if row["category"]==category]
        if not subset: continue
        ax.scatter([row["ec_a_mhz"][0] for row in subset],
                   [row["ec_a_mhz"][1] for row in subset], color=color,
                   marker=marker, s=25, linewidths=.8)
        handles.append(Line2D([],[],marker=marker,ls="",color=color,label=f"{category} ({len(subset)})"))
    ax.legend(handles=handles,loc="upper right",fontsize=8)
    ax.set(title="Corrected-cavity worst-anchor campaign status",
           xlabel=r"$E_{C,A1}/h$ (MHz)",ylabel=r"$E_{C,A2}/h$ (MHz)")
    ax.set_aspect("equal")
    fig.savefig(output / "campaign_status_on_rail_a_readout_map.png",dpi=220)
    fig.savefig(output / "campaign_status_on_rail_a_readout_map.pdf")
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    parser.add_argument("--rail-a-results",type=Path,required=True)
    parser.add_argument("--rail-b-results",type=Path,required=True)
    args=parser.parse_args();campaign=args.campaign.resolve()
    output=campaign/"campaign_summary";output.mkdir(parents=True,exist_ok=True)
    idle,domain=readout(args.rail_b_results.resolve())
    status=json.loads((campaign/"campaign_status.json").read_text())
    selected=[];audited=[]
    for job in status["jobs"]:
        record={"selection_id":job["selection_id"],"ec_a_mhz":job["ec_a_mhz"],
                "campaign":job["campaign"],"source_repetition":job["directory"]}
        if job["state"]!="complete":record["category"]="failed";audited.append(record);continue
        try:best,audit=choose(Path(job["directory"]),idle,domain)
        except RuntimeError:record["category"]="complete without selection";audited.append(record);continue
        best.update(record,selection_audit=audit);selected.append(best)
        record["category"]="complete with selection";audited.append(record)
    payload={"cavity_frequencies_ghz":{"rail_A":6.05,"rail_B":5.95},
             "hardware_reconciliation":"excluded",
             "selection_domain":"independent readout/gate overlay",
             "selection_policy":{"raw_fidelity_strictly_greater_than":RAW_MIN,
             "postselected_fidelity_strictly_greater_than":POST_MIN,
             "idle_above_all_interactions_clearance_strictly_greater_than_mhz":CLEARANCE_MIN_MHZ,
             "rail_b_tuning_direction":"idle_to_lower_gate_frequency",
             "neighborhood_radius_mhz":RADIUS_MHZ,
             "minimum_same_anchor_margin_mhz":MIN_MARGIN_MHZ,
             "primary_ranking":"minimum neighborhood maximum absolute rail-B bare-frequency shift",
             "fidelity_summary":"neighborhood median"},
             "counts":dict(Counter(row["category"] for row in audited)),
             "selections":selected,"jobs":audited}
    (output/"best_configurations.json").write_text(json.dumps(payload,indent=2)+"\n")
    fields=["campaign","selection_id","ec_a_mhz","ec_b_mhz","selected_anchor_id",
            "neighborhood_raw_fidelity_median","neighborhood_postselected_fidelity_median",
            "neighborhood_maximum_absolute_bare_frequency_shift_mhz",
            "anchor_consistent_robustness_margin_mhz"]
    with (output/"best_configurations.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in selected:writer.writerow({key:json.dumps(row[key]) if isinstance(row[key],list) else row[key] for key in fields})
    if selected:plot_best(selected,args.rail_a_results.resolve(),output)
    plot_status(audited,args.rail_a_results.resolve(),output)
    print(json.dumps({"output":str(output),"counts":payload["counts"]},indent=2))


if __name__=="__main__":main()
