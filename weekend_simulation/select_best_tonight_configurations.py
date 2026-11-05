#!/usr/bin/env python3
"""Select and plot one robust rail-B design for each tonight rail-A repetition."""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2Q-gate_XM"))
from xm_cz.model import local_model_from_ej

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "sampled_runs" / "tonight_first12_B4_5p5"
A_RESULTS = HERE / "readout" / "direct_ec" / "rail_a_step5" / "results.json"
B_RESULTS = HERE / "readout" / "direct_ec" / "rail_b_step2" / "results.json"
RAW_MIN = 0.95
POST_MIN = 0.999
MIN_MARGIN_MHZ = 5.0
NEIGHBORHOOD_RADIUS_MHZ = 5.0
GRID_STEP_MHZ = 2.0


def metrics(item: dict) -> dict:
    gate = item["gate"]
    ec = [float(v) for v in item["ec_b_mhz"]]
    idle = item["rail_b"]["point"]
    gate_ej = gate["EJ_GHz"][2:]
    gate_bare = [
        float(local_model_from_ej(ec[i] / 1000.0, gate_ej[i])[1][1])
        for i in (0, 1)
    ]
    shifts = [
        1000.0 * (gate_bare[0] - idle["bare_f1_ghz"]),
        1000.0 * (gate_bare[1] - idle["bare_f2_ghz"]),
    ]
    return {
        "ec_b_mhz": [int(round(v)) for v in ec],
        "raw_fidelity": float(gate["raw_fidelity"]),
        "postselected_fidelity": float(gate["postselected_fidelity"]),
        "bare_frequency_shift_b1_mhz": shifts[0],
        "bare_frequency_shift_b2_mhz": shifts[1],
        "maximum_absolute_bare_frequency_shift_mhz": max(map(abs, shifts)),
        "selected_anchor_id": item["selected_anchor"]["anchor_id"],
        "selected_anchor_readout_frequency_ghz": item["selected_anchor"]["readout_operating_frequency_GHz"],
    }


def select_one(directory: Path) -> tuple[dict, dict]:
    db = directory / "fixed_hardware_reconciliation" / "fixed_hardware_results.sqlite"
    con = sqlite3.connect(db)
    candidates = []
    for passed, text in con.execute(
        "SELECT reconciled_pass, details_json FROM reconciliation"
    ):
        if not passed:
            continue
        item = json.loads(text)
        item = item.get("selected", item)
        row = metrics(item)
        if row["raw_fidelity"] > RAW_MIN and row["postselected_fidelity"] > POST_MIN:
            candidates.append(row)
    con.close()

    all_b = [
        (int(round(r["ec1_target_mhz"])), int(round(r["ec2_target_mhz"])))
        for r in json.loads(B_RESULTS.read_text())
    ]
    xs = np.arange(min(x for x, _ in all_b), max(x for x, _ in all_b) + GRID_STEP_MHZ, GRID_STEP_MHZ)
    ys = np.arange(min(y for _, y in all_b), max(y for _, y in all_b) + GRID_STEP_MHZ, GRID_STEP_MHZ)
    xi = {int(round(x)): i for i, x in enumerate(xs)}
    yi = {int(round(y)): i for i, y in enumerate(ys)}
    by_coordinate = {tuple(row["ec_b_mhz"]): row for row in candidates}
    anchors = sorted({row["selected_anchor_id"] for row in candidates})
    margins = {}
    for anchor in anchors:
        mask = np.zeros((len(ys), len(xs)), dtype=bool)
        for row in candidates:
            if row["selected_anchor_id"] == anchor:
                x, y = row["ec_b_mhz"]
                mask[yi[y], xi[x]] = True
        margins[anchor] = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * GRID_STEP_MHZ
    for row in candidates:
        x, y = row["ec_b_mhz"]
        row["anchor_consistent_robustness_margin_mhz"] = float(
            margins[row["selected_anchor_id"]][yi[y], xi[x]]
        )
        # Compatibility alias for downstream reports; semantics are now the
        # same-anchor combined margin, not the former feasibility-only margin.
        row["robustness_margin_mhz"] = row["anchor_consistent_robustness_margin_mhz"]
        neighborhood = [
            other for (u, v), other in by_coordinate.items()
            if (u - x) ** 2 + (v - y) ** 2 <= NEIGHBORHOOD_RADIUS_MHZ ** 2 + 1e-9
            and other["selected_anchor_id"] == row["selected_anchor_id"]
        ]
        raw = np.asarray([other["raw_fidelity"] for other in neighborhood])
        post = np.asarray([other["postselected_fidelity"] for other in neighborhood])
        shifts = np.asarray([other["maximum_absolute_bare_frequency_shift_mhz"] for other in neighborhood])
        row.update({
            "neighborhood_radius_mhz": NEIGHBORHOOD_RADIUS_MHZ,
            "neighborhood_point_count": len(neighborhood),
            "neighborhood_raw_fidelity_mean": float(np.mean(raw)),
            "neighborhood_raw_fidelity_median": float(np.median(raw)),
            "neighborhood_raw_fidelity_minimum": float(np.min(raw)),
            "neighborhood_raw_fidelity_std": float(np.std(raw)),
            "neighborhood_postselected_fidelity_mean": float(np.mean(post)),
            "neighborhood_postselected_fidelity_median": float(np.median(post)),
            "neighborhood_postselected_fidelity_minimum": float(np.min(post)),
            "neighborhood_postselected_fidelity_std": float(np.std(post)),
            "neighborhood_maximum_absolute_bare_frequency_shift_mhz": float(np.max(shifts)),
            "neighborhood_mean_absolute_bare_frequency_shift_mhz": float(np.mean(shifts)),
        })

    robust = [r for r in candidates if r["anchor_consistent_robustness_margin_mhz"] >= MIN_MARGIN_MHZ]
    pool = robust or candidates
    if not pool:
        raise RuntimeError(f"No fidelity-passing reconciled points in {directory}")
    selected = min(
        pool,
        key=lambda r: (
            r["neighborhood_maximum_absolute_bare_frequency_shift_mhz"],
            -r["neighborhood_raw_fidelity_median"],
            -r["neighborhood_postselected_fidelity_median"],
            -r["anchor_consistent_robustness_margin_mhz"],
        ),
    )
    audit = {
        "fidelity_passing_points": len(candidates),
        "points_meeting_anchor_consistent_minimum_margin": len(robust),
        "minimum_anchor_consistent_margin_mhz": MIN_MARGIN_MHZ,
        "neighborhood_radius_mhz": NEIGHBORHOOD_RADIUS_MHZ,
        "margin_fallback_used": not bool(robust),
    }
    return selected, audit


def plot_readout_map(rows: list[dict], output: Path) -> None:
    source = json.loads(A_RESULTS.read_text())
    xs = sorted({int(r["ec1_target_mhz"]) for r in source})
    ys = sorted({int(r["ec2_target_mhz"]) for r in source})
    grid = np.full((len(ys), len(xs)), np.nan)
    for r in source:
        x, y = int(r["ec1_target_mhz"]), int(r["ec2_target_mhz"])
        grid[ys.index(y), xs.index(x)] = float(r["status"] == "validated_pass")
    background = ListedColormap(["#e2e6e9", "#d8efe8"])
    background_norm = BoundaryNorm([-.5, .5, 1.5], 2)
    panels = [
        ("neighborhood_raw_fidelity_median", "Median raw CZ fidelity in 5 MHz neighborhood", 100.0, "viridis", None, "%"),
        ("neighborhood_postselected_fidelity_median", "Median post-selected CZ fidelity in 5 MHz neighborhood", 100.0, "viridis", (99.9, 100.0), "%"),
        ("neighborhood_maximum_absolute_bare_frequency_shift_mhz", "Neighborhood maximum |rail-B bare-frequency shift|", 1.0, "magma_r", None, "MHz"),
        ("anchor_consistent_robustness_margin_mhz", "Anchor-consistent robustness margin", 1.0, "viridis", None, "MHz"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), layout="constrained", sharex=True, sharey=True)
    extent = [min(xs)-2.5, max(xs)+2.5, min(ys)-2.5, max(ys)+2.5]
    for ax, (key, title, scale, cmap, limits, unit) in zip(axes.flat, panels):
        ax.imshow(grid, origin="lower", interpolation="nearest", extent=extent, aspect="equal", cmap=background, norm=background_norm)
        values = [r[key] * scale for r in rows]
        norm = Normalize(*(limits or (min(values), max(values))))
        image = None
        for fallback, marker in ((False, "o"), (True, "X")):
            subset = [r for r in rows if r["selection_audit"]["margin_fallback_used"] == fallback]
            if not subset:
                continue
            image = ax.scatter(
                [r["ec_a_mhz"][0] for r in subset], [r["ec_a_mhz"][1] for r in subset],
                c=[r[key] * scale for r in subset], cmap=cmap, norm=norm, s=115, marker=marker,
                edgecolors="black", linewidths=.8, zorder=4,
            )
        for r in rows:
            x, y = r["ec_a_mhz"]
            ax.annotate(r["selection_id"], (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7.5, fontweight="bold")
        fig.colorbar(image, ax=ax, label=unit, shrink=.82)
        ax.legend(handles=[
            Line2D([], [], marker="o", ls="", color="black", markerfacecolor="white", label="Same-anchor margin ≥5 MHz"),
            Line2D([], [], marker="X", ls="", color="black", markerfacecolor="white", label="Best available; margin <5 MHz"),
        ], loc="upper right", fontsize=7)
        ax.set_title(title)
        ax.set_xlabel(r"$E_{C,A1}/h$ (MHz)")
        ax.set_ylabel(r"$E_{C,A2}/h$ (MHz)")
        ax.set_aspect("equal")
    fig.suptitle("Best robust rail-B configuration for each sampled rail-A point\n"
                 "5 MHz same-anchor neighborhoods; ranked by neighborhood maximum |ΔfB,bare|", fontsize=13)
    fig.savefig(output / "best_configurations_on_rail_a_readout_map.png", dpi=220)
    fig.savefig(output / "best_configurations_on_rail_a_readout_map.pdf")
    plt.close(fig)


def plot_table(rows: list[dict], output: Path) -> None:
    columns = ["ID", "EC-A (MHz)", "Selected EC-B (MHz)", "Median raw CZ", "Median post CZ", "neigh. max |ΔfB|", "Same-anchor margin", "A anchor"]
    cells = []
    for r in rows:
        cells.append([
            r["selection_id"], str(tuple(r["ec_a_mhz"])), str(tuple(r["ec_b_mhz"])),
            f"{100*r['neighborhood_raw_fidelity_median']:.4f}%", f"{100*r['neighborhood_postselected_fidelity_median']:.5f}%",
            f"{r['neighborhood_maximum_absolute_bare_frequency_shift_mhz']:.2f}", f"{r['anchor_consistent_robustness_margin_mhz']:.2f}" + ("*" if r["selection_audit"]["margin_fallback_used"] else ""),
            r["selected_anchor_id"].replace("fA_", "") + " GHz",
        ])
    fig, ax = plt.subplots(figsize=(15, 6.2), layout="constrained")
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#315b7d"); cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf3f6")
    ax.set_title("Selected rail-B configuration for each sampled rail-A EC pair\n"
                 "5 MHz neighborhood must be feasible and use one rail-A anchor; * indicates no point met this target", pad=18)
    fig.savefig(output / "best_configurations_table.png", dpi=220)
    fig.savefig(output / "best_configurations_table.pdf")
    plt.close(fig)


def main() -> None:
    status = json.loads((CAMPAIGN / "campaign_status.json").read_text())
    rows = []
    for job in status["jobs"]:
        directory = Path(job["directory"])
        selected, audit = select_one(directory)
        selected.update({
            "selection_id": job["id"],
            "ec_a_mhz": [int(v) for v in job["ec"]],
            "source_repetition": str(directory),
            "selection_audit": audit,
        })
        rows.append(selected)
    output = CAMPAIGN / "best_configuration_selection"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_policy": {
            "raw_fidelity_strictly_greater_than": RAW_MIN,
            "postselected_fidelity_strictly_greater_than": POST_MIN,
            "neighborhood_radius_mhz": NEIGHBORHOOD_RADIUS_MHZ,
            "minimum_anchor_consistent_robustness_margin_mhz": MIN_MARGIN_MHZ,
            "primary_ranking": "minimum neighborhood maximum absolute rail-B bare-frequency shift",
            "tie_breakers": ["higher neighborhood median raw fidelity", "higher neighborhood median post-selected fidelity", "larger anchor-consistent margin"],
            "margin_definition": "Euclidean distance on the 2 MHz EC-B lattice to the nearest point outside the fidelity-passing reconciled region or selecting a different rail-A anchor, including the sampled-domain exterior",
        },
        "selections": rows,
    }
    (output / "best_configurations.json").write_text(json.dumps(payload, indent=2) + "\n")
    columns = [
        "selection_id", "ec_a_mhz", "ec_b_mhz", "raw_fidelity", "postselected_fidelity",
        "maximum_absolute_bare_frequency_shift_mhz", "anchor_consistent_robustness_margin_mhz",
        "bare_frequency_shift_b1_mhz", "bare_frequency_shift_b2_mhz",
        "selected_anchor_id", "selected_anchor_readout_frequency_ghz",
        "neighborhood_radius_mhz", "neighborhood_point_count",
        "neighborhood_raw_fidelity_mean", "neighborhood_raw_fidelity_median", "neighborhood_raw_fidelity_minimum", "neighborhood_raw_fidelity_std",
        "neighborhood_postselected_fidelity_mean", "neighborhood_postselected_fidelity_median", "neighborhood_postselected_fidelity_minimum", "neighborhood_postselected_fidelity_std",
        "neighborhood_maximum_absolute_bare_frequency_shift_mhz", "neighborhood_mean_absolute_bare_frequency_shift_mhz",
    ]
    with (output / "best_configurations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k]) if isinstance(row[k], list) else row[k] for k in columns})
    plot_readout_map(rows, output)
    plot_table(rows, output)
    print(json.dumps({"output": str(output), "selections": rows}, indent=2))


if __name__ == "__main__":
    main()
