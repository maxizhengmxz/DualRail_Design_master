#!/usr/bin/env python3
"""Mask and reselect combined-overlay designs by signed idle clearance.

The required clearance is the lower of the two rail-B idle bare frequencies
minus the highest retained valid logical-source collision frequency for the
selected rail-A anchor at the same ordered rail-B EC coordinate.
"""
from __future__ import annotations

import csv
import importlib.util
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

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "select_combined_overlay_configurations.py"
SPEC = importlib.util.spec_from_file_location("combined_overlay_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

SOURCE = BASE.OUTPUT / "combined_overlay_best_configurations.json"
OUTPUT = BASE.OUTPUT / "idle_above_all_interactions_200MHz"
CLEARANCE_MIN_MHZ = 200.0


def gate_records(
    directory: Path, coordinates: set[tuple[int, int]]
) -> dict[tuple[int, int], dict]:
    database = directory / "gate_output" / "compact_anchor_results.sqlite"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    saved = connection.execute(
        "SELECT ec_b1_mhz, ec_b2_mhz, result_json FROM gate_results"
    ).fetchall()
    connection.close()
    return {
        (round(x), round(y)): json.loads(text)
        for x, y, text in saved
        if (round(x), round(y)) in coordinates
    }


def add_clearance(row: dict, idle: dict, gate: dict) -> bool:
    screening = gate.get("screening") or []
    selected_anchor = row["selected_anchor_id"]
    selected = [
        entry
        for entry in screening
        if abs(float(entry.get("fA_GHz", -1)) - row["selected_anchor_readout_frequency_ghz"])
        < 1e-6
    ]
    if not selected and screening:
        selected = screening
    interactions = []
    logical_sources = {}
    for entry in selected:
        for candidate in entry.get("logical_source_candidates", []):
            if candidate.get("status") == "valid" and candidate.get("fB_GHz") is not None:
                frequency = float(candidate["fB_GHz"])
                interactions.append(frequency)
                logical_sources[candidate["logical_source"]] = frequency
    if not interactions:
        return False
    idle_frequencies = [float(idle["bare_f1_ghz"]), float(idle["bare_f2_ghz"])]
    selected_gate_frequency = float(gate["final"]["fB_GHz"])
    highest_interaction = max(interactions)
    row.update(
        idle_bare_frequencies_ghz=idle_frequencies,
        logical_source_interaction_frequencies_ghz=logical_sources,
        highest_logical_source_interaction_frequency_ghz=highest_interaction,
        selected_gate_interaction_frequency_ghz=selected_gate_frequency,
        idle_operating_minus_selected_gate_frequency_mhz=1000.0
        * (float(idle["operating_frequency_ghz"]) - selected_gate_frequency),
        minimum_idle_above_interactions_clearance_mhz=1000.0
        * (min(idle_frequencies) - highest_interaction),
    )
    return True


def enriched_candidates(
    candidates: list[dict], idle_by_ec: dict, gates: dict[tuple[int, int], dict]
) -> list[dict]:
    result = []
    for row in candidates:
        ec = tuple(row["ec_b_mhz"])
        gate = gates.get(ec)
        idle = idle_by_ec.get(ec)
        if gate is not None and idle is not None and add_clearance(row, idle, gate):
            result.append(row)
    return result


def decorate_and_select(
    candidates: list[dict], b_domain: list[tuple[int, int]]
) -> tuple[dict, dict]:
    eligible = [
        row
        for row in candidates
        if row["minimum_idle_above_interactions_clearance_mhz"] > CLEARANCE_MIN_MHZ
    ]
    if not eligible:
        raise RuntimeError("no candidate has the required signed idle clearance")
    xs = np.arange(min(x for x, _ in b_domain), max(x for x, _ in b_domain) + 2, 2)
    ys = np.arange(min(y for _, y in b_domain), max(y for _, y in b_domain) + 2, 2)
    xi = {round(value): index for index, value in enumerate(xs)}
    yi = {round(value): index for index, value in enumerate(ys)}
    by_coordinate = {tuple(row["ec_b_mhz"]): row for row in eligible}
    margins = {}
    for anchor in {row["selected_anchor_id"] for row in eligible}:
        mask = np.zeros((len(ys), len(xs)), dtype=bool)
        for row in eligible:
            if row["selected_anchor_id"] == anchor:
                x, y = row["ec_b_mhz"]
                mask[yi[y], xi[x]] = True
        margins[anchor] = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * 2
    offsets = [
        (dx, dy)
        for dx in range(-4, 5, 2)
        for dy in range(-4, 5, 2)
        if dx * dx + dy * dy <= BASE.NEIGHBORHOOD_RADIUS_MHZ**2 + 1e-9
    ]
    for row in eligible:
        x, y = row["ec_b_mhz"]
        anchor = row["selected_anchor_id"]
        row["anchor_consistent_robustness_margin_mhz"] = float(
            margins[anchor][yi[y], xi[x]]
        )
        neighborhood = [
            other
            for dx, dy in offsets
            if (other := by_coordinate.get((x + dx, y + dy))) is not None
            and other["selected_anchor_id"] == anchor
        ]
        row.update(
            neighborhood_radius_mhz=BASE.NEIGHBORHOOD_RADIUS_MHZ,
            neighborhood_point_count=len(neighborhood),
            neighborhood_raw_fidelity_median=float(
                np.median([other["raw_fidelity"] for other in neighborhood])
            ),
            neighborhood_postselected_fidelity_median=float(
                np.median([other["postselected_fidelity"] for other in neighborhood])
            ),
            neighborhood_maximum_absolute_bare_frequency_shift_mhz=float(
                max(other["maximum_absolute_bare_frequency_shift_mhz"] for other in neighborhood)
            ),
            neighborhood_minimum_idle_above_interactions_clearance_mhz=float(
                min(other["minimum_idle_above_interactions_clearance_mhz"] for other in neighborhood)
            ),
        )
    robust = [
        row
        for row in eligible
        if row["anchor_consistent_robustness_margin_mhz"] >= BASE.MIN_MARGIN_MHZ
    ]
    if not robust:
        raise RuntimeError(
            "no clearance-valid candidate meets the minimum same-anchor margin"
        )
    pool = robust
    selected = min(
        pool,
        key=lambda row: (
            row["neighborhood_maximum_absolute_bare_frequency_shift_mhz"],
            -row["neighborhood_minimum_idle_above_interactions_clearance_mhz"],
            -row["neighborhood_raw_fidelity_median"],
            -row["neighborhood_postselected_fidelity_median"],
            -row["anchor_consistent_robustness_margin_mhz"],
        ),
    )
    return selected, {
        "fidelity_valid_overlay_points_with_retained_interaction_data": len(candidates),
        "points_with_center_clearance_over_200_mhz": len(eligible),
        "points_with_clearance_and_minimum_same_anchor_margin": len(robust),
        "margin_fallback_used": False,
    }


def plot_metric_grid(
    rows: list[dict], stem: str, title: str, *, neighborhood_clearance: bool
) -> None:
    xs, ys, grid = BASE.combined_a_background()
    clearance_key = (
        "neighborhood_minimum_idle_above_interactions_clearance_mhz"
        if neighborhood_clearance
        else "minimum_idle_above_interactions_clearance_mhz"
    )
    clearance_title = (
        "Neighborhood minimum idle-above-interactions clearance"
        if neighborhood_clearance
        else "Center-point idle-above-all-interactions clearance"
    )
    margin_title = (
        "Clearance- and anchor-consistent robustness margin"
        if neighborhood_clearance
        else "Original anchor-consistent overlay robustness margin"
    )
    panels = [
        ("neighborhood_raw_fidelity_median", "Neighborhood median raw CZ fidelity", 100, "viridis", "%"),
        ("neighborhood_postselected_fidelity_median", "Neighborhood median post-selected CZ fidelity", 100, "viridis", "%"),
        (clearance_key, clearance_title, 1, "plasma", "MHz"),
        (
            "idle_operating_minus_selected_gate_frequency_mhz",
            "Rail-B idle operating frequency minus selected gate frequency",
            1,
            "coolwarm",
            "MHz",
        ),
        ("anchor_consistent_robustness_margin_mhz", margin_title, 1, "viridis", "MHz"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), layout="constrained")
    extent = [min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1]
    for ax, (key, subtitle, scale, cmap, unit) in zip(axes.flat, panels):
        ax.imshow(grid, origin="lower", interpolation="nearest", extent=extent, aspect="equal",
                  cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
                  norm=BoundaryNorm([-.5, .5, 1.5], 2))
        values = [row[key] * scale for row in rows]
        norm = Normalize(min(values), max(values))
        image = None
        for campaign, marker in (("original_6p00_6p00", "^"), ("split_6p05_5p95", "o")):
            subset = [row for row in rows if row["campaign"] == campaign]
            if subset:
                image = ax.scatter([r["ec_a_mhz"][0] for r in subset], [r["ec_a_mhz"][1] for r in subset],
                                   c=[r[key] * scale for r in subset], cmap=cmap, norm=norm,
                                   marker=marker, s=34, edgecolors="black", linewidths=.35, zorder=4)
        if image is not None:
            fig.colorbar(image, ax=ax, label=unit, shrink=.82)
        ax.plot([min(xs), max(xs)], [min(xs), max(xs)], "k--", lw=.7, alpha=.5)
        ax.set(title=subtitle, xlabel=r"$E_{C,A1}/h$ (MHz)", ylabel=r"$E_{C,A2}/h$ (MHz)")
        ax.set_aspect("equal")
    axes.flat[-1].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220)
    fig.savefig(OUTPUT / f"{stem}.pdf")
    plt.close(fig)


def plot_existing_mask(rows: list[dict]) -> None:
    xs, ys, grid = BASE.combined_a_background()
    fig, ax = plt.subplots(figsize=(10, 8.5), layout="constrained")
    ax.imshow(grid, origin="lower", interpolation="nearest",
              extent=[min(xs)-1, max(xs)+1, min(ys)-1, max(ys)+1], aspect="equal",
              cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
              norm=BoundaryNorm([-.5, .5, 1.5], 2))
    for passed, color, marker, label in (
        (True, "#16865c", "o", "Existing selection passes signed 200 MHz clearance"),
        (False, "#cc3311", "x", "Existing selection fails signed 200 MHz clearance"),
    ):
        subset = [r for r in rows if r["passes_idle_above_all_interactions_200mhz"] is passed]
        ax.scatter([r["ec_a_mhz"][0] for r in subset], [r["ec_a_mhz"][1] for r in subset],
                   c=color, marker=marker, s=26, linewidths=.8, label=f"{label} ({len(subset)})", zorder=4)
    ax.plot([min(xs), max(xs)], [min(xs), max(xs)], "k--", lw=.7, alpha=.5)
    ax.legend(fontsize=8)
    ax.set(title="Mask applied to the existing combined-overlay selections",
           xlabel=r"$E_{C,A1}/h$ (MHz)", ylabel=r"$E_{C,A2}/h$ (MHz)")
    ax.set_aspect("equal")
    fig.savefig(OUTPUT / "existing_selection_signed_clearance_mask.png", dpi=220)
    fig.savefig(OUTPUT / "existing_selection_signed_clearance_mask.pdf")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configurations = [
        ("original_6p00_6p00", BASE.ORIGINAL, BASE.ORIGINAL_B),
        ("split_6p05_5p95", BASE.SPLIT, BASE.SPLIT_B),
    ]
    old = json.loads(SOURCE.read_text())["selections"]
    old_by_repetition = {Path(row["source_repetition"]): row for row in old}
    masked = []
    reselected = []
    audits = []
    for campaign_name, campaign, b_results in configurations:
        idle_by_ec, b_domain = BASE.readout_lookup(b_results)
        status = json.loads((campaign / "campaign_status.json").read_text())
        for job in status["jobs"]:
            if job["state"] != "complete":
                continue
            directory = Path(job["directory"])
            old_row = old_by_repetition.get(directory)
            candidates = BASE.overlay_rows(directory, idle_by_ec)
            needed = {tuple(row["ec_b_mhz"]) for row in candidates}
            if old_row is not None:
                needed.add(tuple(old_row["ec_b_mhz"]))
            gates = gate_records(directory, needed)
            if old_row is not None:
                row = dict(old_row)
                ec = tuple(row["ec_b_mhz"])
                if ec in gates and ec in idle_by_ec and add_clearance(row, idle_by_ec[ec], gates[ec]):
                    row["passes_idle_above_all_interactions_200mhz"] = (
                        row["minimum_idle_above_interactions_clearance_mhz"] > CLEARANCE_MIN_MHZ
                    )
                    masked.append(row)
            try:
                candidates = enriched_candidates(candidates, idle_by_ec, gates)
                selected, audit = decorate_and_select(candidates, b_domain)
            except RuntimeError:
                audits.append({"campaign": campaign_name, "selection_id": job["selection_id"], "status": "no_clearance_valid_selection"})
                continue
            selected.update(campaign=campaign_name, selection_id=job["selection_id"],
                            ec_a_mhz=[int(v) for v in job["ec_a_mhz"]],
                            source_repetition=str(directory), selection_audit=audit)
            reselected.append(selected)
            audits.append({"campaign": campaign_name, "selection_id": job["selection_id"], "status": "reselected"})

    masked_pass = [r for r in masked if r["passes_idle_above_all_interactions_200mhz"]]
    (OUTPUT / "existing_selections_masked.json").write_text(json.dumps(masked, indent=2) + "\n")
    (OUTPUT / "reselected_configurations.json").write_text(json.dumps({
        "clearance_definition": "min(B1_idle_bare,B2_idle_bare)-max(valid retained logical-source collision frequencies for selected anchor)",
        "strict_minimum_clearance_mhz": CLEARANCE_MIN_MHZ,
        "ranking_after_clearance_mask": "minimum neighborhood maximum absolute rail-B bare-frequency shift, then larger neighborhood clearance/fidelity/margin",
        "counts": dict(Counter(r["status"] for r in audits)),
        "selections": reselected,
        "audit": audits,
    }, indent=2) + "\n")
    plot_existing_mask(masked)
    if masked_pass:
        plot_metric_grid(masked_pass, "existing_selections_after_signed_clearance_mask",
                         "Existing selections that remain after signed idle-above-all-interactions > 200 MHz mask",
                         neighborhood_clearance=False)
    if reselected:
        plot_metric_grid(reselected, "reselected_with_signed_clearance",
                         "Rail-B configurations reselected with signed idle-above-all-interactions > 200 MHz",
                         neighborhood_clearance=True)
    print(json.dumps({"output": str(OUTPUT), "existing_total": len(masked),
                      "existing_pass": len(masked_pass), "reselected": len(reselected),
                      "reselection_counts": dict(Counter(r["status"] for r in audits))}, indent=2))


if __name__ == "__main__":
    main()
