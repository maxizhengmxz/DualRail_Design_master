#!/usr/bin/env python3
"""Select from the full independent readout/gate overlay across two campaigns.

The historical reconciliation flag is deliberately ignored.  Gate metrics are
the saved fixed-80-MHz results; idle rail-B hardware comes directly from the
corresponding independent readout map.  Selections are therefore candidates
for a subsequent fixed-readout-J gate recalculation, not certifications of it.
"""
from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "sampled_runs" / "rail_a_gt_partition_2026-11-06"
SPLIT = (
    HERE
    / "cavity_split_campaign_2026-11-09"
    / "sampled_runs"
    / "rail_a_lt_partition"
)
ORIGINAL_A = HERE / "readout" / "direct_ec" / "rail_a_step2" / "results.json"
ORIGINAL_B = HERE / "readout" / "direct_ec" / "rail_b_step2" / "results.json"
SPLIT_A = (
    HERE
    / "cavity_split_campaign_2026-11-09"
    / "readout"
    / "direct_ec"
    / "rail_a_step2_fr6p05"
    / "results.json"
)
SPLIT_B = (
    HERE
    / "cavity_split_campaign_2026-11-09"
    / "readout"
    / "direct_ec"
    / "rail_b_step2_fr5p95"
    / "results.json"
)
OUTPUT = HERE / "combined_overlay_selection_2026-11-12"

RAW_MIN = 0.95
POST_MIN = 0.999
GRID_STEP_MHZ = 2
NEIGHBORHOOD_RADIUS_MHZ = 5.0
MIN_MARGIN_MHZ = 5.0


def readout_lookup(path: Path) -> tuple[dict[tuple[int, int], dict], list[tuple[int, int]]]:
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


def overlay_rows(directory: Path, idle_by_ec: dict[tuple[int, int], dict]) -> list[dict]:
    database = directory / "fixed_hardware_reconciliation" / "fixed_hardware_results.sqlite"
    if not database.exists():
        return []
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    query = """
        SELECT ec_b1_mhz, ec_b2_mhz,
               json_extract(details_json, '$.gate.raw_fidelity'),
               json_extract(details_json, '$.gate.postselected_fidelity'),
               json_extract(details_json, '$.gate.EJ_GHz'),
               json_extract(details_json, '$.gate.fB_GHz'),
               json_extract(details_json, '$.selected_anchor.anchor_id'),
               json_extract(details_json, '$.selected_anchor.readout_operating_frequency_GHz'),
               reconciled_pass
        FROM reconciliation
    """
    saved = connection.execute(query).fetchall()
    connection.close()
    rows = []
    for ec1, ec2, raw, post, ej_text, gate_frequency, anchor, anchor_frequency, old_pass in saved:
        ec = (round(ec1), round(ec2))
        idle = idle_by_ec.get(ec)
        if (
            idle is None
            or raw is None
            or post is None
            or ej_text is None
            or gate_frequency is None
            or anchor is None
        ):
            continue
        if not raw > RAW_MIN or not post > POST_MIN:
            continue
        # The saved toy gate constructs both B transmons at this common bare
        # frequency before intra-rail dressing, so no spectrum solve is needed.
        gate_bare = [float(gate_frequency), float(gate_frequency)]
        shifts = [
            1000.0 * (gate_bare[0] - idle["bare_f1_ghz"]),
            1000.0 * (gate_bare[1] - idle["bare_f2_ghz"]),
        ]
        rows.append(
            {
                "ec_b_mhz": list(ec),
                "raw_fidelity": float(raw),
                "postselected_fidelity": float(post),
                "bare_frequency_shift_b1_mhz": shifts[0],
                "bare_frequency_shift_b2_mhz": shifts[1],
                "maximum_absolute_bare_frequency_shift_mhz": max(map(abs, shifts)),
                "selected_anchor_id": anchor,
                "selected_anchor_readout_frequency_ghz": float(anchor_frequency),
                "old_reconciliation_pass": bool(old_pass),
                "idle_J_B_GHz": float(idle["J12_ghz"]),
                "idle_logical_splitting_GHz": float(idle["splitting_ghz"]),
                "idle_operating_frequency_GHz": float(idle["operating_frequency_ghz"]),
            }
        )
    return rows


def select_one(
    directory: Path,
    idle_by_ec: dict[tuple[int, int], dict],
    b_domain: list[tuple[int, int]],
) -> tuple[dict, dict]:
    candidates = overlay_rows(directory, idle_by_ec)
    if not candidates:
        raise RuntimeError("empty independent overlay")
    xs = np.arange(
        min(x for x, _ in b_domain),
        max(x for x, _ in b_domain) + GRID_STEP_MHZ,
        GRID_STEP_MHZ,
    )
    ys = np.arange(
        min(y for _, y in b_domain),
        max(y for _, y in b_domain) + GRID_STEP_MHZ,
        GRID_STEP_MHZ,
    )
    xi = {round(value): index for index, value in enumerate(xs)}
    yi = {round(value): index for index, value in enumerate(ys)}
    by_coordinate = {tuple(row["ec_b_mhz"]): row for row in candidates}
    margins = {}
    for anchor in {row["selected_anchor_id"] for row in candidates}:
        mask = np.zeros((len(ys), len(xs)), dtype=bool)
        for row in candidates:
            if row["selected_anchor_id"] == anchor:
                x, y = row["ec_b_mhz"]
                mask[yi[y], xi[x]] = True
        margins[anchor] = (
            distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * GRID_STEP_MHZ
        )
    offsets = [
        (dx, dy)
        for dx in range(-4, 5, GRID_STEP_MHZ)
        for dy in range(-4, 5, GRID_STEP_MHZ)
        if dx * dx + dy * dy <= NEIGHBORHOOD_RADIUS_MHZ**2 + 1e-9
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
            if (other := by_coordinate.get((x + dx, y + dy))) is not None
            and other["selected_anchor_id"] == anchor
        ]
        raw = np.asarray([other["raw_fidelity"] for other in neighborhood])
        post = np.asarray([other["postselected_fidelity"] for other in neighborhood])
        shifts = np.asarray(
            [other["maximum_absolute_bare_frequency_shift_mhz"] for other in neighborhood]
        )
        row.update(
            neighborhood_radius_mhz=NEIGHBORHOOD_RADIUS_MHZ,
            neighborhood_point_count=len(neighborhood),
            neighborhood_raw_fidelity_median=float(np.median(raw)),
            neighborhood_postselected_fidelity_median=float(np.median(post)),
            neighborhood_maximum_absolute_bare_frequency_shift_mhz=float(np.max(shifts)),
        )
    robust = [
        row
        for row in candidates
        if row["anchor_consistent_robustness_margin_mhz"] >= MIN_MARGIN_MHZ
    ]
    pool = robust or candidates
    selected = min(
        pool,
        key=lambda row: (
            row["neighborhood_maximum_absolute_bare_frequency_shift_mhz"],
            -row["neighborhood_raw_fidelity_median"],
            -row["neighborhood_postselected_fidelity_median"],
            -row["anchor_consistent_robustness_margin_mhz"],
        ),
    )
    return selected, {
        "overlay_points": len(candidates),
        "old_reconciliation_passing_points": sum(
            row["old_reconciliation_pass"] for row in candidates
        ),
        "points_meeting_anchor_consistent_minimum_margin": len(robust),
        "margin_fallback_used": not bool(robust),
    }


def combined_a_background() -> tuple[list[int], list[int], np.ndarray]:
    original = json.loads(ORIGINAL_A.read_text())
    split = json.loads(SPLIT_A.read_text())
    chosen = {}
    for row in original:
        ec = (round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"]))
        if ec[0] > ec[1]:
            chosen[ec] = row["status"] == "validated_pass"
    for row in split:
        ec = (round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"]))
        if ec[0] < ec[1]:
            chosen[ec] = row["status"] == "validated_pass"
    xs = sorted({x for x, _ in chosen})
    ys = sorted({y for _, y in chosen})
    grid = np.full((len(ys), len(xs)), np.nan)
    for (x, y), passed in chosen.items():
        grid[ys.index(y), xs.index(x)] = float(passed)
    return xs, ys, grid


def plot_selections(rows: list[dict]) -> None:
    xs, ys, grid = combined_a_background()
    panels = [
        (
            "neighborhood_raw_fidelity_median",
            "Median raw CZ fidelity in 5 MHz overlay neighborhood",
            100.0,
            "viridis",
            None,
            "%",
        ),
        (
            "neighborhood_postselected_fidelity_median",
            "Median post-selected CZ fidelity in 5 MHz overlay neighborhood",
            100.0,
            "viridis",
            (99.9, 100.0),
            "%",
        ),
        (
            "neighborhood_maximum_absolute_bare_frequency_shift_mhz",
            "Neighborhood maximum |rail-B bare-frequency shift|",
            1.0,
            "magma_r",
            None,
            "MHz",
        ),
        (
            "anchor_consistent_robustness_margin_mhz",
            "Anchor-consistent overlay robustness margin",
            1.0,
            "viridis",
            None,
            "MHz",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), layout="constrained")
    background = ListedColormap(["#e2e6e9", "#d8efe8"])
    norm_background = BoundaryNorm([-.5, .5, 1.5], 2)
    extent = [min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1]
    for ax, (key, title, scale, cmap, limits, unit) in zip(axes.flat, panels):
        ax.imshow(
            grid,
            origin="lower",
            interpolation="nearest",
            extent=extent,
            aspect="equal",
            cmap=background,
            norm=norm_background,
        )
        values = [row[key] * scale for row in rows]
        color_norm = Normalize(*(limits or (min(values), max(values))))
        image = None
        for campaign, marker in (("original_6p00_6p00", "^"), ("split_6p05_5p95", "o")):
            subset = [row for row in rows if row["campaign"] == campaign]
            image = ax.scatter(
                [row["ec_a_mhz"][0] for row in subset],
                [row["ec_a_mhz"][1] for row in subset],
                c=[row[key] * scale for row in subset],
                cmap=cmap,
                norm=color_norm,
                marker=marker,
                s=34,
                edgecolors="black",
                linewidths=0.35,
                zorder=4,
            )
        fig.colorbar(image, ax=ax, label=unit, shrink=0.82)
        ax.plot([min(xs), max(xs)], [min(xs), max(xs)], "k--", lw=0.7, alpha=0.5)
        ax.set(title=title, xlabel=r"$E_{C,A1}/h$ (MHz)", ylabel=r"$E_{C,A2}/h$ (MHz)")
        ax.set_aspect("equal")
    axes[0, 0].legend(
        handles=[
            Line2D([], [], marker="^", ls="", color="black", label="Original cavities: A1 > A2"),
            Line2D([], [], marker="o", ls="", color="black", label="Split cavities: A1 < A2"),
        ],
        fontsize=8,
    )
    fig.suptitle(
        "Best rail-B candidate selected from the full independent readout/gate overlay\n"
        "Historical reconciliation is not used; fixed-readout-J recalibration remains required",
        fontsize=13,
    )
    fig.savefig(OUTPUT / "best_configurations_on_combined_rail_a_readout_map.png", dpi=220)
    fig.savefig(OUTPUT / "best_configurations_on_combined_rail_a_readout_map.pdf")
    plt.close(fig)


def plot_status(jobs: list[dict]) -> None:
    xs, ys, grid = combined_a_background()
    styles = {
        "complete with overlay selection": ("#16865c", "o"),
        "complete without overlay selection": ("#e69f00", "s"),
        "campaign failure": ("#cc3311", "x"),
    }
    fig, ax = plt.subplots(figsize=(10, 8.5), layout="constrained")
    ax.imshow(
        grid,
        origin="lower",
        interpolation="nearest",
        extent=[min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1],
        aspect="equal",
        cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
        norm=BoundaryNorm([-.5, .5, 1.5], 2),
    )
    handles = []
    for category, (color, marker) in styles.items():
        subset = [row for row in jobs if row["category"] == category]
        if not subset:
            continue
        ax.scatter(
            [row["ec_a_mhz"][0] for row in subset],
            [row["ec_a_mhz"][1] for row in subset],
            color=color,
            marker=marker,
            s=25,
            linewidths=0.8,
            zorder=4,
        )
        handles.append(
            Line2D([], [], marker=marker, ls="", color=color, label=f"{category} ({len(subset)})")
        )
    ax.plot([min(xs), max(xs)], [min(xs), max(xs)], "k--", lw=0.7, alpha=0.5)
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set(
        title="Combined campaign status on rail-A readout maps",
        xlabel=r"$E_{C,A1}/h$ (MHz)",
        ylabel=r"$E_{C,A2}/h$ (MHz)",
    )
    ax.set_aspect("equal")
    fig.savefig(OUTPUT / "combined_campaign_status_on_rail_a_readout_map.png", dpi=220)
    fig.savefig(OUTPUT / "combined_campaign_status_on_rail_a_readout_map.pdf")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configurations = [
        ("original_6p00_6p00", ORIGINAL, ORIGINAL_B),
        ("split_6p05_5p95", SPLIT, SPLIT_B),
    ]
    selections = []
    audited_jobs = []
    for campaign_name, campaign, b_results in configurations:
        idle_by_ec, b_domain = readout_lookup(b_results)
        status = json.loads((campaign / "campaign_status.json").read_text())
        for job in status["jobs"]:
            directory = Path(job["directory"])
            record = {
                "selection_id": job["selection_id"],
                "ec_a_mhz": [int(value) for value in job["ec_a_mhz"]],
                "campaign": campaign_name,
                "source_repetition": str(directory),
            }
            if job["state"] != "complete":
                record["category"] = "campaign failure"
                audited_jobs.append(record)
                continue
            try:
                selected, audit = select_one(directory, idle_by_ec, b_domain)
            except RuntimeError:
                record["category"] = "complete without overlay selection"
                audited_jobs.append(record)
                continue
            selected.update(record, selection_audit=audit)
            selections.append(selected)
            record["category"] = "complete with overlay selection"
            audited_jobs.append(record)
    payload = {
        "scope": "full independent readout/gate overlay; old reconciliation flag ignored",
        "fixed_readout_J_status": "provisional candidates requiring fresh gate recalibration",
        "selection_policy": {
            "raw_fidelity_strictly_greater_than": RAW_MIN,
            "postselected_fidelity_strictly_greater_than": POST_MIN,
            "neighborhood_radius_mhz": NEIGHBORHOOD_RADIUS_MHZ,
            "minimum_same_anchor_margin_mhz": MIN_MARGIN_MHZ,
            "primary_ranking": "minimum neighborhood maximum absolute rail-B bare-frequency shift",
            "fidelity_summary": "neighborhood median for raw and post-selected fidelity",
        },
        "counts": dict(Counter(row["category"] for row in audited_jobs)),
        "selections": selections,
        "jobs": audited_jobs,
    }
    (OUTPUT / "combined_overlay_best_configurations.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    fields = [
        "campaign",
        "selection_id",
        "ec_a_mhz",
        "ec_b_mhz",
        "selected_anchor_id",
        "old_reconciliation_pass",
        "idle_J_B_GHz",
        "idle_logical_splitting_GHz",
        "neighborhood_raw_fidelity_median",
        "neighborhood_postselected_fidelity_median",
        "neighborhood_maximum_absolute_bare_frequency_shift_mhz",
        "anchor_consistent_robustness_margin_mhz",
    ]
    with (OUTPUT / "combined_overlay_best_configurations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selections:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row[key], list) else row[key]
                    for key in fields
                }
            )
    plot_selections(selections)
    plot_status(audited_jobs)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "selections": len(selections),
                "counts": payload["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
