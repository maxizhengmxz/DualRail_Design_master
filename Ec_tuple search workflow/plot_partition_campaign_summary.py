#!/usr/bin/env python3
"""Select robust designs and plot one full rail-A partition campaign."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import select_best_tonight_configurations as selection


def job_identity(job: dict) -> tuple[str, list[int]]:
    return (
        job.get("selection_id", job.get("id")),
        [int(v) for v in job.get("ec_a_mhz", job.get("ec"))],
    )


def failure_class(directory: Path) -> str:
    log = directory / "repetition.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    post_log = directory / "post.log"
    if post_log.exists():
        text += "\n" + post_log.read_text(errors="replace")
    if "No connected readout-frequency component" in text:
        return "no connected readout component"
    if "fewer than 5 distinct validated samples" in text:
        return "insufficient anchor samples"
    if "plot_anchor_reconciled_metrics.py" in text:
        return "empty gate-overlay region"
    return "execution failure"


def plot_status(source: Path, jobs: list[dict], output: Path) -> None:
    rows = json.loads(source.read_text())
    xs = sorted({int(r["ec1_target_mhz"]) for r in rows})
    ys = sorted({int(r["ec2_target_mhz"]) for r in rows})
    grid = [[float("nan") for _ in xs] for _ in ys]
    for row in rows:
        x, y = int(row["ec1_target_mhz"]), int(row["ec2_target_mhz"])
        grid[ys.index(y)][xs.index(x)] = float(row["status"] == "validated_pass")
    styles = {
        "complete with selection": ("#16865c", "o"),
        "complete without selection": ("#e69f00", "s"),
        "no connected readout component": ("#cc3311", "x"),
        "insufficient anchor samples": ("#ee7733", "x"),
        "empty gate-overlay region": ("#7a4fa3", "X"),
        "execution failure": ("#000000", "+"),
    }
    fig, ax = plt.subplots(figsize=(9.5, 8), layout="constrained")
    ax.imshow(
        grid, origin="lower", interpolation="nearest",
        extent=[min(xs)-1, max(xs)+1, min(ys)-1, max(ys)+1], aspect="equal",
        cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
        norm=BoundaryNorm([-.5, .5, 1.5], 2),
    )
    grouped = {name: [] for name in styles}
    for job in jobs:
        name, ec = job_identity(job)
        category = job["campaign_category"]
        grouped[category].append((ec[0], ec[1], name))
    handles = []
    for category, points in grouped.items():
        if not points:
            continue
        color, marker = styles[category]
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=30,
                   color=color, marker=marker, linewidths=1.2, zorder=4)
        handles.append(Line2D([], [], marker=marker, ls="", color=color,
                              label=f"{category} ({len(points)})"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set(
        title="Rail-A partition campaign outcomes",
        xlabel=r"$E_{C,A1}/h$ (MHz)", ylabel=r"$E_{C,A2}/h$ (MHz)",
    )
    ax.set_aspect("equal")
    fig.savefig(output / "campaign_status_on_rail_a_readout_map.png", dpi=220)
    fig.savefig(output / "campaign_status_on_rail_a_readout_map.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--rail-a-results", type=Path, required=True)
    parser.add_argument("--rail-b-results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = (args.output or campaign / "campaign_summary").resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection.A_RESULTS = args.rail_a_results.resolve()
    selection.B_RESULTS = args.rail_b_results.resolve()

    status = json.loads((campaign / "campaign_status.json").read_text())
    summary_path = output / "campaign_summary.json"
    cached = {}
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        cached = {row["selection_id"]: row for row in previous.get("selections", [])}
    selected = []
    audited_jobs = []
    for job in status["jobs"]:
        record = dict(job)
        identity, ec = job_identity(job)
        directory = Path(job["directory"])
        if job["state"] == "complete":
            try:
                if identity in cached:
                    best = cached[identity]
                else:
                    best, audit = selection.select_one(directory)
                    best.update(selection_id=identity, ec_a_mhz=ec,
                                source_repetition=str(directory), selection_audit=audit)
                selected.append(best)
                record["campaign_category"] = "complete with selection"
            except RuntimeError:
                record["campaign_category"] = "complete without selection"
        else:
            record["campaign_category"] = failure_class(directory)
        audited_jobs.append(record)

    if selected:
        selection.plot_readout_map(selected, output)
    plot_status(selection.A_RESULTS, audited_jobs, output)
    category_counts = Counter(r["campaign_category"] for r in audited_jobs)
    payload = {
        "campaign": str(campaign),
        "selection_policy": {
            "raw_fidelity_strictly_greater_than": selection.RAW_MIN,
            "postselected_fidelity_strictly_greater_than": selection.POST_MIN,
            "neighborhood_radius_mhz": selection.NEIGHBORHOOD_RADIUS_MHZ,
            "minimum_anchor_consistent_robustness_margin_mhz": selection.MIN_MARGIN_MHZ,
            "primary_ranking": "minimum neighborhood maximum absolute rail-B bare-frequency shift",
            "fidelity_summary": "neighborhood median for raw and post-selected fidelity",
        },
        "counts": dict(category_counts),
        "selections": selected,
        "jobs": audited_jobs,
    }
    (output / "campaign_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "best_configurations.csv").open("w", newline="") as stream:
        fields = [
            "selection_id", "ec_a_mhz", "ec_b_mhz", "selected_anchor_id",
            "neighborhood_raw_fidelity_median",
            "neighborhood_postselected_fidelity_median",
            "neighborhood_maximum_absolute_bare_frequency_shift_mhz",
            "anchor_consistent_robustness_margin_mhz",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], list) else row[key]
                             for key in fields})
    print(json.dumps({"output": str(output), "counts": dict(category_counts),
                      "selected": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
