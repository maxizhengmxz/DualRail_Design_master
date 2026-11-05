#!/usr/bin/env python3
"""Plot one completed direct-EC readout campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def run(campaign: Path, output: Path, title: str) -> dict:
    progress = json.loads((campaign / "progress.json").read_text())
    if progress["phase"] != "complete":
        raise RuntimeError(f"Campaign is not complete: {progress}")
    settings = json.loads((campaign / "settings.json").read_text())
    rows = json.loads((campaign / "results.json").read_text())
    step = float(settings["ec_step_mhz"])
    xs = sorted({row["ec1_target_mhz"] for row in rows})
    ys = sorted({row["ec2_target_mhz"] for row in rows})
    xi = {value: index for index, value in enumerate(xs)}
    yi = {value: index for index, value in enumerate(ys)}
    status = np.full((len(ys), len(xs)), np.nan)
    logical = np.full_like(status, np.nan)
    for row in rows:
        i, j = yi[row["ec2_target_mhz"]], xi[row["ec1_target_mhz"]]
        passed = row["status"] == "validated_pass"
        status[i, j] = float(passed)
        if passed:
            logical[i, j] = 1000.0 * row["point"]["splitting_ghz"]
    passed = [row for row in rows if row["status"] == "validated_pass"]
    frequencies = [1000.0 * row["point"]["splitting_ghz"] for row in passed]
    extent = [min(xs)-step/2, max(xs)+step/2, min(ys)-step/2, max(ys)+step/2]
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.1), layout="constrained")
    cmap = ListedColormap(["#d8dee3", "#2a9d8f"])
    norm = BoundaryNorm([-.5, .5, 1.5], 2)
    axes[0].imshow(status, origin="lower", interpolation="nearest", extent=extent, aspect="equal", cmap=cmap, norm=norm)
    axes[0].legend(handles=[Patch(color=cmap(0), label="No validated witness found"), Patch(color=cmap(1), label="Validated readout")], loc="lower right", fontsize=8)
    axes[0].set_title(f"Feasibility ({len(passed)}/{len(rows)} validated)")
    image = axes[1].imshow(logical, origin="lower", interpolation="nearest", extent=extent, aspect="equal", cmap="viridis", vmin=settings["logical_frequency_mhz"][0], vmax=settings["logical_frequency_mhz"][1])
    axes[1].set_title(f"Selected-witness logical frequency\n{min(frequencies):.2f}-{max(frequencies):.2f} MHz")
    fig.colorbar(image, ax=axes[1], label="Logical frequency (MHz)", shrink=.82)
    for ax in axes:
        ax.set(xlabel=r"$E_{C1}/h$ (MHz)", ylabel=r"$E_{C2}/h$ (MHz)")
    fig.suptitle(f"{title} - {step:g} MHz EC lattice, {settings['hull_scale']:g} scaled two-orientation hull", fontsize=15, fontweight="bold")
    png = output.with_suffix(".png"); pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=220); fig.savefig(pdf); plt.close(fig)
    summary = {
        "campaign": str(campaign), "ec_step_mhz": step, "targets": len(rows),
        "validated": len(passed), "validation_fraction": len(passed)/len(rows),
        "logical_frequency_requested_mhz": settings["logical_frequency_mhz"],
        "logical_frequency_found_range_mhz": [min(frequencies), max(frequencies)],
        "ec1_range_mhz": [min(xs), max(xs)], "ec2_range_mhz": [min(ys), max(ys)],
        "plots": [str(png), str(pdf)],
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output stem without extension")
    parser.add_argument("--title", default="Independent direct-EC readout map")
    args = parser.parse_args()
    print(json.dumps(run(args.campaign.resolve(), args.output.resolve(), args.title), indent=2))
