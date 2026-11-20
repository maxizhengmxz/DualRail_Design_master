#!/usr/bin/env python3
"""Create presentation-ready split-cavity readout-feasibility maps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "cavity_split_campaign_2026-11-09/readout/direct_ec"
OUTPUT = HERE.parent / "output/ec_tuple_selection_slides/readout_feasibility"
SPECS = (
    (
        "Rail A",
        DATA / "rail_a_step2_fr6p05/results.json",
        "Logical frequency: 140–160 MHz   |   Bare cavity: 6.05 GHz",
    ),
    (
        "Rail B",
        DATA / "rail_b_step2_fr5p95/results.json",
        "Logical frequency: 80–100 MHz   |   Bare cavity: 5.95 GHz",
    ),
)

FEASIBLE_COLOR = "#177E89"
AXIS_LIMITS = (145, 290)


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rows = json.loads(path.read_text())
    xs = np.asarray(sorted({float(row["ec1_target_mhz"]) for row in rows}))
    ys = np.asarray(sorted({float(row["ec2_target_mhz"]) for row in rows}))
    mask = np.full((len(ys), len(xs)), np.nan)
    x_index = {value: index for index, value in enumerate(xs)}
    y_index = {value: index for index, value in enumerate(ys)}
    count = 0
    for row in rows:
        if row.get("status") == "validated_pass":
            mask[y_index[float(row["ec2_target_mhz"])], x_index[float(row["ec1_target_mhz"])]] = 1.0
            count += 1
    return xs, ys, mask, count


def draw(axis, title: str, subtitle: str, values) -> int:
    xs, ys, mask, count = values
    axis.imshow(
        mask,
        origin="lower",
        interpolation="nearest",
        extent=(xs.min() - 1, xs.max() + 1, ys.min() - 1, ys.max() + 1),
        cmap=ListedColormap([FEASIBLE_COLOR]),
        vmin=0,
        vmax=1,
        aspect="equal",
    )
    axis.plot(AXIS_LIMITS, AXIS_LIMITS, color="#A9B4BE", linewidth=1.2, linestyle="--", zorder=0)
    axis.set(
        xlim=AXIS_LIMITS,
        ylim=AXIS_LIMITS,
        xlabel=r"$E_{C,1}/h$ (MHz)",
        ylabel=r"$E_{C,2}/h$ (MHz)",
    )
    axis.set_title(title, fontsize=20, fontweight="semibold", pad=28)
    axis.text(
        0.5,
        1.025,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#53606A",
    )
    axis.tick_params(labelsize=11)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#26343D")
    return count


def save_figure(fig, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=240, facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = [(title, subtitle, load(path)) for title, path, subtitle in SPECS]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.3), layout="constrained")
    counts = [draw(axis, title, subtitle, values) for axis, (title, subtitle, values) in zip(axes, loaded)]
    fig.legend(
        handles=[Patch(facecolor=FEASIBLE_COLOR, edgecolor="none", label="Readout feasible")],
        loc="outside lower center",
        frameon=False,
        fontsize=12,
    )
    fig.suptitle("Independent split-cavity readout feasibility", fontsize=22, fontweight="semibold")
    save_figure(fig, "split_cavity_rail_a_rail_b_readout_feasibility")

    for title, subtitle, values in loaded:
        fig, axis = plt.subplots(figsize=(7.2, 6.5), layout="constrained")
        draw(axis, title, subtitle, values)
        axis.legend(
            handles=[Patch(facecolor=FEASIBLE_COLOR, edgecolor="none", label="Readout feasible")],
            loc="upper right",
            frameon=False,
            fontsize=11,
        )
        save_figure(fig, f"split_cavity_{title.lower().replace(' ', '_')}_readout_feasibility")

    summary = {
        "schema": "slide-readout-feasibility-v1",
        "source": {title: str(path.resolve()) for title, path, _ in SPECS},
        "feasible_point_counts": {loaded[index][0]: counts[index] for index in range(len(loaded))},
        "axis_limits_mhz": list(AXIS_LIMITS),
        "figures": sorted(path.name for path in OUTPUT.iterdir() if path.suffix in {".png", ".pdf"}),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
