#!/usr/bin/env python3
"""Plot every N<=4, Delta-N=0 transition below a frequency cutoff."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_transition_frequencies import (
    CATEGORY_COLORS,
    enumerate_transitions,
    projected_model,
    state_text,
)


def write_rows(rows: list[dict], path: Path, frequency_key: str) -> None:
    fields = [
        "initial_N",
        "left",
        "right",
        "category",
        frequency_key,
        "bare_charge_amplitude",
        "coupled_charge_amplitude",
        "dominant_ports",
        "minimum_assignment_weight",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "left": state_text(row["left"]),
                    "right": state_text(row["right"]),
                }
            )


def draw_panel(
    axis: plt.Axes,
    rows: list[dict],
    frequency_key: str,
    title: str,
    cutoff_MHz: float,
) -> None:
    positions = np.arange(len(rows))
    frequencies_MHz = np.asarray([1000.0 * row[frequency_key] for row in rows])
    labels = [
        f"N={row['initial_N']}  {state_text(row['left'])} → {state_text(row['right'])}"
        for row in rows
    ]
    previous_n = None
    for position, row, frequency_MHz in zip(positions, rows, frequencies_MHz):
        if row["initial_N"] != previous_n:
            if position:
                axis.axhline(position - 0.5, color="#222222", lw=0.8, alpha=0.3)
            previous_n = row["initial_N"]
        color = CATEGORY_COLORS[row["category"]]
        axis.hlines(position, 0.0, frequency_MHz, color=color, lw=1.8, alpha=0.85)
        axis.plot(frequency_MHz, position, "o", color=color, ms=5)
        axis.annotate(
            f" {frequency_MHz:.4f}",
            (frequency_MHz, position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    if not rows:
        axis.text(
            0.5,
            0.5,
            "No transitions below cutoff",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#555555",
        )
    axis.set_yticks(positions, labels, fontsize=8)
    axis.invert_yaxis()
    if rows:
        visible_maximum = max(float(np.max(frequencies_MHz)), 1e-6)
        axis.set_xlim(0.0, visible_maximum * 1.28)
    else:
        axis.set_xlim(0.0, cutoff_MHz)
    axis.set_xlabel("Transition frequency (MHz)")
    axis.set_title(f"{title} ({len(rows)} transitions)", fontsize=13, pad=10)
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "current_gate_point_config.json"),
    )
    parser.add_argument("--cutoff-mhz", type=float, default=1.0)
    parser.add_argument("--max-total-excitation", type=int, default=4)
    parser.add_argument(
        "--output-directory", default="low_frequency_delta_N_0_N4"
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw = json.loads(config_path.read_text())
    output = Path(__file__).resolve().parents[2] / "output" / args.output_directory
    output.mkdir(parents=True, exist_ok=True)

    model = projected_model(raw, max_excitation=args.max_total_excitation)
    all_rows = enumerate_transitions(model, delta_n=0)
    cutoff_GHz = args.cutoff_mhz / 1000.0
    bare_rows = sorted(
        [row for row in all_rows if row["bare_frequency_GHz"] < cutoff_GHz],
        key=lambda row: (row["initial_N"], row["bare_frequency_GHz"]),
    )
    coupled_rows = sorted(
        [row for row in all_rows if row["coupled_frequency_GHz"] < cutoff_GHz],
        key=lambda row: (row["initial_N"], row["coupled_frequency_GHz"]),
    )

    height = max(8.0, 0.34 * max(len(bare_rows), len(coupled_rows)) + 2.5)
    figure, axes = plt.subplots(1, 2, figsize=(19, height))
    draw_panel(
        axes[0], bare_rows, "bare_frequency_GHz", "Couplings off", args.cutoff_mhz
    )
    draw_panel(
        axes[1], coupled_rows, "coupled_frequency_GHz", "Couplings on", args.cutoff_mhz
    )
    handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=category)
        for category, color in CATEGORY_COLORS.items()
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"Current TDT gate point: ΔN = 0 transitions below {args.cutoff_mhz:g} MHz, "
        f"N ≤ {args.max_total_excitation}",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(left=0.22, right=0.98, top=0.875, bottom=0.06, wspace=0.42)
    figure.savefig(
        output / "delta_N_0_transitions_below_1MHz.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(figure)

    write_rows(bare_rows, output / "couplings_off_transitions_below_1MHz.csv", "bare_frequency_GHz")
    write_rows(
        coupled_rows,
        output / "couplings_on_transitions_below_1MHz.csv",
        "coupled_frequency_GHz",
    )
    print(f"Couplings off: {len(bare_rows)} transitions below {args.cutoff_mhz:g} MHz")
    print(f"Couplings on: {len(coupled_rows)} transitions below {args.cutoff_mhz:g} MHz")
    print(f"Saved output to {output}")


if __name__ == "__main__":
    main()
