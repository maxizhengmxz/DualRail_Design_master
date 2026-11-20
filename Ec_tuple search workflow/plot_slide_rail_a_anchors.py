#!/usr/bin/env python3
"""Plot the five rail-A anchors and their gate-model logical frequencies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "g5_corrected_cavity_A158_240_2026-11-17"
ANCHORS = CAMPAIGN / "rail_A_anchors.json"
SCAN_SUMMARY = CAMPAIGN / "readout_frequency_scan/summary.json"
OUTPUT = HERE.parent / "output/ec_tuple_selection_slides/rail_a_anchors"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    anchors = json.loads(ANCHORS.read_text())
    summary = json.loads(SCAN_SUMMARY.read_text())
    component = max(summary["intervals"], key=lambda item: item["sample_count"])
    lower = float(component["sampled_lower_ghz"])
    upper = float(component["sampled_upper_ghz"])

    x = np.asarray([float(anchor["readout_operating_frequency_GHz"]) for anchor in anchors])
    y = 1000.0 * np.asarray([float(anchor["gate_logical_frequency_GHz"]) for anchor in anchors])

    fig, axis = plt.subplots(figsize=(10.4, 6.2), layout="constrained")
    axis.axvspan(lower, upper, color="#DDEFEA", alpha=0.9, zorder=0)
    axis.scatter(x, y, s=130, color="#F28E2B", edgecolor="white", linewidth=1.6, zorder=3)

    offsets = ((0, 13), (0, 13), (0, 13), (0, -23), (0, 13))
    for frequency, logical, offset in zip(x, y, offsets):
        axis.annotate(
            f"{logical:.3f} MHz",
            (frequency, logical),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=10.5,
            color="#26343D",
        )

    bracket_y = 144.69
    axis.annotate(
        "",
        xy=(upper, bracket_y),
        xytext=(lower, bracket_y),
        arrowprops=dict(arrowstyle="<->", color="#177E89", linewidth=2.2),
    )
    axis.text(
        (lower + upper) / 2,
        bracket_y + 0.025,
        f"Complete connected readout-feasible range: {lower:.2f}–{upper:.2f} GHz",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#11636C",
    )

    axis.set(
        xlim=(lower - 0.05, upper + 0.05),
        ylim=(144.62, 146.05),
        xlabel=r"$f_A$ (GHz)",
        ylabel=r"$f_{L,A}$ (MHz)",
        title=(
            "Five rail-A operating-frequency anchors\n"
            r"$E_{C,A}/h=(158,240)$ MHz   |   allowed logical band: 140–160 MHz"
        ),
    )
    axis.tick_params(labelsize=11)
    axis.title.set_fontsize(18)
    axis.title.set_fontweight("semibold")
    for spine in axis.spines.values():
        spine.set_color("#26343D")
        spine.set_linewidth(1.0)

    stem = "rail_a_five_anchors_logical_vs_operating_frequency"
    fig.savefig(OUTPUT / f"{stem}.png", dpi=240, facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)

    record = {
        "schema": "slide-rail-a-anchor-frequency-v1",
        "ec_a_mhz": [158, 240],
        "connected_operating_frequency_range_ghz": [lower, upper],
        "logical_frequency_kind": "gate-model splitting computed from each anchor's fixed EJ and J",
        "anchors": [
            {
                "anchor_id": anchor["anchor_id"],
                "operating_frequency_ghz": float(anchor["readout_operating_frequency_GHz"]),
                "gate_logical_frequency_mhz": 1000.0 * float(anchor["gate_logical_frequency_GHz"]),
            }
            for anchor in anchors
        ],
        "figure_png": str((OUTPUT / f"{stem}.png").resolve()),
        "figure_pdf": str((OUTPUT / f"{stem}.pdf").resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
