#!/usr/bin/env python3
"""Plot the five-anchor S1 comparison at the promoted full EC tuple."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2Q-gate_XM"))

from xm_cz.anchor_evaluator import evaluate_anchor_set  # noqa: E402


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "g5_corrected_cavity_A158_240_2026-11-17"
OUTPUT = HERE.parent / "output/ec_tuple_selection_slides/actual_anchor_s1"
EC_B_MHZ = [158, 226]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((CAMPAIGN / "config.json").read_text())
    payload = config["task_defaults"] | {
        "ec_GHz": [0.158, 0.240, EC_B_MHZ[0] / 1000, EC_B_MHZ[1] / 1000]
    }

    # The historical campaign used a hard 5.0 GHz rail-A bare-frequency bound,
    # which excluded the nominal 5.000 GHz anchor because its two bare f01 values
    # are 5.0062 and 5.0003 GHz.  Use the current intended anchor-derived bounds
    # so all five complete anchor witnesses receive the same S1 screening.
    bare = [value for anchor in payload["rail_A_anchors"] for value in anchor["bare_f01_GHz"]]
    anchor_bounds = [min(bare) - 0.01, max(bare) + 0.01]
    payload["bare_f01_bounds_GHz"]["A1"] = anchor_bounds
    payload["bare_f01_bounds_GHz"]["A2"] = list(anchor_bounds)

    result = evaluate_anchor_set(payload)
    screening = result["anchor_screening"]
    if len(screening) != 5 or any(row.get("S1") is None for row in screening):
        raise RuntimeError("Expected five valid anchor-screening scores")

    x = np.asarray([float(row["readout_operating_frequency_GHz"]) for row in screening])
    score = np.asarray([float(row["S1"]) for row in screening])
    minimum = float(score.min())
    relative_ppm = 1e6 * (score - minimum)
    selected_id = result["selected_anchor"]["anchor_id"]
    selected_index = next(index for index, row in enumerate(screening) if row["anchor_id"] == selected_id)

    fig, axis = plt.subplots(figsize=(9.6, 6.2), layout="constrained")
    axis.axvspan(4.55, 5.05, color="#DDEFEA", alpha=0.85, zorder=0)
    axis.scatter(x, relative_ppm, s=150, color="#F28E2B", edgecolor="white", linewidth=1.8,
                 zorder=3, label="Five readout-valid anchors")
    axis.scatter([x[selected_index]], [relative_ppm[selected_index]], s=230, facecolors="none",
                 edgecolors="#177E89", linewidths=2.8, zorder=4, label="Selected minimum-$S_1$ anchor")
    axis.axvline(x[selected_index], color="#177E89", linewidth=1.7, linestyle="--", alpha=0.9)

    for frequency, delta, absolute in zip(x, relative_ppm, score):
        axis.annotate(
            rf"$S_1={absolute:.6f}$",
            (frequency, delta),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#26343D",
        )

    axis.set(
        xlim=(4.54, 5.06),
        ylim=(-7, max(relative_ppm) + 19),
        xlabel=r"$f_A$ (GHz)",
        ylabel=r"Aggregate $S_1$ above five-anchor minimum (ppm)",
        title=(
            r"Actual fixed-rail-A workflow: five-anchor $S_1$ comparison" + "\n"
            r"$E_C/h=(158,240,158,226)$ MHz,  $g/h=5$ MHz"
        ),
    )
    axis.legend(frameon=False, fontsize=10.5, loc="upper left")
    axis.tick_params(labelsize=11)
    axis.title.set_fontsize(17)
    axis.title.set_fontweight("semibold")
    for spine in axis.spines.values():
        spine.set_color("#26343D")
        spine.set_linewidth(1.0)

    stem = "actual_workflow_five_anchor_s1_ec_158_240_158_226"
    png = OUTPUT / f"{stem}.png"
    pdf = OUTPUT / f"{stem}.pdf"
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    record = {
        "schema": "actual-five-anchor-s1-slide-v1",
        "ec_tuple_mhz": [158, 240, *EC_B_MHZ],
        "selected_anchor_id": selected_id,
        "anchor_derived_bare_frequency_bounds_ghz": anchor_bounds,
        "historical_fifth_anchor_note": (
            "The saved g=5 campaign excluded fA_5.000 at screening because the old hard A bare-f01 "
            "bound ended at 5.0 GHz. This presentation rerun uses the current anchor-derived bounds."
        ),
        "screening": screening,
        "fidelity_attempts": result.get("fidelity_attempts"),
        "figure_png": str(png.resolve()),
        "figure_pdf": str(pdf.resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "screening"}, indent=2))


if __name__ == "__main__":
    main()
