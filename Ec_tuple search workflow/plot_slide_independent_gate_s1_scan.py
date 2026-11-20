#!/usr/bin/env python3
"""Independent gate-only S1 scan for the promoted EC tuple."""

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

from xm_cz.evaluator import evaluate_ec  # noqa: E402


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "output/ec_tuple_selection_slides/independent_gate_scan"

EC_MHZ = [158, 240, 158, 226]
LOGICAL_MHZ = [150, 90]
FREQUENCY_RANGE_GHZ = [4.0, 5.5]
SAMPLE_STEP_GHZ = 0.025
TARGET_G_MHZ = 5.0


def payload(screening_only: bool) -> dict:
    result = {
        "ec_GHz": [value / 1000.0 for value in EC_MHZ],
        "logical_splittings_GHz": [value / 1000.0 for value in LOGICAL_MHZ],
        "bare_f01_bounds_GHz": {
            name: list(FREQUENCY_RANGE_GHZ) for name in ("A1", "A2", "B1", "B2")
        },
        "frequency_GHz": [*FREQUENCY_RANGE_GHZ, SAMPLE_STEP_GHZ],
        "resonance_channel": "0020",
        "g_GHz": TARGET_G_MHZ / 1000.0,
        "point_budget_seconds": 300,
    }
    if screening_only:
        result["screening_only"] = True
    return result


def compact_row(row: dict) -> dict:
    return {
        "fA_GHz": row.get("fA_GHz"),
        "status": row.get("status", "valid" if row.get("S1") is not None else None),
        "fB_GHz": row.get("fB_GHz"),
        "logical_source": row.get("logical_source"),
        "aggregate_S1": row.get("S1"),
        "intended_source_score": row.get("intended_source_score"),
        "maximum_logical_ratio": row.get("maximum_logical_ratio"),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    screened = evaluate_ec(payload(True))
    calibrated = evaluate_ec(payload(False))

    samples = [compact_row(row) for row in screened.get("screening", [])]
    valid = [row for row in samples if row["aggregate_S1"] is not None]
    if not valid:
        raise RuntimeError("Independent gate scan found no valid S1 samples")

    selected = screened["S1_position"]
    minimum = min(float(row["aggregate_S1"]) for row in valid)
    x = np.asarray([float(row["fA_GHz"]) for row in valid])
    y = 1e6 * (np.asarray([float(row["aggregate_S1"]) for row in valid]) - minimum)
    subset = np.isclose(np.mod(np.rint((x - FREQUENCY_RANGE_GHZ[0]) * 1000), 50), 0)

    fig, axis = plt.subplots(figsize=(9.4, 6.3), layout="constrained")
    axis.plot(x, y, color="#2C7FB8", linewidth=1.8, marker="o", markersize=3.8,
              label="25 MHz samples")
    axis.scatter(x[subset], y[subset], s=72, facecolors="none", edgecolors="#F28E2B",
                 linewidths=1.8, zorder=3, label="50 MHz subset")
    selected_fa = float(selected["fA_GHz"])
    boundary = abs(selected_fa - FREQUENCY_RANGE_GHZ[0]) < 1e-8 or abs(
        selected_fa - FREQUENCY_RANGE_GHZ[1]
    ) < 1e-8
    axis.axvline(selected_fa, color="#D1495B", linewidth=1.7, linestyle="--",
                 label="Selected minimum" + (" (boundary)" if boundary else ""))

    axis.set(
        xlabel=r"$f_A$ (GHz)",
        ylabel=r"Aggregate $S_1$ above sampled minimum (ppm)",
        title=(
            r"Independent gate-only $S_1$ optimization" + "\n"
            r"$E_C/h=(158,240,158,226)$ MHz,  "
            r"$f_{L,A}=150$ MHz,  $f_{L,B}=90$ MHz"
        ),
    )
    axis.grid(color="#DCE1E5", linewidth=0.8, alpha=0.65)
    axis.legend(frameon=True, framealpha=0.95, fontsize=10, loc="best")
    axis.tick_params(labelsize=11)
    axis.title.set_fontsize(17)
    axis.title.set_fontweight("semibold")
    for spine in axis.spines.values():
        spine.set_color("#26343D")
        spine.set_linewidth(1.0)

    stem = "independent_gate_s1_vs_fa_ec_158_240_158_226_fl_150_90"
    png = OUTPUT / f"{stem}.png"
    pdf = OUTPUT / f"{stem}.pdf"
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    final = calibrated.get("final") or {}
    record = {
        "schema": "independent-gate-s1-scan-slide-v1",
        "ec_tuple_mhz": EC_MHZ,
        "logical_frequencies_mhz": {"rail_a": LOGICAL_MHZ[0], "rail_b": LOGICAL_MHZ[1]},
        "target_g_mhz": TARGET_G_MHZ,
        "frequency_scan_ghz": [*FREQUENCY_RANGE_GHZ, SAMPLE_STEP_GHZ],
        "selected": {
            "fA_GHz": selected_fa,
            "frequency_boundary": boundary,
            "fB_collision_GHz": selected.get("fB_GHz"),
            "logical_source": screened.get("logical_source"),
            "aggregate_S1": selected.get("S1"),
            "intended_source_score": selected.get("intended_source_score"),
            "maximum_logical_ratio": selected.get("maximum_logical_ratio"),
        },
        "calibration": {
            "status": calibrated.get("status"),
            "fB_GHz": final.get("fB_GHz"),
            "gate_time_ns": final.get("time_ns"),
            "raw_fidelity": final.get("raw_fidelity"),
            "postselected_fidelity": final.get("postselected_fidelity"),
            "EJ_GHz": final.get("EJ_GHz"),
            "couplings_GHz": final.get("couplings_GHz"),
        },
        "samples": samples,
        "figure_png": str(png.resolve()),
        "figure_pdf": str(pdf.resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
