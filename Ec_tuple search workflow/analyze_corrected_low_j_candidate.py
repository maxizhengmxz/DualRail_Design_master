#!/usr/bin/env python3
"""Create the complete corrected-cavity record for EC=(158,240,158,226) MHz."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "2Q-gate_XM"), str(ROOT)]
from xm_cz.model import fixed_pair_model, local_model_from_ej

CAMPAIGN = HERE / "corrected_cavity_full_neighborhood_A158_240_B158_226_2026-11-17"
CENTER_REPETITION = CAMPAIGN / "repetitions/A_11_ecA_158_240"
CENTER = (158, 240, 158, 226)
OUT = CAMPAIGN / "candidate_analysis"


def quantiles(values):
    data = np.asarray([value for value in values if value is not None], dtype=float)
    return {
        "minimum": float(data.min()),
        "q05": float(np.quantile(data, 0.05)),
        "median": float(np.median(data)),
        "q95": float(np.quantile(data, 0.95)),
        "maximum": float(data.max()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((CAMPAIGN / "corrected_neighborhood_summary.json").read_text())

    gate_db = CENTER_REPETITION / "gate_output/compact_anchor_results.sqlite"
    connection = sqlite3.connect(gate_db)
    stored = connection.execute(
        "SELECT status,raw_fidelity,postselected_fidelity,selected_anchor_id,result_json "
        "FROM gate_results WHERE round(ec_b1_mhz)=? AND round(ec_b2_mhz)=?",
        CENTER[2:],
    ).fetchone()
    connection.close()
    if stored is None:
        raise RuntimeError("Corrected center gate result is missing")
    status, raw, post, anchor_id, payload = stored
    gate_result = json.loads(payload)
    final = gate_result["final"]

    reconciliation_db = CENTER_REPETITION / "fixed_hardware_reconciliation/fixed_hardware_results.sqlite"
    connection = sqlite3.connect(reconciliation_db)
    saved = connection.execute(
        "SELECT reconciled_pass,details_json FROM reconciliation WHERE ec_b1_mhz=? AND ec_b2_mhz=?",
        CENTER[2:],
    ).fetchone()
    connection.close()
    if saved is None:
        raise RuntimeError("Corrected center reconciliation result is missing")
    reconciled_pass, details_text = saved
    reconciliation = json.loads(details_text)

    ec_b = [value / 1000.0 for value in CENTER[2:]]
    gate_ej_b = list(map(float, reconciliation["gate"]["EJ_GHz"][2:]))
    idle = reconciliation["rail_b"]["point"]
    gate_bare = [float(local_model_from_ej(ec_b[i], gate_ej_b[i])[1][1]) for i in (0, 1)]
    gate_pair = fixed_pair_model(
        ec_b[0], ec_b[1], gate_ej_b[0], gate_ej_b[1],
        reconciliation["gate"]["couplings_GHz"]["B1_B2"], 8,
    )
    gate_operating_b = float(np.mean([gate_pair["energies"][mode] for mode in gate_pair["modes"]]))
    idle_bare = [float(idle["bare_f1_ghz"]), float(idle["bare_f2_ghz"])]
    shifts = [1000.0 * (gate_bare[i] - idle_bare[i]) for i in (0, 1)]

    rows = summary["rows"]
    metrics = {
        "raw_fidelity": quantiles([row["raw_fidelity"] for row in rows]),
        "postselected_fidelity": quantiles([row["postselected_fidelity"] for row in rows]),
        "gate_time_ns": quantiles([row["gate_time_ns"] for row in rows]),
        "required_interrail_coupling_mhz": quantiles(
            [row["required_interrail_coupling_mhz"] for row in rows]
        ),
        "aggregate_s1": quantiles([row["aggregate_s1"] for row in rows]),
        "maximum_logical_ratio": quantiles([row["maximum_logical_ratio"] for row in rows]),
        "gate_frequency_ghz": quantiles([row["gate_frequency_ghz"] for row in rows]),
    }
    j = metrics["required_interrail_coupling_mhz"]
    metrics["j_req_full_relative_spread"] = (
        (j["maximum"] - j["minimum"]) / (0.5 * (j["maximum"] + j["minimum"]))
    )
    metrics["logical_source_counts"] = dict(Counter(row["logical_source"] for row in rows))
    metrics["selected_anchor_counts"] = dict(Counter(row["selected_anchor_id"] for row in rows))
    metrics["hard_gate_pass_count"] = summary["hard_gate_pass_count"]
    metrics["fixed_hardware_pass_count"] = summary["fixed_hardware_pass_count"]

    center_record = {
        "schema": "corrected-low-j-candidate-complete-record-v1",
        "ec_tuple_mhz": list(CENTER),
        "cavity_provenance": {
            "rail_a_bare_resonator_ghz": 6.05,
            "rail_b_bare_resonator_ghz": 5.95,
            "rail_a_readout_results": str(json.loads((CAMPAIGN / "campaign_manifest.json").read_text())["rail_a_results"]),
            "rail_b_readout_results": str(json.loads((CAMPAIGN / "campaign_manifest.json").read_text())["rail_b_results"]),
        },
        "gate_status": status,
        "fixed_hardware_reconciliation_pass": bool(reconciled_pass),
        "selected_anchor_id": anchor_id,
        "selected_anchor": gate_result["selected_anchor"],
        "channel_selection": gate_result["channel_selection"],
        "gate_result": final,
        "rail_a_fixed_hardware_audit": reconciliation["rail_a_fixed_audit"],
        "rail_b_idle_readout_witness": idle,
        "rail_b_fixed_hardware_reconciliation": {
            key: value for key, value in reconciliation["rail_b"].items() if key != "point"
        },
        "derived_physical_metrics": {
            "rail_b_idle_operating_frequency_ghz": float(idle["operating_frequency_ghz"]),
            "rail_b_gate_operating_frequency_ghz": gate_operating_b,
            "rail_b_idle_bare_frequencies_ghz": idle_bare,
            "rail_b_gate_bare_frequencies_ghz": gate_bare,
            "rail_b_bare_frequency_shifts_gate_minus_idle_mhz": shifts,
            "maximum_absolute_rail_b_bare_frequency_shift_mhz": max(map(abs, shifts)),
            "rail_a_bare_frequencies_ghz": gate_result["selected_anchor"]["bare_f01_GHz"],
            "minimum_rail_a_b_idle_bare_detuning_mhz": 1000.0 * min(
                b - a
                for b in idle_bare
                for a in gate_result["selected_anchor"]["bare_f01_GHz"]
            ),
            "minimum_absolute_rail_a_b_gate_bare_separation_mhz": 1000.0 * min(
                abs(b - a)
                for b in gate_bare
                for a in gate_result["selected_anchor"]["bare_f01_GHz"]
            ),
        },
        "corrected_neighborhood_metrics": metrics,
        "raw_gate_result": gate_result,
        "raw_reconciliation_record": reconciliation,
    }
    (OUT / "complete_corrected_configuration.json").write_text(
        json.dumps(center_record, indent=2) + "\n"
    )
    (OUT / "corrected_neighborhood_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), layout="constrained")
    panels = (
        ("required_interrail_coupling_mhz", r"$J_{\rm req}/h$ (MHz)", "#2A9D8F"),
        ("gate_time_ns", "Calibrated gate time (ns)", "#4575B4"),
        ("raw_fidelity", "Raw CZ fidelity (%)", "#E76F51"),
        ("postselected_fidelity", "Post-selected CZ fidelity (%)", "#6A4C93"),
    )
    for axis, (key, label, color) in zip(axes.flat, panels):
        scale = 100.0 if "fidelity" in key else 1.0
        values = np.asarray([row[key] for row in rows], dtype=float) * scale
        axis.hist(values, bins=18, color=color, alpha=0.85, edgecolor="white")
        axis.axvline(np.median(values), color="black", linestyle="--", linewidth=1.2,
                     label=f"median {np.median(values):.4g}")
        axis.set_xlabel(label)
        axis.set_ylabel("Neighborhood tuple count")
        axis.legend(frameon=False)
    figure.suptitle(
        "Corrected-cavity 5 MHz neighborhood: EC=(158,240,158,226) MHz\n"
        "233/233 gate-pass and 233/233 fixed-hardware-pass tuples",
        fontsize=14,
    )
    figure.savefig(OUT / "corrected_neighborhood_metric_distributions.png", dpi=220)
    figure.savefig(OUT / "corrected_neighborhood_metric_distributions.pdf")
    plt.close(figure)

    print(json.dumps({
        "output": str(OUT),
        "center": list(CENTER),
        "center_raw_fidelity": raw,
        "center_postselected_fidelity": post,
        "center_j_req_mhz": 1000.0 * final["couplings_GHz"]["A2_B1"],
        "center_gate_time_ns": final["time_ns"],
        "neighborhood": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
