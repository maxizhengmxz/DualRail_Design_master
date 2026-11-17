#!/usr/bin/env python3
"""Plot the saved toy-model spectrum for representative eight-point design A3."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "2Q-gate_XM"), str(ROOT)]
from xm_cz.model import fixed_pair_model, pair_model

SELECTIONS = (
    HERE / "combined_overlay_selection_2026-11-12" / "idle_above_all_interactions_200MHz"
    / "eight_selected_eca_to_ecb" / "eight_selected_configurations.json"
)
OUTPUT = SELECTIONS.parent / "A3_energy_spectrum"


def assigned_state(model: dict, bare_index: int) -> int:
    return int(np.argmax(np.abs(model["vectors"][bare_index]) ** 2))


def main() -> None:
    summary = json.loads(SELECTIONS.read_text())
    chosen = next(row for row in summary if row["label"] == "A3")
    all_selections = json.loads(
        (SELECTIONS.parents[1] / "reselected_configurations.json").read_text()
    )["selections"]
    selected = next(row for row in all_selections if row["ec_a_mhz"] == chosen["ec_a_mhz"])
    repetition = Path(selected["source_repetition"])
    ec_b1, ec_b2 = selected["ec_b_mhz"]
    connection = sqlite3.connect(repetition / "gate_output" / "compact_anchor_results.sqlite")
    gate = json.loads(connection.execute(
        "SELECT result_json FROM gate_results WHERE ec_b1_mhz=? AND ec_b2_mhz=?",
        (ec_b1, ec_b2),
    ).fetchone()[0])
    connection.close()
    anchor = gate["selected_anchor"]
    ec = [float(value) for value in gate["ec_GHz"]]
    fixed = anchor["fixed_A"]
    rail_a = fixed_pair_model(
        ec[0], ec[1], *fixed["EJ_GHz"], fixed["J_GHz"], 8
    )

    selected_frequency = float(gate["final"]["fB_GHz"])
    candidate_frequencies = {
        row["logical_source"]: float(row["fB_GHz"])
        for screen in gate["screening"]
        for row in screen.get("logical_source_candidates", [])
        if row.get("status") == "valid" and row.get("fB_GHz") is not None
    }
    lower = max(4.0, min(candidate_frequencies.values()) - 0.35)
    upper = min(5.5, max(candidate_frequencies.values()) + 0.35)
    frequencies = np.linspace(lower, upper, 401)
    labels = ["00", "01", "10", "11", "B20", "B02", "B11", "A02", "A20", "A11"]
    traces = {label: [] for label in labels}

    a_modes = rail_a["modes"]
    a_states = {
        "A02": assigned_state(rail_a, 2),
        "A20": assigned_state(rail_a, 8),
        "A11": assigned_state(rail_a, 5),
    }
    for frequency in frequencies:
        rail_b = pair_model(ec[2], ec[3], float(frequency), gate["logical_splittings_GHz"][1])
        b_modes = rail_b["modes"]
        logical = {
            f"{ia}{ib}": rail_a["energies"][a_modes[ia]] + rail_b["energies"][b_modes[ib]]
            for ia in (0, 1) for ib in (0, 1)
        }
        reference = float(np.mean(list(logical.values())))
        for label, energy in logical.items():
            traces[label].append(1000.0 * (energy - reference))
        for label, bare_index in (("B20", 8), ("B02", 2), ("B11", 5)):
            state = assigned_state(rail_b, bare_index)
            traces[label].append(1000.0 * (rail_b["energies"][state] - reference))
        for label, state in a_states.items():
            traces[label].append(1000.0 * (rail_a["energies"][state] - reference))

    colors = {
        "00": "#1b9e77", "01": "#f0a11a", "10": "#3182bd", "11": "#6a5acd",
        "B20": "#ef3b4a", "B02": "#d65ad1", "B11": "#6c7f96",
        "A02": "#f06cb5", "A20": "#9a6a50", "A11": "#18a9b9",
    }
    fig, ax = plt.subplots(figsize=(10.5, 8.2), layout="constrained")
    for label in labels:
        ax.plot(frequencies, traces[label], color=colors[label], lw=1.7, label=label)
        ax.text(frequencies[-1] + .006, traces[label][-1], label,
                color=colors[label], fontsize=9, va="center", fontweight="bold")
    source_colors = {"00": "#1b9e77", "01": "#f0a11a", "10": "#3182bd", "11": "#6a5acd"}
    for source, frequency in sorted(candidate_frequencies.items(), key=lambda item: item[1]):
        ax.axvline(frequency, color=source_colors[source], lw=1, ls="--", alpha=.45)
        ax.annotate(
            f"{source}/B20\n{frequency:.4f} GHz", (frequency, 0),
            xytext=(4, 16 if source in ("00", "10") else -36),
            textcoords="offset points", fontsize=7.5, color=source_colors[source],
        )
    ax.axvline(selected_frequency, color="black", lw=1.7, ls=":",
               label=f"Selected gate point {selected_frequency:.4f} GHz")
    ax.set(
        title=(
            "A3 saved toy-model spectrum\n"
            rf"$E_{{C,A}}=({chosen['ec_a_mhz'][0]},{chosen['ec_a_mhz'][1]})$ MHz, "
            rf"$E_{{C,B}}=({chosen['ec_b_mhz'][0]},{chosen['ec_b_mhz'][1]})$ MHz"
        ),
        xlabel="Common bare rail-B frequency (GHz)",
        ylabel="Energy minus mean logical energy (MHz)",
        xlim=(lower, upper),
    )
    ax.grid(alpha=.18)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "A3_energy_spectrum.png", dpi=220)
    fig.savefig(OUTPUT / "A3_energy_spectrum.pdf")
    plt.close(fig)
    provenance = {
        "selection_label": "A3",
        "ec_a_mhz": chosen["ec_a_mhz"],
        "ec_b_mhz": chosen["ec_b_mhz"],
        "selected_anchor_id": gate["selected_anchor"]["anchor_id"],
        "selected_logical_source": gate["logical_source"],
        "selected_gate_frequency_ghz": selected_frequency,
        "logical_source_collision_frequencies_ghz": candidate_frequencies,
        "rail_a_fixed_EJ_ghz": fixed["EJ_GHz"],
        "rail_a_fixed_J_ghz": fixed["J_GHz"],
        "rail_b_target_logical_splitting_mhz": 1000.0 * gate["logical_splittings_GHz"][1],
        "source_repetition": str(repetition),
        "spectrum_definition": "uncoupled fixed-A/pair-B branches relative to pointwise mean of four logical energies",
    }
    (OUTPUT / "spectrum_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), **provenance}, indent=2))


if __name__ == "__main__":
    main()
