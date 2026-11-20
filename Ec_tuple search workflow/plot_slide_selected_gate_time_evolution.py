#!/usr/bin/env python3
"""Plot exact static-gate dynamics for the promoted EC tuple."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "2Q-gate_XM"), str(ROOT)]

from xm_cz.model import fixed_pair_model, layout, pair_model


RECORD = (
    HERE
    / "corrected_cavity_full_neighborhood_A158_240_B158_226_2026-11-17"
    / "candidate_analysis"
    / "complete_corrected_configuration.json"
)
OUTPUT = ROOT / "output/ec_tuple_selection_slides/selected_gate_time_evolution"


def main() -> None:
    record = json.loads(RECORD.read_text())
    ec = np.asarray(record["ec_tuple_mhz"], dtype=float) / 1000.0
    anchor = record["selected_anchor"]
    fixed = anchor["fixed_A"]
    gate = record["gate_result"]
    channel = record["channel_selection"]

    rail_a = fixed_pair_model(ec[0], ec[1], *fixed["EJ_GHz"], fixed["J_GHz"], 8)
    rail_b = pair_model(
        ec[2], ec[3], float(gate["fB_GHz"]), float(gate["logical_splittings_GHz"][1])
    )
    energies, unit, logical, target, coupling, _ = layout(
        rail_a,
        rail_b,
        float(channel["target_g_mhz"]) / 1000.0,
        "0020",
        logical_source=str(channel["source_logical"]),
    )
    eigenvalues, vectors = np.linalg.eigh(np.diag(energies) + coupling * unit)

    gate_time = float(gate["time_ns"])
    times = np.linspace(0.0, 1.2 * gate_time, 721)
    logical = np.asarray(logical, dtype=int)
    source_index = int(gate["logical_source_index"])
    source = int(logical[source_index])

    returns = np.empty((4, len(times)))
    survival = np.empty_like(returns)
    auxiliary = np.empty(len(times))
    other = np.empty(len(times))
    phase = np.empty(len(times))
    for index, time_ns in enumerate(times):
        propagator = (
            vectors * np.exp(-2j * np.pi * eigenvalues * time_ns)
        ) @ vectors.T.conj()
        amplitudes = propagator[:, logical]
        returns[:, index] = np.abs(amplitudes[logical, np.arange(4)]) ** 2
        survival[:, index] = np.sum(np.abs(amplitudes[logical, :]) ** 2, axis=0)
        auxiliary[index] = abs(propagator[target, source]) ** 2
        other[index] = max(
            0.0, 1.0 - survival[source_index, index] - auxiliary[index]
        )
        diagonal = propagator[logical, logical]
        q = diagonal[3] * diagonal[0] * np.conj(diagonal[2] * diagonal[1])
        phase[index] = np.angle(q)
    phase = np.unwrap(phase)

    colors = ["#159D82", "#F2A51A", "#2E86D1", "#6657C8"]
    labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=True)
    for logical_index, (label, color) in enumerate(zip(labels, colors)):
        axes[0, 0].plot(times, returns[logical_index], color=color, lw=1.8, label=label)
        axes[0, 1].plot(times, survival[logical_index], color=color, lw=1.8, label=label)

    source_label = str(channel["source_logical"])
    axes[1, 0].plot(
        times,
        returns[source_index],
        color=colors[source_index],
        lw=2.2,
        label=rf"Return to $|{source_label}\rangle$",
    )
    axes[1, 0].plot(
        times,
        auxiliary,
        color="#F23B4D",
        lw=2.2,
        label=r"$|\widetilde{0020}\rangle$ population",
    )
    axes[1, 0].plot(times, other, color="#718096", lw=1.5, label="Other states")
    axes[1, 1].plot(times, phase / np.pi, color="#7B2CBF", lw=2.2)

    for axis in axes.flat:
        axis.axvline(gate_time, color="#D1495B", ls="--", lw=1.5)
        axis.set_xlabel("Interaction time (ns)")
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_color("#374151")
    axes[0, 0].set(
        title="Return probability of each logical input",
        ylabel="Probability",
        ylim=(-0.02, 1.02),
    )
    axes[0, 1].set(
        title="Logical-subspace survival",
        ylabel="Probability",
        ylim=(-0.02, 1.02),
    )
    axes[1, 0].set(
        title=r"Selected $|10\rangle\leftrightarrow|\widetilde{0020}\rangle$ dynamics",
        ylabel="Probability",
        ylim=(-0.02, 1.02),
    )
    axes[1, 1].set(
        title="Accumulated conditional phase",
        ylabel=r"Conditional phase / $\pi$",
    )
    axes[0, 0].legend(ncol=2, fontsize=9, frameon=False)
    axes[0, 1].legend(ncol=2, fontsize=9, frameon=False)
    axes[1, 0].legend(fontsize=9, frameon=False)
    axes[1, 1].text(
        gate_time,
        0.04,
        rf"  calibrated hold: {gate_time:.2f} ns",
        color="#B02A3C",
        fontsize=9.5,
        rotation=90,
        va="bottom",
    )
    fig.suptitle(
        "Selected-anchor gate dynamics\n"
        rf"$E_C/h=(158,240,158,226)$ MHz   |   $f_A=4.600$ GHz",
        fontsize=17,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.86, hspace=0.32, wspace=0.23)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = "selected_anchor_gate_time_evolution_A158_240_B158_226"
    fig.savefig(OUTPUT / f"{stem}.png", dpi=240, facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)

    combined_stem = "selected_interaction_dynamics_and_conditional_phase_A158_240_B158_226"
    combined, population_axis = plt.subplots(figsize=(10.6, 6.2))
    phase_axis = population_axis.twinx()
    population_axis.plot(
        times,
        returns[source_index],
        color=colors[source_index],
        lw=2.4,
        label=rf"Return to $|{source_label}\rangle$",
    )
    population_axis.plot(
        times,
        auxiliary,
        color="#F23B4D",
        lw=2.4,
        label=r"$|\widetilde{0020}\rangle$ population",
    )
    population_axis.plot(
        times, other, color="#718096", lw=1.5, label="Other states"
    )
    phase_axis.plot(
        times,
        phase / np.pi,
        color="#7B2CBF",
        lw=2.2,
        ls="--",
        label=r"Conditional phase / $\pi$",
    )
    population_axis.axvline(gate_time, color="#D1495B", ls=":", lw=1.7)
    population_axis.text(
        gate_time - 1.2,
        0.49,
        rf"calibrated hold: {gate_time:.2f} ns",
        rotation=90,
        ha="right",
        va="center",
        color="#B02A3C",
        fontsize=10,
    )
    population_axis.set(
        xlabel="Interaction time (ns)",
        ylabel="Population",
        ylim=(-0.02, 1.02),
        title=(
            r"Selected $|10\rangle\leftrightarrow|\widetilde{0020}\rangle$ dynamics and conditional phase"
            "\n"
            r"$f_A=4.600$ GHz"
        ),
    )
    phase_axis.set_ylabel(r"Conditional phase / $\pi$", color="#7B2CBF")
    phase_axis.tick_params(axis="y", colors="#7B2CBF")
    phase_axis.spines["right"].set_color("#7B2CBF")
    population_axis.grid(False)
    lines_left, labels_left = population_axis.get_legend_handles_labels()
    lines_right, labels_right = phase_axis.get_legend_handles_labels()
    population_axis.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper right",
        frameon=False,
        fontsize=9.5,
    )
    combined.subplots_adjust(left=0.10, right=0.90, bottom=0.13, top=0.84)
    combined.savefig(OUTPUT / f"{combined_stem}.png", dpi=240, facecolor="white")
    combined.savefig(OUTPUT / f"{combined_stem}.pdf", facecolor="white")
    plt.close(combined)

    np.savez_compressed(
        OUTPUT / f"{stem}.npz",
        times_ns=times,
        return_probabilities=returns,
        logical_survival=survival,
        selected_source_auxiliary_probability=auxiliary,
        selected_source_other_probability=other,
        conditional_phase_rad=phase,
    )
    summary = {
        "schema": "selected-anchor-gate-time-evolution-slide-v1",
        "ec_tuple_mhz": record["ec_tuple_mhz"],
        "selected_anchor_id": anchor["anchor_id"],
        "selected_anchor_operating_frequency_ghz": anchor["readout_operating_frequency_GHz"],
        "selected_logical_source": source_label,
        "selected_auxiliary": channel["auxiliary"],
        "calibrated_gate_frequency_ghz": gate["fB_GHz"],
        "calibrated_hold_time_ns": gate_time,
        "raw_fidelity": gate["raw_fidelity"],
        "postselected_fidelity": gate["postselected_fidelity"],
        "figure_png": str((OUTPUT / f"{stem}.png").resolve()),
        "figure_pdf": str((OUTPUT / f"{stem}.pdf").resolve()),
        "combined_figure_png": str((OUTPUT / f"{combined_stem}.png").resolve()),
        "combined_figure_pdf": str((OUTPUT / f"{combined_stem}.pdf").resolve()),
        "data_npz": str((OUTPUT / f"{stem}.npz").resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
