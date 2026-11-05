#!/usr/bin/env python3
"""Reconstruct and plot the selected A4 static CZ evolution."""
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
sys.path.insert(0, str(ROOT / "2Q-gate_XM"))
from xm_cz.model import fixed_pair_model, layout, local_model_from_ej, pair_model

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "sampled_runs" / "tonight_first12_B4_5p5"
OUTPUT = CAMPAIGN / "best_configuration_selection" / "A4_detailed_validation"


def detuning(a: list[float], b: list[float]) -> dict:
    pairwise = [[1000.0 * (bj - ai) for bj in b] for ai in a]
    return {
        "signed_B_minus_A_pairwise_mhz": pairwise,
        "same_index_B_minus_A_mhz": [pairwise[0][0], pairwise[1][1]],
        "center_B_minus_A_mhz": 1000.0 * (np.mean(b) - np.mean(a)),
        "closest_absolute_interrail_detuning_mhz": min(abs(v) for row in pairwise for v in row),
    }


def main() -> None:
    selections = json.loads((CAMPAIGN / "best_configuration_selection" / "best_configurations.json").read_text())["selections"]
    selected = next(row for row in selections if row["selection_id"] == "A4")
    repetition = Path(selected["source_repetition"])
    con = sqlite3.connect(repetition / "fixed_hardware_reconciliation" / "fixed_hardware_results.sqlite")
    text = con.execute(
        "SELECT details_json FROM reconciliation WHERE ec_b1_mhz=? AND ec_b2_mhz=?",
        selected["ec_b_mhz"],
    ).fetchone()[0]
    con.close()
    record = json.loads(text)
    record = record.get("selected", record)

    con = sqlite3.connect(repetition / "gate_output" / "compact_anchor_results.sqlite")
    gate_result = json.loads(con.execute(
        "SELECT result_json FROM gate_results WHERE ec_b1_mhz=? AND ec_b2_mhz=?",
        selected["ec_b_mhz"],
    ).fetchone()[0])
    con.close()
    final = gate_result["final"]
    anchor = gate_result["selected_anchor"]
    full_anchor = next(
        row for row in json.loads((repetition / "rail_A_anchors.json").read_text())
        if row["anchor_id"] == anchor["anchor_id"]
    )

    ec = [float(v) for v in gate_result["ec_GHz"]]
    a = fixed_pair_model(ec[0], ec[1], *anchor["fixed_A"]["EJ_GHz"], anchor["fixed_A"]["J_GHz"], 8)
    b = pair_model(ec[2], ec[3], final["fB_GHz"], gate_result["logical_splittings_GHz"][1])
    energies, unit, logical, target, coupling, scores = layout(
        a, b, gate_result["g_GHz"], gate_result["resonance_channel"],
        logical_source=gate_result["logical_source"]
    )
    hamiltonian = np.diag(energies) + coupling * unit
    eigenvalues, vectors = np.linalg.eigh(hamiltonian)
    gate_time = float(final["time_ns"])
    times = np.linspace(0.0, 130.0, 651)
    logical = np.asarray(logical, dtype=int)
    source = int(logical[gate_result["logical_source_index"]])

    returns = np.empty((4, len(times)))
    logical_survival = np.empty_like(returns)
    source_auxiliary = np.empty(len(times))
    source_other = np.empty(len(times))
    phase = np.empty(len(times))
    for k, t in enumerate(times):
        propagator = (vectors * np.exp(-2j * np.pi * eigenvalues * t)) @ vectors.T.conj()
        amplitudes = propagator[:, logical]
        returns[:, k] = np.abs(amplitudes[logical, np.arange(4)]) ** 2
        logical_survival[:, k] = np.sum(np.abs(amplitudes[logical, :]) ** 2, axis=0)
        source_auxiliary[k] = abs(propagator[target, source]) ** 2
        source_other[k] = 1.0 - logical_survival[gate_result["logical_source_index"], k] - source_auxiliary[k]
        diagonal = propagator[logical, logical]
        q = diagonal[3] * diagonal[0] * np.conj(diagonal[2] * diagonal[1])
        phase[k] = np.angle(q)
    phase = np.unwrap(phase)

    def at_time(t: float):
        propagator = (vectors * np.exp(-2j * np.pi * eigenvalues * t)) @ vectors.T.conj()
        matrix = propagator[np.ix_(logical, logical)]
        q = matrix[3, 3] * matrix[0, 0] * np.conj(matrix[2, 2] * matrix[1, 1])
        acceptance = float(np.sum(abs(matrix) ** 2) / 4)
        raw = float((4 * acceptance + np.sum(abs(np.diag(matrix))) ** 2) / 20)
        return matrix, acceptance, raw, float(np.angle(-q))

    matrix, acceptance, raw, phase_error = at_time(gate_time)
    reconstructed = {
        "raw_fidelity": raw,
        "postselected_fidelity": raw / acceptance,
        "acceptance": acceptance,
        "phase_error_rad": phase_error,
        "return_by_input": list(map(float, abs(np.diag(matrix)) ** 2)),
        "leakage_by_input": list(map(float, 1 - np.sum(abs(matrix) ** 2, axis=0))),
    }

    a_bare = [float(local_model_from_ej(ec[i], final["EJ_GHz"][i])[1][1]) for i in (0, 1)]
    b_gate_bare = [float(local_model_from_ej(ec[i+2], final["EJ_GHz"][i+2])[1][1]) for i in (0, 1)]
    b_idle = record["rail_b"]["point"]
    b_idle_bare = [float(b_idle["bare_f1_ghz"]), float(b_idle["bare_f2_ghz"])]
    report = {
        "rail_A": {
            "ec_mhz": selected["ec_a_mhz"],
            "ej_idle_and_gate_ghz": final["EJ_GHz"][:2],
            "bare_f01_idle_and_gate_ghz": a_bare,
            "readout_operating_frequency_ghz": anchor["readout_operating_frequency_GHz"],
            "readout_logical_frequency_mhz": 1000.0 * anchor["readout_logical_frequency_GHz"],
            "gate_model_logical_frequency_mhz": 1000.0 * gate_result["logical_splittings_GHz"][0],
            "intra_rail_coupling_ghz": final["couplings_GHz"]["A1_A2"],
            "readout_C12_fF": anchor["fixed_A"]["readout_C12_fF"],
            "readout_Cg_fF": anchor["fixed_A"]["readout_Cg_fF"],
            "complete_readout_witness": full_anchor["complete_readout_witness"],
        },
        "rail_B": {
            "ec_mhz": selected["ec_b_mhz"],
            "idle_ej_ghz": [b_idle["ej1_ghz"], b_idle["ej2_ghz"]],
            "gate_ej_ghz": final["EJ_GHz"][2:],
            "idle_bare_f01_ghz": b_idle_bare,
            "gate_bare_f01_ghz": b_gate_bare,
            "bare_frequency_shift_gate_minus_idle_mhz": [1000.0 * (b_gate_bare[i] - b_idle_bare[i]) for i in (0, 1)],
            "idle_readout_operating_frequency_ghz": b_idle["operating_frequency_ghz"],
            "idle_readout_logical_frequency_mhz": 1000.0 * b_idle["splitting_ghz"],
            "gate_common_frequency_parameter_ghz": final["fB_GHz"],
            "gate_model_logical_frequency_mhz": 1000.0 * gate_result["logical_splittings_GHz"][1],
            "idle_intra_rail_coupling_ghz": b_idle["J12_ghz"],
            "gate_intra_rail_coupling_ghz": final["couplings_GHz"]["B1_B2"],
            "idle_readout_device": b_idle["device"],
        },
        "interrail_detuning": {
            "idle": detuning(a_bare, b_idle_bare),
            "gate": detuning(a_bare, b_gate_bare),
        },
        "gate": {
            "logical_source": gate_result["logical_source"],
            "auxiliary_state": "|tilde{0020}>",
            "duration_ns": gate_time,
            "interrail_charge_coupling_ghz": final["couplings_GHz"]["A2_B1"],
            "intended_dressed_matrix_element_ghz": gate_result["g_GHz"],
            "S1": final["S1"],
            "S2": final["S2"],
            "archived": {k: final[k] for k in ("raw_fidelity", "postselected_fidelity", "acceptance", "phase_error_rad", "return_by_input", "leakage_by_input")},
            "reconstructed": reconstructed,
            "absolute_reconstruction_errors": {
                "raw_fidelity": abs(raw - final["raw_fidelity"]),
                "postselected_fidelity": abs(raw / acceptance - final["postselected_fidelity"]),
                "acceptance": abs(acceptance - final["acceptance"]),
            },
        },
        "selection": {
            "robustness_margin_mhz": selected["robustness_margin_mhz"],
            "maximum_absolute_rail_B_bare_frequency_shift_mhz": selected["maximum_absolute_bare_frequency_shift_mhz"],
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "complete_configuration.json").write_text(json.dumps(report, indent=2) + "\n")

    labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained", sharex=True)
    for i, label in enumerate(labels):
        axes[0, 0].plot(times, returns[i], label=label)
        axes[0, 1].plot(times, logical_survival[i], label=label)
    source_label = gate_result["logical_source"]
    axes[1, 0].plot(times, returns[gate_result["logical_source_index"]],
                    label=rf"Return to $|{source_label}\rangle_L$", lw=2)
    axes[1, 0].plot(times, source_auxiliary, label=r"Auxiliary $|\widetilde{0020}\rangle$", lw=2)
    axes[1, 0].plot(times, np.maximum(source_other, 0), label="Other nonlogical states", lw=1.5)
    axes[1, 1].plot(times, phase / np.pi, color="#6a1b9a", lw=2)
    for ax in axes.flat:
        ax.axvline(gate_time, color="black", ls="--", lw=1, label=f"Gate time {gate_time:.2f} ns")
        ax.grid(alpha=.22); ax.set_xlabel("Interaction time (ns)")
    axes[0, 0].set(title="Return probability for each logical input", ylabel="Probability", ylim=(-.02, 1.02))
    axes[0, 1].set(title="Total logical-subspace survival", ylabel="Probability", ylim=(-.02, 1.02))
    axes[1, 0].set(title=r"Selected $|10\rangle \leftrightarrow |\widetilde{0020}\rangle$ collision dynamics", ylabel="Probability", ylim=(-.02, 1.02))
    axes[1, 1].set(title="Accumulated conditional phase", ylabel="Conditional phase / π")
    axes[0, 0].legend(ncol=2, fontsize=8); axes[0, 1].legend(ncol=2, fontsize=8); axes[1, 0].legend(fontsize=8); axes[1, 1].legend(fontsize=8)
    fig.suptitle("A4 selected configuration: exact static-gate time evolution")
    fig.savefig(OUTPUT / "gate_time_evolution.png", dpi=220)
    fig.savefig(OUTPUT / "gate_time_evolution.pdf")
    plt.close(fig)

    np.savez_compressed(
        OUTPUT / "gate_time_evolution.npz", times_ns=times, return_probabilities=returns,
        logical_survival=logical_survival, selected_source_auxiliary_probability=source_auxiliary,
        selected_source_other_nonlogical_probability=source_other, conditional_phase_rad=phase,
    )
    print(json.dumps({"output": str(OUTPUT), "summary": report["gate"], "detuning": report["interrail_detuning"]}, indent=2))


if __name__ == "__main__":
    main()
