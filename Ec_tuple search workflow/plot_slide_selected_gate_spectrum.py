#!/usr/bin/env python3
"""Plot the promoted tuple's gate spectrum at its selected rail-A anchor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "2Q-gate_XM"), str(ROOT)]

from xm_cz.model import fixed_pair_model, pair_model


RECORD = (
    HERE
    / "corrected_cavity_full_neighborhood_A158_240_B158_226_2026-11-17"
    / "candidate_analysis"
    / "complete_corrected_configuration.json"
)
OUTPUT = ROOT / "output/ec_tuple_selection_slides/selected_gate_spectrum"


def assigned_state(model: dict, bare_index: int) -> int:
    return int(np.argmax(np.abs(model["vectors"][bare_index]) ** 2))


def main() -> None:
    record = json.loads(RECORD.read_text())
    ec_mhz = [float(value) for value in record["ec_tuple_mhz"]]
    ec = [value / 1000.0 for value in ec_mhz]
    anchor = record["selected_anchor"]
    fixed = anchor["fixed_A"]
    channel = record["channel_selection"]
    gate = record["gate_result"]

    rail_a = fixed_pair_model(ec[0], ec[1], *fixed["EJ_GHz"], fixed["J_GHz"], 8)
    target_b_splitting = float(gate["logical_splittings_GHz"][1])
    interrail_coupling = float(gate["couplings_GHz"]["A2_B1"])
    interaction_frequency = float(channel["gate_frequency_ghz"])
    calibrated_frequency = float(gate["fB_GHz"])

    frequencies = np.linspace(4.10, 5.35, 801)
    labels = ["00", "01", "10", "11", "B20", "B02", "B11", "A02", "A20", "A11"]
    traces = {label: [] for label in labels}
    assigned_eigenstates = {label: [] for label in labels}

    a_modes = rail_a["modes"]
    a_states = {
        "A02": assigned_state(rail_a, 2),
        "A20": assigned_state(rail_a, 8),
        "A11": assigned_state(rail_a, 5),
    }
    for frequency in frequencies:
        rail_b = pair_model(ec[2], ec[3], float(frequency), target_b_splitting)
        b_modes = rail_b["modes"]
        logical_indices = {
            f"{ia}{ib}": a_modes[ia] * 16 + b_modes[ib]
            for ia in (0, 1) for ib in (0, 1)
        }
        b_states = {
            "B20": assigned_state(rail_b, 8),
            "B02": assigned_state(rail_b, 2),
            "B11": assigned_state(rail_b, 5),
        }
        basis_indices = {
            **logical_indices,
            **{label: state for label, state in b_states.items()},
            **{label: state * 16 for label, state in a_states.items()},
        }
        uncoupled = (
            rail_a["energies"][:, None] + rail_b["energies"][None, :]
        ).ravel()
        unit = np.kron(rail_a["right_charge"], rail_b["left_charge"])
        dressed_energies, dressed_vectors = np.linalg.eigh(
            np.diag(uncoupled) + interrail_coupling * unit
        )
        reference = float(np.mean([uncoupled[index] for index in logical_indices.values()]))
        overlap = np.vstack(
            [np.abs(dressed_vectors[index, :]) ** 2 for index in basis_indices.values()]
        )
        rows, columns = linear_sum_assignment(-overlap)
        assignment = {labels[row]: int(column) for row, column in zip(rows, columns)}
        for label in labels:
            traces[label].append(1000.0 * (dressed_energies[assignment[label]] - reference))
            assigned_eigenstates[label].append(assignment[label])

    colors = {
        "00": "#159D82",
        "01": "#F2A51A",
        "10": "#2E86D1",
        "11": "#6657C8",
        "B20": "#F23B4D",
        "B02": "#CF52C9",
        "B11": "#718096",
        "A02": "#F06CB5",
        "A20": "#9A6A50",
        "A11": "#18A9B9",
    }

    interactions = []
    channel_specs = (("0020", 8, "B20"), ("0002", 2, "B02"))
    for resonance_channel, target_bare_index, auxiliary_label in channel_specs:
        for logical_source in ("00", "01", "10", "11"):
            source_a, source_b = map(int, logical_source)

            def auxiliary_residual(value: float) -> float:
                trial_b = pair_model(
                    ec[2], ec[3], float(value), target_b_splitting, target_bare_index
                )
                source_energy = (
                    rail_a["energies"][rail_a["modes"][source_a]]
                    + trial_b["energies"][trial_b["modes"][source_b]]
                )
                return float(source_energy - trial_b["energies"][trial_b["target"]])

            sampled = np.linspace(4.05, 5.40, 271)
            residuals = [auxiliary_residual(float(value)) for value in sampled]
            root = None
            for left, right, y_left, y_right in zip(
                sampled[:-1], sampled[1:], residuals[:-1], residuals[1:]
            ):
                if y_left * y_right <= 0:
                    root = brentq(auxiliary_residual, left, right)
                    break
            if root is None:
                continue
            root_b = pair_model(
                ec[2], ec[3], float(root), target_b_splitting, target_bare_index
            )
            root_logical = {
                f"{ia}{ib}": rail_a["energies"][rail_a["modes"][ia]]
                + root_b["energies"][root_b["modes"][ib]]
                for ia in (0, 1) for ib in (0, 1)
            }
            root_reference = float(np.mean(list(root_logical.values())))
            interaction_energy = 1000.0 * (
                root_logical[logical_source] - root_reference
            )
            interactions.append(
                {
                    "channel": resonance_channel,
                    "auxiliary": auxiliary_label,
                    "logical_source": logical_source,
                    "frequency_ghz": float(root),
                    "energy_mhz": float(interaction_energy),
                }
            )

    fig, axis = plt.subplots(figsize=(10.5, 7.5))
    for label in labels:
        trace = np.asarray(traces[label])
        # The labels follow reference-state character, so their identities can
        # exchange abruptly at an avoided crossing.  Break the drawn curve at
        # those swaps instead of connecting unrelated dressed eigenvalues.
        steps = np.abs(np.diff(trace))
        typical_step = float(np.median(steps))
        identity_swaps = np.diff(np.asarray(assigned_eigenstates[label])) != 0
        jump_indices = np.flatnonzero(
            identity_swaps | (steps > max(5.0, 8.0 * typical_step))
        )
        plotted_trace = trace.copy()
        plotted_trace[jump_indices] = np.nan
        plotted_trace[jump_indices + 1] = np.nan
        axis.plot(frequencies, plotted_trace, color=colors[label], lw=1.65)
        visible = np.flatnonzero((trace > -292.0) & (trace < 292.0))
        if label in ("00", "01", "10", "11"):
            label_index = int(np.argmin(np.abs(frequencies - 5.075)))
            label_x = frequencies[label_index] - 0.008
            label_ha = "right"
        else:
            label_index = int(visible[-1]) if len(visible) else len(frequencies) - 1
            label_x = frequencies[label_index] + 0.006
            label_ha = "left"
        axis.text(
            label_x,
            trace[label_index],
            label,
            color=colors[label],
            fontsize=9,
            ha=label_ha,
            va="center",
            fontweight="bold",
            clip_on=True,
        )

    interaction_styles = {
        "0020": ("o", "#D1495B", r"$|ij\rangle-|\widetilde{0020}\rangle$"),
        "0002": ("s", "#8E5AA9", r"$|ij\rangle-|\widetilde{0002}\rangle$"),
    }
    for item in interactions:
        marker, color, _ = interaction_styles[item["channel"]]
        axis.scatter(
            item["frequency_ghz"],
            item["energy_mhz"],
            marker=marker,
            s=36,
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            zorder=6,
        )
        axis.annotate(
            item["logical_source"],
            (item["frequency_ghz"], item["energy_mhz"]),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
        )

    idle_witness = record["rail_b_idle_readout_witness"]
    idle_frequency = min(
        float(idle_witness["bare_f1_ghz"]), float(idle_witness["bare_f2_ghz"])
    )
    highest_interaction = max(item["frequency_ghz"] for item in interactions)
    clearance_mhz = 1000.0 * (idle_frequency - highest_interaction)
    axis.axvspan(highest_interaction, idle_frequency, color="#BFE3D0", alpha=0.22, zorder=0)
    axis.axvline(idle_frequency, color="#16865C", lw=2.0, ls="-.")
    axis.annotate(
        "",
        xy=(idle_frequency, -255),
        xytext=(highest_interaction, -255),
        arrowprops=dict(arrowstyle="<->", color="#16865C", lw=2.0),
    )
    axis.text(
        0.5 * (idle_frequency + highest_interaction),
        -245,
        rf"idle clearance = {clearance_mhz:.0f} MHz $>200$ MHz",
        ha="center",
        va="bottom",
        color="#116B4A",
        fontsize=10.5,
        fontweight="semibold",
    )
    axis.text(
        idle_frequency - 0.006,
        270,
        r"min rail-B idle bare $f_{01}$",
        rotation=90,
        ha="right",
        va="top",
        color="#116B4A",
        fontsize=9.5,
    )

    source = str(channel["source_logical"])
    # Evaluate the selected two-state avoided crossing exactly at the screened
    # collision frequency; interpolating overlap-labelled traces can
    # underestimate the gap because their labels exchange across resonance.
    selected_b = pair_model(ec[2], ec[3], interaction_frequency, target_b_splitting)
    selected_logical_indices = {
        f"{ia}{ib}": a_modes[ia] * 16 + selected_b["modes"][ib]
        for ia in (0, 1) for ib in (0, 1)
    }
    source_index = selected_logical_indices[source]
    target_index = assigned_state(selected_b, 8)
    selected_uncoupled = (
        rail_a["energies"][:, None] + selected_b["energies"][None, :]
    ).ravel()
    selected_unit = np.kron(rail_a["right_charge"], selected_b["left_charge"])
    selected_energies, selected_vectors = np.linalg.eigh(
        np.diag(selected_uncoupled) + interrail_coupling * selected_unit
    )
    pair_weight = (
        np.abs(selected_vectors[source_index, :]) ** 2
        + np.abs(selected_vectors[target_index, :]) ** 2
    )
    pair_states = np.argsort(pair_weight)[-2:]
    pair_states = pair_states[np.argsort(selected_energies[pair_states])]
    selected_reference = float(
        np.mean([selected_uncoupled[index] for index in selected_logical_indices.values()])
    )
    selected_y, auxiliary_y = 1000.0 * (
        selected_energies[pair_states] - selected_reference
    )
    marker_y = 0.5 * (selected_y + auxiliary_y)
    axis.axvline(interaction_frequency, color="#D1495B", lw=1.4, ls="--", alpha=0.75)
    axis.annotate(
        rf"$|{source}\rangle\;-\;|\widetilde{{0020}}\rangle$",
        xy=(interaction_frequency, marker_y),
        xycoords="data",
        xytext=(5.03, 67.0),
        textcoords="data",
        color="#B02A3C",
        fontsize=12,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            color="#D1495B",
            lw=1.5,
            connectionstyle="arc3,rad=0.12",
        ),
        clip_on=False,
    )

    axis.set(
        title=(
            "Coupled gate-system spectrum and signed idle clearance\n"
            rf"$E_C/h=({ec_mhz[0]:.0f},{ec_mhz[1]:.0f},{ec_mhz[2]:.0f},{ec_mhz[3]:.0f})$ MHz"
            rf"   |   $f_A={anchor['readout_operating_frequency_GHz']:.3f}$ GHz"
        ),
        xlabel="Common rail-B operating frequency $f_B$ (GHz)",
        ylabel="Energy minus mean logical energy (MHz)",
        xlim=(frequencies[0], frequencies[-1]),
        ylim=(-310, 310),
    )
    axis.tick_params(labelsize=10.5)
    axis.title.set_fontsize(16)
    axis.title.set_fontweight("semibold")
    for spine in axis.spines.values():
        spine.set_color("#374151")
    axis.grid(False)
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.12, top=0.84)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = "selected_tuple_coupled_gate_spectrum_with_idle_clearance_A158_240_B158_226"
    fig.savefig(OUTPUT / f"{stem}.png", dpi=240, facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)

    provenance = {
        "schema": "selected-gate-spectrum-slide-v1",
        "ec_tuple_mhz": ec_mhz,
        "selected_anchor_id": anchor["anchor_id"],
        "selected_anchor_operating_frequency_ghz": anchor["readout_operating_frequency_GHz"],
        "selected_logical_source": source,
        "selected_auxiliary": channel["auxiliary"],
        "screened_interaction_frequency_ghz": interaction_frequency,
        "calibrated_gate_frequency_ghz": calibrated_frequency,
        "rail_b_target_logical_splitting_mhz": 1000.0 * target_b_splitting,
        "interrail_hardware_coupling_mhz": 1000.0 * interrail_coupling,
        "rail_b_minimum_idle_bare_f01_ghz": idle_frequency,
        "highest_interaction_frequency_ghz": highest_interaction,
        "idle_above_all_eight_interactions_clearance_mhz": clearance_mhz,
        "interaction_points": interactions,
        "spectrum_definition": "exact 256-state spectrum with selected A2-B1 hardware coupling on; displayed branches assigned one-to-one by reference-state overlap",
        "source_record": str(RECORD.resolve()),
        "figure_png": str((OUTPUT / f"{stem}.png").resolve()),
        "figure_pdf": str((OUTPUT / f"{stem}.pdf").resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
