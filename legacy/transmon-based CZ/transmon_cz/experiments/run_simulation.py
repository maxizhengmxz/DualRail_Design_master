#!/usr/bin/env python3
"""Run one four-transmon dual-rail gate simulation from a JSON configuration."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..simulation import (
    LABELS,
    TWOPI,
    BasisSpec,
    cz_collision_splittings,
    EvolutionSpec,
    FourTransmonSystem,
    NoiseModel,
    TransmonSpec,
    odd_pi_index,
    simulate_logical_basis,
    simulate_physical_cz_full_system,
    transition_table,
)


def load_configuration(path: Path):
    raw = json.loads(path.read_text())
    if raw.get("noise") is not None:
        raise ValueError("This runner reports coherent CZ metrics; use low-level evolve_state for noise populations")
    specs = {label: TransmonSpec(**raw["transmons"][label]) for label in LABELS}
    basis = BasisSpec(**raw["basis"])
    evolution = EvolutionSpec(**raw["evolution"])
    noise = None if raw.get("noise") is None else NoiseModel.from_dict(raw["noise"])
    return raw, specs, basis, evolution, noise


def complex_matrix_to_json(matrix: np.ndarray) -> list:
    return [
        [{"real": float(value.real), "imag": float(value.imag)} for value in row]
        for row in matrix
    ]


def analyze_odd_pi_points(result: dict, options: dict) -> tuple[dict, dict]:
    """Optionally summarize requested odd-pi crossings for one gate run."""
    if not options.get("enabled", False):
        return {}, {}
    tolerance = float(options.get("tolerance_rad", 0.05))
    orders = [int(value) for value in options.get("orders", [1])]
    if result["solver"] != "sesolve":
        return {
            "enabled": True,
            "available": False,
            "reason": "Odd-pi logical-phase analysis requires coherent sesolve output",
        }, {}
    phases = result["conditional_phase_vs_time_rad"]
    report_points = {}
    indices = np.full(len(orders), -1, dtype=int)
    for position, order in enumerate(orders):
        selected = odd_pi_index(phases, order=order, tolerance_rad=tolerance)
        key = f"order_{order}_{2 * order - 1}pi"
        if selected is None:
            report_points[key] = {"reached": False}
            continue
        indices[position] = selected
        phase = float(np.unwrap(phases)[selected])
        phase_error = abs(
            np.angle(np.exp(1j * (phase - np.sign(phase) * (2 * order - 1) * np.pi)))
        )
        report_points[key] = {
            "reached": True,
            "sample_index": int(selected),
            "time_ns": float(result["times_ns"][selected]),
            "conditional_phase_rad": phase,
            "conditional_phase_error_rad": float(phase_error),
            "unconditional_cz_fidelity": float(
                result["average_cz_fidelity_vs_time"][selected]
            ),
            "conditional_cz_fidelity": float(
                result["conditional_cz_fidelity_vs_time"][selected]
            ),
            "acceptance_probability": float(
                result["acceptance_probability_vs_time"][selected]
            ),
            "wrong_logical_state_population": float(
                result["average_wrong_logical_population_vs_time"][selected]
            ),
            "erasure_detectable_leakage": float(
                result["average_erasure_detectable_vs_time"][selected]
            ),
            "erasure_undetectable_leakage": float(
                result["average_erasure_undetectable_vs_time"][selected]
            ),
            "total_nonlogical_leakage": float(
                result["average_nonlogical_leakage_vs_time"][selected]
            ),
        }
    return {
        "enabled": True,
        "available": True,
        "tolerance_rad": tolerance,
        "points": report_points,
    }, {
        "odd_pi_orders": np.asarray(orders, dtype=int),
        "odd_pi_sample_indices": indices,
        "odd_pi_times_ns": np.asarray(
            [result["times_ns"][index] if index >= 0 else np.nan for index in indices]
        ),
    }


def make_population_plot(result: dict, logical_labels: tuple[str, ...], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    keys = ("logical", "sector_11", "sector_20", "sector_02")
    names = (
        "Idle logical manifold",
        "One excitation in each dual rail",
        "Two in A, zero in B",
        "Zero in A, two in B",
    )
    for axis, key, name in zip(axes.flat, keys, names):
        for label, run in zip(logical_labels, result["runs"]):
            axis.plot(result["times_ns"], run[key], label=label)
        axis.set_title(name)
        axis.set_xlabel("Time (ns)")
        axis.set_ylabel("Population")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_level2_plot(result: dict, logical_labels: tuple[str, ...], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, transmon in zip(axes.flat, LABELS):
        for label, run in zip(logical_labels, result["runs"]):
            axis.plot(result["times_ns"], run[f"{transmon}_level2"], label=label)
        axis.set_title(f"{transmon}: population in local level 2")
        axis.set_xlabel("Time (ns)")
        axis.set_ylabel("Population")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_physical_cz_plot(result: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    times = result["times_ns"]
    axes[0].plot(times, result["population_11"], label="|11>")
    axes[0].plot(times, result["population_20"], label="|20>")
    axes[0].plot(times, result["population_02"], label="|02>")
    axes[0].plot(times, result["population_A1_excited"], ls="--", label="A1 excited")
    axes[0].plot(times, result["population_B2_excited"], ls="--", label="B2 excited")
    axes[0].axvline(result["best_time_ns"], color="black", ls="--", alpha=0.5)
    axes[0].set_ylabel("Population")
    axes[0].set_title("A2-B1 physical collision from |11>")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(times, result["average_cz_fidelity"], label="Reduced-channel CZ fidelity")
    axes[1].plot(times, 1.0 - result["leakage"], label="A2-B1 computational survival")
    axes[1].axvline(result["best_time_ns"], color="black", ls="--", alpha=0.5)
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Probability / fidelity")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_gate_time_comparison_plot(physical: dict, logical: dict, output: Path) -> None:
    times = physical["times_ns"]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(times, physical["average_cz_fidelity"], label="Reduced A2-B1 channel")
    axis.plot(times, logical["average_cz_fidelity_vs_time"], label="Full logical map")
    axis.axvline(physical["best_time_ns"], color="C0", ls="--", alpha=0.6)
    axis.axvline(logical["best_time_ns"], color="C1", ls="--", alpha=0.6)
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Average CZ fidelity")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_reduced_population_plot(result: dict, output: Path) -> None:
    """Plot reduced A2-B1 output populations for every basis input."""
    labels = ("00", "01", "10", "11")
    populations = result["reduced_population_transfer"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for input_index, axis in enumerate(axes.flat):
        for output_index, output_label in enumerate(labels):
            axis.plot(
                result["times_ns"],
                populations[:, output_index, input_index],
                label=f"|{output_label}> output",
            )
        survival = np.sum(populations[:, :, input_index], axis=1)
        axis.plot(
            result["times_ns"], 1.0 - survival, "k--", label="A2-B1 leakage"
        )
        axis.axvline(result["best_time_ns"], color="gray", ls=":", alpha=0.7)
        axis.set_title(f"Reduced A2-B1 input |{labels[input_index]}>")
        axis.set_xlabel("Time (ns)")
        axis.set_ylabel("Population")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_logical_population_transfer_plot(result: dict, output: Path) -> None:
    """Plot logical output populations for every logical basis input."""
    labels = ("00L", "01L", "10L", "11L")
    populations = np.abs(result["logical_maps"]) ** 2
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for input_index, axis in enumerate(axes.flat):
        for output_index, output_label in enumerate(labels):
            axis.plot(
                result["times_ns"],
                populations[:, output_index, input_index],
                label=f"|{output_label}> output",
            )
        survival = np.sum(populations[:, :, input_index], axis=1)
        axis.plot(
            result["times_ns"],
            1.0 - survival,
            "k--",
            label="Nonlogical leakage",
        )
        axis.axvline(result["best_time_ns"], color="gray", ls=":", alpha=0.7)
        axis.set_title(f"Logical input |{labels[input_index]}>")
        axis.set_xlabel("Time (ns)")
        axis.set_ylabel("Population")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_logical_outcome_histograms(
    result: dict,
    output_directory: Path,
    time_index: int,
    gate_label: str,
) -> None:
    """Create one six-outcome population histogram per logical input."""
    labels = ("00L", "01L", "10L", "11L")
    populations = result["logical_populations_vs_time"][time_index]
    detectable = result["erasure_detectable_leakage_vs_time"][time_index]
    undetectable = result["erasure_undetectable_leakage_vs_time"][time_index]
    time_ns = result["times_ns"][time_index]
    output_labels = (
        "$|00\\rangle_L$",
        "$|01\\rangle_L$",
        "$|10\\rangle_L$",
        "$|11\\rangle_L$",
        "Erasure-\ndetectable",
        "Undetectable\nnonlogical",
    )
    colors = ("C0", "C1", "C2", "C3", "C4", "C5")
    for input_index, input_label in enumerate(labels):
        values = np.concatenate(
            (
                populations[:, input_index],
                [detectable[input_index], undetectable[input_index]],
            )
        )
        figure, axis = plt.subplots(figsize=(10, 6))
        bars = axis.bar(np.arange(6), values, color=colors)
        axis.set_xticks(np.arange(6), output_labels)
        axis.set_ylabel("Population")
        axis.set_ylim(0.0, 1.05)
        axis.set_title(
            f"Input $|{input_label}\\rangle$: outcomes at "
            f"{time_ns:.3f} ns ({gate_label.replace('_', ' ')})"
        )
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.015, 1.025),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        figure.tight_layout()
        figure.savefig(
            output_directory
            / f"logical_input_{input_label}_{gate_label}_outcomes.png",
            dpi=180,
        )
        plt.close(figure)


def make_logical_gate_metrics_plot(result: dict, output: Path) -> None:
    """Plot phase, raw/conditional fidelity, and leakage classification."""
    times = result["times_ns"]
    unconditional_time = result["best_time_ns"]
    conditional_time = result["best_conditional_time_ns"]
    figure, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(
        times,
        result["conditional_phase_vs_time_rad"],
        color="C4",
        label="Residual conditional phase",
    )
    axes[0].axhline(np.pi, color="black", ls=":", label="$\\pi$")
    axes[0].axhline(-np.pi, color="black", ls=":")
    axes[0].set_ylabel("Phase (rad)")
    axes[0].legend()

    axes[1].plot(
        times,
        result["average_cz_fidelity_vs_time"],
        label="Unconditional CZ fidelity",
    )
    axes[1].plot(
        times,
        result["conditional_cz_fidelity_vs_time"],
        label="Conditional CZ fidelity",
    )
    axes[1].set_ylabel("Average fidelity")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend()

    axes[2].plot(
        times,
        result["average_erasure_detectable_vs_time"],
        label="Erasure-detectable leakage",
    )
    axes[2].plot(
        times,
        result["average_erasure_undetectable_vs_time"],
        label="Undetectable nonlogical leakage",
    )
    axes[2].set_xlabel("Gate duration (ns)")
    axes[2].set_ylabel("Average probability")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend()

    for axis in axes:
        axis.axvline(
            unconditional_time,
            color="C0",
            ls="--",
            alpha=0.75,
            label="Best unconditional" if axis is axes[0] else None,
        )
        axis.axvline(
            conditional_time,
            color="C1",
            ls="-.",
            alpha=0.75,
            label="Best conditional" if axis is axes[0] else None,
        )
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2)
    figure.suptitle("Logical CZ metrics versus gate duration")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_leakage_state_histograms(
    result: dict, output: Path, top_state_count: int = 8
) -> None:
    """Resolve the nonlogical component into local product-level states."""
    logical_labels = ("00L", "01L", "10L", "11L")
    levels = result["leakage_product_level_tuples"]
    distributions = result["leakage_product_populations_at_best"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    for input_index, axis in enumerate(axes.flat):
        distribution = distributions[input_index]
        selected = np.argsort(distribution)[-top_state_count:]
        selected = selected[np.argsort(distribution[selected])]
        values = distribution[selected]
        state_labels = [
            "|" + "".join(str(level) for level in levels[index]) + ">"
            for index in selected
        ]
        omitted = max(float(np.sum(distribution) - np.sum(values)), 0.0)
        if omitted > 1e-10:
            values = np.concatenate(([omitted], values))
            state_labels = ["other"] + state_labels
        axis.barh(state_labels, values)
        axis.set_xlabel("Absolute leaked-state population")
        axis.set_title(
            f"Input |{logical_labels[input_index]}>; total={np.sum(distribution):.3f}"
        )
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle(
        f"Nonlogical leakage destinations at {result['best_time_ns']:.3f} ns",
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_uncoupled_four_qubit_spectrum(
    system: FourTransmonSystem,
    output: Path,
    collision_window_GHz: float = 0.150,
) -> dict:
    """Plot the spectrum with A2--B1 removed and flag multiphoton collisions."""
    h0 = system.hamiltonian() - TWOPI * system.couplings_GHz["A2_B1"] * (
        system.n_full["A2"] * system.n_full["B1"]
    )
    energies_rad, eigenvectors = np.linalg.eigh(h0.full())
    energies_GHz = (energies_rad - energies_rad[0]) / TWOPI
    level_tuples = list(product(range(system.dim), repeat=4))
    bare_vectors = np.column_stack(
        [system.local_product_ket(levels).full().ravel() for levels in level_tuples]
    )
    overlaps = np.abs(bare_vectors.conj().T @ eigenvectors) ** 2
    dominant_indices = np.argmax(overlaps, axis=0)
    dominant_levels = [level_tuples[index] for index in dominant_indices]
    dominant_weights = overlaps[dominant_indices, np.arange(eigenvectors.shape[1])]
    excitation_groups = np.asarray([sum(levels) for levels in dominant_levels])

    one_mask = excitation_groups == 1
    one_frequencies = energies_GHz[one_mask]
    frequency_low = float(np.min(one_frequencies) - collision_window_GHz)
    frequency_high = float(np.max(one_frequencies) + collision_window_GHz)
    photon_frequencies = np.divide(
        energies_GHz,
        excitation_groups,
        out=np.full_like(energies_GHz, np.nan),
        where=excitation_groups > 0,
    )
    selected = (excitation_groups <= 2) | (
        (excitation_groups > 2)
        & (photon_frequencies >= frequency_low)
        & (photon_frequencies <= frequency_high)
    )
    selected_indices = np.flatnonzero(selected)
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    for index in selected_indices:
        excitation = int(excitation_groups[index])
        label = "|" + "".join(map(str, dominant_levels[index])) + ">"
        color = colors(excitation % 10)
        axes[0].plot(
            [excitation - 0.28, excitation + 0.28],
            [energies_GHz[index], energies_GHz[index]],
            color=color,
            lw=2,
        )
        axes[0].annotate(
            label,
            (excitation + 0.31, energies_GHz[index]),
            fontsize=6,
            va="center",
        )
        if excitation > 0:
            axes[1].scatter(
                excitation,
                photon_frequencies[index],
                color=color,
                s=30,
                zorder=3,
            )
            axes[1].annotate(
                f"{label}/{excitation}",
                (excitation + 0.08, photon_frequencies[index]),
                fontsize=6,
                va="center",
            )
    axes[0].set_xlabel("Dominant total excitation group N")
    axes[0].set_ylabel(r"$(E_k-E_0)/h$ (GHz)")
    axes[0].set_title("Four-transmon spectrum without A2--B1 interaction")
    axes[1].axhspan(
        np.min(one_frequencies), np.max(one_frequencies), color="C1", alpha=0.12,
        label="N=1 frequency band",
    )
    axes[1].set_xlabel("Dominant total excitation group N")
    axes[1].set_ylabel(r"$(E_k-E_0)/(Nh)$ (GHz per photon)")
    axes[1].set_title("Ground-state multiphoton frequencies near N=1 band")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xticks(sorted(set(excitation_groups[selected_indices].tolist())))
    figure.savefig(output, dpi=180)
    plt.close(figure)

    collisions = []
    one_indices = np.flatnonzero(one_mask)
    for higher_index in np.flatnonzero(excitation_groups >= 2):
        for one_index in one_indices:
            detuning = photon_frequencies[higher_index] - energies_GHz[one_index]
            if abs(detuning) <= collision_window_GHz:
                collisions.append(
                    {
                        "one_photon_state": "|" + "".join(
                            map(str, dominant_levels[one_index])
                        ) + ">",
                        "one_photon_frequency_GHz": float(energies_GHz[one_index]),
                        "higher_state": "|" + "".join(
                            map(str, dominant_levels[higher_index])
                        ) + ">",
                        "photon_order": int(excitation_groups[higher_index]),
                        "equivalent_frequency_GHz": float(
                            photon_frequencies[higher_index]
                        ),
                        "detuning_MHz": float(1000.0 * detuning),
                        "dominant_state_weight": float(
                            dominant_weights[higher_index]
                        ),
                    }
                )
    collisions.sort(key=lambda item: abs(item["detuning_MHz"]))
    return {
        "A2_B1_interaction_GHz": 0.0,
        "collision_window_MHz": 1000.0 * collision_window_GHz,
        "state_label_convention": "dominant |A1 A2 B1 B2> local energy levels",
        "multiphoton_frequency_definition": "(E_k-E_0)/(N h)",
        "near_multiphoton_collisions": collisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "example_config.json"),
    )
    arguments = parser.parse_args()

    config_path = Path(arguments.config).resolve()
    raw, specs, basis, evolution, noise = load_configuration(config_path)
    output = Path(raw.get("output_directory", "../output/simulation_output"))
    if not output.is_absolute():
        output = config_path.parent / output
    output.mkdir(parents=True, exist_ok=True)

    system = FourTransmonSystem(specs, raw["couplings_GHz"], basis, noise_model=noise)

    spectrum_options = raw.get("uncoupled_spectrum", {})
    spectrum_diagnostics = make_uncoupled_four_qubit_spectrum(
        system,
        output / "four_qubit_spectrum_no_A2_B1.png",
        collision_window_GHz=float(
            spectrum_options.get("collision_window_GHz", 0.150)
        ),
    )

    physical_cz = simulate_physical_cz_full_system(system, evolution)
    collision_splittings = cz_collision_splittings(system)
    logical = system.logical_basis()
    result = simulate_logical_basis(system, logical, evolution)
    odd_pi_report, odd_pi_arrays = analyze_odd_pi_points(
        result, raw.get("odd_pi_analysis", {})
    )

    transitions = transition_table(system)
    report = {
        "isolated_pair_diagnostics": {k: result[k] for k in ("H_A", "H_B", "logical_frequency_A_GHz", "logical_frequency_B_GHz", "logical_detuning_GHz")},
        "transition_table": transitions,
        "four_qubit_spectrum_without_A2_B1": spectrum_diagnostics,
        "physical_A2_B1_CZ": {
            "frequencies": physical_cz["frequencies"],
            "static_collision_splittings": {
                name: {
                    **{
                        key: value
                        for key, value in data.items()
                        if key not in {
                            "direct_coupling_GHz",
                            "dressed_eigenenergies_GHz",
                            "dressed_subspace_weights",
                            "dressed_component_weights",
                        }
                    },
                    "direct_coupling_GHz": {
                        "real": float(data["direct_coupling_GHz"].real),
                        "imag": float(data["direct_coupling_GHz"].imag),
                    },
                    "dressed_eigenenergies_GHz": data[
                        "dressed_eigenenergies_GHz"
                    ].tolist(),
                    "dressed_subspace_weights": data[
                        "dressed_subspace_weights"
                    ].tolist(),
                    "dressed_component_weights": data[
                        "dressed_component_weights"
                    ].tolist(),
                }
                for name, data in collision_splittings.items()
            },
            "best_time_ns": physical_cz["best_time_ns"],
            "best_average_cz_fidelity": physical_cz["best_average_cz_fidelity"],
            "best_leakage": physical_cz["best_leakage"],
            "best_conditional_phase_rad": physical_cz[
                "best_conditional_phase_rad"
            ],
            "best_projected_map": complex_matrix_to_json(
                physical_cz["best_projected_map"]
            ),
        },
        "full_dual_rail_logical_simulation": {
            "solver": result["solver"],
            "collapse_operator_count": result["collapse_operator_count"],
            "survival": result["survival"],
            "leakage": result["leakage"],
        },
    }
    logical_report = report["full_dual_rail_logical_simulation"]
    if odd_pi_report:
        logical_report["odd_pi_analysis"] = odd_pi_report
    if result["solver"] == "sesolve":
        logical_report.update(
            {
                "singular_values": result["singular_values"].tolist(),
                "conditional_phase_rad_if_nearly_diagonal": result["conditional_phase_rad"],
                "best_time_ns": result["best_time_ns"],
                "best_average_cz_fidelity": result["best_average_cz_fidelity"],
                "acceptance_at_best_unconditional": result[
                    "acceptance_at_best_unconditional"
                ],
                "best_conditional_time_ns": result["best_conditional_time_ns"],
                "best_conditional_cz_fidelity": result[
                    "best_conditional_cz_fidelity"
                ],
                "acceptance_at_best_conditional": result[
                    "acceptance_at_best_conditional"
                ],
                "population_errors_at_best_unconditional": {
                    "wrong_logical_state_population": result[
                        "wrong_logical_population_at_best_unconditional"
                    ],
                    "erasure_detectable_leakage": result[
                        "erasure_detectable_leakage_at_best_unconditional"
                    ],
                    "erasure_undetectable_leakage": result[
                        "erasure_undetectable_leakage_at_best_unconditional"
                    ],
                    "total_nonlogical_leakage": result[
                        "nonlogical_leakage_at_best_unconditional"
                    ],
                },
                "population_errors_at_best_conditional": {
                    "wrong_logical_state_population": result[
                        "wrong_logical_population_at_best_conditional"
                    ],
                    "erasure_detectable_leakage": result[
                        "erasure_detectable_leakage_at_best_conditional"
                    ],
                    "erasure_undetectable_leakage": result[
                        "erasure_undetectable_leakage_at_best_conditional"
                    ],
                    "total_nonlogical_leakage": result[
                        "nonlogical_leakage_at_best_conditional"
                    ],
                },
                "best_time_difference_from_reduced_channel_ns": float(
                    result["best_time_ns"] - physical_cz["best_time_ns"]
                ),
                "logical_map": complex_matrix_to_json(result["logical_map"]),
                "closest_unitary": complex_matrix_to_json(result["closest_unitary"]),
            }
        )
    else:
        logical_report["logical_population_transfer"] = result[
            "logical_population_transfer"
        ].tolist()
    (output / "report.json").write_text(json.dumps(report, indent=2))

    arrays = {"times_ns": result["times_ns"], **odd_pi_arrays}
    arrays.update(
        {
            "physical_cz_projected_maps": physical_cz["projected_maps"],
            "physical_cz_leakage": physical_cz["leakage"],
            "physical_cz_conditional_phase_rad": physical_cz[
                "conditional_phase_rad"
            ],
            "physical_cz_average_fidelity": physical_cz["average_cz_fidelity"],
            "physical_cz_population_11": physical_cz["population_11"],
            "physical_cz_population_20": physical_cz["population_20"],
            "physical_cz_population_02": physical_cz["population_02"],
            "physical_cz_population_A1_excited": physical_cz[
                "population_A1_excited"
            ],
            "physical_cz_population_B2_excited": physical_cz[
                "population_B2_excited"
            ],
            "physical_cz_basis_input_reduced_purity": physical_cz[
                "basis_input_reduced_purity"
            ],
            "physical_cz_reduced_population_transfer": physical_cz[
                "reduced_population_transfer"
            ],
            "physical_cz_plus_input_fidelity": physical_cz[
                "plus_input_cz_fidelity"
            ],
        }
    )
    for input_label, run in zip(logical.labels, result["runs"]):
        for observable, values in run.items():
            arrays[f"{input_label}_{observable}"] = values
    if result["solver"] == "sesolve":
        arrays["logical_map"] = result["logical_map"]
        arrays["logical_maps"] = result["logical_maps"]
        arrays["logical_populations_at_best"] = result[
            "logical_populations_at_best"
        ]
        arrays["logical_leakage_product_level_tuples"] = result[
            "leakage_product_level_tuples"
        ]
        arrays["logical_leakage_product_populations_at_best"] = result[
            "leakage_product_populations_at_best"
        ]
        arrays["logical_erasure_detectable_leakage_at_best"] = result[
            "erasure_detectable_leakage_at_best"
        ]
        arrays["logical_erasure_undetectable_leakage_at_best"] = result[
            "erasure_undetectable_leakage_at_best"
        ]
        arrays["closest_unitary"] = result["closest_unitary"]
        arrays["logical_average_cz_fidelity"] = result[
            "average_cz_fidelity_vs_time"
        ]
        arrays["logical_survival_vs_time"] = result["logical_survival_vs_time"]
        arrays["logical_correct_population_by_input_vs_time"] = result[
            "correct_logical_population_by_input_vs_time"
        ]
        arrays["logical_wrong_population_by_input_vs_time"] = result[
            "wrong_logical_population_by_input_vs_time"
        ]
        arrays["logical_nonlogical_leakage_by_input_vs_time"] = result[
            "nonlogical_leakage_by_input_vs_time"
        ]
        arrays["logical_average_correct_population_vs_time"] = result[
            "average_correct_logical_population_vs_time"
        ]
        arrays["logical_average_wrong_population_vs_time"] = result[
            "average_wrong_logical_population_vs_time"
        ]
        arrays["logical_average_nonlogical_leakage_vs_time"] = result[
            "average_nonlogical_leakage_vs_time"
        ]
        arrays["logical_conditional_phase_rad"] = result[
            "conditional_phase_vs_time_rad"
        ]
        arrays["logical_conditional_cz_fidelity"] = result[
            "conditional_cz_fidelity_vs_time"
        ]
        arrays["logical_acceptance_probability"] = result[
            "acceptance_probability_vs_time"
        ]
        arrays["logical_erasure_detectable_leakage_vs_time"] = result[
            "erasure_detectable_leakage_vs_time"
        ]
        arrays["logical_erasure_undetectable_leakage_vs_time"] = result[
            "erasure_undetectable_leakage_vs_time"
        ]
        arrays["logical_average_erasure_detectable_vs_time"] = result[
            "average_erasure_detectable_vs_time"
        ]
        arrays["logical_average_erasure_undetectable_vs_time"] = result[
            "average_erasure_undetectable_vs_time"
        ]
    else:
        arrays["logical_population_transfer"] = result["logical_population_transfer"]
    np.savez(output / "simulation_data.npz", **arrays)

    make_population_plot(result, logical.labels, output / "populations.png")
    make_level2_plot(result, logical.labels, output / "level2_populations.png")
    make_physical_cz_plot(physical_cz, output / "physical_A2_B1_CZ.png")
    make_reduced_population_plot(
        physical_cz, output / "reduced_A2_B1_populations.png"
    )
    if result["solver"] == "sesolve":
        make_logical_population_transfer_plot(
            result, output / "logical_gate_populations.png"
        )
        make_logical_gate_metrics_plot(
            result, output / "logical_gate_metrics_vs_duration.png"
        )
        make_logical_outcome_histograms(
            result,
            output,
            result["best_index"],
            "best_unconditional",
        )
        make_logical_outcome_histograms(
            result,
            output,
            result["best_conditional_index"],
            "best_conditional",
        )
        make_leakage_state_histograms(
            result, output / "logical_leakage_state_histograms.png"
        )
        make_gate_time_comparison_plot(
            physical_cz, result, output / "gate_time_comparison.png"
        )

    print(json.dumps(report, indent=2))
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
