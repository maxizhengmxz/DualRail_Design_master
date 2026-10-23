#!/usr/bin/env python3
"""Plot near-degenerate Delta-N=0 state pairs through three coupling stages."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..simulation import BasisSpec, FourTransmonSystem, LABELS, TransmonSpec, TWOPI
from .plot_transition_frequencies import CATEGORY_COLORS, classify_transition, state_text


def projected_hamiltonian(
    raw: dict, couplings: dict[str, float], states: list[tuple[int, ...]]
) -> tuple[np.ndarray, np.ndarray]:
    max_excitation = max(map(sum, states))
    specs = {
        label: TransmonSpec(**{**raw["transmons"][label], "ng": 0.0})
        for label in LABELS
    }
    system = FourTransmonSystem(
        specs,
        couplings,
        BasisSpec(
            ncut=raw["basis"]["ncut"],
            levels_per_transmon=max_excitation + 1,
        ),
    )
    basis_vectors = np.column_stack(
        [system.local_product_ket(levels).full().ravel() for levels in states]
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        matrix = (
            basis_vectors.conj().T
            @ (system.hamiltonian().full() / TWOPI)
            @ basis_vectors
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("Non-finite projected Hamiltonian")
    return matrix, basis_vectors


def track_eigenstates(
    start: np.ndarray,
    end: np.ndarray,
    reference_vectors: np.ndarray,
    steps: int = 81,
) -> tuple[np.ndarray, np.ndarray]:
    """Adiabatically match eigenvectors to reference columns by overlap."""
    tracked = reference_vectors
    energies = np.real(np.diag(start))
    for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
        matrix = (1.0 - fraction) * start + fraction * end
        raw_energies, raw_vectors = np.linalg.eigh(matrix)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            overlaps = np.abs(tracked.conj().T @ raw_vectors) ** 2
        if not np.isfinite(overlaps).all():
            raise FloatingPointError("Non-finite overlap during eigenstate tracking")
        reference_indices, new_indices = linear_sum_assignment(-overlaps)
        ordering = np.empty(len(raw_energies), dtype=int)
        ordering[reference_indices] = new_indices
        energies = raw_energies[ordering]
        tracked = raw_vectors[:, ordering]
    return energies, tracked


def near_pairs(
    energies: np.ndarray,
    states: list[tuple[int, ...]],
    cutoff_MHz: float,
) -> list[dict]:
    rows = []
    for left, right in combinations(range(len(states)), 2):
        if sum(states[left]) != sum(states[right]):
            continue
        difference_MHz = 1000.0 * abs(energies[right] - energies[left])
        if difference_MHz >= cutoff_MHz:
            continue
        rows.append(
            {
                "left_index": left,
                "right_index": right,
                "left": states[left],
                "right": states[right],
                "N": sum(states[left]),
                "detuning_MHz": float(difference_MHz),
                "category": classify_transition(states[left], states[right]),
            }
        )
    return sorted(rows, key=lambda row: (row["N"], row["detuning_MHz"]))


def compact_state_label(
    stage: str,
    index: int,
    states: list[tuple[int, ...]],
    vectors: np.ndarray,
    parent_vectors: np.ndarray | None,
) -> str:
    ancestry = "".join(map(str, states[index]))
    if stage == "bare":
        return state_text(states[index])
    if parent_vectors is None:
        weight = abs(vectors[index, index]) ** 2
    else:
        weight = abs(np.vdot(parent_vectors[:, index], vectors[:, index])) ** 2
    prefix = "DR" if stage == "dual rail" else "FD←DR"
    return f"{prefix}[{ancestry}; w={weight:.2f}]"


def draw_panel(
    axis: plt.Axes,
    rows: list[dict],
    stage: str,
    states: list[tuple[int, ...]],
    vectors: np.ndarray,
    parent_vectors: np.ndarray | None,
    cutoff_MHz: float,
) -> None:
    positions = np.arange(len(rows))
    labels = []
    previous_n = None
    for position, row in zip(positions, rows):
        if row["N"] != previous_n:
            if position:
                axis.axhline(position - 0.5, color="#222222", lw=0.8, alpha=0.3)
            previous_n = row["N"]
        left_label = compact_state_label(
            stage, row["left_index"], states, vectors, parent_vectors
        )
        right_label = compact_state_label(
            stage, row["right_index"], states, vectors, parent_vectors
        )
        labels.append(f"N={row['N']}  {left_label} ↔ {right_label}")
        color = CATEGORY_COLORS[row["category"]]
        axis.hlines(position, 0.0, row["detuning_MHz"], color=color, lw=1.8)
        axis.plot(row["detuning_MHz"], position, "o", color=color, ms=5)
        axis.annotate(
            f" {row['detuning_MHz']:.4f}",
            (row["detuning_MHz"], position),
            xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
        )
    if not rows:
        axis.text(
            0.5, 0.5, "No state pairs below cutoff", transform=axis.transAxes,
            ha="center", va="center", fontsize=13, color="#555555",
        )
    axis.set_yticks(positions, labels, fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlim(0.0, cutoff_MHz * 1.08)
    axis.set_xlabel("State-pair detuning |Ej − Ei| (MHz)")
    axis.set_title(f"{stage.title()} ({len(rows)} pairs)", fontsize=13)
    axis.grid(axis="x", alpha=0.2)


def write_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as stream:
        fields = ("N", "left", "right", "category", "detuning_MHz")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "N": row["N"],
                    "left": state_text(row["left"]),
                    "right": state_text(row["right"]),
                    "category": row["category"],
                    "detuning_MHz": row["detuning_MHz"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "current_gate_point_config.json"),
    )
    parser.add_argument("--cutoff-mhz", type=float, default=5.0)
    parser.add_argument("--max-total-excitation", type=int, default=4)
    parser.add_argument("--output-directory", default="three_stage_near_degenerate_N4")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw = json.loads(config_path.read_text())
    output = Path(__file__).resolve().parents[2] / "output" / args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    states = [
        levels
        for levels in product(range(args.max_total_excitation + 1), repeat=4)
        if sum(levels) <= args.max_total_excitation
    ]

    zero = {key: 0.0 for key in raw["couplings_GHz"]}
    intra = {**raw["couplings_GHz"], "A2_B1": 0.0}
    full = dict(raw["couplings_GHz"])
    h0, _ = projected_hamiltonian(raw, zero, states)
    h_intra, _ = projected_hamiltonian(raw, intra, states)
    h_full, _ = projected_hamiltonian(raw, full, states)

    identity = np.eye(len(states), dtype=complex)
    bare_energies = np.real(np.diag(h0))
    intra_energies, intra_vectors = track_eigenstates(h0, h_intra, identity)
    full_energies, full_vectors = track_eigenstates(
        h_intra, h_full, intra_vectors
    )
    stages = (
        ("bare", bare_energies, identity, None),
        ("dual rail", intra_energies, intra_vectors, None),
        ("fully coupled", full_energies, full_vectors, intra_vectors),
    )
    stage_rows = [near_pairs(energies, states, args.cutoff_mhz) for _, energies, _, _ in stages]

    maximum_rows = max(map(len, stage_rows), default=0)
    figure_height = max(8.5, 0.34 * maximum_rows + 3.0)
    figure, axes = plt.subplots(1, 3, figsize=(27, figure_height))
    for axis, (stage, _, vectors, parent_vectors), rows in zip(axes, stages, stage_rows):
        draw_panel(
            axis, rows, stage, states, vectors, parent_vectors, args.cutoff_mhz
        )
        write_rows(rows, output / f"{stage.replace(' ', '_')}_pairs.csv")

    handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=category)
        for category, color in CATEGORY_COLORS.items()
    ]
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958),
        ncol=4, frameon=False,
    )
    figure.suptitle(
        f"Current TDT gate point: near-degenerate ΔN=0 state pairs below "
        f"{args.cutoff_mhz:g} MHz, N ≤ {args.max_total_excitation}",
        fontsize=16, y=0.992,
    )
    figure.subplots_adjust(left=0.12, right=0.99, top=0.89, bottom=0.06, wspace=0.68)
    figure.savefig(output / "three_stage_delta_N_0_below_5MHz.png", dpi=190, bbox_inches="tight")
    plt.close(figure)
    print("Pair counts:", {stage[0]: len(rows) for stage, rows in zip(stages, stage_rows)})
    print(f"Saved output to {output}")


if __name__ == "__main__":
    main()
