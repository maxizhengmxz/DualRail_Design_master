#!/usr/bin/env python3
"""Create readable N<=4 transition-frequency plots for one case study."""

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


CATEGORY_COLORS = {
    "single-qubit": "#2878B5",
    "intra-rail": "#59A14F",
    "inter-rail": "#E17C05",
    "multi-qubit": "#B449A0",
}


def classify_transition(left: tuple[int, ...], right: tuple[int, ...]) -> str:
    changed = [index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]
    if len(changed) == 1:
        return "single-qubit"
    if len(changed) >= 3:
        return "multi-qubit"
    if set(changed) <= {0, 1} or set(changed) <= {2, 3}:
        return "intra-rail"
    return "inter-rail"


def projected_model(raw: dict, max_excitation: int = 4) -> dict:
    local_levels = max_excitation + 1
    specs = {
        label: TransmonSpec(**{**raw["transmons"][label], "ng": 0.0})
        for label in LABELS
    }
    basis = BasisSpec(
        ncut=raw["basis"]["ncut"], levels_per_transmon=local_levels
    )
    coupled = FourTransmonSystem(specs, raw["couplings_GHz"], basis)
    zero_couplings = {key: 0.0 for key in raw["couplings_GHz"]}
    uncoupled = FourTransmonSystem(specs, zero_couplings, basis)

    bare_states = [
        levels
        for levels in product(range(local_levels), repeat=4)
        if sum(levels) <= max_excitation
    ]
    bare_vectors = np.column_stack(
        [uncoupled.local_product_ket(levels).full().ravel() for levels in bare_states]
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        h0 = bare_vectors.conj().T @ (uncoupled.hamiltonian().full() / TWOPI) @ bare_vectors
        hc = bare_vectors.conj().T @ (coupled.hamiltonian().full() / TWOPI) @ bare_vectors
    if not np.isfinite(h0).all() or not np.isfinite(hc).all():
        raise FloatingPointError("Non-finite projected Hamiltonian")

    bare_energies = np.real(np.diag(h0))
    coupled_energies_raw, coupled_vectors = np.linalg.eigh(hc)
    overlap = np.abs(coupled_vectors) ** 2
    bare_indices, eigen_indices = linear_sum_assignment(-overlap)
    assignment = np.empty(len(bare_states), dtype=int)
    assignment[bare_indices] = eigen_indices
    coupled_energies = coupled_energies_raw[assignment]
    assigned_weights = overlap[np.arange(len(bare_states)), assignment]

    charge_bare = {}
    charge_coupled = {}
    for label in LABELS:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            projected = bare_vectors.conj().T @ coupled.n_full[label].full() @ bare_vectors
            dressed = coupled_vectors.conj().T @ projected @ coupled_vectors
        charge_bare[label] = projected
        charge_coupled[label] = dressed[np.ix_(assignment, assignment)]

    return {
        "states": bare_states,
        "bare_energies": bare_energies,
        "coupled_energies": coupled_energies,
        "assigned_weights": assigned_weights,
        "charge_bare": charge_bare,
        "charge_coupled": charge_coupled,
    }


def enumerate_transitions(model: dict, delta_n: int) -> list[dict]:
    rows = []
    states = model["states"]
    for left_index, right_index in combinations(range(len(states)), 2):
        left, right = states[left_index], states[right_index]
        if abs(sum(right) - sum(left)) != delta_n:
            continue
        if sum(left) > sum(right):
            left_index, right_index = right_index, left_index
            left, right = right, left
        bare_frequency = abs(
            model["bare_energies"][right_index] - model["bare_energies"][left_index]
        )
        coupled_frequency = abs(
            model["coupled_energies"][right_index]
            - model["coupled_energies"][left_index]
        )
        bare_ports = {
            label: abs(model["charge_bare"][label][right_index, left_index])
            for label in LABELS
        }
        coupled_ports = {
            label: abs(model["charge_coupled"][label][right_index, left_index])
            for label in LABELS
        }
        rows.append(
            {
                "left": left,
                "right": right,
                "initial_N": min(sum(left), sum(right)),
                "category": classify_transition(left, right),
                "bare_frequency_GHz": float(bare_frequency),
                "coupled_frequency_GHz": float(coupled_frequency),
                "bare_charge_amplitude": float(max(bare_ports.values())),
                "coupled_charge_amplitude": float(max(coupled_ports.values())),
                "dominant_ports": ",".join(
                    label
                    for label, value in coupled_ports.items()
                    if value >= 0.95 * max(coupled_ports.values(), default=0.0)
                    and value > 1e-12
                ),
                "minimum_assignment_weight": float(
                    min(
                        model["assigned_weights"][left_index],
                        model["assigned_weights"][right_index],
                    )
                ),
            }
        )
    return rows


def select_close_transitions(
    rows: list[dict], bare_window_MHz: float, coupled_window_MHz: float
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Build separate bare and coupled collision groups of at least two rows."""
    for row in rows:
        row["close_in_bare"] = False
        row["close_in_coupled"] = False
        row["nearest_bare_separation_MHz"] = np.inf
        row["nearest_coupled_separation_MHz"] = np.inf
        row["bare_collision_group"] = ""
        row["coupled_collision_group"] = ""

    all_groups = []
    for frequency_key, flag_key, separation_key, group_key, window_MHz in (
        (
            "bare_frequency_GHz", "close_in_bare",
            "nearest_bare_separation_MHz", "bare_collision_group",
            bare_window_MHz,
        ),
        (
            "coupled_frequency_GHz", "close_in_coupled",
            "nearest_coupled_separation_MHz", "coupled_collision_group",
            coupled_window_MHz,
        ),
    ):
        window_GHz = window_MHz / 1000.0
        ordered = sorted(range(len(rows)), key=lambda index: rows[index][frequency_key])
        groups = []
        current_group = [ordered[0]] if ordered else []
        for left_index, right_index in zip(ordered, ordered[1:]):
            separation_GHz = (
                rows[right_index][frequency_key] - rows[left_index][frequency_key]
            )
            separation_MHz = 1000.0 * separation_GHz
            rows[left_index][separation_key] = min(
                rows[left_index][separation_key], separation_MHz
            )
            rows[right_index][separation_key] = min(
                rows[right_index][separation_key], separation_MHz
            )
            if separation_GHz <= window_GHz:
                current_group.append(right_index)
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [right_index]
        if len(current_group) >= 2:
            groups.append(current_group)

        row_groups = []
        for group_number, indices in enumerate(groups, start=1):
            group_name = f"G{group_number:03d}"
            group_rows = [rows[index] for index in indices]
            for row in group_rows:
                row[flag_key] = True
                row[group_key] = group_name
            row_groups.append(group_rows)
        all_groups.append(row_groups)
    return all_groups[0], all_groups[1]


def paginate_groups(groups: list[list[dict]], rows_per_page: int) -> list[list[dict]]:
    """Paginate without ever separating members of one collision group."""
    pages: list[list[dict]] = []
    page: list[dict] = []
    for group in groups:
        if page and len(page) + len(group) > rows_per_page:
            pages.append(page)
            page = []
        page.extend(group)
    if page:
        pages.append(page)
    return pages


def state_text(state: tuple[int, ...]) -> str:
    return "|" + "".join(map(str, state)) + "⟩"


def make_plot(
    rows: list[dict], delta_n: int, output: Path, page: int, page_count: int,
    bare_window_MHz: float, coupled_window_MHz: float, focus: str,
    max_excitation: int,
) -> None:
    count = len(rows)
    figure_height = max(9.0, 0.30 * count + 2.8)
    figure, axes = plt.subplots(1, 2, figsize=(18, figure_height), sharey=True)
    titles = ("Couplings off: bare resonance map", "Couplings on: physical frequencies")
    frequency_keys = ("bare_frequency_GHz", "coupled_frequency_GHz")
    strength_keys = ("bare_charge_amplitude", "coupled_charge_amplitude")
    maximum_strength = max(
        (row["coupled_charge_amplitude"] for row in rows), default=1.0
    )
    threshold = max(1e-10, maximum_strength * 1e-4)
    y = np.arange(count)
    labels = [
        f"{row[f'{focus}_collision_group']}  N={row['initial_N']}  "
        f"{state_text(row['left'])} → {state_text(row['right'])}"
        for row in rows
    ]

    for axis, title, frequency_key, strength_key in zip(
        axes, titles, frequency_keys, strength_keys
    ):
        if not rows:
            axis.text(
                0.5, 0.5, "No collision groups found",
                transform=axis.transAxes, ha="center", va="center", fontsize=14,
                color="#555555",
            )
        previous_group = None
        for position, row in zip(y, rows):
            group = row[f"{focus}_collision_group"]
            if group != previous_group:
                if position > 0:
                    axis.axhline(position - 0.5, color="#222222", lw=0.8, alpha=0.28)
                previous_group = group
            if row["initial_N"] % 2 == 0:
                axis.axhspan(position - 0.5, position + 0.5, color="#F5F7FA", zorder=-2)
            frequency = row[frequency_key]
            allowed = row[strength_key] >= threshold
            axis.hlines(
                position,
                0.0,
                frequency,
                color=CATEGORY_COLORS[row["category"]],
                lw=1.8 if allowed else 1.0,
                ls="-" if allowed else "--",
                alpha=0.9 if allowed else 0.42,
            )
            axis.plot(
                frequency,
                position,
                marker="o" if allowed else "x",
                ms=5,
                color=CATEGORY_COLORS[row["category"]],
                alpha=0.95 if allowed else 0.5,
            )
            axis.annotate(
                f" {frequency:.4f}",
                (frequency, position),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
            )
        axis.set_title(title, fontsize=13, pad=12)
        axis.set_xlabel("Transition frequency (GHz)")
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        axis.margins(x=0.09)

    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=category)
        for category, color in CATEGORY_COLORS.items()
    ]
    legend_handles.extend(
        [
            plt.Line2D([0], [0], color="#555555", lw=2, ls="-", label="charge-allowed"),
            plt.Line2D([0], [0], color="#777777", lw=1, ls="--", label="charge-dark/suppressed"),
        ]
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.977),
        ncol=3,
        frameon=False,
    )
    focus_window_MHz = bare_window_MHz if focus == "bare" else coupled_window_MHz
    window_text = (
        f"{focus_window_MHz:g} MHz"
        if focus_window_MHz >= 1.0
        else f"{focus_window_MHz * 1000:g} kHz"
    )
    figure.suptitle(
        f"Current TDT gate point: {'ΔN = 0' if delta_n == 0 else '|ΔN| = 1'} "
        f"{focus} collision groups ({window_text}) "
        f"with N ≤ {max_excitation} — page {page}/{page_count}",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(left=0.23, right=0.98, top=0.942, bottom=0.04, wspace=0.10)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_csv(rows: list[dict], output: Path) -> None:
    fields = [
        "initial_N", "left", "right", "category", "bare_frequency_GHz",
        "coupled_frequency_GHz", "bare_charge_amplitude",
        "coupled_charge_amplitude", "dominant_ports", "minimum_assignment_weight",
        "close_in_bare", "close_in_coupled", "nearest_bare_separation_MHz",
        "nearest_coupled_separation_MHz", "bare_collision_group",
        "coupled_collision_group",
    ]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = {**row, "left": state_text(row["left"]), "right": state_text(row["right"])}
            writer.writerow(serialized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "current_gate_point_config.json"),
    )
    parser.add_argument("--bare-collision-window-mhz", type=float, default=5.0)
    parser.add_argument("--coupled-collision-window-mhz", type=float, default=5.0)
    parser.add_argument("--max-total-excitation", type=int, default=2)
    parser.add_argument("--rows-per-page", type=int, default=60)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    raw = json.loads(config_path.read_text())
    output = Path(__file__).resolve().parents[2] / "output" / "transition_frequency_plots"
    output.mkdir(exist_ok=True)

    model = projected_model(raw, max_excitation=args.max_total_excitation)
    for delta_n in (0, 1):
        all_rows = enumerate_transitions(model, delta_n)
        bare_groups, coupled_groups = select_close_transitions(
            all_rows,
            args.bare_collision_window_mhz,
            args.coupled_collision_window_mhz,
        )
        stem = "delta_N_0" if delta_n == 0 else "delta_N_1"
        write_csv(all_rows, output / f"{stem}_all_transitions.csv")
        for old_page in output.glob(f"{stem}_*_collision_page_*.png"):
            old_page.unlink()
        for focus, groups in (("bare", bare_groups), ("coupled", coupled_groups)):
            plotted_rows = [row for group in groups for row in group]
            write_csv(
                plotted_rows,
                output / f"{stem}_{focus}_collision_transitions.csv",
            )
            pages = paginate_groups(groups, args.rows_per_page)
            if not pages:
                pages = [[]]
            for page_index, page_rows in enumerate(pages, start=1):
                make_plot(
                    page_rows,
                    delta_n,
                    output / f"{stem}_{focus}_collision_page_{page_index:02d}.png",
                    page_index,
                    len(pages),
                    args.bare_collision_window_mhz,
                    args.coupled_collision_window_mhz,
                    focus,
                    args.max_total_excitation,
                )
            print(
                f"{stem} {focus}: {len(plotted_rows)} transitions in "
                f"{len(groups)} groups across {len(pages)} pages"
            )
    print(f"Saved plots and transition tables to {output}")


if __name__ == "__main__":
    main()
