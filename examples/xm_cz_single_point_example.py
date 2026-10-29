"""Interactive inspection and single-point use of the 2Q-gate_XM workflow.

Run cells marked ``# %%`` in VS Code, Spyder, or a Jupyter-compatible editor.
The default cells inspect the inexpensive local/pair/layout data structures.
The optional final cell runs the complete frequency search and CZ calibration.
"""

# %% Locate the repository and the hyphenated workflow directory.
from pathlib import Path
from functools import lru_cache
import sys


def _find_project_root() -> Path:
    starts = [Path.cwd().resolve()]
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists() and (
                candidate / "2Q-gate_XM" / "xm_cz"
            ).is_dir():
                return candidate
    raise RuntimeError("Start inside DualRail_Design_master or install the project.")


PROJECT_ROOT = _find_project_root()
WORKFLOW_ROOT = PROJECT_ROOT / "2Q-gate_XM"
for path in (PROJECT_ROOT, WORKFLOW_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# %% Configure numerical libraries before importing NumPy/SciPy users.
from core.runtime import configure_single_threaded_blas


configure_single_threaded_blas()

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from xm_cz.evaluator import evaluate_ec
from xm_cz.model import (
    collision_residual,
    frequency_limits,
    layout,
    local_model,
    pair_model,
    target_overlap,
)


# %% Reference point used by the workflow's numerical physics test.
# Physical order is A1,A2,B1,B2. All energies are cyclic GHz.
EC_GHZ = [
    0.16087962339131393,
    0.29333085491231115,
    0.2285344500027132,
    0.133892466917573,
]
F_A_GHZ = 4.0588683965207455
F_B_GHZ = 4.306565336876355
SPLIT_A_GHZ = 0.150
SPLIT_B_GHZ = 0.090
G_GHZ = 0.005
CHANNEL = "0020"


# %% Inspect one local transmon result.
ej_a1, local_energies, local_charge, charge_basis_vectors = local_model(
    EC_GHZ[0], F_A_GHZ
)
local_summary = {
    "EJ_GHz": ej_a1,
    "energies_GHz": local_energies,
    "projected_charge_shape": local_charge.shape,
    "charge_basis_vectors_shape": charge_basis_vectors.shape,
    "achieved_f01_GHz": local_energies[1] - local_energies[0],
    "EJ_over_EC": ej_a1 / EC_GHZ[0],
}
local_summary


# %% Build pair A and solve the common B frequency from the collision condition.
# pair_model solves each pair's intra-rail J to reproduce the requested splitting.
# Its default target index 8 means the most-|20>-like pair eigenstate.
a = pair_model(EC_GHZ[0], EC_GHZ[1], F_A_GHZ, SPLIT_A_GHZ)


def solve_common_b_frequency(
    pair_a: dict,
    ec_b1_ghz: float,
    ec_b2_ghz: float,
    split_b_ghz: float,
    channel: str,
    bounds_ghz: tuple[float, float] = (4.0, 5.0),
) -> tuple[float, dict]:
    """Find one B-frequency root while staying on one auxiliary branch."""

    physical_limits = [frequency_limits(ec_b1_ghz), frequency_limits(ec_b2_ghz)]
    lower = max(bounds_ghz[0], *(limits[0] for limits in physical_limits))
    upper = min(bounds_ghz[1], *(limits[1] for limits in physical_limits))
    previous = None
    for frequency in np.linspace(lower + 1e-7, upper - 1e-7, 80):
        pair_b = pair_model(ec_b1_ghz, ec_b2_ghz, float(frequency), split_b_ghz)
        residual = collision_residual(pair_a, pair_b, channel)
        if previous is not None:
            old_frequency, old_residual, old_target = previous
            same_branch = channel == "0200" or old_target == pair_b["target"]
            if same_branch and old_residual * residual <= 0:
                branch = pair_b["target"]

                def equation(candidate_frequency: float) -> float:
                    candidate = pair_model(
                        ec_b1_ghz,
                        ec_b2_ghz,
                        candidate_frequency,
                        split_b_ghz,
                    )
                    if channel == "0020" and candidate["target"] != branch:
                        raise ValueError("Auxiliary branch changed inside root bracket")
                    return collision_residual(pair_a, candidate, channel)

                root = brentq(equation, old_frequency, float(frequency), xtol=1e-9)
                return root, pair_model(ec_b1_ghz, ec_b2_ghz, root, split_b_ghz)
        previous = float(frequency), residual, pair_b["target"]
    raise RuntimeError(
        "No collision root was found in the requested B-frequency bounds"
    )


solved_f_b_ghz, b = solve_common_b_frequency(
    a,
    EC_GHZ[2],
    EC_GHZ[3],
    SPLIT_B_GHZ,
    CHANNEL,
)
frequency_root_summary = {
    "stored_reference_fB_GHz": F_B_GHZ,
    "solved_fB_GHz": solved_f_b_ghz,
    "difference_Hz": 1e9 * (solved_f_b_ghz - F_B_GHZ),
    "collision_residual_MHz": 1e3 * collision_residual(a, b, CHANNEL),
}
frequency_root_summary


def pair_summary(pair: dict) -> dict:
    lower, upper = pair["modes"]
    logical_weight = (
        np.abs(pair["vectors"][1, :]) ** 2
        + np.abs(pair["vectors"][4, :]) ** 2
    )
    return {
        "keys": sorted(pair),
        "energies_shape": pair["energies"].shape,
        "vectors_shape": pair["vectors"].shape,
        "logical_mode_indices": pair["modes"],
        "logical_mode_energies_GHz": pair["energies"][[lower, upper]],
        "logical_mode_weights": logical_weight[[lower, upper]],
        "logical_splitting_GHz": pair["energies"][upper]
        - pair["energies"][lower],
        "target_index": pair["target"],
        "target_bare_weight": pair["target_weight"],
        "solved_EJ_GHz": pair["EJ"],
        "solved_intra_pair_J_GHz": pair["J"],
    }


pair_data = {"A": pair_summary(a), "B": pair_summary(b)}
pair_data


# %% Reproduce the mode-selection statement in explicit steps.
logical_weights_b = (
    np.abs(b["vectors"][1, :]) ** 2 + np.abs(b["vectors"][4, :]) ** 2
)
indices_sorted_by_weight = np.argsort(logical_weights_b)
two_largest_weight_indices = indices_sorted_by_weight[-2:]
modes_reconstructed = sorted(
    two_largest_weight_indices,
    key=lambda index: b["energies"][index],
)
assert modes_reconstructed == b["modes"]

mode_selection_debug = {
    "two_largest_weight_indices_before_energy_sort": two_largest_weight_indices,
    "modes_after_energy_sort": modes_reconstructed,
    "selected_weights": logical_weights_b[modes_reconstructed],
    "selected_energies_GHz": b["energies"][modes_reconstructed],
}
mode_selection_debug


# %% Inspect the most-|20>-like B auxiliary and its bare decomposition.
target_b = b["target"]
target_probabilities = np.abs(b["vectors"][:, target_b]) ** 2
largest_components = np.argsort(target_probabilities)[::-1][:6]


def pair_label(flat_index: int) -> str:
    return f"|{flat_index // 4}{flat_index % 4}>"


target_components = [
    {
        "bare_index": int(index),
        "bare_state": pair_label(int(index)),
        "probability": float(target_probabilities[index]),
    }
    for index in largest_components
]
target_components


# %% Plot the principal bare components of the B auxiliary.
figure, axis = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
axis.bar(
    [item["bare_state"] for item in target_components],
    [item["probability"] for item in target_components],
)
axis.set(
    xlabel="B-pair bare product state",
    ylabel="Squared overlap",
    title="Bare decomposition of the selected B20-like eigenstate",
)
axis.grid(axis="y", alpha=0.25)
figure


# %% Reconstruct the phase-calibration functions hidden inside evaluate_ec.
@lru_cache(maxsize=64)
def build_calibration_point(offset_mhz: float) -> dict:
    """Rebuild B and the full Hamiltonian at one common-frequency offset."""

    b_at_offset = pair_model(
        EC_GHZ[2],
        EC_GHZ[3],
        solved_f_b_ghz + 0.001 * offset_mhz,
        SPLIT_B_GHZ,
    )
    if CHANNEL == "0020" and (
        b_at_offset["target"] != b["target"]
        or target_overlap(b, b_at_offset) < 0.9
    ):
        raise ValueError("auxiliary_branch_change")
    es, interaction, indices, auxiliary, coupling, point_scores = layout(
        a, b_at_offset, G_GHZ, CHANNEL
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.diag(es) + coupling * interaction
    )
    coefficients = eigenvectors[indices].T
    return {
        "b": b_at_offset,
        "eigenvalues": eigenvalues,
        "coefficients": coefficients,
        "logical_indices": indices,
        "auxiliary_index": auxiliary,
        "coupling_GHz": coupling,
        "scores": point_scores,
    }


def logical_map_at(offset_mhz: float, duration_ns: float) -> np.ndarray:
    """Return the four-by-four logical map after static evolution."""

    point = build_calibration_point(offset_mhz)
    eigenvalues = point["eigenvalues"]
    coefficients = point["coefficients"]
    phases = np.exp(-2j * np.pi * eigenvalues * duration_ns)
    return coefficients.T @ (phases[:, None] * coefficients)


def calibration_metrics(offset_mhz: float, duration_ns: float) -> dict:
    """Calculate exact-pi error, return, acceptance, and raw fidelity."""

    logical_map = logical_map_at(offset_mhz, duration_ns)
    diagonal = np.diag(logical_map)
    if np.min(np.abs(diagonal)) < 0.1:
        raise ValueError("small_return_amplitude")
    q = diagonal[3] * diagonal[0] * np.conj(diagonal[2] * diagonal[1])
    phase_error = float(np.angle(-q))
    acceptance = float(np.sum(np.abs(logical_map) ** 2) / 4)
    raw_fidelity = float(
        (4 * acceptance + np.sum(np.abs(diagonal)) ** 2) / 20
    )
    return {
        "offset_MHz": float(offset_mhz),
        "duration_ns": float(duration_ns),
        "phase_error_rad": phase_error,
        "acceptance": acceptance,
        "raw_fidelity": raw_fidelity,
        "postselected_fidelity": raw_fidelity / acceptance,
        "return_by_input": np.abs(diagonal) ** 2,
        "leakage_by_input": 1 - np.sum(np.abs(logical_map) ** 2, axis=0),
        "logical_map": logical_map,
    }


def exact_pi_solutions(duration_ns: float) -> list[dict]:
    """Root-solve all continuous exact-pi crossings in the offset window."""

    offsets = np.asarray([-12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 12.0])
    errors = []
    for offset in offsets:
        try:
            errors.append(calibration_metrics(offset, duration_ns)["phase_error_rad"])
        except ValueError:
            errors.append(np.nan)
    solutions = []
    for left, right, left_error, right_error in zip(
        offsets[:-1], offsets[1:], errors[:-1], errors[1:], strict=True
    ):
        if (
            not np.isfinite(left_error + right_error)
            or left_error * right_error > 0
            or abs(left_error - right_error) > np.pi
        ):
            continue
        root = brentq(
            lambda offset: calibration_metrics(offset, duration_ns)[
                "phase_error_rad"
            ],
            left,
            right,
            xtol=2e-5,
        )
        solutions.append(calibration_metrics(root, duration_ns))
    return solutions


def trace_exact_pi_contour(durations_ns: np.ndarray) -> list[dict]:
    """Return exact-pi solutions; fidelity can then choose the saved gate."""

    return [
        solution
        for duration in durations_ns
        for solution in exact_pi_solutions(float(duration))
    ]


# This reproduces the expensive calibration logic without runner/checkpoint code.
RUN_EXACT_PI_CONTOUR = False
if RUN_EXACT_PI_CONTOUR:
    pi_contour = trace_exact_pi_contour(np.linspace(70.0, 130.0, 21))
    best_on_contour = max(pi_contour, key=lambda point: point["raw_fidelity"])
else:
    pi_contour = []
    best_on_contour = {
        "status": "disabled",
        "reason": "Set RUN_EXACT_PI_CONTOUR=True to run the expensive contour.",
    }
best_on_contour


# %% Build the two-pair layout and verify the selected collision.
energies, unit, logical, target, inter_pair_j, scores = layout(
    a, b, G_GHZ, CHANNEL
)
logical_names = ("00", "01", "10", "11")
logical_indices = dict(zip(logical_names, logical, strict=True))
source = logical_indices["01"]

# For 0020, target == 0 * 16 + b["target"]: A is in its absolute pair ground
# state and B is in its separately identified B20-like pair eigenstate.
assert target == b["target"]
assert abs(collision_residual(a, b, CHANNEL)) < 1e-8

layout_summary = {
    "energies_shape": energies.shape,
    "interaction_shape": unit.shape,
    "logical_indices": logical_indices,
    "target_index": target,
    "target_pair_indices": {"A": target // 16, "B": target % 16},
    "collision_residual_MHz": 1e3 * collision_residual(a, b, CHANNEL),
    "intended_matrix_element_MHz": 1e3
    * abs(inter_pair_j * unit[source, target]),
    "solved_inter_pair_J_MHz": 1e3 * inter_pair_j,
    "scores": scores,
}
layout_summary


# %% Compare the implemented and alternative logical sources for the same target.
# Both matrix elements can be nonzero. Only logical 01 is tuned into exact
# resonance by the current 0020 collision equation.
source_comparison = []
for name in ("01", "10"):
    index = logical_indices[name]
    source_comparison.append(
        {
            "logical_source": name,
            "detuning_MHz": 1e3 * (energies[index] - energies[target]),
            "matrix_element_MHz": 1e3
            * abs(inter_pair_j * unit[index, target]),
        }
    )
source_comparison


# %% Map the experimental B-frequency control to actual dressed collision detuning.
# The two quantities are not equal: changing fB moves both the logical and
# auxiliary eigenenergies and changes their eigenvectors.
b_offsets_mhz = np.linspace(-12.0, 12.0, 49)
collision_detuning_mhz = []
target_indices = []
for offset_mhz in b_offsets_mhz:
    b_at_offset = pair_model(
        EC_GHZ[2],
        EC_GHZ[3],
        solved_f_b_ghz + 0.001 * offset_mhz,
        SPLIT_B_GHZ,
    )
    collision_detuning_mhz.append(
        1e3 * collision_residual(a, b_at_offset, CHANNEL)
    )
    target_indices.append(b_at_offset["target"])

detuning_map = {
    "B_frequency_offset_MHz": b_offsets_mhz,
    "collision_detuning_MHz": np.asarray(collision_detuning_mhz),
    "target_indices": np.asarray(target_indices),
}

figure, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
axis.plot(
    detuning_map["B_frequency_offset_MHz"],
    detuning_map["collision_detuning_MHz"],
)
axis.axhline(0.0, color="black", linewidth=0.8)
axis.axvline(0.0, color="black", linewidth=0.8)
axis.set(
    xlabel="Common B bare-frequency offset (MHz)",
    ylabel="Dressed collision detuning (MHz)",
    title="Control coordinate versus actual collision detuning",
)
axis.grid(alpha=0.25)
figure


# %% Optional complete one-EC-point search and exact-pi calibration.
# This is substantially slower than the inspection cells above.
RUN_FULL_POINT_EVALUATION = False
payload = {
    "ec_GHz": EC_GHZ,
    "frequency_GHz": [4.0, 4.1, 0.05],
    "logical_splittings_GHz": [SPLIT_A_GHZ, SPLIT_B_GHZ],
    "bare_f01_bounds_GHz": {
        label: [4.0, 5.0] for label in ("A1", "A2", "B1", "B2")
    },
    "resonance_channel": CHANNEL,
    "g_GHz": G_GHZ,
    "point_budget_seconds": 180,
}

if RUN_FULL_POINT_EVALUATION:
    result = evaluate_ec(payload)
else:
    result = {
        "status": "disabled",
        "reason": (
            "Set RUN_FULL_POINT_EVALUATION=True to run selection and calibration."
        ),
        "payload": payload,
    }
result


# %% Extract the most frequently inspected calibrated-result fields safely.
if result["status"] == "calibrated":
    final = result["final"]
    calibrated_summary = {
        "channel": result["resonance_channel"],
        "logical_source": result["logical_source"],
        "fA_GHz": result["fA_GHz"],
        "fB_GHz": final["fB_GHz"],
        "time_ns": final["time_ns"],
        "raw_fidelity": final["raw_fidelity"],
        "postselected_fidelity": final["postselected_fidelity"],
        "acceptance": final["acceptance"],
        "phase_error_rad": final["phase_error_rad"],
        "leakage_by_input": final["leakage_by_input"],
        "EJ_GHz": final["EJ_GHz"],
        "couplings_GHz": final["couplings_GHz"],
        "S1": final["S1"],
        "S2": final["S2"],
    }
else:
    calibrated_summary = result
calibrated_summary
