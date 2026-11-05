"""Full-cosine spectra, separately hybridized rails and logical-origin scores.

Energies are cyclic GHz. The 61 charge states implement ncut=30; the lowest
four local eigenstates give a 16-state rail and a 256-state two-rail model.
Charge coefficient J in J*n_i*n_j is distinct from a dressed matrix element g.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from .channel_policy import LOGICAL_ORDER, interaction_scores, one_excitation_indices

charges = np.arange(-30, 31, dtype=float)


@lru_cache(maxsize=1024)
def local_model(ec: float, frequency: float) -> tuple:
    """Solve EJ for a bare f01 and return spectrum and projected charge."""

    def eig(ej, vectors=False):
        return eigh_tridiagonal(
            4 * ec * charges**2,
            np.full(60, -ej / 2),
            select="i",
            select_range=(0, 3),
            eigvals_only=not vectors,
        )

    ej = brentq(lambda value: np.diff(eig(value))[0] - frequency, 1, 29.999, xtol=1e-10)
    energies, vectors = eig(ej, True)
    return ej, energies - energies[0], vectors.T @ (charges[:, None] * vectors), vectors


@lru_cache(maxsize=1024)
def local_model_from_ej(ec: float, ej: float) -> tuple:
    """Return the same local representation without re-solving a fixed EJ."""
    if not 1 <= ej < 30:
        raise ValueError("EJ_limit")
    if ej / ec < 25:
        raise ValueError("EJ_over_EC")
    energies, vectors = eigh_tridiagonal(
        4 * ec * charges**2,
        np.full(60, -ej / 2),
        select="i",
        select_range=(0, 3),
        eigvals_only=False,
    )
    return ej, energies - energies[0], vectors.T @ (charges[:, None] * vectors), vectors


def _pair_from_locals(left, right, coupling, target_bare_index):
    diagonal = np.diag((left[1][:, None] + right[1][None, :]).ravel())
    interaction = np.kron(left[2], right[2])
    energies, vectors = np.linalg.eigh(diagonal + coupling * interaction)
    modes = list(
        one_excitation_indices(energies, abs(vectors[1]) ** 2 + abs(vectors[4]) ** 2)
    )
    energies -= energies[0]
    target = int(np.argmax(abs(vectors[target_bare_index]) ** 2))
    return {
        "energies": energies,
        "vectors": vectors,
        "modes": modes,
        "target": target,
        "J": coupling,
        "EJ": [left[0], right[0]],
        "left_charge": vectors.T @ np.kron(left[2], np.eye(4)) @ vectors,
        "right_charge": vectors.T @ np.kron(np.eye(4), right[2]) @ vectors,
        "local_vectors": [left[3], right[3]],
        "target_weight": float(abs(vectors[target_bare_index, target]) ** 2),
        "logical_splitting_GHz": float(energies[modes[1]] - energies[modes[0]]),
        "bare_f01_GHz": [float(left[1][1]), float(right[1][1])],
    }


@lru_cache(maxsize=256)
def pair_model(
    left_ec: float,
    right_ec: float,
    frequency: float,
    splitting: float,
    target_bare_index: int = 8,
) -> dict:
    """Solve rail J for the logical splitting; assign auxiliary by bare overlap.

    In left,right product order, index 8 is bare20 and index 2 is bare02.
    This is a pointwise assignment, not an adiabatic branch continuation.
    """
    left, right = local_model(left_ec, frequency), local_model(right_ec, frequency)
    if min(left[0] / left_ec, right[0] / right_ec) < 25:
        raise ValueError("EJ_over_EC")
    diagonal = np.diag((left[1][:, None] + right[1][None, :]).ravel())
    interaction = np.kron(left[2], right[2])

    def diagonalize(coupling):
        energies, vectors = np.linalg.eigh(diagonal + coupling * interaction)
        modes = list(
            one_excitation_indices(
                energies, abs(vectors[1]) ** 2 + abs(vectors[4]) ** 2
            )
        )
        return energies, vectors, modes

    def residual(coupling):
        energies, _, modes = diagonalize(coupling)
        return energies[modes[1]] - energies[modes[0]] - splitting

    coupling = brentq(residual, 0.0001, 0.25, xtol=1e-11)
    return _pair_from_locals(left, right, coupling, target_bare_index)


@lru_cache(maxsize=256)
def fixed_pair_model(
    left_ec, right_ec, left_ej, right_ej, coupling, target_bare_index=8
):
    """Build a rail from fixed fabrication/bias values without retuning EJ or J."""
    if not 0 < coupling <= 0.25:
        raise ValueError("invalid_intra_pair_coupling")
    left = local_model_from_ej(left_ec, left_ej)
    right = local_model_from_ej(right_ec, right_ej)
    return _pair_from_locals(left, right, coupling, target_bare_index)


def target_overlap(first: dict, second: dict) -> float:
    """Compare physical states, including changes in local eigenbases."""
    transform = np.kron(
        first["local_vectors"][0].T @ second["local_vectors"][0],
        first["local_vectors"][1].T @ second["local_vectors"][1],
    )
    return float(
        abs(
            first["vectors"][:, first["target"]].T
            @ transform
            @ second["vectors"][:, second["target"]]
        )
        ** 2
    )


def frequency_limits(ec: float) -> tuple[float, float]:
    """Bare f01 interval implied by EJ/EC>=25 and EJ<=29.999 GHz."""

    def transition(ej):
        return float(
            np.diff(
                eigh_tridiagonal(
                    4 * ec * charges**2,
                    np.full(60, -ej / 2),
                    select="i",
                    select_range=(0, 1),
                    eigvals_only=True,
                )
            )[0]
        )

    return transition(max(1.0, 25 * ec)), transition(29.999)


def layout(
    a: dict, b: dict, g: float, channel: str = "0020", *, logical_source: str
) -> tuple:
    """Score unwanted transfer from all four inputs around an intended CZ excursion.

    Only the selected source should make the intentional auxiliary excursion.
    Aggregate S1 screens unwanted mixing of every logical component. The
    source-only score measures target-transition isolation and is diagnostic.
    Phase accumulation and finite-time return still require gate calibration.
    """
    if channel not in ("0020", "0200"):
        raise ValueError("Unknown physical auxiliary channel")
    energies = (a["energies"][:, None] + b["energies"][None, :]).ravel()
    unit = np.kron(a["right_charge"], b["left_charge"])
    logical = [i * 16 + j for i in a["modes"] for j in b["modes"]]
    intended_source = logical[LOGICAL_ORDER.index(logical_source)]
    target = b["target"] if channel == "0020" else a["target"] * 16
    element = abs(unit[intended_source, target])
    if element < 1e-10:
        raise ValueError("vanishing_intended_element")
    coupling = g / element
    if coupling > 0.020:
        raise ValueError("inter_pair_J_cap")
    scores = interaction_scores(
        energies,
        coupling * unit,
        logical,
        intended_source,
        target,
        active_mask=abs(unit) > 1e-10,
    )
    source_index = LOGICAL_ORDER.index(logical_source)
    scores.update(
        logical_source=logical_source,
        logical_source_index=source_index,
        intended_source_score=scores["S1_by_input"][source_index],
    )
    return energies, unit, logical, target, coupling, scores


def collision_residual(a, b, channel, *, logical_source):
    """Energy of the selected logical product minus its auxiliary energy."""
    if logical_source not in LOGICAL_ORDER:
        raise ValueError("Unknown logical source")
    ia, ib = map(int, logical_source)
    source_energy = a["energies"][a["modes"][ia]] + b["energies"][b["modes"][ib]]
    if channel == "0020":
        return source_energy - b["energies"][b["target"]]
    if channel == "0200":
        return source_energy - a["energies"][a["target"]]
    raise ValueError("Unknown physical auxiliary channel")
