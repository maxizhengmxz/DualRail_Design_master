"""Eigenstate tracking helpers."""

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

TRACKED_STATE_LABELS = ("g", "p", "m", "r", "pp", "pm", "mm", "pr", "mr")


def _operator_at_point(operator, point_idx: int, point_count: int, name: str):
    """Resolve a shared operator or a point-dependent operator provider."""
    if callable(operator):
        return operator(point_idx)
    if getattr(operator, "ndim", None) == 3:
        if operator.shape[0] != point_count:
            raise ValueError(f"{name} must contain one operator per state-table point")
        return operator[point_idx]
    if not hasattr(operator, "shape") and isinstance(operator, Sequence):
        if len(operator) != point_count:
            raise ValueError(f"{name} must contain one operator per state-table point")
        return operator[point_idx]
    return operator


def _matrixelement_strength(eigenstates: np.ndarray, operator, name: str):
    basis_dim = eigenstates.shape[0]
    if getattr(operator, "shape", None) != (basis_dim, basis_dim):
        raise ValueError(
            f"{name} shape must match the eigenstate basis dimension {basis_dim}"
        )
    matrixelements = eigenstates.conj().T @ (operator @ eigenstates)
    return np.abs(matrixelements) ** 2


def _normalized_rows(scores: np.ndarray, labels):
    maxima = np.max(scores, axis=1)
    missing = [
        label for label, maximum in zip(labels, maxima, strict=True) if maximum <= 0
    ]
    if missing:
        raise ValueError(f"no nonzero assignment score for labels {missing}")
    return scores / maxima[:, None]


def _assign_one_excitation(q3_strength, p_strength, m_strength):
    candidate_levels = np.arange(1, q3_strength.shape[0])
    labels = ("p", "m", "r")
    raw_scores = np.vstack(
        (
            p_strength[0, candidate_levels],
            m_strength[0, candidate_levels],
            q3_strength[0, candidate_levels],
        )
    )
    scores = _normalized_rows(raw_scores, labels)
    label_rows, candidate_cols = linear_sum_assignment(-scores)
    assignment = {
        labels[row]: int(candidate_levels[col])
        for row, col in zip(label_rows, candidate_cols, strict=True)
    }
    assignment_score = {
        labels[row]: float(scores[row, col])
        for row, col in zip(label_rows, candidate_cols, strict=True)
    }
    return assignment, assignment_score


def _rail_hybridization(q3_strength, n1_strength, n2_strength):
    """Return the mixing of the two rail-like one-excitation states."""
    rail_assignment, _ = _assign_one_excitation(q3_strength, n1_strength, n2_strength)
    rail_levels = [rail_assignment["p"], rail_assignment["m"]]
    selected = np.vstack(
        (
            n1_strength[0, rail_levels],
            n2_strength[0, rail_levels],
        )
    )
    participation = selected / np.sum(selected, axis=1, keepdims=True)
    return float(np.clip(1.0 - 2.0 * np.mean(np.abs(participation - 0.5)), 0.0, 1.0))


def _assign_two_excitation(q3_strength, p_strength, m_strength, one_excitation):
    p = one_excitation["p"]
    m = one_excitation["m"]
    r = one_excitation["r"]
    excluded = np.array([0, p, m, r])
    candidate_levels = np.setdiff1d(np.arange(q3_strength.shape[0]), excluded)
    labels = ("pp", "pm", "mm", "pr", "mr")

    raw_scores = np.vstack(
        (
            p_strength[p, candidate_levels],
            np.sqrt(p_strength[m, candidate_levels] * m_strength[p, candidate_levels]),
            m_strength[m, candidate_levels],
            np.sqrt(q3_strength[p, candidate_levels] * p_strength[r, candidate_levels]),
            np.sqrt(q3_strength[m, candidate_levels] * m_strength[r, candidate_levels]),
        )
    )
    parent_threshold = np.array([p, max(p, m), m, max(p, r), max(m, r)])
    raw_scores[candidate_levels[None, :] <= parent_threshold[:, None]] = 0.0
    scores = _normalized_rows(raw_scores, labels)
    label_rows, candidate_cols = linear_sum_assignment(-scores)
    assignment = {
        labels[row]: int(candidate_levels[col])
        for row, col in zip(label_rows, candidate_cols, strict=True)
    }
    assignment_score = {
        labels[row]: float(scores[row, col])
        for row, col in zip(label_rows, candidate_cols, strict=True)
    }
    return assignment, assignment_score


def track_levels_by_overlap(
    state_table,
    energy_table: np.ndarray | None = None,
    *,
    q3_operator=None,
    n1_operator=None,
    n2_operator=None,
    hybridization_threshold: float | None = 0.5,
):
    """Label the dressed states through two total excitations, excluding ``rr``.

    At strongly hybridized points, the common rail mode ``p`` is driven by
    ``n1 + n2`` and the differential rail mode ``m`` by ``n1 - n2``. If the
    rail hybridization is below ``hybridization_threshold``, the tracking falls
    back to the physical ``n1`` and ``n2`` operators, respectively. Set the
    threshold to ``None`` to disable the fallback. Mixed two-excitation states
    are scored from both allowed parent transitions, and assignments within
    each excitation manifold are globally one-to-one.

    Operators may be shared matrices, sequences containing one matrix per
    state-table point, or callables accepting the point index. Point-dependent
    rail operators are required when the eigenstate basis changes across a
    parameter sweep.
    """
    if isinstance(state_table, np.ndarray) and state_table.ndim == 2:
        states = [np.asarray(state_table)]
    else:
        states = [np.asarray(state_entry) for state_entry in state_table]
    if len(states) == 0:
        raise ValueError("state_table must contain at least one parameter point")
    if any(state.ndim != 2 for state in states):
        raise ValueError("state entries must be 2D arrays")
    if q3_operator is None or n1_operator is None or n2_operator is None:
        raise ValueError("q3_operator, n1_operator, and n2_operator are required")
    if hybridization_threshold is not None:
        hybridization_threshold = float(hybridization_threshold)
        if (
            not np.isfinite(hybridization_threshold)
            or not 0.0 <= hybridization_threshold <= 1.0
        ):
            raise ValueError("hybridization_threshold must be between 0 and 1")

    level_count = states[0].shape[1]
    if level_count < len(TRACKED_STATE_LABELS):
        raise ValueError(
            f"at least {len(TRACKED_STATE_LABELS)} eigenstates are required"
        )
    if any(state.shape[1] != level_count for state in states):
        raise ValueError(
            "all state entries must contain the same number of eigenstates"
        )

    tracked_level = np.full((len(states), len(TRACKED_STATE_LABELS)), -1, dtype=int)
    assignment_score = np.full_like(tracked_level, np.nan, dtype=float)

    for point_idx, current in enumerate(states):
        q3_matrix = _operator_at_point(
            q3_operator, point_idx, len(states), "q3_operator"
        )
        n1_matrix = _operator_at_point(
            n1_operator, point_idx, len(states), "n1_operator"
        )
        n2_matrix = _operator_at_point(
            n2_operator, point_idx, len(states), "n2_operator"
        )
        q3_strength = _matrixelement_strength(current, q3_matrix, "q3_operator")
        n1_strength = _matrixelement_strength(current, n1_matrix, "n1_operator")
        n2_strength = _matrixelement_strength(current, n2_matrix, "n2_operator")
        p_strength = _matrixelement_strength(
            current, n1_matrix + n2_matrix, "n1_operator + n2_operator"
        )
        m_strength = _matrixelement_strength(
            current, n1_matrix - n2_matrix, "n1_operator - n2_operator"
        )
        if (
            hybridization_threshold is not None
            and _rail_hybridization(q3_strength, n1_strength, n2_strength)
            < hybridization_threshold
        ):
            p_strength = n1_strength
            m_strength = n2_strength

        one_excitation, one_score = _assign_one_excitation(
            q3_strength, p_strength, m_strength
        )
        two_excitation, two_score = _assign_two_excitation(
            q3_strength, p_strength, m_strength, one_excitation
        )
        assignment = {"g": 0, **one_excitation, **two_excitation}
        scores = {"g": 1.0, **one_score, **two_score}
        tracked_level[point_idx] = [assignment[label] for label in TRACKED_STATE_LABELS]
        assignment_score[point_idx] = [scores[label] for label in TRACKED_STATE_LABELS]

    if energy_table is None:
        return tracked_level, assignment_score

    energy_table = np.asarray(energy_table)
    if energy_table.ndim == 1 and len(states) == 1:
        energy_table = energy_table[None, :]
    if energy_table.shape != (len(states), level_count):
        raise ValueError(
            "energy_table shape must match the state-table point and level counts"
        )
    tracked_energy = np.take_along_axis(energy_table, tracked_level, axis=1)
    return tracked_level, assignment_score, tracked_energy


__all__ = ["TRACKED_STATE_LABELS", "track_levels_by_overlap"]
