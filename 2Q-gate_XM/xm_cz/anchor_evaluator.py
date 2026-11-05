"""Rank fixed-A designs by aggregate S1, then calibrate with fidelity fallback.

The target may make its intentional CZ excursion while every logical input
should avoid unwanted population transfer. A source-only score cannot screen
the other three components of a superposition. It remains a diagnostic and an
optional explicit ceiling, never a resonance/anchor-ranking tie-breaker.
"""

from __future__ import annotations

from .evaluator import evaluate_ec


def evaluate_anchor_set(payload: dict) -> dict:
    anchors = payload.get("rail_A_anchors", [])
    if not anchors:
        raise ValueError("rail_A_anchors must not be empty")
    maximum_s1 = float(payload.get("maximum_S1", float("inf")))
    maximum_source = float(payload.get("maximum_source_score", float("inf")))
    fallback = int(payload.get("fidelity_fallback_count", len(anchors)))
    if fallback < 1:
        raise ValueError("fidelity_fallback_count must be positive")
    base = {
        k: v
        for k, v in payload.items()
        if k
        not in {
            "rail_A_anchors",
            "maximum_S1",
            "maximum_source_score",
            "fidelity_fallback_count",
        }
    }
    screening = []
    passing = []
    for anchor in anchors:
        trial = base | {
            "fixed_A": anchor["fixed_A"],
            "logical_splittings_GHz": [
                anchor["gate_logical_frequency_GHz"],
                base["logical_splittings_GHz"][1],
            ],
            "screening_only": True,
        }
        result = evaluate_ec(trial)
        position = result.get("S1_position") or {}
        record = {
            "anchor_id": anchor["anchor_id"],
            "anchor_hash": anchor["anchor_hash"],
            "readout_operating_frequency_GHz": anchor[
                "readout_operating_frequency_GHz"
            ],
            "status": result.get("status"),
            "S1": position.get("S1"),
            "intended_source_score": position.get("intended_source_score"),
            "maximum_logical_ratio": position.get("maximum_logical_ratio"),
            "logical_source": result.get("logical_source"),
            "fB_GHz": position.get("fB_GHz"),
        }
        accepted = (
            result.get("status") == "screened"
            and record["S1"] <= maximum_s1
            and record["intended_source_score"] <= maximum_source
        )
        record["passed_score_requirements"] = accepted
        screening.append(record)
        if accepted:
            passing.append((record, anchor))
    if not passing:
        return {
            "status": "no_anchor_passed_screening",
            "anchor_screening": screening,
            "anchors_considered": len(anchors),
        }
    passing.sort(key=lambda item: item[0]["S1"])
    attempts = []
    best_failure = None
    for rank, (_screen, anchor) in enumerate(passing[:fallback], 1):
        trial = base | {
            "fixed_A": anchor["fixed_A"],
            "logical_splittings_GHz": [
                anchor["gate_logical_frequency_GHz"],
                base["logical_splittings_GHz"][1],
            ],
        }
        result = evaluate_ec(trial)
        final = result.get("final") or {}
        passes = (
            result.get("status") == "calibrated"
            and final.get("raw_fidelity", 0) > 0.95
            and final.get("postselected_fidelity", 0) > 0.999
        )
        attempts.append(
            {
                "rank": rank,
                "anchor_id": anchor["anchor_id"],
                "status": result.get("status"),
                "raw_fidelity": final.get("raw_fidelity"),
                "postselected_fidelity": final.get("postselected_fidelity"),
                "passed_fidelity": passes,
            }
        )
        if best_failure is None or final.get("raw_fidelity", -1) > (
            best_failure.get("final") or {}
        ).get("raw_fidelity", -1):
            best_failure = result | {"selected_anchor": anchor}
        if passes:
            return result | {
                "selected_anchor": anchor,
                "selected_anchor_rank": rank,
                "anchor_screening": screening,
                "fidelity_attempts": attempts,
            }
    return (best_failure or {"status": "no_anchor_calibrated"}) | {
        "workflow_status": "no_ranked_anchor_passed_fidelity",
        "anchor_screening": screening,
        "fidelity_attempts": attempts,
    }
