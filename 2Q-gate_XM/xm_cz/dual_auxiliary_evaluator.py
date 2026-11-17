"""Rank fixed-A anchors across both supported CZ auxiliaries."""

from __future__ import annotations

from .evaluator import evaluate_ec


def evaluate_dual_auxiliary_anchor_set(payload: dict) -> dict:
    """Screen anchor/channel combinations at common g, then use fidelity fallback.

    Each channel independently compares all logical sources with the established
    aggregate-S1 policy.  This wrapper then compares the resulting verified
    minima across anchors and physical auxiliaries.  It does not use a
    source-specific score or crossing order for ranking.
    """
    anchors = payload.get("rail_A_anchors", [])
    if not anchors:
        raise ValueError("rail_A_anchors must not be empty")
    channels = payload.get("resonance_channels", ["0020", "0200"])
    if (
        not isinstance(channels, list)
        or not channels
        or len(set(channels)) != len(channels)
        or any(channel not in {"0020", "0200"} for channel in channels)
    ):
        raise ValueError("resonance_channels must be a unique nonempty list of 0020/0200")
    fallback = int(payload.get("fidelity_fallback_count", len(anchors) * len(channels)))
    if fallback < 1:
        raise ValueError("fidelity_fallback_count must be positive")
    maximum_s1 = float(payload.get("maximum_S1", float("inf")))
    maximum_source = float(payload.get("maximum_source_score", float("inf")))
    base = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "rail_A_anchors",
            "resonance_channels",
            "fidelity_fallback_count",
            "maximum_S1",
            "maximum_source_score",
            "anchor_ranking",
        }
    }
    screening = []
    passing = []
    for anchor in anchors:
        for channel in channels:
            trial = base | {
                "fixed_A": anchor["fixed_A"],
                "logical_splittings_GHz": [
                    anchor["gate_logical_frequency_GHz"],
                    base["logical_splittings_GHz"][1],
                ],
                "resonance_channel": channel,
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
                "resonance_channel": channel,
                "physical_auxiliary": "B20" if channel == "0020" else "A02",
                "status": result.get("status"),
                "logical_source": result.get("logical_source"),
                "S1": position.get("S1"),
                "intended_source_score": position.get("intended_source_score"),
                "maximum_logical_ratio": position.get("maximum_logical_ratio"),
                "fB_GHz": position.get("fB_GHz"),
                "logical_source_candidates": position.get("logical_source_candidates"),
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
            "status": "no_anchor_auxiliary_passed_screening",
            "dual_auxiliary_screening": screening,
            "anchors_considered": len(anchors),
            "channels_considered": channels,
        }
    passing.sort(key=lambda item: item[0]["S1"])
    attempts = []
    best_failure = None
    for rank, (screen, anchor) in enumerate(passing[:fallback], 1):
        channel = screen["resonance_channel"]
        trial = base | {
            "fixed_A": anchor["fixed_A"],
            "logical_splittings_GHz": [
                anchor["gate_logical_frequency_GHz"],
                base["logical_splittings_GHz"][1],
            ],
            "resonance_channel": channel,
        }
        result = evaluate_ec(trial)
        final = result.get("final") or {}
        passed = (
            result.get("status") == "calibrated"
            and final.get("raw_fidelity", 0) > 0.95
            and final.get("postselected_fidelity", 0) > 0.999
        )
        attempts.append(
            {
                "rank": rank,
                "anchor_id": anchor["anchor_id"],
                "resonance_channel": channel,
                "physical_auxiliary": screen["physical_auxiliary"],
                "logical_source": screen["logical_source"],
                "S1": screen["S1"],
                "status": result.get("status"),
                "raw_fidelity": final.get("raw_fidelity"),
                "postselected_fidelity": final.get("postselected_fidelity"),
                "passed_fidelity": passed,
            }
        )
        enriched = result | {
            "selected_anchor": anchor,
            "selected_auxiliary_rank": rank,
            "dual_auxiliary_screening": screening,
            "dual_auxiliary_fidelity_attempts": attempts,
            "resonance_channels_considered": channels,
            "selection_scope": "minimum_aggregate_S1_across_anchors_sources_and_auxiliaries",
        }
        if best_failure is None or final.get("raw_fidelity", -1) > (
            best_failure.get("final") or {}
        ).get("raw_fidelity", -1):
            best_failure = enriched
        if passed:
            return enriched
    return (best_failure or {}) | {
        "status": "no_dual_auxiliary_candidate_passed_fidelity",
        "dual_auxiliary_screening": screening,
        "dual_auxiliary_fidelity_attempts": attempts,
        "resonance_channels_considered": channels,
        "selection_scope": "minimum_aggregate_S1_across_anchors_sources_and_auxiliaries",
    }
