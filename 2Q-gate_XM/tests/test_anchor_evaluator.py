"""Anchor ranking and fallback preserve aggregate-S1 resonance selection."""

# ruff: noqa: E402 -- checkout path setup precedes workflow imports.
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from xm_cz.anchor_evaluator import evaluate_anchor_set


class AnchorEvaluatorTests(unittest.TestCase):
    def test_anchor_ranking_protects_all_logical_inputs(self):
        anchors = [
            {
                "anchor_id": name,
                "anchor_hash": name,
                "readout_operating_frequency_GHz": frequency,
                "gate_logical_frequency_GHz": 0.145,
                "fixed_A": {},
            }
            for name, frequency in (("clean_source", 4.8), ("clean_gate", 4.9))
        ]
        screening = [
            dict(
                status="screened",
                S1_position=dict(
                    S1=aggregate,
                    intended_source_score=source,
                    maximum_logical_ratio=0.1,
                ),
            )
            for aggregate, source in ((0.02575, 0.001), (0.005, 0.005))
        ]
        calibrated = dict(
            status="calibrated",
            final=dict(raw_fidelity=0.96, postselected_fidelity=0.9995),
        )
        with patch(
            "xm_cz.anchor_evaluator.evaluate_ec", side_effect=[*screening, calibrated]
        ) as evaluate:
            result = evaluate_anchor_set(
                dict(rail_A_anchors=anchors, logical_splittings_GHz=[0.145, 0.08])
            )
        self.assertEqual(result["selected_anchor"]["anchor_id"], "clean_gate")
        self.assertEqual(evaluate.call_count, 3)

    def test_all_screening_failures_skip_fidelity(self):
        payload = {
            "rail_A_anchors": [
                {
                    "anchor_id": "a",
                    "anchor_hash": "h",
                    "readout_operating_frequency_GHz": 4.8,
                    "gate_logical_frequency_GHz": 0.145,
                    "fixed_A": {},
                }
            ],
            "logical_splittings_GHz": [0.145, 0.08],
        }
        with patch(
            "xm_cz.anchor_evaluator.evaluate_ec",
            return_value={"status": "no_feasible_S1_position"},
        ) as evaluate:
            result = evaluate_anchor_set(payload)
        self.assertEqual(result["status"], "no_anchor_passed_screening")
        self.assertEqual(evaluate.call_count, 1)

    def test_ranked_fallback_stops_at_first_fidelity_pass(self):
        anchors = [
            {
                "anchor_id": name,
                "anchor_hash": name,
                "readout_operating_frequency_GHz": f,
                "gate_logical_frequency_GHz": 0.145,
                "fixed_A": {},
            }
            for name, f in (("a", 4.8), ("b", 4.9))
        ]
        screened = [
            {
                "status": "screened",
                "S1_position": {
                    "S1": s,
                    "intended_source_score": s,
                    "maximum_logical_ratio": 0.1,
                },
            }
            for s in (0.02, 0.01)
        ]
        failed = {
            "status": "calibrated",
            "final": {"raw_fidelity": 0.94, "postselected_fidelity": 0.9995},
        }
        passed = {
            "status": "calibrated",
            "final": {"raw_fidelity": 0.96, "postselected_fidelity": 0.9995},
        }
        with patch(
            "xm_cz.anchor_evaluator.evaluate_ec",
            side_effect=[*screened, failed, passed],
        ):
            result = evaluate_anchor_set(
                {"rail_A_anchors": anchors, "logical_splittings_GHz": [0.145, 0.08]}
            )
        self.assertEqual(result["selected_anchor"]["anchor_id"], "a")
        self.assertEqual(result["selected_anchor_rank"], 2)


if __name__ == "__main__":
    unittest.main()
