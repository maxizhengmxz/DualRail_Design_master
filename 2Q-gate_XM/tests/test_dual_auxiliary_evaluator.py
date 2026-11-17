"""Dual-auxiliary ranking keeps aggregate-S1 and fidelity-fallback policy."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from xm_cz.dual_auxiliary_evaluator import evaluate_dual_auxiliary_anchor_set


def anchor():
    return {
        "anchor_id": "a",
        "anchor_hash": "h",
        "readout_operating_frequency_GHz": 4.2,
        "gate_logical_frequency_GHz": 0.145,
        "fixed_A": {},
    }


def screened(s1, source, channel):
    return {
        "status": "screened",
        "logical_source": source,
        "S1_position": {
            "S1": s1,
            "intended_source_score": s1 / 2,
            "maximum_logical_ratio": 0.1,
            "fB_GHz": 4.3,
        },
        "resonance_channel": channel,
    }


class DualAuxiliaryEvaluatorTests(unittest.TestCase):
    def test_lower_aggregate_s1_auxiliary_is_calibrated_first(self):
        calibrated = {
            "status": "calibrated",
            "resonance_channel": "0200",
            "final": {"raw_fidelity": 0.98, "postselected_fidelity": 0.9995},
        }
        with patch(
            "xm_cz.dual_auxiliary_evaluator.evaluate_ec",
            side_effect=[screened(0.02, "10", "0020"), screened(0.01, "00", "0200"), calibrated],
        ) as evaluate:
            result = evaluate_dual_auxiliary_anchor_set(
                {"rail_A_anchors": [anchor()], "logical_splittings_GHz": [0.145, 0.08]}
            )
        self.assertEqual(result["resonance_channel"], "0200")
        self.assertEqual(result["selected_auxiliary_rank"], 1)
        self.assertEqual(evaluate.call_count, 3)

    def test_fidelity_failure_falls_back_to_other_auxiliary(self):
        failed = {
            "status": "calibrated",
            "resonance_channel": "0200",
            "final": {"raw_fidelity": 0.94, "postselected_fidelity": 0.9995},
        }
        passed = {
            "status": "calibrated",
            "resonance_channel": "0020",
            "final": {"raw_fidelity": 0.98, "postselected_fidelity": 0.9995},
        }
        with patch(
            "xm_cz.dual_auxiliary_evaluator.evaluate_ec",
            side_effect=[screened(0.02, "10", "0020"), screened(0.01, "00", "0200"), failed, passed],
        ):
            result = evaluate_dual_auxiliary_anchor_set(
                {"rail_A_anchors": [anchor()], "logical_splittings_GHz": [0.145, 0.08]}
            )
        self.assertEqual(result["resonance_channel"], "0020")
        self.assertEqual(result["selected_auxiliary_rank"], 2)
        self.assertEqual(len(result["dual_auxiliary_fidelity_attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
