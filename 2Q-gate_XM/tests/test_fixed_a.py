"""Fixed readout anchors keep their physical rail controls."""

# ruff: noqa: E402 -- checkout path setup precedes workflow imports.
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from threadpoolctl import threadpool_limits
from xm_cz.config import validate_payload
from xm_cz.evaluator import evaluate_ec
from xm_cz.model import fixed_pair_model


class FixedRailATests(unittest.TestCase):
    def test_calibration_preserves_anchor_and_aggregate_selection(self):
        anchor = dict(
            EJ_GHz=[19.146849682002202, 13.079689412705315],
            J_GHz=0.048760704768226165,
            operating_frequency_GHz=4.85,
        )
        payload = dict(
            ec_GHz=[0.165, 0.250, 0.150, 0.164],
            fixed_A=anchor,
            bare_f01_bounds_GHz={name: [4.0, 5.5] for name in ("A1", "A2", "B1", "B2")},
            logical_splittings_GHz=[0.15, 0.09],
        )
        with threadpool_limits(limits=1):
            result = evaluate_ec(payload)
        self.assertEqual(result["status"], "calibrated")
        final = result["final"]
        self.assertEqual(final["EJ_GHz"][:2], anchor["EJ_GHz"])
        self.assertEqual(final["couplings_GHz"]["A1_A2"], anchor["J_GHz"])
        self.assertEqual(final["logical_source"], result["logical_source"])
        self.assertEqual(final["channel_selection"], result["channel_selection"])
        self.assertEqual(
            result["logical_source_policy"], "minimum_S1_at_common_intended_g"
        )
        self.assertAlmostEqual(
            result["logical_splittings_GHz"][0],
            fixed_pair_model(0.165, 0.25, *anchor["EJ_GHz"], anchor["J_GHz"])[
                "logical_splitting_GHz"
            ],
        )
        self.assertLess(abs(final["phase_error_rad"]), 0.01)

    def test_readout_witness_is_not_retuned(self):
        model = fixed_pair_model(
            0.165,
            0.250,
            19.146849682002202,
            13.079689412705315,
            0.048760704768226165,
            8,
        )
        self.assertEqual(model["EJ"], [19.146849682002202, 13.079689412705315])
        self.assertAlmostEqual(model["J"], 0.048760704768226165)
        self.assertTrue(0.14 <= model["logical_splitting_GHz"] <= 0.16)

    def test_fixed_a_ports_are_validated(self):
        payload = {
            "ec_GHz": [0.165, 0.250, 0.150, 0.164],
            "fixed_A": {
                "EJ_GHz": [19.1, 13.1],
                "J_GHz": 0.049,
                "operating_frequency_GHz": 4.85,
            },
        }
        validate_payload(payload)
        payload["fixed_A"].pop("J_GHz")
        with self.assertRaises(ValueError):
            validate_payload(payload)


if __name__ == "__main__":
    unittest.main()
