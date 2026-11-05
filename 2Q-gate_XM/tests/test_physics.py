"""Reference invariants, channel mirror symmetry and public-input checks."""
# ruff: noqa: E402 -- path and BLAS setup must precede numerical imports.

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from core.runtime import configure_single_threaded_blas

configure_single_threaded_blas()
import numpy as np
from xm_cz.config import validate_payload
from xm_cz.evaluator import evaluate_ec
from xm_cz.model import collision_residual, layout, pair_model


class PhysicsTests(unittest.TestCase):
    def test_clean_source_does_not_outweigh_dirty_spectator(self):
        # Resonance 10 has the cleanest target (0.001) but contaminates 11.
        # Resonance 01 protects all four inputs and must win aggregate S1.
        roots = dict(zip(("00", "01", "10", "11"), (4.2, 4.4, 4.6, 4.8), strict=True))
        by_input = {
            "00": [0.02] * 4,
            "01": [0.005] * 4,
            "10": [0.001, 0.001, 0.001, 0.1],
            "11": [0.03] * 4,
        }

        def pair(ec1, ec2, frequency, splitting, target_bare_index=8):
            return {"frequency": frequency, "target": 1}

        def residual(a, b, channel, *, logical_source):
            return b["frequency"] - roots[logical_source]

        def scores(a, b, g, channel, *, logical_source):
            values = by_input[logical_source]
            index = ("00", "01", "10", "11").index(logical_source)
            return (
                None,
                None,
                None,
                None,
                None,
                dict(
                    S1=float(np.mean(values)),
                    S1_by_input=values,
                    logical_source=logical_source,
                    logical_source_index=index,
                    intended_source_score=values[index],
                    maximum_logical_ratio=0.1,
                ),
            )

        payload = dict(
            ec_GHz=[0.165, 0.25, 0.16, 0.26],
            frequency_GHz=[4.5, 4.5, 0.05],
            screening_only=True,
        )
        with (
            patch("xm_cz.evaluator.pair_model", side_effect=pair),
            patch("xm_cz.evaluator.frequency_limits", return_value=(4.0, 5.0)),
            patch("xm_cz.evaluator.collision_residual", side_effect=residual),
            patch("xm_cz.evaluator.layout", side_effect=scores),
        ):
            result = evaluate_ec(payload)
            restricted = evaluate_ec(dict(payload, logical_sources=["10"]))
        self.assertEqual(result["status"], "screened")
        self.assertEqual(result["logical_source"], "01")
        self.assertAlmostEqual(result["S1_position"]["S1"], 0.005)
        self.assertEqual(
            result["channel_selection"]["selection_rule"],
            "minimum_S1_at_common_intended_g",
        )
        self.assertEqual(restricted["logical_source"], "10")

    def test_0200_calibration_and_matrix_element_time_scaling(self):
        payload = {
            "ec_GHz": [
                0.133892466917573,
                0.2285344500027132,
                0.29333085491231115,
                0.16087962339131393,
            ],
            "frequency_GHz": [4.30, 4.35, 0.05],
            "logical_splittings_GHz": [0.09, 0.15],
            "resonance_channel": "0200",
            "logical_sources": ["10"],
            "g_GHz": 0.0025,
        }
        result = evaluate_ec(payload)
        self.assertEqual(result["status"], "calibrated")
        self.assertEqual(result["logical_source"], "10")
        candidates = result["S1_position"]["logical_source_candidates"]
        self.assertEqual(
            [value["logical_source"] for value in candidates], ["00", "01", "10", "11"]
        )
        valid = [value for value in candidates if value["status"] == "valid"]
        self.assertEqual(
            result["logical_source"],
            min(valid, key=lambda value: value["S1"])["logical_source"],
        )
        self.assertEqual(len(valid), 1)
        self.assertEqual(
            result["channel_selection"]["selection_rule"],
            "minimum_S1_at_common_intended_g",
        )
        source_index = result["logical_source_index"]
        self.assertAlmostEqual(
            result["final"]["intended_source_score"],
            result["final"]["S1_by_input"][source_index],
            places=12,
        )
        self.assertEqual(result["duration_window_ns"], [140.0, 260.0])
        self.assertLess(abs(result["final"]["phase_error_rad"]), 0.01)

    def test_weekend_reference_and_reversed_channel(self):
        ec = [
            0.16087962339131393,
            0.29333085491231115,
            0.2285344500027132,
            0.133892466917573,
        ]
        fa, fb = 4.0588683965207455, 4.306565336876355
        a = pair_model(*ec[:2], fa, 0.15)
        b = pair_model(*ec[2:], fb, 0.09)
        energies, unit, logical, target, coupling, score = layout(
            a, b, 0.005, logical_source="01"
        )
        self.assertAlmostEqual(
            collision_residual(a, b, "0020", logical_source="01"), 0, places=8
        )
        self.assertAlmostEqual(energies[logical[1]] - energies[target], 0, places=8)
        self.assertAlmostEqual(
            abs(coupling * unit[logical[1], target]), 0.005, places=12
        )
        self.assertAlmostEqual(score["S1"], 0.0080940711241, places=8)
        self.assertTrue(np.allclose(unit, unit.T))
        # Reverse physical node order: A1,A2,B1,B2 -> B2,B1,A2,A1.
        ma = pair_model(ec[3], ec[2], fb, 0.09, 2)
        mb = pair_model(ec[1], ec[0], fa, 0.15)
        es, v, indices, x, j, mirrored = layout(
            ma, mb, 0.005, "0200", logical_source="10"
        )
        self.assertAlmostEqual(
            collision_residual(ma, mb, "0200", logical_source="10"), 0, places=8
        )
        self.assertEqual(x, ma["target"] * 16)
        self.assertAlmostEqual(abs(j * v[indices[2], x]), 0.005, places=12)
        self.assertAlmostEqual(score["S1"], mirrored["S1"], places=8)

    def test_invalid_ports_and_uniform_grid(self):
        from xm_cz.runner import make_tasks

        cfg = json.loads((ROOT / "configs/example_grid.json").read_text())
        tasks = make_tasks(cfg)
        self.assertEqual(len(tasks), 6727)
        p = tasks[0]["payload"]
        validate_payload(p)
        for key, value in [
            ("logical_splittings_GHz", [0.15, -0.09]),
            ("g_GHz", 0),
            ("resonance_channel", "2000"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_payload({**p, key: value})
        bad = copy.deepcopy(p)
        bad["bare_f01_bounds_GHz"]["A2"] = [6.0, 7.0]
        with self.assertRaisesRegex(ValueError, "intersection"):
            validate_payload(bad)

        extended = copy.deepcopy(p)
        extended["ec_GHz"] = [0.150, 0.315, 0.315, 0.150]
        validate_payload(extended)
        extended["ec_GHz"][1] = 0.400
        with self.assertRaisesRegex(ValueError, "0,0.400"):
            validate_payload(extended)

        logical_cfg = json.loads((ROOT / "configs/logical_b_grid.json").read_text())
        logical_tasks = make_tasks(logical_cfg)
        self.assertEqual(len(logical_tasks), 4805)
        self.assertEqual(logical_tasks[0]["payload"]["ec_GHz"][:2], [0.15, 0.21])
        self.assertEqual(
            sorted(
                {task["payload"]["logical_splittings_GHz"][1] for task in logical_tasks}
            ),
            [0.08, 0.085, 0.09, 0.095, 0.1],
        )

        logical_a_cfg = json.loads((ROOT / "configs/logical_a_grid.json").read_text())
        logical_a_tasks = make_tasks(logical_a_cfg)
        self.assertEqual(len(logical_a_tasks), 4805)
        self.assertEqual(
            sorted(
                {
                    task["payload"]["logical_splittings_GHz"][0]
                    for task in logical_a_tasks
                }
            ),
            [0.14, 0.145, 0.15, 0.155, 0.16],
        )
        self.assertEqual(
            {task["payload"]["logical_splittings_GHz"][1] for task in logical_a_tasks},
            {0.09},
        )

    def test_calibration_preserves_independent_frequency_bounds(self):
        p = json.loads((ROOT / "configs/smoke.json").read_text())
        payload = {**p["task_defaults"], **p["tasks"][1]}
        payload["bare_f01_bounds_GHz"]["B1"] = [4.1, 4.8]
        payload["bare_f01_bounds_GHz"]["B2"] = [4.2, 4.9]
        result = evaluate_ec(payload)
        self.assertEqual(result["status"], "calibrated")
        final = result["final"]
        self.assertEqual(
            result["logical_source"], result["S1_position"]["logical_source"]
        )
        self.assertEqual(final["channel_selection"], result["channel_selection"])
        self.assertEqual(
            result["channel_selection"]["selection_rule"],
            "minimum_S1_at_common_intended_g",
        )
        self.assertAlmostEqual(result["channel_selection"]["target_g_mhz"], 5.0)
        self.assertTrue(4.2 <= final["fB_GHz"] <= 4.8)
        self.assertLess(abs(final["phase_error_rad"]), 0.01)
        self.assertLess(final["maximum_logical_ratio"], 0.2)
        self.assertEqual(result["duration_window_ns"], [70.0, 130.0])
        self.assertAlmostEqual(
            final["raw_fidelity"] / final["acceptance"],
            final["postselected_fidelity"],
            places=12,
        )

    def test_score_and_logical_definitions_have_one_authority(self):
        from xm_cz import channel_policy as xm

        from core import cz_channel_policy as shared

        self.assertIs(xm.interaction_scores, shared.interaction_scores)
        self.assertIs(xm.select_channel, shared.select_channel)
        self.assertIs(xm.one_excitation_indices, shared.one_excitation_indices)
        self.assertIs(xm.off_target_definition, shared.off_target_definition)
        self.assertIs(xm.effective_logical_exchange, shared.effective_logical_exchange)


if __name__ == "__main__":
    unittest.main()
