"""Reference invariants, channel mirror symmetry and public-input checks."""
# ruff: noqa: E402 -- path and BLAS setup must precede numerical imports.

import copy
import json
import sys
import unittest
from pathlib import Path

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
            "g_GHz": 0.0025,
        }
        result = evaluate_ec(payload)
        self.assertEqual(result["status"], "calibrated")
        self.assertEqual(result["logical_source"], "10")
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
        energies, unit, logical, target, coupling, score = layout(a, b, 0.005)
        self.assertAlmostEqual(collision_residual(a, b, "0020"), 0, places=8)
        self.assertAlmostEqual(energies[logical[1]] - energies[target], 0, places=8)
        self.assertAlmostEqual(
            abs(coupling * unit[logical[1], target]), 0.005, places=12
        )
        self.assertAlmostEqual(score["S1"], 0.0080940711241, places=8)
        self.assertTrue(np.allclose(unit, unit.T))
        # Reverse physical node order: A1,A2,B1,B2 -> B2,B1,A2,A1.
        ma = pair_model(ec[3], ec[2], fb, 0.09, 2)
        mb = pair_model(ec[1], ec[0], fa, 0.15)
        es, v, indices, x, j, mirrored = layout(ma, mb, 0.005, "0200")
        self.assertAlmostEqual(collision_residual(ma, mb, "0200"), 0, places=8)
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

    def test_calibration_preserves_independent_frequency_bounds(self):
        p = json.loads((ROOT / "configs/smoke.json").read_text())
        payload = {**p["task_defaults"], **p["tasks"][1]}
        payload["bare_f01_bounds_GHz"]["B1"] = [4.1, 4.8]
        payload["bare_f01_bounds_GHz"]["B2"] = [4.2, 4.9]
        result = evaluate_ec(payload)
        self.assertEqual(result["status"], "calibrated")
        final = result["final"]
        self.assertTrue(4.2 <= final["fB_GHz"] <= 4.8)
        self.assertLess(abs(final["phase_error_rad"]), 0.01)
        self.assertLess(final["maximum_logical_ratio"], 0.2)
        self.assertEqual(result["duration_window_ns"], [70.0, 130.0])
        self.assertAlmostEqual(
            final["raw_fidelity"] / final["acceptance"],
            final["postselected_fidelity"],
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
