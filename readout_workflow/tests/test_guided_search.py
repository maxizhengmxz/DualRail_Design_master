import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from guided_search import PARAMETERS, guided_search, predict_resonances, trial_model
from model import Basis, Device, ReadoutModel, _transmon, save_results


class GuidedSearchTests(unittest.TestCase):
    def setUp(self):
        self.basis = Basis(ncut=10, transmon_levels=4, resonator_levels=4, eigenlevels=16)
        self.device = replace(Device(), C2_shunt_fF=Device().C1_shunt_fF)
        self.base = ReadoutModel(self.device, self.basis)

    def test_every_supported_parameter_predicts_symmetric_resonance(self):
        bounds = {"EJ2_GHz": (10, 30), "C1_shunt_fF": (30, 90),
                  "C2_shunt_fF": (30, 90), "C1r_fF": (5, 30), "C2r_fF": (5, 30)}
        for parameter in PARAMETERS:
            with self.subTest(parameter=parameter):
                roots, status = predict_resonances(self.base, 20, parameter, bounds[parameter])
                expected = 20 if parameter == "EJ2_GHz" else getattr(self.device, parameter)
                self.assertEqual(status, "predicted")
                self.assertEqual(len(roots), 1)
                self.assertAlmostEqual(roots[0], expected, places=5)

    def test_capacitance_trial_updates_both_charging_energies(self):
        changed, _ = trial_model(self.base, "C1r_fF", 25, 20)
        self.assertNotEqual(changed.ec_ghz[0], self.base.ec_ghz[0])
        self.assertNotEqual(changed.ec_ghz[1], self.base.ec_ghz[1])
        self.assertEqual(self.base.device.C1r_fF, 15)

    def test_no_resonance_produces_diagnostics_and_savable_empty_result(self):
        points, scan, diagnostics = guided_search(self.base, [20], "EJ2_GHz", (1, 2))
        self.assertEqual(points, [])
        self.assertEqual(scan, [])
        self.assertIn("no bracketed resonance", diagnostics[0]["status"])
        with tempfile.TemporaryDirectory() as tmp:
            path = save_results(Path(tmp)/"empty.npz", self.base, points, scan, {}, diagnostics)
            with np.load(path) as result:
                self.assertEqual(result["points"].shape, (0, 0))
                self.assertEqual(len(json.loads(str(result["metadata"]))["diagnostics"]), 1)

    def test_fractional_neighborhood_finds_crossing(self):
        base = ReadoutModel(replace(self.device, C1r_fF=0, C2r_fF=0), self.basis)
        points, scan, diagnostics = guided_search(base, [12, 14], "EJ2_GHz", (5, 25), samples=9)
        self.assertEqual(len(points), 2)
        self.assertEqual(len(scan), 18)
        for point in points:
            self.assertAlmostEqual(point["sweep_value"], point["ej1_ghz"], delta=0.05)
        self.assertTrue(all(d["accepted"] == 1 for d in diagnostics))

    def test_absolute_step_and_validation(self):
        _, scan, _ = guided_search(self.base, [20], "EJ2_GHz", (10, 30),
                                   mode="absolute", absolute_half_width=1, step=0.5)
        np.testing.assert_allclose([p["sweep_value"] for p in scan], [19, 19.5, 20, 20.5, 21], atol=1e-7)
        with self.assertRaises(ValueError):
            guided_search(self.base, [20], "C12_fF", (1, 3))


if __name__ == "__main__":
    unittest.main()
