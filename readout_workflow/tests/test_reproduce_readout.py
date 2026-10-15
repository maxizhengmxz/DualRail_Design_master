"""Regression checks for the consolidated scientific workflow."""

import ast
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from readout_workflow import reproduce_readout as study


class StandaloneReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.designs = study.selected_designs()

    def test_selected_circuits_and_inverse_capacitance(self):
        for point in self.designs.values():
            self.assertEqual(study.failures(point), [])
            device = study.Device(**point["device"])
            matrix = device.capacitance_matrix()
            self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0))
            expected_ec = (
                study.e**2 * np.diag(np.linalg.inv(matrix)) / (2 * study.h * 1e9)
            )
            np.testing.assert_allclose(
                expected_ec[:2] * 1000,
                [point["loaded_ec1_mhz"], point["loaded_ec2_mhz"]],
            )

    def test_transmon_exchange_symmetry(self):
        point = self.designs["2"]
        device = study.Device(**point["device"])
        mirror = replace(
            device, C1_shunt_fF=device.C2_shunt_fF, C2_shunt_fF=device.C1_shunt_fF
        )
        with threadpool_limits(limits=1):
            reflected = study.annotate(
                study.ReadoutModel(mirror, study.VALIDATION_BASIS),
                point["ej2_ghz"],
                point["ej1_ghz"],
            )
        for key in (
            "ratio",
            "hybridization",
            "lambda_max",
            "splitting_ghz",
            "operating_frequency_ghz",
        ):
            self.assertAlmostEqual(point[key], reflected[key], delta=2e-8)

    def test_resonator_error_keeps_impedance_and_breaks_equal_caps(self):
        point = self.designs["1"]
        perturbed = study.perturb(point, [0.02, -0.02, 0.02, -0.02, 0.02, 0.02])
        frequency = (
            1
            / (2 * np.pi * np.sqrt(perturbed.Cr_fF * 1e-15 * perturbed.Lr_nH * 1e-9))
            / 1e9
        )
        impedance = np.sqrt(perturbed.Lr_nH * 1e-9 / (perturbed.Cr_fF * 1e-15))
        self.assertAlmostEqual(frequency, 6.12)
        self.assertAlmostEqual(impedance, 50)
        self.assertNotEqual(perturbed.C1r_fF, perturbed.C2r_fF)
        self.assertAlmostEqual(
            perturbed.C1_shunt_fF, point["device"]["C1_shunt_fF"] * 1.02
        )

    def test_rescue_order_and_strict_energy_bounds(self):
        point = dict(
            self.designs["2"],
            loaded_ec1_mhz=130,
            splitting_ghz=0.085,
            hybridization=0.92,
        )
        target = point["operating_frequency_ghz"]
        self.assertIn("EC_range", study.reasons(point, study.TIERS[0], target))
        self.assertNotIn("EC_range", study.reasons(point, study.TIERS[1], target))
        self.assertIn("logical_frequency", study.reasons(point, study.TIERS[2], target))
        self.assertNotIn(
            "logical_frequency", study.reasons(point, study.TIERS[3], target)
        )
        self.assertIn("hybridization", study.reasons(point, study.TIERS[3], target))
        self.assertEqual(study.reasons(point, study.TIERS[4], target), [])
        self.assertIn(
            "EC_range", study.failures(dict(self.designs["2"], loaded_ec2_mhz=400))
        )
        self.assertIn("EJ_limit", study.failures(dict(self.designs["2"], ej1_ghz=25)))
        self.assertIn(
            "EJ_over_EC", study.failures(dict(self.designs["2"], ej1_over_ec1=50))
        )

    def test_sampling_matches_archive(self):
        path = study.DEFAULT_DATA_DIR / study.ROBUSTNESS_ARCHIVE / "sampling.json"
        self.assertEqual(study.sampling_scenarios(), study.load_json(path)["scenarios"])
        scenarios = study.sampling_scenarios()
        self.assertEqual(len(scenarios), 300)
        self.assertEqual(sum(row["kind"] == "corner" for row in scenarios), 64)

    def test_standalone_source_has_no_local_code_imports(self):
        tree = ast.parse(Path(study.__file__).read_text(encoding="utf-8"))
        local_names = {p.stem for p in Path(study.__file__).parent.glob("*.py")}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                self.assertNotIn((node.module or "").split(".")[0], local_names)
            elif isinstance(node, ast.Import):
                for name in node.names:
                    self.assertNotIn(name.name.split(".")[0], local_names)


if __name__ == "__main__":
    unittest.main()
