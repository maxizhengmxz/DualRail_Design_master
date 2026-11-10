"""Physical invariants for the constant-impedance cavity-frequency sweep."""

import unittest
from copy import deepcopy

import numpy as np

from readout_workflow.model import ReadoutModel
from readout_workflow.resonator_frequency_tolerance import device_at_frequency


class ConstantImpedanceTests(unittest.TestCase):
    def setUp(self):
        self.ro = {
            "case": {"resonator_node": "r"},
            "physical": {
                "pad_pairs": {"A1": ["a", "b"], "A2": ["c", "d"]},
                "ground_capacitances_ff": {"a": 20.0, "b": 20.0, "c": 60.0, "d": 60.0},
                "pad_mutual_capacitances_ff": {"A1": 100.0, "A2": 60.0},
                "physical_links": [
                    {"from": "b", "to": "c", "capacitance_ff": 15.0},
                    {"from": "b", "to": "r", "capacitance_ff": 7.0},
                    {"from": "c", "to": "r", "capacitance_ff": 7.0},
                ],
                "resonator_self_capacitance_ff": 530.0,
                "resonator_inductance_nh": 1.3,
            },
        }

    def test_frequency_impedance_and_unchanged_other_capacitors(self):
        original = deepcopy(self.ro)
        for frequency in [5.0, 6.0, 7.0]:
            device, p = device_at_frequency(self.ro, frequency)
            cr = p["resonator_self_capacitance_ff"] * 1e-15
            lr = p["resonator_inductance_nh"] * 1e-9
            self.assertAlmostEqual(np.sqrt(lr / cr), 50.0, places=11)
            self.assertAlmostEqual(
                1 / (2 * np.pi * np.sqrt(lr * cr)) / 1e9, frequency, places=11
            )
            for name in [
                "pad_pairs",
                "ground_capacitances_ff",
                "pad_mutual_capacitances_ff",
                "physical_links",
            ]:
                self.assertEqual(p[name], original["physical"][name])
        self.assertEqual(self.ro, original)

    def test_loading_is_recomputed_without_ec_compensation(self):
        low, _ = device_at_frequency(self.ro, 5.0)
        high, _ = device_at_frequency(self.ro, 7.0)
        a, b = ReadoutModel(low), ReadoutModel(high)
        self.assertGreater(np.max(np.abs(a.ec_ghz[:2] - b.ec_ghz[:2])), 1e-7)
        self.assertGreater(np.max(np.abs(a.J_ghz - b.J_ghz)), 1e-7)


class MatchingBoundaryRegression(unittest.TestCase):
    def test_guarded_candidate_is_not_discarded_after_failed_edge_audit(self):
        import json

        from readout_workflow import calculation as study
        from readout_workflow.resonator_operation_map import solve_pixel

        original_conditions, original_baseline = study.CONDITIONS, study.BASELINE.copy()
        try:
            readout = json.loads(
                '{"case": {"name": "B", "nodes": ["B1", "B2"], "pad_pairs": {"B1": ["B11", "B12"], "B2": ["B21", "B22"]}, "loaded_ec_ghz": [0.16, 0.26], "resonator_node": "r2", "bare_frequency_guide_ghz": 5.39, "readout_target_ghz": 6.01, "isolated_split_mhz": 90, "dressed_split_window_ghz": [0.08, 0.1]}, "physical": {"pad_pairs": {"B1": ["B11", "B12"], "B2": ["B21", "B22"]}, "ground_capacitances_ff": {"B11": 24.212786655823905, "B12": 24.212786655823905, "B21": 126.65149943046342, "B22": 126.65149943046342}, "pad_mutual_capacitances_ff": {"B1": 106.49423982708973, "B2": 8.400547792031494}, "physical_links": [{"from": "B12", "to": "B21", "capacitance_ff": 8.35921657343108}, {"from": "B12", "to": "r2", "capacitance_ff": 4.527793627363148}, {"from": "B21", "to": "r2", "capacitance_ff": 4.527793627363148}], "resonator_self_capacitance_ff": 525.5519041730017, "resonator_inductance_nh": 1.313879760432504, "nominal_unloaded_impedance_ohm": 50, "effective_device_for_exact_equivalent_H": {"C1_shunt_fF": 117.67456349346588, "C2_shunt_fF": 70.49703106658428, "C12_fF": 1.5774032032151373, "C1r_fF": 1.852139323071599, "C2r_fF": 2.4585328813578227, "Cr_fF": 529.8625763774312, "Lr_nH": 1.313879760432504}, "note": "Effective modal capacitors are not fabricated capacitor values. Each readout physical link appears exactly once."}}'
            )
            row = solve_pixel(dict(readout=readout, fr=5.45, operation=4.97))
            self.assertEqual(row["status"], "validated_pass")
            self.assertFalse(study.failures(row["point"], require_sign_check=True))
            self.assertLess(abs(row["point"]["operating_frequency_ghz"] - 4.97), 1e-6)
        finally:
            study.CONDITIONS, study.BASELINE = original_conditions, original_baseline


if __name__ == "__main__":
    unittest.main()
