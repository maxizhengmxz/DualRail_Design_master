"""Readout chi objective, audited H fallback and integration rejection policy."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from floating_coupler.readout_optimization import maximize_chi, readout_handoff


class ReadoutStrengthOptimizationTests(unittest.TestCase):
    def run_search(self, screen_h=0.97, audit_h=None, reject_strong=False):
        def evaluate(x):
            return dict(hybridization=screen_h, chi=-float(x[0]))

        def audit(x):
            point = evaluate(x)
            point["hybridization"] = screen_h if audit_h is None else audit_h
            point["rejected"] = reject_strong and x[0] > 1.5
            return dict(point=point, physical={"fixed_ec": [0.165, 0.250]})

        return maximize_chi(
            [1.0, 0.0],
            [],
            ([0.0, -1.0], [2.0, 1.0]),
            evaluate,
            lambda x: np.array([x[1]]),
            lambda x, floor: np.array([screen_h - floor - 1e-4]),
            audit,
            lambda point: ["chi_sign_stability"] if point["rejected"] else [],
            lambda point: abs(point["chi"]),
        )

    def test_maximizes_absolute_chi_with_primary_H_margin(self):
        result = self.run_search()
        self.assertEqual(result["status"], "validated_pass")
        self.assertAlmostEqual(result["coordinates"][0], 2.0, places=6)
        self.assertEqual(result["selected_H_floor"], 0.96)
        self.assertFalse(result["fallback_used"])
        self.assertIsNone(result["warning"])
        self.assertEqual(len(result["tiers"]), 1)

    def test_failed_primary_search_falls_back_and_warns(self):
        result = self.run_search(screen_h=0.955)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["selected_H_floor"], 0.95)
        self.assertIn("H > 96%", result["warning"])
        self.assertAlmostEqual(result["coordinates"][0], 2.0, places=6)

    def test_primary_must_pass_H_at_audit_basis(self):
        for h in (0.955, 0.96):
            with self.subTest(audit_H=h):
                result = self.run_search(audit_h=h)
                self.assertEqual(result["selected_H_floor"], 0.95)
                self.assertTrue(result["fallback_used"])
                self.assertIn(
                    "requested_strict_H_floor",
                    result["tiers"][0]["audits"][0]["failures"],
                )

    def test_audit_rejection_keeps_valid_seed_instead_of_bad_stronger_point(self):
        result = self.run_search(reject_strong=True)
        self.assertEqual(result["status"], "validated_pass")
        self.assertEqual(result["coordinates"][0], 1.0)

    def test_both_tiers_fail_without_accepted_handoff(self):
        result = self.run_search(screen_h=0.94)
        self.assertEqual(result["status"], "no_validated_solution_found")
        self.assertIsNone(result["selected"])
        self.assertEqual(len(result["tiers"]), 2)
        with self.assertRaisesRegex(ValueError, "rejected standalone"):
            readout_handoff({"B": {"readout_strength_optimization": result}})

    def test_integration_retains_fallback_and_labels_legacy(self):
        result = dict(
            status="validated_pass",
            selected_H_floor=0.95,
            fallback_used=True,
            warning="96% not found; 95% fallback",
            scope="prescribed_range_validated",
            range_validation=dict(passed=True),
        )
        records = readout_handoff(
            {"A": {"readout_strength_optimization": result}, "B": {}}
        )
        self.assertTrue(records["A"]["fallback_used"])
        self.assertEqual(records["A"]["warning"], result["warning"])
        self.assertEqual(
            records["B"]["optimization_status"], "legacy_no_optimization_record"
        )
        with self.assertRaises(ValueError):
            readout_handoff({"A": {"workflow_failures": ["chi_matching"]}})
        with self.assertRaisesRegex(ValueError, "nominal-frequency initialization"):
            readout_handoff({"A": {"readout_strength_optimization": self.run_search()}})
        # Earlier nominal-only records predate the explicit scope field.
        with self.assertRaisesRegex(ValueError, "nominal-frequency initialization"):
            readout_handoff(
                {
                    "A": {
                        "readout_strength_optimization": dict(
                            status="validated_pass",
                            objective="maximize_absolute_signed_mean_chi",
                        )
                    }
                }
            )


class PhysicalPadRangeTests(unittest.TestCase):
    def test_independent_range_inputs_precede_any_design_work(self):
        from floating_coupler.readout_range import case_for_pair
        from standalone_pad_readout import design

        request = dict(
            pairs={"A": {}, "B": {}},
            standalone_readout_optimization=dict(
                bare_resonator_frequency_range_ghz={"A": [5.3, 5.5], "B": [6.2, 6.4]}
            ),
        )
        self.assertEqual(
            case_for_pair(request, "A")["bare_resonator_frequency_range_ghz"],
            [5.3, 5.5],
        )
        self.assertEqual(
            case_for_pair(request, "B")["bare_resonator_frequency_range_ghz"],
            [6.2, 6.4],
        )
        self.assertEqual(request["pairs"]["A"], {})
        with patch("standalone_pad_readout.least_squares") as fit:
            with self.assertRaisesRegex(ValueError, "Supply bare_resonator"):
                design({}, "unused-output.json")
            fit.assert_not_called()

    def test_actual_pad_layout_at_non6GHz_reference_and_fixed_scenarios(self):
        from floating_coupler.readout_range import PadRangeObjective
        from standalone_pad_readout import build_device, evaluate

        from core.transmon import Basis
        from readout_workflow import calculation as study

        case = dict(
            nodes=["A1", "A2"],
            pad_pairs={"A1": ["a", "b"], "A2": ["c", "d"]},
            loaded_ec_ghz=[0.165, 0.25],
            resonator_node="r",
            bare_resonator_frequency_range_ghz=[5.7, 5.9],
        )
        seed = np.array([4.61, 4.61, 16.0, 10.0, 5.8, 0.3, 0.65])
        basis = Basis(15, 4, 3, 16)
        model = PadRangeObjective(
            case,
            seed,
            (
                [4.58, 4.58, 0.1, 0.1, 5.75, 0.1, 0.1],
                [4.64, 4.64, 40, 25, 5.9, 0.85, 0.85],
            ),
            lambda x: evaluate(case, x, basis)[0],
            lambda x: np.zeros(3),
            lambda x: np.ones(1),
            lambda x: None,
            build_device,
            evaluate,
            basis,
        )
        device, physical = model.hardware(seed)
        original = deepcopy(physical)
        np.testing.assert_allclose(
            study.ReadoutModel(device, basis).ec_ghz[:2], [0.165, 0.25], atol=1e-12
        )
        low = model.point(seed, 5.7, [4.61, 4.61])
        high = model.point(seed, 5.9, [4.61, 4.61])
        self.assertGreater(abs(low["loaded_ec1_mhz"] - high["loaded_ec1_mhz"]), 1e-5)
        self.assertEqual(model.hardware(seed)[1], original)
        self.assertEqual(model.hardware_builds, 1)
        self.assertEqual(
            [e["capacitance_ff"] for e in physical["physical_links"][1:]], [10.0, 10.0]
        )
        self.assertNotEqual(device.C1r_fF, device.C2r_fF)
        self.assertNotIn(6.0, model.frequencies)
        z = model.start(seed)
        result = model.evaluate(z)
        self.assertEqual(len(result["points"]), len(model.frequencies))
        constraints = model.inequalities(z, 0.96)
        self.assertEqual(len(constraints), model.constraint_count)
        self.assertTrue(np.isfinite(constraints).all())

    def test_interior_refinement_precedes_H_fallback_and_preserves_nominal_handoff(
        self,
    ):
        from floating_coupler.readout_range import optimize_pad_range

        from core.transmon import Basis

        models = []

        def factory(*args, extra_frequencies=(), **kwargs):
            model = SimpleNamespace(
                low=5.7,
                high=5.9,
                step=0.2,
                frequencies=np.array(sorted({5.7, 5.9, *extra_frequencies})),
                failed_frequencies={5.8},
                hardware_builds=1,
                scenario_evaluations=2,
                bounds=(np.zeros(8), np.ones(8) * 10),
                start=lambda x: np.r_[x, 1.0],
                evaluate=None,
                equalities=None,
                inequalities=None,
                audit=None,
            )
            models.append(model)
            return model

        seed = [4.61, 4.61, 16.0, 10.0, 5.8, 0.3, 0.65]
        common = dict(
            seconds=0.1,
            warning=None,
            coordinates=seed + [1.0],
            scope="prescribed_range_validated",
            selected_H_floor=0.96,
            fallback_used=False,
        )
        failure = dict(
            common,
            status="no_validated_solution_found",
            selected=None,
            tiers=[dict(H_floor=0.96, audits=[dict(x=seed + [1.0])])],
        )
        selected = dict(
            nominal={"hybridization": 0.97},
            physical={"pad_model": "nominal"},
            range_validation=dict(passed=True, range_ghz=[5.7, 5.9]),
        )
        success = dict(
            common,
            status="validated_pass",
            selected=selected,
            tiers=[dict(H_floor=0.96, audits=[])],
        )
        with (
            patch(
                "floating_coupler.readout_range.PadRangeObjective", side_effect=factory
            ),
            patch(
                "floating_coupler.readout_range.maximize_chi",
                side_effect=[failure, success],
            ) as optimize,
        ):
            result = optimize_pad_range(
                {},
                seed,
                [],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Basis(15, 4, 3, 16),
            )
        self.assertEqual(optimize.call_count, 2)
        self.assertTrue(
            all(call.kwargs["h_floors"] == (0.96,) for call in optimize.call_args_list)
        )
        self.assertIn(5.8, models[1].frequencies)
        self.assertEqual(result["coordinates"], seed)
        self.assertEqual(result["selected"]["physical"], {"pad_model": "nominal"})
        self.assertEqual(result["range_ghz"], [5.7, 5.9])
        record = readout_handoff({"A": {"readout_strength_optimization": result}})["A"]
        self.assertEqual(record["range_ghz"], [5.7, 5.9])

    def test_interior_failure_prevents_a_range_certificate(self):
        from floating_coupler.readout_range import PadRangeObjective

        model = PadRangeObjective.__new__(PadRangeObjective)
        point = dict(hybridization=0.97, chi_m_ghz=0.0012, chi_p_ghz=0.0012)
        model.nominal_audit = lambda x: dict(point=point, physical={})
        model.nominal_equalities = lambda x: np.zeros(3)
        model.low, model.high, model.step = 5.7, 5.9, 0.2
        model.frequencies = np.array([5.7, 5.9])
        model.failed_frequencies = set()
        model.case = {"loaded_ec_ghz": [0.165, 0.25]}

        def validate(hardware, frequency, *args):
            failed = abs(frequency - 5.8) < 1e-8
            return dict(
                frequency_ghz=frequency,
                point=None if failed else point,
                failures=["interior"] if failed else [],
            )

        model.validate_frequency = validate
        x = np.r_[[4.61, 4.61, 16.0, 10.0, 5.8, 0.3, 0.65], [4.61, 4.61] * 2, 1.1]
        with patch("floating_coupler.readout_range.checked_sample", return_value=[]):
            result = model.audit(x, 0.96)
        self.assertFalse(result["range_validation"]["passed"])
        self.assertEqual(model.failed_frequencies, {5.8})
        self.assertTrue(result["point"]["failures"])

    def test_epigraph_optimizes_worst_chi_not_peak(self):
        def evaluate(x):
            return dict(hybridization=0.97, worst=min(1 + x[0], 3 - x[0]), failures=[])

        result = maximize_chi(
            [0.2, 0.0, 1.0],
            [],
            ([0.0, -1.0, 1.0], [2.0, 1.0, 3.0]),
            evaluate,
            lambda x: np.array([x[1]]),
            lambda x, floor: np.array(
                [1 + x[0] - x[-1], 3 - x[0] - x[-1], 0.97 - floor - 1e-4]
            ),
            lambda x, floor: dict(point=evaluate(x)),
            lambda p: p["failures"],
            lambda p: p["worst"],
            objective=lambda x: -x[-1],
            scope="prescribed_range_validated",
            audit_with_floor=True,
        )
        self.assertEqual(result["status"], "validated_pass")
        self.assertAlmostEqual(result["coordinates"][0], 1.0, places=6)
        self.assertAlmostEqual(result["coordinates"][-1], 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
