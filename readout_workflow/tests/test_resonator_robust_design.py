"""Range coverage and fixed-fabrication invariants for robust design."""

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from readout_workflow import calculation as s
from readout_workflow.ec_resonator_tolerance import at_frequency
from readout_workflow.resonator_robust_design import (
    design_device,
    frequency_grid,
    point_slacks,
)
from readout_workflow.validation import INTERMEDIATE


class RobustRangeTests(unittest.TestCase):
    def test_grid_includes_endpoints_reference_without_overshoot(self):
        f = frequency_grid(5.93, 6.12, 0.05)
        self.assertEqual(f[0], 5.93)
        self.assertEqual(f[-1], 6.12)
        self.assertIn(6.0, f)
        self.assertTrue(np.all(np.diff(f) > 0))
        np.testing.assert_array_equal(frequency_grid(6, 6, 0.05), [6.0])
        for low, high in [(6.1, 6.3), (5.0, 5.9)]:
            with self.assertRaises(ValueError):
                frequency_grid(low, high, 0.05)

    def test_one_capacitor_design_not_resynthesized_at_every_frequency(self):
        # Covers EC synthesis, equal links, fixed shared hardware, impedance,
        # input immutability and uncompensated loading in one physics check.
        for c12, cg in [(1.5, 2.5), (1.7, 6.8), (2.1, 5.2)]:
            with self.subTest(C12=c12, Cg=cg):
                d = design_device([200, 238], c12, cg)
                nominal = s.ReadoutModel(d, INTERMEDIATE)
                np.testing.assert_allclose(
                    nominal.ec_ghz[:2] * 1000, [200, 238], atol=1e-9
                )
                self.assertEqual(d.C1r_fF, d.C2r_fF)
                original = asdict(d)
                for f in (5.0, 6.0, 7.0):
                    changed = at_frequency(original, f)
                    for key in s.CAPS:
                        self.assertEqual(getattr(changed, key), getattr(d, key))
                    self.assertAlmostEqual(
                        np.sqrt(changed.Lr_nH * 1e-9 / (changed.Cr_fF * 1e-15)),
                        50.0,
                        places=10,
                    )
                    self.assertAlmostEqual(
                        1
                        / (
                            2
                            * np.pi
                            * np.sqrt(changed.Lr_nH * 1e-9 * changed.Cr_fF * 1e-15)
                        )
                        / 1e9,
                        f,
                        places=10,
                    )
                self.assertEqual(asdict(d), original)
                self.assertGreater(
                    np.max(
                        abs(
                            s.ReadoutModel(changed, INTERMEDIATE).ec_ghz[:2]
                            - nominal.ec_ghz[:2]
                        )
                    ),
                    1e-7,
                )

    def test_original_ec_policy_is_used_without_coupler_exception(self):
        from readout_workflow.ec_resonator_tolerance import set_policy

        with (
            patch.object(s, "CONDITIONS", s.CONDITIONS),
            patch.object(s, "BASELINE", s.BASELINE),
        ):
            set_policy(
                asdict(
                    s.Conditions(logical_frequency_ghz=(0.08, 0.1), ej_over_ec_min=50.0)
                )
            )
            self.assertEqual(s.BASELINE["ej_ec"], 50)
            self.assertEqual(s.BASELINE["logical_frequency"], (0.08, 0.1))
            self.assertFalse(s.BASELINE["fixed"])
            self.assertEqual(s.CONDITIONS.fixed_frequency_tolerance_ghz, 1e-7)

    def test_ec_constraints_remain_active_at_swept_frequencies(self):
        from readout_workflow.direct_ec_feasibility import point

        p = point(200, 238, [4.8, 0.3, 1.8, 6.4], INTERMEDIATE)
        original = point_slacks(p, 0.95)
        p = dict(p, loaded_ec1_mhz=149.0)
        self.assertLess(point_slacks(p, 0.95)[-4], 0)
        self.assertGreaterEqual(original[-4], 0)


class ObjectivePolicyTests(unittest.TestCase):
    def test_cached_point_with_failures_does_not_certify_range(self):
        from readout_workflow.resonator_robust_design import validate_range

        point = dict(device=asdict(s.Device()))
        solution = dict(frequencies_ghz=[6.0], points=[point], worst_mean_chi_mhz=1.2)
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "frequency_6.0000000000.json").write_text(
                json.dumps(dict(frequency_ghz=6.0, point=point, failures=["H_target"]))
            )
            result = validate_range(solution, [200, 238], 0.96, 6.0, 6.0, 0.05, folder)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_frequencies_ghz"], [6.0])

    def test_exploration_has_no_extra_chi_or_h_objective(self):
        from readout_workflow.resonator_robust_design import (
            selected_h_targets,
            validation_chi_floor,
        )

        self.assertEqual(selected_h_targets("tolerance", None, 0.95), [0.95])
        self.assertEqual(selected_h_targets("tolerance", [0.95], 0.95), [0.95])
        with self.assertRaises(ValueError):
            selected_h_targets("tolerance", [0.95, 0.98], 0.95)
        self.assertEqual(
            validation_chi_floor({"worst_mean_chi_mhz": 1.8}, False),
            s.CONDITIONS.abs_mean_chi_min_mhz,
        )

    def test_prescribed_range_retains_worst_chi_optimization(self):
        from readout_workflow.resonator_robust_design import (
            selected_h_targets,
            validation_chi_floor,
        )

        self.assertEqual(
            selected_h_targets("prescribed", None, 0.95), [0.95, 0.97, 0.98]
        )
        self.assertAlmostEqual(
            validation_chi_floor({"worst_mean_chi_mhz": 1.8}, True), 1.799
        )


class WorkflowRoutingTests(unittest.TestCase):
    """Exercise CLI routing without requiring ignored historical study data."""

    def exercise(self, mode):
        from readout_workflow import resonator_robust_design as workflow

        point = dict(
            device=asdict(s.Device()),
            loaded_ec1_mhz=200.0,
            loaded_ec2_mhz=238.0,
        )
        certified = dict(
            status="validated_sampled_range",
            validation=dict(samples=[dict(frequency_ghz=6.0, point=point)]),
            worst_mean_chi_mhz=1.2,
            worst_H=0.96,
        )
        quick = dict(
            status="validated_sampled_range",
            range_ghz=[5.9, 6.1],
            boundary=dict(validated_halfwidth_ghz=0.1),
            validated_result=certified,
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            seed = root / "seed.json"
            conditions = root / "conditions.json"
            output = root / "output"
            seed.write_text(json.dumps(dict(point=point)), encoding="utf-8")
            conditions.write_text(
                json.dumps(dict(conditions=asdict(s.Conditions()))), encoding="utf-8"
            )
            argv = [
                "robust",
                "--run",
                "--mode",
                mode,
                "--ec",
                "200",
                "238",
                "--input",
                str(seed),
                "--conditions",
                str(conditions),
                "--output",
                str(output),
            ]
            if mode == "prescribed":
                argv += ["--resonator-range-ghz", "5.9", "6.1", "--h-targets", "0.95"]
            with (
                patch("sys.argv", argv),
                patch.object(s, "CONDITIONS", s.CONDITIONS),
                patch.object(s, "BASELINE", s.BASELINE),
                patch.object(workflow, "quick_tolerance", return_value=quick) as search,
                patch.object(workflow, "run_range", return_value=certified) as strength,
            ):
                workflow.main()
            result = json.loads((output / "results.json").read_text())[0]["result"]
            return search.call_count, strength.call_count, result

    def test_exploration_never_calls_secondary_strength_optimizer(self):
        searches, optimizations, result = self.exercise("tolerance")
        self.assertEqual(searches, 1)
        self.assertEqual(optimizations, 0)
        self.assertFalse(result["strength_optimized"])

    def test_prescribed_mode_calls_strength_optimizer(self):
        searches, optimizations, result = self.exercise("prescribed")
        self.assertEqual(searches, 0)
        self.assertEqual(optimizations, 1)
        self.assertTrue(result["strength_optimized"])


if __name__ == "__main__":
    unittest.main()
