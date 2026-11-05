"""Target-domain selection for the direct loaded-EC feasibility driver."""

import json
import unittest
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from readout_workflow import direct_ec_feasibility as direct
from readout_workflow.direct_ec_feasibility import (
    logical_range_ghz,
    mirrored_target_grid,
    target_grid,
)


class DirectECTargetGridTests(unittest.TestCase):
    def test_fractional_steps_are_not_truncated(self):
        targets = target_grid(
            [(150, 200), (155, 200), (150, 205)], "rectangle", step_mhz=2.5
        )
        self.assertIn((152.5, 202.5), targets)
        self.assertEqual(len(targets), 9)

    def test_frequency_task_units_are_explicit_and_compatible(self):
        with (
            patch.object(direct.s, "CONDITIONS", direct.s.CONDITIONS),
            patch.object(direct.s, "BASELINE", direct.s.BASELINE),
        ):
            for key, bounds in [
                ("logical_frequency_mhz", [80, 100]),
                ("logical_frequency_ghz", [0.08, 0.1]),
            ]:
                result = direct.solve(dict(ec=(165, 250), warm=[], **{key: bounds}))
                self.assertEqual(result["status"], "no_validated_solution_found")
                self.assertEqual(direct.s.CONDITIONS.logical_frequency_ghz, (0.08, 0.1))
            with self.assertRaisesRegex(ValueError, "Conflicting MHz and GHz"):
                direct.solve(
                    dict(
                        ec=(165, 250),
                        warm=[],
                        logical_frequency_mhz=[140, 160],
                        logical_frequency_ghz=[0.08, 0.1],
                    )
                )

    def test_cli_preserves_explicit_ranges_and_full_mirrored_domain(self):
        cases = [
            (
                [
                    "--ec1-range",
                    "150",
                    "152.5",
                    "--ec2-range",
                    "200",
                    "202.5",
                    "--ec-step-mhz",
                    "2.5",
                ],
                False,
            ),
            (["--orientation-domain", "full", "--hull-scale", "1.5"], True),
        ]
        for options, mirrored in cases:
            with self.subTest(options=options), TemporaryDirectory() as directory:
                source = Path(directory) / "source.json"
                witnesses = []
                for ec1, ec2 in ((150, 200), (160, 200), (150, 210)):
                    c1, c2 = direct.capacitances(ec1, ec2, 5, 3)
                    witnesses.append(
                        dict(
                            status="validated_pass",
                            point=dict(
                                loaded_ec1_mhz=ec1,
                                loaded_ec2_mhz=ec2,
                                bare_f1_ghz=4.6,
                                bare_f2_ghz=4.6,
                                device=dict(
                                    C1_shunt_fF=c1, C2_shunt_fF=c2, C12_fF=5, C1r_fF=3
                                ),
                            ),
                        )
                    )
                source.write_text(json.dumps(witnesses))
                output = Path(directory) / "output"
                tasks = []

                class Pool:
                    def __init__(self, submitted):
                        self.submitted = submitted

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass

                    def submit(self, worker, task):
                        self.submitted.append(task)
                        future = Future()
                        future.set_result(
                            dict(
                                ec1_target_mhz=task["ec"][0],
                                ec2_target_mhz=task["ec"][1],
                                status="no_validated_solution_found",
                                seconds=0,
                            )
                        )
                        return future

                arguments = [
                    "direct_ec_feasibility.py",
                    "--run",
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    *options,
                ]
                with (
                    patch("sys.argv", arguments),
                    patch.object(
                        direct, "ProcessPoolExecutor", return_value=Pool(tasks)
                    ),
                    patch.object(direct, "OUT", direct.OUT),
                    patch.object(direct.s, "CONDITIONS", direct.s.CONDITIONS),
                    patch.object(direct.s, "BASELINE", direct.s.BASELINE),
                    patch("builtins.print"),
                ):
                    direct.main()
                targets = {tuple(task["ec"]) for task in tasks}
                if mirrored:
                    self.assertIn((200.0, 150.0), targets)
                    self.assertIn((150.0, 200.0), targets)
                else:
                    self.assertEqual(
                        targets,
                        {
                            (150.0, 200.0),
                            (150.0, 202.5),
                            (152.5, 200.0),
                            (152.5, 202.5),
                        },
                    )
                self.assertEqual(len(tasks), len(targets))
                self.assertTrue(
                    all(
                        task["logical_frequency_ghz"] == (0.14, 0.16)
                        and "logical_frequency_mhz" not in task
                        for task in tasks
                    )
                )
                settings = json.loads((output / "settings.json").read_text())
                self.assertEqual(settings["readout_label_swap_symmetry"], mirrored)

    def test_mirroring_changes_seed_coordinates_not_saved_observables(self):
        witness = dict(
            bare_f1_ghz=4.7,
            bare_f2_ghz=4.6,
            device=dict(C12_fF=5.0, C1r_fF=3.0),
            ratio=0.003,
        )
        seed = direct.mirrored_warm_point(witness)
        self.assertAlmostEqual(seed["x"][0], 4.65)
        self.assertAlmostEqual(seed["x"][1], -10.0)
        self.assertNotIn("ratio", seed)
        self.assertEqual(witness["bare_f1_ghz"], 4.7)

    def test_rectangle_keeps_points_outside_triangular_hull(self):
        witnesses = np.array([(150, 200), (160, 200), (150, 210)], float)
        rectangle = target_grid(witnesses, "rectangle")
        hull = target_grid(witnesses, "hull")
        self.assertEqual(len(rectangle), 9)
        self.assertEqual(len(hull), 6)
        self.assertIn((160, 210), rectangle)
        self.assertNotIn((160, 210), hull)

    def test_invalid_domain_is_rejected(self):
        with self.assertRaises(ValueError):
            target_grid(np.array([(150, 200), (160, 200), (150, 210)]), "unknown")

    def test_scaled_hull_expands_and_respects_ec_limits(self):
        witnesses = np.array([(170, 220), (210, 220), (170, 260)], float)
        original = set(target_grid(witnesses, "hull", hull_scale=1.0))
        expanded = set(target_grid(witnesses, "hull", hull_scale=1.5))
        self.assertTrue(original < expanded)
        self.assertTrue(
            all(150 <= value < 400 for point in expanded for value in point)
        )

    def test_finer_step_increases_target_density(self):
        witnesses = np.array([(170, 220), (210, 220), (170, 260)], float)
        coarse = target_grid(witnesses, "hull", step_mhz=5, hull_scale=1.5)
        fine = target_grid(witnesses, "hull", step_mhz=2, hull_scale=1.5)
        self.assertGreater(len(fine), len(coarse))
        self.assertTrue(all(value % 2 == 0 for point in fine for value in point))

    def test_hull_scale_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            target_grid(
                np.array([(150, 200), (160, 200), (150, 210)]), "hull", hull_scale=0.9
            )

    def test_full_orientation_is_union_not_combined_convex_hull(self):
        witnesses = np.array([(155, 210), (165, 210), (155, 230)], float)
        targets, original, mirrored = mirrored_target_grid(witnesses, hull_scale=1.5)
        self.assertEqual(set(targets), original | mirrored)
        self.assertIn((155, 210), original)
        self.assertIn((210, 155), mirrored)
        self.assertIn((155, 210), targets)
        self.assertIn((210, 155), targets)

    def test_logical_frequency_interval_conversion(self):
        self.assertEqual(logical_range_ghz((80, 100)), (0.08, 0.1))
        for invalid in ((100, 80), (0, 100), (90, 90)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                logical_range_ghz(invalid)


if __name__ == "__main__":
    unittest.main()
