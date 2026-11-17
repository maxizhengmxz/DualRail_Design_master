import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from readout_workflow import fine_readout_feasibility as fine
from readout_workflow.fine_readout_feasibility import logical_range_ghz


class FineReadoutFrequencyTests(unittest.TestCase):
    def test_background_preserves_explicit_mhz_and_feasibility_mode(self):
        with (
            TemporaryDirectory() as directory,
            patch(
                "sys.argv",
                [
                    "fine_readout_feasibility.py",
                    "--run",
                    "--background",
                    "--output",
                    directory,
                    "--logical-frequency-mhz",
                    "80",
                    "100",
                    "--abs-mean-chi-min-mhz",
                    "5",
                    "--bare-resonator-ghz",
                    "5.95",
                    "--capacitance-step-ff",
                    "2",
                    "--feasibility-only",
                ],
            ),
            patch.object(fine.subprocess, "Popen") as popen,
            patch("builtins.print"),
        ):
            popen.return_value.pid = 123
            fine.main()
            command = popen.call_args.args[0]
            self.assertEqual(command[0], fine.sys.executable)
            self.assertIn("--feasibility-only", command)
            self.assertEqual(
                command[command.index("--abs-mean-chi-min-mhz") + 1], "5.0"
            )
            self.assertEqual(
                command[command.index("--bare-resonator-ghz") + 1], "5.95"
            )
            self.assertEqual(
                command[command.index("--capacitance-step-ff") + 1], "2"
            )
            launch = fine.json.loads((Path(directory) / "launch.json").read_text())
            self.assertEqual(launch["logical_frequency_ghz"], [0.08, 0.1])
            self.assertEqual(launch["abs_mean_chi_min_mhz"], 5.0)
            self.assertEqual(launch["bare_resonator_ghz"], 5.95)
            self.assertEqual(launch["capacitance_step_ff"], 2)
            self.assertFalse(launch["failure_diagnosis_enabled"])

    def test_cli_converts_mhz_before_run(self):
        with (
            patch(
                "sys.argv",
                [
                    "fine_readout_feasibility.py",
                    "--run",
                    "--logical-frequency-mhz",
                    "80",
                    "100",
                    "--abs-mean-chi-min-mhz",
                    "5",
                    "--feasibility-only",
                ],
            ),
            patch.object(fine, "run") as run,
        ):
            fine.main()
        self.assertEqual(
            run.call_args.args[2:], ((0.08, 0.1), True, 5.0, 6.0, 1)
        )

    def test_ghz_initialization_scales_cold_seeds_once_and_preserves_warm(self):
        with (
            patch.object(fine.s, "CONDITIONS", fine.s.CONDITIONS),
            patch.object(fine.s, "BASELINE", fine.s.BASELINE),
            patch.object(fine.s, "seed_candidates", fine.s.seed_candidates),
            patch.object(
                fine, "_BASE_SEEDS", return_value=[np.array([4.5, 0.0, 2.0, 3.0])]
            ),
        ):
            fine.initialize((0.14, 0.16), 5.0)
            self.assertAlmostEqual(fine.s.seed_candidates(100, 75)[0][2], 3.0)
            self.assertEqual(fine.s.CONDITIONS.abs_mean_chi_min_mhz, 5.0)
            fine.initialize((0.08, 0.1))
            warm = [4.6, 0.1, 4.0, 5.0]
            seeds = fine.s.seed_candidates(100, 75, [{"x": warm}])
            np.testing.assert_array_equal(seeds[0], warm)
            self.assertAlmostEqual(seeds[1][2], 1.8)
            self.assertEqual(fine.s.CONDITIONS.logical_frequency_ghz, (0.08, 0.1))

    def test_frequency_interval_conversion(self):
        self.assertEqual(logical_range_ghz((80, 100)), (0.08, 0.1))
        self.assertEqual(logical_range_ghz((140, 160)), (0.14, 0.16))

    def test_frequency_interval_rejects_invalid_bounds(self):
        for bounds in ((100, 80), (0, 100), (100, 100)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                logical_range_ghz(bounds)

    def test_chi_minimum_rejects_invalid_values(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fine.initialize((0.14, 0.16), value)

    def test_bare_resonator_frequency_is_explicit(self):
        with patch.object(fine.s, "BARE_RESONATOR_GHZ", 6.0):
            fine.initialize((0.14, 0.16), 1.0, 5.95)
            self.assertEqual(fine.s.BARE_RESONATOR_GHZ, 5.95)

    def test_capacitance_step_is_validated(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                fine.run(Path(directory), 1, capacitance_step_ff=0)


if __name__ == "__main__":
    unittest.main()
