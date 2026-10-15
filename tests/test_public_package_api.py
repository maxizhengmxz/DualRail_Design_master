import os
import unittest

import dtc_dualrail as dtc
from dtc_dualrail.circuits import gen_dtc
from dtc_dualrail.io import save_dataset
from dtc_dualrail.monte_carlo import run_monte_carlo
from dtc_dualrail.runtime import THREAD_ENV_VARS, configure_single_threaded_blas
from dtc_dualrail.sampling import _sample_parameters
from dtc_dualrail.tracking import track_levels_by_overlap
from dtc_dualrail.tunable_resonator import evaluate_dispersive_shifts


class PublicPackageApiTests(unittest.TestCase):
    def test_top_level_exports_core_workflows(self):
        for name in (
            "gen_dtc",
            "gen_dt",
            "find_operating_points",
            "load_or_compute_operating_points",
            "build_logical_contour",
            "calibrate_reduced_pair",
            "build_dualrail_fsim_model",
            "DualRailPulse",
            "DualRailPulseConstraints",
            "simulate_dualrail_pulse",
            "extract_fsim_angles",
            "run_static_interaction_landscape",
            "load_or_compute_static_interaction_landscape",
            "run_operating_point_dispersive_sweep",
            "load_or_compute_operating_point_dispersive_sweep",
            "track_levels_by_overlap",
            "run_monte_carlo",
            "run_tunable_resonator_monte_carlo",
            "run_tunable_resonator_sample_sweep",
            "run_two_param_sweep",
            "save_dataset",
        ):
            self.assertTrue(hasattr(dtc, name), name)

    def test_core_helpers_are_owned_by_package_modules(self):
        self.assertEqual(gen_dtc.__module__, "dtc_dualrail.circuits")
        self.assertEqual(
            track_levels_by_overlap.__module__,
            "dtc_dualrail.tracking",
        )
        self.assertEqual(_sample_parameters.__module__, "dtc_dualrail.sampling")
        self.assertEqual(save_dataset.__module__, "dtc_dualrail.io")
        self.assertEqual(run_monte_carlo.__module__, "dtc_dualrail.monte_carlo")
        self.assertEqual(
            evaluate_dispersive_shifts.__module__,
            "dtc_dualrail.tunable_resonator",
        )

    def test_runtime_helper_sets_thread_environment(self):
        configure_single_threaded_blas("2")
        try:
            for env_var in THREAD_ENV_VARS:
                with self.subTest(env_var=env_var):
                    self.assertEqual(os.environ[env_var], "2")
        finally:
            configure_single_threaded_blas()


if __name__ == "__main__":
    unittest.main()
