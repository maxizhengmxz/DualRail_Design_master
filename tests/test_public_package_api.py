import os
import unittest

import core as dtc
import logical_resonance_gate as gates
from core.circuits import gen_dtc
from core.io import save_dataset
from legacy.analysis.monte_carlo import run_monte_carlo
from core.runtime import THREAD_ENV_VARS, configure_single_threaded_blas
from legacy.analysis.sampling import _sample_parameters
from core.tracking import track_levels_by_overlap
from legacy.analysis.tunable_resonator import evaluate_dispersive_shifts


class PublicPackageApiTests(unittest.TestCase):
    def test_top_level_exports_core_workflows(self):
        for name in (
            "gen_dtc",
            "gen_dt",
            "find_operating_points",
            "load_or_compute_operating_points",
            "run_operating_point_dispersive_sweep",
            "load_or_compute_operating_point_dispersive_sweep",
            "track_levels_by_overlap",
            "save_dataset",
            "transmon_spectrum",
            "charge_coupling_ghz",
            "exchange_j01_ghz",
        ):
            self.assertTrue(hasattr(dtc, name), name)

    def test_gate_api_is_owned_by_gate_package(self):
        for name in (
            "build_logical_contour", "calibrate_reduced_pair",
            "build_dualrail_fsim_model", "DualRailPulse", "DualRailPulseConstraints",
            "simulate_dualrail_pulse", "extract_fsim_angles",
            "run_static_interaction_landscape", "load_or_compute_static_interaction_landscape",
        ):
            self.assertTrue(hasattr(gates, name), name)
            self.assertFalse(hasattr(dtc, name), name)

    def test_core_does_not_forward_to_gate_or_legacy_packages(self):
        self.assertTrue(all(module.startswith("core.") for module, _ in dtc._EXPORTS.values()))
        for name in ("run_monte_carlo", "run_two_param_sweep", "evaluate_dispersive_shifts"):
            self.assertFalse(hasattr(dtc, name), name)

    def test_core_helpers_are_owned_by_package_modules(self):
        self.assertEqual(gen_dtc.__module__, "core.circuits")
        self.assertEqual(
            track_levels_by_overlap.__module__,
            "core.tracking",
        )
        self.assertEqual(_sample_parameters.__module__, "legacy.analysis.sampling")
        self.assertEqual(save_dataset.__module__, "core.io")
        self.assertEqual(run_monte_carlo.__module__, "legacy.analysis.monte_carlo")
        self.assertEqual(
            evaluate_dispersive_shifts.__module__,
            "legacy.analysis.tunable_resonator",
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
