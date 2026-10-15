import copy
import importlib
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import xarray as xr

from dtc_dualrail.dualrail_fsim import DualRailPulse, build_dualrail_fsim_model
from dtc_dualrail.logical_fsim_config import LogicalFSimWorkflowConfig
from dtc_dualrail.logical_fsim_workflow import (
    LOGICAL_FSIM_PULSE_SWEEP_SCHEMA,
    load_compatible_pulse_sweep,
    load_or_create_configured_reference,
    pulse_results_to_dataset,
)
from tests.test_dualrail_fsim import pulse_contours, tiny_config

ROOT = Path(__file__).parents[1]
SMOKE_CONFIG = ROOT / "configs" / "logical_fsim_smoke.toml"
PRODUCTION_CONFIG = ROOT / "configs" / "logical_fsim.toml"


class WorkflowConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = LogicalFSimWorkflowConfig.load(SMOKE_CONFIG)

    def changed(self, path, value):
        values = copy.deepcopy(self.config.values)
        target = values
        for name in path[:-1]:
            target = target[name]
        target[path[-1]] = value
        return LogicalFSimWorkflowConfig(self.config.source, values)

    def test_model_contour_grid_pulse_and_solver_changes_alter_fingerprints(self):
        cases = (
            (("model", "a2_c_ghz"), 0.041, ("model",)),
            (("contours", "segment_a"), 1, ("contours",)),
            (("static", "frequency_a_mhz"), [99.0], ("static",)),
            (("pulse", "padding_ns"), 0.2, ("pulse",)),
            (("pulse", "solver", "rtol"), 2e-6, ("pulse",)),
            (("phase1", "path_samples"), 5, ("phase1",)),
            (("phase2", "coherent_ceiling_time_samples"), 7, ("phase2",)),
        )
        for path, value, sections in cases:
            with self.subTest(path=path):
                changed = self.changed(path, value)
                self.assertNotEqual(
                    self.config.fingerprint(*sections),
                    changed.fingerprint(*sections),
                )

    def test_production_branch_tracking_uses_resolved_path(self):
        production = LogicalFSimWorkflowConfig.load(PRODUCTION_CONFIG)

        self.assertGreaterEqual(production.section("phase1", "path_samples"), 25)

    def test_production_selected_paths_only_tune_the_converged_coupler_basis(self):
        production = LogicalFSimWorkflowConfig.load(PRODUCTION_CONFIG)
        idle = production.section("idle")

        for gate in ("iswap", "cz"):
            point = production.section("gates", gate)
            self.assertEqual(point["frequency_a_mhz"], idle["frequency_a_mhz"])
            self.assertEqual(point["frequency_b_mhz"], idle["frequency_b_mhz"])
        self.assertGreaterEqual(
            production.section("phase1", "pulse_coupler_truncated_dim"), 7
        )

    def test_configured_reference_cache_loads_only_identical_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reference.nc"
            expected = self.config.configured_operating_points("a")
            first, _ = load_or_create_configured_reference(
                lambda: expected.copy(),
                path=path,
                fingerprint="same",
            )
            first.close()
            loaded, _ = load_or_create_configured_reference(
                lambda: self.fail("compatible cache should load"),
                path=path,
                fingerprint="same",
            )
            xr.testing.assert_allclose(loaded, expected)
            loaded.close()
            with self.assertRaisesRegex(ValueError, "force=True"):
                load_or_create_configured_reference(
                    lambda: expected.copy(),
                    path=path,
                    fingerprint="changed",
                )


class PulsePersistenceTests(unittest.TestCase):
    def test_complete_result_and_transient_maxima_round_trip(self):
        config = tiny_config()
        model = build_dualrail_fsim_model(config)
        contour_a, contour_b = pulse_contours(config)
        pulse = DualRailPulse(
            coupler_peak_flux=0.22,
            duration_ns=1.0,
            rise_time_ns=0.0,
            output_samples=7,
        )
        result = model.simulate(pulse, contour_a, contour_b)
        dataset = pulse_results_to_dataset(
            [result],
            [(0.0, 1.0)],
            gate_type="iswap",
            fingerprint="workflow-fingerprint",
            gate_point={
                "frequency_a_mhz": 100.0,
                "frequency_b_mhz": 100.0,
                "coupler_flux": 0.22,
            },
        )

        required = {
            "leakage",
            "parasitic_amplitude",
            "coupler_excited_probability",
            "pair_a_invalid_probability",
            "pair_b_invalid_probability",
            "iswap_virtual_z_rad",
            "cphase_virtual_z_rad",
            "solver_status",
            "maximum_transient_leakage",
            "maximum_transient_coupler_excited_probability",
            "maximum_transient_pair_a_invalid_probability",
            "maximum_transient_pair_b_invalid_probability",
        }
        self.assertTrue(required <= set(dataset.data_vars))
        self.assertEqual(dataset.solver_status.item(), "success")
        self.assertTrue(np.all(dataset.maximum_transient_leakage >= dataset.leakage))
        self.assertEqual(
            dataset.attrs["logical_fsim_pulse_sweep_schema"],
            LOGICAL_FSIM_PULSE_SWEEP_SCHEMA,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pulse.nc"
            dataset.to_netcdf(path)
            loaded = load_compatible_pulse_sweep(path, "workflow-fingerprint")
            self.assertIsNotNone(loaded)
            xr.testing.assert_allclose(loaded, dataset)
            loaded.close()
            self.assertIsNone(load_compatible_pulse_sweep(path, "changed"))


class HeadlessNotebookSmokeTests(unittest.TestCase):
    def test_notebook_dag_runs_twice_from_an_initially_empty_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = {
                "LOGICAL_FSIM_CONFIG": str(SMOKE_CONFIG),
                "LOGICAL_FSIM_DATA": tmpdir,
                "MPLCONFIGDIR": tmpdir,
            }
            with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                notebook = importlib.import_module("logical_fsim")
                _, first = notebook.app.run()
                first_fingerprint = first["iswap_smooth_sweep"].attrs["fingerprint"]
                _, second = notebook.app.run()

            self.assertEqual(
                second["iswap_smooth_sweep"].attrs["fingerprint"],
                first_fingerprint,
            )
            self.assertTrue(Path(second["static_atlas_path"]).exists())
            self.assertTrue(Path(second["gate_comparison_path"]).exists())
            self.assertEqual(
                second["iswap_smooth_sweep"].solver_status.item(), "success"
            )
            self.assertEqual(len(second["iswap_trajectories"]), 4)
            self.assertEqual(len(second["cz_trajectories"]), 4)
            self.assertEqual(
                set(second["transmon_validation"]), {"A1", "A2", "C", "B1", "B2"}
            )
            self.assertEqual(
                second["calibration_reports"]["A"].sizes["logical_frequency_mhz"],
                3,
            )
            self.assertIn("diabatic_level", second["iswap_branch_tracking"])
            self.assertIn("adiabatic_level", second["cz_branch_tracking"])
            self.assertEqual(second["iswap_basis_validation"].sizes["path_point"], 3)
            self.assertIn("iswap", second["static_dynamic_rate_reports"])
            self.assertIn("cz", second["static_dynamic_rate_reports"])
            self.assertIsInstance(second["phase1_execution_gate"], bool)
            self.assertIsInstance(second["phase2_execution_gate"], bool)
            self.assertIn(
                "static_coherent_ceiling",
                second["iswap_static_candidate_report"],
            )


if __name__ == "__main__":
    unittest.main()
