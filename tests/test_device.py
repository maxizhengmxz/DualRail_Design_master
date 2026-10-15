import json
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np

from dtc_dualrail import (
    LogicalContour,
    LogicalFSimDeviceConfig,
    LogicalModeMapping,
    TDTPairConfig,
    TunableTransmonConfig,
    calibrate_reduced_pair_multi_point,
    run_single_transmon_sweep,
)
from dtc_dualrail.runtime import configure_single_threaded_blas


def reference_pair():
    return TDTPairConfig(
        Ej1=12.2,
        Cj1=5.0,
        Ej1_2=7.8,
        Cj1_2=5.0,
        C1=354.0,
        Ej2=12.2,
        Cj2=5.0,
        Ej2_2=7.8,
        Cj2_2=5.0,
        C2=434.0,
        C12=2.9,
        C1r=15.0,
        C2r=15.0,
        Cr=531.0,
        Lr=1.33,
    )


def reference_contour(carrier=4.75):
    return LogicalContour(
        logical_frequency_mhz=np.array([120.0, 125.0, 130.0]),
        phi_1=np.array([0.1, 0.15, 0.2]),
        phi_2=np.array([0.2, 0.25, 0.3]),
        carrier_frequency_ghz=np.full(3, carrier),
        hybridization=np.ones(3),
        polarization=np.zeros(3),
    )


class DeviceDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_single_threaded_blas()

    def setUp(self):
        self.pair = reference_pair()
        self.device = LogicalFSimDeviceConfig(
            pair_a=self.pair,
            pair_b=replace(self.pair, C1=294.0, C2=364.0),
            coupler=TunableTransmonConfig(
                EJmax=24.4,
                EC=0.6,
                d=0.02,
                flux=0.4,
                ncut=15,
                truncated_dim=4,
            ),
            min_assignment_probability=0.8,
            min_leakage_gap_mhz=5.0,
        )

    def test_reference_circuit_preserves_factory_units_and_junctions(self):
        pair = replace(self.pair, Ej1_2=6.0, Ej2=13.0, C12=3.1)
        with patch("dtc_dualrail.circuits._configured_dual_rail_circuit") as build:
            self.assertIs(pair.build_circuit(), build.return_value)
        yaml = build.call_args.args[0]
        self.assertIn("Ej1_2=6.0GHz", yaml)
        self.assertIn("Ej2=13.0GHz", yaml)
        self.assertIn("C1=354.0MHz", yaml)
        self.assertIn("C2=434.0MHz", yaml)
        self.assertIn("C12=3.1fF", yaml)
        self.assertIn("531.0fF", yaml)
        self.assertIn("1.33nH", yaml)

    def test_unified_calibration_fits_and_reports_the_full_contour(self):
        contours = reference_contour(), reference_contour(4.9)
        config, calibration_a, calibration_b = self.device.calibrate(*contours)
        for contour, actual in zip(
            contours, (calibration_a, calibration_b), strict=True
        ):
            np.testing.assert_array_equal(
                actual.sample_logical_frequency_mhz,
                contour.logical_frequency_mhz,
            )
            self.assertEqual(actual.carrier_residual_mhz.shape, (3,))
            self.assertEqual(actual.logical_splitting_residual_mhz.shape, (3,))
            self.assertTrue(np.all(np.isfinite(actual.carrier_residual_mhz)))
            self.assertTrue(np.all(np.isfinite(actual.logical_splitting_residual_mhz)))
            report = actual.to_dataset()
            self.assertIn("reference_carrier_frequency_ghz", report)
            self.assertIn("reduced_logical_splitting_mhz", report)
            self.assertEqual(report.attrs["fit_method"], "multi-point least squares")
        self.assertEqual(config.rail_a1, calibration_a.rail_1)
        self.assertEqual(config.rail_b2, calibration_b.rail_2)
        self.assertEqual(config.coupling.a1_a2, calibration_a.coupling_ghz)
        self.assertEqual(config.coupling.b1_b2, calibration_b.coupling_ghz)

    def test_changes_propagate_to_correct_rails_links_and_metadata(self):
        device = replace(
            self.device,
            pair_a=replace(self.pair, Ej1=13.0, Ej2_2=8.0),
            pair_b=replace(self.pair, Ej1_2=6.0, Ej2=14.0),
            coupler=replace(self.device.coupler, EC=0.7),
            a2_c_ghz=0.08,
            c_b1_ghz=0.12,
            logical_mapping=LogicalModeMapping(
                logical_zero="upper", logical_one="lower"
            ),
        )
        config, _, _ = device.calibrate(reference_contour(), reference_contour(4.9))
        rails = config.rail_a1, config.rail_a2, config.rail_b1, config.rail_b2
        for rail, ejmax, d in zip(
            rails,
            (20.8, 20.2, 18.2, 21.8),
            (5.2 / 20.8, 4.2 / 20.2, 6.2 / 18.2, 6.2 / 21.8),
            strict=True,
        ):
            self.assertAlmostEqual(rail.EJmax, ejmax)
            self.assertAlmostEqual(rail.d, d)
        self.assertEqual(config.coupler, device.coupler)
        self.assertEqual(config.coupling.a2_c, 0.08)
        self.assertEqual(config.coupling.c_b1, 0.12)
        self.assertEqual(config.logical_mapping, device.logical_mapping)
        self.assertEqual(config.min_assignment_probability, 0.8)
        self.assertEqual(config.min_leakage_gap_mhz, 5.0)
        self.assertEqual(
            json.loads(config.parameter_source),
            json.loads(json.dumps(asdict(device))),
        )

    def test_direct_reference_is_selected_when_a_pair_fit_misses_tolerance(self):
        contours = reference_contour(), reference_contour(4.9)
        _, calibration_a, calibration_b = self.device.calibrate(*contours)
        expected = not (
            calibration_a.residual_within_tolerance
            and calibration_b.residual_within_tolerance
        )
        self.assertEqual(
            self.device.requires_direct_reference(calibration_a, calibration_b),
            expected,
        )
        with patch("dtc_dualrail.device.gen_five_transmon_dual_rail") as build:
            self.assertIs(self.device.build_reference_circuit(), build.return_value)
        kwargs = build.call_args.kwargs
        self.assertEqual(kwargs["coupling_capacitance_ff"]["A1_A2"], self.pair.C12)
        self.assertEqual(kwargs["coupling_capacitance_ff"]["B1_B2"], self.pair.C12)
        np.testing.assert_allclose(kwargs["coupler_ej_pair_ghz"], (12.444, 11.956))

    def test_every_physical_parameter_changes_reference_cache_identity(self):
        for name, value in asdict(self.pair).items():
            with self.subTest(name=name):
                changed = replace(self.pair, **{name: value + 0.1})
                self.assertNotEqual(self.pair.fingerprint(), changed.fingerprint())
        self.assertEqual(
            self.pair.fingerprint(), replace(self.pair, Cj1=5).fingerprint()
        )

    def test_invalid_physical_and_basis_inputs_fail_early(self):
        for changes in (
            {"Ej1": -1},
            {"C12": float("nan")},
            {"C1": 0},
            {"Ej1": 0, "Ej1_2": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.pair, **changes)
        for changes in (
            {"rail_ncut": 0},
            {"rail_truncated_dim": 1},
            {"rail_ncut": 1, "rail_truncated_dim": 4},
            {"a2_c_ghz": float("inf")},
            {"a2_c_ff": 0.0},
            {"c_b1_ff": float("nan")},
            {"coupler_junction_capacitance_ff": (0.2, -0.1)},
            {"calibration_residual_tolerance_fraction": 1.0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.device, **changes)


class SingleTransmonValidationTests(unittest.TestCase):
    def test_sweep_records_spectrum_charge_elements_and_ncut_convergence(self):
        config = TunableTransmonConfig(
            EJmax=12.0,
            EC=0.24,
            d=0.15,
            ncut=4,
            truncated_dim=3,
        )

        result = run_single_transmon_sweep(
            config,
            flux_values=[0.0, 0.2],
            ncut_values=[3, 4, 5],
            convergence_tolerance_khz=50.0,
        )

        self.assertEqual(result.sizes["ncut"], 3)
        self.assertEqual(result.sizes["flux"], 2)
        self.assertEqual(
            {"f01_ghz", "f12_ghz", "anharmonicity_ghz", "n01", "n12"},
            {
                name
                for name in result.data_vars
                if name in {"f01_ghz", "f12_ghz", "anharmonicity_ghz", "n01", "n12"}
            },
        )
        np.testing.assert_allclose(
            result.anharmonicity_ghz,
            result.f12_ghz - result.f01_ghz,
        )
        self.assertTrue(np.all(result.n01 > 0))
        self.assertTrue(np.all(result.n12 > 0))
        np.testing.assert_array_equal(result.ncut_delta_khz.ncut_step, [4, 5])
        self.assertEqual(
            result.ncut_delta_khz.dims, ("ncut_step", "flux", "transition")
        )
        self.assertIn(result.attrs["ncut_converged"], (0, 1))
        self.assertEqual(result.attrs["convergence_tolerance_khz"], 50.0)

    def test_sweep_rejects_an_inadequate_basis_or_grid(self):
        config = TunableTransmonConfig(
            EJmax=12.0,
            EC=0.24,
            d=0.15,
            truncated_dim=3,
        )
        with self.assertRaisesRegex(ValueError, "ncut_values"):
            run_single_transmon_sweep(config, [0.0], ncut_values=[1, 2])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            run_single_transmon_sweep(config, [0.0], ncut_values=[4, 4])


class MultiPointCalibrationTests(unittest.TestCase):
    def test_richer_contour_reports_two_excitation_and_charge_residuals(self):
        frequency = np.array([95.0, 105.0])
        contour = LogicalContour(
            logical_frequency_mhz=frequency,
            phi_1=np.array([0.0, 0.01]),
            phi_2=np.array([0.0, 0.01]),
            carrier_frequency_ghz=np.array([4.5, 4.49]),
            hybridization=np.ones(2),
            polarization=np.zeros(2),
            two_excitation_frequency_ghz=np.array(
                [[8.98, 9.03, 9.08], [8.96, 9.01, 9.06]]
            ),
            charge_matrix_element=np.full((2, 2, 2), 0.8),
        )

        result = calibrate_reduced_pair_multi_point(
            contour,
            ejmax_ghz=(18.0, 18.0),
            asymmetry=(0.15, 0.15),
            ncut=4,
            truncated_dim=3,
        )

        self.assertEqual(result.two_excitation_residual_mhz.shape, (2, 3))
        self.assertEqual(result.charge_matrix_element_residual.shape, (2, 2, 2))
        self.assertAlmostEqual(result.residual_tolerance_mhz, 2.0)
        self.assertLess(np.max(np.abs(result.logical_splitting_residual_mhz)), 10.0)
        self.assertEqual(result.fit_selection, "core-preserving-fallback")
        report = result.to_dataset()
        self.assertIn("two_excitation_residual_mhz", report)
        self.assertIn("charge_matrix_element_residual", report)


if __name__ == "__main__":
    unittest.main()
