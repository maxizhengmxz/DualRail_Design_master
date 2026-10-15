import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from dtc_dualrail.operating_points import (
    OPERATING_POINT_SCHEMA,
    evaluate_rail_hybridization,
    find_operating_points,
    load_or_compute_operating_points,
)


class AvoidedCrossingCircuit:
    def __init__(self, coupling=0.05):
        self.Φ1 = 0.0
        self.Φ2 = 0.0
        self.coupling = coupling

    def eigensys(self, *, evals_count):
        if evals_count != 3:
            raise AssertionError("test circuit expects three eigenstates")
        detuning = self.Φ1 - self.Φ2
        rail_hamiltonian = np.array(
            [
                [5.0 + detuning / 2, self.coupling],
                [self.coupling, 5.0 - detuning / 2],
            ]
        )
        rail_energies, rail_states = np.linalg.eigh(rail_hamiltonian)
        states = np.zeros((3, 3))
        states[0, 0] = 1.0
        states[1:, 1:] = rail_states
        return np.concatenate(([0.0], rail_energies)), states

    @staticmethod
    def n1_operator():
        operator = np.zeros((3, 3))
        operator[0, 1] = operator[1, 0] = 1.0
        return operator

    @staticmethod
    def n2_operator():
        operator = np.zeros((3, 3))
        operator[0, 2] = operator[2, 0] = 1.0
        return operator


class RailHybridizationTests(unittest.TestCase):
    def test_score_is_unity_and_polarization_zero_at_resonance(self):
        circuit = AvoidedCrossingCircuit(coupling=0.05)

        result = evaluate_rail_hybridization(circuit, 0.3, 0.3, evals_count=3)

        self.assertAlmostEqual(result["polarization"], 0.0)
        self.assertAlmostEqual(result["hybridization"], 1.0)
        self.assertAlmostEqual(result["operating_frequency_ghz"], 5.0)
        self.assertAlmostEqual(result["splitting_ghz"], 0.1)
        np.testing.assert_array_equal(result["rail_levels"], [1, 2])

    def test_polarization_changes_sign_across_avoided_crossing(self):
        circuit = AvoidedCrossingCircuit()

        left = evaluate_rail_hybridization(circuit, 0.2, 0.4, evals_count=3)
        right = evaluate_rail_hybridization(circuit, 0.6, 0.4, evals_count=3)

        self.assertLess(left["polarization"] * right["polarization"], 0)
        self.assertLess(left["hybridization"], 1.0)
        self.assertLess(right["hybridization"], 1.0)


class OperatingPointFinderTests(unittest.TestCase):
    def test_preserves_calibration_observables_returned_by_evaluator(self):
        expected_charge = np.asarray([[0.1, 0.2], [0.3, 0.4]])
        expected_two_excitation = np.asarray([8.9, 9.0, 9.1])

        def rich_evaluator(circuit, phi_1, phi_2, **kwargs):
            return {
                "polarization": phi_1 - phi_2,
                "hybridization": 1.0,
                "operating_frequency_ghz": 4.5,
                "splitting_ghz": 0.1 + (phi_1 - phi_2) ** 2,
                "rail_levels": np.array([1, 2]),
                "charge_matrix_element": expected_charge,
                "two_excitation_frequency_ghz": expected_two_excitation,
            }

        result = find_operating_points(
            object(),
            np.linspace(0.0, 1.0, 11),
            [0.3],
            evaluator=rich_evaluator,
        )

        np.testing.assert_allclose(
            result.charge_matrix_element.isel(operating_point=0), expected_charge
        )
        np.testing.assert_allclose(
            result.two_excitation_frequency_ghz.isel(operating_point=0),
            expected_two_excitation,
        )

    def test_refines_one_operating_point_per_phi2_slice(self):
        circuit = AvoidedCrossingCircuit()
        original_flux = (circuit.Φ1, circuit.Φ2)

        result = find_operating_points(
            circuit,
            np.linspace(0.0, 1.0, 9),
            [0.2, 0.6],
            evals_count=3,
            minimize_xtol=1e-10,
        )

        np.testing.assert_allclose(result.phi_1, [0.2, 0.6], atol=1e-8)
        np.testing.assert_allclose(result.phi_2, [0.2, 0.6])
        np.testing.assert_allclose(result.hybridization, 1.0)
        np.testing.assert_allclose(result.splitting_ghz, 0.1)
        self.assertEqual((circuit.Φ1, circuit.Φ2), original_flux)

    def test_retains_all_symmetry_related_roots_on_each_slice(self):
        def symmetric_evaluator(circuit, phi_1, phi_2, **kwargs):
            distance = min(abs(phi_1 - phi_2), abs(phi_1 - (1.0 - phi_2)))
            return {
                "polarization": (phi_1 - phi_2) * (phi_1 - (1.0 - phi_2)),
                "hybridization": 1.0 - distance,
                "operating_frequency_ghz": 4.5,
                "splitting_ghz": 0.1 + distance**2,
                "rail_levels": np.array([1, 2]),
            }

        result = find_operating_points(
            object(),
            np.linspace(0.0, 1.0, 11),
            [0.2, 0.3],
            evaluator=symmetric_evaluator,
        )

        np.testing.assert_allclose(result.phi_1, [0.2, 0.8, 0.3, 0.7], atol=1e-5)
        np.testing.assert_allclose(result.phi_2, [0.2, 0.2, 0.3, 0.3])

    def test_location_is_set_by_gap_minimum_not_polarization_zero(self):
        def shifted_evaluator(circuit, phi_1, phi_2, **kwargs):
            return {
                "polarization": phi_1 - phi_2,
                "hybridization": 1.0,
                "operating_frequency_ghz": 4.5,
                "splitting_ghz": 0.1 + (phi_1 - phi_2 - 0.1) ** 2,
                "rail_levels": np.array([1, 2]),
            }

        result = find_operating_points(
            object(),
            np.linspace(0.0, 1.0, 11),
            [0.3],
            evaluator=shifted_evaluator,
        )

        self.assertAlmostEqual(result.phi_1.item(), 0.4, places=5)
        self.assertAlmostEqual(result.polarization.item(), 0.1, places=5)

    def test_parallel_matches_serial_and_preserves_slice_order(self):
        phi_1_values = np.linspace(0.0, 1.0, 9)
        phi_2_values = [0.2, 0.4, 0.6, 0.8]

        serial = find_operating_points(
            AvoidedCrossingCircuit(),
            phi_1_values,
            phi_2_values,
            evals_count=3,
            minimize_xtol=1e-10,
            cpus=1,
        )
        parallel = find_operating_points(
            AvoidedCrossingCircuit(),
            phi_1_values,
            phi_2_values,
            evals_count=3,
            minimize_xtol=1e-10,
            cpus=2,
        )

        np.testing.assert_allclose(parallel.phi_1, serial.phi_1)
        np.testing.assert_allclose(parallel.phi_2, serial.phi_2)
        np.testing.assert_allclose(parallel.hybridization, serial.hybridization)
        np.testing.assert_allclose(parallel.splitting_ghz, serial.splitting_ghz)
        np.testing.assert_array_equal(parallel.rail_level, serial.rail_level)
        self.assertEqual(parallel.attrs["cpus"], 2)
        self.assertEqual(parallel.attrs["worker_count"], 2)

    def test_rejects_invalid_search_configuration(self):
        circuit = AvoidedCrossingCircuit()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            find_operating_points(circuit, [0.0, 0.5, 0.4], [0.2])
        with self.assertRaisesRegex(ValueError, "at least 3"):
            find_operating_points(circuit, [0.0, 1.0], [0.2])
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            find_operating_points(
                circuit, [0.0, 0.5, 1.0], [0.2], min_hybridization=1.1
            )
        with self.assertRaisesRegex(ValueError, "cpus"):
            find_operating_points(circuit, [0.0, 0.5, 1.0], [0.2], cpus=0)


class OperatingPointCacheTests(unittest.TestCase):
    def test_cache_round_trip_preserves_operating_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "operating-points.nc"
            computed, computed_path = load_or_compute_operating_points(
                AvoidedCrossingCircuit(),
                np.linspace(0.0, 1.0, 9),
                [0.2, 0.6],
                path=path,
                force=True,
                evals_count=3,
            )
            loaded, loaded_path = load_or_compute_operating_points(path=path)

            self.assertEqual(computed_path, path)
            self.assertEqual(loaded_path, path)
            np.testing.assert_allclose(loaded.phi_1, computed.phi_1)
            np.testing.assert_allclose(loaded.phi_2, computed.phi_2)
            self.assertEqual(
                loaded.attrs["operating_point_schema"], OPERATING_POINT_SCHEMA
            )
            loaded.close()

    def test_stale_cache_requires_force_recompute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stale.nc"
            xr.Dataset({"phi_1": ("operating_point", [0.2])}).to_netcdf(path)

            with self.assertRaisesRegex(ValueError, "force=True"):
                load_or_compute_operating_points(path=path)

    def test_explicit_input_fingerprint_rejects_changed_circuit_or_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fingerprinted.nc"
            computed, _ = load_or_compute_operating_points(
                AvoidedCrossingCircuit(),
                np.linspace(0.0, 1.0, 9),
                [0.2, 0.6],
                path=path,
                input_fingerprint="original-inputs",
                evals_count=3,
            )
            self.assertEqual(computed.attrs["input_fingerprint"], "original-inputs")

            loaded, _ = load_or_compute_operating_points(
                path=path,
                input_fingerprint="original-inputs",
            )
            loaded.close()
            with self.assertRaisesRegex(ValueError, "force=True"):
                load_or_compute_operating_points(
                    path=path,
                    input_fingerprint="changed-inputs",
                )


if __name__ == "__main__":
    unittest.main()
