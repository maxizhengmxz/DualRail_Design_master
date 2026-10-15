import tempfile
import unittest
from pathlib import Path

import numpy as np

from dtc_dualrail.constants import BRANCH_LABELS
from dtc_dualrail.tdt_operating_point_monte_carlo import (
    TDT_OPERATING_POINT_MONTE_CARLO_SCHEMA,
    load_or_compute_tdt_operating_point_monte_carlo,
    run_tdt_operating_point_monte_carlo,
)


class PipelineCircuit:
    def __init__(self, device_value):
        self.device_value = device_value
        self.Φ1 = 0.0
        self.Φ2 = 0.0


def pipeline_factory(device_value):
    return PipelineCircuit(device_value)


def operating_point_evaluator(circuit, phi_1, phi_2, **kwargs):
    distance = phi_1 - phi_2
    return {
        "polarization": distance,
        "hybridization": 1.0 - abs(distance),
        "operating_frequency_ghz": circuit.device_value + phi_2,
        "splitting_ghz": 0.1 + distance**2,
        "rail_levels": np.array([1, 2]),
    }


def dispersive_evaluator(circuit, phi_1, phi_2, *, evals_count, **kwargs):
    base = circuit.device_value + phi_1 + 10 * phi_2
    chi_p = -0.002 * (1.0 + phi_1)
    chi_m = chi_p + 0.0001 * (1.0 + phi_2)
    delta_chi = chi_p - chi_m
    return {
        "energy": np.arange(evals_count, dtype=float) + base,
        "tracked_energy": np.arange(len(BRANCH_LABELS), dtype=float) + base,
        "tracked_level": np.arange(len(BRANCH_LABELS)),
        "assignment_score": np.ones(len(BRANCH_LABELS)),
        "chi_p": chi_p,
        "chi_m": chi_m,
        "delta_chi": delta_chi,
        "abs_delta_chi_over_chi_p": abs(delta_chi / chi_p),
    }


def monotonic_operating_point_evaluator(circuit, phi_1, phi_2, **kwargs):
    return {
        "polarization": 1.0,
        "hybridization": 1.0,
        "operating_frequency_ghz": circuit.device_value,
        "splitting_ghz": phi_1,
        "rail_levels": np.array([1, 2]),
    }


class TdtOperatingPointMonteCarloTests(unittest.TestCase):
    def run_pipeline(self, **overrides):
        kwargs = {
            "samples": 3,
            "seed": 17,
            "param_sigma": {"device_value": 0.0},
            "param_rel_sigma": {"device_value": 0.0},
            "circuit_factory": pipeline_factory,
            "operating_point_evals_count": 3,
            "dispersive_evals_count": len(BRANCH_LABELS),
            "cpus": 1,
            "operating_point_evaluator": operating_point_evaluator,
            "dispersive_evaluator": dispersive_evaluator,
        }
        kwargs.update(overrides)
        return run_tdt_operating_point_monte_carlo(
            {"device_value": 5.0},
            np.linspace(0.0, 1.0, 9),
            [0.2, 0.6],
            **kwargs,
        )

    def test_finds_and_evaluates_operating_points_for_every_sample(self):
        dataset = self.run_pipeline()

        self.assertEqual(dataset.sizes["sample"], 3)
        self.assertEqual(dataset.sizes["operating_point"], 6)
        np.testing.assert_array_equal(dataset.operating_point_count, [2, 2, 2])
        np.testing.assert_array_equal(dataset.sample_index, [0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(
            dataset.sample_operating_point,
            [0, 1, 0, 1, 0, 1],
        )
        np.testing.assert_allclose(dataset.phi_1, [0.2, 0.6] * 3, atol=1e-5)
        np.testing.assert_allclose(dataset.phi_2, [0.2, 0.6] * 3)
        np.testing.assert_allclose(dataset.param_value, 5.0)
        np.testing.assert_allclose(dataset.delta_chi, dataset.chi_p - dataset.chi_m)
        np.testing.assert_allclose(
            dataset.abs_delta_chi_mhz,
            1e3 * np.abs(dataset.delta_chi),
        )
        self.assertEqual(
            dataset.attrs["tdt_operating_point_monte_carlo_schema"],
            TDT_OPERATING_POINT_MONTE_CARLO_SCHEMA,
        )

    def test_sampling_is_reproducible(self):
        first = self.run_pipeline(
            param_sigma=None,
            param_rel_sigma={"device_value": 0.1},
        )
        second = self.run_pipeline(
            param_sigma=None,
            param_rel_sigma={"device_value": 0.1},
        )

        np.testing.assert_allclose(first.param_value, second.param_value)
        np.testing.assert_allclose(first.energy, second.energy)

    def test_samples_with_no_operating_points_produce_an_empty_result(self):
        dataset = self.run_pipeline(
            operating_point_evaluator=monotonic_operating_point_evaluator,
        )

        self.assertEqual(dataset.sizes["operating_point"], 0)
        np.testing.assert_array_equal(dataset.operating_point_count, [0, 0, 0])
        self.assertEqual(dataset.energy.shape, (0, len(BRANCH_LABELS)))
        self.assertEqual(dataset.tracked_energy.shape, (0, len(BRANCH_LABELS)))

    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tdt-operating-point-monte-carlo.nc"
            computed, computed_path = load_or_compute_tdt_operating_point_monte_carlo(
                {"device_value": 5.0},
                np.linspace(0.0, 1.0, 9),
                [0.2, 0.6],
                path=path,
                force=True,
                samples=1,
                param_sigma={"device_value": 0.0},
                param_rel_sigma={"device_value": 0.0},
                circuit_factory=pipeline_factory,
                operating_point_evals_count=3,
                dispersive_evals_count=len(BRANCH_LABELS),
                operating_point_evaluator=operating_point_evaluator,
                dispersive_evaluator=dispersive_evaluator,
            )
            loaded, loaded_path = load_or_compute_tdt_operating_point_monte_carlo(
                path=path
            )

            self.assertEqual(computed_path, path)
            self.assertEqual(loaded_path, path)
            np.testing.assert_allclose(loaded.delta_chi, computed.delta_chi)
            loaded.close()


if __name__ == "__main__":
    unittest.main()
