import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from core.constants import BRANCH_LABELS, TRACKING_SCHEMA
from core.operating_point_dispersive import (
    OPERATING_POINT_DISPERSIVE_SCHEMA,
    evaluate_dispersive_shifts_at_flux_pair,
    load_or_compute_operating_point_dispersive_sweep,
    run_operating_point_dispersive_sweep,
)
from core.operating_points import OPERATING_POINT_SCHEMA


def _symmetric_operator(size, edges):
    operator = np.zeros((size, size))
    for left, right, value in edges:
        operator[left, right] = operator[right, left] = value
    return operator


class TrackingCircuit:
    def __init__(self):
        self.Φ1 = 0.0
        self.Φ2 = 0.0

    @staticmethod
    def eigensys(*, evals_count):
        return np.arange(evals_count, dtype=float), np.eye(evals_count)

    @staticmethod
    def Q3_operator():
        return _symmetric_operator(9, [(0, 3, 1.0), (1, 7, 1.0), (2, 8, 1.0)])

    @classmethod
    def n1_operator(cls):
        return (cls._p_operator() + cls._m_operator()) / 2

    @classmethod
    def n2_operator(cls):
        return (cls._p_operator() - cls._m_operator()) / 2

    @staticmethod
    def _p_operator():
        return _symmetric_operator(
            9,
            [(0, 1, 1.0), (1, 4, 1.0), (2, 5, 1.0), (3, 7, 1.0)],
        )

    @staticmethod
    def _m_operator():
        return _symmetric_operator(
            9,
            [(0, 2, 1.0), (2, 6, 1.0), (1, 5, 1.0), (3, 8, 1.0)],
        )


def tracking_factory(**params):
    return TrackingCircuit()


def deterministic_pair_evaluator(circuit, phi_1, phi_2, *, evals_count, **kwargs):
    base = float(phi_1 + 10 * phi_2)
    chi_p = -0.002 * (1 + phi_1)
    chi_m = chi_p + 0.0001 * (1 + phi_2)
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


def operating_points_fixture(phi_1=(0.2, 0.6), phi_2=(0.3, 0.7)):
    point_count = len(phi_1)
    return xr.Dataset(
        data_vars={
            "phi_1": ("operating_point", np.asarray(phi_1)),
            "phi_2": ("operating_point", np.asarray(phi_2)),
            "hybridization": ("operating_point", np.full(point_count, 0.99)),
            "operating_frequency_ghz": (
                "operating_point",
                np.full(point_count, 4.5),
            ),
            "splitting_ghz": ("operating_point", np.full(point_count, 0.1)),
            "rail_level": (
                ("operating_point", "rail_state"),
                np.tile([1, 2], (point_count, 1)),
            ),
        },
        coords={
            "operating_point": np.arange(point_count),
            "rail_state": ["lower", "upper"],
        },
        attrs={"operating_point_schema": OPERATING_POINT_SCHEMA},
    )


class PairEvaluationTests(unittest.TestCase):
    def test_evaluator_returns_tracking_and_chi_diagnostics(self):
        circuit = TrackingCircuit()

        result = evaluate_dispersive_shifts_at_flux_pair(
            circuit, 0.2, 0.3, evals_count=9
        )

        self.assertEqual(circuit.Φ1, 0.2)
        self.assertEqual(circuit.Φ2, 0.3)
        np.testing.assert_array_equal(result["tracked_level"], np.arange(9))
        self.assertEqual(result["chi_p"], 3.0)
        self.assertEqual(result["chi_m"], 3.0)
        self.assertEqual(result["delta_chi"], 0.0)


class DispersiveSweepTests(unittest.TestCase):
    def test_dataset_keeps_operating_points_parameters_and_tracking(self):
        operating_points = operating_points_fixture()
        params = {"Ej1": 12.2, "Cr": 379.0}

        dataset = run_operating_point_dispersive_sweep(
            operating_points,
            params,
            circuit_factory=tracking_factory,
            evals_count=9,
            evaluator=deterministic_pair_evaluator,
        )

        self.assertEqual(dataset.energy.dims, ("operating_point", "level"))
        self.assertEqual(dataset.tracked_energy.dims, ("operating_point", "branch"))
        np.testing.assert_allclose(dataset.phi_1, operating_points.phi_1)
        np.testing.assert_allclose(
            dataset.hybridization, operating_points.hybridization
        )
        np.testing.assert_allclose(dataset.delta_chi, dataset.chi_p - dataset.chi_m)
        np.testing.assert_allclose(
            dataset.abs_delta_chi_mhz, 1e3 * np.abs(dataset.delta_chi)
        )
        np.testing.assert_allclose(dataset.param_value, [12.2, 379.0])
        self.assertEqual(dataset.param.values.tolist(), ["Ej1", "Cr"])
        self.assertEqual(dataset.attrs["tracking_schema"], TRACKING_SCHEMA)
        self.assertEqual(
            dataset.attrs["operating_point_dispersive_schema"],
            OPERATING_POINT_DISPERSIVE_SCHEMA,
        )
        for name in ("energy", "tracked_energy", "chi_p", "chi_m", "delta_chi"):
            self.assertEqual(dataset[name].attrs["units"], "GHz")
        self.assertEqual(dataset.abs_delta_chi_mhz.attrs["units"], "MHz")
        self.assertEqual(
            dataset.abs_delta_chi_over_chi_p.attrs["units"], "dimensionless"
        )

    def test_parallel_matches_serial_and_preserves_point_order(self):
        operating_points = operating_points_fixture(
            phi_1=(0.2, 0.4, 0.6, 0.8),
            phi_2=(0.3, 0.5, 0.7, 0.9),
        )
        kwargs = {
            "circuit_factory": tracking_factory,
            "evals_count": 9,
            "evaluator": deterministic_pair_evaluator,
        }

        serial = run_operating_point_dispersive_sweep(
            operating_points, {"Ej1": 12.2}, cpus=1, **kwargs
        )
        parallel = run_operating_point_dispersive_sweep(
            operating_points, {"Ej1": 12.2}, cpus=2, **kwargs
        )

        for name in (
            "energy",
            "tracked_energy",
            "tracked_level",
            "assignment_score",
            "chi_p",
            "chi_m",
            "delta_chi",
        ):
            np.testing.assert_allclose(parallel[name], serial[name])
        self.assertEqual(parallel.attrs["worker_count"], 2)


class DispersiveCacheTests(unittest.TestCase):
    def test_cache_round_trip_and_input_fingerprints(self):
        operating_points = operating_points_fixture()
        params = {"Ej1": 12.2}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dispersive.nc"
            computed, computed_path = load_or_compute_operating_point_dispersive_sweep(
                operating_points,
                params,
                path=path,
                force=True,
                circuit_factory=tracking_factory,
                evals_count=9,
                evaluator=deterministic_pair_evaluator,
            )
            loaded, loaded_path = load_or_compute_operating_point_dispersive_sweep(
                operating_points,
                params,
                path=path,
                circuit_factory=tracking_factory,
            )

            self.assertEqual(computed_path, path)
            self.assertEqual(loaded_path, path)
            np.testing.assert_allclose(loaded.delta_chi, computed.delta_chi)
            loaded.close()

            changed = operating_points_fixture(phi_1=(0.21, 0.6))
            with self.assertRaisesRegex(ValueError, "different flux pairs"):
                load_or_compute_operating_point_dispersive_sweep(
                    changed,
                    params,
                    path=path,
                    circuit_factory=tracking_factory,
                )
            with self.assertRaisesRegex(ValueError, "different circuit parameters"):
                load_or_compute_operating_point_dispersive_sweep(
                    operating_points,
                    {"Ej1": 12.3},
                    path=path,
                    circuit_factory=tracking_factory,
                )

    def test_stale_cache_requires_force_recompute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stale.nc"
            xr.Dataset({"chi_p": ("operating_point", [-0.002])}).to_netcdf(path)

            with self.assertRaisesRegex(ValueError, "force=True"):
                load_or_compute_operating_point_dispersive_sweep(path=path)


if __name__ == "__main__":
    unittest.main()
