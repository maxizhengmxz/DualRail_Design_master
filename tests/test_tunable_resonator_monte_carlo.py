import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from dtc_dualrail.sampling import _sample_parameters
from dtc_dualrail.tunable_resonator import (
    _abs_delta_chi_over_chi_p,
    evaluate_dispersive_shifts,
    load_or_compute_tunable_resonator_monte_carlo,
    optimize_resonator_inductance,
    run_tunable_resonator_monte_carlo,
    run_tunable_resonator_sample_sweep,
)


class FrozenParameterSamplingTests(unittest.TestCase):
    def test_frozen_parameters_remain_nominal_with_nonzero_sigmas(self):
        params = {"Ej1": 14.0, "Cr": 385.0, "Lr": 1.56}

        sampled, sigmas = _sample_parameters(
            params,
            np.random.default_rng(123),
            param_sigma={"Ej1": 1.0, "Cr": 100.0, "Lr": 1.0},
            param_rel_sigma={},
            frozen_params={"Cr", "Lr"},
        )

        self.assertNotEqual(sampled["Ej1"], params["Ej1"])
        self.assertEqual(sampled["Cr"], params["Cr"])
        self.assertEqual(sampled["Lr"], params["Lr"])
        self.assertEqual(sigmas["Cr"], 0.0)
        self.assertEqual(sigmas["Lr"], 0.0)

    def test_frozen_parameters_may_remain_in_correlation_groups(self):
        params = {"C1": 194.0, "Cr": 385.0, "Lr": 1.56}

        sampled, _ = _sample_parameters(
            params,
            np.random.default_rng(456),
            param_sigma=None,
            param_rel_sigma={"C1": 0.1, "Cr": 0.1, "Lr": 0.1},
            param_correlation_groups=[
                {"params": ["C1", "Cr", "Lr"], "correlation": 0.8}
            ],
            frozen_params={"Cr", "Lr"},
        )

        self.assertNotEqual(sampled["C1"], params["C1"])
        self.assertEqual(sampled["Cr"], params["Cr"])
        self.assertEqual(sampled["Lr"], params["Lr"])


class FakeCircuit:
    def __init__(self):
        self.flux = None

    def eigensys(self, evals_count):
        return np.arange(evals_count, dtype=float), np.eye(evals_count)

    def Q3_operator(self):
        operator = np.zeros((12, 12))
        operator[0, 3] = operator[3, 0] = 1.0
        operator[1, 7] = operator[7, 1] = 2.0
        operator[2, 8] = operator[8, 2] = 3.0
        return operator

    def n1_operator(self):
        return (self._p_operator() + self._m_operator()) / 2

    def n2_operator(self):
        return (self._p_operator() - self._m_operator()) / 2

    @staticmethod
    def _p_operator():
        operator = np.zeros((12, 12))
        for left, right, value in (
            (0, 1, 1.0),
            (1, 4, 2.0),
            (2, 5, 1.0),
            (3, 7, 1.0),
        ):
            operator[left, right] = operator[right, left] = value
        return operator

    @staticmethod
    def _m_operator():
        operator = np.zeros((12, 12))
        for left, right, value in (
            (0, 2, 1.0),
            (2, 6, 2.0),
            (1, 5, 1.0),
            (3, 8, 1.0),
        ):
            operator[left, right] = operator[right, left] = value
        return operator


class DispersiveShiftEvaluationTests(unittest.TestCase):
    def test_ratio_is_absolute_delta_chi_over_chi_p(self):
        self.assertAlmostEqual(_abs_delta_chi_over_chi_p(-2.0, -1.5), 0.25)

    def test_ratio_is_infinite_when_chi_p_is_zero_or_nonfinite(self):
        self.assertTrue(np.isinf(_abs_delta_chi_over_chi_p(0.0, 1.0)))
        self.assertTrue(np.isinf(_abs_delta_chi_over_chi_p(np.nan, 1.0)))

    def test_evaluator_sets_flux_and_returns_tracking_diagnostics(self):
        circuit = FakeCircuit()

        result = evaluate_dispersive_shifts(
            {"Lr": 1.5},
            circuit_factory=lambda **params: circuit,
            flux=0.5,
            flux_param="flux",
            evals_count=12,
        )

        self.assertEqual(circuit.flux, 0.5)
        self.assertEqual(
            result["tracked_level"].tolist(),
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
        )
        self.assertEqual(result["chi_p"], 3.0)
        self.assertEqual(result["chi_m"], 3.0)
        self.assertEqual(result["objective"], 0.0)
        self.assertEqual(result["assignment_score"].shape, (9,))


class ResonatorOptimizationTests(unittest.TestCase):
    @staticmethod
    def quadratic_evaluator(params, **kwargs):
        objective = (params["Lr"] - 1.7) ** 2 + 0.01
        return {
            "chi_p": -0.002,
            "chi_m": -0.002 + objective * 0.002,
            "delta_chi": -objective * 0.002,
            "objective": objective,
            "energy": np.arange(24, dtype=float),
            "tracked_energy": np.arange(9, dtype=float),
            "tracked_level": np.arange(9),
            "assignment_score": np.ones(9),
        }

    def test_optimizer_finds_interior_minimum(self):
        result = optimize_resonator_inductance(
            {"Lr": 1.56, "Cr": 385.0},
            evaluator=self.quadratic_evaluator,
            lr_bounds=(0.8, 2.5),
            grid_size=9,
            xatol=1e-7,
        )

        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["optimal_lr"], 1.7, places=5)
        self.assertFalse(result["boundary_hit"])
        self.assertGreater(result["nfev"], 9)

    def test_optimizer_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "increasing"):
            optimize_resonator_inductance(
                {"Lr": 1.56, "Cr": 385.0},
                evaluator=self.quadratic_evaluator,
                lr_bounds=(2.5, 0.8),
            )
        with self.assertRaisesRegex(ValueError, "at least 3"):
            optimize_resonator_inductance(
                {"Lr": 1.56, "Cr": 385.0},
                evaluator=self.quadratic_evaluator,
                grid_size=2,
            )
        with self.assertRaisesRegex(ValueError, "Cr and Lr"):
            optimize_resonator_inductance(
                {"Lr": 1.56}, evaluator=self.quadratic_evaluator
            )

    def test_optimizer_reports_all_nonfinite_evaluations(self):
        def invalid_evaluator(params, **kwargs):
            return {"objective": np.inf}

        result = optimize_resonator_inductance(
            {"Lr": 1.56, "Cr": 385.0},
            evaluator=invalid_evaluator,
            grid_size=3,
        )

        self.assertFalse(result["success"])
        self.assertTrue(np.isnan(result["optimal_lr"]))
        self.assertEqual(result["nfev"], 3)


def deterministic_optimizer(params, **kwargs):
    return {
        "success": True,
        "optimal_lr": 1.7,
        "boundary_hit": False,
        "nfev": 5,
        "energy": np.arange(24, dtype=float),
        "tracked_energy": np.arange(9, dtype=float),
        "tracked_level": np.arange(9),
        "assignment_score": np.ones(9),
        "chi_p": -0.002,
        "chi_m": -0.0019,
        "delta_chi": -0.0001,
        "objective": 0.05,
    }


def deterministic_original_evaluator(params, **kwargs):
    if params["Lr"] != 1.56:
        raise AssertionError("original evaluation did not use nominal Lr")
    return {
        "chi_p": -0.003,
        "chi_m": -0.0028,
        "delta_chi": -0.0002,
    }


class TunableMonteCarloDatasetTests(unittest.TestCase):
    def test_dataset_records_frozen_parameters_and_optimizer_outputs(self):
        poi = [
            {
                "name": "test",
                "params": {"Ej1": 14.0, "Cr": 385.0, "Lr": 1.56},
                "flux": None,
            }
        ]

        ds = run_tunable_resonator_monte_carlo(
            poi,
            samples=2,
            cpus=1,
            seed=123,
            param_rel_sigma={"Ej1": 0.1, "Cr": 1.0, "Lr": 1.0},
            circuit_factory=lambda **params: params,
            flux_param=None,
            optimizer=deterministic_optimizer,
            evaluator=deterministic_original_evaluator,
        )

        np.testing.assert_array_equal(
            ds.param_value.sel(poi="test", param="Cr"), [385.0, 385.0]
        )
        np.testing.assert_array_equal(
            ds.param_value.sel(poi="test", param="Lr"), [1.56, 1.56]
        )
        np.testing.assert_array_equal(ds.optimal_lr, [[1.7, 1.7]])
        np.testing.assert_array_equal(ds.optimizer_success, [[True, True]])
        self.assertEqual(ds.attrs["frozen_params"], "Cr,Lr")
        self.assertEqual(ds["chi_p"].attrs["units"], "GHz")
        self.assertEqual(ds["optimal_lr"].attrs["units"], "nH")
        self.assertIn("abs_delta_chi_over_chi_p", ds)
        np.testing.assert_array_equal(ds.original_chi_p, [[-0.003, -0.003]])
        np.testing.assert_array_equal(ds.original_chi_m, [[-0.0028, -0.0028]])
        np.testing.assert_array_equal(ds.original_delta_chi, [[-0.0002, -0.0002]])
        expected_frequency = 1 / np.sqrt(1.7e-9 * 385.0e-15) / (2 * np.pi) / 1e9
        np.testing.assert_allclose(ds.resonator_frequency_ghz, expected_frequency)
        for name in (
            "original_chi_p",
            "original_chi_m",
            "original_delta_chi",
            "resonator_frequency_ghz",
        ):
            self.assertEqual(ds[name].attrs["units"], "GHz")

    def test_failed_optimization_has_nan_resonator_frequency(self):
        poi = [
            {
                "name": "test",
                "params": {"Ej1": 14.0, "Cr": 385.0, "Lr": 1.56},
                "flux": None,
            }
        ]

        def failed_optimizer(params, **kwargs):
            return {
                "success": False,
                "optimal_lr": np.nan,
                "boundary_hit": False,
                "nfev": 3,
            }

        ds = run_tunable_resonator_monte_carlo(
            poi,
            samples=1,
            cpus=1,
            circuit_factory=lambda **params: params,
            flux_param=None,
            optimizer=failed_optimizer,
            evaluator=deterministic_original_evaluator,
        )

        self.assertTrue(np.isnan(ds.resonator_frequency_ghz.item()))


class TunableMonteCarloCacheTests(unittest.TestCase):
    def test_cache_round_trip_preserves_optimal_lr(self):
        poi = [
            {
                "name": "test",
                "params": {"Ej1": 14.0, "Cr": 385.0, "Lr": 1.56},
                "flux": None,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tunable.nc"
            computed, returned_path = load_or_compute_tunable_resonator_monte_carlo(
                poi,
                path=path,
                force=True,
                samples=1,
                cpus=1,
                circuit_factory=lambda **params: params,
                flux_param=None,
                optimizer=deterministic_optimizer,
                evaluator=deterministic_original_evaluator,
            )
            loaded, loaded_path = load_or_compute_tunable_resonator_monte_carlo(
                poi, path=path, force=False
            )

            self.assertEqual(returned_path, path)
            self.assertEqual(loaded_path, path)
            np.testing.assert_array_equal(loaded.optimal_lr, computed.optimal_lr)
            for name in (
                "original_chi_p",
                "original_chi_m",
                "original_delta_chi",
                "resonator_frequency_ghz",
            ):
                np.testing.assert_array_equal(loaded[name], computed[name])

    def test_stale_cache_requires_force_recompute(self):
        poi = [
            {
                "name": "test",
                "params": {"Ej1": 14.0, "Cr": 385.0, "Lr": 1.56},
                "flux": None,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stale.nc"
            xr.Dataset({"optimal_lr": (("poi", "sample"), [[1.7]])}).to_netcdf(path)

            with self.assertRaisesRegex(ValueError, "force=True"):
                load_or_compute_tunable_resonator_monte_carlo(
                    poi, path=path, force=False
                )


SWEEP_LR_CALLS = []


def recording_sweep_evaluator(params, **kwargs):
    lr = params["Lr"]
    SWEEP_LR_CALLS.append(lr)
    return {
        "energy": np.arange(24, dtype=float) + lr,
        "tracked_energy": np.arange(9, dtype=float) + lr,
        "tracked_level": np.arange(9),
        "assignment_score": np.ones(9),
        "chi_p": -0.002 * lr,
        "chi_m": -0.0018 * lr,
        "delta_chi": -0.0002 * lr,
        "objective": 0.1,
    }


class SingleSampleResonatorSweepTests(unittest.TestCase):
    def test_sweep_reuses_one_sample_and_preserves_lr_order(self):
        SWEEP_LR_CALLS.clear()
        params = {"Ej1": 14.0, "Cr": 385.0, "Lr": 1.56}
        lr_values = np.array([2.0, 1.0, 1.5])

        ds = run_tunable_resonator_sample_sweep(
            params,
            lr_values,
            seed=123,
            cpus=1,
            param_rel_sigma={"Ej1": 0.1, "Cr": 1.0, "Lr": 1.0},
            circuit_factory=lambda **kwargs: kwargs,
            flux_param=None,
            evaluator=recording_sweep_evaluator,
        )

        np.testing.assert_array_equal(SWEEP_LR_CALLS, lr_values)
        np.testing.assert_array_equal(ds.lr, lr_values)
        self.assertEqual(ds.sampled_param.sel(param="Cr").item(), 385.0)
        self.assertEqual(ds.sampled_param.sel(param="Lr").item(), 1.56)
        expected = 1 / np.sqrt(lr_values * 1e-9 * 385e-15) / (2 * np.pi) / 1e9
        np.testing.assert_allclose(ds.resonator_frequency_ghz, expected)
        np.testing.assert_allclose(ds.delta_chi, -0.0002 * lr_values)
        self.assertEqual(ds.lr.attrs["units"], "nH")
        self.assertEqual(ds.delta_chi.attrs["units"], "GHz")
        self.assertEqual(ds.abs_delta_chi_mhz.attrs["units"], "MHz")
        self.assertEqual(ds.resonator_frequency_ghz.attrs["units"], "GHz")
        self.assertEqual(ds.energy.dims, ("lr", "level"))
        self.assertEqual(ds.tracked_energy.dims, ("lr", "branch"))

    def test_parallel_sweep_preserves_lr_order(self):
        params = {"Ej1": 14.0, "Cr": 385.0, "Lr": 1.56}
        lr_values = np.array([2.0, 1.0, 1.5])

        ds = run_tunable_resonator_sample_sweep(
            params,
            lr_values,
            seed=123,
            cpus=2,
            param_rel_sigma={"Ej1": 0.1},
            circuit_factory=lambda **kwargs: kwargs,
            flux_param=None,
            evaluator=recording_sweep_evaluator,
        )

        np.testing.assert_array_equal(ds.lr, lr_values)
        np.testing.assert_allclose(ds.delta_chi, -0.0002 * lr_values)
        self.assertEqual(ds.attrs["cpus"], 2)

    def test_sweep_rejects_invalid_cpu_count(self):
        with self.assertRaisesRegex(ValueError, "cpus must be at least 1"):
            run_tunable_resonator_sample_sweep(
                {"Cr": 385.0, "Lr": 1.56},
                [1.0],
                cpus=0,
            )

    def test_sweep_rejects_invalid_lr_values(self):
        params = {"Cr": 385.0, "Lr": 1.56}
        invalid_values = ([], [[1.0]], [1.0, np.nan], [1.0, 0.0], [-1.0])

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    run_tunable_resonator_sample_sweep(
                        params,
                        values,
                        circuit_factory=lambda **kwargs: kwargs,
                        flux_param=None,
                        evaluator=recording_sweep_evaluator,
                    )

    def test_sweep_requires_resonator_parameters(self):
        with self.assertRaisesRegex(ValueError, "Cr and Lr"):
            run_tunable_resonator_sample_sweep(
                {"Lr": 1.56},
                [1.0],
                circuit_factory=lambda **kwargs: kwargs,
                flux_param=None,
                evaluator=recording_sweep_evaluator,
            )


if __name__ == "__main__":
    unittest.main()
