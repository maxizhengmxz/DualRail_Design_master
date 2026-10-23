import tempfile
import unittest
from pathlib import Path

import numpy as np

from legacy.analysis.sweeps import (
    _target_scales,
    load_or_compute_two_param_sweep,
    run_two_param_sweep,
)


class FakeSweep:
    def __init__(self, energy_table):
        self.energy_table = energy_table
        self.state_table = None


class FakeCircuit:
    def __init__(self, **params):
        self.params = params

    def get_spectrum_vs_paramvals(
        self,
        flux_param,
        param_vals,
        *,
        evals_count,
        get_eigenstates,
        num_cpus,
    ):
        base = (
            self.params.get("left", 0.0)
            + 10.0 * self.params.get("right", 0.0)
            + float(param_vals[0])
        )
        return FakeSweep(np.arange(evals_count, dtype=float)[None, :] + base)


def fake_factory(**params):
    return FakeCircuit(**params)


class TwoParamSweepTests(unittest.TestCase):
    def test_target_scales_from_none_sequence_and_mapping(self):
        self.assertEqual(_target_scales("C2", None), {"C2": 1.0})
        self.assertEqual(_target_scales("axis", ["C1", "C2"]), {"C1": 1.0, "C2": 1.0})
        self.assertEqual(_target_scales("axis", {"C1": 2.0}), {"C1": 2.0})

    def test_run_two_param_sweep_records_energy_and_scaled_coords(self):
        ds = run_two_param_sweep(
            gen_dtc=fake_factory,
            param1_name="x",
            param1_values=[1.0, 2.0],
            param1_targets={"left": 2.0},
            param2_name="y",
            param2_values=[3.0],
            param2_targets={"right": 0.5},
            fixed_flux=0.25,
            evals_count=4,
            cpus=1,
        )

        self.assertEqual(ds.energy.dims, ("x", "y", "level"))
        np.testing.assert_allclose(ds.left, [2.0, 4.0])
        np.testing.assert_allclose(ds.right, [1.5])
        np.testing.assert_allclose(ds.energy[0, 0], [17.25, 18.25, 19.25, 20.25])
        self.assertEqual(ds.attrs["sweep_param_1_targets"], "left")
        self.assertEqual(ds.attrs["sweep_param_2_targets"], "right")

    def test_rejects_invalid_axis_configuration(self):
        with self.assertRaisesRegex(ValueError, "required"):
            run_two_param_sweep(gen_dtc=fake_factory, param1_values=[1.0])
        with self.assertRaisesRegex(ValueError, "must differ"):
            run_two_param_sweep(
                gen_dtc=fake_factory,
                param1_name="x",
                param1_values=[1.0],
                param2_name="x",
                param2_values=[2.0],
            )
        with self.assertRaisesRegex(ValueError, "cannot be shared"):
            run_two_param_sweep(
                gen_dtc=fake_factory,
                param1_name="x",
                param1_values=[1.0],
                param1_targets=["shared"],
                param2_name="y",
                param2_values=[2.0],
                param2_targets=["shared"],
            )

    def test_load_or_compute_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "two-param.nc"
            computed, computed_path = load_or_compute_two_param_sweep(
                path=path,
                force=True,
                gen_dtc=fake_factory,
                param1_name="x",
                param1_values=[1.0],
                param2_name="y",
                param2_values=[2.0],
                evals_count=3,
                cpus=1,
            )
            loaded, loaded_path = load_or_compute_two_param_sweep(path=path)

            self.assertEqual(computed_path, path)
            self.assertEqual(loaded_path, path)
            np.testing.assert_allclose(loaded.energy, computed.energy)


if __name__ == "__main__":
    unittest.main()
