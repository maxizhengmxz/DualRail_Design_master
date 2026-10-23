import copy
import importlib
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from core.transmon import Basis, transmon_spectrum
from transmon_cz import simulation as sim
from transmon_cz.gate_search import _collision_detuning, find_gate_operating_points


def raw_model():
    return dict(
        transmons={label: dict(EC_GHz=ec, EJ_GHz=ej, ng=0.0)
                   for label, ec, ej in zip(sim.LABELS, (.2, .35, .21, .36), (20., 21., 19., 20.))},
        couplings_GHz=dict(A1_A2=.08, B1_B2=.08, A2_B1=.015),
        basis=dict(ncut=12, levels_per_transmon=3),
    )


def system(raw):
    return sim.FourTransmonSystem(
        {k: sim.TransmonSpec(**v) for k, v in raw["transmons"].items()},
        raw["couplings_GHz"], sim.BasisSpec(**raw["basis"]),
    )


class TransmonCZTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=1)

    def tearDown(self):
        self.threads.restore_original_limits()

    def test_shared_local_spectrum_and_conversion(self):
        local = sim.build_local_model("test", sim.TransmonSpec(.25, 17), sim.BasisSpec(20, 4))
        energies, charge = transmon_spectrum(17, .25, Basis(20, 4, 3, 12))
        np.testing.assert_allclose(local.levels_GHz, energies, atol=1e-11)
        np.testing.assert_allclose(abs(local.n_op.full()), abs(charge), atol=1e-11)
        j = sim.mutual_capacitance_to_charge_coupling_from_EC_GHz(2.3, .2, .35)
        self.assertAlmostEqual(sim.charge_coupling_to_mutual_capacitance_from_EC_GHz(j, .2, .35), 2.3)

    def test_coherent_logical_map_and_population_closure(self):
        device = system(raw_model())
        result = sim.simulate_logical_basis(device, device.logical_basis(), sim.EvolutionSpec(10, 21))
        np.testing.assert_allclose(result["logical_maps"][0], np.eye(4), atol=1e-11)
        np.testing.assert_allclose(
            result["average_correct_logical_population_vs_time"]
            + result["average_wrong_logical_population_vs_time"]
            + result["average_nonlogical_leakage_vs_time"], 1, atol=1e-11,
        )
        for pair in ("A", "B"):
            self.assertTrue(0 <= result[f"H_{pair}"] <= 1)
            self.assertGreater(result[f"logical_frequency_{pair}_GHz"], 0)

    def test_search_solves_constraints_with_four_distinct_ecs(self):
        raw = raw_model()
        saved = copy.deepcopy(raw)
        device = system(raw)
        target_a = device.local["A1"].levels_GHz[1] - device.local["A2"].levels_GHz[1]
        target_b = device.local["B1"].levels_GHz[1] - device.local["B2"].levels_GHz[1]
        target_cz = _collision_detuning(device, "0110_to_0200")
        found = find_gate_operating_points(
            raw, [16, 18, 20, 22, 24], minimum_EJ_over_EC=0,
            target_a_detuning_GHz=target_a, target_b_detuning_GHz=target_b,
            target_cz_detuning_GHz=target_cz,
        )
        self.assertGreater(found.sizes["gate_operating_point"], 0)
        for label, spec in raw["transmons"].items():
            self.assertEqual(found.attrs[f"EC_{label}_GHz"], spec["EC_GHz"])
        row = found.isel(gate_operating_point=int(np.argmin(abs(found.EJ_A1_GHz.values-20))))
        for label in sim.LABELS:
            self.assertAlmostEqual(float(row[f"EJ_{label}_GHz"]), raw["transmons"][label]["EJ_GHz"], places=6)
        self.assertEqual(raw, saved)

    def test_import_sweeps_does_not_execute_calculations(self):
        from unittest.mock import patch
        with patch("pathlib.Path.mkdir", side_effect=AssertionError("import wrote a directory")), \
             patch("numpy.load", side_effect=AssertionError("import loaded saved results")):
            importlib.reload(importlib.import_module("transmon_cz.experiments.sweeps"))

    def test_no_crossing_is_explicit(self):
        self.assertIsNone(sim.first_odd_pi_index(np.zeros(10)))
