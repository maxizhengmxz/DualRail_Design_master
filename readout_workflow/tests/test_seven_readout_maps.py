"""Sparse/dense solver equivalence at identical tiny synthetic truncations.

The small basis tests the eigensolver, not production readout convergence.
"""

import unittest
import numpy as np
from readout_workflow.seven_readout_shift_map import solve_point
from core.transmon import Basis
from floating_coupler.model import Model, FULL
from floating_coupler.design import ej_for_frequency


class SevenMapSolverTests(unittest.TestCase):
    def test_matching_tracks_do_not_jump_when_one_branch_disappears(self):
        from readout_workflow.readout_shift_map import matching_band_tracks

        bands = [
            dict(intervals_ej1_ghz=[[1, 1.01], [4, 4.01]]),
            dict(intervals_ej1_ghz=[[4.1, 4.11]]),
            dict(intervals_ej1_ghz=[[1.2, 1.21], [4.2, 4.21]]),
        ]
        tracks = matching_band_tracks(bands, 0.5)
        self.assertEqual(len(tracks), 3)
        self.assertTrue(np.isnan(tracks[0][1:]).all())
        np.testing.assert_allclose(tracks[1][:, 0], [4, 4.1, 4.2])
        self.assertTrue(np.isnan(tracks[2][:2]).all())

    def test_estimated_narrow_band_keeps_both_boundaries(self):
        from readout_workflow.plot_seven_readout_maps import intervals

        band = intervals(
            np.array([0.0, 1.0, 2.0]), np.array([-1.0, 0.0, 1.0]), np.ones(3)
        )
        np.testing.assert_allclose(band, [[0.99, 1.01]])

    def test_sparse_and_dense_agree_for_both_spectators_and_rails(self):
        ec = np.array([0.165, 0.25, 0.45, 0.16, 0.26])
        f = [4.61, 4.62, 9.9, 5.42, 5.39]
        ej = np.array([ej_for_frequency(fr, c, 25) for fr, c in zip(f, ec)])
        j = np.zeros((5, 5))
        for a, b, value in [
            (0, 1, 0.04),
            (3, 4, 0.025),
            (1, 2, 0.12),
            (2, 3, 0.15),
            (1, 3, 0.003),
        ]:
            j[a, b] = j[b, a] = value
        model = Model(FULL, ec, ej, j, Basis(ncut=25, transmon_levels=4), {})
        physical = dict(
            K_per_ff=np.eye(7).tolist(),
            resonator_parameters={
                "r1": dict(
                    fr_ghz=5.99,
                    gamma_ghz_per_n=[0.02, -0.025, 0.001, 0.0001, -0.0001],
                    q_zpf_coulomb=1e-19,
                ),
                "r2": dict(
                    fr_ghz=6.01,
                    gamma_ghz_per_n=[0.0001, -0.0001, 0.001, 0.018, -0.022],
                    q_zpf_coulomb=1e-19,
                ),
            },
        )
        basis = dict(local_levels=4, pair_states=6, coupler_levels=3, ncut=25)
        for pair in ("A", "B"):
            fast = solve_point(model, physical, 9.9, pair, basis)
            dense = solve_point(model, physical, 9.9, pair, basis, dense=True)
            np.testing.assert_allclose(fast[0], dense[0], atol=1e-7, rtol=0)
            np.testing.assert_allclose(fast[1], dense[1], atol=1e-8, rtol=0)
            self.assertEqual(fast[0].shape, (2, 3))
