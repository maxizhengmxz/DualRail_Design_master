import unittest
from dataclasses import replace
import numpy as np
from model import Basis, Device, ReadoutModel, find_readout_points


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.basis = Basis(ncut=10, transmon_levels=4, resonator_levels=4, eigenlevels=16)

    def test_capacitance_matrix_has_no_junction_parasitics(self):
        d = Device()
        c = d.capacitance_matrix()*1e15
        self.assertAlmostEqual(c[0, 0], d.C1_shunt_fF+d.C12_fF+d.C1r_fF)
        self.assertAlmostEqual(c[1, 1], d.C2_shunt_fF+d.C12_fF+d.C2r_fF)
        np.testing.assert_allclose(c, c.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(c) > 0))

    def test_disconnected_resonator_has_zero_conditional_shifts(self):
        d = replace(Device(), C1r_fF=0, C2r_fF=0)
        p = ReadoutModel(d, self.basis).evaluate(16, 18)
        self.assertAlmostEqual(p["chi_p_ghz"], 0, places=10)
        self.assertAlmostEqual(p["chi_m_ghz"], 0, places=10)
        self.assertGreater(p["min_assignment_overlap"], 0.999)

    def test_identical_rails_at_resonance_are_strongly_hybridized(self):
        d = replace(Device(), C2_shunt_fF=Device().C1_shunt_fF, C1r_fF=0, C2r_fF=0)
        p = ReadoutModel(d, self.basis).evaluate(16, 16)
        # Counter-rotating terms change the two mode transition strengths, so
        # this strength-based score need not be exactly one even at symmetry.
        self.assertGreater(p["hybridization"], 0.95)
        self.assertGreater(p["splitting_ghz"], 0)

    def test_swapping_transmons_preserves_dressed_observables(self):
        d = Device()
        swapped = replace(d, C1_shunt_fF=d.C2_shunt_fF, C2_shunt_fF=d.C1_shunt_fF,
                          C1r_fF=d.C2r_fF, C2r_fF=d.C1r_fF)
        a = ReadoutModel(d, self.basis).evaluate(16, 18)
        b = ReadoutModel(swapped, self.basis).evaluate(18, 16)
        for key in ("splitting_ghz", "operating_frequency_ghz", "chi_p_ghz", "chi_m_ghz"):
            self.assertAlmostEqual(a[key], b[key], places=9)

    def test_search_refines_resonance_and_rejects_boundaries(self):
        class AnalyticModel:
            def evaluate(self, e1, e2):
                return {"ej1_ghz": e1, "ej2_ghz": e2,
                        "splitting_ghz": np.sqrt((e1-e2)**2+0.01),
                        "hybridization": 1.0}
        points, scan = find_readout_points(AnalyticModel(), [1., 2., 3., 4.], [2.2, 6.])
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0]["ej1_ghz"], 2.2, places=3)
        self.assertEqual(len(scan), 8)


if __name__ == "__main__":
    unittest.main()
