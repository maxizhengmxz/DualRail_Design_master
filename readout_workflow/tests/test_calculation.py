"""Physical conventions and acceptance rules of the consolidated batch model."""

import unittest
from dataclasses import replace

import numpy as np
from threadpoolctl import threadpool_limits

from readout_workflow import calculation as calc


class CalculationTests(unittest.TestCase):
    def test_capacitance_matrix_and_loaded_ec(self):
        d = calc.Device()
        c = d.capacitance_matrix() * 1e15
        self.assertAlmostEqual(c[0, 0], d.C1_shunt_fF + d.C12_fF + d.C1r_fF)
        self.assertAlmostEqual(c[1, 1], d.C2_shunt_fF + d.C12_fF + d.C2r_fF)
        np.testing.assert_allclose(c, c.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(c) > 0))
        model = calc.ReadoutModel(d)
        np.testing.assert_allclose(
            model.ec_ghz, calc.e**2 * np.diag(model.cinv) / (2 * calc.h * 1e9)
        )

    def test_matching_uses_signed_mean(self):
        p = dict(chi_p_ghz=-0.001, chi_m_ghz=-0.00101)
        self.assertAlmostEqual(calc.chi_ratio(p), 0.00001 / 0.001005)
        self.assertAlmostEqual(calc.absolute_mean_chi_mhz(p), 1.005)
        p["chi_m_ghz"] = 0.001
        self.assertEqual(calc.absolute_mean_chi_mhz(p), 0)
        self.assertTrue(np.isinf(calc.chi_ratio(p)))

    def test_transmon_label_symmetry(self):
        d = calc.Device()
        other = replace(
            d,
            C1_shunt_fF=d.C2_shunt_fF,
            C2_shunt_fF=d.C1_shunt_fF,
            C1r_fF=d.C2r_fF,
            C2r_fF=d.C1r_fF,
        )
        with threadpool_limits(limits=1):
            a = calc.ReadoutModel(d).evaluate(14, 12)
            b = calc.ReadoutModel(other).evaluate(12, 14)
        for name in (
            "chi_p_ghz",
            "chi_m_ghz",
            "operating_frequency_ghz",
            "splitting_ghz",
            "hybridization",
        ):
            self.assertAlmostEqual(a[name], b[name], places=9)

    def test_strict_energy_bounds_on_fresh_circuit(self):
        with threadpool_limits(limits=1):
            p = calc.annotate(
                calc.ReadoutModel(calc.Device(), calc.CONVERGENCE_BASIS),
                14,
                12,
            )
        self.assertTrue(np.isfinite(calc.absolute_mean_chi_mhz(p)))
        self.assertIn("EC_range", calc.failures(p | dict(loaded_ec1_mhz=400)))
        self.assertIn("EJ_limit", calc.failures(p | dict(ej1_ghz=25)))
        self.assertIn("EJ_over_EC", calc.failures(p | dict(ej1_over_ec1=50)))

    def test_frequency_perturbation_preserves_impedance(self):
        d = calc.Device()
        p = dict(device={k: getattr(d, k) for k in d.__dataclass_fields__})
        other = calc.perturb(p, [0, 0, 0, 0, 0, 0.02])
        self.assertAlmostEqual(other.Lr_nH / other.Cr_fF, d.Lr_nH / d.Cr_fF)
        self.assertAlmostEqual(
            np.sqrt(d.Lr_nH * d.Cr_fF / (other.Lr_nH * other.Cr_fF)), 1.02
        )


if __name__ == "__main__":
    unittest.main()
