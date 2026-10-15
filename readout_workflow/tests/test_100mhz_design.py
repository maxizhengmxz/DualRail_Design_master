import unittest
from model import Basis,Device,ReadoutModel
from study_dispersive import diagnostics


class FeasibleDesignTests(unittest.TestCase):
    def test_six_ff_design_meets_energy_and_dispersive_constraints(self):
        # Rounded copyable design, checked with a larger basis than the notebook.
        d=Device(C1_shunt_fF=115,C2_shunt_fF=80,C12_fF=2.460963,
                 C1r_fF=6,C2r_fF=6,Cr_fF=530.5164769729845,Lr_nH=1.3262911924324612)
        m=ReadoutModel(d,Basis(ncut=20,transmon_levels=9,resonator_levels=8,eigenlevels=30))
        ej1,ej2=16.607386,12.238665
        p=m.evaluate(ej1,ej2)
        p.update(diagnostics(m,p))
        self.assertLess(max(p["abs_delta_chi_over_chi_p"],p["ratio_m"]),.01)
        self.assertLess(p["lambda_max"],.1)
        self.assertAlmostEqual(p["splitting_ghz"],.1,delta=1e-5)
        self.assertAlmostEqual(p["operating_frequency_ghz"],4.4,delta=1e-5)
        self.assertGreater(p["hybridization"],.95)
        self.assertGreater(p["min_assignment_overlap"],.95)
        for ej,ec in zip((ej1,ej2),m.ec_ghz[:2]):
            self.assertLess(ej,25)
            self.assertGreaterEqual(ec,.15)
            self.assertLessEqual(ec,.35)
            self.assertGreater(ej/ec,50)


if __name__ == "__main__":
    unittest.main()
