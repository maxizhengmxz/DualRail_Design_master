import unittest

from threadpoolctl import threadpool_limits
from model import Basis, Device, ReadoutModel
from study_dispersive import diagnostics


class EqualCouplingDesignTests(unittest.TestCase):
    def test_rounded_design_has_equal_g_and_meets_all_constraints(self):
        device=Device(C1_shunt_fF=118,C2_shunt_fF=78,C12_fF=2.430402617,
            C1r_fF=6,C2r_fF=4.933679823,Cr_fF=530.516476973,
            Lr_nH=1.326291192432)
        with threadpool_limits(limits=1):
            model=ReadoutModel(device,Basis(20,9,8,30))
            ej1,ej2=16.979628090,11.850355554
            p=model.evaluate(ej1,ej2)
            p.update(diagnostics(model,p))
        self.assertAlmostEqual(p["g1r_over_2pi_mhz"],p["g2r_over_2pi_mhz"],delta=1e-5)
        self.assertLess(max(p["ratio_m"],p["abs_delta_chi_over_chi_p"]),.01)
        self.assertLessEqual(p["lambda_max"],.1)
        self.assertAlmostEqual(p["operating_frequency_ghz"],4.4,delta=1e-5)
        self.assertAlmostEqual(p["splitting_ghz"],.1,delta=1e-5)
        self.assertGreaterEqual(p["hybridization"],.95)
        self.assertGreaterEqual(p["min_assignment_overlap"],.95)
        for ej,ec in zip((ej1,ej2),model.ec_ghz[:2]):
            self.assertLess(ej,25)
            self.assertGreaterEqual(ec,.15)
            self.assertLessEqual(ec,.35)
            self.assertGreater(ej/ec,50)


if __name__ == "__main__":
    unittest.main()
