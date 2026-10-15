import unittest
from dataclasses import replace

from threadpoolctl import threadpool_limits
from model import Device, ReadoutModel
from study_capacitance_robustness import CarrierSlice


class RecalibrationDetuningTests(unittest.TestCase):
    def test_capacitance_error_recenters_the_hybridization_search(self):
        device=Device(C1_shunt_fF=118,C2_shunt_fF=78,C12_fF=2.430402617,
            C1r_fF=6,C2r_fF=4.933679823,Cr_fF=530.516476973,
            Lr_nH=1.326291192432)
        with threadpool_limits(limits=1):
            reference=ReadoutModel(device).evaluate(16.979628090,11.850355554)
            perturbed=ReadoutModel(replace(device,C2_shunt_fF=78*1.02))
            sl=CarrierSlice(perturbed,reference)
            peak=sl.maximum()
            chosen=sl.constrained()
        self.assertGreater(sl.reference["ej2_ghz"]-reference["ej2_ghz"],.18)
        self.assertGreater(peak["hybridization"],.98)
        self.assertIsNotNone(chosen)
        self.assertGreaterEqual(chosen["hybridization"],.95)
        ratio=abs(chosen["delta_chi_ghz"])/min(abs(chosen["chi_m_ghz"]),abs(chosen["chi_p_ghz"]))
        self.assertLessEqual(ratio,.0090001)

    def test_chi_constraint_moves_ejs_away_from_maximum_hybridization(self):
        device=Device(C1_shunt_fF=118,C2_shunt_fF=78,C12_fF=2.430402617,
            C1r_fF=6,C2r_fF=4.933679823,Cr_fF=530.516476973,
            Lr_nH=1.326291192432)
        with threadpool_limits(limits=1):
            sl=CarrierSlice(ReadoutModel(device),
                {"ej1_ghz":16.979628090,"ej2_ghz":11.850355554})
            peak=sl.maximum()
            chosen=sl.constrained(limit=.009)
        self.assertIsNotNone(chosen)
        self.assertGreater(peak["abs_delta_chi_over_chi_p"],.01)
        self.assertLess(chosen["hybridization"],peak["hybridization"]-.02)
        self.assertGreaterEqual(chosen["hybridization"],.95)
        ratio=abs(chosen["delta_chi_ghz"])/min(abs(chosen["chi_m_ghz"]),abs(chosen["chi_p_ghz"]))
        self.assertLessEqual(ratio,.0090001)
        self.assertGreater(abs(chosen["ej2_ghz"]-peak["ej2_ghz"]),.01)
        self.assertAlmostEqual(chosen["operating_frequency_ghz"],4.4,delta=1e-7)


if __name__ == "__main__": unittest.main()
