import unittest
from dataclasses import asdict
import numpy as np
from model import Device,ReadoutModel
from robustness_rescue import perturb,TIERS,reasons


class RescueChecks(unittest.TestCase):
    def test_frequency_and_impedance(self):
        d=Device();reference={'device':asdict(d)}
        for error in (-.02,.02):
            q=perturb(reference,[0,0,0,0,0,error])
            self.assertAlmostEqual(np.sqrt(q.Lr_nH/q.Cr_fF),np.sqrt(d.Lr_nH/d.Cr_fF),places=13)
            self.assertAlmostEqual(np.sqrt(d.Lr_nH*d.Cr_fF/(q.Lr_nH*q.Cr_fF)),1+error,places=13)
    def test_readout_caps_independent(self):
        d=Device();q=perturb({'device':asdict(d)},[0,0,0,.02,-.02,0])
        self.assertAlmostEqual(q.C1r_fF,1.02*d.C1r_fF)
        self.assertAlmostEqual(q.C2r_fF,.98*d.C2r_fF)
    def test_rescue_ladder(self):
        p=dict(ratio=.009,lambda_max=.09,hybridization=.92,splitting_ghz=.085,
            loaded_ec1_mhz=180,loaded_ec2_mhz=220,ej1_ghz=15,ej2_ghz=14,
            ej1_over_ec1=45,ej2_over_ec2=55,min_assignment_overlap=.97,operating_frequency_ghz=4.5)
        self.assertEqual(reasons(p,TIERS[4],4.5),[])
        self.assertIn('hybridization',reasons(p,TIERS[1],4.5))
        self.assertIn('logical_gap',reasons(p,TIERS[2],4.5))
        self.assertIn('EJ_over_EC',reasons(p,TIERS[3],4.5))
        p['ratio']=.0101
        self.assertIn('chi_matching',reasons(p,TIERS[4],4.5))


if __name__=='__main__': unittest.main()
