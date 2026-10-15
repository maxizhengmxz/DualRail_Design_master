import unittest
from reordered_robustness import TIERS,reasons


class ReorderedRescueChecks(unittest.TestCase):
    def point(self):
        return dict(ratio=.009,lambda_max=.09,hybridization=.96,splitting_ghz=.10,
            loaded_ec1_mhz=130.,loaded_ec2_mhz=220.,ej1_ghz=15.,ej2_ghz=14.,
            ej1_over_ec1=55.,ej2_over_ec2=60.,min_assignment_overlap=.97,operating_frequency_ghz=4.5)
    def test_EC_first(self):
        p=self.point()
        self.assertEqual(reasons(p,TIERS[0],4.5),['EC_range'])
        self.assertEqual(reasons(p,TIERS[1],4.5),[])
        p['loaded_ec1_mhz']=119.9
        self.assertIn('EC_range',reasons(p,TIERS[5],4.5))
    def test_gap_before_H(self):
        p=self.point();p['splitting_ghz']=.085
        self.assertIn('logical_gap',reasons(p,TIERS[2],4.5))
        self.assertEqual(reasons(p,TIERS[3],4.5),[])
        p['hybridization']=.92
        self.assertIn('hybridization',reasons(p,TIERS[3],4.5))
        self.assertEqual(reasons(p,TIERS[4],4.5),[])
    def test_operation_frequency_and_hard_screens(self):
        p=self.point();p['operating_frequency_ghz']=4.6
        self.assertIn('fixed_operation_frequency',reasons(p,TIERS[1],4.5))
        self.assertEqual(reasons(p,TIERS[2],4.5),[])
        p['lambda_max']=.101
        self.assertIn('dispersive',reasons(p,TIERS[5],4.5))


if __name__=='__main__': unittest.main()
