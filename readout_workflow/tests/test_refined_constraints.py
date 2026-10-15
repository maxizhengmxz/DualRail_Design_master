import unittest
from threadpoolctl import threadpool_limits
from model import Basis
from refined_pair_search import failures, inverse_frequency, point_from_x, search


class RefinedConstraintsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=[4.161521873743831,.3843682749421451,2.5387728221387356,9.284433553084366]
        with threadpool_limits(limits=1):
            cls.point=point_from_x(104,92,cls.x,Basis(20,9,8,40))

    def test_physical_witness_uses_the_new_gap_window(self):
        self.assertGreaterEqual(self.point["splitting_ghz"],.09)
        self.assertLess(self.point["splitting_ghz"],.095)
        self.assertEqual(failures(self.point),[])

    def test_charging_energy_upper_limit_is_strict(self):
        point=dict(self.point,loaded_ec2_mhz=400.)
        self.assertIn("EC",failures(point))
        point["loaded_ec2_mhz"]=399.999
        self.assertNotIn("EC",failures(point))

    def test_search_excludes_the_diagonal_and_mirrored_half(self):
        for pair in ((80,80),(79,80)):
            with self.assertRaises(ValueError): search({"pair":pair})

    def test_physical_label_exchange_preserves_observables(self):
        x=list(self.x);x[1]*=-1
        with threadpool_limits(limits=1):
            reflected=point_from_x(92,104,x,Basis(20,9,8,40))
        for key in ("ratio","hybridization","lambda_max","splitting_ghz","operating_frequency_ghz"):
            self.assertAlmostEqual(self.point[key],reflected[key],delta=2e-8)
        self.assertAlmostEqual(self.point["loaded_ec1_mhz"],reflected["loaded_ec2_mhz"],delta=1e-8)


if __name__=="__main__": unittest.main()
