"""Target-domain selection for the direct loaded-EC feasibility driver."""

import unittest

import numpy as np

from readout_workflow.direct_ec_feasibility import logical_range_ghz,mirrored_target_grid,target_grid


class DirectECTargetGridTests(unittest.TestCase):
    def test_rectangle_keeps_points_outside_triangular_hull(self):
        witnesses=np.array([(150,200),(160,200),(150,210)],float)
        rectangle=target_grid(witnesses,'rectangle')
        hull=target_grid(witnesses,'hull')
        self.assertEqual(len(rectangle),9)
        self.assertEqual(len(hull),6)
        self.assertIn((160,210),rectangle)
        self.assertNotIn((160,210),hull)

    def test_invalid_domain_is_rejected(self):
        with self.assertRaises(ValueError):
            target_grid(np.array([(150,200),(160,200),(150,210)]),'unknown')

    def test_scaled_hull_expands_and_respects_ec_limits(self):
        witnesses=np.array([(170,220),(210,220),(170,260)],float)
        original=set(target_grid(witnesses,'hull',hull_scale=1.0))
        expanded=set(target_grid(witnesses,'hull',hull_scale=1.5))
        self.assertTrue(original < expanded)
        self.assertTrue(all(150<=value<400 for point in expanded for value in point))

    def test_finer_step_increases_target_density(self):
        witnesses=np.array([(170,220),(210,220),(170,260)],float)
        coarse=target_grid(witnesses,'hull',step_mhz=5,hull_scale=1.5)
        fine=target_grid(witnesses,'hull',step_mhz=2,hull_scale=1.5)
        self.assertGreater(len(fine),len(coarse))
        self.assertTrue(all(value%2==0 for point in fine for value in point))

    def test_hull_scale_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            target_grid(np.array([(150,200),(160,200),(150,210)]),'hull',hull_scale=.9)

    def test_full_orientation_is_union_not_combined_convex_hull(self):
        witnesses=np.array([(155,210),(165,210),(155,230)],float)
        targets,original,mirrored=mirrored_target_grid(witnesses,hull_scale=1.5)
        self.assertEqual(set(targets),original|mirrored)
        self.assertIn((155,210),original)
        self.assertIn((210,155),mirrored)
        self.assertIn((155,210),targets)
        self.assertIn((210,155),targets)

    def test_logical_frequency_interval_conversion(self):
        self.assertEqual(logical_range_ghz((80,100)),(0.08,0.1))
        for invalid in ((100,80),(0,100),(90,90)):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                logical_range_ghz(invalid)


if __name__=='__main__':unittest.main()
