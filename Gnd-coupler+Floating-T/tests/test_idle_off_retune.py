import unittest
from calibrate_idle import retune_off


class IdleOffRetuneTests(unittest.TestCase):
    def test_free_idle_motion_can_move_root_outside_local_bracket(self):
        root,record=retune_off(lambda f:f-9.2,9.8,(7.001,9.9999))
        self.assertAlmostEqual(root,9.2)
        self.assertEqual(record['scope'],'full_authorized_interval')

    def test_never_accepts_root_above_hard_bound(self):
        with self.assertRaises(ValueError):
            retune_off(lambda f:f-10.1,9.8,(7.001,9.9999))

    def test_local_branch_still_used_first(self):
        root,record=retune_off(lambda f:f-9.81,9.8,(7.001,9.9999))
        self.assertAlmostEqual(root,9.81)
        self.assertEqual(record['scope'],'local')

    def test_outside_seed_never_queries_outside_authorized_interval(self):
        def exchange(f):
            self.assertTrue(7.001<=f<=9.9999)
            return f-9.2
        root,_=retune_off(exchange,12.,(7.001,9.9999))
        self.assertAlmostEqual(root,9.2)
