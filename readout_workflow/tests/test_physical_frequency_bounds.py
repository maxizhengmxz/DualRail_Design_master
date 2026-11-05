import unittest

from readout_workflow.physical_frequency_bounds import connected_intervals


class ConnectedIntervalsTests(unittest.TestCase):
    def test_disconnected_pass_runs_remain_separate(self):
        rows=[{"target_frequency_ghz":f,"status":s,"component":None} for f,s in
              [(4.0,"validated_pass"),(4.1,"validated_pass"),(4.2,"fail"),
               (4.3,"validated_pass"),(4.4,"fail")]]
        intervals=connected_intervals(rows)
        self.assertEqual([(i["sampled_lower_ghz"],i["sampled_upper_ghz"]) for i in intervals],[(4.0,4.1),(4.3,4.3)])
        self.assertEqual([r["component"] for r in rows],[1,1,None,2,None])


if __name__ == "__main__": unittest.main()
