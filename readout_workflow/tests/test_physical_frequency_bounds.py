import unittest
from unittest.mock import patch

from readout_workflow.physical_frequency_bounds import (
    connected_intervals,
    point_bare_resonator_ghz,
    require_bare_resonator,
    solve_frequency,
)


class ConnectedIntervalsTests(unittest.TestCase):
    def test_disconnected_pass_runs_remain_separate(self):
        rows=[{"target_frequency_ghz":f,"status":s,"component":None} for f,s in
              [(4.0,"validated_pass"),(4.1,"validated_pass"),(4.2,"fail"),
               (4.3,"validated_pass"),(4.4,"fail")]]
        intervals=connected_intervals(rows)
        self.assertEqual([(i["sampled_lower_ghz"],i["sampled_upper_ghz"]) for i in intervals],[(4.0,4.1),(4.3,4.3)])
        self.assertEqual([r["component"] for r in rows],[1,1,None,2,None])

    def test_bare_resonator_provenance_check(self):
        frequency = 5.95
        impedance = 50.0
        omega = 2 * 3.141592653589793 * frequency * 1e9
        point = {"device": {
            "Cr_fF": 1 / (omega * impedance) * 1e15,
            "Lr_nH": impedance / omega * 1e9,
        }}
        self.assertAlmostEqual(point_bare_resonator_ghz(point), frequency)
        self.assertAlmostEqual(require_bare_resonator(point, frequency), frequency)
        with self.assertRaisesRegex(RuntimeError, "expected 6.05"):
            require_bare_resonator(point, 6.05)

    def test_solve_frequency_initializes_requested_resonator(self):
        with patch(
            "readout_workflow.physical_frequency_bounds.initialize",
            side_effect=RuntimeError("initialized"),
        ) as initialize:
            with self.assertRaisesRegex(RuntimeError, "initialized"):
                solve_frequency((156, 254), 4.85, [], (140, 160), 55, 6.05)
        initialize.assert_called_once_with((140, 160), 6.05)


if __name__ == "__main__": unittest.main()
