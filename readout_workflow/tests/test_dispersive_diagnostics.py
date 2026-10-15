import unittest
from dataclasses import replace
from model import Device,ReadoutModel
from study_dispersive import diagnostics


class DispersiveDiagnosticsTests(unittest.TestCase):
    def test_zero_resonator_coupling_has_zero_dispersive_ratios(self):
        m=ReadoutModel(replace(Device(),C1r_fF=0,C2r_fF=0))
        p=m.evaluate(12,14)
        d=diagnostics(m,p)
        self.assertEqual(d["lambda_max"],0)
        self.assertEqual(d["lambda_ground"],0)

    def test_higher_transitions_do_not_relax_ground_transition_screen(self):
        m=ReadoutModel(Device())
        p=m.evaluate(11.5,10.3)
        d=diagnostics(m,p)
        self.assertGreaterEqual(d["lambda_max"],d["lambda_ground"])
        self.assertGreater(d["lambda_max"],0)


if __name__ == "__main__":
    unittest.main()
