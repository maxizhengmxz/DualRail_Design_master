import unittest
from unittest.mock import patch

from xm_cz.anchor_evaluator import evaluate_anchor_set


class AnchorEvaluatorTests(unittest.TestCase):
    def test_all_screening_failures_skip_fidelity(self):
        payload={"rail_A_anchors":[{"anchor_id":"a","anchor_hash":"h","readout_operating_frequency_GHz":4.8,"gate_logical_frequency_GHz":.145,"fixed_A":{}}],"logical_splittings_GHz":[.145,.08]}
        with patch("xm_cz.anchor_evaluator.evaluate_ec",return_value={"status":"no_feasible_S1_position"}) as evaluate:
            result=evaluate_anchor_set(payload)
        self.assertEqual(result["status"],"no_anchor_passed_screening")
        self.assertEqual(evaluate.call_count,1)

    def test_ranked_fallback_stops_at_first_fidelity_pass(self):
        anchors=[{"anchor_id":name,"anchor_hash":name,"readout_operating_frequency_GHz":f,"gate_logical_frequency_GHz":.145,"fixed_A":{}} for name,f in (("a",4.8),("b",4.9))]
        screened=[{"status":"screened","S1_position":{"S1":s,"intended_source_score":s,"maximum_logical_ratio":.1}} for s in (.02,.01)]
        failed={"status":"calibrated","final":{"raw_fidelity":.94,"postselected_fidelity":.9995}}
        passed={"status":"calibrated","final":{"raw_fidelity":.96,"postselected_fidelity":.9995}}
        with patch("xm_cz.anchor_evaluator.evaluate_ec",side_effect=[*screened,failed,passed]):
            result=evaluate_anchor_set({"rail_A_anchors":anchors,"logical_splittings_GHz":[.145,.08]})
        self.assertEqual(result["selected_anchor"]["anchor_id"],"a")
        self.assertEqual(result["selected_anchor_rank"],2)


if __name__=="__main__":unittest.main()
