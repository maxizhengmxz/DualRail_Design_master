import unittest

from xm_cz.config import validate_payload
from xm_cz.model import fixed_pair_model


class FixedRailATests(unittest.TestCase):
    def test_readout_witness_is_not_retuned(self):
        model=fixed_pair_model(.165,.250,19.146849682002202,13.079689412705315,.048760704768226165,8)
        self.assertEqual(model["EJ"],[19.146849682002202,13.079689412705315])
        self.assertAlmostEqual(model["J"],.048760704768226165)
        self.assertTrue(.14 <= model["logical_splitting_GHz"] <= .16)

    def test_fixed_a_ports_are_validated(self):
        payload={"ec_GHz":[.165,.250,.150,.164],
                 "fixed_A":{"EJ_GHz":[19.1,13.1],"J_GHz":.049,"operating_frequency_GHz":4.85}}
        validate_payload(payload)
        payload["fixed_A"].pop("J_GHz")
        with self.assertRaises(ValueError): validate_payload(payload)


if __name__=="__main__": unittest.main()
