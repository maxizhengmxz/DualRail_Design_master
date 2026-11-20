"""Anchored cavity tolerance preserves EC reference and tuning budgets."""

import unittest

import numpy as np
from unittest.mock import patch

from readout_workflow import calculation as study
from readout_workflow.anchored_resonator_tolerance import (
    continuation,
    frequency_bounds,
    synthesize_device,
)
from readout_workflow.validation import INTERMEDIATE


class AnchoredToleranceTests(unittest.TestCase):
    def test_individual_frequency_budget_is_centered_on_each_target(self):
        low, high = frequency_bounds([4.606403, 4.600550], 0.1)
        np.testing.assert_allclose(low, [4.506403, 4.500550])
        np.testing.assert_allclose(high, [4.706403, 4.700550])
        with self.assertRaises(ValueError):
            frequency_bounds([4.6], 0.1)

    def test_ec_synthesis_uses_each_rail_nominal_cavity(self):
        original = study.BARE_RESONATOR_GHZ
        for reference, ec, caps in [
            (6.05, [158, 240], [3.104857864158004, 5.016092792494639]),
            (5.95, [158, 226], [1.6888646767689799, 2.655021614046039]),
        ]:
            with self.subTest(reference=reference):
                device = synthesize_device(ec, *caps, reference)
                model = study.ReadoutModel(device, INTERMEDIATE)
                np.testing.assert_allclose(model.ec_ghz[:2] * 1000, ec, atol=1e-9)
                cavity = 1 / (2*np.pi*np.sqrt(device.Lr_nH*1e-9*device.Cr_fF*1e-15))/1e9
                self.assertAlmostEqual(cavity, reference, places=12)
        self.assertEqual(study.BARE_RESONATOR_GHZ, original)

    def test_narrow_ranges_still_validate_both_endpoints(self):
        def accepted(device, frequency, targets, halfwidth, warm, exact=False, max_nfev=70):
            point = dict(hybridization=.96, chi_m_ghz=.0011, chi_p_ghz=.0011)
            return dict(status="validated_pass", frequency_ghz=frequency,
                        bare_f01_ghz=list(targets), point=point)

        with patch("readout_workflow.anchored_resonator_tolerance.solve_frequency",
                   side_effect=accepted):
            result = continuation(None, 5.95, [5.29, 5.286], .05,
                                  5.9125, 5.9875, .05)
        self.assertEqual([row["frequency_ghz"] for row in result["samples"]],
                         [5.9125, 5.95, 5.9875])


if __name__ == "__main__":
    unittest.main()
