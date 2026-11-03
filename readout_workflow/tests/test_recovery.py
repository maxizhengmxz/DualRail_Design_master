"""Recovery order and rejected-candidate selection regressions."""

import unittest

from readout_workflow import calculation as study
from readout_workflow.recovery import candidate_rank, recovery_tiers


class RecoveryPolicyTests(unittest.TestCase):
    def test_historical_order_preserves_case_windows_and_approved_baseline(self):
        for window in ([0.14, 0.16], [0.08, 0.10]):
            tiers = recovery_tiers({"dressed_split_window_ghz": window}, 45.0)
            self.assertEqual(
                [t["name"] for t in tiers],
                [
                    "baseline",
                    "EC120",
                    "free_operation_frequency",
                    "wider_logical_frequency",
                    "H90",
                    "EJ_EC40",
                    "chi_matching_5percent",
                ],
            )
            self.assertEqual(tiers[0]["logical_frequency"], window)
            self.assertEqual(tiers[0]["ej_ec"], 45.0)
            self.assertTrue(tiers[1]["fixed"])
            self.assertFalse(tiers[2]["fixed"])
            self.assertAlmostEqual(tiers[3]["logical_frequency"][0], window[0] - 0.01)
            self.assertAlmostEqual(tiers[3]["logical_frequency"][1], window[1] + 0.01)
            self.assertEqual(tiers[4]["H"], 0.9)
            self.assertEqual(tiers[5]["ej_ec"], 40.0)
            self.assertTrue(all(t["mismatch_max"] == 0.01 for t in tiers[:-1]))
            self.assertEqual(tiers[-1]["mismatch_max"], 0.05)
            self.assertTrue(
                all(
                    "ej_upper_ghz" not in t and "abs_mean_chi_min_mhz" not in t
                    for t in tiers
                )
            )

    def test_healthy_strength_limited_point_outranks_destroyed_endpoint(self):
        tier = recovery_tiers({"dressed_split_window_ghz": [0.08, 0.1]})[2]
        good = dict(
            ratio=0.003,
            chi_m_ghz=-0.00078,
            chi_p_ghz=-0.00078,
            hybridization=0.972,
            lambda_max=0.07,
            min_assignment_overlap=0.99,
            splitting_ghz=0.09,
            operating_frequency_ghz=5.39,
            dressed_r_ghz=6.15,
            loaded_ec1_mhz=155.0,
            loaded_ec2_mhz=260.0,
            ej1_ghz=24.9,
            ej2_ghz=15.0,
            ej1_over_ec1=160.0,
            ej2_over_ec2=57.0,
        )
        bad = dict(good, ratio=1.97, hybridization=0.0, splitting_ghz=2.8)
        self.assertGreater(
            candidate_rank(good, tier, 5.39), candidate_rank(bad, tier, 5.39)
        )
        self.assertIn("absolute_mean_chi", study.reasons(good, tier, 5.39))

    def test_five_percent_is_final_tier_only_and_does_not_change_nominal_policy(self):
        tiers = recovery_tiers({"dressed_split_window_ghz": [0.08, 0.10]})
        point = dict(
            ratio=0.03,
            chi_m_ghz=-0.001,
            chi_p_ghz=-0.00103,
            hybridization=0.97,
            lambda_max=0.07,
            min_assignment_overlap=0.99,
            splitting_ghz=0.09,
            operating_frequency_ghz=5.39,
            dressed_r_ghz=6.15,
            loaded_ec1_mhz=155.0,
            loaded_ec2_mhz=260.0,
            ej1_ghz=24.9,
            ej2_ghz=15.0,
            ej1_over_ec1=160.0,
            ej2_over_ec2=57.0,
        )
        self.assertTrue(
            all("chi_matching" in study.reasons(point, t, 5.39) for t in tiers[:-1])
        )
        self.assertNotIn("chi_matching", study.reasons(point, tiers[-1], 5.39))
        self.assertLess(study.slacks(point, tiers[0])[0], 0)
        self.assertGreater(study.slacks(point, tiers[-1])[0], 0)
        self.assertIn(
            "chi_matching", study.reasons(dict(point, ratio=0.051), tiers[-1], 5.39)
        )
        self.assertIn(
            "absolute_mean_chi",
            study.reasons(
                dict(point, chi_m_ghz=-0.00075, chi_p_ghz=-0.00075), tiers[-1], 5.39
            ),
        )
        self.assertEqual(study.CONDITIONS.mismatch_max, 0.01)
