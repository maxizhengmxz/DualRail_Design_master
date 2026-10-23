import unittest

import numpy as np

from legacy.analysis.sampling import _sample_parameters


class NegativeIndependentRng:
    def normal(self, center=0.0, sigma=1.0):
        return center - sigma


class NegativeStandardRng:
    def normal(self):
        return -1.0


class CorrelatedParameterSamplingTests(unittest.TestCase):
    def test_perfect_correlation_group_uses_same_relative_deviation(self):
        rng = np.random.default_rng(123)
        params = {"Ej1": 10.0, "Ej2": 20.0, "C1": 100.0}

        sampled, sigmas = _sample_parameters(
            params,
            rng,
            param_sigma=None,
            param_rel_sigma={"Ej1": 0.1, "Ej2": 0.1, "C1": 0.01},
            param_correlation_groups=[("Ej1", "Ej2")],
        )

        ej1_rel = (sampled["Ej1"] - params["Ej1"]) / params["Ej1"]
        ej2_rel = (sampled["Ej2"] - params["Ej2"]) / params["Ej2"]

        self.assertAlmostEqual(ej1_rel, ej2_rel)
        self.assertEqual(sigmas["Ej1"], 1.0)
        self.assertEqual(sigmas["Ej2"], 2.0)

    def test_partial_correlation_group_mixes_shared_and_independent_noise(self):
        seed = 456
        params = {"Ej1": 10.0, "Ej2": 20.0}
        param_rel_sigma = {"Ej1": 0.1, "Ej2": 0.1}

        sampled, _ = _sample_parameters(
            params,
            np.random.default_rng(seed),
            param_sigma=None,
            param_rel_sigma=param_rel_sigma,
            param_correlation_groups=[
                {"params": ["Ej1", "Ej2"], "correlation": 0.25},
            ],
        )

        rng = np.random.default_rng(seed)
        shared = rng.normal()
        independent_ej1 = rng.normal()
        independent_ej2 = rng.normal()
        expected_ej1 = 10.0 * (
            1.0 + 0.1 * (0.5 * shared + np.sqrt(0.75) * independent_ej1)
        )
        expected_ej2 = 20.0 * (
            1.0 + 0.1 * (0.5 * shared + np.sqrt(0.75) * independent_ej2)
        )

        self.assertAlmostEqual(sampled["Ej1"], expected_ej1)
        self.assertAlmostEqual(sampled["Ej2"], expected_ej2)

    def test_rejects_parameters_in_multiple_correlation_groups(self):
        with self.assertRaisesRegex(ValueError, "multiple correlation groups"):
            _sample_parameters(
                {"Ej1": 10.0, "Ej2": 20.0, "Ej3": 1.0},
                np.random.default_rng(123),
                param_sigma=None,
                param_rel_sigma={"Ej1": 0.1, "Ej2": 0.1, "Ej3": 0.1},
                param_correlation_groups=[
                    ("Ej1", "Ej2"),
                    {"params": ["Ej1", "Ej3"], "correlation": 0.5},
                ],
            )

    def test_default_relative_floor_is_one_percent_of_parameter_value(self):
        sampled, _ = _sample_parameters(
            {"Cj3": 0.2},
            NegativeIndependentRng(),
            param_sigma={"Cj3": 10.0},
            param_rel_sigma={},
        )

        self.assertEqual(sampled["Cj3"], 0.002)

    def test_custom_relative_floor_applies_to_correlated_parameters(self):
        sampled, _ = _sample_parameters(
            {"Ej1": 14.0, "Ej2": 14.0},
            NegativeStandardRng(),
            param_sigma={"Ej1": 100.0, "Ej2": 100.0},
            param_rel_sigma={},
            param_correlation_groups=[("Ej1", "Ej2")],
            param_rel_min=0.05,
        )

        self.assertAlmostEqual(sampled["Ej1"], 0.7)
        self.assertAlmostEqual(sampled["Ej2"], 0.7)

    def test_relative_floor_can_be_overridden_per_parameter(self):
        sampled, _ = _sample_parameters(
            {"Cj1": 5.0, "Cj2": 10.0},
            NegativeIndependentRng(),
            param_sigma={"Cj1": 100.0, "Cj2": 100.0},
            param_rel_sigma={},
            param_rel_min={"Cj1": 0.1},
        )

        self.assertAlmostEqual(sampled["Cj1"], 0.5)
        self.assertAlmostEqual(sampled["Cj2"], 0.1)

    def test_rejects_negative_relative_floor(self):
        with self.assertRaisesRegex(ValueError, "param_rel_min"):
            _sample_parameters(
                {"Cj1": 5.0},
                NegativeIndependentRng(),
                param_sigma={"Cj1": 100.0},
                param_rel_sigma={},
                param_rel_min=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
