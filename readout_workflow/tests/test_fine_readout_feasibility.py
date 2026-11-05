import unittest

from readout_workflow.fine_readout_feasibility import logical_range_ghz


class FineReadoutFrequencyTests(unittest.TestCase):
    def test_frequency_interval_conversion(self):
        self.assertEqual(logical_range_ghz((80, 100)), (0.08, 0.1))
        self.assertEqual(logical_range_ghz((140, 160)), (0.14, 0.16))

    def test_frequency_interval_rejects_invalid_bounds(self):
        for bounds in ((100, 80), (0, 100), (100, 100)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                logical_range_ghz(bounds)


if __name__ == "__main__":
    unittest.main()
