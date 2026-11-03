"""Optional-study archive, sampling and noise-model regression coverage."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from studies.archive import write_archive, verify_archive, iter_records
from studies.robustness import draw_errors
from studies.noise import dephasing_time, exponential_cz, ECHO_FACTOR


class OptionalStudyTests(unittest.TestCase):
    def record(self):
        return dict(
            id=1,
            kind="example",
            readout={"A": {"passed": True}, "B": {"passed": False}},
            off={"found": True},
            gate={
                "found": False,
                "best": dict(raw_cz_fidelity=0.9, postselected_cz_fidelity=0.95),
            },
        )

    def test_archive_preserves_rejected_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.tar.xz"
            row = self.record()
            result = write_archive(path, [row], {"nominal": True})
            self.assertEqual(list(iter_records(path)), [row])
            self.assertEqual(verify_archive(path), result["summary"])
            self.assertEqual(result["summary"]["B_failed"], 1)
            self.assertEqual(result["summary"]["gate_best_available"], 1)

    def test_archive_rejects_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_archive(
                    Path(directory) / "data.tar.xz", [self.record(), self.record()], {}
                )

    def test_stratified_maximum_is_inside_requested_bin(self):
        names = [str(i) for i in range(24)]
        maxima = []
        for i in range(1000):
            values = draw_errors(names, 113, 2, i, 0.01, 0.0125)
            maxima.append(max(map(abs, values.values())))
            self.assertEqual(len(values), 24)
            self.assertTrue(0.01 <= maxima[-1] < 0.0125)
        self.assertAlmostEqual(np.mean(maxima), 0.01125, delta=0.00008)

    def test_sampler_is_reproducible_without_cloning_components(self):
        names = [str(i) for i in range(24)]
        a = draw_errors(names, 5, 1, 0, 0.01, 0.02)
        self.assertEqual(a, draw_errors(names, 5, 1, 0, 0.01, 0.02))
        self.assertNotEqual(a, draw_errors(names, 5, 1, 1, 0.01, 0.02))
        self.assertGreater(len(set(a.values())), 20)

    def test_echo_reference_scaling(self):
        self.assertAlmostEqual(dephasing_time(100.0, "echo"), 1 / (ECHO_FACTOR * 100))
        self.assertAlmostEqual(
            dephasing_time(10.0, "echo") / dephasing_time(100.0, "echo"), 10.0
        )

    def test_ramsey_short_cutoff_reference(self):
        t = dephasing_time(1000.0, "ramsey")
        exponent = 2 * np.pi**2 * 1000**2 * t * t * np.log(0.4007 / t)
        self.assertAlmostEqual(exponent, 1.0, delta=0.001)

    def test_ideal_cycle_has_unit_fidelity(self):
        result = exponential_cz(
            np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 5.0]]), 0.0, "echo"
        )
        for key in ["raw_fidelity", "postselected_fidelity", "survival"]:
            self.assertAlmostEqual(result[key], 1.0, places=12)

    def test_known_auxiliary_inclusive_reference(self):
        derivatives = np.array(
            [
                [-2199875481.033325, -2196693351.6979218],
                [-2236669510.2214813, -2233983230.2570343],
                [-8455349964.714049, -408904871.94061273],
            ]
        )
        result = exponential_cz(derivatives, 10.0, "echo")
        self.assertAlmostEqual(
            result["postselected_fidelity"], 0.99725600455143, places=10
        )
        self.assertLess(result["raw_fidelity"], result["postselected_fidelity"])


if __name__ == "__main__":
    unittest.main()
