"""Readout anchor sampling cannot silently include feasible-range edges."""

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

# Load by file path because the descriptive workflow directory contains spaces.
_path = (
    Path(__file__).resolve().parents[1]
    / "Ec_tuple search workflow"
    / "run_sampled_anchor_repetition.py"
)
_spec = importlib.util.spec_from_file_location("ec_tuple_anchor_repetition", _path)
workflow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(workflow)
anchors, main = workflow.anchors, workflow.main


def samples(start, count, step=0.025):
    return [
        dict(status="validated_pass", target_frequency_ghz=start + i * step)
        for i in range(count)
    ]


class AnchorFrequencySamplingTests(unittest.TestCase):
    def test_cached_endpoint_anchors_cannot_bypass_new_selection(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            scan = output / "readout_frequency_scan"
            scan.mkdir()
            (scan / "results.json").write_text(json.dumps(samples(4.0, 41)))
            saved = [
                dict(readout_operating_frequency_GHz=frequency)
                for frequency in (4.0, 4.3, 4.5, 4.7, 5.0)
            ]
            (output / "config.json").write_text(
                json.dumps(dict(task_defaults=dict(rail_A_anchors=saved)))
            )
            arguments = [
                "run_sampled_anchor_repetition.py",
                "--selection-id",
                "test",
                "--ec-a-mhz",
                "165",
                "250",
                "--output",
                directory,
            ]
            with (
                patch("sys.argv", arguments),
                patch.object(workflow, "command") as command,
            ):
                with self.assertRaisesRegex(RuntimeError, "fresh output directory"):
                    main()
            command.assert_not_called()
            self.assertFalse((output / "anchor_selection.json").exists())

    def test_default_fractions_are_kept(self):
        selected, bounds, count = anchors(samples(4.0, 41))
        self.assertEqual(bounds, (4.0, 5.0))
        self.assertEqual(count, 41)
        for actual, expected in zip(selected, (4.1, 4.3, 4.5, 4.7, 4.9), strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_coarse_grid_rounding_stays_inside_band(self):
        selected, (low, high), _ = anchors(samples(4.0, 7))
        self.assertEqual(len(set(selected)), 5)
        for frequency in selected:
            self.assertGreaterEqual((frequency - low) / (high - low), 0.1 - 1e-12)
            self.assertLessEqual((frequency - low) / (high - low), 0.9 + 1e-12)

    def test_sparse_interval_requires_refinement_instead_of_endpoints(self):
        with self.assertRaisesRegex(RuntimeError, "Refine the frequency scan"):
            anchors(samples(4.0, 5))

    def test_largest_component_is_used(self):
        selected, bounds, _ = anchors(samples(4.0, 9) + samples(5.0, 41))
        self.assertEqual(bounds, (5.0, 6.0))
        self.assertGreater(min(selected), 5.0)
        self.assertLess(max(selected), 6.0)

    def test_invalid_sampling_requests_fail_explicitly(self):
        for fractions in ((0.0, 0.5, 1.0), (0.1, 0.1), (), (0.1, float("nan"))):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                anchors(samples(4.0, 41), fractions=fractions)
