"""Coarse-to-fine reuse must save work without changing physical acceptance."""

import unittest
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from readout_workflow import calculation as study
from readout_workflow.resonator_refinement import (
    PHYSICS_FUNCTIONS,
    PHYSICS_MODULES,
    ROOT,
    load_refinement,
    refine_tolerance,
)
from readout_workflow.storage import atomic_json


def certified(width):
    from readout_workflow.ec_resonator_tolerance import at_frequency

    frequencies = sorted(
        set(
            [6 - width, 6.0, 6 + width]
            + [
                round(6 - width + i * 0.025, 10)
                for i in range(int(2 * width / 0.025) + 1)
            ]
        )
    )
    samples = [
        dict(
            frequency_ghz=f,
            point=dict(
                device=asdict(at_frequency(asdict(study.Device()), f)),
                loaded_ec1_mhz=200.0,
                loaded_ec2_mhz=238.0,
            ),
            failures=[],
        )
        for f in frequencies
    ]
    return dict(
        status="validated_sampled_range",
        range_ghz=[6 - width, 6 + width],
        worst_mean_chi_mhz=1.2,
        worst_H=0.96,
        validation=dict(
            passed=True,
            failed_frequencies_ghz=[],
            validation_step_ghz=0.025,
            samples=samples,
        ),
    )


def state():
    return dict(
        source="coarse",
        good=0.1,
        bad=0.125,
        censored=False,
        certified=certified(0.1),
        provenance={},
    )


class RefinementSearchTests(unittest.TestCase):
    def test_only_two_new_widths_and_no_coarse_expansion(self):
        from readout_workflow import resonator_robust_design as workflow

        def dense(ec, nominal, low, high, h, folder, *args, **kwargs):
            result = certified((high - low) / 2)
            atomic_json(Path(folder) / "result.json", result)
            return result

        with TemporaryDirectory() as directory:
            with (
                patch.object(
                    workflow,
                    "optimize_scenarios",
                    return_value={"best": {"seed": "new"}},
                ) as opt,
                patch.object(workflow, "run_range", side_effect=dense) as audit,
            ):
                result = refine_tolerance(
                    state(), [200, 238], {}, directory, 3, 50, 0.05, 0.00625
                )
            widths = [
                (call.args[2][-1] - call.args[2][0]) / 2 for call in opt.call_args_list
            ]
            self.assertEqual(len(widths), 2)
            self.assertTrue(all(0.1 < w < 0.125 for w in widths))
            self.assertAlmostEqual(
                result["boundary"]["validated_halfwidth_ghz"], 0.11875
            )
            self.assertEqual(audit.call_count, 2)
            # A restart must reuse refined trials too, not just the coarse result.
            with (
                patch.object(
                    workflow,
                    "optimize_scenarios",
                    side_effect=AssertionError("repeated solve"),
                ),
                patch.object(
                    workflow, "run_range", side_effect=AssertionError("repeated audit")
                ),
            ):
                repeated = refine_tolerance(
                    state(), [200, 238], {}, directory, 3, 50, 0.05, 0.00625
                )
            self.assertEqual(repeated["refinement"]["new_width_trials"], 0)
            self.assertEqual(repeated["refinement"]["cached_width_trials"], 2)

    def test_failed_refinement_never_loses_the_coarse_design(self):
        from readout_workflow import resonator_robust_design as workflow

        original = state()
        with (
            TemporaryDirectory() as directory,
            patch.object(workflow, "optimize_scenarios", return_value={"best": None}),
            patch.object(workflow, "run_range") as audit,
        ):
            result = refine_tolerance(
                original, [200, 238], {}, directory, 3, 50, 0.05, 0.00625
            )
        self.assertIs(result["validated_result"], original["certified"])
        self.assertEqual(result["boundary"]["validated_halfwidth_ghz"], 0.1)
        self.assertLessEqual(
            result["boundary"]["failed_search_halfwidth_ghz"] - 0.1, 0.00625 + 1e-10
        )
        audit.assert_not_called()

    def test_already_fine_or_censored_source_needs_no_new_solves(self):
        from readout_workflow import resonator_robust_design as workflow

        for bad, censored in [(0.125, False), (None, True)]:
            original = state()
            original.update(bad=bad, censored=censored)
            with (
                TemporaryDirectory() as directory,
                patch.object(workflow, "optimize_scenarios") as opt,
            ):
                result = refine_tolerance(
                    original, [200, 238], {}, directory, 3, 50, 0.05, 0.025
                )
            opt.assert_not_called()
            self.assertEqual(result["boundary"]["censored_by_search_ceiling"], censored)


class RefinementCompatibilityTests(unittest.TestCase):
    def make_coarse(self, folder):
        hashes = {}
        for relative in set(PHYSICS_MODULES) | {f for f, fn in PHYSICS_FUNCTIONS}:
            file = folder / "source_snapshot" / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes((ROOT / relative).read_bytes())
            hashes[relative] = sha256(file.read_bytes()).hexdigest()
        settings = dict(
            arguments=dict(
                mode="tolerance",
                ec=[200, 238],
                frequency_step_ghz=0.05,
                max_halfwidth_ghz=1.0,
                boundary_resolution_ghz=0.025,
            ),
            conditions=asdict(study.CONDITIONS),
            source_hashes=hashes,
        )
        atomic_json(folder / "settings.json", settings)
        atomic_json(
            folder / "completion.json",
            dict(
                phase="complete",
                boundary=dict(
                    validated_halfwidth_ghz=0.1,
                    failed_search_halfwidth_ghz=0.125,
                    censored_by_search_ceiling=False,
                ),
            ),
        )
        atomic_json(
            folder / "results.json", [dict(H_target=0.95, result=certified(0.1))]
        )
        return settings

    def test_compatible_source_is_read_only_and_resolution_can_change(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(study, "failures", return_value=[]),
        ):
            folder = Path(directory)
            self.make_coarse(folder)
            before = {
                str(p): sha256(p.read_bytes()).hexdigest()
                for p in folder.rglob("*")
                if p.is_file()
            }
            result = load_refinement(folder, [200, 238], 0.05, 1.0, 0.00625)
            self.assertEqual(result["good"], 0.1)
            self.assertEqual(result["provenance"]["target_resolution_ghz"], 0.00625)
            self.assertEqual(
                before,
                {
                    str(p): sha256(p.read_bytes()).hexdigest()
                    for p in folder.rglob("*")
                    if p.is_file()
                },
            )

    def test_changed_conditions_are_rejected(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            settings = self.make_coarse(folder)
            settings["conditions"]["mismatch_max"] = 0.02
            atomic_json(folder / "settings.json", settings)
            with self.assertRaisesRegex(ValueError, "acceptance conditions differ"):
                load_refinement(folder, [200, 238], 0.05, 1.0, 0.00625)

    def test_changed_physics_is_rejected_even_with_updated_snapshot_hash(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            settings = self.make_coarse(folder)
            relative = "readout_workflow/model.py"
            snapshot = folder / "source_snapshot" / relative
            snapshot.write_text(snapshot.read_text() + "\nCHANGED_PHYSICS_TEST = 1\n")
            settings["source_hashes"][relative] = sha256(
                snapshot.read_bytes()
            ).hexdigest()
            atomic_json(folder / "settings.json", settings)
            with self.assertRaisesRegex(ValueError, "Physics implementation changed"):
                load_refinement(folder, [200, 238], 0.05, 1.0, 0.00625)

    def test_changed_tuple_grid_or_ceiling_is_rejected(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            self.make_coarse(folder)
            for ec, step, ceiling, pattern in [
                ([202, 238], 0.05, 1.0, "EC tuple"),
                ([200, 238], 0.025, 1.0, "Frequency-grid"),
                ([200, 238], 0.05, 2.0, "ceiling differs"),
            ]:
                with self.assertRaisesRegex(ValueError, pattern):
                    load_refinement(folder, ec, step, ceiling, 0.00625)


if __name__ == "__main__":
    unittest.main()
