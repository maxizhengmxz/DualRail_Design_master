"""Saved symmetric certificates seed asymmetric limits without redoing the band."""

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from readout_workflow import asymmetric_design_bounds as bounds
from readout_workflow import calculation as study


class AsymmetricBoundsTests(unittest.TestCase):
    def test_reuses_band_and_flexible_contains_fixed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            out = root / "out"
            nominal = dict(device=asdict(study.Device()), operating_frequency_ghz=4.5)
            (source / "H_0.950000").mkdir(parents=True)
            (source / "H_0.950000" / "nominal.json").write_text(
                json.dumps({"point": nominal})
            )
            for f in [5.9, 6.1]:
                p = dict(
                    nominal, device=asdict(bounds.at_frequency(nominal["device"], f))
                )
                path = source / "round" / "validation" / f"frequency_{f:.10f}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(point=p, failures=[])))

            def solve(nominal, f, mode, warm, settings, extra):
                low, high = (5.95, 6.05) if mode == "fixed" else (5.8, 6.3)
                p = dict(
                    nominal, device=asdict(bounds.at_frequency(nominal["device"], f))
                )
                ok = low - 1e-9 <= f <= high + 1e-9
                return dict(
                    frequency_ghz=f,
                    mode=mode,
                    point=p if ok else None,
                    closest=p,
                    status="validated_pass" if ok else "no_validated_solution_found",
                    failures=[] if ok else ["test_limit"],
                    seconds=0,
                )

            task = dict(
                pair="B",
                ec=[200, 238],
                source_point=str(source),
                radius_ghz=0.1,
                output=str(out),
                config=dict(
                    conditions=asdict(study.CONDITIONS),
                    frequency_range_ghz=[4.0, 12.0],
                    resonator_step_ghz=0.05,
                    boundary_resolution_ghz=0.00625,
                    starts=3,
                    max_nfev=35,
                ),
            )
            with (
                patch.object(study, "CONDITIONS", study.CONDITIONS),
                patch.object(study, "BASELINE", study.BASELINE),
                patch.object(study, "failures", return_value=[]),
                patch.object(bounds, "solve_frequency", side_effect=solve) as solver,
            ):
                r = bounds.run_design(task)
                for call in solver.call_args_list:
                    if call.args[2] == "flexible":
                        self.assertTrue(call.args[1] < 5.9 or call.args[1] > 6.1)
                solver.reset_mock()
                again = bounds.run_design(task)
                solver.assert_not_called()
            for side in ("lower", "upper"):
                self.assertGreaterEqual(
                    r["bounds"]["flexible"][side]["tolerance_mhz"] + 1e-6,
                    r["bounds"]["fixed"][side]["tolerance_mhz"],
                )
            self.assertAlmostEqual(r["bounds"]["fixed"]["lower"]["tolerance_mhz"], 50)
            self.assertAlmostEqual(
                r["bounds"]["flexible"]["upper"]["tolerance_mhz"], 300
            )
            self.assertEqual(r["ec_mhz"], again["ec_mhz"])


if __name__ == "__main__":
    unittest.main()
