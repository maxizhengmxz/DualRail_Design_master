"""Sequence order, half-plane seeding and restart protection."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from readout_workflow.resonator_design_campaign import selected_rows
from readout_workflow.resonator_pair_sequence import exclusive_run, stage_plan


class PairSequenceTests(unittest.TestCase):
    def test_order_resolution_and_same_half_plane(self):
        p = stage_plan(Path("sequence"), 4)
        self.assertEqual([s["name"] for s in p], ["B_fine", "A_coarse", "A_refined"])
        self.assertEqual(
            [s["boundary_resolution_ghz"] for s in p], [0.00625, 0.025, 0.00625]
        )
        for stage in p:
            cmd = stage["command"]
            self.assertEqual(cmd[cmd.index("--orientation") + 1], "ec2-gt-ec1")
            self.assertEqual(cmd[cmd.index("--ec-step-mhz") + 1], "2")
            self.assertIn("--stop-file", cmd)
        self.assertNotIn("--refine-from", p[0]["command"])
        self.assertNotIn("--refine-from", p[1]["command"])
        self.assertIn("--refine-from", p[2]["command"])

    def test_second_scheduler_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            with exclusive_run(path):
                with self.assertRaisesRegex(RuntimeError, "live scheduler"):
                    with exclusive_run(path):
                        pass
            self.assertFalse((path / "sequence.lock").exists())

    def test_mirroring_is_a_seed_not_a_copied_certificate(self):
        point = dict(
            device={
                "C1_shunt_fF": 80,
                "C2_shunt_fF": 90,
                "C12_fF": 2,
                "C1r_fF": 5,
                "C2r_fF": 5,
            },
            ej1_ghz=15,
            ej2_ghz=16,
            bare_f1_ghz=4.5,
            bare_f2_ghz=4.6,
            loaded_ec1_mhz=238,
            loaded_ec2_mhz=198,
            chi_sign_stability={"passed": True},
            validation={"fake": "old"},
            x=[1],
        )
        data = [dict(ec1_target_mhz=238, ec2_target_mhz=198, point=point)]
        rows = selected_rows(data, "ec2-gt-ec1", 2)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        q = r["nominal"]
        self.assertEqual((r["ec1_target_mhz"], r["ec2_target_mhz"]), (198, 238))
        self.assertEqual(q["ej1_ghz"], 16)
        self.assertTrue(q["label_swapped_seed_only"])
        self.assertNotIn("chi_sign_stability", q)
        self.assertNotIn("validation", q)
        self.assertIn("chi_sign_stability", point)


if __name__ == "__main__":
    unittest.main()
