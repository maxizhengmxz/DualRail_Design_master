"""Campaign launch arguments must preserve the production half-plane policy."""

import unittest
from pathlib import Path

from readout_workflow.resonator_design_campaign import command_for


class CampaignCommandTests(unittest.TestCase):
    def test_launch_uses_robust_tolerance_objective_and_original_h_floor(self):
        cmd = command_for([238, 200], Path("test-output"), 3, 70, 1.0)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "tolerance")
        self.assertEqual(
            cmd[cmd.index("--ec") + 1 : cmd.index("--ec") + 3], ["238", "200"]
        )
        self.assertNotIn("--h-targets", cmd)  # Derive H from the original conditions.
        self.assertEqual(cmd[cmd.index("--frequency-step-ghz") + 1], "0.05")
        self.assertNotIn("--diagnose-tolerance", cmd)

    def test_quick_profile_changes_search_effort_not_acceptance(self):
        cmd = command_for([238, 200], Path("test-output"), 3, 50, 1.0, quick=True)
        self.assertIn("--quick-tolerance", cmd)
        self.assertEqual(cmd[cmd.index("--boundary-resolution-ghz") + 1], "0.025")
        self.assertNotIn("--h-targets", cmd)  # Derive H from the original conditions.
        self.assertEqual(cmd[cmd.index("--frequency-step-ghz") + 1], "0.05")

    def test_custom_conditions_and_current_interpreter_are_forwarded(self):
        import sys

        conditions = Path("custom") / "settings.json"
        cmd = command_for([238, 200], Path("out"), 3, 50, 1.0, conditions=conditions)
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[cmd.index("--conditions") + 1], str(conditions))

    def test_campaign_range_resolution_can_be_refined_explicitly(self):
        cmd = command_for(
            [238, 200], Path("out"), 3, 50, 1.0, boundary_resolution=0.00625
        )
        self.assertEqual(cmd[cmd.index("--boundary-resolution-ghz") + 1], "0.00625")


if __name__ == "__main__":
    unittest.main()
