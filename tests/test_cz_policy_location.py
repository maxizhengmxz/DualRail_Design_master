"""Active workflows must work without importing the legacy CZ package."""
import subprocess
import sys
import unittest
from pathlib import Path

from core import cz_channel_policy as policy


class PolicyLocationTests(unittest.TestCase):
    def test_active_imports_with_legacy_package_blocked(self):
        root = Path(__file__).resolve().parents[1]
        code = '''
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root / "2Q-gate_XM"), str(root / "Gnd-coupler+Floating-T"),
               str(root / "logical-resonance gate"), str(root)]
class RejectLegacy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "transmon_cz" or fullname.startswith("transmon_cz."):
            raise ImportError("Legacy CZ package is unavailable")
sys.meta_path.insert(0, RejectLegacy())
from core import cz_channel_policy as policy
from xm_cz import model, evaluator, runner
from floating_coupler import channel_policy, calibrate_on, fabrication
from floating_coupler import gate, on_gate, idle_gate, pulsed_gate, filtered_pulse
import final_seven_validation
import pulse_cz
import pulse_shape_comparison
from studies import robustness_worker, minimum_change_validation, parasitic_absorption
from logical_resonance_gate import dualrail_fsim, fsim
assert channel_policy.select_channel is policy.select_channel
assert model.interaction_scores is policy.interaction_scores
assert evaluator.select_channel is policy.select_channel
assert not any(name.startswith("transmon_cz") for name in sys.modules)
'''
        subprocess.run([sys.executable, "-c", code, str(root)], cwd=root, check=True)

    def test_current_policy_version_and_off_scope_are_unchanged(self):
        self.assertEqual(policy.DEFINITION_VERSION, "dual-rail-cz-s1-v2")
        self.assertEqual(policy.off_target_definition("three")["source"], "01")
        for model in ("five", "seven"):
            self.assertEqual(policy.off_target_definition(model)["source"], "0100")
            self.assertEqual(policy.off_target_definition(model)["status"], "temporary_open_question")
