"""Opt-in legacy import compatibility; active policy tests live at repository root."""
import unittest
from core import cz_channel_policy as policy
from transmon_cz import channel_policy as legacy


class LegacyPolicyCompatibilityTests(unittest.TestCase):
    def test_same_public_objects(self):
        for name, value in vars(policy).items():
            if not name.startswith("_"):
                self.assertIs(getattr(legacy, name), value, name)
