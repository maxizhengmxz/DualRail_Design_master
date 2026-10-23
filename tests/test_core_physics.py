"""Shared physics ownership, electrostatic units, and lightweight imports."""

import subprocess
import sys
import unittest

import numpy as np
from scipy.constants import elementary_charge as e, h

from core.electrostatics import charge_coupling_ghz, loaded_charging_energies_ghz
from core.transmon import Basis, transmon_spectrum


class SharedPhysicsTests(unittest.TestCase):
    def test_loaded_network_coefficients(self):
        c1, c2, cm = 90e-15, 75e-15, 3e-15
        inverse = np.linalg.inv([[c1, -cm], [-cm, c2]])
        determinant = c1*c2-cm**2
        expected_ec = e**2*np.array([c2, c1])/(2*h*1e9*determinant)
        np.testing.assert_allclose(loaded_charging_energies_ghz(inverse), expected_ec)
        self.assertAlmostEqual(
            charge_coupling_ghz(inverse)[0, 1], 4*e**2*cm/(h*1e9*determinant)
        )

    def test_readout_uses_shared_definitions(self):
        from readout_workflow import calculation, model
        for module in (calculation, model):
            self.assertIs(module.Basis, Basis)
            self.assertIs(module._transmon, transmon_spectrum)
            self.assertIs(module.transmon_spectrum, transmon_spectrum)

    def test_physics_import_does_not_load_workflows(self):
        subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys; import core.transmon; import core.electrostatics; "
             "assert not any(k.startswith(('logical_resonance_gate', 'readout_workflow', "
             "'coupler_turn_off', 'legacy')) for k in sys.modules)"],
            check=True, capture_output=True, text=True,
        )
