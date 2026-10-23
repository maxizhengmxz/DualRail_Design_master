"""Check J01 against an actual product-space Hamiltonian matrix element."""

import unittest
import numpy as np
from readout_workflow.calculation import Basis, Device, ReadoutModel, _transmon
from readout_workflow.coupling import exchange_j01_ghz


class CouplingTests(unittest.TestCase):
    def test_local_exchange_is_not_charge_coefficient_or_gap(self):
        basis = Basis(15, 4, 3, 12)
        model = ReadoutModel(Device(), basis)
        n1 = _transmon(16, model.ec_ghz[0], basis)[1]
        n2 = _transmon(18, model.ec_ghz[1], basis)[1]
        j = model.J_ghz[0, 1]
        interaction = j * np.kron(n1, n2)
        j01 = exchange_j01_ghz(j, n1[0, 1], n2[0, 1])
        self.assertAlmostEqual(j01, abs(interaction[4, 1]), places=14)
        gap = np.ptp(np.linalg.eigvalsh([[0, j01], [j01, 0]]))
        self.assertAlmostEqual(gap, 2*j01, places=14)
        np.testing.assert_allclose(np.diag(model.J_ghz), 8*model.ec_ghz)
        self.assertAlmostEqual(
            j01, exchange_j01_ghz(j, -1j*n1[0, 1], -n2[0, 1]), places=14
        )
