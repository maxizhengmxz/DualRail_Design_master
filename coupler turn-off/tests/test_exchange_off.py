"""Ensure exchange extraction distinguishes local rotations from interactions."""

import unittest

import numpy as np
from coupler_turn_off.logical import energy_exchange


class ExchangeFrameTests(unittest.TestCase):
    def test_oriented_exchange_is_invariant_to_logical_basis_signs(self):
        paulis = dict(
            I=np.eye(2),
            X=np.array([[0, 1], [1, 0]]),
            Y=np.array([[0, -1j], [1j, 0]]),
            Z=np.diag([1, -1]),
        )
        original = dict(
            II=0.0, XI=3.0, IX=-2.0, ZI=-45.0, IZ=-46.0, XX=0.0001, YY=0.0001, ZZ=0.0002
        )
        ham = sum(
            value * np.kron(paulis[key[0]], paulis[key[1]])
            for key, value in original.items()
        )
        flip = np.kron(paulis["Z"], paulis["I"])
        changed = flip @ ham @ flip

        def extract(matrix):
            coefficients = {
                a + b: float(np.trace(np.kron(pa, pb) @ matrix).real / 4)
                for a, pa in paulis.items()
                for b, pb in paulis.items()
            }
            return energy_exchange(dict(pauli_mhz=coefficients))

        self.assertAlmostEqual(
            extract(ham)["oriented_logical_exchange_mhz"],
            extract(changed)["oriented_logical_exchange_mhz"],
            places=12,
        )

    def test_known_exchange_survives_independent_local_rotations(self):
        identity = np.eye(2)
        x = np.array([[0, 1], [1, 0]])
        y = np.array([[0, -1j], [1j, 0]])
        z = np.diag([1, -1])
        j, zz = 0.0002, 0.0007
        ham = (
            -45 * np.kron(z, identity)
            - 46 * np.kron(identity, z)
            + j / 2 * (np.kron(x, x) + np.kron(y, y))
            + zz / 4 * np.kron(z, z)
        )

        def rotation(angle):
            return np.array(
                [
                    [np.cos(angle / 2), -np.sin(angle / 2)],
                    [np.sin(angle / 2), np.cos(angle / 2)],
                ]
            )

        u = np.kron(rotation(0.4), rotation(-0.3))
        rotated = u @ ham @ u.T
        paulis = dict(I=identity, X=x, Y=y, Z=z)
        coefficients = {
            a + b: float(np.trace(np.kron(pa, pb) @ rotated).real / 4)
            for a, pa in paulis.items()
            for b, pb in paulis.items()
        }
        result = energy_exchange(dict(pauli_mhz=coefficients))
        self.assertAlmostEqual(
            abs(result["energy_basis_logical_exchange_mhz"]), j, places=11
        )
        self.assertAlmostEqual(result["energy_basis_logical_zz_mhz"], zz, places=11)


if __name__ == "__main__":
    unittest.main()
