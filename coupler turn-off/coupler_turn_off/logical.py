"""Logical exchange and ZZ in the local energy eigenstate convention (MHz).

The full Hamiltonian is diagonalized by BudgetModel. Isolated-pair states
identify the logical products; this helper removes local logical rotations.
"""

import numpy as np
from scipy.linalg import eigh


def energy_exchange(result):
    """Remove local rotations using local terms of the effective logical H."""
    paulis = dict(
        I=np.eye(2),
        X=np.array([[0, 1], [1, 0]]),
        Y=np.array([[0, -1j], [1j, 0]]),
        Z=np.diag([1, -1]),
    )
    p = result["pauli_mhz"]
    ham = sum(
        value * np.kron(paulis[key[0]], paulis[key[1]]) for key, value in p.items()
    )
    ha = sum(p[a + "I"] * paulis[a] for a in "XYZ")
    hb = sum(p["I" + a] * paulis[a] for a in "XYZ")
    _, ua = eigh(ha)
    _, ub = eigh(hb)
    # Fix the arbitrary eigenvector signs for a continuous local axis convention.
    for rotation in (ua, ub):
        for column in range(2):
            pivot = np.argmax(np.abs(rotation[:, column]))
            rotation[:, column] *= np.sign(rotation[pivot, column].real)
    u = np.kron(ua, ub)
    rotated = u.conj().T @ ham @ u
    orientation_product = p["XI"] * p["IX"]
    orientation = (
        np.sign(orientation_product) if abs(orientation_product) > 1e-12 else 1
    )
    return dict(
        energy_basis_logical_exchange_mhz=float(rotated[1, 2].real),
        oriented_logical_exchange_mhz=float(orientation * rotated[1, 2].real),
        energy_basis_logical_zz_mhz=float(
            (rotated[3, 3] - rotated[2, 2] - rotated[1, 1] + rotated[0, 0]).real
        ),
    )
