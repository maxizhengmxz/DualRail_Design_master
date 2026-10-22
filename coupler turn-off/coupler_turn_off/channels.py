"""Residual channels and connected-state detunings, with explicit units.

Logical product order is 00,01,10,11. The ground state of a pair is g,
not 0L: each logical state contains one transmon excitation. Photon indices
count occupations of the nearby readout resonators rA and rB.
"""

import numpy as np
from scipy.linalg import eigh

PAULIS = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def local_rotation(terms):
    rotations = []
    for side in (0, 1):
        ham = sum(terms[a + "I" if side == 0 else "I" + a] * PAULIS[a] for a in "XYZ")
        _, rotation = eigh(ham)
        for j in range(2):
            pivot = np.argmax(abs(rotation[:, j]))
            rotation[:, j] *= np.sign(rotation[pivot, j].real)
        rotations.append(rotation)
    return np.kron(*rotations)


def channels(result):
    """J2L is a logical double flip; R changes nearby readout photon occupation.

    Detunings are diagonal energy differences in the SAME effective matrix,
    not bare frequencies. Matrix elements alone are not leakage probabilities.
    Photon-sector and vacuum-only effective matrices use different retained
    reference spaces; their vacuum entries need not coincide exactly.
    """
    terms = result["pauli_mhz"]
    rotation = local_rotation(terms)
    ham = sum(v * np.kron(PAULIS[k[0]], PAULIS[k[1]]) for k, v in terms.items())
    logical = (rotation.conj().T @ ham @ rotation).real
    lookup = {
        (x["side"], x["spectator"], x["mode"], x["photon"]): x["row"]
        for x in result["readout_frame_lookup"]
    }
    sectors = []
    for na, nb in ((0, 0), (1, 0), (0, 1)):
        side = 1 if nb else 0
        sectors.append(
            [
                lookup[(side, b if side == 0 else a, a if side == 0 else b, na + nb)]
                for a in (1, 2)
                for b in (1, 2)
            ]
        )
    rows = sum(sectors, [])
    full = np.asarray(result["readout_frame_matrix_ghz"])[np.ix_(rows, rows)]
    u = np.kron(np.eye(3), rotation)
    photon = (1000 * u.conj().T @ full @ u).real  # GHz -> MHz
    photon -= photon[0, 0] * np.eye(12)
    single = np.asarray(result["single_excitation_effective_matrix_mhz"])
    return dict(
        logical_order=["0L,0L", "0L,1L", "1L,0L", "1L,1L"],
        photon_sector_order=[[0, 0], [1, 0], [0, 1]],
        vacuum_matrix_mhz=logical.tolist(),
        photon_matrix_mhz=photon.tolist(),
        vacuum_transitions=[
            dict(
                bra=f"{i:02b}",
                ket=f"{j:02b}",
                matrix_element_hz=float(1e6 * logical[i, j]),
                detuning_hz=float(1e6 * (logical[i, i] - logical[j, j])),
            )
            for i in range(4)
            for j in range(i + 1, 4)
        ],
        J1_photon_sectors_hz=[
            float(1e6 * photon[4 * k + 1, 4 * k + 2]) for k in range(3)
        ],
        J2L_photon_sectors_hz=[float(1e6 * photon[4 * k, 4 * k + 3]) for k in range(3)],
        R_mhz=photon[4:8, 8:12].tolist(),
        R_detuning_mhz=(
            np.diag(photon)[4:8, None] - np.diag(photon)[None, 8:12]
        ).tolist(),
        single_excitation_transfer_mhz=result["single_excitation_exchange_matrix_mhz"],
        single_excitation_detuning_mhz=(
            np.diag(single)[:2, None] - np.diag(single)[None, 2:]
        ).tolist(),
    )
