"""Cyclic charge coupling J and local 01-exchange magnitude J01.

H_int/h = J_ij n_i n_j, with n in Cooper-pair units. J is in GHz;
J01 = |<10|H_int/h|01>| = |J n_i[1,0] n_j[0,1]| is also in GHz.
Neither is an angular rate or the resonant avoided-crossing gap 2*J01.
EC and J must come from the same loaded inverse capacitance matrix.
"""

import numpy as np
from scipy.constants import elementary_charge as e, h


def charge_coupling_ghz(inverse_capacitance_per_f):
    """Return 4 e^2 C^-1/h in GHz; use off-diagonals as J_ij.

    The diagonal is 8*EC, not an additional interaction term.
    """
    return 4 * e**2 * np.asarray(inverse_capacitance_per_f) / (h * 1e9)


def exchange_j01_ghz(j_ghz, charge_01_i, charge_01_j):
    """Magnitude of the local product-state 10 <-> 01 matrix element."""
    return float(abs(j_ghz * np.conj(charge_01_i) * charge_01_j))


def loaded_charging_energies_ghz(inverse_capacitance_per_f):
    """EC_i/h = e^2 (C^-1)_ii/(2h), in GHz, from the loaded network."""
    return e**2 * np.diag(np.asarray(inverse_capacitance_per_f)) / (2 * h * 1e9)


# Reduced two-node toy-model conversion convention; not a full-circuit shunt reconstruction.
def charging_energy_to_capacitance_fF(EC_GHz: float) -> float:
    """Convert EC/h in GHz to C=e^2/(2 EC) in fF."""
    ec_hz = EC_GHz * 1.0e9
    capacitance_f = e**2 / (2.0 * h * ec_hz)
    return capacitance_f * 1.0e15


def capacitance_to_charging_energy_GHz(capacitance_fF: float) -> float:
    """Convert capacitance in fF to EC/h in GHz."""
    capacitance_f = capacitance_fF * 1.0e-15
    ec_hz = e**2 / (2.0 * h * capacitance_f)
    return ec_hz / 1.0e9


def mutual_capacitance_to_charge_coupling_GHz(
    mutual_capacitance_fF: float,
    node_capacitance_1_fF: float,
    node_capacitance_2_fF: float,
) -> float:
    """Return J/h in GHz for H_int/h = J n1 n2.

    The two-node capacitance matrix is assumed to be
    [[C1, -Cm], [-Cm, C2]], where C1 and C2 are its full diagonal entries.
    """
    cm = mutual_capacitance_fF * 1.0e-15
    c1 = node_capacitance_1_fF * 1.0e-15
    c2 = node_capacitance_2_fF * 1.0e-15
    if cm < 0.0 or c1 <= 0.0 or c2 <= 0.0:
        raise ValueError("Capacitances must be positive, with Cm allowed to be zero")
    determinant = c1 * c2 - cm**2
    if determinant <= 0.0:
        raise ValueError("Capacitance matrix is not positive definite")
    coupling_hz = (
        4.0 * e**2 / h * cm / determinant
    )
    return coupling_hz / 1.0e9


def charge_coupling_to_mutual_capacitance_fF(
    coupling_GHz: float,
    node_capacitance_1_fF: float,
    node_capacitance_2_fF: float,
) -> float:
    """Invert mutual_capacitance_to_charge_coupling_GHz for positive Cm."""
    if coupling_GHz < 0.0:
        raise ValueError("This helper expects a non-negative coupling magnitude")
    if coupling_GHz == 0.0:
        return 0.0
    c1 = node_capacitance_1_fF * 1.0e-15
    c2 = node_capacitance_2_fF * 1.0e-15
    if c1 <= 0.0 or c2 <= 0.0:
        raise ValueError("Node capacitances must be positive")
    coupling_hz = coupling_GHz * 1.0e9
    scale = 4.0 * e**2 / h
    # Stable positive solution of J Cm^2 + scale Cm - J C1 C2 = 0.
    discriminant_root = np.sqrt(scale**2 + 4.0 * coupling_hz**2 * c1 * c2)
    cm = 2.0 * coupling_hz * c1 * c2 / (scale + discriminant_root)
    return float(cm * 1.0e15)


def mutual_capacitance_to_charge_coupling_from_EC_GHz(
    mutual_capacitance_fF: float,
    EC1_GHz: float,
    EC2_GHz: float,
) -> float:
    """Convenience estimate using C_i=e^2/(2h EC_i)."""
    return mutual_capacitance_to_charge_coupling_GHz(
        mutual_capacitance_fF,
        charging_energy_to_capacitance_fF(EC1_GHz),
        charging_energy_to_capacitance_fF(EC2_GHz),
    )


def charge_coupling_to_mutual_capacitance_from_EC_GHz(
    coupling_GHz: float,
    EC1_GHz: float,
    EC2_GHz: float,
) -> float:
    """Convenience inverse using C_i=e^2/(2h EC_i)."""
    return charge_coupling_to_mutual_capacitance_fF(
        coupling_GHz,
        charging_energy_to_capacitance_fF(EC1_GHz),
        charging_energy_to_capacitance_fF(EC2_GHz),
    )
