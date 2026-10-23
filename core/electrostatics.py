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
