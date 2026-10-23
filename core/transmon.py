"""Shared full-cosine single-junction transmon primitives.

Energies represent H/h in GHz; n counts Cooper pairs; offset charge is zero.
EC is the loaded electrostatic coefficient, not a dressed transition frequency.
Basis retains the existing truncation fields for transmon/resonator callers;
transmon_spectrum uses only ncut and transmon_levels. No circuit topology,
readout acceptance criteria, or optimization is defined here.
"""

from dataclasses import asdict, dataclass
import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class Basis:
    """Charge cutoff, retained local levels, photons and tracked eigenlevels."""

    ncut: int = 15  # Charge states -ncut ... ncut, for each transmon.
    transmon_levels: int = 6  # Local eigenstates kept per transmon.
    resonator_levels: int = 5
    eigenlevels: int = 20  # Full-circuit eigenstates used for branch assignment.

    def __post_init__(self):
        if any(not isinstance(v, int) for v in asdict(self).values()):
            raise ValueError("Basis sizes must be integers")
        if self.ncut < 2 or not 3 <= self.transmon_levels <= 2 * self.ncut + 1:
            raise ValueError("Invalid transmon basis")
        if self.resonator_levels < 3:
            raise ValueError("Keep at least three resonator levels")
        if not 6 <= self.eigenlevels <= self.transmon_levels**2 * self.resonator_levels:
            raise ValueError("Invalid eigenlevel count")


def transmon_spectrum(ej_ghz, ec_ghz, basis):
    """Diagonalize 4 EC n^2 - EJ cos(phi), returning local energies and n."""
    if not np.isfinite(ej_ghz) or ej_ghz <= 0:
        raise ValueError("Josephson energies must be positive, in GHz")
    n = np.arange(-basis.ncut, basis.ncut + 1)
    # One ideal junction: -EJ*cos(phi); cos(phi) shifts charge by one Cooper pair.
    ham = np.diag(4 * ec_ghz * n * n) - ej_ghz / 2 * (
        np.diag(np.ones(len(n) - 1), 1) + np.diag(np.ones(len(n) - 1), -1)
    )
    energy, states = eigh(ham, subset_by_index=(0, basis.transmon_levels - 1))
    charge = states.T @ (n[:, None] * states)
    return energy - energy[0], charge

