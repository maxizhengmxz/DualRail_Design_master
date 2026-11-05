"""Fixed-control basis checks for saved parasitic OFF-idle diagnostics."""
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh

WF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WF), str(WF.parent)]
from studies.parasitic_absorption import AUDIT, rebuild
from integrated_readout_range import frozen_model
from seven_readout import SevenReadout
from floating_coupler.channel_policy import logical_exchange_from_eigenpairs, logical_exchange_definition


def logical_audit(physical, off_ghz):
    """Same seven-mode eigenproblem and four-logical projection, even sector only."""
    ph, _ = rebuild(physical)
    system = SevenReadout(frozen_model(ph), ph, off_ghz, basis=AUDIT, off_only=True)
    sector = np.flatnonzero(system.parity == 1)
    block = system.ham[sector, :][:, sector].toarray()
    energy, vectors = eigh(block, subset_by_index=(0, system.eigenlevels - 1), check_finite=False)
    refs = [system.refs[(label, 0, 0)] for label in ('00', '01', '10', '11')]
    positions = np.searchsorted(sector, refs)
    exchange = logical_exchange_from_eigenpairs(energy, vectors[positions, :])
    exchange['observable_definition'] = logical_exchange_definition('seven')
    matrix = np.asarray(exchange['local_energy_hamiltonian_ghz'])
    channels = []
    for left, right, i, j in [('01L', '10L', 1, 2), ('00L', '11L', 0, 3)]:
        coupling = float(matrix[i, j] * 1e9)
        detuning = float((matrix[i, i] - matrix[j, j]) * 1e9)
        channels.append(dict(left=left, right=right, g_hz=coupling,
                             detuning_mhz=detuning / 1e6,
                             amplitude_ratio=abs(coupling / detuning),
                             eigenstate_admixture=(coupling / detuning)**2))
    return dict(logical_exchange=exchange, logical_channels=channels,
                logical_basis=AUDIT, scope='Audit-basis even-sector solve. Physical OFF and participation use the separately recorded odd-sector basis.')


