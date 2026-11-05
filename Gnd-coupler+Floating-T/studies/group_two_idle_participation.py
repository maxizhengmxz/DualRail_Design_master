"""Physical rail occupations in dressed idle states for saved group-two circuits.

This opt-in diagnostic freezes every circuit and its verified OFF bias. It uses
full seven-mode odd-sector eigenvectors, retaining their absolute normalization.
"""
from pathlib import Path
import sys

import numpy as np

WF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WF), str(WF.parent)]
from integrated_readout_range import frozen_model
from seven_readout import SevenReadout
from floating_coupler.off_observables import physical_single_excitation_references
from studies.parasitic_absorption import AUDIT, rebuild


def participation(physical, frequency):
    """Project normalized dressed eigenvectors on the four physical rail states."""
    physical, _ = rebuild(physical)
    system = SevenReadout(frozen_model(physical), physical, frequency,
                          basis=AUDIT, off_only=True, eigenlevels=48).solve()
    return participation_detail(system)


def participation_detail(system):
    """Reuse an already solved odd sector without another diagonalization."""
    vacuum = np.zeros(system.photons**2)
    vacuum[0] = 1.
    references = np.kron(physical_single_excitation_references(system.pair), vacuum[None, :])
    labels = ('1000', '0100', '0010', '0001')
    result = {}
    for name, key in (('A0L', '0g'), ('A1L', '1g'), ('B0L', 'g0'), ('B1L', 'g1')):
        vector = system.vectors[(key, 0, 0)]
        overlap = system.weights[(key, 0, 0)]
        if overlap < .9:
            raise ValueError(f'Ambiguous dressed idle branch {name}: overlap {overlap}')
        weights = abs(references @ vector)**2
        result[name] = dict(physical_weights=dict(zip(labels, weights.tolist())),
                            A_weight=float(sum(weights[:2])), B_weight=float(sum(weights[2:])),
                            four_rail_weight=float(sum(weights)),
                            dressed_energy_ghz=system.eigen[(key, 0, 0)],
                            isolated_pair_reference_overlap=overlap,
                            full_eigenvector_norm=float(vector @ vector))
    return dict(basis=system.basis, off_ghz=system.fc, states=result,
                reference_norms=np.sum(references**2, axis=1).tolist(),
                definition='Absolute squared projections onto physical 1000,0100,0010,0001 with C ground and both resonators vacuum; no four-state renormalization.')


