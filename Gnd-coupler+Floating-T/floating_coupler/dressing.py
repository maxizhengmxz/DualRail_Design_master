"""Four-rail local frequency and coupler participation diagnostics.

Local self frequencies are diagonals of a symmetric-orthogonalized effective
four-rail Hamiltonian in the physical single-excitation reference basis. They
are not the four hybridized eigenfrequencies. No RWA is used.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from .pairs import PairModel


def rail_dressing(model, fc, basis):
    system = PairModel(model, **basis)
    ham, ej_c = system.hamiltonian(fc)
    p, c, _ = system.dims
    sector = np.flatnonzero(system.parity == -1)
    even = np.flatnonzero(system.parity == 1)
    ground_energy, ground_vector = eigh(ham[np.ix_(even, even)], subset_by_index=(0, 0))
    energy, vectors = eigh(ham[np.ix_(sector, sector)], subset_by_index=(0, min(40, len(sector)) - 1))
    local_levels = basis["local_levels"]
    physical_states = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    c0 = np.eye(c)[:, 0]
    references = []
    for a1, a2, b1, b2 in physical_states:
        va = system.a["vectors"][a1 * local_levels + a2, :p]
        vb = system.b["vectors"][b1 * local_levels + b2, :p]
        references.append(np.kron(np.kron(va, c0), vb)[sector])
    reference = np.column_stack(references)
    gram_energy, gram_vectors = eigh(reference.T @ reference)
    if gram_energy.min() < 1e-8:
        raise ValueError("Physical rail reference projection is rank deficient")
    reference = reference @ ((gram_vectors / np.sqrt(gram_energy)) @ gram_vectors.T)
    projection = reference.T @ vectors
    _, columns = linear_sum_assignment(-abs(projection)**2)
    u, singular, vh = np.linalg.svd(projection[:, columns])
    rotation = u @ vh
    effective = (rotation * (energy[columns] - ground_energy[0])) @ rotation.T
    # Assign the SAME selected states to isolated-pair lower/upper mode references.
    pair_labels = ["0g", "1g", "g0", "g1"]
    pair_indices = [system.ref_indices[system.labels.index(label)] for label in pair_labels]
    pair_projection = vectors[np.searchsorted(sector, pair_indices), :][:, columns]
    _, mode_assignment = linear_sum_assignment(-abs(pair_projection)**2)
    c_level = np.indices(system.dims)[1].ravel()
    ground_pc = float((c_level[even] > 0) @ abs(ground_vector[:, 0])**2)
    modes = []
    for label, k in zip(pair_labels, mode_assignment, strict=True):
        state = vectors[:, columns[k]]
        pc = float((c_level[sector] > 0) @ abs(state)**2)
        modes.append({"label": label, "frequency_ghz": float(energy[columns[k]] - ground_energy[0]),
                      "coupler_excited_probability": pc,
                      "excess_coupler_excited_probability": pc - ground_pc})
    pairs = []
    for indices in ((0, 1), (2, 3)):
        block = effective[np.ix_(indices, indices)]
        energies, states = eigh(block)
        pairs.append({"local_detuning_mhz": float((block[0, 0] - block[1, 1]) * 1000),
                      "exchange_mhz": float(block[0, 1] * 1000),
                      "splitting_mhz": float((energies[1] - energies[0]) * 1000),
                      "lower_mode_right_port_weight": float(abs(states[1, 0])**2),
                      "eigenvectors": states.tolist()})
    return {"fc_ghz": float(fc), "ej_c_ghz": ej_c, "basis": basis,
            "node_order": ["A1", "A2", "B1", "B2"],
            "local_self_frequencies_ghz": effective.diagonal().tolist(),
            "effective_four_rail_hamiltonian_ghz": effective.tolist(),
            "minimum_subspace_overlap": float(singular.min()**2),
            "ground_coupler_excited_probability": ground_pc,
            "modes": modes, "pair_blocks": pairs}
