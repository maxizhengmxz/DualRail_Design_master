"""Static ON-position CZ in canonically coupler-dressed matter references.

The retained matter space includes all six rail two-excitation auxiliaries,
not just four logical eigenstates. Its canonical embedding into full-system
eigenstates dresses the references without eliminating the CZ excursion.
Logical axes are then aligned to their local energy terms. No switching/ramp.
"""

from dataclasses import replace

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from .design import ej_for_frequency
from .gate import gate_metrics
from .pairs import PairModel


def energy_axis_matter_hamiltonian(detail, logical_rotations=None):
    """Rotate the logical block AND its auxiliary connections consistently.

    Auxiliaries keep their canonical nominal-pair branch labels. Supplying
    rotations explicitly permits a comparison in one fixed logical axis frame.
    """
    ham = np.asarray(detail["effective_hamiltonian_ghz"])
    logical = [detail["reference_labels"].index(label) for label in ("00", "01", "10", "11")]
    ua, ub = (detail["logical_local_energy_frame"]["local_rotations"]
              if logical_rotations is None else logical_rotations)
    transform = np.eye(len(ham))
    transform[np.ix_(logical, logical)] = np.kron(ua, ub)
    return transform.T @ ham @ transform


def secular_logical_shifts(detail, minimum_assignment_weight=0.9):
    """Conditional eigenenergy shift including retained auxiliary mediation.

    A logical-block diagonal ZZ omits virtual shifts from auxiliaries that are
    still retained. Return state weights so a near-resonant ambiguous assignment
    cannot be mistaken for a reliable four-state dispersive description.
    """
    ham = energy_axis_matter_hamiltonian(detail)
    logical = [detail["reference_labels"].index(label) for label in ("00", "01", "10", "11")]
    energy, vectors = eigh(ham)
    _, columns = linear_sum_assignment(-abs(vectors[logical, :])**2)
    assigned = energy[columns]
    weights = [float(abs(vectors[i, j])**2) for i, j in zip(logical, columns, strict=True)]
    return {"logical_eigenenergies_ghz": assigned.tolist(), "assignment_weights": weights,
            "secular_zz_hz": float((assigned[3] - assigned[2] - assigned[1] + assigned[0]) * 1e9),
            "well_identified": bool(min(weights) >= minimum_assignment_weight)}


def dressed_on_spectrum(model, fb, fc, basis):
    ej = model.ej.copy()
    for i in (3, 4):
        ej[i] = ej_for_frequency(fb, model.ec[i], basis["ncut"])
    result = PairModel(replace(model, ej=ej), **basis).evaluate(fc)
    ham = np.asarray(result["effective_hamiltonian_ghz"])
    logical = [result["reference_labels"].index(label) for label in ("00", "01", "10", "11")]
    ua, ub = result["logical_local_energy_frame"]["local_rotations"]
    reference = np.zeros((len(ham), 4))
    reference[logical] = np.kron(ua, ub)
    energy, vectors = eigh(ham)
    coefficients = vectors.T @ reference
    # Reuse this exact diagonalization for auxiliary populations too.
    result['on_auxiliary_coefficients']={label:vectors[i,:].tolist()
            for i,label in enumerate(result['reference_labels']) if label.startswith(('A','B'))}
    return energy - energy[0], coefficients, result


def dressed_on_metrics(energy, coefficients, time_ns):
    matrix = coefficients.T @ (np.exp(-2j * np.pi * energy * time_ns)[:, None] * coefficients)
    return dict(gate_metrics(matrix), time_ns=float(time_ns))


def dressed_on_metrics_many(energy,coefficients,times_ns):
    """All logical inputs and durations, one fixed-bias spectral decomposition."""
    times=np.asarray(times_ns,dtype=float)
    phases=np.exp(-2j*np.pi*times[:,None]*np.asarray(energy)[None,:])
    matrices=np.einsum('ki,tk,kj->tij',coefficients,phases,coefficients,optimize=True)
    return [dict(gate_metrics(matrix),time_ns=float(t)) for t,matrix in zip(times,matrices)]
