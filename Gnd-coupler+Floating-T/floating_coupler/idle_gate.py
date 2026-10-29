"""Sudden idle-to-on hold and return, in a common physical charge basis.

Inputs and outputs are four exact dressed IDLE eigenstates assigned to the
nominal logical products. On/off local and pair bases are connected by actual
charge-basis overlaps, including the changing coupler vacuum. No ramp assumed.
"""

from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from .design import ej_for_frequency
from .gate import gate_metrics
from .pairs import PairModel


def charge_vectors(ej, ec, ncut, levels):
    charges = np.arange(-ncut, ncut + 1)
    ham = np.diag(4 * ec * charges**2) - ej / 2 * (
        np.diag(np.ones(2 * ncut), 1) + np.diag(np.ones(2 * ncut), -1))
    _, vectors = eigh(ham, subset_by_index=(0, levels - 1))
    n = vectors.T @ (charges[:, None] * vectors)
    signs = np.ones(levels)
    for i in range(1, levels):
        signs[i] = signs[i - 1] * (1 if n[i - 1, i] >= 0 else -1)
    return vectors * signs


class IdleDressedGate:
    def __init__(self, model, idle_fc, basis):
        self.model, self.idle_fc, self.basis = model, idle_fc, basis
        self.idle_system = PairModel(model, **basis)
        ham, self.idle_ej_c = self.idle_system.hamiltonian(idle_fc)
        self.idle_sector = np.flatnonzero(self.idle_system.parity == 1)
        energy, vectors = eigh(ham[np.ix_(self.idle_sector, self.idle_sector)],
                               subset_by_index=(0, min(48, len(self.idle_sector)) - 1))
        indices = [self.idle_system.ref_indices[self.idle_system.labels.index(label)]
                   for label in ("00", "01", "10", "11")]
        refs = np.searchsorted(self.idle_sector, indices)
        _, columns = linear_sum_assignment(-abs(vectors[refs])**2)
        self.idle_states = np.zeros((ham.shape[0], 4))
        self.idle_states[self.idle_sector] = vectors[:, columns]
        for i in range(4):
            if self.idle_states[indices[i], i] < 0:
                self.idle_states[:, i] *= -1
        self.idle_assignment_weights = abs(self.idle_states[indices, np.arange(4)])**2
        self.idle_energies = energy[columns]
        ncut, levels, c = basis["ncut"], basis["local_levels"], basis["coupler_levels"]
        self.idle_local_vectors = [charge_vectors(model.ej[i], model.ec[i], ncut, levels) for i in range(5)]
        self.idle_local_vectors[2] = charge_vectors(self.idle_ej_c, model.ec[2], ncut, c)

    def _on_problem(self, fb, fc):
        ncut, levels, c = self.basis["ncut"], self.basis["local_levels"], self.basis["coupler_levels"]
        ej = self.model.ej.copy()
        for i in (3, 4):
            ej[i] = ej_for_frequency(fb, self.model.ec[i], ncut)
        on = PairModel(replace(self.model, ej=ej), **self.basis)
        ham, ej_c = on.hamiltonian(fc)
        p = self.basis["pair_states"]
        local_overlaps = []
        for i in (3, 4):
            local_overlaps.append(charge_vectors(ej[i], self.model.ec[i], ncut, levels).T @ self.idle_local_vectors[i])
        tb = on.b["vectors"][:, :p].T @ np.kron(*local_overlaps) @ self.idle_system.b["vectors"][:, :p]
        ta = on.a["vectors"][:, :p].T @ self.idle_system.a["vectors"][:, :p]
        tc = charge_vectors(ej_c, self.model.ec[2], ncut, c).T @ self.idle_local_vectors[2]
        mapped = np.einsum("ia,jb,kc,abcq->ijkq", ta, tc, tb,
                           self.idle_states.reshape(p, c, p, 4), optimize=True).reshape(-1, 4)
        sector = np.flatnonzero(on.parity == 1)
        mapped = mapped[sector]
        norm_eigen, norm_vectors = eigh(mapped.T @ mapped)
        if norm_eigen.min() < 0.99:
            raise ValueError("On basis loses more than 1 percent of an idle logical combination")
        # Orthonormalize the projected inputs, explicitly reporting the truncation loss.
        mapped = mapped @ ((norm_vectors / np.sqrt(norm_eigen)) @ norm_vectors.T)
        energy, vectors = eigh(ham[np.ix_(sector, sector)])
        coefficients = vectors.T @ mapped
        meta = {
            "maximum_idle_state_projection_loss": float(1 - norm_eigen.min()),
            "idle_reference_assignment_weights": self.idle_assignment_weights.tolist(),
            "ej_c_ghz": ej_c}
        return energy - energy[0], coefficients, meta, vectors, sector, (ta, tc, tb)

    @lru_cache(maxsize=24)
    def spectral(self, fb, fc):
        return self._on_problem(fb, fc)[:3]

    def leakage_populations(self, fb, fc, time_ns):
        """Locate the returned component orthogonal to the dressed idle code.

        Disjoint probabilities are in the isolated IDLE pair/C product basis
        AFTER removing the exact dressed-idle logical projection. C occupation
        is local eigenlevel, not a photon count. All results average four inputs.
        """
        energy, coeff, meta, vectors, sector, transforms = self._on_problem(float(fb), float(fc))
        ta, tc, tb = transforms
        p, c, _ = self.idle_system.dims
        full = np.zeros((p * c * p, 4), complex)
        full[sector] = vectors @ (np.exp(-2j * np.pi * energy * time_ns)[:, None] * coeff)
        returned = np.einsum("ia,jb,kc,ijkq->abcq", ta, tc, tb, full.reshape(p, c, p, 4),
                             optimize=True).reshape(-1, 4)
        logical = self.idle_states.T @ returned
        outside = returned - self.idle_states @ logical
        probabilities = np.mean(abs(outside.reshape(p, c, p, 4))**2, axis=3)
        by_c = probabilities.sum(axis=(0, 2))
        nlevels = self.basis["local_levels"]
        local_number = np.indices((nlevels, nlevels)).sum(axis=0).ravel()
        na = np.rint(local_number @ self.idle_system.a["vectors"][:, :p]**2).astype(int)
        nb = np.rint(local_number @ self.idle_system.b["vectors"][:, :p]**2).astype(int)
        categories = {}
        for a in range(p):
            for b in range(p):
                key = f"A{na[a]}_B{nb[b]}"
                categories[key] = categories.get(key, 0.) + float(probabilities[a, 0, b])
        return {"fb_ghz": fb, "fc_ghz": fc, "time_ns": time_ns,
                "outside_idle_logical_probability": float(probabilities.sum()),
                "outside_by_idle_coupler_level": by_c.tolist(),
                "outside_with_c_ground_by_rail_excitation_family": categories,
                "return_projection_norm_deficit": float(1 - np.mean(np.sum(abs(returned)**2, axis=0))),
                "logical_survival_before_complement": float(np.sum(abs(logical)**2) / 4), **meta}

    def at(self, fb, fc, time_ns):
        energy, coefficients, meta = self.spectral(float(fb), float(fc))
        matrix = coefficients.T @ (np.exp(-2j * np.pi * energy * time_ns)[:, None] * coefficients)
        return dict(gate_metrics(matrix), fb_ghz=float(fb), fc_ghz=float(fc), time_ns=float(time_ns), **meta)
