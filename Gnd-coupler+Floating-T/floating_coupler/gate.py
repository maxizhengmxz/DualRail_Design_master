"""Noiseless static-hold CZ diagnostic with frozen five-transmon capacitances.

Initial/readout states are isolated-pair logical products at the ON rail
frequencies, with C in its local ground state. No ramp or idle-to-on transport
is included. Report leakage-inclusive fidelity and survival together.
"""

from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq, minimize

from .design import ej_for_frequency
from .pairs import PairModel
from .channel_policy import require_selection,LOGICAL_ORDER


def gate_metrics(matrix):
    """Same local-Z convention as the standalone CZ workflow; leakage included."""
    diagonal = np.diag(matrix)
    phases = np.angle(diagonal)
    target_phases = np.array([phases[0], phases[1], phases[2], phases[1] + phases[2] - phases[0] + np.pi])
    norm = float(np.sum(abs(matrix)**2))
    fidelity = float((norm + abs(np.sum(np.exp(-1j * target_phases) * diagonal))**2) / 20)
    survival = norm / 4
    phase_error = float(np.angle(-diagonal[3] * diagonal[0] * np.conj(diagonal[2] * diagonal[1])))
    return {"raw_cz_fidelity": fidelity, "postselected_cz_fidelity": fidelity / survival,
            "logical_survival": survival, "average_leakage": 1 - survival,
            "average_wrong_logical_population": float((norm - np.sum(abs(diagonal)**2)) / 4),
            "phase_error_rad": phase_error,
            "return_by_input": (abs(diagonal)**2).tolist(),
            "leakage_by_input": (1 - np.sum(abs(matrix)**2, axis=0)).tolist()}


class HoldGate:
    def __init__(self, model, basis):
        self.model = model
        self.basis = basis
        self.selection=require_selection(model.electrostatics.get('channel_selection'))
        if self.selection.moved_pair!='B':
            raise ValueError('This hold kernel moves B only; do not silently move the wrong rail')

    @lru_cache(maxsize=32)
    def spectral(self, fb, fc):
        ej = self.model.ej.copy()
        for i in (3, 4):
            ej[i] = ej_for_frequency(fb, self.model.ec[i], self.basis["ncut"])
        system = PairModel(replace(self.model, ej=ej), **self.basis)
        ham, ej_c = system.hamiltonian(fc)
        sector = np.flatnonzero(system.parity == 1)
        # FULL even-sector spectrum: exact unitary within this product basis.
        energy, vectors = eigh(ham[np.ix_(sector, sector)])
        logical = [system.labels.index(label) for label in LOGICAL_ORDER]
        references = np.searchsorted(sector, system.ref_indices[logical])
        coefficients = vectors[references, :].T
        aux = np.searchsorted(sector, system.ref_indices[system.labels.index(self.selection.auxiliary)])
        source_energy=sum(pair['energy'][pair['one'][int(bit)]] for pair,bit in
                          zip((system.a,system.b),self.selection.source_logical,strict=True))
        pair=system.a if self.selection.auxiliary.startswith('A') else system.b
        target_energy=pair['energy'][pair['auxiliary'][self.selection.auxiliary[1:]]]
        ej[2] = ej_c
        return energy - energy[0], coefficients, vectors[aux, :], {
            "ej_c_ghz": ej_c, "rail_ej_ghz": ej.tolist(),
            "isolated_pair_splittings_mhz": [system.a["splitting_mhz"], system.b["splitting_mhz"]],
            "channel_selection":self.selection.to_dict(),
            "isolated_collision_detuning_mhz": float((source_energy-target_energy) * 1000)}

    def at(self, fb, fc, time_ns):
        energy, coefficients, aux, meta = self.spectral(float(fb), float(fc))
        evolved = np.exp(-2j * np.pi * energy * time_ns)[:, None] * coefficients
        matrix = coefficients.T @ evolved
        return dict(gate_metrics(matrix), fb_ghz=float(fb), fc_ghz=float(fc), time_ns=float(time_ns),
                    auxiliary_population_by_input=(abs(aux @ evolved)**2).tolist(), **meta)

    def isolated_collision(self, fc, bracket):
        return brentq(lambda fb: self.spectral(float(fb), float(fc))[3]["isolated_collision_detuning_mhz"],
                      *bracket, xtol=1e-10)

    def dressed_collision(self, fc, bracket):
        def residual(fb):
            ej = self.model.ej.copy()
            for i in (3, 4):
                ej[i] = ej_for_frequency(fb, self.model.ec[i], self.basis["ncut"])
            from .paired_design import selected_channel
            return selected_channel(PairModel(replace(self.model, ej=ej), **self.basis).evaluate(fc))["detuning_mhz"]
        return brentq(residual, *bracket, xtol=1e-10)


def search_hold(model, spec):
    """Small explicit scan followed by local searches; no global optimality claim."""
    gate = HoldGate(model, spec["search_basis"])
    starts = []
    for fc in spec["coupler_seeds_ghz"]:
        collision_method = gate.dressed_collision if spec.get("collision_seed", "isolated") == "dressed" else gate.isolated_collision
        collision = collision_method(fc, spec["b_frequency_bounds_ghz"])
        for offset in spec["collision_offsets_mhz"]:
            fb = collision + offset / 1000
            if not spec["b_frequency_bounds_ghz"][0] <= fb <= spec["b_frequency_bounds_ghz"][1]:
                continue
            times = [spec["fixed_time_ns"]] if "fixed_time_ns" in spec else np.linspace(*spec["time_bounds_ns"], spec["time_samples"])
            rows = [gate.at(fb, fc, t) for t in times]
            starts.append(max(rows, key=lambda r: r["raw_cz_fidelity"]))
    starts.sort(key=lambda r: r["raw_cz_fidelity"], reverse=True)
    results = []
    # Scale B/C coordinates in MHz for stable finite differences.
    bounds = [(1000 * x, 1000 * y) for x, y in (spec["b_frequency_bounds_ghz"], spec["coupler_bounds_ghz"])]
    if "fixed_time_ns" not in spec:
        bounds.append(spec["time_bounds_ns"])
    for initial in starts[:spec.get("local_starts", 4)]:
        def objective(x):
            duration = spec["fixed_time_ns"] if "fixed_time_ns" in spec else x[2]
            return -gate.at(x[0] / 1000, x[1] / 1000, duration)["raw_cz_fidelity"]
        initial_x = [initial["fb_ghz"] * 1000, initial["fc_ghz"] * 1000]
        if "fixed_time_ns" not in spec:
            initial_x.append(initial["time_ns"])
        fit = minimize(objective, initial_x,
                       method="Nelder-Mead", bounds=bounds,
                       options={"maxiter": 220, "xatol": 2e-4, "fatol": 1e-10})
        duration = spec["fixed_time_ns"] if "fixed_time_ns" in spec else fit.x[2]
        row = gate.at(fit.x[0] / 1000, fit.x[1] / 1000, duration)
        row.update(optimizer_success=bool(fit.success), optimizer_message=str(fit.message))
        results.append(row)
    return {"grid_best": starts[:10], "local_results": results,
            "best": max(results, key=lambda r: r["raw_cz_fidelity"])}
