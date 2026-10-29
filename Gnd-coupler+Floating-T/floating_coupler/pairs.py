"""Pair-eigenstate refinement of a floating five-transmon static off point.

No readout model. Energies and Hamiltonians are H/h in GHz. Both excitation
parities are retained; no rotating-wave approximation is used. Pair truncation
and local transmon truncation are independent and must be converged separately.
"""

from itertools import product, combinations
from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares, linear_sum_assignment, brentq

from core.transmon import Basis
from .model import FULL, build_model, local_spectrum
from .design import ej_for_frequency
from .channel_policy import (LOGICAL_ORDER,one_excitation_indices,logical_energy_frame,
    logical_exchange_from_eigenpairs,off_target_definition,physical_exchange_from_eigenpairs)


def pair_spectrum(model, indices, levels, ncut):
    """Exact pair spectrum, cached only by all invariant physics inputs."""
    i,j=indices
    result=_pair_spectrum_cached(float(model.ec[i]),float(model.ec[j]),
            float(model.ej[i]),float(model.ej[j]),float(model.j[i,j]),levels,ncut).copy()
    for key in ('one','two','one_excitation_weights'):result[key]=tuple(result[key])
    for key in ('auxiliary','auxiliary_bare_weights'):result[key]=result[key].copy()
    return result


@lru_cache(maxsize=256)
def _pair_spectrum_cached(ec0,ec1,ej0,ej1,coupling,levels,ncut):
    basis=Basis(ncut=ncut,transmon_levels=levels)
    spectra=[local_spectrum(ej0,ec0,basis),local_spectrum(ej1,ec1,basis)]
    identity = np.eye(levels)
    ham = (np.kron(np.diag(spectra[0][0]), identity)
           + np.kron(identity, np.diag(spectra[1][0]))
           + coupling * np.kron(spectra[0][1], spectra[1][1]))
    energies, vectors = eigh(ham)
    one_refs = [np.ravel_multi_index(state, (levels, levels)) for state in ((1, 0), (0, 1))]
    two_refs = [np.ravel_multi_index(state, (levels, levels)) for state in ((2, 0), (1, 1), (0, 2))]
    one = list(one_excitation_indices(energies,np.sum(vectors[one_refs]**2,axis=0)))
    two = sorted(np.argsort(np.sum(vectors[two_refs]**2, axis=0))[-3:], key=lambda k: energies[k])
    # Each physical auxiliary is assigned distinctly within the two-excitation group.
    _, assignment = linear_sum_assignment(-vectors[np.ix_(two_refs, two)]**2)
    auxiliary = {label: two[k] for label, k in zip(("20", "11", "02"), assignment, strict=True)}
    # Phase conventions are fixed by bare reference amplitudes, not arbitrary eigensolver signs.
    for k in range(vectors.shape[1]):
        reference = int(np.argmax(abs(vectors[:, k])))
        if k in one:
            reference = one_refs[0]
        for label, index in auxiliary.items():
            if k == index:
                reference = two_refs[("20", "11", "02").index(label)]
        if vectors[reference, k] < 0:
            vectors[:, k] *= -1
    parity_op = (-1.) ** np.indices((levels, levels)).sum(axis=0).ravel()
    parity = np.rint(parity_op @ (vectors**2)).astype(int)
    charges = [vectors.T @ np.kron(spectra[0][1], identity) @ vectors,
               vectors.T @ np.kron(identity, spectra[1][1]) @ vectors]
    energies=energies-energies[0]
    for array in [energies,vectors,parity,*charges]:array.setflags(write=False)
    return {"energy": energies, "vectors": vectors, "charge": tuple(charges),
            "parity": parity, "one": one, "two": two, "auxiliary": auxiliary,
            "splitting_mhz": float((energies[one[1]] - energies[one[0]]) * 1000),
            "one_excitation_weights": np.sum(vectors[np.ix_(one_refs, one)]**2, axis=0).tolist(),
            "auxiliary_bare_weights": {label: float(vectors[two_refs[i], auxiliary[label]]**2)
                                       for i, label in enumerate(("20", "11", "02"))}}


def calibrate_pairs(spec):
    """Keep the seed's inter-pair capacitors; solve rail capacitors and all shunts."""
    seed = spec["seed_three_config"]
    if not isinstance(seed, dict):
        raise ValueError("Supply the complete seed three-transmon config")
    transmons = {name: dict(seed["transmons"][name]) for name in ("A2", "C", "B1")}
    for name in ("A1", "B2"):
        ec = spec["extra_ec_ghz"][name]
        frequency = spec["pair_bare_frequencies_ghz"][0 if name == "A1" else 1]
        transmons[name] = {"ec_ghz": ec, "ej_ghz": ej_for_frequency(frequency, ec, 35)}
    for name, frequency in zip(("A2", "B1"), spec["pair_bare_frequencies_ghz"], strict=True):
        transmons[name]["ej_ghz"] = ej_for_frequency(frequency, transmons[name]["ec_ghz"], 35)
    original_caps = seed["electrostatics"]["capacitors"]

    def config_at(capacitors):
        caps = [dict(item) for item in original_caps]
        for (a, b), cap in zip((("A1", "A2"), ("B1", "B2")), capacitors, strict=True):
            for sign in ("+", "-"):
                caps.append({"from": a + sign, "to": b + sign, "capacitance_ff": float(cap)})
        result={"nodes": list(FULL), "transmons": transmons,
                "electrostatics": {"mode": "island_capacitors", "capacitors": caps},
                "basis": {"levels": 5, "ncut": 35}}
        selection=spec.get('channel_selection') or seed.get('channel_selection')
        if selection is not None:
            from .channel_policy import require_selection
            result['channel_selection']=require_selection(selection).to_dict()
        return result

    def residual(caps):
        model = build_model(config_at(caps))
        splits = [pair_spectrum(model, pair, 5, 35)["splitting_mhz"] for pair in ((0, 1), (3, 4))]
        return (np.asarray(splits) - spec["pair_splittings_mhz"]) / spec["pair_splittings_mhz"]

    lower, upper = spec.get("pair_capacitor_bounds_ff", [1e-6, 30.])
    fit = least_squares(residual, [3., 5.], bounds=([lower, lower], [upper, upper]),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not fit.success or np.max(abs(residual(fit.x))) > 1e-7:
        raise ValueError("Could not match pair splittings within capacitor search bounds")
    return config_at(fit.x)


class PairModel:
    """Isolated pair A x local coupler x isolated pair B, all cross terms retained."""

    def __init__(self, model, local_levels=5, pair_states=12, coupler_levels=5, ncut=35):
        if model.names != FULL:
            raise ValueError("Pair model requires full five-transmon node order")
        if not 6 <= pair_states <= local_levels**2:
            raise ValueError("Keep at least six pair states and no more than the local product basis")
        self.model = model
        self.basis = {"local_levels": local_levels, "pair_states": pair_states,
                      "coupler_levels": coupler_levels, "ncut": ncut}
        self.a = pair_spectrum(model, (0, 1), local_levels, ncut)
        self.b = pair_spectrum(model, (3, 4), local_levels, ncut)
        self.dims = (pair_states, coupler_levels, pair_states)
        if np.prod(self.dims) > 4096:
            raise ValueError("Pair product dimension exceeds 4096")
        self.references = [("gg", 0, 0)]
        self.references += [(f"{i}g", self.a["one"][i], 0) for i in (0, 1)]
        self.references += [(f"g{j}", 0, self.b["one"][j]) for j in (0, 1)]
        self.references += [(f"{i}{j}", self.a["one"][i], self.b["one"][j]) for i, j in product((0, 1), repeat=2)]
        self.references += [("A" + label, index, 0) for label, index in self.a["auxiliary"].items()]
        self.references += [("B" + label, 0, index) for label, index in self.b["auxiliary"].items()]
        if max(max(a, b) for _, a, b in self.references) >= pair_states:
            raise ValueError("Pair truncation omits a required reference state")
        self.ref_indices = np.array([np.ravel_multi_index((a, 0, b), self.dims) for _, a, b in self.references])
        self.labels = [label for label, _, _ in self.references]
        self.parity = (self.a["parity"][:pair_states, None, None]
                       * (-1.)**np.arange(coupler_levels)[None, :, None]
                       * self.b["parity"][None, None, :pair_states]).ravel()

    def hamiltonian(self, fc):
        p, c, _ = self.dims
        ej_c = ej_for_frequency(fc, self.model.ec[2], self.basis["ncut"])
        energy_c, nc = local_spectrum(ej_c, self.model.ec[2], Basis(
            ncut=self.basis["ncut"], transmon_levels=c))
        energy = (self.a["energy"][:p, None, None] + energy_c[None, :, None]
                  + self.b["energy"][None, None, :p]).ravel()
        ham = np.diag(energy)
        identity_p, identity_c = np.eye(p), np.eye(c)
        qa = [op[:p, :p] for op in self.a["charge"]]
        qb = [op[:p, :p] for op in self.b["charge"]]
        for i in (0, 1):
            ham += self.model.j[i, 2] * np.kron(np.kron(qa[i], nc), identity_p)
        for j in (0, 1):
            ham += self.model.j[2, 3 + j] * np.kron(np.kron(identity_p, nc), qb[j])
        for i, j in product((0, 1), repeat=2):
            ham += self.model.j[i, 3 + j] * np.kron(np.kron(qa[i], identity_c), qb[j])
        return ham, ej_c

    def physical_off(self, fc):
        """Odd sector only for frozen-hardware OFF searches; no ON/readout work."""
        from .off_observables import physical_single_excitation_references
        ham,ej_c=self.hamiltonian(fc)
        sector=np.flatnonzero(self.parity==-1)
        energies,vectors=eigh(ham[np.ix_(sector,sector)],
                             subset_by_index=(0,min(47,len(sector)-1)))
        refs=physical_single_excitation_references(self)
        result=physical_exchange_from_eigenpairs(energies,refs[:,sector]@vectors)
        return dict(off_definition=off_target_definition('five'),physical_exchange=result,
                    coupler_ej_ghz=ej_c,coupler_bare_f01_ghz=float(fc))

    def evaluate(self, fc):
        ham, ej_c = self.hamiltonian(fc)
        count = len(self.references)
        effective = np.zeros((count, count))
        overlaps, weights, occupations, residuals = [], {}, {}, []
        logical_exchange=None;physical_exchange=None
        from .off_observables import physical_single_excitation_references
        physical_refs=physical_single_excitation_references(self)
        c_level = np.indices(self.dims)[1].ravel()
        for parity in (1, -1):
            sector = np.flatnonzero(self.parity == parity)
            reference_rows = np.flatnonzero(self.parity[self.ref_indices] == parity)
            local_refs = np.searchsorted(sector, self.ref_indices[reference_rows])
            sector_ham = ham[np.ix_(sector, sector)]
            n_eigen = min(len(sector), max(48, 4 * len(reference_rows)))
            energies, vectors = eigh(sector_ham, subset_by_index=(0, n_eigen - 1))
            projection = vectors[local_refs]
            if parity == 1:
                logical_refs=np.searchsorted(sector,[self.ref_indices[self.labels.index(label)] for label in LOGICAL_ORDER])
                logical_exchange=logical_exchange_from_eigenpairs(energies,vectors[logical_refs,:])
            if parity == -1:
                physical_exchange=physical_exchange_from_eigenpairs(energies,physical_refs[:,sector]@vectors)
            _, columns = linear_sum_assignment(-abs(projection)**2)
            if max(columns) >= n_eigen - 2 and n_eigen < len(sector):
                raise ValueError("Tracked eigenlevel window is too small")
            u, singular, vh = np.linalg.svd(projection[:, columns])
            rotation = u @ vh
            effective[np.ix_(reference_rows, reference_rows)] = (rotation * energies[columns]) @ rotation.T
            overlaps.append(float(singular.min()**2))
            occupation = c_level[sector] @ abs(vectors[:, columns])**2
            for k, row in enumerate(reference_rows):
                weights[self.labels[row]] = float(abs(projection[k, columns[k]])**2)
                occupations[self.labels[row]] = float(occupation[k])
            residuals.append(float(np.max(np.linalg.norm(sector_ham @ vectors[:, columns]
                                                          - vectors[:, columns] * energies[columns], axis=0))))
        index = {label: k for k, label in enumerate(self.labels)}

        def channel(left, right):
            i, j = index[left], index[right]
            return {"from": left, "to": right, "matrix_element_mhz": float(effective[i, j] * 1000),
                    "diagonal_detuning_mhz": float((effective[i, i] - effective[j, j]) * 1000)}

        diagonal = effective.diagonal()
        logical = list(LOGICAL_ORDER)
        auxiliaries = [label for label in self.labels if label.startswith(("A", "B"))]
        logical_block = effective[np.ix_([index[label] for label in logical], [index[label] for label in logical])]
        energy_logical = local_energy_frame(logical_block)
        result = {"coupler_bare_f01_ghz": float(fc), "coupler_ej_ghz": ej_c,
                "channel_selection": self.model.electrostatics.get('channel_selection'),
                "logical_exchange":logical_exchange,"physical_exchange":physical_exchange,
                "off_definition":off_target_definition('five'),
                "legacy_fixed_source_diagnostics": {'cz_0020':'01-B20','cz_0200':'10-A02'},
                "minimum_subspace_overlap": min(overlaps),
                "assigned_reference_weights": weights,
                "coupler_mean_local_level": occupations,
                "maximum_eigen_residual_ghz": max(residuals),
                "cz_0020": channel("01", "B20"), "cz_0200": channel("10", "A02"),
                "logical_zz_mhz": float((diagonal[index["11"]] - diagonal[index["10"]]
                                          - diagonal[index["01"]] + diagonal[index["00"]]) * 1000),
                "logical_local_energy_frame": energy_logical,
                "logical_local_energy_frame_scope":"raw retained logical-block diagnostic; OFF uses logical_exchange with virtual auxiliaries included",
                "logical_offdiagonals": [channel(a, b) for a, b in combinations(logical, 2)],
                "logical_to_auxiliary": [channel(a, b) for a, b in product(logical, auxiliaries)],
                "single_excitation_transfer": [channel(f"{i}g", f"g{j}") for i, j in product((0, 1), repeat=2)],
                "reference_labels": self.labels, "effective_hamiltonian_ghz": effective.tolist()}
        if result['channel_selection'] is not None:
            from .paired_design import selected_channel,channel_scores
            result['selected_cz_channel']=selected_channel(result)
            chosen=result['selected_cz_channel']
            result['selected_channel_score_diagnostics']=channel_scores(result,chosen['source_logical'],chosen['auxiliary'])
        return result


def local_energy_frame(logical_ghz):
    """Shared local-axis diagnostic; raw retained block is not the OFF observable."""
    return logical_energy_frame(logical_ghz)


def refine_pair_off(model, channel, bracket, basis, scan_points=17,selection=None):
    from .channel_policy import require_selection
    from .off_search import exchange_roots
    if channel not in (None,'physical_0100_0010'):
        raise ValueError('Temporary OFF target is physical 0100-0010; logical exchange and CZ auxiliaries are diagnostics')
    if selection is not None:
        selected=require_selection(selection)
        model=replace(model,electrostatics=dict(model.electrostatics,channel_selection=selected.to_dict()))
    system=PairModel(model,**basis)
    cache={}
    def at(fc):
        if fc not in cache:cache[fc]=system.physical_off(float(fc))
        return cache[fc]
    def objective(fc):
        row=at(float(fc))
        if not row['physical_exchange']['valid']:
            raise ValueError('Unreliable physical single-excitation subspace in OFF search')
        return row['physical_exchange']['g_mhz']
    grid=np.linspace(*bracket,scan_points)
    samples=[at(float(fc)) for fc in grid]
    roots=[system.evaluate(fc) for fc in exchange_roots(objective,grid,tolerance=1e-6)]
    return {"off_definition":off_target_definition('five'),"basis": basis, "samples": samples, "roots": roots,
            "pair_a": {k: system.a[k] for k in ("splitting_mhz", "one_excitation_weights", "auxiliary_bare_weights")},
            "pair_b": {k: system.b[k] for k in ("splitting_mhz", "one_excitation_weights", "auxiliary_bare_weights")}}
