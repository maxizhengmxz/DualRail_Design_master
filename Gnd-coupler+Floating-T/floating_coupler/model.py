"""H/h in GHz; charge in Cooper pairs; capacitances in fF.

Floating common coordinates are eliminated in the zero total island-charge
sector of each transmon. No ground capacitor or artificial ground is added.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.constants import elementary_charge as e, h
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.optimize import least_squares, linear_sum_assignment, brentq

from core.transmon import Basis, transmon_spectrum

SCALE = e**2 / (2 * h * 1e9) * 1e15
FULL = ("A1", "A2", "C", "B1", "B2")
PORT = ("A2", "C", "B1")


@lru_cache(maxsize=1024)
def _local_spectrum(ej, ec, ncut, levels):
    """Exact full-cosine charge Hamiltonian; cache immutable local operators.

    Same tridiagonal eigensolve as the active toy. No charge-cutoff rounding,
    Duffing approximation, or change to the shared readout primitive.
    """
    if not np.isfinite(ej) or ej<=0 or not np.isfinite(ec) or ec<=0:
        raise ValueError('EC and EJ must be finite and positive')
    n=np.arange(-ncut,ncut+1,dtype=float)
    energy,vectors=eigh_tridiagonal(4*ec*n*n,np.full(len(n)-1,-ej/2),
                                    select='i',select_range=(0,levels-1))
    energy=energy-energy[0]
    charge=vectors.T@(n[:,None]*vectors)
    signs = np.ones(len(energy))
    for i in range(1, len(signs)):
        signs[i] = signs[i - 1] * (1 if charge[i - 1, i] >= 0 else -1)
    charge=charge*signs[:,None]*signs[None,:]
    energy.setflags(write=False);charge.setflags(write=False)
    return energy,charge


def local_spectrum(ej, ec, basis):
    """Immutable cached spectra, with adjacent charge elements positive."""
    return _local_spectrum(float(ej),float(ec),basis.ncut,basis.transmon_levels)


def positive(value, name):
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def floating_inverse(names, junction_ff, capacitors):
    """D.T L^+ D, where Q_islands = D q and D has columns (+1,-1).

    L may have multiple gauge null modes. Its pseudoinverse is used only on
    neutral charge vectors; this also supports fully disconnected transmons.
    """
    size = len(names)
    labels = [name + sign for name in names for sign in ("+", "-")]
    laplacian = np.zeros((2 * size, 2 * size))

    def add(a, b, cap):
        if a == b or not np.isfinite(cap) or cap < 0:
            raise ValueError("Capacitors require distinct endpoints and finite nonnegative fF")
        i, j = labels.index(a), labels.index(b)
        laplacian[i, i] += cap
        laplacian[j, j] += cap
        laplacian[i, j] -= cap
        laplacian[j, i] -= cap

    for name, cap in zip(names, junction_ff, strict=True):
        add(name + "+", name + "-", positive(cap, "junction capacitance"))
    for item in capacitors:
        cap = item["capacitance_ff"]
        if cap is None:
            raise ValueError("Missing capacitance_ff; supply fF or explicit zero")
        add(item["from"], item["to"], float(cap))
    d = np.kron(np.eye(size), np.array([[1.0], [-1.0]]))
    inverse = d.T @ np.linalg.pinv(laplacian, hermitian=True, rcond=1e-13) @ d
    return inverse, laplacian


@dataclass
class Model:
    names: tuple
    ec: np.ndarray
    ej: np.ndarray
    j: np.ndarray
    basis: Basis
    electrostatics: dict

    def subset(self, names=PORT):
        """Keep loaded coefficients; omit spectator dynamics, not capacitors."""
        indices = [self.names.index(name) for name in names]
        return Model(tuple(names), self.ec[indices], self.ej[indices],
                     self.j[np.ix_(indices, indices)], self.basis, self.electrostatics)

    def hamiltonian(self, coupler_ej=None):
        ej = self.ej.copy()
        if coupler_ej is not None:
            ej[self.names.index("C")] = positive(coupler_ej, "coupler EJ")
        spectra = [local_spectrum(a, b, self.basis) for a, b in zip(ej, self.ec, strict=True)]
        levels = self.basis.transmon_levels
        dim = levels ** len(self.names)
        if dim > 4096:
            raise ValueError("Dense Hamiltonian exceeds 4096 states; reduce retained levels")

        def embed(operators):
            out = np.ones((1, 1))
            for i in range(len(self.names)):
                out = np.kron(out, operators.get(i, np.eye(levels)))
            return out

        ham = sum(embed({i: np.diag(s[0])}) for i, s in enumerate(spectra))
        for i, j in combinations(range(len(self.names)), 2):
            ham += self.j[i, j] * embed({i: spectra[i][1], j: spectra[j][1]})
        return ham, spectra


def build_model(config):
    names = tuple(config.get("nodes", FULL))
    if names not in (FULL, PORT):
        raise ValueError("nodes must be A1,A2,C,B1,B2 or A2,C,B1 in that order")
    ec = np.array([positive(config["transmons"][n]["ec_ghz"], n + " EC") for n in names])
    ej = np.array([positive(config["transmons"][n]["ej_ghz"], n + " EJ") for n in names])
    numerical = config.get("basis", {})
    basis = Basis(ncut=numerical.get("ncut", 15), transmon_levels=numerical.get("levels", 3))
    mode = config["electrostatics"]["mode"]
    data = {"mode": mode}
    if mode == "island_capacitors":
        caps = config["electrostatics"]["capacitors"]

        def residual(log_caps):
            inverse, _ = floating_inverse(names, np.exp(log_caps), caps)
            return (SCALE * np.diag(inverse) - ec) / ec

        fit = least_squares(residual, np.log(SCALE / ec),
                            bounds=(np.full(len(names), -20.), np.full(len(names), 25.)),
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if not fit.success or np.max(np.abs(residual(fit.x))) > 1e-7:
            raise ValueError("Cannot realize loaded EC with positive intra-transmon capacitances")
        junctions = np.exp(fit.x)
        inverse, laplacian = floating_inverse(names, junctions, caps)
        j = 8 * SCALE * inverse
        data.update(junction_capacitances_ff=dict(zip(names, junctions.tolist(), strict=True)),
                    island_order=[n + s for n in names for s in ("+", "-")],
                    island_laplacian_ff=laplacian.tolist(),
                    inverse_differential_capacitance_per_ff=inverse.tolist())
    elif mode == "explicit_pads":
        from .pad_electrostatics import build_pad_circuit
        physical=config['electrostatics']
        if tuple(physical['pad_pairs'])!=names:
            raise ValueError('Explicit pad transmon order must match nodes')
        circuit=build_pad_circuit(physical['pad_pairs'],physical['ground_capacitances_ff'],
             physical.get('resonator_capacitances_ff',{}),physical['physical_links'],
             physical['pad_mutual_capacitances_ff'])
        if circuit.resonators:raise ValueError('This five-transmon Hamiltonian cannot discard explicit resonator modes')
        achieved=np.array(list(circuit.loaded_ec_ghz.values()))
        if np.max(abs(achieved/ec-1))>1e-8:raise ValueError('Loaded EC inputs disagree with explicit physical pad circuit')
        j=circuit.charge_coefficients_ghz.copy()
        data.update(circuit.summary())
    elif mode == "strengths":
        j = np.diag(8 * ec)
        seen = set()
        spectra = [transmon_spectrum(a, b, basis) for a, b in zip(ej, ec, strict=True)]
        for link in config["electrostatics"]["links"]:
            i, k = [names.index(link[key]) for key in ("from", "to")]
            pair = tuple(sorted((i, k)))
            if i == k or pair in seen:
                raise ValueError("Strength links must be unique, distinct transmon pairs")
            seen.add(pair)
            keys = [key for key in ("charge_coupling_ghz", "signed_j01_mhz") if key in link]
            if len(keys) != 1:
                raise ValueError("Specify exactly one signed J or signed J01 per strength link")
            if link[keys[0]] is None:
                raise ValueError("Missing coupling strength; supply signed value or explicit zero")
            value = float(link[keys[0]])
            if not np.isfinite(value):
                raise ValueError("Coupling must be finite")
            if keys[0] == "signed_j01_mhz":
                value /= 1000 * abs(spectra[i][1][0, 1] * spectra[k][1][0, 1])
            j[i, k] = j[k, i] = value
        data["physical_capacitors_reconstructed"] = False
    else:
        raise ValueError("Unknown electrostatics mode")
    if np.linalg.eigvalsh(j).min() <= 0:
        raise ValueError("Charging quadratic form is not positive definite")
    np.fill_diagonal(j, 0.)
    data.update(node_order=list(names), loaded_ec_ghz=ec.tolist(), charge_coupling_ghz=j.tolist())
    if config.get('channel_selection') is not None:
        from .channel_policy import require_selection
        data['channel_selection']=require_selection(config['channel_selection']).to_dict()
    return Model(names, ec, ej, j, basis, data)


def port_exchange(model, coupler_ej, minimum_overlap=0.8):
    """Two-state symmetric-orthogonalized dressed Hamiltonian.

    Select distinct exact eigenstates by maximum joint product-state overlap.
    Singular values measure validity of the retained two-state description.
    """
    ham, spectra = model.hamiltonian(coupler_ej)
    energy, vectors = eigh(ham)
    levels = model.basis.transmon_levels
    references = []
    for excited in ("A2", "B1"):
        state = [int(n == excited) for n in model.names]
        references.append(np.ravel_multi_index(tuple(state), (levels,) * len(state)))
    projection = vectors[references, :]
    _, columns = linear_sum_assignment(-abs(projection)**2)
    selected = projection[:, columns]
    u, singular, vh = np.linalg.svd(selected)
    rotation = u @ vh
    effective = (rotation * energy[columns]) @ rotation.T
    overlap = float(singular.min()**2)
    from .channel_policy import off_target_definition
    return {"off_definition":off_target_definition('three') if model.names==PORT else None,"coupler_ej_ghz": float(coupler_ej),
            "coupler_bare_f01_ghz": float(spectra[model.names.index("C")][0][1]),
            "exchange_mhz": float(effective[0, 1] * 1000),
            "reference_labels":['10','01'],
            "diagonal_detuning_convention":'E_10-E_01',
            "diagonal_detuning_mhz": float((effective[0, 0] - effective[1, 1]) * 1000),
            "minimum_subspace_overlap": overlap,
            "valid": bool(overlap >= minimum_overlap),
            "effective_hamiltonian_ghz": effective.tolist(),
            "selected_eigenstate_indices": columns.tolist()}


def scan_coupler(model, start, stop, points, minimum_overlap=0.8):
    start, stop = positive(start, "scan start"), positive(stop, "scan stop")
    if stop <= start or not isinstance(points, int) or points < 2:
        raise ValueError("Scan requires increasing EJ bounds and at least two points")
    if not 0 < minimum_overlap <= 1:
        raise ValueError("minimum_overlap must be in (0,1]")
    from .off_search import exchange_roots
    grid=np.linspace(start,stop,points)
    cache={}
    def at(x):
        if x not in cache:cache[x]=port_exchange(model,float(x),minimum_overlap)
        return cache[x]
    def objective(x):
        row=at(float(x))
        if not row['valid']:raise ValueError('Unreliable two-state subspace in physical exchange scan')
        return row['exchange_mhz']
    rows=[at(float(x)) for x in grid]
    roots=[at(x) for x in exchange_roots(objective,grid)]
    return {'node_order':list(model.names),'samples':rows,'physical_port_roots':roots,
            'off_definition':rows[0]['off_definition'],'residual_tolerance_mhz':1e-6}
