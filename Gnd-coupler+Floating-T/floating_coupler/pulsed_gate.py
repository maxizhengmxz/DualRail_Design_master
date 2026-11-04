"""Five-transmon idle -> EJ-tanh -> hold -> EJ-tanh -> idle CZ.

All propagation uses ONE fixed local charge-eigenfunction basis. Changing EJ
changes -EJ*cos(phi), including off-diagonal terms in that basis. No moving-basis
connection is omitted. EC, charge couplings and the stationary A rail stay fixed.
Total parity is exact at ng=0; the additional product-excitation cutoff is a
numerical truncation, NOT an excitation-conserving/RWA Hamiltonian. Audit it.
"""

from itertools import combinations, product
from time import perf_counter

import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.optimize import brentq, linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from .design import ej_for_frequency
from .gate import gate_metrics
from .model import FULL
from .precision import GATE_PHASE_TOL


def edge_parameters(rise_ns=10.0, truncation=4.0):
    """Exactly specified 10-90% width for a finite, endpoint-normalized tanh.

    s=(tanh((t-T/2)/tau)+tanh(K))/(2*tanh(K)), T=2*K*tau.
    The tails are INCLUDED in wall time, not counted as plateau time.
    """
    if rise_ns <= 0 or truncation <= 0:
        raise ValueError("Positive rise time and tanh truncation required")
    tau = rise_ns / (2 * np.arctanh(0.8 * np.tanh(truncation)))
    return float(tau), float(2 * truncation * tau)


def edge_envelope(t, rise_ns=10.0, truncation=4.0):
    tau, duration = edge_parameters(rise_ns, truncation)
    x = np.clip(np.asarray(t), 0, duration)
    return (np.tanh((x - duration / 2) / tau) + np.tanh(truncation)) / (
        2 * np.tanh(truncation)
    )


def propagate_edge(h0, delta, vectors, rise_ns=10.0, step_ns=0.1, truncation=4.0):
    """Unitary fourth-order commutator-free Magnus integration of one edge.

    The two exponentials use Gauss nodes, keeping the affine H(s) sparse.
    scipy expm_multiply applies each exponential without forming a dense U.
    """
    _, duration = edge_parameters(rise_ns, truncation)
    return propagate_schedule(
        h0,
        delta,
        vectors,
        lambda t: edge_envelope(t, rise_ns, truncation),
        duration,
        step_ns,
    )


def propagate_schedule(h0, delta, vectors, envelope, duration, step_ns=0.1):
    """Propagate an arbitrary scalar EJ envelope in one fixed Hamiltonian frame."""
    if duration <= 0 or step_ns <= 0:
        raise ValueError("Positive duration and time step required")
    steps = int(np.ceil(duration / step_ns))
    dt = duration / steps
    h0, delta = csr_matrix(h0), csr_matrix(delta)
    # A time-dependent scalar identity only changes a global phase.
    identity = csr_matrix(np.eye(h0.shape[0]))
    h0 = h0 - identity * (h0.diagonal().sum() / h0.shape[0])
    delta = delta - identity * (delta.diagonal().sum() / delta.shape[0])
    a1, a2 = (3 - 2 * np.sqrt(3)) / 12, (3 + 2 * np.sqrt(3)) / 12
    result = np.asarray(vectors, dtype=complex).copy()
    for k in range(steps):
        middle = (k + 0.5) * dt
        s1, s2 = envelope(
            np.array([middle - dt * np.sqrt(3) / 6, middle + dt * np.sqrt(3) / 6])
        )
        # First exponential acts first (rightmost factor in the propagator).
        for weight in (a2 * s1 + a1 * s2, a1 * s1 + a2 * s2):
            generator = (-2j * np.pi * dt) * (0.5 * h0 + weight * delta)
            result = expm_multiply(generator, result, traceA=0.0)
    return result, dict(edge_duration_ns=duration, steps=steps, step_ns=dt)


def symmetric_pulse_matrix(energy, vectors, ramped_inputs, hold_ns):
    """Return M for a real symmetric H and an exactly time-reversed falling edge.

    U_down=U_up.T, NOT U_up.conj().T. Thus only four forward edge states
    are needed: M=C.T exp(-2*pi*i*E*t) C. This retains coherent edge errors.
    """
    coefficients = vectors.T @ ramped_inputs
    return coefficients.T @ (
        np.exp(-2j * np.pi * energy * hold_ns)[:, None] * coefficients
    )


class PulseModel:
    """Fixed full-cosine basis, even total parity, explicit cutoff convergence."""

    def __init__(
        self,
        model,
        off_ghz,
        gate_fb,
        gate_fc,
        *,
        local_levels=6,
        coupler_levels=5,
        excitation_cutoff=6,
        ncut=35,
    ):
        if model.names != FULL:
            raise ValueError("Require physical order A1,A2,C,B1,B2")
        self.model = model
        self.ncut = ncut
        self.idle_ej = model.ej.copy()
        self.idle_ej[2] = ej_for_frequency(off_ghz, model.ec[2], ncut)
        self.levels = [
            local_levels,
            local_levels,
            coupler_levels,
            local_levels,
            local_levels,
        ]
        gate_ej = self.controls(gate_fb, gate_fc)
        reference_ej = (self.idle_ej + gate_ej) / 2
        self.reference_ej = reference_ej
        self.local = []
        for ec, ej, levels in zip(model.ec, reference_ej, self.levels, strict=True):
            n = np.arange(-ncut, ncut + 1, dtype=float)
            energy, v = eigh_tridiagonal(
                4 * ec * n**2,
                np.full(2 * ncut, -ej / 2),
                select="i",
                select_range=(0, levels - 1),
            )
            charge = v.T @ (n[:, None] * v)
            signs = np.ones(levels)
            for k in range(1, levels):
                signs[k] = signs[k - 1] * (1 if charge[k - 1, k] >= 0 else -1)
            v *= signs
            charge = v.T @ (n[:, None] * v)
            cos = (v[:-1].T @ v[1:] + v[1:].T @ v[:-1]) / 2
            parity = np.arange(levels) % 2
            charge[parity[:, None] == parity[None, :]] = 0
            cos[parity[:, None] != parity[None, :]] = 0
            self.local.append(dict(energy=energy - energy[0], charge=charge, cos=cos))
        self.states = np.array(
            [
                s
                for s in product(*map(range, self.levels))
                if sum(s) % 2 == 0 and sum(s) <= excitation_cutoff
            ]
        )
        self.basis = dict(
            local_levels=local_levels,
            coupler_levels=coupler_levels,
            excitation_cutoff=excitation_cutoff,
            ncut=ncut,
            dimension=len(self.states),
            parity="even",
            frame="fixed_endpoint_midpoint_EJ",
        )
        self.cos = [self.embed({i: local["cos"]}) for i, local in enumerate(self.local)]
        diagonal = sum(
            local["energy"][self.states[:, i]] for i, local in enumerate(self.local)
        )
        ham = np.diag(diagonal)
        for i, j in combinations(range(5), 2):
            ham += model.j[i, j] * self.embed(
                {i: self.local[i]["charge"], j: self.local[j]["charge"]}
            )
        self.h_ref = ham
        self.h_idle = self.hamiltonian(self.idle_ej)
        self.idle_energy, idle_vectors = eigh(self.h_idle)
        nominal = self.logical_products()
        self.product_norms = np.sum(nominal**2, axis=0)
        nominal /= np.sqrt(self.product_norms)
        overlaps = nominal.T @ idle_vectors
        _, columns = linear_sum_assignment(-(abs(overlaps) ** 2))
        self.inputs = idle_vectors[:, columns].copy()
        for k, column in enumerate(columns):
            self.inputs[:, k] *= 1 if overlaps[k, column] >= 0 else -1
        self.input_weights = abs(overlaps[np.arange(4), columns]) ** 2
        self.idle_logical_energies = self.idle_energy[columns]
        self.idle_indices = columns
        if min(self.input_weights) < 0.9:
            raise ValueError("Idle logical-state assignment is ambiguous")

    def embed(self, operators):
        dim = len(self.states)
        value = np.ones((dim, dim))
        for i in range(5):
            left, right = self.states[:, i, None], self.states[None, :, i]
            value *= operators[i][left, right] if i in operators else left == right
        return value

    def controls(self, fb, fc):
        ej = self.idle_ej.copy()
        for i, frequency in ((2, fc), (3, fb), (4, fb)):
            ej[i] = ej_for_frequency(float(frequency), self.model.ec[i], self.ncut)
        return ej

    def hamiltonian(self, ej):
        return self.h_ref - sum(
            (ej[i] - self.reference_ej[i]) * self.cos[i] for i in range(5)
        )

    def logical_products(self):
        pairs = []
        for i, j in ((0, 1), (3, 4)):
            li, lj = self.levels[i], self.levels[j]
            local_h = [
                np.diag(self.local[k]["energy"])
                - (self.idle_ej[k] - self.reference_ej[k]) * self.local[k]["cos"]
                for k in (i, j)
            ]
            pair_h = np.kron(local_h[0], np.eye(lj)) + np.kron(np.eye(li), local_h[1])
            pair_h += self.model.j[i, j] * np.kron(
                self.local[i]["charge"], self.local[j]["charge"]
            )
            energy, v = eigh(pair_h)
            one = [lj, 1]
            columns = np.argsort(np.sum(v[one] ** 2, axis=0))[-2:]
            columns = sorted(columns, key=lambda k: energy[k])
            pairs.append(v[:, columns].reshape(li, lj, 2))
        hc = (
            np.diag(self.local[2]["energy"])
            - (self.idle_ej[2] - self.reference_ej[2]) * self.local[2]["cos"]
        )
        _, vc = eigh(hc)
        result = []
        a1, a2, c, b1, b2 = self.states.T
        for a, b in product(range(2), repeat=2):
            result.append(pairs[0][a1, a2, a] * vc[c, 0] * pairs[1][b1, b2, b])
        return np.array(result).T

    def spectrum(self, fb, fc, rise_ns=10.0, step_ns=0.1, truncation=4.0, shape="tanh"):
        started = perf_counter()
        ej = self.controls(fb, fc)
        ham = self.hamiltonian(ej)
        energy, vectors = eigh(ham)
        if shape == "tanh":
            ramped, edge = propagate_edge(
                self.h_idle,
                ham - self.h_idle,
                self.inputs,
                rise_ns,
                step_ns,
                truncation,
            )
        elif shape == "cosine":
            duration = rise_ns / (1 - 2 * np.arccos(0.8) / np.pi)
            ramped, edge = propagate_schedule(
                self.h_idle,
                ham - self.h_idle,
                self.inputs,
                lambda t: 0.5 * (1 - np.cos(np.pi * np.clip(t / duration, 0, 1))),
                duration,
                step_ns,
            )
        else:
            raise ValueError("Unknown pulse shape")
        coefficients = vectors.T @ ramped
        return dict(
            energy=energy - energy[0],
            vectors=vectors,
            coefficients=coefficients,
            ramped=ramped,
            fb_ghz=float(fb),
            fc_ghz=float(fc),
            gate_ej=ej,
            edge=edge,
            seconds=perf_counter() - started,
            edge_norm_error=float(np.max(abs(ramped.conj().T @ ramped - np.eye(4)))),
        )


def metrics_at(spectrum, hold_ns):
    c = spectrum["coefficients"]
    matrix = c.T @ (np.exp(-2j * np.pi * spectrum["energy"] * hold_ns)[:, None] * c)
    result = gate_metrics(matrix)
    result.update(
        hold_ns=float(hold_ns),
        total_ns=float(hold_ns + 2 * spectrum["edge"]["edge_duration_ns"]),
        fb_ghz=spectrum["fb_ghz"],
        fc_ghz=spectrum["fc_ghz"],
        worst_nonlogical_population=max(result["leakage_by_input"]),
        worst_input_survival=1 - max(result["leakage_by_input"]),
    )
    return result


def phase_pi_candidates(spectrum, time_bounds=(40.0, 170.0), points=521):
    """Reuse one captured ramp and hold spectrum for the duration scan/roots."""
    times = np.linspace(*time_bounds, points)
    rows = [metrics_at(spectrum, time) for time in times]
    roots = []
    for left, right in zip(rows[:-1], rows[1:]):
        a, b = left["phase_error_rad"], right["phase_error_rad"]
        if a * b < 0 and abs(a - b) < np.pi:
            time = brentq(
                lambda t: metrics_at(spectrum, t)["phase_error_rad"],
                left["hold_ns"],
                right["hold_ns"],
                xtol=1e-7,
            )
            row = metrics_at(spectrum, time)
            if abs(row["phase_error_rad"]) <= GATE_PHASE_TOL:
                roots.append(row)
    return roots
