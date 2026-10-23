"""Multilevel readout dispersive screen including all five transmon dynamics.

Readout oscillators are excluded from the matter Hamiltonian; their charge
couplings to every matter node come from the same full inverse capacitance.
This extends the original isolated-pair |g_ij / (omega_ij - omega_r)| screen.
"""

import numpy as np
from scipy.constants import e, hbar
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from coupler_turn_off.circuit import embed
from core.transmon import Basis, transmon_spectrum


def network_dispersive(model, ej_c, levels=6, pair_keep=10, c_levels=5):
    basis = Basis(25, levels, 3, 10)
    pair_data = []
    for nodes, ejs in (
        ((0, 1), model.rail_ej_ghz[:2]),
        ((4, 5), model.rail_ej_ghz[2:]),
    ):
        en, ops = [], []
        for pos, (node, ej) in enumerate(zip(nodes, ejs, strict=True)):
            energy, charge = transmon_spectrum(ej, model.ec[node], basis)
            en.append(energy)
            ops.append(embed(charge, pos, (levels, levels)))
        ham = np.diag((en[0][:, None] + en[1][None, :]).ravel())
        ham += model.k[nodes[0], nodes[1]] * ops[0] @ ops[1]
        vals, vecs = eigh(ham, subset_by_index=(0, pair_keep - 1))
        pair_data.append((vals - vals[0], [vecs.T @ op @ vecs for op in ops]))
    ec, nc = transmon_spectrum(ej_c, model.ec_c, Basis(25, c_levels, 3, 10))
    ea, oa = pair_data[0]
    eb, ob = pair_data[1]
    dims = (pair_keep, c_levels, pair_keep)
    ham = np.diag((ea[:, None, None] + ec[None, :, None] + eb[None, None, :]).ravel())
    blocks = [([0, 1], oa), ([3], [nc]), ([4, 5], ob)]
    for left in range(3):
        for right in range(left + 1, 3):
            for ni, oi in zip(*blocks[left], strict=True):
                for nj, oj in zip(*blocks[right], strict=True):
                    factors = [np.eye(n) for n in dims]
                    factors[left], factors[right] = oi, oj
                    ham += model.k[ni, nj] * np.kron(
                        np.kron(factors[0], factors[1]), factors[2]
                    )
    vals, vecs = eigh(ham)
    refs = [(a, b) for a in range(3) for b in range(3)]
    indices = [np.ravel_multi_index((a, 0, b), dims) for a, b in refs]
    _, labels = linear_sum_assignment(-(vecs[indices, :] ** 2))
    lookup = dict(zip(refs, labels, strict=True))
    reports = []
    for side, resonator in enumerate((2, 6)):
        cr = 1 / model.cinv[resonator, resonator]
        omega = 1 / np.sqrt(model.resonator_inductances_nh[side] * 1e-9 * cr)
        fr = omega / (2 * np.pi * 1e9)
        qr = np.sqrt(hbar * cr * omega / 2) / (2 * e)
        coupling = np.zeros_like(ham)
        for block, (nodes, ops) in enumerate(blocks):
            local = sum(
                model.k[node, resonator] * qr * op
                for node, op in zip(nodes, ops, strict=True)
            )
            coupling += embed(local, block, dims)
        initial_refs = [(a, b) for a, b in refs if (b >= 1 if side == 0 else a >= 1)]
        initial = [lookup[ref] for ref in initial_refs]
        g = np.abs(vecs[:, initial].T @ coupling @ vecs)
        transition = vals[None, :] - vals[initial, None]
        detuning = np.abs(transition - fr)
        ratios = np.where(
            (transition > 1e-9) & (g > 1e-9), g / np.maximum(detuning, 1e-15), 0
        )
        i, j = np.unravel_index(np.argmax(ratios), ratios.shape)
        reports.append(
            dict(
                resonator="A" if side == 0 else "B",
                dispersive_ratio=float(ratios[i, j]),
                initial_pair_reference=initial_refs[i],
                initial_eigenstate=int(initial[i]),
                final_eigenstate=int(j),
                transition_ghz=float(transition[i, j]),
                detuning_ghz=float(detuning[i, j]),
                coupling_mhz=float(1000 * g[i, j]),
                resonator_frequency_ghz=float(fr),
                minimum_initial_assignment=float(
                    min(
                        vecs[index, label] ** 2
                        for index, label in zip(indices, labels, strict=True)
                    )
                ),
                basis=[levels, pair_keep, c_levels],
                scope="five-transmon matter Hamiltonian; linear readout coupling; original upward-transition dispersive screen",
            )
        )
    return reports
