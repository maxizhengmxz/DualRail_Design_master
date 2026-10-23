"""Reusable grounded five-transmon + two-resonator Hamiltonian.

No file reads, optimization, or saved-case defaults live here. Node order is
A1,A2,rA,C,B1,B2,rB. Energies are H/h in GHz, capacitances fF. Use
``python "coupler turn-off/run.py" --help`` for the staged workflow.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import elementary_charge as e
from scipy.constants import h, hbar
from scipy.linalg import eigh
from scipy.optimize import brentq, linear_sum_assignment, root

from core.transmon import Basis, transmon_spectrum
from core.electrostatics import charge_coupling_ghz, exchange_j01_ghz, loaded_charging_energies_ghz


def embed(operator, index, dims):
    result = np.ones((1, 1))
    for i, dimension in enumerate(dims):
        result = np.kron(result, operator if i == index else np.eye(dimension))
    return result


def restricted_hamiltonian(energies, blocks, dims, coefficients, indices):
    """Exact selected submatrix of the Kronecker Hamiltonian, without building it all."""
    coordinates = np.asarray(np.unravel_index(indices, dims))
    ham = np.diag(energies[indices])
    for left in range(3):
        for right in range(left + 1, 3):
            spectator = 3 - left - right
            mask = coordinates[spectator, :, None] == coordinates[spectator, None, :]
            li, ri = coordinates[left], coordinates[right]
            for ni, oi in zip(*blocks[left], strict=True):
                a = oi[li[:, None], li[None, :]]
                for nj, oj in zip(*blocks[right], strict=True):
                    ham += (
                        coefficients[ni, nj] * a * oj[ri[:, None], ri[None, :]] * mask
                    )
    return ham


def capacitance_matrix(point, link_ff, coupler_ec_ghz, direct_ff=0.0):
    """Nodes A1,A2,RA,C,B1,B2,RB; ECc sets the loaded coupler EC exactly.

    First remove links from connected shunts, then compensate inverse-matrix
    loading to preserve all four validated rail ECs. Set the coupler diagonal
    by the Schur complement, so ECc includes both links and their loading.
    """
    d = point["device"]
    link_a, link_b = (link_ff, link_ff) if np.isscalar(link_ff) else link_ff
    if not (np.isfinite(direct_ff) and direct_ff >= 0):
        raise ValueError("Direct capacitance must be finite and nonnegative")
    if not (
        0 <= link_a + direct_ff < d["C2_shunt_fF"]
        and 0 <= link_b + direct_ff < d["C1_shunt_fF"]
    ):
        raise ValueError("Link exceeds available shunt budget")
    matrix = np.zeros((7, 7))
    shunts = [
        d["C1_shunt_fF"],
        d["C2_shunt_fF"] - link_a - direct_ff,
        d["Cr_fF"],
        0,
        d["C1_shunt_fF"] - link_b - direct_ff,
        d["C2_shunt_fF"],
        d["Cr_fF"],
    ]
    matrix += np.diag(shunts)
    links = [
        (0, 1, d["C12_fF"]),
        (0, 2, d["C1r_fF"]),
        (1, 2, d["C2r_fF"]),
        (4, 5, d["C12_fF"]),
        (4, 6, d["C1r_fF"]),
        (5, 6, d["C2r_fF"]),
        (1, 3, link_a),
        (3, 4, link_b),
        (1, 4, direct_ff),
    ]
    for i, j, value in links:
        matrix[i, i] += value
        matrix[j, j] += value
        matrix[i, j] -= value
        matrix[j, i] -= value
    others = [0, 1, 2, 4, 5, 6]
    v = matrix[3, others]
    desired_effective_ff = e**2 / (2 * h * coupler_ec_ghz * 1e9) * 1e15
    rail_nodes = [0, 1, 4, 5]
    target_ec = np.array([point["loaded_ec1_mhz"], point["loaded_ec2_mhz"]] * 2) / 1000

    def compensated(delta):
        candidate = matrix.copy()
        candidate[rail_nodes, rail_nodes] += delta
        candidate[3, 3] = desired_effective_ff + v @ np.linalg.solve(
            candidate[np.ix_(others, others)], v
        )
        return candidate

    def residual(delta):
        inverse = np.linalg.inv(compensated(delta))
        ecs = e**2 * np.diag(inverse)[rail_nodes] * 1e15 / (2 * h * 1e9)
        return (ecs - target_ec) / target_ec

    fit = root(residual, np.zeros(4), tol=1e-10)
    if np.max(np.abs(residual(fit.x))) > 1e-10:
        raise ValueError("Could not preserve rail EC budget")
    matrix = compensated(fit.x)
    diagonal = matrix[3, 3]
    coupler_shunt_ff = diagonal - link_a - link_b
    if coupler_shunt_ff <= 0:
        raise ValueError("Coupler has no positive shunt capacitance budget")
    matrix[3, 3] = diagonal
    if np.any(matrix.sum(axis=1) <= 0):
        raise ValueError("Capacitive loading exhausted a physical ground-shunt budget")
    if np.linalg.eigvalsh(matrix).min() <= 0:
        raise ValueError("Non-positive capacitance matrix")
    return matrix, coupler_shunt_ff


class BudgetModel:
    def __init__(
        self,
        point,
        link_ff,
        ec_c,
        levels=5,
        photons=4,
        pair_keep=10,
        c_levels=4,
        equal_g=True,
        direct_ff=0.0,
        fixed_links_ff=None,
        rail_ej_ghz=None,
        eigenenergy_max_ghz=None,
        resonator_inductances_nh=None,
        basis_energy_max_ghz=None,
        readout_window_ghz=(4.5, 7.0),
    ):
        self.readout_window_ghz = tuple(readout_window_ghz)
        self.point, self.link_ff, self.ec_c = point, link_ff, ec_c
        self.resonator_inductances_nh = (
            tuple(resonator_inductances_nh)
            if resonator_inductances_nh is not None
            else (point["device"]["Lr_nH"],) * 2
        )
        self.direct_ff = direct_ff
        self.rail_ej_ghz = (
            tuple(rail_ej_ghz)
            if rail_ej_ghz is not None
            else (
                point["ej1_ghz"],
                point["ej2_ghz"],
                point["ej1_ghz"],
                point["ej2_ghz"],
            )
        )
        if len(self.rail_ej_ghz) != 4 or min(self.rail_ej_ghz) <= 0:
            raise ValueError("Provide four positive rail EJs")
        self.eigenenergy_max_ghz = eigenenergy_max_ghz
        self.basis_energy_max_ghz = basis_energy_max_ghz
        if basis_energy_max_ghz is not None and eigenenergy_max_ghz is None:
            raise ValueError("An energy-pruned basis requires a spectral ceiling")
        self.levels, self.photons = levels, photons
        self.pair_keep, self.c_levels = pair_keep, c_levels
        local_basis = Basis(25, max(levels, 3), max(photons, 3), 20)
        n1 = abs(
            transmon_spectrum(point["ej1_ghz"], point["loaded_ec1_mhz"] / 1000, local_basis)[1][
                0, 1
            ]
        )
        n2 = abs(
            transmon_spectrum(point["ej2_ghz"], point["loaded_ec2_mhz"] / 1000, local_basis)[1][
                0, 1
            ]
        )
        link_b = link_ff
        if fixed_links_ff is not None:
            link_ff, link_b = map(float, fixed_links_ff)
            self.link_ff = link_ff
        elif equal_g and link_ff > 0:

            def imbalance(candidate):
                cap, _ = capacitance_matrix(
                    point, (link_ff, candidate), ec_c, direct_ff
                )
                inverse = np.linalg.inv(cap)
                return inverse[1, 3] * n2 - inverse[3, 4] * n1

            link_b = brentq(imbalance, 0.5 * link_ff, 2 * link_ff, xtol=1e-12)
        self.links = [link_ff, link_b]
        self.cap, self.c_shunt = capacitance_matrix(point, self.links, ec_c, direct_ff)
        self.rail_shunts = (self.cap.sum(axis=1)[[0, 1, 4, 5]]).tolist()
        self.cinv = np.linalg.inv(self.cap * 1e-15)
        self.ec = loaded_charging_energies_ghz(self.cinv)
        self.J_ghz = charge_coupling_ghz(self.cinv)
        self.k = self.J_ghz  # Compatibility alias; off-diagonals are charge coefficients.
        self.basis = Basis(25, levels, photons, 20)
        self.pair_parities = []
        self.pairs = [self.pair((0, 1, 2)), self.pair((4, 5, 6))]
        self.dims = (pair_keep, c_levels, pair_keep)
        self.refs = [
            np.ravel_multi_index((a, 0, b), self.dims)
            for a, b in ((1, 1), (1, 2), (2, 1), (2, 2))
        ]

    def pair(self, nodes):
        dims = (self.levels, self.levels, self.photons)
        energies, operators = [], []
        for i, node in enumerate(nodes[:2]):
            en, charge = transmon_spectrum(
                self.rail_ej_ghz[[0, 1, 4, 5].index(node)], self.ec[node], self.basis
            )
            energies.append(en)
            operators.append(embed(charge, i, dims))
        node = nodes[2]
        cr_eff = 1 / self.cinv[node, node]
        inductance = self.resonator_inductances_nh[0 if node == 2 else 1]
        omega = 1 / np.sqrt(inductance * 1e-9 * cr_eff)
        energies.append(np.arange(self.photons) * omega / (2 * np.pi * 1e9))
        a = np.diag(np.sqrt(np.arange(1, self.photons)), 1)
        # Oscillator phase rotation makes charge real without changing physics.
        qr = np.sqrt(hbar * cr_eff * omega / 2) * (a + a.T) / (2 * e)
        operators.append(embed(qr, 2, dims))
        ham = sum(embed(np.diag(en), i, dims) for i, en in enumerate(energies))
        for i in range(3):
            for j in range(i + 1, 3):
                ham += self.J_ghz[nodes[i], nodes[j]] * (operators[i] @ operators[j])
        en, states = eigh(ham, subset_by_index=(0, self.pair_keep - 1))
        en -= en[0]
        parity = np.kron(
            np.kron((-1.0) ** np.arange(dims[0]), (-1.0) ** np.arange(dims[1])),
            (-1.0) ** np.arange(dims[2]),
        )
        self.pair_parities.append(np.sign(np.sum(states**2 * parity[:, None], axis=0)))
        return en, [states.T @ op @ states for op in operators]

    def evaluate(self, ej_c, readout=False):
        ec, nc = transmon_spectrum(ej_c, self.ec_c, Basis(25, self.c_levels, 3, 6))
        ea, ops_a = self.pairs[0]
        eb, ops_b = self.pairs[1]
        energies = (ea[:, None, None] + ec[None, :, None] + eb[None, None, :]).ravel()
        # All cross-block interactions from the same C^-1, including non-neighbors.
        blocks = [([0, 1, 2], ops_a), ([3], [nc]), ([4, 5, 6], ops_b)]
        ham = None
        if self.eigenenergy_max_ghz is None:
            ham = restricted_hamiltonian(
                energies, blocks, self.dims, self.J_ghz, np.arange(len(energies))
            )
            en, states = eigh(ham)
        else:
            # Exact total-parity blocks reduce the cost of the low-spectrum solve.
            parity = (
                self.pair_parities[0][:, None, None]
                * ((-1.0) ** np.arange(self.c_levels))[None, :, None]
                * self.pair_parities[1][None, None, :]
            ).ravel()
            values, vectors = [], []
            for sign in (-1, 1):
                selected = parity == sign
                if self.basis_energy_max_ghz is not None:
                    selected &= energies <= self.basis_energy_max_ghz
                indices = np.flatnonzero(selected)
                block_ham = restricted_hamiltonian(
                    energies, blocks, self.dims, self.J_ghz, indices
                )
                eigenvalues, eigenvectors = eigh(
                    block_ham,
                    subset_by_value=(-np.inf, self.eigenenergy_max_ghz),
                )
                embedded = np.zeros((len(energies), len(eigenvalues)))
                embedded[indices, :] = eigenvectors
                values.append(eigenvalues)
                vectors.append(embedded)
            en = np.concatenate(values)
            states = np.column_stack(vectors)
            order = np.argsort(en)
            en, states = en[order], states[:, order]
        overlap = states[self.refs, :]
        single_refs = [
            np.ravel_multi_index(index, self.dims)
            for index in ((1, 0, 0), (2, 0, 0), (0, 0, 1), (0, 0, 2))
        ]
        single_overlap = states[single_refs, :]
        _, single_labels = linear_sum_assignment(-(single_overlap**2))
        su, ss, sv = np.linalg.svd(single_overlap[:, single_labels])
        single_frame = su @ sv
        single_effective = (single_frame * en[single_labels]) @ single_frame.T
        single_rotation = np.zeros((4, 4))
        for start in (0, 2):
            _, rotation = eigh(single_effective[start : start + 2, start : start + 2])
            single_rotation[start : start + 2, start : start + 2] = rotation
        single_effective = single_rotation.T @ single_effective @ single_rotation
        # Charge-transition balance of A's local energy modes in the dressed
        # one-excitation effective Hamiltonian (B ground reference).
        single_dressed = states[:, single_labels] @ single_frame.T
        columns_a = single_dressed[:, :2].reshape(*self.dims, 2)
        strengths_a = []
        for op in ops_a[:2]:
            acted = np.einsum("ij,jklm->iklm", op, columns_a, optimize=True)
            amplitudes = states[:, 0] @ acted.reshape(-1, 2) @ single_rotation[:2, :2]
            strengths_a.append(np.abs(amplitudes) ** 2)
        strengths_a = np.asarray(strengths_a)
        balance_a = strengths_a / strengths_a.sum(axis=1, keepdims=True)
        single_a_hybridization = 1 - 2 * np.mean(np.abs(balance_a - 0.5))
        _, assigned = linear_sum_assignment(-(overlap**2))
        frame = overlap[:, assigned]
        u, singular, vh = np.linalg.svd(frame)
        unitary = u @ vh
        effective = (unitary * (en[assigned] - en[assigned[0]])) @ unitary.T
        exchange = 1000 * abs(effective[1, 2])
        zz = 1000 * (
            effective[3, 3] - effective[2, 2] - effective[1, 1] + effective[0, 0]
        )
        leakage = 1000 * np.min(
            np.abs(en[assigned, None] - np.delete(en, assigned)[None, :])
        )
        c_occupations = np.unravel_index(np.arange(len(energies)), self.dims)[1]
        c_excitation = np.sum(states[:, assigned] ** 2 * c_occupations[:, None], axis=0)
        paulis = [
            np.eye(2),
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.diag([1, -1]),
        ]
        terms = {
            a + b: float(np.trace(np.kron(paulis[i], paulis[j]) @ effective).real * 250)
            for i, a in enumerate("IXYZ")
            for j, b in enumerate("IXYZ")
        }
        nonlocal_norm = float(
            np.linalg.norm([v for key, v in terms.items() if "I" not in key])
        )
        result = dict(
            basis_energy_max_ghz=self.basis_energy_max_ghz,
            retained_product_states=int(
                np.count_nonzero(energies <= self.basis_energy_max_ghz)
            )
            if self.basis_energy_max_ghz is not None
            else len(energies),
            single_excitation_exchange_matrix_mhz=(
                1000 * single_effective[:2, 2:]
            ).tolist(),
            single_excitation_effective_matrix_mhz=(1000 * single_effective).tolist(),
            single_excitation_subspace_sigma_min=float(ss.min()),
            single_excitation_a_hybridization=float(single_a_hybridization),
            single_excitation_a_charge_balance=balance_a.tolist(),
            dressed_coupler_transition_ghz=float(
                en[
                    np.argmax(
                        states[np.ravel_multi_index((0, 1, 0), self.dims), :] ** 2
                    )
                ]
                - en[0]
            ),
            link_ff=self.link_ff,
            rail_ej_ghz=list(self.rail_ej_ghz),
            direct_a2_b1_ff=self.direct_ff,
            coupler_ec_ghz=self.ec_c,
            link_a2_c_ff=self.links[0],
            link_c_b1_ff=self.links[1],
            coupler_ej_ghz=float(ej_c),
            coupler_f01_ghz=float(ec[1]),
            exchange_mhz=float(exchange),
            signed_logical_exchange_mhz=float(1000 * effective[1, 2]),
            zz_mhz=float(zz),
            residual_mhz=float(np.hypot(exchange, zz)),
            sigma_min=float(singular.min()),
            assignment_min=float(np.min(np.diag(frame) ** 2)),
            leakage_gap_mhz=float(leakage),
            maximum_coupler_excitation=float(c_excitation.max()),
            nonlocal_pauli_norm_mhz=nonlocal_norm,
            pauli_mhz=terms,
            pair_splittings_mhz=[float(1000 * (x[0][2] - x[0][1])) for x in self.pairs],
        )
        if getattr(self, "diagnose_logical_mixing", False):
            result["logical_mixing"] = []
            for label, column in enumerate(assigned):
                top = np.argsort(states[:, column] ** 2)[-5:][::-1]
                result["logical_mixing"].append(
                    dict(
                        label=int(label),
                        energy_ghz=float(en[column] - en[0]),
                        components=[
                            dict(
                                pair_a=int(np.unravel_index(int(q), self.dims)[0]),
                                coupler=int(np.unravel_index(int(q), self.dims)[1]),
                                pair_b=int(np.unravel_index(int(q), self.dims)[2]),
                                weight=float(states[q, column] ** 2),
                            )
                            for q in top
                        ],
                    )
                )
            result["isolated_pair_energies_ghz"] = [ea.tolist(), eb.tolist()]
        n1 = abs(transmon_spectrum(self.rail_ej_ghz[2], self.ec[4], self.basis)[1][0, 1])
        n2 = abs(transmon_spectrum(self.rail_ej_ghz[1], self.ec[1], self.basis)[1][0, 1])
        for name, i, j, ni, nj in (
            ("a2_c", 1, 3, n2, nc[0, 1]),
            ("c_b1", 3, 4, nc[0, 1], n1),
            ("a2_b1", 1, 4, n2, n1),
        ):
            result[f"J_{name}_ghz"] = float(self.J_ghz[i, j])
            result[f"J01_{name}_mhz"] = 1000 * exchange_j01_ghz(self.J_ghz[i, j], ni, nj)
        result["charge_coupling_J_ghz"] = self.J_ghz.tolist()
        result["charge_coupling_node_order"] = ["A1", "A2", "rA", "C", "B1", "B2", "rB"]
        result["g_a2_c_mhz"] = result["J01_a2_c_mhz"]  # Legacy cyclic-rate alias.
        result["g_c_b1_mhz"] = result["J01_c_b1_mhz"]
        if readout:
            photon_modes = []
            for energy, ops in self.pairs:
                targets = []
                for mode in range(3):
                    mask = (energy - energy[mode] > self.readout_window_ghz[0]) & (
                        energy - energy[mode] < self.readout_window_ghz[1]
                    )
                    strengths = np.where(mask, np.abs(ops[2][mode]) ** 2, 0)
                    if strengths.max() == 0:
                        raise ValueError("Pair basis has no readout photon branch")
                    targets.append(int(np.argmax(strengths)))
                photon_modes.append(targets)
            indices, lookup = set(), {}
            for side in range(2):
                for spectator in (1, 2):
                    for mode in range(3):
                        for photon in range(2):
                            a = photon_modes[side][mode] if photon else mode
                            coordinates = (
                                (a, 0, spectator) if side == 0 else (spectator, 0, a)
                            )
                            index = int(np.ravel_multi_index(coordinates, self.dims))
                            lookup[side, spectator, mode, photon] = index
                            indices.add(index)
            indices = sorted(indices)
            _, labels = linear_sum_assignment(-(states[indices, :] ** 2))
            overlap_readout = states[np.ix_(indices, labels)]
            ur, sr, vr = np.linalg.svd(overlap_readout)
            local_frame = ur @ vr
            readout_effective = (local_frame * en[labels]) @ local_frame.T
            # Archive the projected frame so photon-changing matrix elements can
            # be audited without confusing them with eigenenergy splittings.
            result["readout_frame_matrix_ghz"] = readout_effective.tolist()
            result["readout_frame_product_indices"] = [
                list(map(int, np.unravel_index(index, self.dims))) for index in indices
            ]
            result["readout_frame_lookup"] = [
                dict(
                    side=side,
                    spectator=spectator,
                    mode=mode,
                    photon=photon,
                    row=indices.index(index),
                )
                for (side, spectator, mode, photon), index in lookup.items()
            ]
            # Identical resonators hybridize: assigned eigenenergies alone do
            # not describe a local readout. Report local-frame diagonal shifts.
            dressed = dict(zip(indices, np.diag(readout_effective), strict=True))
            checks = []
            for side in range(2):
                for spectator in (1, 2):
                    frequencies = [
                        dressed[lookup[side, spectator, mode, 1]]
                        - dressed[lookup[side, spectator, mode, 0]]
                        for mode in range(3)
                    ]
                    chis = 1000 * (np.array(frequencies[1:]) - frequencies[0])
                    checks.append(
                        dict(
                            resonator="A" if side == 0 else "B",
                            spectator_logical=spectator - 1,
                            chi_minus_mhz=float(chis[0]),
                            chi_plus_mhz=float(chis[1]),
                            mean_chi_mhz=float(chis.mean()),
                            mismatch=float(abs((chis[1] - chis[0]) / chis.mean())),
                        )
                    )
            result["readout_checks"] = checks
            energy_checks = []
            for side in range(2):
                for spectator in (1, 2):
                    local_indices = [
                        indices.index(lookup[side, spectator, mode, 0])
                        for mode in (1, 2)
                    ]
                    photon_indices = [
                        indices.index(lookup[side, spectator, mode, 1])
                        for mode in (1, 2)
                    ]
                    local_energy, local_modes = np.linalg.eigh(
                        readout_effective[np.ix_(local_indices, local_indices)]
                    )
                    photon_energy = np.linalg.eigvalsh(
                        readout_effective[np.ix_(photon_indices, photon_indices)]
                    )
                    ground = indices.index(lookup[side, spectator, 0, 0])
                    ground_photon = indices.index(lookup[side, spectator, 0, 1])
                    resonator = (
                        readout_effective[ground_photon, ground_photon]
                        - readout_effective[ground, ground]
                    )
                    chi = 1000 * (photon_energy - local_energy - resonator)
                    dressed_basis = states[:, labels] @ local_frame.T
                    block = 0 if side == 0 else 2
                    ops = ops_a if side == 0 else ops_b
                    strengths = []
                    for op in ops[:2]:
                        columns = dressed_basis[:, local_indices].reshape(*self.dims, 2)
                        acted = (
                            np.einsum("ij,jklm->iklm", op, columns, optimize=True)
                            if block == 0
                            else np.einsum("ij,kljm->klim", op, columns, optimize=True)
                        )
                        amplitudes = (
                            dressed_basis[:, ground]
                            @ acted.reshape(-1, 2)
                            @ local_modes
                        )
                        strengths.append(np.abs(amplitudes) ** 2)
                    strengths = np.asarray(strengths)
                    balance = strengths / strengths.sum(axis=1, keepdims=True)
                    hybridization = 1 - 2 * np.mean(np.abs(balance - 0.5))
                    energy_checks.append(
                        dict(
                            resonator="A" if side == 0 else "B",
                            spectator_logical=spectator - 1,
                            chi_minus_mhz=float(chi[0]),
                            chi_plus_mhz=float(chi[1]),
                            mean_chi_mhz=float(chi.mean()),
                            mismatch=float(abs((chi[1] - chi[0]) / chi.mean())),
                            hybridization=float(hybridization),
                            logical_frequency_mhz=float(
                                1000 * (local_energy[1] - local_energy[0])
                            ),
                            operation_frequency_ghz=float(
                                local_energy.mean() - readout_effective[ground, ground]
                            ),
                            resonator_frequency_ghz=float(resonator),
                        )
                    )
            result["energy_mode_readout_checks"] = energy_checks
            result["readout_assignment_min"] = float(
                np.min(states[indices, labels] ** 2)
            )
            result["readout_subspace_sigma_min"] = float(sr.min())
            result["readout_interpretation"] = (
                "local-frame diagonal shifts; identical resonator eigenmodes can hybridize"
            )
        return result

    def physical_exchange(self, ej_c, levels=5):
        """Signed boundary-rail exchange after eliminating C in a three-mode diagnostic.

        Uses the full seven-node inverse capacitance matrix, but omits A1/B2
        and readout dynamics. The full logical calculation remains separate.
        Fix local charge signs so the direct exchange is positive by convention.
        """
        dims = (levels,) * 3
        basis = Basis(25, levels, 3, 6)
        energies, charges = [], []
        for node, ej in (
            (1, self.rail_ej_ghz[1]),
            (3, ej_c),
            (4, self.rail_ej_ghz[2]),
        ):
            energy, charge = transmon_spectrum(ej, self.ec[node], basis)
            charge = charge * np.sign(charge[0, 1])
            energies.append(energy)
            charges.append(charge)
        ham = sum(embed(np.diag(energy), i, dims) for i, energy in enumerate(energies))
        nodes = [1, 3, 4]
        for i in range(3):
            for j in range(i + 1, 3):
                factors = [np.eye(levels) for _ in range(3)]
                factors[i], factors[j] = charges[i], charges[j]
                ham += self.J_ghz[nodes[i], nodes[j]] * np.kron(
                    np.kron(factors[0], factors[1]), factors[2]
                )
        en, states = eigh(ham)
        refs = [np.ravel_multi_index(index, dims) for index in ((1, 0, 0), (0, 0, 1))]
        overlap = states[refs, :]
        _, assigned = linear_sum_assignment(-(overlap**2))
        u, s, v = np.linalg.svd(overlap[:, assigned])
        frame = u @ v
        effective = (frame * (en[assigned] - en[assigned[0]])) @ frame.T
        absolute_effective = (frame * en[assigned]) @ frame.T - en[0] * np.eye(2)
        return dict(
            physical_signed_exchange_mhz=float(1000 * effective[0, 1]),
            physical_effective_detuning_mhz=float(
                1000 * (absolute_effective[0, 0] - absolute_effective[1, 1])
            ),
            physical_dressed_rail_frequencies_ghz=np.diag(absolute_effective).tolist(),
            physical_sigma_min=float(s.min()),
            coupler_f01_ghz=float(energies[1][1]),
            rail_f01_ghz=[float(energies[0][1]), float(energies[2][1])],
            physical_diagnostic_levels=levels,
        )
