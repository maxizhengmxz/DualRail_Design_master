"""Full-cosine four-transmon dual-rail interaction simulation.

All energies supplied by the user are in GHz. QuTiP Hamiltonians are built in
rad/ns, so every energy is multiplied by 2*pi. Time is measured in ns.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
import qutip as qt
from core.transmon import projected_transmon_operators
from .channel_policy import LOGICAL_ORDER,one_excitation_indices
from core.electrostatics import (
    charging_energy_to_capacitance_fF,
    capacitance_to_charging_energy_GHz,
    mutual_capacitance_to_charge_coupling_GHz,
    charge_coupling_to_mutual_capacitance_fF,
    mutual_capacitance_to_charge_coupling_from_EC_GHz,
    charge_coupling_to_mutual_capacitance_from_EC_GHz
)


TWOPI = 2.0 * np.pi
LABELS = ("A1", "A2", "B1", "B2")


def first_odd_pi_index(
    phases_rad: np.ndarray,
    tolerance_rad: float = 0.05,
) -> int | None:
    """Closest sampled point in the first nearby odd-pi phase encounter."""
    phases_rad = np.asarray(phases_rad, dtype=float)
    signed_error = np.angle(np.exp(1j * (phases_rad - np.pi)))
    phase_error = np.abs(signed_error)
    matches = np.flatnonzero(
        np.isfinite(phase_error) & (phase_error <= tolerance_rad)
    )
    matches = matches[matches > 0]
    if not matches.size:
        return None
    first = int(matches[0])
    stop = first
    while (
        stop + 1 < phase_error.size
        and phase_error[stop + 1] <= 2.0 * tolerance_rad
    ):
        stop += 1
    for right in range(first + 1, stop + 1):
        left = right - 1
        if signed_error[left] == 0.0:
            return left
        if signed_error[left] * signed_error[right] <= 0.0:
            return left if phase_error[left] <= phase_error[right] else right
    encounter = np.arange(first, stop + 1)
    return int(encounter[np.argmin(phase_error[encounter])])


def odd_pi_index(
    phases_rad: np.ndarray,
    order: int = 1,
    tolerance_rad: float = 0.05,
) -> int | None:
    """Closest sampled point at the requested odd-pi phase crossing."""
    if order < 1:
        raise ValueError("order must be at least 1")
    if order == 1:
        return first_odd_pi_index(phases_rad, tolerance_rad)
    phases = np.unwrap(np.asarray(phases_rad, dtype=float))
    finite = np.flatnonzero(np.isfinite(phases))
    if finite.size < 2:
        return None
    direction = np.sign(phases[finite[-1]] - phases[finite[0]])
    if direction == 0.0:
        return None
    target = direction * (2 * order - 1) * np.pi
    residual = phases - target
    for right in range(max(1, int(finite[0]) + 1), phases.size):
        left = right - 1
        if not np.isfinite(residual[left:right + 1]).all():
            continue
        if residual[left] == 0.0:
            return left
        if residual[left] * residual[right] <= 0.0:
            return left if abs(residual[left]) <= abs(residual[right]) else right
    return None


def end_of_window_slope(
    times_ns: np.ndarray,
    values: np.ndarray,
    window_ns: float = 10.0,
) -> float:
    """Return the linear slope over the final finite part of a time trace."""
    times_ns = np.asarray(times_ns, dtype=float)
    values = np.asarray(values, dtype=float)
    if times_ns.ndim != 1 or values.shape != times_ns.shape:
        raise ValueError("times_ns and values must be one-dimensional arrays of equal length")
    if not np.isfinite(window_ns) or window_ns <= 0.0:
        raise ValueError("window_ns must be positive and finite")
    keep = np.isfinite(values) & (times_ns >= times_ns[-1] - window_ns)
    if np.count_nonzero(keep) < 2:
        return np.nan
    return float(np.polyfit(times_ns[keep], values[keep], 1)[0])


def odd_pi_gate_metrics(
    times_ns: np.ndarray,
    conditional_phase_rad: np.ndarray,
    conditional_fidelity: np.ndarray,
    unconditional_fidelity: np.ndarray,
    acceptance_probability: np.ndarray,
    *,
    order: int = 1,
    tolerance_rad: float = 0.05,
) -> dict:
    """Return gate metrics at a selected odd-pi phase encounter."""
    if not np.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
        raise ValueError("tolerance_rad must be positive and finite")
    arrays = [
        np.asarray(value, dtype=float)
        for value in (
            times_ns,
            conditional_phase_rad,
            conditional_fidelity,
            unconditional_fidelity,
            acceptance_probability,
        )
    ]
    if any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays):
        raise ValueError("odd-pi metric inputs must be equal-length one-dimensional arrays")
    selected = odd_pi_index(arrays[1], order=order, tolerance_rad=tolerance_rad)
    result = {
        "index": selected,
        "time_ns": np.nan,
        "conditional_cz_fidelity": np.nan,
        "unconditional_cz_fidelity": np.nan,
        "acceptance_probability": np.nan,
        "conditional_phase_rad": np.nan,
        "conditional_phase_error_rad": np.nan,
    }
    if selected is None:
        return result
    phase = arrays[1][selected]
    result.update(
        time_ns=float(arrays[0][selected]),
        conditional_cz_fidelity=float(arrays[2][selected]),
        unconditional_cz_fidelity=float(arrays[3][selected]),
        acceptance_probability=float(arrays[4][selected]),
        conditional_phase_rad=float(phase),
        conditional_phase_error_rad=float(
            abs(np.angle(np.exp(1j * (phase - np.pi))))
        ),
    )
    return result


@dataclass(frozen=True)
class TransmonSpec:
    EC_GHz: float
    EJ_GHz: float
    ng: float = 0.0


@dataclass(frozen=True)
class BasisSpec:
    ncut: int = 30
    levels_per_transmon: int = 4


@dataclass(frozen=True)
class EvolutionSpec:
    duration_ns: float = 100.0
    samples: int = 801


@dataclass(frozen=True)
class TransmonNoiseSpec:
    """Phenomenological noise parameters for one transmon, in ns."""

    T1_ns: float | None = None
    Tphi_ns: float | None = None
    thermal_population: float = 0.0

    def __post_init__(self) -> None:
        if self.T1_ns is not None and self.T1_ns <= 0.0:
            raise ValueError("T1_ns must be positive")
        if self.Tphi_ns is not None and self.Tphi_ns <= 0.0:
            raise ValueError("Tphi_ns must be positive")
        if self.thermal_population < 0.0:
            raise ValueError("thermal_population must be non-negative")


@dataclass(frozen=True)
class NoiseModel:
    """Independent local relaxation and dephasing for the four transmons."""

    transmons: dict[str, TransmonNoiseSpec]

    @classmethod
    def from_dict(cls, values: dict) -> "NoiseModel":
        unknown = set(values) - set(LABELS)
        if unknown:
            raise ValueError(f"Unknown transmon labels in noise model: {sorted(unknown)}")
        return cls(
            transmons={
                label: TransmonNoiseSpec(**values.get(label, {})) for label in LABELS
            }
        )


@dataclass
class LocalModel:
    label: str
    spec: TransmonSpec
    n_op: qt.Qobj
    n2_op: qt.Qobj
    cos_phi_op: qt.Qobj
    levels_GHz: np.ndarray
    level_kets: list[qt.Qobj]


@dataclass
class LogicalBasis:
    states: list[qt.Qobj]
    labels: tuple[str, ...]
    projector: qt.Qobj
    ideal_product_states: list[qt.Qobj]
















def build_local_model(label: str, spec: TransmonSpec, basis: BasisSpec) -> LocalModel:
    """Project a full cosine transmon into one fixed reference eigenbasis."""
    matrices = projected_transmon_operators(
        spec.EJ_GHz, spec.EC_GHz, basis.ncut, basis.levels_per_transmon, spec.ng
    )
    n_op, n2_op, cos_phi_op = [qt.Qobj(m) for m in matrices]

    h_local_GHz = 4.0 * spec.EC_GHz * n2_op - spec.EJ_GHz * cos_phi_op
    evals, kets = h_local_GHz.eigenstates()
    evals = np.asarray(evals) - evals[0]

    return LocalModel(
        label=label,
        spec=spec,
        n_op=n_op,
        n2_op=n2_op,
        cos_phi_op=cos_phi_op,
        levels_GHz=evals,
        level_kets=list(kets),
    )


def embed(operator: qt.Qobj, index: int, local_dim: int) -> qt.Qobj:
    operators = [qt.qeye(local_dim) for _ in LABELS]
    operators[index] = operator
    return qt.tensor(operators)


class FourTransmonSystem:
    def __init__(
        self,
        specs: dict[str, TransmonSpec],
        couplings_GHz: dict[str, float],
        basis: BasisSpec,
        noise_model: NoiseModel | None = None,
    ) -> None:
        self.specs = specs
        self.couplings_GHz = couplings_GHz
        self.basis = basis
        self.noise_model = noise_model
        self.local = {
            label: build_local_model(label, specs[label], basis) for label in LABELS
        }
        self.dim = basis.levels_per_transmon

        self.n_full = {
            label: embed(self.local[label].n_op, index, self.dim)
            for index, label in enumerate(LABELS)
        }
        self.n2_full = {
            label: embed(self.local[label].n2_op, index, self.dim)
            for index, label in enumerate(LABELS)
        }
        self.cos_full = {
            label: embed(self.local[label].cos_phi_op, index, self.dim)
            for index, label in enumerate(LABELS)
        }

        self.h_charging = sum(
            TWOPI * 4.0 * specs[label].EC_GHz * self.n2_full[label]
            for label in LABELS
        )
        self.h_coupling = TWOPI * (
            couplings_GHz["A1_A2"] * self.n_full["A1"] * self.n_full["A2"]
            + couplings_GHz["B1_B2"] * self.n_full["B1"] * self.n_full["B2"]
            + couplings_GHz["A2_B1"] * self.n_full["A2"] * self.n_full["B1"]
        )
        self._hamiltonian: qt.Qobj | None = None
        self._spectral_cache: tuple[np.ndarray, np.ndarray] | None = None

    def hamiltonian(self) -> qt.Qobj:
        if self._hamiltonian is None:
            hamiltonian = self.h_charging + self.h_coupling
            for label in LABELS:
                hamiltonian += -TWOPI * self.specs[label].EJ_GHz * self.cos_full[label]
            self._hamiltonian = hamiltonian
        return self._hamiltonian

    def spectral_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """Return and cache eigenvalues/eigenvectors of the static Hamiltonian."""
        if self._spectral_cache is None:
            self._spectral_cache = np.linalg.eigh(self.hamiltonian().full())
        return self._spectral_cache

    def unitary_trajectories(
        self,
        initial_states: list[qt.Qobj],
        times_ns: np.ndarray,
    ) -> list[list[qt.Qobj]]:
        """Evolve several kets together using one cached diagonalization."""
        if not initial_states or any(not state.isket for state in initial_states):
            raise TypeError("initial_states must be a non-empty list of kets")
        eigenvalues, eigenvectors = self.spectral_decomposition()
        initial_matrix = np.column_stack(
            [state.full().ravel() for state in initial_states]
        )
        # Apple Accelerate may emit spurious floating-point warnings for this
        # finite dense product; validate the result explicitly instead.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            coefficients = eigenvectors.conj().T @ initial_matrix
        if not np.isfinite(coefficients).all():
            raise FloatingPointError("Non-finite coefficients in unitary evolution")
        phases = np.exp(-1j * np.outer(np.asarray(times_ns), eigenvalues))
        # Shape: time, physical coordinate, initial-state index.
        evolved = np.einsum(
            "de,te,ei->tdi", eigenvectors, phases, coefficients, optimize=True
        )
        return [
            [
                qt.Qobj(evolved[t, :, i].reshape(-1, 1), dims=state.dims)
                for t in range(len(times_ns))
            ]
            for i, state in enumerate(initial_states)
        ]

    def local_product_ket(self, levels: tuple[int, int, int, int]) -> qt.Qobj:
        return qt.tensor(
            [self.local[label].level_kets[level] for label, level in zip(LABELS, levels)]
        )

    def pair_logical_states(self, pair: str) -> list[qt.Qobj]:
        """Return the two single-excitation eigenstates of an isolated dual rail."""
        if pair == "A":
            left, right, coupling_key = "A1", "A2", "A1_A2"
        elif pair == "B":
            left, right, coupling_key = "B1", "B2", "B1_B2"
        else:
            raise ValueError("pair must be 'A' or 'B'")

        lm, rm = self.local[left], self.local[right]
        identity = qt.qeye(self.dim)
        h_pair = TWOPI * (
            4.0 * lm.spec.EC_GHz * qt.tensor(lm.n2_op, identity)
            - lm.spec.EJ_GHz * qt.tensor(lm.cos_phi_op, identity)
            + 4.0 * rm.spec.EC_GHz * qt.tensor(identity, rm.n2_op)
            - rm.spec.EJ_GHz * qt.tensor(identity, rm.cos_phi_op)
            + self.couplings_GHz[coupling_key] * qt.tensor(lm.n_op, rm.n_op)
        )
        evals, states = h_pair.eigenstates()
        bare_10 = qt.tensor(lm.level_kets[1], rm.level_kets[0])
        bare_01 = qt.tensor(lm.level_kets[0], rm.level_kets[1])
        weights = np.asarray(
            [abs(bare_10.overlap(state)) ** 2 + abs(bare_01.overlap(state)) ** 2 for state in states]
        )
        candidates = one_excitation_indices(evals,weights)
        return [states[index] for index in candidates]

    def logical_basis(self) -> LogicalBasis:
        """Logical products of the isolated A and B dual-rail eigenstates.

        The A2-B1 interaction is deliberately excluded from state preparation
        and included during evolution, allowing its population dynamics and
        logical/nonlogical mixing to be observed.
        """
        logical_a = self.pair_logical_states("A")
        logical_b = self.pair_logical_states("B")
        ideal_products = [qt.tensor(logical_a[i], logical_b[j]) for i, j in product(range(2), repeat=2)]
        projector = sum(state * state.dag() for state in ideal_products)
        return LogicalBasis(
            states=ideal_products,
            labels=tuple(label+'L' for label in LOGICAL_ORDER),
            projector=projector,
            ideal_product_states=ideal_products,
        )

    def sector_projector(self, excitation_a: int, excitation_b: int) -> qt.Qobj:
        """Project using idle local-transmon level labels as excitation numbers."""
        projector = 0 * self.hamiltonian()
        for levels in product(range(self.dim), repeat=4):
            if levels[0] + levels[1] != excitation_a:
                continue
            if levels[2] + levels[3] != excitation_b:
                continue
            ket = self.local_product_ket(levels)
            projector += ket * ket.dag()
        return projector

    def level_projector(self, label: str, level: int) -> qt.Qobj:
        local_projector = self.local[label].level_kets[level]
        local_projector = local_projector * local_projector.dag()
        return embed(local_projector, LABELS.index(label), self.dim)

    def local_energy_lowering_operator(self, label: str) -> qt.Qobj:
        """Annihilation operator in the transmon's local energy eigenbasis."""
        model = self.local[label]
        lowering = 0 * model.n_op
        for level in range(1, self.dim):
            lowering += (
                np.sqrt(level)
                * model.level_kets[level - 1]
                * model.level_kets[level].dag()
            )
        return lowering

    def local_energy_number_operator(self, label: str) -> qt.Qobj:
        """Level-number operator used for phenomenological pure dephasing."""
        model = self.local[label]
        number = 0 * model.n_op
        for level in range(self.dim):
            ket = model.level_kets[level]
            number += level * ket * ket.dag()
        return number

    def collapse_operators(self) -> list[qt.Qobj]:
        """Build collapse operators from the optional noise model."""
        if self.noise_model is None:
            return []

        operators: list[qt.Qobj] = []
        for label in LABELS:
            noise = self.noise_model.transmons[label]
            index = LABELS.index(label)

            if noise.T1_ns is not None:
                lowering = embed(self.local_energy_lowering_operator(label), index, self.dim)
                downward_rate = (1.0 + noise.thermal_population) / noise.T1_ns
                operators.append(np.sqrt(downward_rate) * lowering)
                if noise.thermal_population > 0.0:
                    upward_rate = noise.thermal_population / noise.T1_ns
                    operators.append(np.sqrt(upward_rate) * lowering.dag())

            if noise.Tphi_ns is not None:
                number = embed(self.local_energy_number_operator(label), index, self.dim)
                # sqrt(2/Tphi) makes the |0><1| coherence decay at 1/Tphi.
                operators.append(np.sqrt(2.0 / noise.Tphi_ns) * number)

        return operators


def closest_unitary(matrix: np.ndarray) -> np.ndarray:
    left, _, right_dag = np.linalg.svd(matrix)
    return left @ right_dag


def logical_isometry(logical: LogicalBasis) -> np.ndarray:
    """Return V_L whose four columns are the physical logical basis vectors.

    V_L maps a four-component logical ket into the truncated physical Hilbert
    space. It obeys V_L^dagger V_L = I_4, while V_L V_L^dagger is the logical
    projector in the physical space.
    """
    return np.column_stack([state.full().ravel() for state in logical.states])


def logical_to_physical(
    logical_state: np.ndarray | qt.Qobj,
    logical: LogicalBasis,
) -> qt.Qobj:
    """Encode a four-component logical ket into the physical Hilbert space."""
    if isinstance(logical_state, qt.Qobj):
        if not logical_state.isket:
            raise TypeError("logical_state must be a ket")
        amplitudes = logical_state.full().ravel()
    else:
        amplitudes = np.asarray(logical_state, dtype=complex).ravel()
    if amplitudes.size != 4:
        raise ValueError("logical_state must contain exactly four amplitudes")
    return sum(
        amplitude * basis_state
        for amplitude, basis_state in zip(amplitudes, logical.states)
    )


def physical_to_logical(
    physical_state: qt.Qobj,
    logical: LogicalBasis,
    normalize: bool = False,
) -> dict:
    """Project a physical ket or density matrix into the logical subspace.

    The returned ket or density matrix is generally subnormalized. Set
    ``normalize=True`` to condition on survival in the logical subspace.
    """
    projection = project_onto_logical(physical_state, logical)
    survival = projection["survival"]

    if projection["kind"] == "ket":
        logical_object = qt.Qobj(projection["amplitudes"], dims=[[4], [1]])
        if normalize and survival > 0.0:
            logical_object = logical_object / np.sqrt(survival)
    else:
        logical_object = qt.Qobj(
            projection["density_matrix"], dims=[[4], [4]]
        )
        if normalize and survival > 0.0:
            logical_object = logical_object / survival

    return {
        **projection,
        "logical_state": logical_object,
        "normalized": normalize and survival > 0.0,
    }


def evolve_state(
    system: FourTransmonSystem,
    initial_state: qt.Qobj,
    evolution: EvolutionSpec,
    observables: dict[str, qt.Qobj] | None = None,
) -> dict:
    """Evolve one arbitrary full-system state.

    A ket with no configured collapse operators uses ``sesolve``. A density
    matrix, or any enabled noise channel, uses ``mesolve``. The initial state
    must already live in the four-transmon physical Hilbert space.
    """
    if not (initial_state.isket or initial_state.isoper):
        raise TypeError("initial_state must be a QuTiP ket or density matrix")

    times = np.linspace(0.0, evolution.duration_ns, evolution.samples)
    collapse_operators = system.collapse_operators()
    observables = {} if observables is None else observables
    use_master_equation = bool(collapse_operators) or initial_state.isoper
    if use_master_equation:
        solver_options = {
            "store_states": True,
            "atol": 1e-9,
            "rtol": 1e-8,
            "nsteps": 20000,
        }
        density_matrix = (
            initial_state * initial_state.dag() if initial_state.isket else initial_state
        )
        result = qt.mesolve(
            system.hamiltonian(),
            density_matrix,
            times,
            c_ops=collapse_operators,
            e_ops=observables,
            options=solver_options,
        )
        solver = "mesolve"
    else:
        states = system.unitary_trajectories([initial_state], times)[0]
        solver = "sesolve"
        result = None

    if use_master_equation:
        if hasattr(result, "e_data") and result.e_data:
            expectation_data = result.e_data
        else:
            expectation_data = dict(zip(observables, result.expect))
        states = result.states
    else:
        expectation_data = {
            name: np.asarray([qt.expect(operator, state) for state in states])
            for name, operator in observables.items()
        }

    return {
        "times_ns": times,
        "states": states,
        "final_state": states[-1],
        "expectations": {
            key: np.asarray(value) for key, value in expectation_data.items()
        },
        "solver": solver,
        "collapse_operator_count": len(collapse_operators),
        "raw_result": result,
    }


def project_onto_logical(state: qt.Qobj, logical: LogicalBasis) -> dict:
    """Project a physical ket or density matrix onto the four logical states."""
    if state.isket:
        amplitudes = np.asarray(
            [logical_state.overlap(state) for logical_state in logical.states],
            dtype=complex,
        )
        populations = np.abs(amplitudes) ** 2
        return {
            "kind": "ket",
            "amplitudes": amplitudes,
            "populations": populations,
            "survival": float(np.sum(populations).real),
        }

    if state.isoper:
        def matrix_element(left: qt.Qobj, right: qt.Qobj) -> complex:
            value = left.dag() * state * right
            if isinstance(value, qt.Qobj):
                return complex(value.full()[0, 0])
            return complex(value)

        logical_density_matrix = np.asarray(
            [
                [matrix_element(left, right) for right in logical.states]
                for left in logical.states
            ],
            dtype=complex,
        )
        populations = np.real(np.diag(logical_density_matrix))
        return {
            "kind": "density_matrix",
            "density_matrix": logical_density_matrix,
            "populations": populations,
            "survival": float(np.trace(logical_density_matrix).real),
        }

    raise TypeError("state must be a QuTiP ket or density matrix")


def static_two_state_splitting(
    system: FourTransmonSystem,
    first_levels: tuple[int, int, int, int],
    second_levels: tuple[int, int, int, int],
) -> dict:
    """Analyze one bare-state collision in the full static Hamiltonian.

    The dressed splitting is obtained from the two full-system eigenstates with
    the greatest combined overlap with the requested two-dimensional bare-state
    subspace. Returned energies use GHz rather than angular-frequency units.
    """
    first = system.local_product_ket(first_levels)
    second = system.local_product_ket(second_levels)
    hamiltonian_GHz = system.hamiltonian() / TWOPI

    def matrix_element(left: qt.Qobj, right: qt.Qobj) -> complex:
        value = left.dag() * hamiltonian_GHz * right
        if isinstance(value, qt.Qobj):
            return complex(value.full()[0, 0])
        return complex(value)

    first_energy = float(matrix_element(first, first).real)
    second_energy = float(matrix_element(second, second).real)
    direct_coupling = matrix_element(first, second)
    bare_detuning = second_energy - first_energy
    two_state_splitting = float(
        np.sqrt(bare_detuning**2 + 4.0 * abs(direct_coupling) ** 2)
    )

    eigenenergies_rad, eigenvectors = system.spectral_decomposition()
    eigenenergies = eigenenergies_rad / TWOPI
    eigenstates = [
        qt.Qobj(eigenvectors[:, index].reshape(-1, 1), dims=first.dims)
        for index in range(eigenvectors.shape[1])
    ]
    subspace_weights = np.asarray(
        [
            abs(first.overlap(state)) ** 2 + abs(second.overlap(state)) ** 2
            for state in eigenstates
        ]
    )
    selected = np.argsort(subspace_weights)[-2:]
    selected = sorted(selected, key=lambda index: eigenenergies[index])
    dressed_energies = np.asarray([eigenenergies[index] for index in selected])
    dressed_weights = np.asarray([subspace_weights[index] for index in selected])
    component_weights = np.asarray(
        [
            [
                abs(first.overlap(eigenstates[index])) ** 2,
                abs(second.overlap(eigenstates[index])) ** 2,
            ]
            for index in selected
        ]
    )

    return {
        "first_levels": first_levels,
        "second_levels": second_levels,
        "bare_first_energy_GHz": first_energy,
        "bare_second_energy_GHz": second_energy,
        "bare_detuning_GHz": bare_detuning,
        "direct_coupling_GHz": direct_coupling,
        "resonant_two_level_gap_GHz": float(2.0 * abs(direct_coupling)),
        "two_level_predicted_splitting_GHz": two_state_splitting,
        "dressed_eigenenergies_GHz": dressed_energies,
        "full_dressed_splitting_GHz": float(dressed_energies[1] - dressed_energies[0]),
        "dressed_subspace_weights": dressed_weights,
        "dressed_component_weights": component_weights,
    }


def cz_collision_splittings(system: FourTransmonSystem) -> dict[str, dict]:
    """Return full-system static diagnostics for both A2-B1 CZ collisions."""
    state_11 = (0, 1, 1, 0)
    return {
        "0110_to_0200": static_two_state_splitting(
            system, state_11, (0, 2, 0, 0)
        ),
        "0110_to_0020": static_two_state_splitting(
            system, state_11, (0, 0, 2, 0)
        ),
    }


def _cz_metrics_from_projected_maps(maps: np.ndarray) -> dict[str, np.ndarray]:
    """Score projected coherent maps as CZ gates up to local Z phases."""
    sample_count = maps.shape[0]
    survival = np.empty(sample_count)
    conditional_phase = np.empty(sample_count)
    average_fidelity = np.empty(sample_count)
    for index, projected_map in enumerate(maps):
        diagonal_phases = np.angle(np.diag(projected_map))
        conditional_phase[index] = np.angle(
            projected_map[3, 3]
            * projected_map[0, 0]
            * np.conj(projected_map[2, 2] * projected_map[1, 1])
        )
        target_phases = np.asarray(
            [
                diagonal_phases[0],
                diagonal_phases[1],
                diagonal_phases[2],
                diagonal_phases[1]
                + diagonal_phases[2]
                - diagonal_phases[0]
                + np.pi,
            ]
        )
        target = np.diag(np.exp(1j * target_phases))
        norm = np.trace(projected_map.conj().T @ projected_map).real
        survival[index] = norm / 4.0
        average_fidelity[index] = (
            norm + abs(np.trace(target.conj().T @ projected_map)) ** 2
        ) / 20.0
    return {
        "survival": survival,
        "leakage": 1.0 - survival,
        "conditional_phase_rad": np.unwrap(conditional_phase),
        "average_cz_fidelity": average_fidelity,
    }


def simulate_physical_cz_full_system(
    system: FourTransmonSystem,
    evolution: EvolutionSpec,
) -> dict:
    """Diagnose the reduced A2-B1 channel inside the complete system.

    A1 and B2 begin in |0>, while A2/B1 span |00>, |01>, |10>, |11>.
    Evolution uses the complete Hamiltonian. A1 and B2 are then traced out,
    rather than required to remain in |0>. The resulting A2-B1 channel is
    scored against CZ, while A2/B1 higher levels remain leakage.
    """
    left = system.local["A2"]
    right = system.local["B1"]

    # Ordering is A1, A2, B1, B2. A1 and B2 are physical spectators in |0>.
    computational_states = [
        system.local_product_ket((0, 0, 0, 0)),
        system.local_product_ket((0, 0, 1, 0)),
        system.local_product_ket((0, 1, 0, 0)),
        system.local_product_ket((0, 1, 1, 0)),
    ]
    ket_20 = system.local_product_ket((0, 2, 0, 0))
    ket_02 = system.local_product_ket((0, 0, 2, 0))
    times = np.linspace(0.0, evolution.duration_ns, evolution.samples)

    trajectories = system.unitary_trajectories(computational_states, times)

    # Each (A1, B2) output pair labels one Kraus operator of the reduced
    # A2-B1 channel. Rows retain all local A2/B1 levels; columns are the four
    # computational inputs.
    kraus = np.empty((times.size, system.dim**2, system.dim**2, 4), dtype=complex)
    for time_index in range(times.size):
        for input_index in range(4):
            tensor = trajectories[input_index][time_index].full().reshape(
                system.dim, system.dim, system.dim, system.dim
            )
            for a1_level in range(system.dim):
                for b2_level in range(system.dim):
                    environment_index = a1_level * system.dim + b2_level
                    kraus[time_index, environment_index, :, input_index] = tensor[
                        a1_level, :, :, b2_level
                    ].reshape(-1)

    computational_rows = np.asarray(
        [0, 1, system.dim, system.dim + 1], dtype=int
    )
    reduced_maps = kraus[:, :, computational_rows, :]
    reduced_population_transfer = np.sum(np.abs(reduced_maps) ** 2, axis=1)
    # The environment-ground Kraus operator supplies the coherent residual
    # phase convention. All environment outcomes contribute to channel score.
    ground_environment_index = 0
    coherent_maps = reduced_maps[:, ground_environment_index]
    coherent_metrics = _cz_metrics_from_projected_maps(coherent_maps)

    reduced_survival = np.empty(times.size)
    reduced_fidelity = np.empty(times.size)
    reduced_purity = np.empty((4, times.size))
    plus_fidelity = np.empty(times.size)
    plus = np.ones(4, dtype=complex) / 2.0
    for time_index in range(times.size):
        phases = np.angle(np.diag(coherent_maps[time_index]))
        target_phases = np.asarray(
            [phases[0], phases[1], phases[2], phases[1] + phases[2] - phases[0] + np.pi]
        )
        target = np.diag(np.exp(1j * target_phases))
        channel_matrices = reduced_maps[time_index]
        survival_numerator = sum(
            np.trace(matrix.conj().T @ matrix).real for matrix in channel_matrices
        )
        overlap_numerator = sum(
            abs(np.trace(target.conj().T @ matrix)) ** 2
            for matrix in channel_matrices
        )
        reduced_survival[time_index] = survival_numerator / 4.0
        reduced_fidelity[time_index] = (
            survival_numerator + overlap_numerator
        ) / 20.0
        target_plus = target @ plus
        plus_fidelity[time_index] = sum(
            abs(np.vdot(target_plus, matrix @ plus)) ** 2
            for matrix in channel_matrices
        )
        for input_index in range(4):
            output_vectors = kraus[time_index, :, :, input_index]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                rho_reduced = output_vectors.T @ output_vectors.conj()
            if not np.isfinite(rho_reduced).all():
                raise FloatingPointError("Non-finite reduced A2-B1 density matrix")
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                purity = np.trace(rho_reduced @ rho_reduced).real
            reduced_purity[input_index, time_index] = purity

    input_11_trajectory = trajectories[3]
    population_11 = np.asarray(
        [abs(computational_states[3].overlap(state)) ** 2 for state in input_11_trajectory]
    )
    population_20 = np.asarray(
        [abs(ket_20.overlap(state)) ** 2 for state in input_11_trajectory]
    )
    population_02 = np.asarray(
        [abs(ket_02.overlap(state)) ** 2 for state in input_11_trajectory]
    )
    a1_ground = system.level_projector("A1", 0)
    b2_ground = system.level_projector("B2", 0)
    population_a1_excited = np.asarray(
        [1.0 - float(qt.expect(a1_ground, state).real) for state in input_11_trajectory]
    )
    population_b2_excited = np.asarray(
        [1.0 - float(qt.expect(b2_ground, state).real) for state in input_11_trajectory]
    )

    best_index = int(np.argmax(reduced_fidelity))
    frequencies = {
        "A2_w01_GHz": float(left.levels_GHz[1] - left.levels_GHz[0]),
        "A2_w12_GHz": float(left.levels_GHz[2] - left.levels_GHz[1]),
        "B1_w01_GHz": float(right.levels_GHz[1] - right.levels_GHz[0]),
        "B1_w12_GHz": float(right.levels_GHz[2] - right.levels_GHz[1]),
    }
    frequencies["A2_w12_minus_B1_w01_GHz"] = (
        frequencies["A2_w12_GHz"] - frequencies["B1_w01_GHz"]
    )
    frequencies["A2_w01_minus_B1_w12_GHz"] = (
        frequencies["A2_w01_GHz"] - frequencies["B1_w12_GHz"]
    )

    return {
        "times_ns": times,
        "frequencies": frequencies,
        "projected_maps": coherent_maps,
        "reduced_computational_survival": reduced_survival,
        "reduced_population_transfer": reduced_population_transfer,
        "leakage": 1.0 - reduced_survival,
        "conditional_phase_rad": coherent_metrics["conditional_phase_rad"],
        "average_cz_fidelity": reduced_fidelity,
        "basis_input_reduced_purity": reduced_purity,
        "plus_input_cz_fidelity": plus_fidelity,
        "population_11": population_11,
        "population_20": population_20,
        "population_02": population_02,
        "population_A1_excited": population_a1_excited,
        "population_B2_excited": population_b2_excited,
        "best_index": best_index,
        "best_time_ns": float(times[best_index]),
        "best_average_cz_fidelity": float(reduced_fidelity[best_index]),
        "best_leakage": float(1.0 - reduced_survival[best_index]),
        "best_conditional_phase_rad": float(
            coherent_metrics["conditional_phase_rad"][best_index]
        ),
        "best_projected_map": coherent_maps[best_index],
    }


def simulate_logical_basis(
    system: FourTransmonSystem,
    logical: LogicalBasis,
    evolution: EvolutionSpec,
    *,
    odd_pi_tolerance_rad: float | None = 0.05,
    end_slope_window_ns: float | None = 10.0,
) -> dict:
    observables = {
        "logical": logical.projector,
        "sector_11": system.sector_projector(1, 1),
        "sector_20": system.sector_projector(2, 0),
        "sector_02": system.sector_projector(0, 2),
    }
    for label in LABELS:
        observables[f"{label}_level2"] = system.level_projector(label, 2)

    runs = []
    state_runs = []
    final_objects = []
    solver = None
    collapse_operator_count = None
    for state in logical.states:
        evolution_result = evolve_state(system, state, evolution, observables)
        solver = evolution_result["solver"]
        collapse_operator_count = evolution_result["collapse_operator_count"]
        runs.append(evolution_result["expectations"])
        state_runs.append(evolution_result["states"])
        final_objects.append(evolution_result["final_state"])

    common_result = {
        **logical_diagnostics(system),
        "times_ns": evolution_result["times_ns"],
        "runs": runs,
        "solver": solver,
        "collapse_operator_count": collapse_operator_count,
    }

    if solver == "mesolve":
        population_transfer = np.asarray(
            [
                project_onto_logical(final_state, logical)["populations"]
                for final_state in final_objects
            ]
        ).T
        survival = float(np.sum(population_transfer) / 4.0)
        common_result.update(
            {
                "logical_population_transfer": population_transfer,
                "survival": survival,
                "leakage": 1.0 - survival,
            }
        )
        return common_result

    logical_maps = np.asarray(
        [
            [
                [
                    bra.overlap(state_runs[input_index][time_index])
                    for input_index in range(4)
                ]
                for bra in logical.states
            ]
            for time_index in range(len(common_result["times_ns"]))
        ],
        dtype=complex,
    )
    logical_metrics = _cz_metrics_from_projected_maps(logical_maps)
    product_level_tuples = list(product(range(system.dim), repeat=4))
    detectable_mask = np.asarray(
        [
            levels[0] + levels[1] == 0 or levels[2] + levels[3] == 0
            for levels in product_level_tuples
        ]
    )
    erasure_projector = 0 * system.hamiltonian()
    for levels, is_detectable in zip(product_level_tuples, detectable_mask):
        if is_detectable:
            ket = system.local_product_ket(levels)
            erasure_projector += ket * ket.dag()

    erasure_detectable_vs_time = np.asarray(
        [
            [float(qt.expect(erasure_projector, state).real) for state in states]
            for states in state_runs
        ]
    ).T
    logical_populations_vs_time = np.abs(logical_maps) ** 2
    logical_survival_by_input = np.sum(logical_populations_vs_time, axis=1)
    correct_logical_population_by_input = np.stack(
        [logical_populations_vs_time[:, index, index] for index in range(4)],
        axis=1,
    )
    wrong_logical_population_by_input = (
        logical_survival_by_input - correct_logical_population_by_input
    )
    nonlogical_leakage_by_input = 1.0 - logical_survival_by_input
    average_correct_logical_population = np.mean(
        correct_logical_population_by_input, axis=1
    )
    average_wrong_logical_population = np.mean(
        wrong_logical_population_by_input, axis=1
    )
    average_nonlogical_leakage = np.mean(nonlogical_leakage_by_input, axis=1)
    erasure_undetectable_vs_time = np.clip(
        1.0 - logical_survival_by_input - erasure_detectable_vs_time,
        0.0,
        1.0,
    )
    average_erasure_detectable = np.mean(erasure_detectable_vs_time, axis=1)
    average_erasure_undetectable = np.mean(erasure_undetectable_vs_time, axis=1)
    acceptance_probability = 1.0 - average_erasure_detectable
    conditional_cz_fidelity = np.divide(
        logical_metrics["average_cz_fidelity"],
        acceptance_probability,
        out=np.zeros_like(acceptance_probability),
        where=acceptance_probability > 1e-12,
    )
    best_index = int(np.argmax(logical_metrics["average_cz_fidelity"]))
    best_conditional_index = int(np.argmax(conditional_cz_fidelity))
    first_odd_pi = None
    if odd_pi_tolerance_rad is not None:
        first_odd_pi = odd_pi_gate_metrics(
            common_result["times_ns"],
            logical_metrics["conditional_phase_rad"],
            conditional_cz_fidelity,
            logical_metrics["average_cz_fidelity"],
            acceptance_probability,
            tolerance_rad=odd_pi_tolerance_rad,
        )
    conditional_end_slope = np.nan
    if end_slope_window_ns is not None:
        conditional_end_slope = end_of_window_slope(
            common_result["times_ns"],
            conditional_cz_fidelity,
            end_slope_window_ns,
        )
    logical_map = logical_maps[best_index]
    logical_populations_at_best = logical_populations_vs_time[best_index]
    leakage_product_populations = np.empty((4, len(product_level_tuples)))
    for input_index in range(4):
        state = state_runs[input_index][best_index]
        leaked_state = state - logical.projector * state
        product_populations = np.asarray(
            [
                abs(system.local_product_ket(levels).overlap(leaked_state)) ** 2
                for levels in product_level_tuples
            ]
        )
        leakage_product_populations[input_index] = product_populations
    erasure_detectable_leakage = erasure_detectable_vs_time[best_index]
    erasure_undetectable_leakage = erasure_undetectable_vs_time[best_index]
    survival = float(np.trace(logical_map.conj().T @ logical_map).real / 4.0)
    unitary = closest_unitary(logical_map)
    singular_values = np.linalg.svd(logical_map, compute_uv=False)

    diagonal = np.diag(unitary)
    conditional_phase = float(
        np.angle(diagonal[3])
        - np.angle(diagonal[2])
        - np.angle(diagonal[1])
        + np.angle(diagonal[0])
    )
    conditional_phase = float(np.angle(np.exp(1j * conditional_phase)))
    common_result.update(
        {
            "logical_map": logical_map,
            "logical_maps": logical_maps,
            "logical_populations_at_best": logical_populations_at_best,
            "leakage_product_level_tuples": np.asarray(
                product_level_tuples, dtype=int
            ),
            "leakage_product_populations_at_best": leakage_product_populations,
            "erasure_detectable_leakage_at_best": erasure_detectable_leakage,
            "erasure_undetectable_leakage_at_best": erasure_undetectable_leakage,
            "logical_populations_vs_time": logical_populations_vs_time,
            "correct_logical_population_by_input_vs_time": correct_logical_population_by_input,
            "wrong_logical_population_by_input_vs_time": wrong_logical_population_by_input,
            "nonlogical_leakage_by_input_vs_time": nonlogical_leakage_by_input,
            "average_correct_logical_population_vs_time": average_correct_logical_population,
            "average_wrong_logical_population_vs_time": average_wrong_logical_population,
            "average_nonlogical_leakage_vs_time": average_nonlogical_leakage,
            "erasure_detectable_leakage_vs_time": erasure_detectable_vs_time,
            "erasure_undetectable_leakage_vs_time": erasure_undetectable_vs_time,
            "average_erasure_detectable_vs_time": average_erasure_detectable,
            "average_erasure_undetectable_vs_time": average_erasure_undetectable,
            "acceptance_probability_vs_time": acceptance_probability,
            "conditional_cz_fidelity_vs_time": conditional_cz_fidelity,
            "closest_unitary": unitary,
            "survival": survival,
            "leakage": 1.0 - survival,
            "singular_values": singular_values,
            "conditional_phase_rad": conditional_phase,
            "conditional_phase_vs_time_rad": logical_metrics[
                "conditional_phase_rad"
            ],
            "average_cz_fidelity_vs_time": logical_metrics[
                "average_cz_fidelity"
            ],
            "logical_survival_vs_time": logical_metrics["survival"],
            "best_index": best_index,
            "best_time_ns": float(common_result["times_ns"][best_index]),
            "best_average_cz_fidelity": float(
                logical_metrics["average_cz_fidelity"][best_index]
            ),
            "best_conditional_index": best_conditional_index,
            "best_conditional_time_ns": float(
                common_result["times_ns"][best_conditional_index]
            ),
            "best_conditional_cz_fidelity": float(
                conditional_cz_fidelity[best_conditional_index]
            ),
            "first_odd_pi": first_odd_pi,
            "first_odd_pi_index": (
                None if first_odd_pi is None else first_odd_pi["index"]
            ),
            "first_odd_pi_time_ns": (
                np.nan if first_odd_pi is None else first_odd_pi["time_ns"]
            ),
            "conditional_cz_fidelity_at_first_odd_pi": (
                np.nan
                if first_odd_pi is None
                else first_odd_pi["conditional_cz_fidelity"]
            ),
            "unconditional_cz_fidelity_at_first_odd_pi": (
                np.nan
                if first_odd_pi is None
                else first_odd_pi["unconditional_cz_fidelity"]
            ),
            "acceptance_at_first_odd_pi": (
                np.nan
                if first_odd_pi is None
                else first_odd_pi["acceptance_probability"]
            ),
            "conditional_phase_error_at_first_odd_pi_rad": (
                np.nan
                if first_odd_pi is None
                else first_odd_pi["conditional_phase_error_rad"]
            ),
            "conditional_fidelity_end_slope_per_ns": conditional_end_slope,
            "acceptance_at_best_unconditional": float(
                acceptance_probability[best_index]
            ),
            "acceptance_at_best_conditional": float(
                acceptance_probability[best_conditional_index]
            ),
            "wrong_logical_population_at_best_unconditional": float(
                average_wrong_logical_population[best_index]
            ),
            "wrong_logical_population_at_best_conditional": float(
                average_wrong_logical_population[best_conditional_index]
            ),
            "nonlogical_leakage_at_best_unconditional": float(
                average_nonlogical_leakage[best_index]
            ),
            "nonlogical_leakage_at_best_conditional": float(
                average_nonlogical_leakage[best_conditional_index]
            ),
            "erasure_detectable_leakage_at_best_unconditional": float(
                average_erasure_detectable[best_index]
            ),
            "erasure_detectable_leakage_at_best_conditional": float(
                average_erasure_detectable[best_conditional_index]
            ),
            "erasure_undetectable_leakage_at_best_unconditional": float(
                average_erasure_undetectable[best_index]
            ),
            "erasure_undetectable_leakage_at_best_conditional": float(
                average_erasure_undetectable[best_conditional_index]
            ),
        }
    )
    return common_result


def transition_table(system: FourTransmonSystem) -> dict[str, dict[str, float]]:
    table = {}
    for label in LABELS:
        levels = system.local[label].levels_GHz
        table[label] = {
            "w01_GHz": float(levels[1] - levels[0]),
            "w12_GHz": float(levels[2] - levels[1]),
            "alpha_GHz": float(levels[2] - 2.0 * levels[1] + levels[0]),
        }
    return table


def logical_diagnostics(system):
    """Isolated-pair logical splittings and readout-convention charge H-scores.

    These omit resonator dressing and are diagnostics, not idle/off qualification.
    """
    result = {}
    for pair, left, right, key in (("A", "A1", "A2", "A1_A2"), ("B", "B1", "B2", "B1_B2")):
        lm, rm = system.local[left], system.local[right]
        identity = qt.qeye(system.dim)
        h = (qt.tensor(4*lm.spec.EC_GHz*lm.n2_op-lm.spec.EJ_GHz*lm.cos_phi_op, identity)
             + qt.tensor(identity, 4*rm.spec.EC_GHz*rm.n2_op-rm.spec.EJ_GHz*rm.cos_phi_op)
             + system.couplings_GHz[key]*qt.tensor(lm.n_op, rm.n_op))
        _, ground = h.groundstate()
        modes = system.pair_logical_states(pair)
        energies = [float(qt.expect(h, state).real) for state in modes]
        strengths = np.asarray([[abs(ground.overlap(op*state))**2 for state in modes]
                                for op in (qt.tensor(lm.n_op, identity), qt.tensor(identity, rm.n_op))])
        totals = strengths.sum(axis=1, keepdims=True)
        balance = np.divide(strengths, totals, out=np.zeros_like(strengths), where=totals>0)
        result[f"H_{pair}"] = float(np.clip(1-2*np.mean(np.abs(balance-.5)), 0, 1))
        result[f"logical_frequency_{pair}_GHz"] = energies[1]-energies[0]
    result["logical_detuning_GHz"] = result["logical_frequency_A_GHz"]-result["logical_frequency_B_GHz"]
    return result
