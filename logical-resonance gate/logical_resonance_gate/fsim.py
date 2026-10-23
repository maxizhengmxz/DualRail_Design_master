"""Time-domain fSim simulation for two transmons and a tunable coupler.

All energies are expressed as frequencies in GHz and all times in ns. The
factor of ``2*pi`` needed by the Schrodinger equation is applied internally.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np
import qutip as qt
import scqubits as scq
import xarray as xr
from pathos.pools import ProcessPool
from scipy.optimize import linear_sum_assignment

from core.constants import DATA_DIR
from core.io import save_dataset

FSIM_LANDSCAPE_SCHEMA = "three-transmon-rectangular-fsim-v1"


def _validate_cpus(cpus: int) -> int:
    if not isinstance(cpus, (int, np.integer)) or isinstance(cpus, bool) or cpus < 1:
        raise ValueError("cpus must be an integer of at least 1")
    return int(cpus)


def _process_map(function, jobs, worker_count: int):
    if worker_count == 1:
        return list(map(function, jobs))
    pool = ProcessPool(nodes=worker_count)
    try:
        return pool.map(function, jobs)
    finally:
        pool.close()
        pool.join()
        pool.clear()


def _solve_final_state(job):
    hamiltonian_terms, initial_state, integration_times, solver_options = job
    evolution = qt.sesolve(
        qt.QobjEvo(hamiltonian_terms),
        initial_state,
        integration_times,
        options=solver_options,
    )
    return evolution.final_state


@dataclass(frozen=True)
class TunableTransmonConfig:
    """Parameters of one ``scqubits.TunableTransmon`` subsystem."""

    EJmax: float
    EC: float
    d: float
    flux: float = 0.0
    ng: float = 0.0
    ncut: int = 30
    truncated_dim: int = 3


@dataclass(frozen=True)
class CouplingConfig:
    """Charge-charge coupling strengths in GHz."""

    qubit1_coupler: float = 0.1
    qubit2_coupler: float = 0.1
    qubit1_qubit2: float = 0.1


@dataclass(frozen=True)
class ThreeTransmonConfig:
    """Complete static configuration of the three-transmon device."""

    qubit1: TunableTransmonConfig = field(
        default_factory=lambda: TunableTransmonConfig(
            EJmax=17.2, EC=0.344, d=0.42
        )
    )
    qubit2: TunableTransmonConfig = field(
        default_factory=lambda: TunableTransmonConfig(
            EJmax=17.2, EC=0.334, d=0.42
        )
    )
    coupler: TunableTransmonConfig = field(
        default_factory=lambda: TunableTransmonConfig(
            EJmax=24.4, EC=0.6, d=0.1, truncated_dim=4
        )
    )
    coupling: CouplingConfig = field(default_factory=CouplingConfig)


@dataclass(frozen=True)
class CouplerPulse:
    """Synchronous cosine-ramp flux pulses for an fSim gate.

    ``rise_time_ns == duration_ns / 2`` gives a smooth pulse with no flat top.
    Smaller values insert a plateau. ``peak_flux`` controls the coupler and the
    optional qubit peaks control their detuning during the interaction. A
    qubit peak of ``None`` leaves that qubit at its idle flux.
    """

    peak_flux: float = 0.25
    duration_ns: float = 40.0
    rise_time_ns: float = 10.0
    output_samples: int = 401
    qubit1_peak_flux: float | None = None
    qubit2_peak_flux: float | None = None

    def _envelope(self, time_ns: float | np.ndarray) -> np.ndarray:
        time = np.asarray(time_ns, dtype=float)
        clipped = np.clip(time, 0.0, self.duration_ns)
        envelope = np.ones_like(clipped)
        if self.rise_time_ns > 0:
            rising = clipped < self.rise_time_ns
            falling = clipped > self.duration_ns - self.rise_time_ns
            envelope[rising] = 0.5 * (
                1.0 - np.cos(np.pi * clipped[rising] / self.rise_time_ns)
            )
            envelope[falling] = 0.5 * (
                1.0
                - np.cos(
                    np.pi
                    * (self.duration_ns - clipped[falling])
                    / self.rise_time_ns
                )
            )
        return envelope

    def subsystem_flux(
        self,
        time_ns: float | np.ndarray,
        idle_flux: float,
        peak_flux: float | None,
    ) -> np.ndarray:
        """Return one subsystem's flux waveform in flux quanta."""

        self.validate()
        peak_flux = idle_flux if peak_flux is None else peak_flux
        return idle_flux + (peak_flux - idle_flux) * self._envelope(time_ns)

    def flux(self, time_ns: float | np.ndarray, idle_flux: float) -> np.ndarray:
        """Return the coupler flux, in flux quanta, at the requested time(s)."""

        self.validate()
        return self.subsystem_flux(time_ns, idle_flux, self.peak_flux)

    def validate(self) -> None:
        if not np.isfinite(self.peak_flux):
            raise ValueError("peak_flux must be finite")
        for name, value in (
            ("qubit1_peak_flux", self.qubit1_peak_flux),
            ("qubit2_peak_flux", self.qubit2_peak_flux),
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite or None")
        if not np.isfinite(self.duration_ns) or self.duration_ns <= 0:
            raise ValueError("duration_ns must be positive and finite")
        if not np.isfinite(self.rise_time_ns) or self.rise_time_ns < 0:
            raise ValueError("rise_time_ns must be nonnegative and finite")
        if self.rise_time_ns > self.duration_ns / 2:
            raise ValueError("rise_time_ns cannot exceed half of duration_ns")
        if self.output_samples < 2:
            raise ValueError("output_samples must be at least 2")


@dataclass(frozen=True)
class SolverConfig:
    """Numerical tolerances for QuTiP's Schrodinger-equation solver."""

    atol: float = 1e-12
    rtol: float = 1e-10
    nsteps: int = 300_000
    integrator: str = "adams"

    def validate(self) -> None:
        if not np.isfinite(self.atol) or self.atol <= 0:
            raise ValueError("atol must be positive and finite")
        if not np.isfinite(self.rtol) or self.rtol <= 0:
            raise ValueError("rtol must be positive and finite")
        if self.nsteps < 1:
            raise ValueError("nsteps must be at least 1")
        if self.integrator not in qt.SESolver.avail_integrators():
            raise ValueError(
                "integrator must be one of "
                f"{sorted(qt.SESolver.avail_integrators())}"
            )


@dataclass(frozen=True)
class FSimResult:
    """Projected logical evolution and fSim diagnostics."""

    time_ns: np.ndarray
    qubit1_flux: np.ndarray
    qubit2_flux: np.ndarray
    coupler_flux: np.ndarray
    projected_unitary: np.ndarray
    theta_rad: float
    phi_rad: float
    leakage_by_state: np.ndarray
    average_leakage: float
    swap_probability: float
    parasitic_amplitude: float
    computational_assignment: np.ndarray
    computational_assignment_probability: np.ndarray

    @property
    def theta_over_pi(self) -> float:
        return self.theta_rad / np.pi

    @property
    def phi_over_pi(self) -> float:
        return self.phi_rad / np.pi


@dataclass(frozen=True)
class SwapOscillationSweep:
    """Fig. 1-style population-transfer sweep for rectangular pulses."""

    coupler_flux: np.ndarray
    duration_ns: np.ndarray
    population_01_to_10: np.ndarray
    leakage_from_01: np.ndarray
    swap_frequency_mhz: np.ndarray
    effective_coupling_mhz: np.ndarray


@dataclass(frozen=True)
class FSimModel:
    """Reusable fixed-device model; call ``simulate`` for different pulses."""

    config: ThreeTransmonConfig
    hamiltonian_ghz: qt.Qobj
    flux_operators_ghz: tuple[qt.Qobj, ...]
    coupler_flux_operator_ghz: qt.Qobj
    computational_states: tuple[qt.Qobj, ...]
    computational_assignment: np.ndarray
    computational_assignment_probability: np.ndarray

    def simulate(
        self,
        pulse: CouplerPulse,
        *,
        solver: SolverConfig | None = None,
        cpus: int = 1,
    ) -> FSimResult:
        """Propagate the four dressed computational states through one pulse.

        For shaped pulses, ``cpus > 1`` distributes the four independent ODE
        solves across ``pathos`` processes. Rectangular pulses use one exact
        diagonalization and intentionally remain serial.
        """

        pulse.validate()
        solver = SolverConfig() if solver is None else solver
        solver.validate()
        cpus = _validate_cpus(cpus)
        subsystem_configs = (
            self.config.qubit1,
            self.config.qubit2,
            self.config.coupler,
        )
        peak_fluxes = (
            pulse.qubit1_peak_flux,
            pulse.qubit2_peak_flux,
            pulse.peak_flux,
        )
        hamiltonian_terms: list = [2 * np.pi * self.hamiltonian_ghz]
        for subsystem, peak_flux, operator in zip(
            subsystem_configs,
            peak_fluxes,
            self.flux_operators_ghz,
            strict=True,
        ):
            if peak_flux is None or peak_flux == subsystem.flux:
                continue
            idle_ej = _effective_ej(subsystem, subsystem.flux)

            def delta_ej(
                time_ns: float,
                _args=None,
                *,
                subsystem=subsystem,
                peak_flux=peak_flux,
                idle_ej=idle_ej,
            ) -> float:
                flux = float(
                    pulse.subsystem_flux(time_ns, subsystem.flux, peak_flux)
                )
                return _effective_ej(subsystem, flux) - idle_ej

            hamiltonian_terms.append([2 * np.pi * operator, delta_ej])

        if pulse.rise_time_ns == 0:
            # For the nominally rectangular pulses used in Foxen et al., the
            # Hamiltonian is constant between instantaneous turn-on and turn-off.
            # Exact diagonalization is both faster and more accurate than an ODE.
            peak_hamiltonian = self.hamiltonian_ghz.copy()
            for subsystem, peak_flux, operator in zip(
                subsystem_configs,
                peak_fluxes,
                self.flux_operators_ghz,
                strict=True,
            ):
                peak_flux = subsystem.flux if peak_flux is None else peak_flux
                peak_hamiltonian += (
                    _effective_ej(subsystem, peak_flux)
                    - _effective_ej(subsystem, subsystem.flux)
                ) * operator
            eigenvalues, eigenstates = peak_hamiltonian.eigenstates()
            eigenvectors = np.column_stack(
                [state.full().ravel() for state in eigenstates]
            )
            phases = np.exp(-2j * np.pi * eigenvalues * pulse.duration_ns)
            final_states = []
            for initial_state in self.computational_states:
                amplitudes = eigenvectors.conj().T @ initial_state.full().ravel()
                final = eigenvectors @ (phases * amplitudes)
                final_states.append(qt.Qobj(final, dims=initial_state.dims))
        else:
            solver_options = {
                "atol": solver.atol,
                "rtol": solver.rtol,
                "nsteps": solver.nsteps,
                "method": solver.integrator,
                "store_states": False,
                "store_final_state": True,
            }
            integration_times = np.array([0.0, pulse.duration_ns])
            jobs = [
                (
                    hamiltonian_terms,
                    initial_state,
                    integration_times,
                    solver_options,
                )
                for initial_state in self.computational_states
            ]
            final_states = _process_map(
                _solve_final_state,
                jobs,
                min(cpus, len(jobs)),
            )

        projected = np.asarray(
            [
                [complex(output.overlap(final)) for final in final_states]
                for output in self.computational_states
            ]
        )
        leakage = 1.0 - np.sum(np.abs(projected) ** 2, axis=0)
        leakage = np.clip(np.real(leakage), 0.0, 1.0)
        theta, phi, swap_probability = _extract_fsim_angles(projected)

        allowed = np.zeros((4, 4), dtype=bool)
        allowed[0, 0] = allowed[1, 1] = allowed[1, 2] = True
        allowed[2, 1] = allowed[2, 2] = allowed[3, 3] = True
        parasitic_amplitude = float(np.max(np.abs(projected[~allowed])))

        time_ns = np.linspace(0.0, pulse.duration_ns, pulse.output_samples)
        qubit1_flux = pulse.subsystem_flux(
            time_ns, self.config.qubit1.flux, pulse.qubit1_peak_flux
        )
        qubit2_flux = pulse.subsystem_flux(
            time_ns, self.config.qubit2.flux, pulse.qubit2_peak_flux
        )
        return FSimResult(
            time_ns=time_ns,
            qubit1_flux=qubit1_flux,
            qubit2_flux=qubit2_flux,
            coupler_flux=pulse.flux(time_ns, self.config.coupler.flux),
            projected_unitary=projected,
            theta_rad=theta,
            phi_rad=phi,
            leakage_by_state=leakage,
            average_leakage=float(np.mean(leakage)),
            swap_probability=swap_probability,
            parasitic_amplitude=parasitic_amplitude,
            computational_assignment=self.computational_assignment.copy(),
            computational_assignment_probability=(
                self.computational_assignment_probability.copy()
            ),
        )


def _validate_transmon(config: TunableTransmonConfig, name: str) -> None:
    positive = {
        "EJmax": config.EJmax,
        "EC": config.EC,
    }
    for parameter, value in positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name}.{parameter} must be positive and finite")
    if not np.isfinite(config.d) or not 0 <= config.d <= 1:
        raise ValueError(f"{name}.d must lie in [0, 1]")
    if not np.isfinite(config.flux) or not np.isfinite(config.ng):
        raise ValueError(f"{name}.flux and {name}.ng must be finite")
    if config.ncut < 1:
        raise ValueError(f"{name}.ncut must be at least 1")
    if config.truncated_dim < 2 or config.truncated_dim > 2 * config.ncut + 1:
        raise ValueError(
            f"{name}.truncated_dim must lie between 2 and 2*ncut + 1"
        )


def _make_transmon(config: TunableTransmonConfig, id_str: str):
    return scq.TunableTransmon(
        EJmax=config.EJmax,
        EC=config.EC,
        d=config.d,
        flux=config.flux,
        ng=config.ng,
        ncut=config.ncut,
        truncated_dim=config.truncated_dim,
        id_str=id_str,
    )


def _effective_ej(config: TunableTransmonConfig, flux: float) -> float:
    return config.EJmax * np.sqrt(
        np.cos(np.pi * flux) ** 2
        + config.d**2 * np.sin(np.pi * flux) ** 2
    )


def _embed(operator: np.ndarray, subsystem: int, dimensions: tuple[int, ...]):
    factors = [qt.qeye(dimension) for dimension in dimensions]
    factors[subsystem] = qt.Qobj(operator)
    return qt.tensor(factors)


def build_fsim_model(config: ThreeTransmonConfig | None = None) -> FSimModel:
    """Build a truncated Hamiltonian in each subsystem's idle eigenbasis."""

    config = ThreeTransmonConfig() if config is None else config
    for name, transmon in (
        ("qubit1", config.qubit1),
        ("qubit2", config.qubit2),
        ("coupler", config.coupler),
    ):
        _validate_transmon(transmon, name)
    coupling_values = vars(config.coupling).values()
    if not all(np.isfinite(value) for value in coupling_values):
        raise ValueError("all coupling strengths must be finite")

    subsystems = (
        _make_transmon(config.qubit1, "qubit1"),
        _make_transmon(config.qubit2, "qubit2"),
        _make_transmon(config.coupler, "coupler"),
    )
    dimensions = tuple(item.truncated_dim for item in subsystems)
    eigensystems = tuple(
        item.eigensys(evals_count=item.truncated_dim) for item in subsystems
    )

    hamiltonian = sum(
        (
            _embed(np.diag(esys[0]), index, dimensions)
            for index, esys in enumerate(eigensystems)
        ),
        0 * qt.tensor([qt.qeye(dimension) for dimension in dimensions]),
    )
    charge = tuple(
        item.n_operator(energy_esys=esys)
        for item, esys in zip(subsystems, eigensystems, strict=True)
    )
    charge_full = tuple(
        _embed(operator, index, dimensions)
        for index, operator in enumerate(charge)
    )
    hamiltonian += (
        config.coupling.qubit1_coupler * charge_full[0] * charge_full[2]
        + config.coupling.qubit2_coupler * charge_full[1] * charge_full[2]
        + config.coupling.qubit1_qubit2 * charge_full[0] * charge_full[1]
    )

    flux_operators = tuple(
        _embed(-item.cos_phi_operator(energy_esys=esys), index, dimensions)
        for index, (item, esys) in enumerate(
            zip(subsystems, eigensystems, strict=True)
        )
    )

    bare_labels = ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0))
    bare_states = tuple(
        qt.tensor([qt.basis(dimensions[i], label[i]) for i in range(3)])
        for label in bare_labels
    )
    _, dressed_states = hamiltonian.eigenstates()
    overlaps = np.asarray(
        [
            [abs(bare.overlap(dressed)) ** 2 for dressed in dressed_states]
            for bare in bare_states
        ]
    )
    rows, columns = linear_sum_assignment(-overlaps)
    assignment = np.empty(4, dtype=int)
    assignment[rows] = columns
    computational_states = tuple(dressed_states[index] for index in assignment)
    assignment_probability = overlaps[np.arange(4), assignment]

    return FSimModel(
        config=config,
        hamiltonian_ghz=hamiltonian,
        flux_operators_ghz=flux_operators,
        coupler_flux_operator_ghz=flux_operators[2],
        computational_states=computational_states,
        computational_assignment=assignment,
        computational_assignment_probability=assignment_probability,
    )


def _finite_1d(values, name: str, *, minimum_size: int = 1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < minimum_size:
        raise ValueError(
            f"{name} must be one-dimensional with at least {minimum_size} values"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _peak_hamiltonian_matrix(
    model: FSimModel,
    peak_fluxes: tuple[float, float, float],
) -> np.ndarray:
    hamiltonian = model.hamiltonian_ghz.full().copy()
    subsystem_configs = (
        model.config.qubit1,
        model.config.qubit2,
        model.config.coupler,
    )
    for subsystem, peak_flux, operator in zip(
        subsystem_configs,
        peak_fluxes,
        model.flux_operators_ghz,
        strict=True,
    ):
        hamiltonian += (
            _effective_ej(subsystem, peak_flux)
            - _effective_ej(subsystem, subsystem.flux)
        ) * operator.full()
    return hamiltonian


def _rectangular_projected_evolution(
    model: FSimModel,
    peak_fluxes: tuple[float, float, float],
    duration_ns: np.ndarray,
) -> np.ndarray:
    """Return projected logical matrices for many durations at one pulse bias."""

    hamiltonian = _peak_hamiltonian_matrix(model, peak_fluxes)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    computational = np.column_stack(
        [state.full().ravel() for state in model.computational_states]
    )
    amplitudes = eigenvectors.conj().T @ computational
    phases = np.exp(-2j * np.pi * np.outer(duration_ns, eigenvalues))
    return np.einsum(
        "ka,tk,kb->tab",
        amplitudes.conj(),
        phases,
        amplitudes,
        optimize=True,
    )


def _swap_oscillation_chunk(job):
    (
        row_indices,
        model,
        coupler_flux_values,
        duration_ns,
        qubit1_peak_flux,
        qubit2_peak_flux,
    ) = job
    shape = (len(row_indices), duration_ns.size)
    population = np.empty(shape)
    leakage = np.empty(shape)
    swap_frequency_mhz = np.zeros(len(row_indices))
    time_step = duration_ns[1] - duration_ns[0]
    fft_size = 8 * 2 ** int(np.ceil(np.log2(duration_ns.size)))
    frequencies_ghz = np.fft.rfftfreq(fft_size, d=time_step)
    window = np.hanning(duration_ns.size)

    for local_index, coupler_flux in enumerate(coupler_flux_values):
        projected = _rectangular_projected_evolution(
            model,
            (
                qubit1_peak_flux,
                qubit2_peak_flux,
                float(coupler_flux),
            ),
            duration_ns,
        )
        population[local_index] = np.abs(projected[:, 2, 1]) ** 2
        logical_survival = np.sum(np.abs(projected[:, :, 1]) ** 2, axis=1)
        leakage[local_index] = np.clip(1.0 - logical_survival, 0.0, 1.0)
        centered = population[local_index] - population[local_index].mean()
        spectrum = np.abs(np.fft.rfft(centered * window, n=fft_size))
        if spectrum.size > 1 and np.max(spectrum[1:]) > 1e-10:
            peak = 1 + int(np.argmax(spectrum[1:]))
            swap_frequency_mhz[local_index] = 1e3 * frequencies_ghz[peak]

    return row_indices, population, leakage, swap_frequency_mhz


def _isolated_transition_frequency(
    config: TunableTransmonConfig, flux: float
) -> float:
    transmon = _make_transmon(
        TunableTransmonConfig(
            EJmax=config.EJmax,
            EC=config.EC,
            d=config.d,
            flux=float(flux),
            ng=config.ng,
            ncut=config.ncut,
            truncated_dim=max(2, config.truncated_dim),
        ),
        "frequency_probe",
    )
    energies = transmon.eigenvals(evals_count=2)
    return float(energies[1] - energies[0])


def _array_fingerprint(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values, dtype="<f8")
    digest = sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _fsim_model_fingerprint(model: FSimModel) -> str:
    """Return a stable fingerprint of the device configuration and model basis."""

    digest = sha256()
    digest.update(
        json.dumps(asdict(model.config), sort_keys=True, separators=(",", ":")).encode()
    )
    arrays = (
        model.hamiltonian_ghz.full(),
        *(operator.full() for operator in model.flux_operators_ghz),
        np.column_stack(
            [state.full().ravel() for state in model.computational_states]
        ),
    )
    for values in arrays:
        values = np.ascontiguousarray(values, dtype="<c16")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _fsim_landscape_chunk(job):
    (
        row_indices,
        coupler_delta_ej,
        qubit1_delta_ej,
        qubit2_delta_ej,
        resonance_delta_ej,
        duration_ns,
        hamiltonian,
        flux_operators,
        computational,
    ) = job
    row_count = len(row_indices)
    column_count = len(qubit1_delta_ej)
    shape = (row_count, column_count)
    theta = np.empty(shape)
    phi = np.empty(shape)
    leakage_11 = np.empty(shape)
    average_leakage = np.empty(shape)
    swap_probability = np.empty(shape)
    effective_coupling_mhz = np.empty(row_count)
    qubit1_operator, qubit2_operator, coupler_operator = flux_operators
    fixed_hamiltonian = hamiltonian + qubit2_delta_ej * qubit2_operator

    for row_index, coupler_delta in enumerate(coupler_delta_ej):
        row_hamiltonian = fixed_hamiltonian + coupler_delta * coupler_operator
        resonant_hamiltonian = (
            row_hamiltonian + resonance_delta_ej * qubit1_operator
        )
        resonant_energies, resonant_states = np.linalg.eigh(resonant_hamiltonian)
        odd_weight = np.sum(
            np.abs(resonant_states.conj().T @ computational[:, 1:3]) ** 2,
            axis=1,
        )
        odd_levels = np.argpartition(odd_weight, -2)[-2:]
        effective_coupling_mhz[row_index] = (
            500.0 * abs(np.diff(resonant_energies[odd_levels]))[0]
        )

        for column_index, qubit1_delta in enumerate(qubit1_delta_ej):
            eigenvalues, eigenvectors = np.linalg.eigh(
                row_hamiltonian + qubit1_delta * qubit1_operator
            )
            amplitudes = eigenvectors.conj().T @ computational
            phases = np.exp(-2j * np.pi * duration_ns * eigenvalues)
            projected = np.einsum(
                "ka,k,kb->ab",
                amplitudes.conj(),
                phases,
                amplitudes,
                optimize=True,
            )
            theta_value, phi_value, swap_value = _extract_fsim_angles(projected)
            state_leakage = np.clip(
                1.0 - np.sum(np.abs(projected) ** 2, axis=0), 0.0, 1.0
            )
            theta[row_index, column_index] = theta_value
            phi[row_index, column_index] = phi_value
            leakage_11[row_index, column_index] = state_leakage[3]
            average_leakage[row_index, column_index] = np.mean(state_leakage)
            swap_probability[row_index, column_index] = swap_value

    return (
        row_indices,
        effective_coupling_mhz,
        theta,
        phi,
        leakage_11,
        average_leakage,
        swap_probability,
    )


def run_swap_oscillation_sweep(
    model: FSimModel,
    coupler_flux_values,
    duration_ns,
    *,
    qubit1_peak_flux: float,
    qubit2_peak_flux: float,
    cpus: int = 1,
) -> SwapOscillationSweep:
    """Simulate Fig. 1-style swap chevrons and extract coupling by FFT.

    The two qubit peak fluxes should place the qubits on resonance. The
    population oscillation frequency is twice the effective coupling, matching
    the convention used in Fig. 1 of Foxen et al. With ``cpus > 1``,
    contiguous coupler-flux rows run in separate ``pathos`` processes.
    """

    coupler_flux_values = _finite_1d(
        coupler_flux_values, "coupler_flux_values", minimum_size=2
    )
    duration_ns = _finite_1d(duration_ns, "duration_ns", minimum_size=3)
    if np.any(duration_ns < 0) or np.any(np.diff(duration_ns) <= 0):
        raise ValueError("duration_ns must be nonnegative and strictly increasing")
    time_steps = np.diff(duration_ns)
    if not np.allclose(time_steps, time_steps[0], rtol=1e-7, atol=1e-12):
        raise ValueError("duration_ns must be uniformly spaced for FFT extraction")
    cpus = _validate_cpus(cpus)

    shape = (coupler_flux_values.size, duration_ns.size)
    population = np.empty(shape)
    leakage = np.empty(shape)
    swap_frequency_mhz = np.zeros(coupler_flux_values.size)
    worker_count = min(cpus, coupler_flux_values.size)
    row_chunks = np.array_split(np.arange(coupler_flux_values.size), worker_count)
    jobs = [
        (
            rows,
            model,
            coupler_flux_values[rows],
            duration_ns,
            float(qubit1_peak_flux),
            float(qubit2_peak_flux),
        )
        for rows in row_chunks
    ]
    for rows, chunk_population, chunk_leakage, chunk_frequency in _process_map(
        _swap_oscillation_chunk,
        jobs,
        worker_count,
    ):
        population[rows] = chunk_population
        leakage[rows] = chunk_leakage
        swap_frequency_mhz[rows] = chunk_frequency

    return SwapOscillationSweep(
        coupler_flux=coupler_flux_values,
        duration_ns=duration_ns,
        population_01_to_10=population,
        leakage_from_01=leakage,
        swap_frequency_mhz=swap_frequency_mhz,
        effective_coupling_mhz=0.5 * swap_frequency_mhz,
    )


def run_fsim_landscape(
    model: FSimModel,
    qubit1_flux_values,
    coupler_flux_values,
    *,
    qubit2_peak_flux: float,
    duration_ns: float,
    cpus: int = 1,
) -> xr.Dataset:
    """Simulate Fig. 2-style leakage, conditional-phase, and swap-angle maps.

    With ``cpus > 1``, contiguous groups of coupler-flux rows are evaluated in
    separate ``pathos`` processes. The returned xarray dataset is directly
    serializable with :func:`load_or_compute_fsim_landscape`.
    """

    qubit1_flux_values = _finite_1d(
        qubit1_flux_values, "qubit1_flux_values", minimum_size=2
    )
    coupler_flux_values = _finite_1d(
        coupler_flux_values, "coupler_flux_values", minimum_size=2
    )
    if not np.isfinite(qubit2_peak_flux):
        raise ValueError("qubit2_peak_flux must be finite")
    if not np.isfinite(duration_ns) or duration_ns <= 0:
        raise ValueError("duration_ns must be positive and finite")
    cpus = _validate_cpus(cpus)

    qubit2_frequency = _isolated_transition_frequency(
        model.config.qubit2, qubit2_peak_flux
    )
    detuning_mhz = 1e3 * np.asarray(
        [
            _isolated_transition_frequency(model.config.qubit1, flux)
            - qubit2_frequency
            for flux in qubit1_flux_values
        ]
    )
    resonance_order = np.argsort(detuning_mhz)
    resonance_flux = float(
        np.interp(
            0.0,
            detuning_mhz[resonance_order],
            qubit1_flux_values[resonance_order],
        )
    )

    hamiltonian = model.hamiltonian_ghz.full()
    flux_operators = tuple(
        operator.full() for operator in model.flux_operators_ghz
    )
    computational = np.column_stack(
        [state.full().ravel() for state in model.computational_states]
    )
    subsystem_configs = (
        model.config.qubit1,
        model.config.qubit2,
        model.config.coupler,
    )
    qubit1_delta_ej = np.asarray(
        [
            _effective_ej(subsystem_configs[0], flux)
            - _effective_ej(subsystem_configs[0], subsystem_configs[0].flux)
            for flux in qubit1_flux_values
        ]
    )
    qubit2_delta_ej = _effective_ej(
        subsystem_configs[1], qubit2_peak_flux
    ) - _effective_ej(subsystem_configs[1], subsystem_configs[1].flux)
    coupler_delta_ej = np.asarray(
        [
            _effective_ej(subsystem_configs[2], flux)
            - _effective_ej(subsystem_configs[2], subsystem_configs[2].flux)
            for flux in coupler_flux_values
        ]
    )
    resonance_delta_ej = _effective_ej(
        subsystem_configs[0], resonance_flux
    ) - _effective_ej(subsystem_configs[0], subsystem_configs[0].flux)

    worker_count = min(cpus, coupler_flux_values.size)
    row_chunks = np.array_split(np.arange(coupler_flux_values.size), worker_count)
    jobs = [
        (
            rows,
            coupler_delta_ej[rows],
            qubit1_delta_ej,
            qubit2_delta_ej,
            resonance_delta_ej,
            duration_ns,
            hamiltonian,
            flux_operators,
            computational,
        )
        for rows in row_chunks
    ]
    if worker_count == 1:
        chunk_results = list(map(_fsim_landscape_chunk, jobs))
    else:
        pool = ProcessPool(nodes=worker_count)
        try:
            chunk_results = pool.map(_fsim_landscape_chunk, jobs)
        finally:
            pool.close()
            pool.join()
            pool.clear()

    shape = (coupler_flux_values.size, qubit1_flux_values.size)
    theta = np.empty(shape)
    phi = np.empty(shape)
    leakage_11 = np.empty(shape)
    average_leakage = np.empty(shape)
    swap_probability = np.empty(shape)
    effective_coupling_mhz = np.empty(coupler_flux_values.size)
    for result in chunk_results:
        (
            rows,
            effective_coupling_mhz[rows],
            theta[rows],
            phi[rows],
            leakage_11[rows],
            average_leakage[rows],
            swap_probability[rows],
        ) = result

    dataset = xr.Dataset(
        data_vars={
            "detuning_mhz": (("qubit1_flux",), detuning_mhz),
            "effective_coupling_mhz": (
                ("coupler_flux",),
                effective_coupling_mhz,
            ),
            "theta_rad": (("coupler_flux", "qubit1_flux"), theta),
            "phi_rad": (("coupler_flux", "qubit1_flux"), phi),
            "leakage_11": (("coupler_flux", "qubit1_flux"), leakage_11),
            "average_leakage": (
                ("coupler_flux", "qubit1_flux"),
                average_leakage,
            ),
            "swap_probability": (
                ("coupler_flux", "qubit1_flux"),
                swap_probability,
            ),
        },
        coords={
            "qubit1_flux": qubit1_flux_values,
            "coupler_flux": coupler_flux_values,
        },
        attrs={
            "description": "Rectangular-pulse fSim landscape.",
            "fsim_landscape_schema": FSIM_LANDSCAPE_SCHEMA,
            "model_fingerprint": _fsim_model_fingerprint(model),
            "qubit1_flux_fingerprint": _array_fingerprint(qubit1_flux_values),
            "coupler_flux_fingerprint": _array_fingerprint(coupler_flux_values),
            "qubit2_peak_flux": float(qubit2_peak_flux),
            "duration_ns": float(duration_ns),
            "cpus": int(cpus),
            "worker_count": worker_count,
        },
    )
    dataset["qubit1_flux"].attrs["units"] = "flux quantum"
    dataset["coupler_flux"].attrs["units"] = "flux quantum"
    dataset["detuning_mhz"].attrs["units"] = "MHz"
    dataset["effective_coupling_mhz"].attrs["units"] = "MHz"
    dataset["theta_rad"].attrs["units"] = "rad"
    dataset["phi_rad"].attrs["units"] = "rad"
    for name in ("leakage_11", "average_leakage", "swap_probability"):
        dataset[name].attrs["units"] = "dimensionless"
    return dataset


def load_or_compute_fsim_landscape(
    model: FSimModel | None = None,
    qubit1_flux_values=None,
    coupler_flux_values=None,
    *,
    qubit2_peak_flux: float | None = None,
    duration_ns: float | None = None,
    path=DATA_DIR / "fsim_landscape.nc",
    force: bool = False,
    cpus: int = 1,
):
    """Load a validated fSim landscape or compute and save it as NetCDF."""

    path = Path(path)
    if path.exists() and not force:
        dataset = xr.load_dataset(path)
        required = {
            "detuning_mhz",
            "effective_coupling_mhz",
            "theta_rad",
            "phi_rad",
            "leakage_11",
            "average_leakage",
            "swap_probability",
        }
        missing = sorted(required - set(dataset.data_vars))
        if missing:
            dataset.close()
            raise ValueError(
                f"stale fSim landscape cache {path} is missing {missing}; "
                "rerun with force=True"
            )
        if dataset.attrs.get("fsim_landscape_schema") != FSIM_LANDSCAPE_SCHEMA:
            dataset.close()
            raise ValueError(
                f"stale fSim landscape cache {path} uses an older schema; "
                "rerun with force=True"
            )
        expected = []
        if model is not None:
            expected.append(
                ("model_fingerprint", _fsim_model_fingerprint(model), "device model")
            )
        if qubit1_flux_values is not None:
            values = _finite_1d(
                qubit1_flux_values, "qubit1_flux_values", minimum_size=2
            )
            expected.append(
                (
                    "qubit1_flux_fingerprint",
                    _array_fingerprint(values),
                    "qubit-1 flux grid",
                )
            )
        if coupler_flux_values is not None:
            values = _finite_1d(
                coupler_flux_values, "coupler_flux_values", minimum_size=2
            )
            expected.append(
                (
                    "coupler_flux_fingerprint",
                    _array_fingerprint(values),
                    "coupler flux grid",
                )
            )
        if qubit2_peak_flux is not None:
            expected.append(
                ("qubit2_peak_flux", float(qubit2_peak_flux), "qubit-2 peak flux")
            )
        if duration_ns is not None:
            expected.append(("duration_ns", float(duration_ns), "pulse duration"))
        for attribute, value, description in expected:
            if dataset.attrs.get(attribute) != value:
                dataset.close()
                raise ValueError(
                    f"fSim landscape cache {path} used a different {description}; "
                    "rerun with force=True"
                )
        return dataset, path

    if (
        model is None
        or qubit1_flux_values is None
        or coupler_flux_values is None
        or qubit2_peak_flux is None
        or duration_ns is None
    ):
        raise ValueError(
            "model, both flux grids, qubit2_peak_flux, and duration_ns are "
            "required to compute an fSim landscape"
        )
    dataset = run_fsim_landscape(
        model,
        qubit1_flux_values,
        coupler_flux_values,
        qubit2_peak_flux=qubit2_peak_flux,
        duration_ns=duration_ns,
        cpus=cpus,
    )
    return dataset, save_dataset(dataset, path)


def _extract_fsim_angles(unitary: np.ndarray) -> tuple[float, float, float]:
    """Extract principal fSim angles after ignoring global and local-Z phases."""

    odd_stay = np.sqrt((abs(unitary[1, 1]) ** 2 + abs(unitary[2, 2]) ** 2) / 2)
    odd_swap = np.sqrt((abs(unitary[1, 2]) ** 2 + abs(unitary[2, 1]) ** 2) / 2)
    theta = float(np.arctan2(odd_swap, odd_stay))
    odd_survival = odd_stay**2 + odd_swap**2
    swap_probability = float(odd_swap**2 / odd_survival) if odd_survival > 0 else np.nan

    diagonal_denominator = unitary[1, 1] * unitary[2, 2]
    swap_denominator = unitary[1, 2] * unitary[2, 1]
    if abs(diagonal_denominator) >= abs(swap_denominator):
        if abs(diagonal_denominator) < 1e-14:
            phi = np.nan
        else:
            invariant = unitary[0, 0] * unitary[3, 3] / diagonal_denominator
            phi = float(np.angle(np.exp(-1j * np.angle(invariant))))
    elif abs(swap_denominator) < 1e-14:
        phi = np.nan
    else:
        invariant = unitary[0, 0] * unitary[3, 3] / swap_denominator
        phi = float(np.angle(np.exp(1j * (np.pi - np.angle(invariant)))))
    return theta, phi, swap_probability


def simulate_fsim(
    config: ThreeTransmonConfig | None = None,
    pulse: CouplerPulse | None = None,
    *,
    solver: SolverConfig | None = None,
    cpus: int = 1,
) -> FSimResult:
    """Convenience wrapper for building a device model and simulating one pulse."""

    model = build_fsim_model(config)
    return model.simulate(
        CouplerPulse() if pulse is None else pulse,
        solver=solver,
        cpus=cpus,
    )


def ideal_fsim(theta_rad: float, phi_rad: float) -> np.ndarray:
    """Return the standard fSim matrix in the ``00, 01, 10, 11`` ordering."""

    cosine = np.cos(theta_rad)
    swap = -1j * np.sin(theta_rad)
    return np.asarray(
        [
            [1, 0, 0, 0],
            [0, cosine, swap, 0],
            [0, swap, cosine, 0],
            [0, 0, 0, np.exp(-1j * phi_rad)],
        ],
        dtype=complex,
    )


__all__ = [
    "CouplerPulse",
    "CouplingConfig",
    "FSIM_LANDSCAPE_SCHEMA",
    "FSimModel",
    "FSimResult",
    "SolverConfig",
    "SwapOscillationSweep",
    "ThreeTransmonConfig",
    "TunableTransmonConfig",
    "build_fsim_model",
    "ideal_fsim",
    "load_or_compute_fsim_landscape",
    "run_fsim_landscape",
    "run_swap_oscillation_sweep",
    "simulate_fsim",
]
