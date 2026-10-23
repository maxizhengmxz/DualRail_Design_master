# Self-contained logical-transition iSWAP/CZ study; see README.md.
# Relocation bootstrap only: numerical code and parameter values are unchanged.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()

    from dataclasses import dataclass, field

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import qutip as qt
    import scqubits as scq
    from scipy.optimize import linear_sum_assignment

    from logical_resonance_gate import (
        build_logical_contour,
    )
    from core import (
        find_operating_points,
    )

    return (
        build_logical_contour,
        dataclass,
        field,
        find_operating_points,
        linear_sum_assignment,
        mo,
        np,
        plt,
        qt,
        scq,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Logical fSim from two dual-rail transmons

    This five-transmon calculation uses subsystem order
    $[A_1,A_2,C,B_1,B_2]$.  The physical transmons in a pair need not be
    identical.  Their two fluxes are therefore resolved from a numerically
    calculated maximal-hybridization contour; in general
    $\Phi_{A_1}\ne\Phi_{A_2}$ and $\Phi_{B_1}\ne\Phi_{B_2}$.

    The sequence is
    $$(f_A,\Phi_C,f_B)_\mathrm{idle}
      \xrightarrow{\text{cosine rise}}
      (f_A,\Phi_C,f_B)_\mathrm{interaction}
      \xrightarrow{t_\mathrm{hold}}
      (f_A,\Phi_C,f_B)_\mathrm{idle}.$$

    Here $f_A$ and $f_B$ are logical $|+\rangle$--$|-\rangle$ splittings.
    During each ramp they are interpolated first, then each pair's two physical
    fluxes are evaluated on its contour. Thus the complete rise and fall—not
    only their endpoints—remain maximally hybridized. There is intentionally no
    single physical control called $\Phi_A$ or $\Phi_B$ in this notebook.

    Energies are in GHz, times are in ns, and fluxes are in units of $\Phi_0$.
    The reported logical map is ordered as
    $|++\rangle,|+-\rangle,|-+\rangle,|--\rangle$.  Leakage is therefore any
    final population outside the four dressed idle logical states.
    """)
    return


@app.cell
def _(
    build_logical_contour,
    dataclass,
    field,
    find_operating_points,
    linear_sum_assignment,
    np,
    qt,
    scq,
):
    LABELS = ("A1", "A2", "C", "B1", "B2")
    LOGICAL_LABELS = ("++", "+-", "-+", "--")

    @dataclass(frozen=True)
    class TransmonSpec:
        """One asymmetric-SQUID transmon in the scqubits convention."""

        EJmax: float
        EC: float
        d: float
        flux: float = 0.0
        ng: float = 0.0
        ncut: int = 15
        truncated_dim: int = 3

    @dataclass(frozen=True)
    class DeviceSpec:
        """Five transmons and nearest-neighbour charge couplings, all in GHz."""

        rail_a1: TransmonSpec = field(
            default_factory=lambda: TransmonSpec(
                EJmax=17.2, EC=0.320, d=0.20
            )
        )
        rail_a2: TransmonSpec = field(
            default_factory=lambda: TransmonSpec(
                EJmax=16.8, EC=0.325, d=0.22
            )
        )
        rail_b1: TransmonSpec = field(
            default_factory=lambda: TransmonSpec(
                EJmax=17.0, EC=0.315, d=0.19
            )
        )
        rail_b2: TransmonSpec = field(
            default_factory=lambda: TransmonSpec(
                EJmax=16.6, EC=0.330, d=0.21
            )
        )
        coupler: TransmonSpec = field(
            default_factory=lambda: TransmonSpec(
                EJmax=24.4,
                EC=0.50,
                d=0.08,
                truncated_dim=3,
            )
        )
        g_a1_a2: float = 0.055
        g_a2_c: float = 0.110
        g_c_b1: float = 0.110
        g_b1_b2: float = 0.055

        def subsystems(self):
            return (
                self.rail_a1,
                self.rail_a2,
                self.coupler,
                self.rail_b1,
                self.rail_b2,
            )

        def with_idle_fluxes(self, fluxes):
            """Return the same physical device in local bases at five fluxes."""

            def at_flux(spec, flux):
                return TransmonSpec(
                    EJmax=spec.EJmax,
                    EC=spec.EC,
                    d=spec.d,
                    flux=float(flux),
                    ng=spec.ng,
                    ncut=spec.ncut,
                    truncated_dim=spec.truncated_dim,
                )

            if len(fluxes) != 5:
                raise ValueError("idle fluxes must contain A1, A2, C, B1, B2")
            return DeviceSpec(
                rail_a1=at_flux(self.rail_a1, fluxes[0]),
                rail_a2=at_flux(self.rail_a2, fluxes[1]),
                coupler=at_flux(self.coupler, fluxes[2]),
                rail_b1=at_flux(self.rail_b1, fluxes[3]),
                rail_b2=at_flux(self.rail_b2, fluxes[4]),
                g_a1_a2=self.g_a1_a2,
                g_a2_c=self.g_a2_c,
                g_c_b1=self.g_c_b1,
                g_b1_b2=self.g_b1_b2,
            )

        def couplings(self):
            return (
                (0, 1, self.g_a1_a2),
                (1, 2, self.g_a2_c),
                (2, 3, self.g_c_b1),
                (3, 4, self.g_b1_b2),
            )

    @dataclass(frozen=True)
    class ContourPulse:
        """Cosine pulse that keeps both rail pairs on their operating contours."""

        contour_a: object
        contour_b: object
        idle_frequency_a_mhz: float
        interaction_frequency_a_mhz: float
        idle_coupler_flux: float
        interaction_coupler_flux: float
        idle_frequency_b_mhz: float
        interaction_frequency_b_mhz: float
        rise_time_ns: float
        hold_time_ns: float

        @property
        def duration_ns(self):
            return 2.0 * self.rise_time_ns + self.hold_time_ns

        def envelope(self, time_ns):
            time = np.asarray(time_ns, dtype=float)
            total = self.duration_ns
            if self.rise_time_ns == 0:
                return np.where((time >= 0) & (time < total), 1.0, 0.0)
            rising = 0.5 * (
                1.0
                - np.cos(
                    np.pi
                    * np.clip(time, 0, self.rise_time_ns)
                    / self.rise_time_ns
                )
            )
            falling_time = np.clip(total - time, 0, self.rise_time_ns)
            falling = 0.5 * (
                1.0 - np.cos(np.pi * falling_time / self.rise_time_ns)
            )
            return np.minimum(rising, falling)

        def logical_frequencies(self, time_ns):
            envelope = np.atleast_1d(self.envelope(time_ns))
            frequency_a = self.idle_frequency_a_mhz + envelope * (
                self.interaction_frequency_a_mhz
                - self.idle_frequency_a_mhz
            )
            frequency_b = self.idle_frequency_b_mhz + envelope * (
                self.interaction_frequency_b_mhz
                - self.idle_frequency_b_mhz
            )
            return np.vstack((frequency_a, frequency_b))

        def _fluxes_at_envelope(self, envelope):
            envelope = np.atleast_1d(np.asarray(envelope, dtype=float))
            frequencies = self.logical_frequencies_from_envelope(envelope)
            phi_a1, phi_a2 = self.contour_a.flux(frequencies[0])
            phi_b1, phi_b2 = self.contour_b.flux(frequencies[1])
            phi_c = self.idle_coupler_flux + envelope * (
                self.interaction_coupler_flux - self.idle_coupler_flux
            )
            return np.vstack((phi_a1, phi_a2, phi_c, phi_b1, phi_b2))

        def logical_frequencies_from_envelope(self, envelope):
            return np.vstack(
                (
                    self.idle_frequency_a_mhz
                    + envelope
                    * (
                        self.interaction_frequency_a_mhz
                        - self.idle_frequency_a_mhz
                    ),
                    self.idle_frequency_b_mhz
                    + envelope
                    * (
                        self.interaction_frequency_b_mhz
                        - self.idle_frequency_b_mhz
                    ),
                )
            )

        def fluxes(self, time_ns):
            return self._fluxes_at_envelope(self.envelope(time_ns))

        @property
        def idle(self):
            return tuple(self._fluxes_at_envelope(0.0)[:, 0])

        @property
        def interaction(self):
            return tuple(self._fluxes_at_envelope(1.0)[:, 0])

    @dataclass(frozen=True)
    class FiveTransmonModel:
        spec: DeviceSpec
        dimensions: tuple[int, ...]
        hamiltonian_idle_ghz: np.ndarray
        flux_operators_ghz: tuple[np.ndarray, ...]
        logical_states: np.ndarray
        logical_assignment: np.ndarray
        logical_assignment_probability: np.ndarray
        logical_reference_sign_probability: np.ndarray
        logical_idle_energies_ghz: np.ndarray

    @dataclass(frozen=True)
    class GateMetrics:
        projected_map: np.ndarray
        theta_rad: float
        phi_rad: float
        swap_probability: float
        leakage_by_state: np.ndarray
        average_leakage: float
        parasitic_amplitude: float

    def _validate_device(spec):
        for label, subsystem in zip(LABELS, spec.subsystems(), strict=True):
            if subsystem.EJmax <= 0 or subsystem.EC <= 0:
                raise ValueError(f"{label}: EJmax and EC must be positive")
            if not 0 <= subsystem.d <= 1:
                raise ValueError(f"{label}: d must lie in [0, 1]")
            if (
                subsystem.ncut < 1
                or not 2
                <= subsystem.truncated_dim
                <= 2 * subsystem.ncut + 1
            ):
                raise ValueError(f"{label}: invalid charge cutoff or truncation")
        if not np.all(
            np.isfinite([value for _, _, value in spec.couplings()])
        ):
            raise ValueError("all coupling strengths must be finite")

    def _transmon(spec, label):
        return scq.TunableTransmon(
            EJmax=spec.EJmax,
            EC=spec.EC,
            d=spec.d,
            flux=spec.flux,
            ng=spec.ng,
            ncut=spec.ncut,
            truncated_dim=spec.truncated_dim,
            id_str=label,
        )

    def _spec_at_flux(spec, flux):
        return TransmonSpec(
            EJmax=spec.EJmax,
            EC=spec.EC,
            d=spec.d,
            flux=float(flux),
            ng=spec.ng,
            ncut=spec.ncut,
            truncated_dim=spec.truncated_dim,
        )

    class ReducedPairCircuit:
        """Adapter exposing a reduced transmon pair to the contour finder."""

        def __init__(self, rail_1, rail_2, coupling_ghz):
            self.rail_1 = rail_1
            self.rail_2 = rail_2
            self.coupling_ghz = float(coupling_ghz)
            self.Φ1 = float(rail_1.flux)
            self.Φ2 = float(rail_2.flux)
            self._n1 = None
            self._n2 = None

        def eigensys(self, *, evals_count):
            transmons = (
                _transmon(_spec_at_flux(self.rail_1, self.Φ1), "rail_1"),
                _transmon(_spec_at_flux(self.rail_2, self.Φ2), "rail_2"),
            )
            dimensions = tuple(item.truncated_dim for item in transmons)
            eigensystems = tuple(
                item.eigensys(evals_count=dimension)
                for item, dimension in zip(
                    transmons, dimensions, strict=True
                )
            )
            charges = tuple(
                item.n_operator(energy_esys=esys)
                for item, esys in zip(
                    transmons, eigensystems, strict=True
                )
            )
            identities = tuple(qt.qeye(dimension) for dimension in dimensions)
            hamiltonian = (
                qt.tensor(qt.Qobj(np.diag(eigensystems[0][0])), identities[1])
                + qt.tensor(identities[0], qt.Qobj(np.diag(eigensystems[1][0])))
                + self.coupling_ghz
                * qt.tensor(qt.Qobj(charges[0]), qt.Qobj(charges[1]))
            )
            self._n1 = qt.tensor(qt.Qobj(charges[0]), identities[1]).full()
            self._n2 = qt.tensor(identities[0], qt.Qobj(charges[1])).full()
            energies, states = hamiltonian.eigenstates(eigvals=evals_count)
            return np.asarray(energies), np.column_stack(
                [state.full().ravel() for state in states]
            )

        def n1_operator(self):
            return self._n1

        def n2_operator(self):
            return self._n2

    def build_pair_contour(
        rail_1,
        rail_2,
        coupling_ghz,
        *,
        phi_1_values=None,
        phi_2_values=None,
        min_hybridization=0.97,
        segment=None,
    ):
        """Run the project's avoided-crossing search and contour builder."""

        phi_1_values = (
            np.linspace(0.0, 0.45, 61)
            if phi_1_values is None
            else np.asarray(phi_1_values, dtype=float)
        )
        phi_2_values = (
            np.linspace(0.02, 0.38, 25)
            if phi_2_values is None
            else np.asarray(phi_2_values, dtype=float)
        )
        operating_points = find_operating_points(
            ReducedPairCircuit(rail_1, rail_2, coupling_ghz),
            phi_1_values,
            phi_2_values,
            flux_params=("Φ1", "Φ2"),
            evals_count=6,
            min_hybridization=min_hybridization,
            minimize_xtol=1e-7,
            cpus=1,
        )
        contour = build_logical_contour(
            operating_points,
            segment=segment,
            min_hybridization=min_hybridization,
            max_path_gap_factor=4.0,
        )
        return contour, operating_points

    def _effective_ej(spec, flux):
        flux = np.asarray(flux, dtype=float)
        return spec.EJmax * np.sqrt(
            np.cos(np.pi * flux) ** 2
            + spec.d**2 * np.sin(np.pi * flux) ** 2
        )

    def _embed(operator, subsystem, dimensions):
        factors = [qt.qeye(dimension) for dimension in dimensions]
        factors[subsystem] = qt.Qobj(operator)
        return qt.tensor(factors)

    def _pair_modes(energies, charges, coupling):
        dimensions = (len(energies[0]), len(energies[1]))
        identity = [qt.qeye(dimension) for dimension in dimensions]
        hamiltonian = (
            qt.tensor(qt.Qobj(np.diag(energies[0])), identity[1])
            + qt.tensor(identity[0], qt.Qobj(np.diag(energies[1])))
            + coupling
            * qt.tensor(qt.Qobj(charges[0]), qt.Qobj(charges[1]))
        )
        bare_10 = qt.tensor(
            qt.basis(dimensions[0], 1), qt.basis(dimensions[1], 0)
        )
        bare_01 = qt.tensor(
            qt.basis(dimensions[0], 0), qt.basis(dimensions[1], 1)
        )
        signs = ((bare_10 + bare_01).unit(), (bare_10 - bare_01).unit())
        _, eigenstates = hamiltonian.eigenstates()
        overlaps = np.asarray(
            [
                [abs(sign.overlap(state)) ** 2 for state in eigenstates]
                for sign in signs
            ]
        )
        rows, columns = linear_sum_assignment(-overlaps)
        assignment = np.empty(2, dtype=int)
        assignment[rows] = columns
        return (
            tuple(eigenstates[index] for index in assignment),
            overlaps[np.arange(2), assignment],
        )

    def build_model(spec=None):
        """Build the coupled device in the five local idle eigenbases."""

        spec = DeviceSpec() if spec is None else spec
        _validate_device(spec)
        subsystem_specs = spec.subsystems()
        subsystems = tuple(
            _transmon(subsystem, label)
            for subsystem, label in zip(subsystem_specs, LABELS, strict=True)
        )
        dimensions = tuple(item.truncated_dim for item in subsystem_specs)
        eigensystems = tuple(
            subsystem.eigensys(evals_count=dimension)
            for subsystem, dimension in zip(
                subsystems, dimensions, strict=True
            )
        )
        energies = tuple(
            np.asarray(esys[0], dtype=float) for esys in eigensystems
        )
        charges = tuple(
            subsystem.n_operator(energy_esys=esys)
            for subsystem, esys in zip(subsystems, eigensystems, strict=True)
        )
        zero = 0 * qt.tensor(
            [qt.qeye(dimension) for dimension in dimensions]
        )
        hamiltonian = sum(
            (
                _embed(np.diag(local_energies), index, dimensions)
                for index, local_energies in enumerate(energies)
            ),
            zero,
        )
        charge_operators = tuple(
            _embed(charge, index, dimensions)
            for index, charge in enumerate(charges)
        )
        for left, right, coupling in spec.couplings():
            hamiltonian += (
                coupling * charge_operators[left] * charge_operators[right]
            )

        flux_operators = tuple(
            _embed(
                -subsystem.cos_phi_operator(energy_esys=esys),
                index,
                dimensions,
            )
            for index, (subsystem, esys) in enumerate(
                zip(subsystems, eigensystems, strict=True)
            )
        )
        pair_a, sign_probability_a = _pair_modes(
            energies[:2], charges[:2], spec.g_a1_a2
        )
        pair_b, sign_probability_b = _pair_modes(
            energies[3:], charges[3:], spec.g_b1_b2
        )
        coupler_ground = qt.basis(dimensions[2], 0)
        references = tuple(
            qt.tensor(pair_a[bit_a], coupler_ground, pair_b[bit_b])
            for bit_a, bit_b in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
        eigenvalues, eigenstates = hamiltonian.eigenstates()
        overlaps = np.asarray(
            [
                [abs(reference.overlap(state)) ** 2 for state in eigenstates]
                for reference in references
            ]
        )
        rows, columns = linear_sum_assignment(-overlaps)
        assignment = np.empty(4, dtype=int)
        assignment[rows] = columns
        logical_states = np.column_stack(
            [eigenstates[index].full().ravel() for index in assignment]
        )
        return FiveTransmonModel(
            spec=spec,
            dimensions=dimensions,
            hamiltonian_idle_ghz=hamiltonian.full(),
            flux_operators_ghz=tuple(
                operator.full() for operator in flux_operators
            ),
            logical_states=logical_states,
            logical_assignment=assignment,
            logical_assignment_probability=overlaps[np.arange(4), assignment],
            logical_reference_sign_probability=np.concatenate(
                (sign_probability_a, sign_probability_b)
            ),
            logical_idle_energies_ghz=np.asarray(eigenvalues)[assignment],
        )

    def hamiltonian_at(model, fluxes):
        """Hamiltonian at one five-flux point in the fixed idle basis."""

        if len(fluxes) != 5 or not np.all(np.isfinite(fluxes)):
            raise ValueError("fluxes must contain five finite values")
        hamiltonian = model.hamiltonian_idle_ghz.copy()
        for subsystem, idle_flux, flux, operator in zip(
            model.spec.subsystems(),
            (item.flux for item in model.spec.subsystems()),
            fluxes,
            model.flux_operators_ghz,
            strict=True,
        ):
            hamiltonian += (
                _effective_ej(subsystem, flux)
                - _effective_ej(subsystem, idle_flux)
            ) * operator
        return hamiltonian

    def extract_gate_metrics(projected):
        """Remove global/local-Z phases and report fSim invariants and leakage."""

        projected = np.asarray(projected, dtype=complex)
        odd_stay = np.sqrt(
            (abs(projected[1, 1]) ** 2 + abs(projected[2, 2]) ** 2) / 2
        )
        odd_swap = np.sqrt(
            (abs(projected[1, 2]) ** 2 + abs(projected[2, 1]) ** 2) / 2
        )
        theta = float(np.arctan2(odd_swap, odd_stay))
        odd_survival = odd_stay**2 + odd_swap**2
        swap_probability = (
            float(odd_swap**2 / odd_survival)
            if odd_survival > 1e-15
            else np.nan
        )
        diagonal_denominator = projected[1, 1] * projected[2, 2]
        swap_denominator = projected[1, 2] * projected[2, 1]
        if abs(diagonal_denominator) >= abs(swap_denominator):
            invariant = (
                projected[0, 0] * projected[3, 3] / diagonal_denominator
                if abs(diagonal_denominator) > 1e-14
                else np.nan
            )
            phi = (
                float(-np.angle(invariant))
                if np.isfinite(invariant)
                else np.nan
            )
        else:
            invariant = (
                projected[0, 0] * projected[3, 3] / swap_denominator
                if abs(swap_denominator) > 1e-14
                else np.nan
            )
            phi = (
                float(np.angle(-invariant))
                if np.isfinite(invariant)
                else np.nan
            )
        leakage = np.clip(
            1.0 - np.sum(np.abs(projected) ** 2, axis=0), 0.0, 1.0
        )
        allowed = np.eye(4, dtype=bool)
        allowed[1, 2] = allowed[2, 1] = True
        return GateMetrics(
            projected_map=projected,
            theta_rad=theta,
            phi_rad=phi,
            swap_probability=swap_probability,
            leakage_by_state=leakage,
            average_leakage=float(np.mean(leakage)),
            parasitic_amplitude=float(np.max(np.abs(projected[~allowed]))),
        )

    def rectangular_sweep(model, interaction_fluxes, hold_times_ns):
        """Exact constant-H evolution used to locate promising hold times."""

        hold_times_ns = np.asarray(hold_times_ns, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(
            hamiltonian_at(model, interaction_fluxes)
        )
        amplitudes = eigenvectors.conj().T @ model.logical_states
        phases = np.exp(-2j * np.pi * np.outer(hold_times_ns, eigenvalues))
        projected = np.einsum(
            "ka,tk,kb->tab",
            amplitudes.conj(),
            phases,
            amplitudes,
            optimize=True,
        )
        metrics = tuple(extract_gate_metrics(matrix) for matrix in projected)
        return {
            "time_ns": hold_times_ns,
            "projected": projected,
            "theta_rad": np.asarray([item.theta_rad for item in metrics]),
            "phi_rad": np.asarray([item.phi_rad for item in metrics]),
            "swap_probability": np.asarray(
                [item.swap_probability for item in metrics]
            ),
            "average_leakage": np.asarray(
                [item.average_leakage for item in metrics]
            ),
            "parasitic_amplitude": np.asarray(
                [item.parasitic_amplitude for item in metrics]
            ),
        }

    def track_static_path(model, flux_path):
        """Adiabatically track the four logical eigenbranches from idle."""

        tracked_states = model.logical_states.copy()
        energies = []
        probabilities = []
        for fluxes in flux_path:
            eigenvalues, eigenvectors = np.linalg.eigh(
                hamiltonian_at(model, fluxes)
            )
            overlap = np.abs(tracked_states.conj().T @ eigenvectors) ** 2
            rows, columns = linear_sum_assignment(-overlap)
            assignment = np.empty(4, dtype=int)
            assignment[rows] = columns
            energies.append(eigenvalues[assignment])
            probabilities.append(overlap[np.arange(4), assignment])
            tracked_states = eigenvectors[:, assignment]
        energies = np.asarray(energies)
        zz_mhz = 1e3 * (
            energies[:, 0]
            - energies[:, 1]
            - energies[:, 2]
            + energies[:, 3]
        )
        return energies, np.asarray(probabilities), zz_mhz

    def simulate_shaped_pulse(model, pulse, ramp_slices=32):
        """Propagate the four logical states with midpoint-exact ramp slices.

        Every slice is unitary.  The flat hold is one exact propagation; only
        the two smooth edges are discretized.  Increase ``ramp_slices`` to
        verify convergence at a final candidate.
        """

        if pulse.rise_time_ns < 0 or pulse.hold_time_ns < 0:
            raise ValueError("rise and hold times must be nonnegative")
        if not isinstance(ramp_slices, int) or ramp_slices < 1:
            raise ValueError("ramp_slices must be a positive integer")

        def evolve(states, fluxes, duration_ns):
            if duration_ns == 0:
                return states
            eigenvalues, eigenvectors = np.linalg.eigh(
                hamiltonian_at(model, fluxes)
            )
            phases = np.exp(-2j * np.pi * eigenvalues * duration_ns)
            return eigenvectors @ (
                phases[:, None] * (eigenvectors.conj().T @ states)
            )

        states = model.logical_states.copy()
        if pulse.rise_time_ns > 0:
            time_step = pulse.rise_time_ns / ramp_slices
            for step in range(ramp_slices):
                midpoint = (step + 0.5) * time_step
                fluxes = tuple(pulse.fluxes(midpoint)[:, 0])
                states = evolve(states, fluxes, time_step)
        states = evolve(states, pulse.interaction, pulse.hold_time_ns)
        if pulse.rise_time_ns > 0:
            time_step = pulse.rise_time_ns / ramp_slices
            for step in range(ramp_slices):
                midpoint = (
                    pulse.rise_time_ns
                    + pulse.hold_time_ns
                    + (step + 0.5) * time_step
                )
                fluxes = tuple(pulse.fluxes(midpoint)[:, 0])
                states = evolve(states, fluxes, time_step)
        projected = model.logical_states.conj().T @ states
        return extract_gate_metrics(projected)

    def choose_candidates(sweep):
        """Rank coarse rectangular iSWAP and CZ candidates."""

        leakage = sweep["average_leakage"]
        parasitic = sweep["parasitic_amplitude"]
        phase_from_zero = np.abs(
            np.angle(np.exp(1j * sweep["phi_rad"]))
        )
        iswap_cost = (
            (sweep["theta_rad"] - np.pi / 2) ** 2
            + phase_from_zero**2
            + 4 * leakage
            + parasitic**2
        )
        phase_distance = np.abs(
            np.angle(np.exp(1j * (sweep["phi_rad"] - np.pi)))
        )
        cz_cost = (
            phase_distance**2
            + 2 * sweep["swap_probability"]
            + 4 * leakage
            + parasitic**2
        )
        valid = sweep["time_ns"] > 0
        return (
            int(np.flatnonzero(valid)[np.argmin(iswap_cost[valid])]),
            int(np.flatnonzero(valid)[np.argmin(cz_cost[valid])]),
        )

    return (
        ContourPulse,
        DeviceSpec,
        LOGICAL_LABELS,
        build_model,
        build_pair_contour,
        choose_candidates,
        rectangular_sweep,
        simulate_shaped_pulse,
        track_static_path,
    )


@app.cell
def _(DeviceSpec, build_pair_contour):
    physical_device = DeviceSpec()
    contour_a, operating_points_a = build_pair_contour(
        physical_device.rail_a1,
        physical_device.rail_a2,
        physical_device.g_a1_a2,
    )
    contour_b, operating_points_b = build_pair_contour(
        physical_device.rail_b1,
        physical_device.rail_b2,
        physical_device.g_b1_b2,
    )
    return contour_a, contour_b, physical_device


@app.cell
def _(contour_a, contour_b, mo):
    frequency_a_idle = mo.ui.slider(
        *contour_a.frequency_range_mhz,
        step=0.5,
        value=125.0,
        label=r"Idle $f_A$ (MHz)",
    )
    frequency_b_idle = mo.ui.slider(
        *contour_b.frequency_range_mhz,
        step=0.5,
        value=90.0,
        label=r"Idle $f_B$ (MHz)",
    )
    phi_c_idle = mo.ui.slider(
        0.0,
        0.35,
        step=0.005,
        value=0.0,
        label=r"Idle $\Phi_C/\Phi_0$",
    )
    frequency_a_gate = mo.ui.slider(
        *contour_a.frequency_range_mhz,
        step=0.5,
        value=105.0,
        label=r"Gate $f_A$ (MHz)",
    )
    frequency_b_gate = mo.ui.slider(
        *contour_b.frequency_range_mhz,
        step=0.5,
        value=105.0,
        label=r"Gate $f_B$ (MHz)",
    )
    phi_c_gate = mo.ui.slider(
        0.0,
        0.45,
        step=0.005,
        value=0.20,
        label=r"Gate $\Phi_C/\Phi_0$",
    )
    rise_time = mo.ui.slider(
        0.0, 30.0, step=1.0, value=8.0, label="Rise/fall (ns)"
    )
    hold_time = mo.ui.slider(
        0.0, 600.0, step=1.0, value=584.0, label="Selected hold (ns)"
    )
    max_hold = mo.ui.slider(
        50.0, 600.0, step=10.0, value=600.0, label="Scan limit (ns)"
    )
    run = mo.ui.run_button(label="Run logical-gate exploration")

    mo.vstack(
        [
            mo.md("### Idle point"),
            mo.hstack(
                [frequency_a_idle, phi_c_idle, frequency_b_idle],
                widths="equal",
            ),
            mo.md("### Interaction point"),
            mo.hstack(
                [frequency_a_gate, phi_c_gate, frequency_b_gate],
                widths="equal",
            ),
            mo.hstack([rise_time, hold_time, max_hold], widths="equal"),
            mo.callout(
                "f_A and f_B parameterize independently calculated "
                "maximal-hybridization contours. The five physical fluxes "
                "are displayed after the run; neither rail pair is forced to "
                "use equal fluxes.",
                kind="warn",
            ),
            run,
        ]
    )
    return (
        frequency_a_gate,
        frequency_a_idle,
        frequency_b_gate,
        frequency_b_idle,
        hold_time,
        max_hold,
        phi_c_gate,
        phi_c_idle,
        rise_time,
        run,
    )


@app.cell
def _(
    ContourPulse,
    build_model,
    contour_a,
    contour_b,
    frequency_a_gate,
    frequency_a_idle,
    frequency_b_gate,
    frequency_b_idle,
    hold_time,
    max_hold,
    mo,
    np,
    phi_c_gate,
    phi_c_idle,
    physical_device,
    rectangular_sweep,
    rise_time,
    run,
    simulate_shaped_pulse,
    track_static_path,
):
    mo.stop(
        not run.value,
        mo.callout(
            "Set the logical frequencies and coupler flux, then run the exploration.",
            kind="info",
        ),
    )

    pulse = ContourPulse(
        contour_a=contour_a,
        contour_b=contour_b,
        idle_frequency_a_mhz=frequency_a_idle.value,
        interaction_frequency_a_mhz=frequency_a_gate.value,
        idle_coupler_flux=phi_c_idle.value,
        interaction_coupler_flux=phi_c_gate.value,
        idle_frequency_b_mhz=frequency_b_idle.value,
        interaction_frequency_b_mhz=frequency_b_gate.value,
        rise_time_ns=rise_time.value,
        hold_time_ns=hold_time.value,
    )
    idle_point = pulse.idle
    interaction_point = pulse.interaction
    device = physical_device.with_idle_fluxes(idle_point)
    model = build_model(device)
    path_coordinate = np.linspace(0.0, 1.0, 31)
    flux_table = pulse._fluxes_at_envelope(path_coordinate)
    flux_path = [tuple(flux_table[:, index]) for index in range(31)]
    branch_energies, branch_overlap, zz_mhz = track_static_path(
        model, flux_path
    )
    duration = np.linspace(
        0.0, max_hold.value, int(max_hold.value * 2) + 1
    )
    coarse = rectangular_sweep(model, interaction_point, duration)
    shaped = simulate_shaped_pulse(model, pulse)
    return (
        branch_energies,
        branch_overlap,
        coarse,
        idle_point,
        interaction_point,
        model,
        path_coordinate,
        pulse,
        shaped,
        zz_mhz,
    )


@app.cell
def _(
    LOGICAL_LABELS,
    branch_energies,
    branch_overlap,
    choose_candidates,
    coarse,
    contour_a,
    contour_b,
    idle_point,
    interaction_point,
    np,
    path_coordinate,
    plt,
    pulse,
    shaped,
    zz_mhz,
):
    iswap_index, cz_index = choose_candidates(coarse)

    figure, axes = plt.subplots(
        2, 2, figsize=(11, 8), constrained_layout=True
    )
    relative_mhz = 1e3 * (branch_energies - branch_energies[:, :1])
    for index, label in enumerate(LOGICAL_LABELS):
        axes[0, 0].plot(
            path_coordinate, relative_mhz[:, index], label=label
        )
    axes[0, 0].set(
        title="Tracked logical branches along synchronous approach",
        xlabel="idle-to-interaction coordinate",
        ylabel=r"$(E-E_{++})/h$ (MHz)",
    )
    axes[0, 0].legend(ncol=2)

    axes[0, 1].plot(
        path_coordinate,
        np.min(branch_overlap, axis=1),
        label="min step overlap",
    )
    axes[0, 1].set(
        title="Adiabatic branch continuity and static ZZ",
        xlabel="idle-to-interaction coordinate",
        ylabel="minimum overlap",
        ylim=(0, 1.02),
    )
    zz_axis = axes[0, 1].twinx()
    zz_axis.plot(
        path_coordinate, zz_mhz, color="tab:red", label=r"$\zeta/2\pi$"
    )
    zz_axis.set_ylabel("conditional frequency (MHz)", color="tab:red")

    axes[1, 0].plot(
        coarse["time_ns"],
        coarse["swap_probability"],
        label="swap probability",
    )
    axes[1, 0].plot(
        coarse["time_ns"],
        np.abs(coarse["phi_rad"]) / np.pi,
        label=r"$|\phi|/\pi$",
    )
    axes[1, 0].plot(
        coarse["time_ns"],
        coarse["average_leakage"],
        label="average leakage",
    )
    axes[1, 0].axvline(
        coarse["time_ns"][iswap_index],
        color="tab:green",
        ls="--",
        alpha=0.7,
    )
    axes[1, 0].axvline(
        coarse["time_ns"][cz_index],
        color="tab:purple",
        ls=":",
        alpha=0.7,
    )
    axes[1, 0].set(
        title="Rectangular-hold feasibility scan",
        xlabel="hold time (ns)",
        ylabel="dimensionless metric",
        ylim=(-0.02, 1.05),
    )
    axes[1, 0].legend()

    pulse_times = np.linspace(0.0, pulse.duration_ns, 301)
    waveforms = pulse.fluxes(pulse_times)
    for index, label in enumerate(("A1", "A2", "C", "B1", "B2")):
        axes[1, 1].plot(pulse_times, waveforms[index], label=label)
    axes[1, 1].set(
        title="Five applied flux waveforms",
        xlabel="time (ns)",
        ylabel=r"$\Phi/\Phi_0$",
    )
    axes[1, 1].legend(ncol=3)
    for axis in axes.flat:
        axis.grid(alpha=0.25)

    map_figure, map_axes = plt.subplots(
        1, 2, figsize=(9, 3.8), constrained_layout=True
    )
    magnitude = np.abs(shaped.projected_map)
    phase = np.angle(shaped.projected_map) / np.pi
    magnitude_image = map_axes[0].imshow(
        magnitude, vmin=0, vmax=1, cmap="viridis"
    )
    phase_image = map_axes[1].imshow(
        phase, vmin=-1, vmax=1, cmap="twilight"
    )
    for axis, title in zip(
        map_axes,
        (r"$|P_L U P_L|$", r"$\arg(P_L U P_L)/\pi$"),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xticks(range(4), LOGICAL_LABELS)
        axis.set_yticks(range(4), LOGICAL_LABELS)
        axis.set_xlabel("input")
        axis.set_ylabel("output")
    map_figure.colorbar(magnitude_image, ax=map_axes[0], shrink=0.8)
    map_figure.colorbar(phase_image, ax=map_axes[1], shrink=0.8)

    contour_figure, contour_axes = plt.subplots(
        1, 2, figsize=(9, 3.8), constrained_layout=True
    )
    for contour, name, color in (
        (contour_a, "A", "tab:blue"),
        (contour_b, "B", "tab:orange"),
    ):
        contour_axes[0].plot(
            contour.phi_1,
            contour.phi_2,
            marker=".",
            color=color,
            label=f"pair {name}",
        )
        contour_axes[1].plot(
            contour.logical_frequency_mhz,
            contour.hybridization,
            color=color,
            label=f"pair {name}",
        )
    for point, marker, label in (
        (idle_point, "o", "idle"),
        (interaction_point, "*", "interaction"),
    ):
        contour_axes[0].scatter(
            [point[0], point[3]],
            [point[1], point[4]],
            marker=marker,
            s=80,
            c=["tab:blue", "tab:orange"],
            edgecolor="black",
            linewidth=0.5,
            label=label,
            zorder=3,
        )
    contour_axes[0].set(
        title="Resolved maximal-hybridization contours",
        xlabel=r"rail-1 flux $\Phi_1/\Phi_0$",
        ylabel=r"rail-2 flux $\Phi_2/\Phi_0$",
    )
    contour_axes[1].set(
        title="Accepted mixing along each contour",
        xlabel="logical splitting (MHz)",
        ylabel="hybridization",
        ylim=(0.94, 1.005),
    )
    for axis in contour_axes:
        axis.grid(alpha=0.25)
        axis.legend()
    return contour_figure, cz_index, figure, iswap_index, map_figure


@app.cell(hide_code=True)
def _(
    coarse,
    contour_figure,
    cz_index,
    figure,
    idle_point,
    interaction_point,
    iswap_index,
    map_figure,
    mo,
    model,
    np,
    shaped,
):
    def _candidate_row(name, index):
        return (
            f"| {name} | {coarse['time_ns'][index]:.1f} | "
            f"{coarse['theta_rad'][index] / np.pi:.4f} | "
            f"{coarse['phi_rad'][index] / np.pi:.4f} | "
            f"{coarse['swap_probability'][index]:.4f} | "
            f"{coarse['average_leakage'][index]:.3e} |"
        )

    summary = mo.md(
        fr"""
        ### Result at the selected smooth pulse

        | quantity | value |
        |---|---:|
        | $\theta/\pi$ | {shaped.theta_rad / np.pi:.5f} |
        | $\phi/\pi$ | {shaped.phi_rad / np.pi:.5f} |
        | odd-subspace swap probability | {shaped.swap_probability:.5f} |
        | average leakage | {shaped.average_leakage:.3e} |
        | worst-state leakage | {np.max(shaped.leakage_by_state):.3e} |
        | largest non-fSim logical amplitude | {shaped.parasitic_amplitude:.3e} |
        | minimum idle logical assignment | {np.min(model.logical_assignment_probability):.5f} |
        | minimum $+/-$ reference purity | {np.min(model.logical_reference_sign_probability):.5f} |

        ### Resolved physical flux points

        | point | $Phi_{{A1}}$ | $Phi_{{A2}}$ | $Phi_C$ | $Phi_{{B1}}$ | $Phi_{{B2}}$ |
        |---|---:|---:|---:|---:|---:|
        | idle | {idle_point[0]:.6f} | {idle_point[1]:.6f} | {idle_point[2]:.6f} | {idle_point[3]:.6f} | {idle_point[4]:.6f} |
        | interaction | {interaction_point[0]:.6f} | {interaction_point[1]:.6f} | {interaction_point[2]:.6f} | {interaction_point[3]:.6f} | {interaction_point[4]:.6f} |

        ### Coarse constant-H candidates

        Dashed/dotted lines in the hold scan mark these score minima. They are
        diagnostics, not calibrated gates; rerun the smooth pulse at the
        candidate hold and inspect leakage and the full projected map.

        | target | hold (ns) | $\theta/\pi$ | $\phi/\pi$ | swap | avg. leakage |
        |---|---:|---:|---:|---:|---:|
        {_candidate_row("iSWAP", iswap_index)}
        {_candidate_row("CZ", cz_index)}

        A credible iSWAP requires $\theta\simeq\pi/2$, while a credible CZ
        requires $\theta\simeq0$ and $|\phi|\simeq\pi$, in both cases with small
        leakage and small non-fSim amplitudes. The smooth-pulse result includes
        the two ramps; the coarse candidates do not.
        """
    )
    mo.vstack([summary, contour_figure, figure, map_figure])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
