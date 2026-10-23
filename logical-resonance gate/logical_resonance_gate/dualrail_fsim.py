"""Static feasibility tools for fSim gates between two dual-rail qubits.

This module deliberately stops at the static go/no-go stage.  It builds a
fixed-basis five-transmon reduced model, assigns the four logical states, and
maps the exchange and conditional-phase rates that determine whether pulse
optimization is worth doing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import qutip as qt
import scqubits as scq
import xarray as xr
from pathos.pools import ProcessPool
from scipy.interpolate import PchipInterpolator
from scipy.linalg import expm
from scipy.optimize import (
    brentq,
    least_squares,
    linear_sum_assignment,
    minimize,
    minimize_scalar,
)

from core.constants import DATA_DIR
from .fsim import SolverConfig, TunableTransmonConfig, ideal_fsim
from core.io import save_dataset

ISOLATED_LOGICAL_CONTOUR_SCHEMA = "dual-rail-logical-contour-v1"
STATIC_DUALRAIL_FSIM_SCHEMA = "five-transmon-static-fsim-v5"
DYNAMIC_DUALRAIL_FSIM_SCHEMA = "five-transmon-dynamic-fsim-v2"
POPULATION_TRAJECTORY_SCHEMA = "five-transmon-population-trajectory-v1"

LOGICAL_STATE_LABELS = ("00", "01", "10", "11")
LOGICAL_MODE_LABELS = ("lower", "upper")
MODE_CHARACTER_LABELS = ("+", "-")
SUBSYSTEM_LABELS = ("A1", "A2", "C", "B1", "B2")
PAULI_LABELS = ("I", "X", "Y", "Z")
PAULI_TERMS = tuple(left + right for left in PAULI_LABELS for right in PAULI_LABELS)


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


def _solve_state_trajectory(job):
    (
        hamiltonian_terms,
        time_ns,
        interpolation_order,
        initial_state,
        solver_options,
    ) = job
    angular_hamiltonian = qt.QobjEvo(
        hamiltonian_terms,
        tlist=time_ns,
        order=interpolation_order,
    )
    evolution = qt.sesolve(
        angular_hamiltonian,
        initial_state,
        time_ns,
        options=solver_options,
    )
    return np.column_stack([state.full().ravel() for state in evolution.states])


def _finite_1d(values, name: str, *, minimum_size: int = 1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < minimum_size:
        raise ValueError(
            f"{name} must be one-dimensional with at least {minimum_size} values"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _state_vector(state) -> np.ndarray:
    if isinstance(state, qt.Qobj):
        return state.full().ravel()
    state = np.asarray(state, dtype=complex)
    if state.ndim != 1:
        raise ValueError("states must be one-dimensional vectors or QuTiP kets")
    return state


@dataclass(frozen=True)
class LogicalModeMapping:
    """Map logical bits to energy modes and record the optional +/- convention."""

    logical_zero: Literal["lower", "upper"] = "lower"
    logical_one: Literal["lower", "upper"] = "upper"
    lower_character: Literal["+", "-"] | None = None

    def validate(self) -> None:
        modes = {self.logical_zero, self.logical_one}
        if modes != set(LOGICAL_MODE_LABELS):
            raise ValueError(
                "logical_zero and logical_one must map lower and upper once"
            )
        if self.lower_character not in (None, *MODE_CHARACTER_LABELS):
            raise ValueError("lower_character must be '+', '-', or None")

    def mode_for_bit(self, bit: int) -> str:
        self.validate()
        if bit not in (0, 1):
            raise ValueError("logical bit must be 0 or 1")
        return self.logical_zero if bit == 0 else self.logical_one

    def character_for_bit(self, bit: int) -> str | None:
        mode = self.mode_for_bit(bit)
        if self.lower_character is None:
            return None
        if mode == "lower":
            return self.lower_character
        return "+" if self.lower_character == "-" else "-"


@dataclass(frozen=True, eq=False)
class LogicalContour:
    """Single-valued maximal-hybridization contour parameterized by splitting."""

    logical_frequency_mhz: np.ndarray
    phi_1: np.ndarray
    phi_2: np.ndarray
    carrier_frequency_ghz: np.ndarray
    hybridization: np.ndarray
    polarization: np.ndarray
    source_segment: int = 0
    flux_params: tuple[str, str] = ("Phi_1", "Phi_2")
    two_excitation_frequency_ghz: np.ndarray | None = None
    charge_matrix_element: np.ndarray | None = None

    def __post_init__(self) -> None:
        frequency = _finite_1d(
            self.logical_frequency_mhz,
            "logical_frequency_mhz",
            minimum_size=2,
        )
        arrays = {
            "phi_1": self.phi_1,
            "phi_2": self.phi_2,
            "carrier_frequency_ghz": self.carrier_frequency_ghz,
            "hybridization": self.hybridization,
            "polarization": self.polarization,
        }
        normalized = {}
        for name, values in arrays.items():
            values = _finite_1d(values, name, minimum_size=2)
            if values.shape != frequency.shape:
                raise ValueError(f"{name} must match logical_frequency_mhz")
            normalized[name] = values.copy()
        if np.any(np.diff(frequency) <= 0):
            raise ValueError("logical_frequency_mhz must be strictly increasing")
        if np.any(
            (normalized["hybridization"] < 0) | (normalized["hybridization"] > 1)
        ):
            raise ValueError("hybridization must lie in [0, 1]")
        if len(self.flux_params) != 2 or self.flux_params[0] == self.flux_params[1]:
            raise ValueError("flux_params must contain two distinct names")

        optional = {
            "two_excitation_frequency_ghz": (
                self.two_excitation_frequency_ghz,
                (frequency.size, 3),
            ),
            "charge_matrix_element": (
                self.charge_matrix_element,
                (frequency.size, 2, 2),
            ),
        }
        for name, (values, expected_shape) in optional.items():
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            if values.shape != expected_shape or np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must have finite shape {expected_shape}")
            object.__setattr__(self, name, values.copy())

        object.__setattr__(self, "logical_frequency_mhz", frequency.copy())
        for name, values in normalized.items():
            object.__setattr__(self, name, values)

    @property
    def frequency_range_mhz(self) -> tuple[float, float]:
        return (
            float(self.logical_frequency_mhz[0]),
            float(self.logical_frequency_mhz[-1]),
        )

    def _interpolator(self, values: np.ndarray) -> PchipInterpolator:
        return PchipInterpolator(self.logical_frequency_mhz, values, extrapolate=False)

    def flux(self, logical_frequency_mhz) -> tuple[np.ndarray, np.ndarray]:
        """Return both rail fluxes at one or more logical frequencies."""

        frequency = np.asarray(logical_frequency_mhz, dtype=float)
        if not np.all(np.isfinite(frequency)):
            raise ValueError("logical_frequency_mhz must be finite")
        lower, upper = self.frequency_range_mhz
        if np.any((frequency < lower) | (frequency > upper)):
            raise ValueError(
                "logical frequency lies outside contour range "
                f"[{lower:.6g}, {upper:.6g}] MHz"
            )
        return self._interpolator(self.phi_1)(frequency), self._interpolator(
            self.phi_2
        )(frequency)

    def carrier_frequency(self, logical_frequency_mhz) -> np.ndarray:
        """Return the pair ground-to-one-excitation carrier frequency in GHz."""

        frequency = np.asarray(logical_frequency_mhz, dtype=float)
        self.flux(frequency)
        return np.asarray(self._interpolator(self.carrier_frequency_ghz)(frequency))

    def inverse_contour(self, phi_1: float, phi_2: float) -> float:
        """Project a flux pair onto the contour and return its logical frequency."""

        if not np.isfinite(phi_1) or not np.isfinite(phi_2):
            raise ValueError("phi_1 and phi_2 must be finite")
        phi_1_scale = max(float(np.ptp(self.phi_1)), 1e-12)
        phi_2_scale = max(float(np.ptp(self.phi_2)), 1e-12)
        interp_1 = self._interpolator(self.phi_1)
        interp_2 = self._interpolator(self.phi_2)

        def distance(frequency: float) -> float:
            return ((float(interp_1(frequency)) - phi_1) / phi_1_scale) ** 2 + (
                (float(interp_2(frequency)) - phi_2) / phi_2_scale
            ) ** 2

        result = minimize_scalar(
            distance,
            bounds=self.frequency_range_mhz,
            method="bounded",
        )
        if not result.success:
            raise RuntimeError("failed to project flux pair onto logical contour")
        return float(result.x)

    def to_dataset(self) -> xr.Dataset:
        frequency = self.logical_frequency_mhz
        data_vars = {
            "phi_1": ("logical_frequency_mhz", self.phi_1),
            "phi_2": ("logical_frequency_mhz", self.phi_2),
            "carrier_frequency_ghz": (
                "logical_frequency_mhz",
                self.carrier_frequency_ghz,
            ),
            "hybridization": ("logical_frequency_mhz", self.hybridization),
            "polarization": ("logical_frequency_mhz", self.polarization),
            "dphi_1_df": (
                "logical_frequency_mhz",
                self._interpolator(self.phi_1).derivative()(frequency),
            ),
            "dphi_2_df": (
                "logical_frequency_mhz",
                self._interpolator(self.phi_2).derivative()(frequency),
            ),
        }
        if self.two_excitation_frequency_ghz is not None:
            data_vars["two_excitation_frequency_ghz"] = (
                ("logical_frequency_mhz", "two_excitation_state"),
                self.two_excitation_frequency_ghz,
            )
        if self.charge_matrix_element is not None:
            data_vars["charge_matrix_element"] = (
                ("logical_frequency_mhz", "physical_rail", "logical_mode"),
                self.charge_matrix_element,
            )
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={"logical_frequency_mhz": frequency},
            attrs={
                "isolated_logical_contour_schema": ISOLATED_LOGICAL_CONTOUR_SCHEMA,
                "source_segment": self.source_segment,
                "flux_param_1": self.flux_params[0],
                "flux_param_2": self.flux_params[1],
                "fingerprint": self.fingerprint(),
            },
        )
        dataset.logical_frequency_mhz.attrs["units"] = "MHz"
        dataset.phi_1.attrs["units"] = "flux quantum"
        dataset.phi_2.attrs["units"] = "flux quantum"
        dataset.carrier_frequency_ghz.attrs["units"] = "GHz"
        if "two_excitation_frequency_ghz" in dataset:
            dataset.two_excitation_frequency_ghz.attrs["units"] = "GHz"
        dataset.dphi_1_df.attrs["units"] = "flux quantum / MHz"
        dataset.dphi_2_df.attrs["units"] = "flux quantum / MHz"
        return dataset

    def fingerprint(self) -> str:
        payload = {
            "schema": ISOLATED_LOGICAL_CONTOUR_SCHEMA,
            "logical_frequency_mhz": self.logical_frequency_mhz.tolist(),
            "phi_1": self.phi_1.tolist(),
            "phi_2": self.phi_2.tolist(),
            "carrier_frequency_ghz": self.carrier_frequency_ghz.tolist(),
            "hybridization": self.hybridization.tolist(),
            "polarization": self.polarization.tolist(),
            "source_segment": self.source_segment,
            "flux_params": self.flux_params,
            "two_excitation_frequency_ghz": (
                None
                if self.two_excitation_frequency_ghz is None
                else self.two_excitation_frequency_ghz.tolist()
            ),
            "charge_matrix_element": (
                None
                if self.charge_matrix_element is None
                else self.charge_matrix_element.tolist()
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _monotonic_segments(values: np.ndarray) -> list[slice]:
    differences = np.diff(values)
    scale = max(float(np.max(np.abs(values))), 1.0)
    if np.any(np.abs(differences) <= 1e-12 * scale):
        raise ValueError("logical frequency has repeated values and is not invertible")
    signs = np.sign(differences)
    extrema = (np.flatnonzero(signs[1:] != signs[:-1]) + 1).tolist()
    boundaries = [0, *extrema, values.size - 1]
    return [
        slice(left, right + 1)
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def build_logical_contour(
    operating_points: xr.Dataset,
    *,
    segment: int | None = None,
    frequency_range_mhz: tuple[float, float] | None = None,
    min_hybridization: float = 0.95,
    max_path_gap_factor: float = 3.0,
) -> LogicalContour:
    """Build one invertible contour branch from an operating-point dataset.

    A folded contour is rejected unless ``segment`` explicitly chooses one of
    its monotonic runs.  This prevents interpolation across two physical flux
    points with the same logical frequency.
    """

    required = {
        "phi_1",
        "phi_2",
        "splitting_ghz",
        "operating_frequency_ghz",
        "hybridization",
    }
    missing = sorted(required - set(operating_points.data_vars))
    if missing:
        raise ValueError(f"operating-point dataset is missing {missing}")
    if not 0 <= min_hybridization <= 1:
        raise ValueError("min_hybridization must lie in [0, 1]")
    if not np.isfinite(max_path_gap_factor) or max_path_gap_factor <= 1:
        raise ValueError("max_path_gap_factor must be finite and greater than 1")

    phi_1 = _finite_1d(operating_points.phi_1.values, "phi_1", minimum_size=2)
    phi_2 = _finite_1d(operating_points.phi_2.values, "phi_2", minimum_size=2)
    frequency = 1e3 * _finite_1d(
        operating_points.splitting_ghz.values,
        "splitting_ghz",
        minimum_size=2,
    )
    carrier = _finite_1d(
        operating_points.operating_frequency_ghz.values,
        "operating_frequency_ghz",
        minimum_size=2,
    )
    hybridization = _finite_1d(
        operating_points.hybridization.values,
        "hybridization",
        minimum_size=2,
    )
    if "polarization" in operating_points:
        polarization = _finite_1d(
            operating_points.polarization.values,
            "polarization",
            minimum_size=2,
        )
    else:
        polarization = np.full_like(frequency, np.nan)
    optional = {}
    for name, trailing_shape in (
        ("two_excitation_frequency_ghz", (3,)),
        ("charge_matrix_element", (2, 2)),
    ):
        if name in operating_points:
            values = np.asarray(operating_points[name].values, dtype=float)
            expected_shape = (frequency.size, *trailing_shape)
            if values.shape != expected_shape or np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must have finite shape {expected_shape}")
            optional[name] = values

    order = np.lexsort((phi_1, phi_2))
    arrays = [
        item[order]
        for item in (
            frequency,
            phi_1,
            phi_2,
            carrier,
            hybridization,
            polarization,
        )
    ]
    frequency, phi_1, phi_2, carrier, hybridization, polarization = arrays
    optional = {name: values[order] for name, values in optional.items()}
    path_steps = np.hypot(np.diff(phi_1), np.diff(phi_2))
    positive_steps = path_steps[path_steps > 0]
    if positive_steps.size == 0:
        raise ValueError("operating-point contour has no path extent")
    if np.any(path_steps > max_path_gap_factor * np.median(positive_steps)):
        raise ValueError("operating-point contour contains a disconnected branch")

    segments = _monotonic_segments(frequency)
    if len(segments) > 1 and segment is None:
        raise ValueError(
            f"logical contour is folded into {len(segments)} monotonic segments; "
            "choose segment explicitly"
        )
    segment = 0 if segment is None else segment
    if segment < 0 or segment >= len(segments):
        raise ValueError(f"segment must be between 0 and {len(segments) - 1}")
    selected = segments[segment]
    frequency, phi_1, phi_2, carrier, hybridization, polarization = [
        item[selected]
        for item in (frequency, phi_1, phi_2, carrier, hybridization, polarization)
    ]
    optional = {name: values[selected] for name, values in optional.items()}
    if frequency[0] > frequency[-1]:
        frequency, phi_1, phi_2, carrier, hybridization, polarization = [
            item[::-1]
            for item in (
                frequency,
                phi_1,
                phi_2,
                carrier,
                hybridization,
                polarization,
            )
        ]
        optional = {name: values[::-1] for name, values in optional.items()}

    if frequency_range_mhz is not None:
        lower, upper = map(float, frequency_range_mhz)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("frequency_range_mhz must be an increasing finite pair")
        available_lower = float(np.min(frequency))
        available_upper = float(np.max(frequency))
        if lower < available_lower or upper > available_upper:
            raise ValueError(
                f"selected contour covers [{available_lower:.6g}, "
                f"{available_upper:.6g}] MHz, not [{lower:.6g}, {upper:.6g}] MHz"
            )
        interior = frequency[(frequency > lower) & (frequency < upper)]
        target_frequency = np.unique(np.concatenate(([lower], interior, [upper])))

        def interpolate(values: np.ndarray) -> np.ndarray:
            if not np.all(np.isfinite(values)):
                return np.full(target_frequency.shape, np.nan)
            return np.asarray(
                PchipInterpolator(frequency, values)(target_frequency),
                dtype=float,
            )

        phi_1, phi_2, carrier, hybridization, polarization = [
            interpolate(item)
            for item in (phi_1, phi_2, carrier, hybridization, polarization)
        ]
        optional = {
            name: np.asarray(
                PchipInterpolator(
                    frequency,
                    values,
                    axis=0,
                )(target_frequency),
                dtype=float,
            )
            for name, values in optional.items()
        }
        frequency = target_frequency

    if np.any(hybridization < min_hybridization):
        weakest = float(np.min(hybridization))
        raise ValueError(
            f"selected contour hybridization reaches {weakest:.6g}, below "
            f"min_hybridization={min_hybridization:.6g}"
        )
    return LogicalContour(
        logical_frequency_mhz=frequency,
        phi_1=phi_1,
        phi_2=phi_2,
        carrier_frequency_ghz=carrier,
        hybridization=hybridization,
        polarization=polarization,
        source_segment=segment,
        flux_params=(
            operating_points.attrs.get("flux_param_1", "Phi_1"),
            operating_points.attrs.get("flux_param_2", "Phi_2"),
        ),
        two_excitation_frequency_ghz=optional.get("two_excitation_frequency_ghz"),
        charge_matrix_element=optional.get("charge_matrix_element"),
    )


@dataclass(frozen=True)
class DualRailCouplingConfig:
    """Charge-charge coupling strengths in GHz for the five-transmon model."""

    a1_a2: float = 0.05
    a2_c: float = 0.10
    c_b1: float = 0.10
    b1_b2: float = 0.05
    a2_b1: float = 0.0
    a1_c: float = 0.0
    c_b2: float = 0.0
    a1_b1: float = 0.0
    a1_b2: float = 0.0
    a2_b2: float = 0.0

    def indexed_values(self) -> tuple[tuple[int, int, float], ...]:
        return (
            (0, 1, self.a1_a2),
            (1, 2, self.a2_c),
            (2, 3, self.c_b1),
            (3, 4, self.b1_b2),
            (1, 3, self.a2_b1),
            (0, 2, self.a1_c),
            (2, 4, self.c_b2),
            (0, 3, self.a1_b1),
            (0, 4, self.a1_b2),
            (1, 4, self.a2_b2),
        )


def _default_rail_a1() -> TunableTransmonConfig:
    return TunableTransmonConfig(EJmax=12.2, EC=0.24, d=0.18, truncated_dim=3)


def _default_rail_a2() -> TunableTransmonConfig:
    return TunableTransmonConfig(EJmax=12.2, EC=0.22, d=0.18, truncated_dim=3)


def _default_coupler() -> TunableTransmonConfig:
    return TunableTransmonConfig(
        EJmax=24.4,
        EC=0.9,
        d=0.1,
        flux=0.25,
        truncated_dim=4,
    )


def _default_rail_b1() -> TunableTransmonConfig:
    return TunableTransmonConfig(EJmax=12.2, EC=0.26, d=0.18, truncated_dim=3)


def _default_rail_b2() -> TunableTransmonConfig:
    return TunableTransmonConfig(EJmax=12.2, EC=0.24, d=0.18, truncated_dim=3)


@dataclass(frozen=True)
class FiveTransmonConfig:
    """Complete reduced-model configuration in A1, A2, C, B1, B2 order."""

    rail_a1: TunableTransmonConfig = field(default_factory=_default_rail_a1)
    rail_a2: TunableTransmonConfig = field(default_factory=_default_rail_a2)
    coupler: TunableTransmonConfig = field(default_factory=_default_coupler)
    rail_b1: TunableTransmonConfig = field(default_factory=_default_rail_b1)
    rail_b2: TunableTransmonConfig = field(default_factory=_default_rail_b2)
    coupling: DualRailCouplingConfig = field(default_factory=DualRailCouplingConfig)
    logical_mapping: LogicalModeMapping = field(default_factory=LogicalModeMapping)
    parameter_source: str = "nominal workflow-validation values"
    min_assignment_probability: float = 0.5
    min_leakage_gap_mhz: float = 1.0

    def subsystems(self) -> tuple[TunableTransmonConfig, ...]:
        return (
            self.rail_a1,
            self.rail_a2,
            self.coupler,
            self.rail_b1,
            self.rail_b2,
        )


@dataclass(frozen=True)
class IsolatedPairModes:
    ground: qt.Qobj
    lower: qt.Qobj
    upper: qt.Qobj
    energies_ghz: np.ndarray
    logical_splitting_mhz: float
    carrier_frequency_ghz: float
    lower_character: str
    charge_participation: np.ndarray
    charge_matrix_element: np.ndarray
    two_excitation_frequency_ghz: np.ndarray | None


@dataclass(frozen=True)
class ReducedPairCalibration:
    """Reduced pair parameters and residuals over a reference contour."""

    rail_1: TunableTransmonConfig
    rail_2: TunableTransmonConfig
    coupling_ghz: float
    anchor_logical_frequency_mhz: float
    anchor_carrier_frequency_ghz: float
    achieved_logical_frequency_mhz: float
    sample_logical_frequency_mhz: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )
    reference_carrier_frequency_ghz: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )
    reduced_carrier_frequency_ghz: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )
    reduced_logical_splitting_mhz: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )
    carrier_residual_mhz: np.ndarray = field(default_factory=lambda: np.empty(0))
    logical_splitting_residual_mhz: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )
    reference_two_excitation_frequency_ghz: np.ndarray | None = None
    reduced_two_excitation_frequency_ghz: np.ndarray | None = None
    two_excitation_residual_mhz: np.ndarray | None = None
    reference_charge_matrix_element: np.ndarray | None = None
    reduced_charge_matrix_element: np.ndarray | None = None
    charge_matrix_element_residual: np.ndarray | None = None
    residual_tolerance_mhz: float = np.nan
    maximum_spectral_residual_mhz: float = np.nan
    residual_within_tolerance: bool = False
    optimizer_success: bool = True
    fit_selection: str = "joint"

    def to_dataset(self) -> xr.Dataset:
        """Return a serializable calibration report with every fit residual."""

        frequency = np.asarray(self.sample_logical_frequency_mhz, dtype=float)
        data_vars = {
            "reference_carrier_frequency_ghz": (
                "logical_frequency_mhz",
                self.reference_carrier_frequency_ghz,
            ),
            "reduced_carrier_frequency_ghz": (
                "logical_frequency_mhz",
                self.reduced_carrier_frequency_ghz,
            ),
            "carrier_residual_mhz": (
                "logical_frequency_mhz",
                self.carrier_residual_mhz,
            ),
            "reference_logical_splitting_mhz": (
                "logical_frequency_mhz",
                frequency,
            ),
            "reduced_logical_splitting_mhz": (
                "logical_frequency_mhz",
                self.reduced_logical_splitting_mhz,
            ),
            "logical_splitting_residual_mhz": (
                "logical_frequency_mhz",
                self.logical_splitting_residual_mhz,
            ),
        }
        if self.reference_two_excitation_frequency_ghz is not None:
            data_vars.update(
                reference_two_excitation_frequency_ghz=(
                    ("logical_frequency_mhz", "two_excitation_state"),
                    self.reference_two_excitation_frequency_ghz,
                ),
                reduced_two_excitation_frequency_ghz=(
                    ("logical_frequency_mhz", "two_excitation_state"),
                    self.reduced_two_excitation_frequency_ghz,
                ),
                two_excitation_residual_mhz=(
                    ("logical_frequency_mhz", "two_excitation_state"),
                    self.two_excitation_residual_mhz,
                ),
            )
        if self.reference_charge_matrix_element is not None:
            charge_dims = (
                "logical_frequency_mhz",
                "physical_rail",
                "logical_mode",
            )
            data_vars.update(
                reference_charge_matrix_element=(
                    charge_dims,
                    self.reference_charge_matrix_element,
                ),
                reduced_charge_matrix_element=(
                    charge_dims,
                    self.reduced_charge_matrix_element,
                ),
                charge_matrix_element_residual=(
                    charge_dims,
                    self.charge_matrix_element_residual,
                ),
            )
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "logical_frequency_mhz": frequency,
                "two_excitation_state": ["20-like", "11-like", "02-like"],
                "physical_rail": ["rail_1", "rail_2"],
                "logical_mode": ["lower", "upper"],
            },
            attrs={
                "fit_method": "multi-point least squares",
                "optimizer_success": int(self.optimizer_success),
                "residual_tolerance_mhz": self.residual_tolerance_mhz,
                "maximum_spectral_residual_mhz": (self.maximum_spectral_residual_mhz),
                "residual_within_tolerance": int(self.residual_within_tolerance),
                "fit_selection": self.fit_selection,
                "rail_1": json.dumps(asdict(self.rail_1), sort_keys=True),
                "rail_2": json.dumps(asdict(self.rail_2), sort_keys=True),
                "coupling_ghz": self.coupling_ghz,
            },
        )
        for name in dataset.data_vars:
            if "frequency" in name:
                dataset[name].attrs["units"] = "GHz"
            elif "residual_mhz" in name or "splitting_mhz" in name:
                dataset[name].attrs["units"] = "MHz"
        return dataset


@dataclass(frozen=True)
class StaticInteractionResult:
    projected_hamiltonian_ghz: np.ndarray
    effective_hamiltonian_ghz: np.ndarray
    logical_energy_ghz: np.ndarray
    logical_assignment: np.ndarray
    logical_assignment_probability: np.ndarray
    logical_singular_values: np.ndarray
    logical_subspace_sigma_min: float
    logical_detuning_mhz: float
    direct_exchange_mhz: float
    effective_exchange_mhz: float
    logical_exchange_mhz: float
    effective_zz_mhz: float
    logical_zz_mhz: float
    nearest_leakage_gap_mhz: float
    nearest_leakage_level: int
    nearest_leakage_logical_state: str
    nearest_leakage_branch_label: str
    nearest_leakage_branch_probability: float
    coupler_participation: np.ndarray
    pair_a_occupation: np.ndarray
    pair_b_occupation: np.ndarray
    estimated_iswap_time_ns: float
    estimated_cz_time_ns: float
    logical_assignment_valid: bool
    spectrally_isolated: bool


@dataclass(frozen=True)
class StaticCoherentCeilingResult:
    """Best local-Z-corrected coherent fidelity in a bounded static hold."""

    gate_type: Literal["iswap", "cz"]
    fidelity: float
    duration_ns: float
    virtual_z_rad: np.ndarray


@dataclass(frozen=True)
class StaticDynamicRateComparison:
    """Flat-top dynamic rates compared with a static logical Hamiltonian."""

    static_exchange_mhz: float
    dynamic_exchange_mhz: float
    exchange_relative_error: float
    static_zz_mhz: float
    dynamic_zz_mhz: float
    zz_relative_error: float
    relative_tolerance: float
    normalization_rate_mhz: float
    rates_agree: bool


@dataclass(frozen=True)
class DualRailPulse:
    """Synchronous endpoint-returning pulse in logical-frequency coordinates.

    ``duration_ns`` is the active pulse duration. ``padding_ns`` adds an idle
    interval of that length before and after the active pulse.
    """

    coupler_peak_flux: float
    duration_ns: float
    rise_time_ns: float = 0.0
    logical_frequency_a_peak_mhz: float | None = None
    logical_frequency_b_peak_mhz: float | None = None
    output_samples: int = 401
    target_cphase_rad: float = np.pi
    padding_ns: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("coupler_peak_flux", self.coupler_peak_flux),
            ("duration_ns", self.duration_ns),
            ("padding_ns", self.padding_ns),
            ("rise_time_ns", self.rise_time_ns),
            ("target_cphase_rad", self.target_cphase_rad),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in (
            ("logical_frequency_a_peak_mhz", self.logical_frequency_a_peak_mhz),
            ("logical_frequency_b_peak_mhz", self.logical_frequency_b_peak_mhz),
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite or None")
        if self.duration_ns <= 0:
            raise ValueError("duration_ns must be positive")
        if self.padding_ns < 0:
            raise ValueError("padding_ns must be nonnegative")
        if self.rise_time_ns < 0:
            raise ValueError("rise_time_ns must be nonnegative")
        if self.rise_time_ns > self.duration_ns / 2:
            raise ValueError("rise_time_ns cannot exceed half of duration_ns")
        if self.output_samples < 2:
            raise ValueError("output_samples must be at least 2")

    @property
    def total_duration_ns(self) -> float:
        """Return the complete idle--active--idle evolution duration."""

        return self.duration_ns + 2.0 * self.padding_ns

    def envelope(self, time_ns: float | np.ndarray) -> np.ndarray:
        """Return the shared normalized pulse envelope, including return endpoints."""

        self.validate()
        time = np.asarray(time_ns, dtype=float)
        if not np.all(np.isfinite(time)):
            raise ValueError("time_ns must be finite")
        active_time = time - self.padding_ns
        clipped = np.clip(active_time, 0.0, self.duration_ns)
        if self.rise_time_ns == 0:
            envelope = np.ones_like(clipped)
            envelope = np.where(
                (active_time <= 0.0) | (active_time >= self.duration_ns),
                0.0,
                envelope,
            )
            return envelope
        envelope = np.ones_like(clipped)
        rising = clipped < self.rise_time_ns
        falling = clipped > self.duration_ns - self.rise_time_ns
        envelope[rising] = 0.5 * (
            1.0 - np.cos(np.pi * clipped[rising] / self.rise_time_ns)
        )
        envelope[falling] = 0.5 * (
            1.0
            - np.cos(np.pi * (self.duration_ns - clipped[falling]) / self.rise_time_ns)
        )
        envelope = np.where(
            (active_time < 0.0) | (active_time > self.duration_ns),
            0.0,
            envelope,
        )
        return envelope


def _default_flux_bounds() -> tuple[tuple[float, float], ...]:
    return ((-0.5, 0.5),) * len(SUBSYSTEM_LABELS)


@dataclass(frozen=True)
class DualRailPulseConstraints:
    """Physical-domain and slew constraints in A1, A2, C, B1, B2 order."""

    flux_bounds: tuple[tuple[float, float], ...] = field(
        default_factory=_default_flux_bounds
    )
    max_flux_slew_per_ns: tuple[float, ...] | None = None
    contour_match_tolerance: float = 1e-5
    validation_samples: int = 1001

    def validate(self) -> None:
        if len(self.flux_bounds) != len(SUBSYSTEM_LABELS):
            raise ValueError("flux_bounds must contain five (lower, upper) pairs")
        for name, bounds in zip(
            SUBSYSTEM_LABELS,
            self.flux_bounds,
            strict=True,
        ):
            if len(bounds) != 2:
                raise ValueError(f"{name} flux bounds must contain two values")
            lower, upper = bounds
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(f"{name} flux bounds must be finite and increasing")
        if self.max_flux_slew_per_ns is not None:
            if len(self.max_flux_slew_per_ns) != len(SUBSYSTEM_LABELS):
                raise ValueError("max_flux_slew_per_ns must contain five values")
            slew = np.asarray(self.max_flux_slew_per_ns, dtype=float)
            if np.any(~np.isfinite(slew)) or np.any(slew <= 0):
                raise ValueError("maximum flux slews must be positive and finite")
        if (
            not np.isfinite(self.contour_match_tolerance)
            or self.contour_match_tolerance <= 0
        ):
            raise ValueError("contour_match_tolerance must be positive and finite")
        if self.validation_samples < 3:
            raise ValueError("validation_samples must be at least 3")


@dataclass(frozen=True)
class DualRailPulseResult:
    """Projected logical pulse evolution with coherent and leakage diagnostics."""

    time_ns: np.ndarray
    logical_frequency_a_mhz: np.ndarray
    logical_frequency_b_mhz: np.ndarray
    subsystem_flux: np.ndarray
    max_flux_slew_per_ns: np.ndarray
    projected_unitary: np.ndarray
    closest_unitary: np.ndarray
    theta_rad: float
    phi_rad: float
    swap_probability: float
    survival_by_state: np.ndarray
    leakage_by_state: np.ndarray
    average_leakage: float
    bare_code_probability_by_state: np.ndarray
    coupler_excited_probability_by_state: np.ndarray
    pair_a_invalid_probability_by_state: np.ndarray
    pair_b_invalid_probability_by_state: np.ndarray
    maximum_transient_leakage_by_state: np.ndarray
    maximum_transient_coupler_excited_probability_by_state: np.ndarray
    maximum_transient_pair_a_invalid_probability_by_state: np.ndarray
    maximum_transient_pair_b_invalid_probability_by_state: np.ndarray
    parasitic_amplitude: float
    unitarity_error: float
    coherent_fsim_error: float
    iswap_coherent_fidelity: float
    iswap_fidelity_with_leakage: float
    cphase_coherent_fidelity: float
    cphase_fidelity_with_leakage: float
    iswap_virtual_z_rad: np.ndarray
    cphase_virtual_z_rad: np.ndarray
    dynamic_phase_rad: np.ndarray
    logical_assignment: np.ndarray
    logical_assignment_probability: np.ndarray
    solver_method: str
    solver_status: str
    target_cphase_rad: float
    fingerprint: str

    @property
    def theta_over_pi(self) -> float:
        return self.theta_rad / np.pi

    @property
    def phi_over_pi(self) -> float:
        return self.phi_rad / np.pi

    def to_dataset(self) -> xr.Dataset:
        """Convert the result to a portable real/imaginary xarray dataset."""

        dataset = xr.Dataset(
            data_vars={
                "logical_frequency_a_mhz": (
                    "time_ns",
                    self.logical_frequency_a_mhz,
                ),
                "logical_frequency_b_mhz": (
                    "time_ns",
                    self.logical_frequency_b_mhz,
                ),
                "subsystem_flux": (
                    ("time_ns", "subsystem"),
                    self.subsystem_flux,
                ),
                "max_flux_slew_per_ns": (
                    "subsystem",
                    self.max_flux_slew_per_ns,
                ),
                "projected_unitary_real": (
                    ("logical_output", "logical_input"),
                    self.projected_unitary.real,
                ),
                "projected_unitary_imag": (
                    ("logical_output", "logical_input"),
                    self.projected_unitary.imag,
                ),
                "closest_unitary_real": (
                    ("logical_output", "logical_input"),
                    self.closest_unitary.real,
                ),
                "closest_unitary_imag": (
                    ("logical_output", "logical_input"),
                    self.closest_unitary.imag,
                ),
                "survival": ("logical_input", self.survival_by_state),
                "leakage": ("logical_input", self.leakage_by_state),
                "bare_code_probability": (
                    "logical_input",
                    self.bare_code_probability_by_state,
                ),
                "coupler_excited_probability": (
                    "logical_input",
                    self.coupler_excited_probability_by_state,
                ),
                "pair_a_invalid_probability": (
                    "logical_input",
                    self.pair_a_invalid_probability_by_state,
                ),
                "pair_b_invalid_probability": (
                    "logical_input",
                    self.pair_b_invalid_probability_by_state,
                ),
                "maximum_transient_leakage": (
                    "logical_input",
                    self.maximum_transient_leakage_by_state,
                ),
                "maximum_transient_coupler_excited_probability": (
                    "logical_input",
                    self.maximum_transient_coupler_excited_probability_by_state,
                ),
                "maximum_transient_pair_a_invalid_probability": (
                    "logical_input",
                    self.maximum_transient_pair_a_invalid_probability_by_state,
                ),
                "maximum_transient_pair_b_invalid_probability": (
                    "logical_input",
                    self.maximum_transient_pair_b_invalid_probability_by_state,
                ),
                "dynamic_phase_rad": (
                    "logical_input",
                    self.dynamic_phase_rad,
                ),
                "logical_assignment": (
                    "logical_input",
                    self.logical_assignment,
                ),
                "logical_assignment_probability": (
                    "logical_input",
                    self.logical_assignment_probability,
                ),
                "iswap_virtual_z_rad": (
                    "local_z_parameter",
                    self.iswap_virtual_z_rad,
                ),
                "cphase_virtual_z_rad": (
                    "local_z_parameter",
                    self.cphase_virtual_z_rad,
                ),
                "theta_rad": self.theta_rad,
                "phi_rad": self.phi_rad,
                "swap_probability": self.swap_probability,
                "average_leakage": self.average_leakage,
                "parasitic_amplitude": self.parasitic_amplitude,
                "unitarity_error": self.unitarity_error,
                "coherent_fsim_error": self.coherent_fsim_error,
                "iswap_coherent_fidelity": self.iswap_coherent_fidelity,
                "iswap_fidelity_with_leakage": self.iswap_fidelity_with_leakage,
                "cphase_coherent_fidelity": self.cphase_coherent_fidelity,
                "cphase_fidelity_with_leakage": self.cphase_fidelity_with_leakage,
            },
            coords={
                "time_ns": self.time_ns,
                "subsystem": list(SUBSYSTEM_LABELS),
                "logical_input": list(LOGICAL_STATE_LABELS),
                "logical_output": list(LOGICAL_STATE_LABELS),
                "local_z_parameter": [
                    "before_A",
                    "before_B",
                    "after_A",
                    "after_B",
                ],
            },
            attrs={
                "description": "Five-transmon dual-rail pulse evolution.",
                "dynamic_dualrail_fsim_schema": DYNAMIC_DUALRAIL_FSIM_SCHEMA,
                "solver_method": self.solver_method,
                "solver_status": self.solver_status,
                "target_cphase_rad": self.target_cphase_rad,
                "fingerprint": self.fingerprint,
            },
        )
        dataset.time_ns.attrs["units"] = "ns"
        dataset.logical_frequency_a_mhz.attrs["units"] = "MHz"
        dataset.logical_frequency_b_mhz.attrs["units"] = "MHz"
        dataset.subsystem_flux.attrs["units"] = "flux quantum"
        dataset.max_flux_slew_per_ns.attrs["units"] = "flux quantum / ns"
        for name in (
            "theta_rad",
            "phi_rad",
            "dynamic_phase_rad",
            "iswap_virtual_z_rad",
            "cphase_virtual_z_rad",
        ):
            dataset[name].attrs["units"] = "rad"
        return dataset


@dataclass(frozen=True)
class DualRailPopulationTrajectory:
    """Logical-state populations for one selected pulse input."""

    time_ns: np.ndarray
    logical_frequency_a_mhz: np.ndarray
    logical_frequency_b_mhz: np.ndarray
    subsystem_flux: np.ndarray
    logical_population: np.ndarray
    leakage: np.ndarray
    initial_state: str
    solver_method: str
    fingerprint: str

    def to_dataset(self) -> xr.Dataset:
        """Convert the trajectory to an xarray dataset."""

        dataset = xr.Dataset(
            data_vars={
                "logical_frequency_a_mhz": (
                    "time_ns",
                    self.logical_frequency_a_mhz,
                ),
                "logical_frequency_b_mhz": (
                    "time_ns",
                    self.logical_frequency_b_mhz,
                ),
                "subsystem_flux": (
                    ("time_ns", "subsystem"),
                    self.subsystem_flux,
                ),
                "logical_population": (
                    ("time_ns", "logical_state"),
                    self.logical_population,
                ),
                "leakage": ("time_ns", self.leakage),
            },
            coords={
                "time_ns": self.time_ns,
                "subsystem": list(SUBSYSTEM_LABELS),
                "logical_state": list(LOGICAL_STATE_LABELS),
            },
            attrs={
                "description": (
                    "Five-transmon populations during one dual-rail pulse."
                ),
                "population_trajectory_schema": POPULATION_TRAJECTORY_SCHEMA,
                "initial_state": self.initial_state,
                "solver_method": self.solver_method,
                "fingerprint": self.fingerprint,
            },
        )
        dataset.time_ns.attrs["units"] = "ns"
        dataset.logical_frequency_a_mhz.attrs["units"] = "MHz"
        dataset.logical_frequency_b_mhz.attrs["units"] = "MHz"
        dataset.subsystem_flux.attrs["units"] = "flux quantum"
        return dataset


@dataclass(frozen=True)
class DualRailFSimModel:
    config: FiveTransmonConfig
    hamiltonian_ghz: qt.Qobj
    flux_operators_ghz: tuple[qt.Qobj, ...]
    excitation_operators: tuple[qt.Qobj, ...]
    logical_references: tuple[qt.Qobj, ...]
    logical_states: tuple[qt.Qobj, ...]
    eigenvalues_ghz: np.ndarray
    eigenstates: tuple[qt.Qobj, ...]
    logical_assignment: np.ndarray
    logical_assignment_probability: np.ndarray
    pair_a_modes: IsolatedPairModes
    pair_b_modes: IsolatedPairModes

    def static_interaction(self) -> StaticInteractionResult:
        return evaluate_static_interaction(self)

    def static_logical_propagators(self, duration_ns) -> np.ndarray:
        """Project exact flat-top evolution into the bare logical references."""

        duration = _finite_1d(duration_ns, "duration_ns", minimum_size=1)
        overlap = np.asarray(
            [
                [reference.overlap(state) for state in self.eigenstates]
                for reference in self.logical_references
            ],
            dtype=complex,
        )
        phase = np.exp(-2j * np.pi * duration[:, None] * self.eigenvalues_ghz[None, :])
        return np.einsum(
            "ok,tk,ik->toi",
            overlap,
            phase,
            overlap.conj(),
            optimize=True,
        )

    def simulate(
        self,
        pulse: DualRailPulse,
        contour_a: LogicalContour,
        contour_b: LogicalContour,
        *,
        solver: SolverConfig | None = None,
        constraints: DualRailPulseConstraints | None = None,
        method: Literal["auto", "exact", "solver"] = "auto",
        cpus: int = 1,
    ) -> DualRailPulseResult:
        """Propagate the four idle dressed logical states through one pulse.

        For shaped or solver-forced pulses, ``cpus > 1`` distributes the four
        independent ODE solves across ``pathos`` processes. Exact rectangular
        propagation intentionally remains serial.
        """

        return simulate_dualrail_pulse(
            self,
            pulse,
            contour_a,
            contour_b,
            solver=solver,
            constraints=constraints,
            method=method,
            cpus=cpus,
        )

    def simulate_trajectory(
        self,
        pulse: DualRailPulse,
        contour_a: LogicalContour,
        contour_b: LogicalContour,
        initial_state: str | int,
        *,
        solver: SolverConfig | None = None,
        constraints: DualRailPulseConstraints | None = None,
        method: Literal["auto", "exact", "solver"] = "auto",
    ) -> DualRailPopulationTrajectory:
        """Propagate one logical input and retain populations at every time."""

        return simulate_dualrail_population_trajectory(
            self,
            pulse,
            contour_a,
            contour_b,
            initial_state,
            solver=solver,
            constraints=constraints,
            method=method,
        )


@dataclass(frozen=True)
class LogicalBranchTrackingResult:
    """Fixed-seed and overlap-continuous labels along one control path."""

    diabatic_level: np.ndarray
    diabatic_overlap: np.ndarray
    adiabatic_level: np.ndarray
    adiabatic_overlap: np.ndarray
    phase_alignment_rad: np.ndarray
    logical_weight: np.ndarray
    dominant_leakage_label: np.ndarray
    dominant_leakage_probability: np.ndarray
    continuity_valid: bool
    minimum_continuity_overlap: float
    logical_labels: tuple[str, ...]

    def to_dataset(self, path_coordinate=None) -> xr.Dataset:
        point_count, branch_count = self.diabatic_level.shape
        path_coordinate = (
            np.arange(point_count)
            if path_coordinate is None
            else np.asarray(path_coordinate)
        )
        if path_coordinate.shape != (point_count,):
            raise ValueError("path_coordinate must contain one value per path point")
        dimensions = ("path_point", "logical_state")
        return xr.Dataset(
            data_vars={
                "diabatic_level": (dimensions, self.diabatic_level),
                "diabatic_overlap": (dimensions, self.diabatic_overlap),
                "adiabatic_level": (dimensions, self.adiabatic_level),
                "adiabatic_overlap": (dimensions, self.adiabatic_overlap),
                "phase_alignment_rad": (dimensions, self.phase_alignment_rad),
                "logical_weight": (dimensions, self.logical_weight),
                "dominant_leakage_label": (
                    dimensions,
                    self.dominant_leakage_label.astype(str),
                ),
                "dominant_leakage_probability": (
                    dimensions,
                    self.dominant_leakage_probability,
                ),
            },
            coords={
                "path_point": path_coordinate,
                "logical_state": list(self.logical_labels[:branch_count]),
            },
            attrs={
                "tracking_method": "fixed-seed diabatic and neighbor-overlap adiabatic",
                "minimum_continuity_overlap": self.minimum_continuity_overlap,
                "continuity_valid": int(self.continuity_valid),
            },
        )


def assign_dressed_logical_states(
    reference_states,
    eigenstates,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign distinct eigenstates to references by global maximum overlap."""

    references = [_state_vector(state) for state in reference_states]
    eigenvectors = [_state_vector(state) for state in eigenstates]
    if not references or len(eigenvectors) < len(references):
        raise ValueError("need at least as many eigenstates as reference states")
    dimension = references[0].size
    if any(state.size != dimension for state in (*references, *eigenvectors)):
        raise ValueError("all reference and eigenstates must have the same dimension")
    overlap = np.asarray(
        [
            [abs(np.vdot(reference, eigenvector)) ** 2 for eigenvector in eigenvectors]
            for reference in references
        ],
        dtype=float,
    )
    rows, columns = linear_sum_assignment(-overlap)
    assignment = np.empty(len(references), dtype=int)
    assignment[rows] = columns
    return assignment, overlap[np.arange(len(references)), assignment]


def _eigenstate_columns(state_entry) -> np.ndarray:
    if isinstance(state_entry, np.ndarray) and state_entry.ndim == 2:
        return np.asarray(state_entry, dtype=complex)
    vectors = [_state_vector(state) for state in state_entry]
    if not vectors:
        raise ValueError("each path point must contain at least one eigenstate")
    return np.column_stack(vectors)


def track_logical_branches(
    reference_states,
    eigenstate_table,
    *,
    logical_labels=None,
    leakage_reference_states=(),
    leakage_labels=None,
    minimum_continuity_overlap: float = 0.5,
) -> LogicalBranchTrackingResult:
    """Track diabatic labels and phase-aligned adiabatic branches separately."""

    references = np.column_stack([_state_vector(state) for state in reference_states])
    if references.ndim != 2 or references.shape[1] == 0:
        raise ValueError("reference_states must contain at least one state")
    tables = [_eigenstate_columns(entry) for entry in eigenstate_table]
    if not tables:
        raise ValueError("eigenstate_table must contain at least one path point")
    dimension, branch_count = references.shape
    if any(table.shape[0] != dimension for table in tables):
        raise ValueError("all states must share the reference Hilbert-space dimension")
    if any(table.shape[1] < branch_count for table in tables):
        raise ValueError("each path point needs at least as many states as references")
    if (
        not np.isfinite(minimum_continuity_overlap)
        or not 0 <= minimum_continuity_overlap <= 1
    ):
        raise ValueError("minimum_continuity_overlap must lie in [0, 1]")
    logical_labels = tuple(
        LOGICAL_STATE_LABELS[:branch_count]
        if logical_labels is None and branch_count <= len(LOGICAL_STATE_LABELS)
        else (
            [f"logical_{index}" for index in range(branch_count)]
            if logical_labels is None
            else logical_labels
        )
    )
    if len(logical_labels) != branch_count:
        raise ValueError("logical_labels must match reference_states")

    leakage_vectors = [_state_vector(state) for state in leakage_reference_states]
    if leakage_labels is None:
        leakage_labels = [f"leakage_{index}" for index in range(len(leakage_vectors))]
    leakage_labels = tuple(map(str, leakage_labels))
    if len(leakage_labels) != len(leakage_vectors):
        raise ValueError("leakage_labels must match leakage_reference_states")
    if any(vector.size != dimension for vector in leakage_vectors):
        raise ValueError("leakage references must share the logical-state dimension")

    shape = (len(tables), branch_count)
    diabatic_level = np.empty(shape, dtype=int)
    diabatic_overlap = np.empty(shape)
    adiabatic_level = np.empty(shape, dtype=int)
    adiabatic_overlap = np.ones(shape)
    phase_alignment = np.zeros(shape)
    logical_weight = np.empty(shape)
    dominant_leakage_label = np.full(shape, "", dtype=object)
    dominant_leakage_probability = np.zeros(shape)

    aligned_previous = None
    for point_index, current in enumerate(tables):
        fixed_overlap = np.abs(references.conj().T @ current) ** 2
        rows, columns = linear_sum_assignment(-fixed_overlap)
        diabatic_level[point_index, rows] = columns
        diabatic_overlap[point_index, rows] = fixed_overlap[rows, columns]

        if point_index == 0:
            adiabatic_level[point_index] = diabatic_level[point_index]
            selected = current[:, adiabatic_level[point_index]].copy()
        else:
            neighbor_overlap = np.abs(aligned_previous.conj().T @ current) ** 2
            rows, columns = linear_sum_assignment(-neighbor_overlap)
            adiabatic_level[point_index, rows] = columns
            adiabatic_overlap[point_index, rows] = neighbor_overlap[rows, columns]
            selected = current[:, adiabatic_level[point_index]].copy()
            for branch_index in range(branch_count):
                overlap = np.vdot(
                    aligned_previous[:, branch_index], selected[:, branch_index]
                )
                correction = -float(np.angle(overlap)) if abs(overlap) > 0 else 0.0
                phase_alignment[point_index, branch_index] = correction
                selected[:, branch_index] *= np.exp(1j * correction)

        logical_weight[point_index] = np.sum(
            np.abs(references.conj().T @ selected) ** 2,
            axis=0,
        )
        if leakage_vectors:
            leakage_overlap = (
                np.abs(np.column_stack(leakage_vectors).conj().T @ selected) ** 2
            )
            dominant = np.argmax(leakage_overlap, axis=0)
            dominant_leakage_probability[point_index] = leakage_overlap[
                dominant, np.arange(branch_count)
            ]
            dominant_leakage_label[point_index] = np.asarray(leakage_labels)[dominant]
        aligned_previous = selected

    continuity_valid = bool(
        len(tables) == 1 or np.all(adiabatic_overlap[1:] >= minimum_continuity_overlap)
    )
    return LogicalBranchTrackingResult(
        diabatic_level=diabatic_level,
        diabatic_overlap=diabatic_overlap,
        adiabatic_level=adiabatic_level,
        adiabatic_overlap=adiabatic_overlap,
        phase_alignment_rad=phase_alignment,
        logical_weight=logical_weight,
        dominant_leakage_label=dominant_leakage_label,
        dominant_leakage_probability=dominant_leakage_probability,
        continuity_valid=continuity_valid,
        minimum_continuity_overlap=float(minimum_continuity_overlap),
        logical_labels=logical_labels,
    )


def track_dualrail_model_path(
    reference_model: DualRailFSimModel,
    target_configs,
    *,
    path_coordinate=None,
    minimum_continuity_overlap: float = 0.5,
) -> xr.Dataset:
    """Build instantaneous models and persist both logical label conventions."""

    target_configs = tuple(target_configs)
    if not target_configs:
        raise ValueError("target_configs must contain at least one path point")
    dimensions = tuple(
        config.truncated_dim for config in reference_model.config.subsystems()
    )
    leakage_states = []
    leakage_labels = []
    for flat_index in range(int(np.prod(dimensions))):
        levels = np.unravel_index(flat_index, dimensions)
        pair_a_count = levels[0] + levels[1]
        pair_b_count = levels[3] + levels[4]
        if pair_a_count == 1 and levels[2] == 0 and pair_b_count == 1:
            continue
        leakage_states.append(
            qt.tensor(
                [
                    qt.basis(dimension, level)
                    for dimension, level in zip(dimensions, levels, strict=True)
                ]
            )
        )
        leakage_labels.append(
            ",".join(
                f"{name}={level}"
                for name, level in zip(SUBSYSTEM_LABELS, levels, strict=True)
            )
        )

    reference_local_states = []
    for config, name in zip(
        reference_model.config.subsystems(), SUBSYSTEM_LABELS, strict=True
    ):
        subsystem = _make_transmon(config, f"path_reference_{name}")
        _, states = subsystem.eigensys(evals_count=config.truncated_dim)
        reference_local_states.append(np.asarray(states, dtype=complex))

    state_table = []
    energy_table = []
    for target_config in target_configs:
        if (
            tuple(item.truncated_dim for item in target_config.subsystems())
            != dimensions
        ):
            raise ValueError(
                "all target configs must use the reference truncated dimensions"
            )
        local_transforms = []
        for reference_config, target_config_item, reference_states, name in zip(
            reference_model.config.subsystems(),
            target_config.subsystems(),
            reference_local_states,
            SUBSYSTEM_LABELS,
            strict=True,
        ):
            if reference_config.ncut != target_config_item.ncut:
                raise ValueError(
                    "all target configs must use the reference ncut values"
                )
            subsystem = _make_transmon(target_config_item, f"path_target_{name}")
            _, states = subsystem.eigensys(evals_count=target_config_item.truncated_dim)
            local_transforms.append(
                qt.Qobj(reference_states.conj().T @ np.asarray(states, dtype=complex))
            )
        transform = qt.tensor(local_transforms)
        model = build_dualrail_fsim_model(target_config)
        state_table.append(
            np.column_stack(
                [(transform * state).full().ravel() for state in model.eigenstates]
            )
        )
        energy_table.append(model.eigenvalues_ghz)

    tracked = track_logical_branches(
        reference_model.logical_references,
        state_table,
        leakage_reference_states=leakage_states,
        leakage_labels=leakage_labels,
        minimum_continuity_overlap=minimum_continuity_overlap,
    )
    dataset = tracked.to_dataset(path_coordinate)
    energy_table = np.asarray(energy_table)
    dataset["diabatic_energy_ghz"] = (
        ("path_point", "logical_state"),
        np.take_along_axis(energy_table, tracked.diabatic_level, axis=1),
    )
    dataset["adiabatic_energy_ghz"] = (
        ("path_point", "logical_state"),
        np.take_along_axis(energy_table, tracked.adiabatic_level, axis=1),
    )
    dataset.diabatic_energy_ghz.attrs["units"] = "GHz"
    dataset.adiabatic_energy_ghz.attrs["units"] = "GHz"
    return dataset


def _validate_transmon(config: TunableTransmonConfig, name: str) -> None:
    if not np.isfinite(config.EJmax) or config.EJmax <= 0:
        raise ValueError(f"{name}.EJmax must be positive and finite")
    if not np.isfinite(config.EC) or config.EC <= 0:
        raise ValueError(f"{name}.EC must be positive and finite")
    if not np.isfinite(config.d) or not 0 <= config.d <= 1:
        raise ValueError(f"{name}.d must lie in [0, 1]")
    if not np.isfinite(config.flux) or not np.isfinite(config.ng):
        raise ValueError(f"{name}.flux and {name}.ng must be finite")
    if config.ncut < 1:
        raise ValueError(f"{name}.ncut must be at least 1")
    if config.truncated_dim < 2 or config.truncated_dim > 2 * config.ncut + 1:
        raise ValueError(f"{name}.truncated_dim must lie between 2 and 2*ncut + 1")


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


def _transmon_transition_ghz(config: TunableTransmonConfig) -> float:
    transmon = _make_transmon(config, "calibration")
    energies = np.asarray(transmon.eigenvals(evals_count=2), dtype=float)
    return float(energies[1] - energies[0])


def calibrate_reduced_pair(
    contour: LogicalContour,
    logical_frequency_mhz: float,
    *,
    ejmax_ghz: tuple[float, float] = (20.0, 20.0),
    asymmetry: tuple[float, float] = (0.22, 0.22),
    ng: tuple[float, float] = (0.0, 0.0),
    ncut: int = 15,
    truncated_dim: int = 3,
    ec_bounds_ghz: tuple[float, float] = (0.01, 2.0),
    coupling_bounds_ghz: tuple[float, float] = (0.0, 0.5),
) -> ReducedPairCalibration:
    """Match an independent-transmon pair to one point of a circuit contour.

    Each rail's charging energy is first fitted so its isolated transition equals
    the contour carrier.  The internal charge-charge coupling is then fitted to
    the requested logical splitting.  This is an anchor-point match, not a claim
    that the reduced model reproduces the full reference circuit.
    """

    logical_frequency_mhz = float(logical_frequency_mhz)
    phi_1, phi_2 = contour.flux(logical_frequency_mhz)
    carrier_ghz = float(contour.carrier_frequency(logical_frequency_mhz))
    if len(ejmax_ghz) != 2 or len(asymmetry) != 2 or len(ng) != 2:
        raise ValueError("ejmax_ghz, asymmetry, and ng must each contain two values")
    ec_lower, ec_upper = map(float, ec_bounds_ghz)
    if not 0 < ec_lower < ec_upper:
        raise ValueError("ec_bounds_ghz must be a positive increasing pair")
    coupling_lower, coupling_upper = map(float, coupling_bounds_ghz)
    if not 0 <= coupling_lower < coupling_upper:
        raise ValueError("coupling_bounds_ghz must be a nonnegative increasing pair")

    rails = []
    for index, (ejmax, d, offset_charge, flux) in enumerate(
        zip(ejmax_ghz, asymmetry, ng, (phi_1, phi_2), strict=True),
        start=1,
    ):

        def residual(
            ec: float,
            ejmax=ejmax,
            d=d,
            flux=flux,
            offset_charge=offset_charge,
        ) -> float:
            candidate = TunableTransmonConfig(
                EJmax=float(ejmax),
                EC=ec,
                d=float(d),
                flux=float(flux),
                ng=float(offset_charge),
                ncut=ncut,
                truncated_dim=truncated_dim,
            )
            return _transmon_transition_ghz(candidate) - carrier_ghz

        lower_residual = residual(ec_lower)
        upper_residual = residual(ec_upper)
        if lower_residual * upper_residual > 0:
            raise ValueError(
                f"rail {index} carrier {carrier_ghz:.6g} GHz is not bracketed by "
                f"ec_bounds_ghz={ec_bounds_ghz}"
            )
        ec = brentq(residual, ec_lower, ec_upper, xtol=1e-12, rtol=1e-12)
        rails.append(
            TunableTransmonConfig(
                EJmax=float(ejmax),
                EC=float(ec),
                d=float(d),
                flux=float(flux),
                ng=float(offset_charge),
                ncut=ncut,
                truncated_dim=truncated_dim,
            )
        )

    pair_energies = []
    pair_charges = []
    for index, rail in enumerate(rails, start=1):
        subsystem = _make_transmon(rail, f"rail_{index}")
        esys = subsystem.eigensys(evals_count=truncated_dim)
        pair_energies.append(np.asarray(esys[0], dtype=float))
        pair_charges.append(subsystem.n_operator(energy_esys=esys))

    def splitting_residual(coupling: float) -> float:
        modes = _isolated_pair_modes(
            (pair_energies[0], pair_energies[1]),
            (pair_charges[0], pair_charges[1]),
            coupling,
        )
        return modes.logical_splitting_mhz - logical_frequency_mhz

    lower_residual = splitting_residual(coupling_lower)
    upper_residual = splitting_residual(coupling_upper)
    if lower_residual * upper_residual > 0:
        raise ValueError(
            f"logical splitting {logical_frequency_mhz:.6g} MHz is not bracketed by "
            f"coupling_bounds_ghz={coupling_bounds_ghz}"
        )
    coupling = brentq(
        splitting_residual,
        coupling_lower,
        coupling_upper,
        xtol=1e-12,
        rtol=1e-12,
    )
    achieved = logical_frequency_mhz + splitting_residual(coupling)
    return ReducedPairCalibration(
        rail_1=rails[0],
        rail_2=rails[1],
        coupling_ghz=float(coupling),
        anchor_logical_frequency_mhz=logical_frequency_mhz,
        anchor_carrier_frequency_ghz=carrier_ghz,
        achieved_logical_frequency_mhz=float(achieved),
    )


def _embed(operator, subsystem: int, dimensions: tuple[int, ...]) -> qt.Qobj:
    factors = [qt.qeye(dimension) for dimension in dimensions]
    factors[subsystem] = (
        operator if isinstance(operator, qt.Qobj) else qt.Qobj(operator)
    )
    return qt.tensor(factors)


def _isolated_pair_modes(
    energies: tuple[np.ndarray, np.ndarray],
    charges: tuple[np.ndarray, np.ndarray],
    coupling_ghz: float,
) -> IsolatedPairModes:
    dimensions = (energies[0].size, energies[1].size)
    identity = tuple(qt.qeye(dimension) for dimension in dimensions)
    hamiltonian = qt.tensor(qt.Qobj(np.diag(energies[0])), identity[1])
    hamiltonian += qt.tensor(identity[0], qt.Qobj(np.diag(energies[1])))
    n1 = qt.tensor(qt.Qobj(charges[0]), identity[1])
    n2 = qt.tensor(identity[0], qt.Qobj(charges[1]))
    hamiltonian += coupling_ghz * n1 * n2
    pair_energies, pair_states = hamiltonian.eigenstates()
    pair_energies = np.asarray(pair_energies, dtype=float)

    bare_ground = qt.tensor(qt.basis(dimensions[0], 0), qt.basis(dimensions[1], 0))
    bare_10 = qt.tensor(qt.basis(dimensions[0], 1), qt.basis(dimensions[1], 0))
    bare_01 = qt.tensor(qt.basis(dimensions[0], 0), qt.basis(dimensions[1], 1))
    assignment, _ = assign_dressed_logical_states(
        (bare_ground, bare_10, bare_01), pair_states
    )
    ground = pair_states[assignment[0]]
    one_excitation_indices = assignment[1:]
    one_excitation_indices = one_excitation_indices[
        np.argsort(pair_energies[one_excitation_indices])
    ]
    lower, upper = (pair_states[index] for index in one_excitation_indices)
    mode_energies = pair_energies[one_excitation_indices]

    plus = (bare_10 + bare_01).unit()
    minus = (bare_10 - bare_01).unit()
    lower_character = (
        "+" if abs(plus.overlap(lower)) >= abs(minus.overlap(lower)) else "-"
    )
    raw_strength = np.asarray(
        [
            [abs(ground.overlap(operator * state)) ** 2 for state in (lower, upper)]
            for operator in (n1, n2)
        ]
    )
    totals = np.sum(raw_strength, axis=1, keepdims=True)
    participation = np.divide(
        raw_strength,
        totals,
        out=np.zeros_like(raw_strength),
        where=totals > 0,
    )
    transitions = mode_energies - pair_energies[assignment[0]]
    charge_matrix_element = np.asarray(
        [
            [abs(ground.overlap(operator * state)) for state in (lower, upper)]
            for operator in (n1, n2)
        ],
        dtype=float,
    )
    two_excitation_frequency = None
    if min(dimensions) >= 3:
        bare_two_excitation = (
            qt.tensor(qt.basis(dimensions[0], 2), qt.basis(dimensions[1], 0)),
            qt.tensor(qt.basis(dimensions[0], 1), qt.basis(dimensions[1], 1)),
            qt.tensor(qt.basis(dimensions[0], 0), qt.basis(dimensions[1], 2)),
        )
        two_assignment, _ = assign_dressed_logical_states(
            bare_two_excitation,
            pair_states,
        )
        two_excitation_frequency = (
            pair_energies[two_assignment] - pair_energies[assignment[0]]
        )
    return IsolatedPairModes(
        ground=ground,
        lower=lower,
        upper=upper,
        energies_ghz=mode_energies,
        logical_splitting_mhz=float(1e3 * np.diff(mode_energies)[0]),
        carrier_frequency_ghz=float(np.mean(transitions)),
        lower_character=lower_character,
        charge_participation=participation,
        charge_matrix_element=charge_matrix_element,
        two_excitation_frequency_ghz=two_excitation_frequency,
    )


def calibrate_reduced_pair_multi_point(
    contour: LogicalContour,
    *,
    anchor_logical_frequency_mhz: float | None = None,
    sample_logical_frequency_mhz=None,
    ejmax_ghz: tuple[float, float] = (20.0, 20.0),
    asymmetry: tuple[float, float] = (0.22, 0.22),
    ng: tuple[float, float] = (0.0, 0.0),
    ncut: int = 15,
    truncated_dim: int = 3,
    ec_bounds_ghz: tuple[float, float] = (0.01, 2.0),
    coupling_bounds_ghz: tuple[float, float] = (0.0, 0.5),
    residual_tolerance_fraction: float = 0.1,
) -> ReducedPairCalibration:
    """Fit one reduced pair to all requested points of a reference contour.

    Carrier and logical splitting are always fitted.  When the contour carries
    reference two-excitation frequencies or physical-rail charge matrix
    elements, those enter the same fit and residual report.
    """

    sample_frequency = (
        contour.logical_frequency_mhz.copy()
        if sample_logical_frequency_mhz is None
        else _finite_1d(
            sample_logical_frequency_mhz,
            "sample_logical_frequency_mhz",
            minimum_size=2,
        )
    )
    if np.any(np.diff(sample_frequency) <= 0):
        raise ValueError("sample_logical_frequency_mhz must be strictly increasing")
    contour.flux(sample_frequency)
    anchor = (
        float(sample_frequency[len(sample_frequency) // 2])
        if anchor_logical_frequency_mhz is None
        else float(anchor_logical_frequency_mhz)
    )
    contour.flux(anchor)
    if (
        not np.isfinite(residual_tolerance_fraction)
        or not 0 < residual_tolerance_fraction < 1
    ):
        raise ValueError("residual_tolerance_fraction must lie between 0 and 1")

    initial = calibrate_reduced_pair(
        contour,
        anchor,
        ejmax_ghz=ejmax_ghz,
        asymmetry=asymmetry,
        ng=ng,
        ncut=ncut,
        truncated_dim=truncated_dim,
        ec_bounds_ghz=ec_bounds_ghz,
        coupling_bounds_ghz=coupling_bounds_ghz,
    )
    reference_carrier = np.asarray(
        contour.carrier_frequency(sample_frequency), dtype=float
    )

    def interpolate_optional(values):
        if values is None:
            return None
        return np.asarray(
            PchipInterpolator(
                contour.logical_frequency_mhz,
                values,
                axis=0,
                extrapolate=False,
            )(sample_frequency),
            dtype=float,
        )

    reference_two = interpolate_optional(contour.two_excitation_frequency_ghz)
    reference_charge = interpolate_optional(contour.charge_matrix_element)
    if reference_two is not None and truncated_dim < 3:
        raise ValueError(
            "truncated_dim must be at least 3 to fit two-excitation frequencies"
        )

    phi_1, phi_2 = contour.flux(sample_frequency)

    def evaluate(parameters):
        ec_1, ec_2, coupling = map(float, parameters)
        carriers = []
        splittings = []
        two_excitation = []
        charge_elements = []
        for flux_1, flux_2 in zip(phi_1, phi_2, strict=True):
            rails = (
                TunableTransmonConfig(
                    EJmax=float(ejmax_ghz[0]),
                    EC=ec_1,
                    d=float(asymmetry[0]),
                    flux=float(flux_1),
                    ng=float(ng[0]),
                    ncut=ncut,
                    truncated_dim=truncated_dim,
                ),
                TunableTransmonConfig(
                    EJmax=float(ejmax_ghz[1]),
                    EC=ec_2,
                    d=float(asymmetry[1]),
                    flux=float(flux_2),
                    ng=float(ng[1]),
                    ncut=ncut,
                    truncated_dim=truncated_dim,
                ),
            )
            pair_energies = []
            pair_charges = []
            for index, rail in enumerate(rails):
                subsystem = _make_transmon(rail, f"fit_rail_{index + 1}")
                esys = subsystem.eigensys(evals_count=truncated_dim)
                pair_energies.append(np.asarray(esys[0], dtype=float))
                pair_charges.append(subsystem.n_operator(energy_esys=esys))
            modes = _isolated_pair_modes(
                (pair_energies[0], pair_energies[1]),
                (pair_charges[0], pair_charges[1]),
                coupling,
            )
            carriers.append(modes.carrier_frequency_ghz)
            splittings.append(modes.logical_splitting_mhz)
            charge_elements.append(modes.charge_matrix_element)
            if modes.two_excitation_frequency_ghz is not None:
                two_excitation.append(modes.two_excitation_frequency_ghz)
        return (
            np.asarray(carriers),
            np.asarray(splittings),
            None if not two_excitation else np.asarray(two_excitation),
            np.asarray(charge_elements),
        )

    relevant_scales = [float(np.min(np.abs(sample_frequency)))]
    if reference_two is not None:
        # The reference values are energies from the ground state in the
        # two-excitation manifold.  The logical |11> energy is the sum of the
        # lower and upper one-excitation modes, namely twice their carrier.
        leakage_gap = 1e3 * np.abs(reference_two - 2.0 * reference_carrier[:, None])
        positive_gap = leakage_gap[leakage_gap > np.finfo(float).eps]
        if positive_gap.size:
            relevant_scales.append(float(np.min(positive_gap)))
    residual_tolerance_mhz = residual_tolerance_fraction * min(relevant_scales)

    def core_objective(parameters):
        carrier, splitting, _, _ = evaluate(parameters)
        return np.concatenate(
            (
                1e3 * (carrier - reference_carrier),
                splitting - sample_frequency,
            )
        )

    # Spectral residuals use MHz throughout.  A 10 MHz scale for dimensionless
    # matrix elements keeps them relevant without overwhelming the spectrum.
    def objective(parameters):
        carrier, splitting, two_excitation, charge = evaluate(parameters)
        residuals = [1e3 * (carrier - reference_carrier), splitting - sample_frequency]
        if reference_two is not None:
            residuals.append(1e3 * (two_excitation - reference_two).ravel())
        if reference_charge is not None:
            residuals.append(10.0 * (charge - reference_charge).ravel())
        return np.concatenate([np.ravel(item) for item in residuals])

    lower = np.asarray([ec_bounds_ghz[0], ec_bounds_ghz[0], coupling_bounds_ghz[0]])
    upper = np.asarray([ec_bounds_ghz[1], ec_bounds_ghz[1], coupling_bounds_ghz[1]])
    fit_options = {
        "bounds": (lower, upper),
        "xtol": 1e-10,
        "ftol": 1e-10,
        "gtol": 1e-10,
        "max_nfev": 500,
    }
    core_fit = least_squares(
        core_objective,
        x0=np.asarray([initial.rail_1.EC, initial.rail_2.EC, initial.coupling_ghz]),
        **fit_options,
    )
    joint_fit = least_squares(objective, x0=core_fit.x, **fit_options)

    def summarize(parameters):
        reduced_carrier, reduced_splitting, reduced_two, reduced_charge = evaluate(
            parameters
        )
        carrier_residual = 1e3 * (reduced_carrier - reference_carrier)
        splitting_residual = reduced_splitting - sample_frequency
        two_residual = (
            None if reference_two is None else 1e3 * (reduced_two - reference_two)
        )
        spectral_residuals = [np.abs(carrier_residual), np.abs(splitting_residual)]
        if two_residual is not None:
            spectral_residuals.append(np.abs(two_residual).ravel())
        maximum_spectral_residual = float(
            np.max(np.concatenate([np.ravel(item) for item in spectral_residuals]))
        )
        charge_residual = (
            None if reference_charge is None else reduced_charge - reference_charge
        )
        charge_within_tolerance = bool(
            reference_charge is None
            or np.all(
                np.abs(charge_residual)
                <= residual_tolerance_fraction
                * np.maximum(np.abs(reference_charge), 1e-12)
            )
        )
        return (
            reduced_carrier,
            reduced_splitting,
            reduced_two,
            reduced_charge,
            carrier_residual,
            splitting_residual,
            two_residual,
            charge_residual,
            maximum_spectral_residual,
            bool(
                maximum_spectral_residual < residual_tolerance_mhz
                and charge_within_tolerance
            ),
        )

    joint_summary = summarize(joint_fit.x)
    if joint_summary[-1]:
        fit = joint_fit
        fit_selection = "joint"
        summary = joint_summary
    else:
        # A rejected rich fit must not corrupt the reduced surrogate consumed
        # by downstream diagnostics.  Preserve the best carrier/splitting fit;
        # its full residual report still fails and selects direct truth.
        fit = core_fit
        fit_selection = "core-preserving-fallback"
        summary = summarize(core_fit.x)
    (
        reduced_carrier,
        reduced_splitting,
        reduced_two,
        reduced_charge,
        carrier_residual,
        splitting_residual,
        two_residual,
        charge_residual,
        maximum_spectral_residual,
        residual_within_tolerance,
    ) = summary

    rail_1 = replace(initial.rail_1, EC=float(fit.x[0]))
    rail_2 = replace(initial.rail_2, EC=float(fit.x[1]))
    anchor_index = int(np.argmin(np.abs(sample_frequency - anchor)))
    return ReducedPairCalibration(
        rail_1=rail_1,
        rail_2=rail_2,
        coupling_ghz=float(fit.x[2]),
        anchor_logical_frequency_mhz=anchor,
        anchor_carrier_frequency_ghz=float(contour.carrier_frequency(anchor)),
        achieved_logical_frequency_mhz=float(reduced_splitting[anchor_index]),
        sample_logical_frequency_mhz=sample_frequency,
        reference_carrier_frequency_ghz=reference_carrier,
        reduced_carrier_frequency_ghz=reduced_carrier,
        reduced_logical_splitting_mhz=reduced_splitting,
        carrier_residual_mhz=carrier_residual,
        logical_splitting_residual_mhz=splitting_residual,
        reference_two_excitation_frequency_ghz=reference_two,
        reduced_two_excitation_frequency_ghz=(
            None if reference_two is None else reduced_two
        ),
        two_excitation_residual_mhz=two_residual,
        reference_charge_matrix_element=reference_charge,
        reduced_charge_matrix_element=(
            None if reference_charge is None else reduced_charge
        ),
        charge_matrix_element_residual=charge_residual,
        residual_tolerance_mhz=float(residual_tolerance_mhz),
        maximum_spectral_residual_mhz=maximum_spectral_residual,
        residual_within_tolerance=residual_within_tolerance,
        optimizer_success=bool(fit.success),
        fit_selection=fit_selection,
    )


def build_dualrail_fsim_model(
    config: FiveTransmonConfig | None = None,
) -> DualRailFSimModel:
    """Build the fixed-basis reduced five-transmon static model."""

    config = FiveTransmonConfig() if config is None else config
    config.logical_mapping.validate()
    if not 0 <= config.min_assignment_probability <= 1:
        raise ValueError("min_assignment_probability must lie in [0, 1]")
    if not np.isfinite(config.min_leakage_gap_mhz) or config.min_leakage_gap_mhz < 0:
        raise ValueError("min_leakage_gap_mhz must be nonnegative and finite")
    subsystem_configs = config.subsystems()
    for name, subsystem in zip(SUBSYSTEM_LABELS, subsystem_configs, strict=True):
        _validate_transmon(subsystem, name)
    coupling_values = [value for _, _, value in config.coupling.indexed_values()]
    if not np.all(np.isfinite(coupling_values)):
        raise ValueError("all coupling strengths must be finite")

    subsystems = tuple(
        _make_transmon(subsystem, name)
        for subsystem, name in zip(subsystem_configs, SUBSYSTEM_LABELS, strict=True)
    )
    dimensions = tuple(subsystem.truncated_dim for subsystem in subsystems)
    eigensystems = tuple(
        subsystem.eigensys(evals_count=subsystem.truncated_dim)
        for subsystem in subsystems
    )
    energies = tuple(np.asarray(esys[0], dtype=float) for esys in eigensystems)
    charges = tuple(
        subsystem.n_operator(energy_esys=esys)
        for subsystem, esys in zip(subsystems, eigensystems, strict=True)
    )
    zero = 0 * qt.tensor([qt.qeye(dimension) for dimension in dimensions])
    hamiltonian = sum(
        (
            _embed(np.diag(local_energies), index, dimensions)
            for index, local_energies in enumerate(energies)
        ),
        zero,
    )
    charge_operators = tuple(
        _embed(charge, index, dimensions) for index, charge in enumerate(charges)
    )
    for left, right, coupling in config.coupling.indexed_values():
        if coupling != 0:
            hamiltonian += coupling * charge_operators[left] * charge_operators[right]

    flux_operators = tuple(
        _embed(-subsystem.cos_phi_operator(energy_esys=esys), index, dimensions)
        for index, (subsystem, esys) in enumerate(
            zip(subsystems, eigensystems, strict=True)
        )
    )
    excitation_operators = tuple(
        _embed(np.diag(np.arange(dimension, dtype=float)), index, dimensions)
        for index, dimension in enumerate(dimensions)
    )

    pair_a = _isolated_pair_modes(
        (energies[0], energies[1]),
        (charges[0], charges[1]),
        config.coupling.a1_a2,
    )
    pair_b = _isolated_pair_modes(
        (energies[3], energies[4]),
        (charges[3], charges[4]),
        config.coupling.b1_b2,
    )
    pair_modes = []
    for pair in (pair_a, pair_b):
        modes = {"lower": pair.lower, "upper": pair.upper}
        pair_modes.append(
            (
                modes[config.logical_mapping.logical_zero],
                modes[config.logical_mapping.logical_one],
            )
        )
    coupler_ground = qt.basis(dimensions[2], 0)
    references = tuple(
        qt.tensor(pair_modes[0][bit_a], coupler_ground, pair_modes[1][bit_b])
        for bit_a, bit_b in ((0, 0), (0, 1), (1, 0), (1, 1))
    )

    eigenvalues, eigenstates = hamiltonian.eigenstates()
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    assignment, probability = assign_dressed_logical_states(references, eigenstates)
    logical_states = tuple(eigenstates[index] for index in assignment)
    return DualRailFSimModel(
        config=config,
        hamiltonian_ghz=hamiltonian,
        flux_operators_ghz=flux_operators,
        excitation_operators=excitation_operators,
        logical_references=references,
        logical_states=logical_states,
        eigenvalues_ghz=eigenvalues,
        eigenstates=tuple(eigenstates),
        logical_assignment=assignment,
        logical_assignment_probability=probability,
        pair_a_modes=pair_a,
        pair_b_modes=pair_b,
    )


def _gate_time_ns(rate_mhz: float) -> float:
    return float(
        np.inf if abs(rate_mhz) <= np.finfo(float).eps else 250.0 / abs(rate_mhz)
    )


def evaluate_static_interaction(model: DualRailFSimModel) -> StaticInteractionResult:
    """Compute the static quantities required by the early feasibility gate."""

    projected = np.asarray(
        [
            [
                complex(left.overlap(model.hamiltonian_ghz * right))
                for right in model.logical_references
            ]
            for left in model.logical_references
        ]
    )
    logical_energy = model.eigenvalues_ghz[model.logical_assignment]
    logical_energy = logical_energy - logical_energy[0]
    logical_overlap = np.asarray(
        [
            [complex(reference.overlap(state)) for state in model.logical_states]
            for reference in model.logical_references
        ]
    )
    logical_singular_values = np.linalg.svd(
        logical_overlap,
        compute_uv=False,
    )
    logical_frame = closest_unitary(logical_overlap)
    effective = logical_frame @ np.diag(logical_energy) @ logical_frame.conj().T
    effective_exchange_mhz = float(1e3 * abs(effective[1, 2]))
    effective_zz_mhz = float(
        1e3
        * np.real(effective[3, 3] - effective[2, 2] - effective[1, 1] + effective[0, 0])
    )
    zz_mhz = float(1e3 * (logical_energy[3] - logical_energy[2] - logical_energy[1]))
    detuning_mhz = float(
        model.pair_a_modes.logical_splitting_mhz
        - model.pair_b_modes.logical_splitting_mhz
    )
    odd_gap_mhz = float(1e3 * abs(logical_energy[2] - logical_energy[1]))
    radicand = odd_gap_mhz**2 - detuning_mhz**2
    exchange_mhz = 0.5 * np.sqrt(max(radicand, 0.0))
    direct_exchange_mhz = float(1e3 * abs(projected[1, 2]))

    unassigned_level = np.delete(
        np.arange(model.eigenvalues_ghz.size), model.logical_assignment
    )
    unassigned = model.eigenvalues_ghz[unassigned_level]
    absolute_logical_energy = model.eigenvalues_ghz[model.logical_assignment]
    leakage_distance = np.abs(
        absolute_logical_energy[:, None] - unassigned[None, :]
    )
    logical_index, leakage_index = np.unravel_index(
        int(np.argmin(leakage_distance)), leakage_distance.shape
    )
    nearest_gap_mhz = float(1e3 * leakage_distance[logical_index, leakage_index])
    nearest_leakage_level = int(unassigned_level[leakage_index])
    nearest_leakage_state = model.eigenstates[nearest_leakage_level]
    dimensions = tuple(item.truncated_dim for item in model.config.subsystems())
    bare_probability = np.abs(nearest_leakage_state.full().ravel()) ** 2
    dominant_bare_index = int(np.argmax(bare_probability))
    dominant_levels = np.unravel_index(dominant_bare_index, dimensions)
    nearest_leakage_label = ",".join(
        f"{name}={level}"
        for name, level in zip(SUBSYSTEM_LABELS, dominant_levels, strict=True)
    )
    coupler_participation = np.asarray(
        [
            qt.expect(model.excitation_operators[2], state)
            for state in model.logical_states
        ],
        dtype=float,
    )
    pair_a_occupation = np.asarray(
        [
            qt.expect(
                model.excitation_operators[0] + model.excitation_operators[1], state
            )
            for state in model.logical_states
        ],
        dtype=float,
    )
    pair_b_occupation = np.asarray(
        [
            qt.expect(
                model.excitation_operators[3] + model.excitation_operators[4], state
            )
            for state in model.logical_states
        ],
        dtype=float,
    )
    return StaticInteractionResult(
        projected_hamiltonian_ghz=projected,
        effective_hamiltonian_ghz=effective,
        logical_energy_ghz=logical_energy,
        logical_assignment=model.logical_assignment.copy(),
        logical_assignment_probability=model.logical_assignment_probability.copy(),
        logical_singular_values=logical_singular_values,
        logical_subspace_sigma_min=float(np.min(logical_singular_values)),
        logical_detuning_mhz=detuning_mhz,
        direct_exchange_mhz=direct_exchange_mhz,
        effective_exchange_mhz=effective_exchange_mhz,
        logical_exchange_mhz=float(exchange_mhz),
        effective_zz_mhz=effective_zz_mhz,
        logical_zz_mhz=zz_mhz,
        nearest_leakage_gap_mhz=nearest_gap_mhz,
        nearest_leakage_level=nearest_leakage_level,
        nearest_leakage_logical_state=LOGICAL_STATE_LABELS[logical_index],
        nearest_leakage_branch_label=nearest_leakage_label,
        nearest_leakage_branch_probability=float(bare_probability[dominant_bare_index]),
        coupler_participation=coupler_participation,
        pair_a_occupation=pair_a_occupation,
        pair_b_occupation=pair_b_occupation,
        estimated_iswap_time_ns=_gate_time_ns(exchange_mhz),
        estimated_cz_time_ns=float(
            np.inf if abs(zz_mhz) <= np.finfo(float).eps else 500.0 / abs(zz_mhz)
        ),
        logical_assignment_valid=bool(
            np.min(logical_singular_values) >= model.config.min_assignment_probability
        ),
        spectrally_isolated=bool(nearest_gap_mhz >= model.config.min_leakage_gap_mhz),
    )


def extract_fsim_angles(unitary: np.ndarray) -> tuple[float, float, float]:
    """Extract principal fSim angles while ignoring global and local-Z phases."""

    unitary = np.asarray(unitary, dtype=complex)
    if unitary.shape != (4, 4):
        raise ValueError("unitary must have shape (4, 4)")
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


def compare_static_and_dynamic_rates(
    duration_ns,
    projected_unitary,
    *,
    static_hamiltonian_ghz,
    relative_tolerance: float = 0.05,
    normalization_rate_mhz: float | None = None,
) -> StaticDynamicRateComparison:
    """Fit exchange and ZZ rates while retaining static nuisance terms.

    The complete static logical Hamiltonian is the prediction template.  Only
    its odd-manifold exchange and conditional-phase coefficients are scaled in
    the fit; detuning, local terms, and other coherent nuisance terms remain in
    the model.  This avoids interpreting detuned, non-fSim evolution with the
    pure-fSim swap and phase formulas.
    """

    duration = _finite_1d(duration_ns, "duration_ns", minimum_size=3)
    if np.any(np.diff(duration) <= 0):
        raise ValueError("duration_ns must be strictly increasing")
    unitary = np.asarray(projected_unitary, dtype=complex)
    if unitary.shape != (duration.size, 4, 4):
        raise ValueError("projected_unitary must have shape (duration, 4, 4)")
    static_hamiltonian = np.asarray(static_hamiltonian_ghz, dtype=complex)
    if static_hamiltonian.shape != (4, 4):
        raise ValueError("static_hamiltonian_ghz must have shape (4, 4)")
    if not np.all(np.isfinite(static_hamiltonian)):
        raise ValueError("static_hamiltonian_ghz must be finite")
    if not np.allclose(static_hamiltonian, static_hamiltonian.conj().T):
        raise ValueError("static_hamiltonian_ghz must be Hermitian")
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must lie between 0 and 1")
    if normalization_rate_mhz is not None and (
        not np.isfinite(normalization_rate_mhz) or normalization_rate_mhz <= 0
    ):
        raise ValueError("normalization_rate_mhz must be positive and finite")

    static_exchange = float(1e3 * abs(static_hamiltonian[1, 2]))
    static_zz = float(
        1e3
        * np.real(
            static_hamiltonian[3, 3]
            - static_hamiltonian[2, 2]
            - static_hamiltonian[1, 1]
            + static_hamiltonian[0, 0]
        )
    )

    def predicted_unitaries(scales):
        candidate = static_hamiltonian.copy()
        candidate[1, 2] *= scales[0]
        candidate[2, 1] *= scales[0]
        candidate[3, 3] += (scales[1] - 1.0) * static_zz / 1e3
        return np.asarray([expm(-2j * np.pi * candidate * time) for time in duration])

    def residual(scales):
        predicted = predicted_unitaries(scales)
        aligned = []
        for expected, observed in zip(predicted, unitary, strict=True):
            phase_overlap = np.vdot(expected, observed)
            phase = (
                1.0
                if abs(phase_overlap) < 1e-15
                else np.exp(1j * np.angle(phase_overlap))
            )
            difference = phase * expected - observed
            aligned.append(difference.ravel())
        aligned = np.concatenate(aligned)
        return np.concatenate((aligned.real, aligned.imag))

    fit = least_squares(
        residual,
        x0=np.ones(2),
        bounds=(np.full(2, 0.25), np.full(2, 2.0)),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    dynamic_exchange = float(fit.x[0] * static_exchange)
    dynamic_zz = float(fit.x[1] * static_zz)

    def relative_error(actual: float, expected: float) -> float:
        denominator = abs(expected)
        if normalization_rate_mhz is not None:
            denominator = max(denominator, float(normalization_rate_mhz))
        if denominator <= np.finfo(float).eps:
            return 0.0 if abs(actual) <= 1e-9 else np.inf
        return abs(actual - expected) / denominator

    exchange_error = relative_error(dynamic_exchange, static_exchange)
    zz_error = relative_error(dynamic_zz, static_zz)
    return StaticDynamicRateComparison(
        static_exchange_mhz=static_exchange,
        dynamic_exchange_mhz=dynamic_exchange,
        exchange_relative_error=float(exchange_error),
        static_zz_mhz=static_zz,
        dynamic_zz_mhz=float(dynamic_zz),
        zz_relative_error=float(zz_error),
        relative_tolerance=float(relative_tolerance),
        normalization_rate_mhz=(
            np.nan if normalization_rate_mhz is None else float(normalization_rate_mhz)
        ),
        rates_agree=bool(
            fit.success
            and exchange_error <= relative_tolerance
            and zz_error <= relative_tolerance
        ),
    )


def closest_unitary(matrix: np.ndarray) -> np.ndarray:
    """Return the unitary polar factor of a square complex matrix."""

    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    left, _, right = np.linalg.svd(matrix)
    return left @ right


def average_gate_fidelity(actual: np.ndarray, target: np.ndarray) -> float:
    """Average code-space fidelity, retaining loss when ``actual`` is nonunitary."""

    actual = np.asarray(actual, dtype=complex)
    target = np.asarray(target, dtype=complex)
    if actual.ndim != 2 or actual.shape[0] != actual.shape[1]:
        raise ValueError("actual must be square")
    if target.shape != actual.shape:
        raise ValueError("actual and target must have the same shape")
    dimension = actual.shape[0]
    overlap = np.trace(target.conj().T @ actual)
    survival = np.trace(actual.conj().T @ actual).real
    fidelity = (abs(overlap) ** 2 + survival) / (dimension * (dimension + 1))
    return float(np.clip(fidelity, 0.0, 1.0))


def pauli_decompose(matrix: np.ndarray) -> dict[str, complex]:
    """Return the two-logical-qubit Pauli coefficients of a 4x4 matrix."""

    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4)")
    paulis = (
        np.eye(2, dtype=complex),
        np.asarray([[0, 1], [1, 0]], dtype=complex),
        np.asarray([[0, -1j], [1j, 0]], dtype=complex),
        np.asarray([[1, 0], [0, -1]], dtype=complex),
    )
    return {
        term: complex(np.trace(np.kron(left, right).conj().T @ matrix) / 4)
        for term, left, right in zip(
            PAULI_TERMS,
            (left for left in paulis for _ in paulis),
            (right for _ in paulis for right in paulis),
            strict=True,
        )
    }


def _local_z_matrix(angle_a: float, angle_b: float) -> np.ndarray:
    return np.diag(
        np.exp(
            -0.5j
            * np.asarray(
                [
                    angle_a + angle_b,
                    angle_a - angle_b,
                    -angle_a + angle_b,
                    -angle_a - angle_b,
                ]
            )
        )
    )


def _apply_local_z(matrix: np.ndarray, angles: np.ndarray) -> np.ndarray:
    before = _local_z_matrix(float(angles[0]), float(angles[1]))
    after = _local_z_matrix(float(angles[2]), float(angles[3]))
    return after @ matrix @ before


def local_z_optimized_fidelity(
    actual: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Optimize pre/post logical-Z rotations and return fidelity and four angles."""

    actual = np.asarray(actual, dtype=complex)
    target = np.asarray(target, dtype=complex)
    if actual.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError("actual and target must both have shape (4, 4)")

    def objective(angles: np.ndarray) -> float:
        return -average_gate_fidelity(_apply_local_z(actual, angles), target)

    starts = (
        np.zeros(4),
        np.full(4, np.pi / 2),
        np.asarray([np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2]),
    )
    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            bounds=[(-np.pi, np.pi)] * 4,
            options={"xtol": 1e-10, "ftol": 1e-12, "maxiter": 2_000},
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not best.success:
        raise RuntimeError("failed to optimize local logical-Z corrections")
    angles = (np.asarray(best.x) + np.pi) % (2 * np.pi) - np.pi
    return float(-best.fun), angles


def evaluate_static_coherent_ceiling(
    logical_hamiltonian_ghz: np.ndarray,
    gate_type: Literal["iswap", "cz"],
    *,
    maximum_duration_ns: float,
    time_samples: int = 101,
    target_cphase_rad: float = np.pi,
) -> StaticCoherentCeilingResult:
    """Maximize coherent target fidelity under a complete static Hamiltonian.

    The search uses the full numerical 4x4 logical Hamiltonian, including every
    Pauli nuisance term.  A vectorized bounded grid supplies starts for a joint
    hold-time/local-Z refinement, so this remains inexpensive enough to run at
    every atlas point.
    """

    hamiltonian = np.asarray(logical_hamiltonian_ghz, dtype=complex)
    if hamiltonian.shape != (4, 4):
        raise ValueError("logical_hamiltonian_ghz must have shape (4, 4)")
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("logical_hamiltonian_ghz must be finite")
    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12):
        raise ValueError("logical_hamiltonian_ghz must be Hermitian")
    if gate_type not in ("iswap", "cz"):
        raise ValueError("gate_type must be 'iswap' or 'cz'")
    if not np.isfinite(maximum_duration_ns) or maximum_duration_ns <= 0:
        raise ValueError("maximum_duration_ns must be positive and finite")
    if (
        not isinstance(time_samples, (int, np.integer))
        or isinstance(time_samples, bool)
        or time_samples < 3
    ):
        raise ValueError("time_samples must be an integer of at least 3")
    if not np.isfinite(target_cphase_rad):
        raise ValueError("target_cphase_rad must be finite")

    target = (
        ideal_fsim(np.pi / 2, 0.0)
        if gate_type == "iswap"
        else ideal_fsim(0.0, target_cphase_rad)
    )
    energies, states = np.linalg.eigh(hamiltonian)

    def unitary(duration_ns: float) -> np.ndarray:
        phases = np.exp(-2j * np.pi * energies * duration_ns)
        return (states * phases) @ states.conj().T

    if gate_type == "iswap":
        support = ((0, 0), (1, 2), (2, 1), (3, 3))
        phase_sign = ((-1, 0), (0, 1), (0, -1), (1, 0))
    else:
        support = ((0, 0), (1, 1), (2, 2), (3, 3))
        phase_sign = ((-1, 0), (0, -1), (0, 1), (1, 0))

    def supported_coefficients(duration_ns: float) -> np.ndarray:
        actual = unitary(duration_ns)
        return np.asarray(
            [target[row, column].conjugate() * actual[row, column] for row, column in support]
        )

    def fidelity_at(parameters: np.ndarray) -> float:
        duration_ns, phase_x, phase_y = parameters
        coefficients = supported_coefficients(float(duration_ns))
        rotations = np.exp(
            1j
            * np.asarray(
                [x_sign * phase_x + y_sign * phase_y for x_sign, y_sign in phase_sign]
            )
        )
        overlap = np.sum(coefficients * rotations)
        return float(np.clip((abs(overlap) ** 2 + 4.0) / 20.0, 0.0, 1.0))

    durations = np.linspace(0.0, maximum_duration_ns, int(time_samples))
    phase_grid = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    coarse = []
    rotations = np.asarray(
        [
            [
                np.exp(
                    1j
                    * np.asarray(
                        [
                            x_sign * phase_x + y_sign * phase_y
                            for x_sign, y_sign in phase_sign
                        ]
                    )
                )
                for phase_y in phase_grid
            ]
            for phase_x in phase_grid
        ]
    )
    for duration_ns in durations:
        overlaps = rotations @ supported_coefficients(float(duration_ns))
        flat_index = int(np.argmax(np.abs(overlaps)))
        x_index, y_index = np.unravel_index(flat_index, overlaps.shape)
        coarse.append(
            (
                float((abs(overlaps[x_index, y_index]) ** 2 + 4.0) / 20.0),
                float(duration_ns),
                float(phase_grid[x_index]),
                float(phase_grid[y_index]),
            )
        )

    best_parameters = np.asarray(max(coarse, key=lambda item: item[0])[1:])
    starts = sorted(coarse, reverse=True)[: min(8, len(coarse))]
    best_fidelity = fidelity_at(best_parameters)
    bounds = ((0.0, float(maximum_duration_ns)), (-np.pi, np.pi), (-np.pi, np.pi))
    for _, duration_ns, phase_x, phase_y in starts:
        optimized = minimize(
            lambda parameters: -fidelity_at(parameters),
            np.asarray([duration_ns, phase_x, phase_y]),
            method="Powell",
            bounds=bounds,
            options={"xtol": 1e-9, "ftol": 1e-12, "maxiter": 500},
        )
        candidate_fidelity = fidelity_at(optimized.x)
        if candidate_fidelity > best_fidelity:
            best_fidelity = candidate_fidelity
            best_parameters = np.asarray(optimized.x)

    duration_ns = float(best_parameters[0])
    fidelity, virtual_z = local_z_optimized_fidelity(unitary(duration_ns), target)
    return StaticCoherentCeilingResult(
        gate_type=gate_type,
        fidelity=fidelity,
        duration_ns=duration_ns,
        virtual_z_rad=virtual_z,
    )


def _effective_ej(config: TunableTransmonConfig, flux: float | np.ndarray):
    flux = np.asarray(flux, dtype=float)
    return config.EJmax * np.sqrt(
        np.cos(np.pi * flux) ** 2 + config.d**2 * np.sin(np.pi * flux) ** 2
    )


def validate_fixed_basis_along_path(
    reference_model: DualRailFSimModel,
    target_configs,
    *,
    path_coordinate=None,
    levels: int = 24,
    minimum_state_overlap: float = 0.5,
    tolerance_fraction: float = 0.1,
    relevant_rate_mhz: float | None = None,
) -> xr.Dataset:
    """Validate one fixed pulse basis against instantaneous models along a path.

    Instantaneous eigenstates are transformed into the reference subsystem
    basis before overlap matching.  Logical energies and the nearest logical-to-
    leakage gap are therefore compared by physical state identity, not by
    energy-order index.
    """

    target_configs = tuple(target_configs)
    if not target_configs:
        raise ValueError("target_configs must contain at least one path point")
    if (
        isinstance(levels, bool)
        or not isinstance(levels, (int, np.integer))
        or levels < 1
    ):
        raise ValueError("levels must be an integer of at least 1")
    if not 0 <= minimum_state_overlap <= 1:
        raise ValueError("minimum_state_overlap must lie in [0, 1]")
    if not np.isfinite(tolerance_fraction) or not 0 < tolerance_fraction < 1:
        raise ValueError("tolerance_fraction must lie between 0 and 1")
    if relevant_rate_mhz is not None and (
        not np.isfinite(relevant_rate_mhz) or relevant_rate_mhz <= 0
    ):
        raise ValueError("relevant_rate_mhz must be positive and finite")

    point_count = len(target_configs)
    coordinate = (
        np.arange(point_count, dtype=float)
        if path_coordinate is None
        else np.asarray(path_coordinate)
    )
    if coordinate.shape != (point_count,):
        raise ValueError("path_coordinate must contain one value per target config")
    dimension = reference_model.hamiltonian_ghz.shape[0]
    level_count = min(int(levels), dimension)
    matched_level = np.empty((point_count, level_count), dtype=int)
    state_overlap = np.empty((point_count, level_count))
    instantaneous_energy = np.empty((point_count, level_count))
    fixed_energy = np.empty((point_count, level_count))
    logical_energy_error = np.empty((point_count, len(LOGICAL_STATE_LABELS)))
    target_leakage_gap = np.empty(point_count)
    fixed_leakage_gap = np.empty(point_count)
    relevant_rate_scale = np.empty(point_count)

    reference_local_states = []
    for subsystem_config, name in zip(
        reference_model.config.subsystems(), SUBSYSTEM_LABELS, strict=True
    ):
        subsystem = _make_transmon(subsystem_config, f"reference_{name}")
        _, states = subsystem.eigensys(evals_count=subsystem_config.truncated_dim)
        reference_local_states.append(np.asarray(states, dtype=complex))

    for point_index, target_config in enumerate(target_configs):
        target_model = build_dualrail_fsim_model(target_config)
        if target_model.hamiltonian_ghz.shape != reference_model.hamiltonian_ghz.shape:
            raise ValueError(
                "all target configs must use the reference truncated dimensions"
            )
        fixed_hamiltonian = reference_model.hamiltonian_ghz.copy()
        local_transforms = []
        for (
            reference_config,
            target_subsystem_config,
            operator,
            reference_states,
            name,
        ) in zip(
            reference_model.config.subsystems(),
            target_config.subsystems(),
            reference_model.flux_operators_ghz,
            reference_local_states,
            SUBSYSTEM_LABELS,
            strict=True,
        ):
            if reference_config.ncut != target_subsystem_config.ncut:
                raise ValueError(
                    "all target configs must use the reference ncut values"
                )
            fixed_hamiltonian += (
                _effective_ej(reference_config, target_subsystem_config.flux)
                - _effective_ej(reference_config, reference_config.flux)
            ) * operator
            target_subsystem = _make_transmon(target_subsystem_config, f"target_{name}")
            _, target_states = target_subsystem.eigensys(
                evals_count=target_subsystem_config.truncated_dim
            )
            local_overlap = reference_states.conj().T @ np.asarray(
                target_states, dtype=complex
            )
            local_transforms.append(qt.Qobj(local_overlap))
        basis_transform = qt.tensor(local_transforms)
        transformed_target_states = tuple(
            basis_transform * state for state in target_model.eigenstates
        )
        fixed_values, fixed_states = fixed_hamiltonian.eigenstates()
        fixed_values = np.asarray(fixed_values, dtype=float)

        logical_indices = target_model.logical_assignment
        unassigned = np.setdiff1d(np.arange(dimension), logical_indices)
        absolute_logical = target_model.eigenvalues_ghz[logical_indices]
        gap_table = np.abs(
            absolute_logical[:, None]
            - target_model.eigenvalues_ghz[unassigned][None, :]
        )
        logical_gap_row, leakage_gap_column = np.unravel_index(
            int(np.argmin(gap_table)), gap_table.shape
        )
        leakage_index = int(unassigned[leakage_gap_column])
        target_leakage_gap[point_index] = (
            1e3 * gap_table[logical_gap_row, leakage_gap_column]
        )

        relevant_indices = np.unique(
            np.concatenate(
                (
                    np.arange(level_count),
                    logical_indices,
                    [leakage_index],
                )
            )
        )
        overlap = np.asarray(
            [
                [
                    abs(target_state.overlap(fixed_state)) ** 2
                    for fixed_state in fixed_states
                ]
                for target_state in (
                    transformed_target_states[index] for index in relevant_indices
                )
            ]
        )
        rows, columns = linear_sum_assignment(-overlap)
        mapping = {
            int(relevant_indices[row]): int(column)
            for row, column in zip(rows, columns, strict=True)
        }
        overlap_mapping = {
            int(relevant_indices[row]): float(overlap[row, column])
            for row, column in zip(rows, columns, strict=True)
        }
        low_indices = np.arange(level_count)
        matched_level[point_index] = [mapping[int(index)] for index in low_indices]
        state_overlap[point_index] = [
            overlap_mapping[int(index)] for index in low_indices
        ]
        target_relative = target_model.eigenvalues_ghz - target_model.eigenvalues_ghz[0]
        fixed_ground = fixed_values[mapping[0]]
        fixed_relative = fixed_values - fixed_ground
        instantaneous_energy[point_index] = 1e3 * target_relative[low_indices]
        fixed_energy[point_index] = 1e3 * fixed_relative[matched_level[point_index]]
        logical_energy_error[point_index] = 1e3 * np.asarray(
            [
                fixed_relative[mapping[int(index)]] - target_relative[int(index)]
                for index in logical_indices
            ]
        )
        fixed_leakage_gap[point_index] = 1e3 * abs(
            fixed_values[mapping[int(logical_indices[logical_gap_row])]]
            - fixed_values[mapping[leakage_index]]
        )
        if relevant_rate_mhz is None:
            static = target_model.static_interaction()
            rate_scales = (
                static.effective_exchange_mhz,
                static.effective_zz_mhz,
            )
        else:
            rate_scales = (relevant_rate_mhz,)
        positive_scales = np.abs([*rate_scales, target_leakage_gap[point_index]])
        positive_scales = positive_scales[positive_scales > np.finfo(float).eps]
        relevant_rate_scale[point_index] = (
            float(np.min(positive_scales)) if positive_scales.size else np.inf
        )

    energy_error = fixed_energy - instantaneous_energy
    leakage_gap_error = fixed_leakage_gap - target_leakage_gap
    tolerance_mhz = tolerance_fraction * float(np.min(relevant_rate_scale))
    maximum_relevant_error = float(
        max(
            np.max(np.abs(logical_energy_error)),
            np.max(np.abs(leakage_gap_error)),
        )
    )
    closest_point = int(np.argmin(target_leakage_gap))
    return xr.Dataset(
        data_vars={
            "instantaneous_level": ("matched_level", np.arange(level_count)),
            "fixed_level": (("path_point", "matched_level"), matched_level),
            "state_overlap": (("path_point", "matched_level"), state_overlap),
            "instantaneous_energy_mhz": (
                ("path_point", "matched_level"),
                instantaneous_energy,
            ),
            "fixed_energy_mhz": (("path_point", "matched_level"), fixed_energy),
            "energy_error_mhz": (("path_point", "matched_level"), energy_error),
            "logical_energy_error_mhz": (
                ("path_point", "logical_state"),
                logical_energy_error,
            ),
            "instantaneous_leakage_gap_mhz": ("path_point", target_leakage_gap),
            "fixed_leakage_gap_mhz": ("path_point", fixed_leakage_gap),
            "leakage_gap_error_mhz": ("path_point", leakage_gap_error),
            "relevant_rate_scale_mhz": ("path_point", relevant_rate_scale),
        },
        coords={
            "path_point": coordinate,
            "matched_level": np.arange(level_count),
            "logical_state": list(LOGICAL_STATE_LABELS),
        },
        attrs={
            "overlap_matching": "instantaneous-to-fixed",
            "minimum_state_overlap": float(minimum_state_overlap),
            "minimum_observed_overlap": float(np.min(state_overlap)),
            "state_overlap_valid": int(np.min(state_overlap) >= minimum_state_overlap),
            "tolerance_fraction": float(tolerance_fraction),
            "relevant_rate_mhz": (
                np.nan if relevant_rate_mhz is None else float(relevant_rate_mhz)
            ),
            "error_tolerance_mhz": tolerance_mhz,
            "maximum_relevant_error_mhz": maximum_relevant_error,
            "fixed_basis_valid": int(maximum_relevant_error < tolerance_mhz),
            "closest_leakage_approach_path_point": coordinate[closest_point].item(),
        },
    )


def _pulse_waveform_builder(
    model: DualRailFSimModel,
    pulse: DualRailPulse,
    contour_a: LogicalContour,
    contour_b: LogicalContour,
    constraints: DualRailPulseConstraints,
):
    pulse.validate()
    constraints.validate()
    idle_a = contour_a.inverse_contour(
        model.config.rail_a1.flux,
        model.config.rail_a2.flux,
    )
    idle_b = contour_b.inverse_contour(
        model.config.rail_b1.flux,
        model.config.rail_b2.flux,
    )
    recovered_a = np.asarray(contour_a.flux(idle_a), dtype=float)
    recovered_b = np.asarray(contour_b.flux(idle_b), dtype=float)
    configured_a = np.asarray([model.config.rail_a1.flux, model.config.rail_a2.flux])
    configured_b = np.asarray([model.config.rail_b1.flux, model.config.rail_b2.flux])
    if np.max(np.abs(recovered_a - configured_a)) > constraints.contour_match_tolerance:
        raise ValueError("pair A idle fluxes do not lie on contour_a")
    if np.max(np.abs(recovered_b - configured_b)) > constraints.contour_match_tolerance:
        raise ValueError("pair B idle fluxes do not lie on contour_b")

    peak_a = (
        idle_a
        if pulse.logical_frequency_a_peak_mhz is None
        else float(pulse.logical_frequency_a_peak_mhz)
    )
    peak_b = (
        idle_b
        if pulse.logical_frequency_b_peak_mhz is None
        else float(pulse.logical_frequency_b_peak_mhz)
    )
    contour_a.flux(peak_a)
    contour_b.flux(peak_b)
    phi_a1_interpolator = contour_a._interpolator(contour_a.phi_1)
    phi_a2_interpolator = contour_a._interpolator(contour_a.phi_2)
    phi_b1_interpolator = contour_b._interpolator(contour_b.phi_1)
    phi_b2_interpolator = contour_b._interpolator(contour_b.phi_2)

    def waveforms(time_ns: float | np.ndarray):
        envelope = pulse.envelope(time_ns)
        frequency_a = idle_a + envelope * (peak_a - idle_a)
        frequency_b = idle_b + envelope * (peak_b - idle_b)
        phi_a1 = phi_a1_interpolator(frequency_a)
        phi_a2 = phi_a2_interpolator(frequency_a)
        phi_b1 = phi_b1_interpolator(frequency_b)
        phi_b2 = phi_b2_interpolator(frequency_b)
        coupler_flux = model.config.coupler.flux + envelope * (
            pulse.coupler_peak_flux - model.config.coupler.flux
        )
        flux = np.stack((phi_a1, phi_a2, coupler_flux, phi_b1, phi_b2), axis=-1)
        return frequency_a, frequency_b, flux

    validation_time = pulse.padding_ns + np.linspace(
        0.0,
        pulse.duration_ns,
        constraints.validation_samples,
    )
    if pulse.padding_ns > 0:
        validation_time = np.unique(
            np.concatenate(
                (
                    [0.0],
                    validation_time,
                    [pulse.total_duration_ns],
                )
            )
        )
    _, _, validation_flux = waveforms(validation_time)
    for index, (name, bounds) in enumerate(
        zip(SUBSYSTEM_LABELS, constraints.flux_bounds, strict=True)
    ):
        lower, upper = bounds
        if np.any(
            (validation_flux[:, index] < lower) | (validation_flux[:, index] > upper)
        ):
            raise ValueError(f"{name} pulse leaves configured flux bounds {bounds}")

    changed = np.ptp(validation_flux, axis=0) > 1e-14
    if pulse.rise_time_ns == 0:
        max_slew = np.where(changed, np.inf, 0.0)
    else:
        max_slew = np.max(
            np.abs(np.gradient(validation_flux, validation_time, axis=0)),
            axis=0,
        )
    if constraints.max_flux_slew_per_ns is not None:
        allowed_slew = np.asarray(constraints.max_flux_slew_per_ns, dtype=float)
        exceeded = np.flatnonzero(max_slew > allowed_slew)
        if exceeded.size:
            names = [SUBSYSTEM_LABELS[index] for index in exceeded]
            raise ValueError(f"pulse exceeds maximum flux slew for {names}")
    return waveforms, max_slew, idle_a, idle_b, peak_a, peak_b


def _pulse_time_grid(pulse: DualRailPulse) -> np.ndarray:
    """Build an output grid that includes both active-pulse boundaries."""

    time_ns = np.linspace(0.0, pulse.total_duration_ns, pulse.output_samples)
    if pulse.padding_ns == 0:
        return time_ns
    return np.unique(
        np.concatenate(
            (
                time_ns,
                [pulse.padding_ns, pulse.padding_ns + pulse.duration_ns],
            )
        )
    )


def _static_evolution(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    initial_vector: np.ndarray,
    duration_ns: float | np.ndarray,
) -> np.ndarray:
    """Evolve a state vector for one or more durations under a static Hamiltonian."""

    duration = np.atleast_1d(np.asarray(duration_ns, dtype=float))
    amplitudes = eigenvectors.conj().T @ initial_vector
    phases = np.exp(-2j * np.pi * np.asarray(eigenvalues)[:, None] * duration[None, :])
    return eigenvectors @ (amplitudes[:, None] * phases)


def _pair_one_excitation_projector(dimension_1: int, dimension_2: int) -> qt.Qobj:
    diagonal = np.asarray(
        [
            float(level_1 + level_2 == 1)
            for level_1 in range(dimension_1)
            for level_2 in range(dimension_2)
        ]
    )
    return qt.Qobj(
        np.diag(diagonal),
        dims=[[dimension_1, dimension_2], [dimension_1, dimension_2]],
    )


def _pulse_fingerprint(
    model: DualRailFSimModel,
    pulse: DualRailPulse,
    contour_a: LogicalContour,
    contour_b: LogicalContour,
    solver: SolverConfig,
    constraints: DualRailPulseConstraints,
    method: str,
) -> str:
    payload = {
        "schema": DYNAMIC_DUALRAIL_FSIM_SCHEMA,
        "config": asdict(model.config),
        "pulse": asdict(pulse),
        "contour_a": contour_a.fingerprint(),
        "contour_b": contour_b.fingerprint(),
        "solver": asdict(solver),
        "constraints": asdict(constraints),
        "method": method,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _logical_state_index(initial_state: str | int) -> int:
    if isinstance(initial_state, str):
        try:
            return LOGICAL_STATE_LABELS.index(initial_state)
        except ValueError as error:
            raise ValueError(
                f"initial_state must be one of {LOGICAL_STATE_LABELS} or an index 0--3"
            ) from error
    if (
        isinstance(initial_state, (int, np.integer))
        and not isinstance(initial_state, bool)
        and 0 <= int(initial_state) < len(LOGICAL_STATE_LABELS)
    ):
        return int(initial_state)
    raise ValueError(
        f"initial_state must be one of {LOGICAL_STATE_LABELS} or an index 0--3"
    )


def simulate_dualrail_population_trajectory(
    model: DualRailFSimModel,
    pulse: DualRailPulse,
    contour_a: LogicalContour,
    contour_b: LogicalContour,
    initial_state: str | int,
    *,
    solver: SolverConfig | None = None,
    constraints: DualRailPulseConstraints | None = None,
    method: Literal["auto", "exact", "solver"] = "auto",
) -> DualRailPopulationTrajectory:
    """Evolve one logical input and record four populations plus leakage."""

    solver = SolverConfig() if solver is None else solver
    constraints = DualRailPulseConstraints() if constraints is None else constraints
    solver.validate()
    if method not in ("auto", "exact", "solver"):
        raise ValueError("method must be 'auto', 'exact', or 'solver'")
    if method == "exact" and pulse.rise_time_ns != 0:
        raise ValueError("exact propagation is available only for rectangular pulses")
    selected_method = (
        "exact"
        if method == "exact" or (method == "auto" and pulse.rise_time_ns == 0)
        else "solver"
    )
    state_index = _logical_state_index(initial_state)
    state_label = LOGICAL_STATE_LABELS[state_index]
    waveforms, _, _, _, _, _ = _pulse_waveform_builder(
        model,
        pulse,
        contour_a,
        contour_b,
        constraints,
    )
    subsystem_configs = model.config.subsystems()
    time_ns = _pulse_time_grid(pulse)
    frequency_a, frequency_b, flux = waveforms(time_ns)
    initial = model.logical_states[state_index]

    if selected_method == "exact":
        peak_flux = np.asarray(
            waveforms(pulse.padding_ns + pulse.duration_ns / 2)[2],
            dtype=float,
        )
        peak_hamiltonian = model.hamiltonian_ghz.copy()
        for subsystem, peak_flux_value, operator in zip(
            subsystem_configs,
            peak_flux,
            model.flux_operators_ghz,
            strict=True,
        ):
            peak_hamiltonian += (
                _effective_ej(subsystem, peak_flux_value)
                - _effective_ej(subsystem, subsystem.flux)
            ) * operator
        peak_eigenvalues, peak_eigenstates = peak_hamiltonian.eigenstates()
        peak_eigenvectors = np.column_stack(
            [state.full().ravel() for state in peak_eigenstates]
        )
        idle_eigenvalues, idle_eigenstates = model.hamiltonian_ghz.eigenstates()
        idle_eigenvectors = np.column_stack(
            [state.full().ravel() for state in idle_eigenstates]
        )
        initial_vector = initial.full().ravel()
        active_start_ns = pulse.padding_ns
        active_end_ns = pulse.padding_ns + pulse.duration_ns
        evolved_vectors = np.empty(
            (initial_vector.size, time_ns.size),
            dtype=complex,
        )
        before_active = time_ns <= active_start_ns
        during_active = (time_ns > active_start_ns) & (time_ns <= active_end_ns)
        after_active = time_ns > active_end_ns
        evolved_vectors[:, before_active] = _static_evolution(
            idle_eigenvalues,
            idle_eigenvectors,
            initial_vector,
            time_ns[before_active],
        )
        active_initial = _static_evolution(
            idle_eigenvalues,
            idle_eigenvectors,
            initial_vector,
            pulse.padding_ns,
        )[:, 0]
        evolved_vectors[:, during_active] = _static_evolution(
            peak_eigenvalues,
            peak_eigenvectors,
            active_initial,
            time_ns[during_active] - active_start_ns,
        )
        active_final = _static_evolution(
            peak_eigenvalues,
            peak_eigenvectors,
            active_initial,
            pulse.duration_ns,
        )[:, 0]
        evolved_vectors[:, after_active] = _static_evolution(
            idle_eigenvalues,
            idle_eigenvectors,
            active_final,
            time_ns[after_active] - active_end_ns,
        )
    else:
        hamiltonian_terms: list = [2 * np.pi * model.hamiltonian_ghz]
        coefficient_flux = np.asarray(flux, dtype=float)
        if pulse.rise_time_ns == 0:
            rectangular_peak_flux = np.asarray(
                waveforms(pulse.padding_ns + pulse.duration_ns / 2)[2],
                dtype=float,
            )
            idle_flux = np.asarray(
                [subsystem.flux for subsystem in subsystem_configs],
                dtype=float,
            )
            active = (time_ns >= pulse.padding_ns) & (
                time_ns < pulse.padding_ns + pulse.duration_ns
            )
            coefficient_flux = np.where(
                active[:, None],
                rectangular_peak_flux[None, :],
                idle_flux[None, :],
            )
        for index, (subsystem, operator) in enumerate(
            zip(subsystem_configs, model.flux_operators_ghz, strict=True)
        ):
            idle_ej = _effective_ej(subsystem, subsystem.flux)
            delta_ej = _effective_ej(subsystem, coefficient_flux[:, index]) - idle_ej
            if np.max(np.abs(delta_ej)) > 1e-14:
                hamiltonian_terms.append([2 * np.pi * operator, delta_ej])
        interpolation_order = 0 if pulse.rise_time_ns == 0 else 3
        angular_hamiltonian = qt.QobjEvo(
            hamiltonian_terms,
            tlist=time_ns,
            order=interpolation_order,
        )
        evolution = qt.sesolve(
            angular_hamiltonian,
            initial,
            time_ns,
            options={
                "atol": solver.atol,
                "rtol": solver.rtol,
                "nsteps": solver.nsteps,
                "method": solver.integrator,
                "store_states": True,
                "store_final_state": True,
            },
        )
        evolved_vectors = np.column_stack(
            [state.full().ravel() for state in evolution.states]
        )

    logical_vectors = np.column_stack(
        [state.full().ravel() for state in model.logical_states]
    )
    logical_amplitude = logical_vectors.conj().T @ evolved_vectors
    logical_population = np.abs(logical_amplitude.T) ** 2
    leakage = np.clip(1.0 - np.sum(logical_population, axis=1), 0.0, 1.0)
    pulse_fingerprint = _pulse_fingerprint(
        model,
        pulse,
        contour_a,
        contour_b,
        solver,
        constraints,
        selected_method,
    )
    fingerprint = hashlib.sha256(
        f"{POPULATION_TRAJECTORY_SCHEMA}:{pulse_fingerprint}:{state_label}".encode()
    ).hexdigest()
    return DualRailPopulationTrajectory(
        time_ns=time_ns,
        logical_frequency_a_mhz=np.asarray(frequency_a, dtype=float),
        logical_frequency_b_mhz=np.asarray(frequency_b, dtype=float),
        subsystem_flux=np.asarray(flux, dtype=float),
        logical_population=np.asarray(logical_population, dtype=float),
        leakage=np.asarray(leakage, dtype=float),
        initial_state=state_label,
        solver_method=selected_method,
        fingerprint=fingerprint,
    )


def simulate_dualrail_pulse(
    model: DualRailFSimModel,
    pulse: DualRailPulse,
    contour_a: LogicalContour,
    contour_b: LogicalContour,
    *,
    solver: SolverConfig | None = None,
    constraints: DualRailPulseConstraints | None = None,
    method: Literal["auto", "exact", "solver"] = "auto",
    cpus: int = 1,
) -> DualRailPulseResult:
    """Evolve all four logical states in the fixed idle subsystem basis.

    ``cpus`` applies to independent QuTiP ODE solves. The exact rectangular
    path shares one diagonalization and does not create worker processes.
    """

    solver = SolverConfig() if solver is None else solver
    constraints = DualRailPulseConstraints() if constraints is None else constraints
    solver.validate()
    cpus = _validate_cpus(cpus)
    if method not in ("auto", "exact", "solver"):
        raise ValueError("method must be 'auto', 'exact', or 'solver'")
    if method == "exact" and pulse.rise_time_ns != 0:
        raise ValueError("exact propagation is available only for rectangular pulses")
    selected_method = (
        "exact"
        if method == "exact" or (method == "auto" and pulse.rise_time_ns == 0)
        else "solver"
    )
    waveforms, max_slew, _, _, _, _ = _pulse_waveform_builder(
        model,
        pulse,
        contour_a,
        contour_b,
        constraints,
    )
    subsystem_configs = model.config.subsystems()
    time_ns = _pulse_time_grid(pulse)
    frequency_a, frequency_b, flux = waveforms(time_ns)

    if selected_method == "exact":
        peak_frequency_a = (
            contour_a.inverse_contour(
                model.config.rail_a1.flux,
                model.config.rail_a2.flux,
            )
            if pulse.logical_frequency_a_peak_mhz is None
            else pulse.logical_frequency_a_peak_mhz
        )
        peak_frequency_b = (
            contour_b.inverse_contour(
                model.config.rail_b1.flux,
                model.config.rail_b2.flux,
            )
            if pulse.logical_frequency_b_peak_mhz is None
            else pulse.logical_frequency_b_peak_mhz
        )
        peak_a1, peak_a2 = contour_a.flux(peak_frequency_a)
        peak_b1, peak_b2 = contour_b.flux(peak_frequency_b)
        peak_flux = np.asarray(
            [peak_a1, peak_a2, pulse.coupler_peak_flux, peak_b1, peak_b2],
            dtype=float,
        )
        peak_hamiltonian = model.hamiltonian_ghz.copy()
        for subsystem, peak_flux_value, operator in zip(
            subsystem_configs,
            peak_flux,
            model.flux_operators_ghz,
            strict=True,
        ):
            peak_hamiltonian += (
                _effective_ej(subsystem, peak_flux_value)
                - _effective_ej(subsystem, subsystem.flux)
            ) * operator
        peak_eigenvalues, peak_eigenstates = peak_hamiltonian.eigenstates()
        peak_eigenvectors = np.column_stack(
            [state.full().ravel() for state in peak_eigenstates]
        )
        idle_eigenvalues, idle_eigenstates = model.hamiltonian_ghz.eigenstates()
        idle_eigenvectors = np.column_stack(
            [state.full().ravel() for state in idle_eigenstates]
        )
        state_trajectories = []
        active_start_ns = pulse.padding_ns
        active_end_ns = pulse.padding_ns + pulse.duration_ns
        before_active = time_ns <= active_start_ns
        during_active = (time_ns > active_start_ns) & (time_ns <= active_end_ns)
        after_active = time_ns > active_end_ns
        for initial_state in model.logical_states:
            initial_vector = initial_state.full().ravel()
            evolved_vectors = np.empty(
                (initial_vector.size, time_ns.size), dtype=complex
            )
            evolved_vectors[:, before_active] = _static_evolution(
                idle_eigenvalues,
                idle_eigenvectors,
                initial_vector,
                time_ns[before_active],
            )
            active_initial = _static_evolution(
                idle_eigenvalues,
                idle_eigenvectors,
                initial_vector,
                pulse.padding_ns,
            )[:, 0]
            active_final = _static_evolution(
                peak_eigenvalues,
                peak_eigenvectors,
                active_initial,
                pulse.duration_ns,
            )[:, 0]
            evolved_vectors[:, during_active] = _static_evolution(
                peak_eigenvalues,
                peak_eigenvectors,
                active_initial,
                time_ns[during_active] - active_start_ns,
            )
            evolved_vectors[:, after_active] = _static_evolution(
                idle_eigenvalues,
                idle_eigenvectors,
                active_final,
                time_ns[after_active] - active_end_ns,
            )
            state_trajectories.append(evolved_vectors)
    else:
        hamiltonian_terms: list = [2 * np.pi * model.hamiltonian_ghz]
        coefficient_flux = flux
        if pulse.rise_time_ns == 0:
            rectangular_peak_flux = np.asarray(
                waveforms(pulse.padding_ns + pulse.duration_ns / 2)[2],
                dtype=float,
            )
            idle_flux = np.asarray(
                [subsystem.flux for subsystem in subsystem_configs],
                dtype=float,
            )
            active = (time_ns >= pulse.padding_ns) & (
                time_ns < pulse.padding_ns + pulse.duration_ns
            )
            coefficient_flux = np.where(
                active[:, None],
                rectangular_peak_flux[None, :],
                idle_flux[None, :],
            )
        for index, (subsystem, operator) in enumerate(
            zip(subsystem_configs, model.flux_operators_ghz, strict=True)
        ):
            idle_ej = _effective_ej(subsystem, subsystem.flux)
            delta_ej = _effective_ej(subsystem, coefficient_flux[:, index]) - idle_ej
            if np.max(np.abs(delta_ej)) > 1e-14:
                hamiltonian_terms.append([2 * np.pi * operator, delta_ej])
        interpolation_order = 0 if pulse.rise_time_ns == 0 else 3
        solver_options = {
            "atol": solver.atol,
            "rtol": solver.rtol,
            "nsteps": solver.nsteps,
            "method": solver.integrator,
            "store_states": True,
            "store_final_state": True,
        }
        jobs = [
            (
                hamiltonian_terms,
                time_ns,
                interpolation_order,
                initial_state,
                solver_options,
            )
            for initial_state in model.logical_states
        ]
        state_trajectories = _process_map(
            _solve_state_trajectory,
            jobs,
            min(cpus, len(jobs)),
        )

    final_states = [
        qt.Qobj(trajectory[:, -1], dims=initial_state.dims)
        for trajectory, initial_state in zip(
            state_trajectories, model.logical_states, strict=True
        )
    ]

    projected = np.asarray(
        [
            [complex(output.overlap(final)) for final in final_states]
            for output in model.logical_states
        ]
    )
    survival = np.sum(np.abs(projected) ** 2, axis=0)
    leakage = np.clip(1.0 - survival.real, 0.0, 1.0)
    polar_unitary = closest_unitary(projected)
    theta, phi, swap_probability = extract_fsim_angles(projected)
    allowed = np.zeros((4, 4), dtype=bool)
    allowed[0, 0] = allowed[1, 1] = allowed[1, 2] = True
    allowed[2, 1] = allowed[2, 2] = allowed[3, 3] = True
    parasitic_amplitude = float(np.max(np.abs(projected[~allowed])))
    unitarity_error = float(
        np.linalg.norm(projected.conj().T @ projected - np.eye(4), ord="fro") / 2
    )

    fitted_fsim = ideal_fsim(theta, phi)
    fitted_fidelity, _ = local_z_optimized_fidelity(polar_unitary, fitted_fsim)
    iswap_target = ideal_fsim(np.pi / 2, 0.0)
    cphase_target = ideal_fsim(0.0, pulse.target_cphase_rad)
    iswap_coherent_fidelity, iswap_z = local_z_optimized_fidelity(
        polar_unitary,
        iswap_target,
    )
    cphase_coherent_fidelity, cphase_z = local_z_optimized_fidelity(
        polar_unitary,
        cphase_target,
    )
    iswap_with_leakage = average_gate_fidelity(
        _apply_local_z(projected, iswap_z),
        iswap_target,
    )
    cphase_with_leakage = average_gate_fidelity(
        _apply_local_z(projected, cphase_z),
        cphase_target,
    )

    dimensions = tuple(item.truncated_dim for item in subsystem_configs)
    pair_a_one = _pair_one_excitation_projector(dimensions[0], dimensions[1])
    pair_b_one = _pair_one_excitation_projector(dimensions[3], dimensions[4])
    identity_a = qt.tensor(qt.qeye(dimensions[0]), qt.qeye(dimensions[1]))
    identity_b = qt.tensor(qt.qeye(dimensions[3]), qt.qeye(dimensions[4]))
    coupler_ground = qt.basis(dimensions[2], 0).proj()
    projector_a = qt.tensor(pair_a_one, qt.qeye(dimensions[2]), identity_b)
    projector_b = qt.tensor(identity_a, qt.qeye(dimensions[2]), pair_b_one)
    projector_c = qt.tensor(identity_a, coupler_ground, identity_b)
    bare_code_projector = qt.tensor(pair_a_one, coupler_ground, pair_b_one)

    def probabilities(projector: qt.Qobj) -> np.ndarray:
        return np.clip(
            np.asarray(
                [qt.expect(projector, state) for state in final_states], dtype=float
            ),
            0.0,
            1.0,
        )

    bare_code_probability = probabilities(bare_code_projector)
    coupler_excited_probability = 1.0 - probabilities(projector_c)
    pair_a_invalid_probability = 1.0 - probabilities(projector_a)
    pair_b_invalid_probability = 1.0 - probabilities(projector_b)

    logical_vectors = np.column_stack(
        [state.full().ravel() for state in model.logical_states]
    )

    def maximum_transient_complement(projector: qt.Qobj) -> np.ndarray:
        diagonal = np.asarray(projector.diag(), dtype=float)
        maxima = []
        for trajectory in state_trajectories:
            expectation = np.sum(np.abs(trajectory) ** 2 * diagonal[:, None], axis=0)
            maxima.append(np.max(np.clip(1.0 - expectation, 0.0, 1.0)))
        return np.asarray(maxima, dtype=float)

    maximum_transient_leakage = []
    for trajectory in state_trajectories:
        logical_amplitude = logical_vectors.conj().T @ trajectory
        survival_trajectory = np.sum(np.abs(logical_amplitude) ** 2, axis=0)
        maximum_transient_leakage.append(
            np.max(np.clip(1.0 - survival_trajectory.real, 0.0, 1.0))
        )

    maximum_transient_leakage = np.maximum(
        np.asarray(maximum_transient_leakage, dtype=float), leakage
    )
    maximum_transient_coupler = np.maximum(
        maximum_transient_complement(projector_c), coupler_excited_probability
    )
    maximum_transient_pair_a = np.maximum(
        maximum_transient_complement(projector_a), pair_a_invalid_probability
    )
    maximum_transient_pair_b = np.maximum(
        maximum_transient_complement(projector_b), pair_b_invalid_probability
    )

    return DualRailPulseResult(
        time_ns=time_ns,
        logical_frequency_a_mhz=np.asarray(frequency_a, dtype=float),
        logical_frequency_b_mhz=np.asarray(frequency_b, dtype=float),
        subsystem_flux=np.asarray(flux, dtype=float),
        max_flux_slew_per_ns=max_slew,
        projected_unitary=projected,
        closest_unitary=polar_unitary,
        theta_rad=theta,
        phi_rad=phi,
        swap_probability=swap_probability,
        survival_by_state=survival.real,
        leakage_by_state=leakage,
        average_leakage=float(np.mean(leakage)),
        bare_code_probability_by_state=bare_code_probability,
        coupler_excited_probability_by_state=coupler_excited_probability,
        pair_a_invalid_probability_by_state=pair_a_invalid_probability,
        pair_b_invalid_probability_by_state=pair_b_invalid_probability,
        maximum_transient_leakage_by_state=maximum_transient_leakage,
        maximum_transient_coupler_excited_probability_by_state=(
            maximum_transient_coupler
        ),
        maximum_transient_pair_a_invalid_probability_by_state=(
            maximum_transient_pair_a
        ),
        maximum_transient_pair_b_invalid_probability_by_state=(
            maximum_transient_pair_b
        ),
        parasitic_amplitude=parasitic_amplitude,
        unitarity_error=unitarity_error,
        coherent_fsim_error=float(1.0 - fitted_fidelity),
        iswap_coherent_fidelity=iswap_coherent_fidelity,
        iswap_fidelity_with_leakage=iswap_with_leakage,
        cphase_coherent_fidelity=cphase_coherent_fidelity,
        cphase_fidelity_with_leakage=cphase_with_leakage,
        iswap_virtual_z_rad=iswap_z,
        cphase_virtual_z_rad=cphase_z,
        dynamic_phase_rad=np.angle(np.diag(projected)),
        logical_assignment=model.logical_assignment.copy(),
        logical_assignment_probability=model.logical_assignment_probability.copy(),
        solver_method=selected_method,
        solver_status="success",
        target_cphase_rad=pulse.target_cphase_rad,
        fingerprint=_pulse_fingerprint(
            model,
            pulse,
            contour_a,
            contour_b,
            solver,
            constraints,
            selected_method,
        ),
    )


def _landscape_fingerprint(
    config: FiveTransmonConfig,
    contour_a: LogicalContour,
    contour_b: LogicalContour,
    frequency_a: np.ndarray,
    frequency_b: np.ndarray,
    coupler_flux: np.ndarray,
    coherent_ceiling_max_time_ns: float,
    coherent_ceiling_time_samples: int,
    target_cphase_rad: float,
) -> str:
    payload = {
        "schema": STATIC_DUALRAIL_FSIM_SCHEMA,
        "config": asdict(config),
        "contour_a": contour_a.fingerprint(),
        "contour_b": contour_b.fingerprint(),
        "logical_frequency_a_mhz": frequency_a.tolist(),
        "logical_frequency_b_mhz": frequency_b.tolist(),
        "coupler_flux": coupler_flux.tolist(),
        "coherent_ceiling_max_time_ns": coherent_ceiling_max_time_ns,
        "coherent_ceiling_time_samples": coherent_ceiling_time_samples,
        "target_cphase_rad": target_cphase_rad,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _static_interaction_chunk(job):
    (
        flat_indices,
        shape,
        config,
        phi_a1,
        phi_a2,
        phi_b1,
        phi_b2,
        coupler_flux,
        coherent_ceiling_max_time_ns,
        coherent_ceiling_time_samples,
        target_cphase_rad,
    ) = job
    results = []
    for flat_index in flat_indices:
        coupler_index, a_index, b_index = np.unravel_index(int(flat_index), shape)
        point_config = replace(
            config,
            rail_a1=replace(config.rail_a1, flux=float(phi_a1[a_index])),
            rail_a2=replace(config.rail_a2, flux=float(phi_a2[a_index])),
            coupler=replace(
                config.coupler,
                flux=float(coupler_flux[coupler_index]),
            ),
            rail_b1=replace(config.rail_b1, flux=float(phi_b1[b_index])),
            rail_b2=replace(config.rail_b2, flux=float(phi_b2[b_index])),
        )
        static = build_dualrail_fsim_model(point_config).static_interaction()
        ceilings = tuple(
            evaluate_static_coherent_ceiling(
                static.effective_hamiltonian_ghz,
                gate_type,
                maximum_duration_ns=coherent_ceiling_max_time_ns,
                time_samples=coherent_ceiling_time_samples,
                target_cphase_rad=target_cphase_rad,
            )
            for gate_type in ("iswap", "cz")
        )
        results.append((static, *ceilings))
    return flat_indices, results


def run_static_interaction_landscape(
    config: FiveTransmonConfig,
    contour_a: LogicalContour,
    contour_b: LogicalContour,
    logical_frequency_a_mhz,
    logical_frequency_b_mhz,
    coupler_flux_values,
    *,
    cpus: int = 1,
    coherent_ceiling_max_time_ns: float = 600.0,
    coherent_ceiling_time_samples: int = 61,
    target_cphase_rad: float = np.pi,
) -> xr.Dataset:
    """Evaluate static go/no-go metrics on a physical/logical control grid.

    With ``cpus > 1``, grid points are divided into balanced chunks and built
    in separate ``pathos`` processes.
    """

    frequency_a = _finite_1d(logical_frequency_a_mhz, "logical_frequency_a_mhz")
    frequency_b = _finite_1d(logical_frequency_b_mhz, "logical_frequency_b_mhz")
    coupler_flux = _finite_1d(coupler_flux_values, "coupler_flux_values")
    cpus = _validate_cpus(cpus)
    if (
        not np.isfinite(coherent_ceiling_max_time_ns)
        or coherent_ceiling_max_time_ns <= 0
    ):
        raise ValueError("coherent_ceiling_max_time_ns must be positive and finite")
    if (
        not isinstance(coherent_ceiling_time_samples, (int, np.integer))
        or isinstance(coherent_ceiling_time_samples, bool)
        or coherent_ceiling_time_samples < 3
    ):
        raise ValueError("coherent_ceiling_time_samples must be at least 3")
    if not np.isfinite(target_cphase_rad):
        raise ValueError("target_cphase_rad must be finite")
    phi_a1, phi_a2 = contour_a.flux(frequency_a)
    phi_b1, phi_b2 = contour_b.flux(frequency_b)
    shape = (coupler_flux.size, frequency_a.size, frequency_b.size)
    logical_shape = (*shape, len(LOGICAL_STATE_LABELS))
    scalar_names = (
        "model_logical_detuning_mhz",
        "logical_exchange_mhz",
        "direct_exchange_mhz",
        "effective_exchange_mhz",
        "effective_zz_mhz",
        "logical_zz_mhz",
        "nearest_leakage_gap_mhz",
        "estimated_iswap_time_ns",
        "estimated_cz_time_ns",
        "logical_subspace_sigma_min",
    )
    values = {name: np.empty(shape) for name in scalar_names}
    logical_energy = np.empty(logical_shape)
    assignment_probability = np.empty(logical_shape)
    logical_singular_values = np.empty(logical_shape)
    coupler_participation = np.empty(logical_shape)
    pair_a_occupation = np.empty(logical_shape)
    pair_b_occupation = np.empty(logical_shape)
    logical_assignment = np.empty(logical_shape, dtype=int)
    nearest_leakage_level = np.empty(shape, dtype=int)
    nearest_leakage_logical_state = np.empty(shape, dtype="U2")
    nearest_leakage_branch_label = np.empty(shape, dtype="U64")
    nearest_leakage_branch_probability = np.empty(shape)
    pauli_coefficient = np.empty((*shape, len(PAULI_TERMS)), dtype=complex)
    iswap_exchange_component = np.empty(shape)
    iswap_pair_creation_component = np.empty(shape)
    iswap_nuisance_pauli_norm = np.empty(shape)
    cz_offdiagonal_pauli_norm = np.empty(shape)
    cz_offdiagonal_to_zz_ratio = np.empty(shape)
    maximum_coupler_participation = np.empty(shape)
    iswap_static_coherent_ceiling = np.empty(shape)
    iswap_static_coherent_time = np.empty(shape)
    iswap_static_virtual_z = np.empty((*shape, 4))
    cz_static_coherent_ceiling = np.empty(shape)
    cz_static_coherent_time = np.empty(shape)
    cz_static_virtual_z = np.empty((*shape, 4))
    projected_hamiltonian = np.empty((*shape, 4, 4), dtype=complex)
    effective_hamiltonian = np.empty((*shape, 4, 4), dtype=complex)
    assignment_valid = np.empty(shape, dtype=bool)
    spectrally_isolated = np.empty(shape, dtype=bool)

    worker_count = min(cpus, int(np.prod(shape)))
    flat_chunks = np.array_split(np.arange(np.prod(shape)), worker_count)
    jobs = [
        (
            flat_indices,
            shape,
            config,
            phi_a1,
            phi_a2,
            phi_b1,
            phi_b2,
            coupler_flux,
            float(coherent_ceiling_max_time_ns),
            int(coherent_ceiling_time_samples),
            float(target_cphase_rad),
        )
        for flat_indices in flat_chunks
    ]
    for flat_indices, results in _process_map(
        _static_interaction_chunk,
        jobs,
        worker_count,
    ):
        for flat_index, packed_result in zip(flat_indices, results, strict=True):
            result, iswap_ceiling, cz_ceiling = packed_result
            index = np.unravel_index(int(flat_index), shape)
            for name in scalar_names:
                source_name = (
                    "logical_detuning_mhz"
                    if name == "model_logical_detuning_mhz"
                    else name
                )
                values[name][index] = getattr(result, source_name)
            logical_energy[index] = result.logical_energy_ghz
            assignment_probability[index] = result.logical_assignment_probability
            logical_singular_values[index] = result.logical_singular_values
            coupler_participation[index] = result.coupler_participation
            pair_a_occupation[index] = result.pair_a_occupation
            pair_b_occupation[index] = result.pair_b_occupation
            logical_assignment[index] = result.logical_assignment
            nearest_leakage_level[index] = result.nearest_leakage_level
            nearest_leakage_logical_state[index] = (
                result.nearest_leakage_logical_state
            )
            nearest_leakage_branch_label[index] = result.nearest_leakage_branch_label
            nearest_leakage_branch_probability[index] = (
                result.nearest_leakage_branch_probability
            )
            projected_hamiltonian[index] = result.projected_hamiltonian_ghz
            effective_hamiltonian[index] = result.effective_hamiltonian_ghz
            decomposition = pauli_decompose(result.effective_hamiltonian_ghz)
            pauli_coefficient[index] = [decomposition[term] for term in PAULI_TERMS]
            coefficients_mhz = {
                term: float(1e3 * np.real(value))
                for term, value in decomposition.items()
            }
            iswap_exchange_component[index] = 0.5 * (
                coefficients_mhz["XX"] + coefficients_mhz["YY"]
            )
            iswap_pair_creation_component[index] = 0.5 * (
                coefficients_mhz["XX"] - coefficients_mhz["YY"]
            )
            iswap_nuisance_pauli_norm[index] = np.linalg.norm(
                [
                    coefficients_mhz[term]
                    for term in ("XI", "IX", "XZ", "ZX", "XY", "YX")
                ]
            )
            offdiagonal_terms = tuple(
                term
                for term in PAULI_TERMS
                if "X" in term or "Y" in term
            )
            cz_offdiagonal_pauli_norm[index] = np.linalg.norm(
                [coefficients_mhz[term] for term in offdiagonal_terms]
            )
            zz_scale = abs(result.effective_zz_mhz)
            cz_offdiagonal_to_zz_ratio[index] = (
                cz_offdiagonal_pauli_norm[index] / zz_scale
                if zz_scale > np.finfo(float).eps
                else np.inf
            )
            maximum_coupler_participation[index] = np.max(
                result.coupler_participation
            )
            iswap_static_coherent_ceiling[index] = iswap_ceiling.fidelity
            iswap_static_coherent_time[index] = iswap_ceiling.duration_ns
            iswap_static_virtual_z[index] = iswap_ceiling.virtual_z_rad
            cz_static_coherent_ceiling[index] = cz_ceiling.fidelity
            cz_static_coherent_time[index] = cz_ceiling.duration_ns
            cz_static_virtual_z[index] = cz_ceiling.virtual_z_rad
            assignment_valid[index] = result.logical_assignment_valid
            spectrally_isolated[index] = result.spectrally_isolated

    dimensions = (
        "coupler_flux",
        "logical_frequency_a_mhz",
        "logical_frequency_b_mhz",
    )
    logical_dimensions = (*dimensions, "logical_state")
    dataset = xr.Dataset(
        data_vars={
            "phi_a1": ("logical_frequency_a_mhz", phi_a1),
            "phi_a2": ("logical_frequency_a_mhz", phi_a2),
            "phi_b1": ("logical_frequency_b_mhz", phi_b1),
            "phi_b2": ("logical_frequency_b_mhz", phi_b2),
            "logical_detuning_mhz": (
                ("logical_frequency_a_mhz", "logical_frequency_b_mhz"),
                frequency_a[:, None] - frequency_b[None, :],
            ),
            **{name: (dimensions, data) for name, data in values.items()},
            "logical_energy_ghz": (logical_dimensions, logical_energy),
            "logical_assignment_probability": (
                logical_dimensions,
                assignment_probability,
            ),
            "logical_singular_values": (
                logical_dimensions,
                logical_singular_values,
            ),
            "coupler_participation": (logical_dimensions, coupler_participation),
            "pair_a_occupation": (logical_dimensions, pair_a_occupation),
            "pair_b_occupation": (logical_dimensions, pair_b_occupation),
            "logical_assignment": (logical_dimensions, logical_assignment),
            "nearest_leakage_level": (dimensions, nearest_leakage_level),
            "nearest_leakage_logical_state": (
                dimensions,
                nearest_leakage_logical_state,
            ),
            "nearest_leakage_branch_label": (
                dimensions,
                nearest_leakage_branch_label,
            ),
            "nearest_leakage_branch_probability": (
                dimensions,
                nearest_leakage_branch_probability,
            ),
            "projected_hamiltonian_real_ghz": (
                (*dimensions, "logical_output", "logical_input"),
                projected_hamiltonian.real,
            ),
            "projected_hamiltonian_imag_ghz": (
                (*dimensions, "logical_output", "logical_input"),
                projected_hamiltonian.imag,
            ),
            "effective_hamiltonian_real_ghz": (
                (*dimensions, "logical_output", "logical_input"),
                effective_hamiltonian.real,
            ),
            "effective_hamiltonian_imag_ghz": (
                (*dimensions, "logical_output", "logical_input"),
                effective_hamiltonian.imag,
            ),
            "pauli_coefficient_real_mhz": (
                (*dimensions, "pauli_term"),
                1e3 * pauli_coefficient.real,
            ),
            "pauli_coefficient_imag_mhz": (
                (*dimensions, "pauli_term"),
                1e3 * pauli_coefficient.imag,
            ),
            "iswap_exchange_component_mhz": (
                dimensions,
                iswap_exchange_component,
            ),
            "iswap_pair_creation_component_mhz": (
                dimensions,
                iswap_pair_creation_component,
            ),
            "iswap_nuisance_pauli_norm_mhz": (
                dimensions,
                iswap_nuisance_pauli_norm,
            ),
            "cz_offdiagonal_pauli_norm_mhz": (
                dimensions,
                cz_offdiagonal_pauli_norm,
            ),
            "cz_offdiagonal_to_zz_ratio": (
                dimensions,
                cz_offdiagonal_to_zz_ratio,
            ),
            "maximum_coupler_participation": (
                dimensions,
                maximum_coupler_participation,
            ),
            "iswap_static_coherent_ceiling": (
                dimensions,
                iswap_static_coherent_ceiling,
            ),
            "iswap_static_coherent_time_ns": (
                dimensions,
                iswap_static_coherent_time,
            ),
            "iswap_static_virtual_z_rad": (
                (*dimensions, "local_z_correction"),
                iswap_static_virtual_z,
            ),
            "cz_static_coherent_ceiling": (
                dimensions,
                cz_static_coherent_ceiling,
            ),
            "cz_static_coherent_time_ns": (
                dimensions,
                cz_static_coherent_time,
            ),
            "cz_static_virtual_z_rad": (
                (*dimensions, "local_z_correction"),
                cz_static_virtual_z,
            ),
            "logical_assignment_valid": (dimensions, assignment_valid),
            "spectrally_isolated": (dimensions, spectrally_isolated),
        },
        coords={
            "coupler_flux": coupler_flux,
            "logical_frequency_a_mhz": frequency_a,
            "logical_frequency_b_mhz": frequency_b,
            "logical_state": list(LOGICAL_STATE_LABELS),
            "logical_output": list(LOGICAL_STATE_LABELS),
            "logical_input": list(LOGICAL_STATE_LABELS),
            "pauli_term": list(PAULI_TERMS),
            "local_z_correction": ["a_before", "b_before", "a_after", "b_after"],
        },
        attrs={
            "description": "Static five-transmon dual-rail fSim feasibility map.",
            "static_dualrail_fsim_schema": STATIC_DUALRAIL_FSIM_SCHEMA,
            "fingerprint": _landscape_fingerprint(
                config,
                contour_a,
                contour_b,
                frequency_a,
                frequency_b,
                coupler_flux,
                float(coherent_ceiling_max_time_ns),
                int(coherent_ceiling_time_samples),
                float(target_cphase_rad),
            ),
            "parameter_source": config.parameter_source,
            "logical_zero_mode": config.logical_mapping.logical_zero,
            "logical_one_mode": config.logical_mapping.logical_one,
            "contour_a_fingerprint": contour_a.fingerprint(),
            "contour_b_fingerprint": contour_b.fingerprint(),
            "cpus": cpus,
            "worker_count": worker_count,
            "coherent_ceiling_max_time_ns": float(
                coherent_ceiling_max_time_ns
            ),
            "coherent_ceiling_time_samples": int(coherent_ceiling_time_samples),
            "target_cphase_rad": float(target_cphase_rad),
        },
    )
    for name in (
        "logical_frequency_a_mhz",
        "logical_frequency_b_mhz",
        "logical_detuning_mhz",
        "model_logical_detuning_mhz",
        "logical_exchange_mhz",
        "direct_exchange_mhz",
        "effective_exchange_mhz",
        "effective_zz_mhz",
        "logical_zz_mhz",
        "nearest_leakage_gap_mhz",
        "iswap_exchange_component_mhz",
        "iswap_pair_creation_component_mhz",
        "iswap_nuisance_pauli_norm_mhz",
        "cz_offdiagonal_pauli_norm_mhz",
    ):
        dataset[name].attrs["units"] = "MHz"
    dataset.coupler_flux.attrs["units"] = "flux quantum"
    for name in ("phi_a1", "phi_a2", "phi_b1", "phi_b2"):
        dataset[name].attrs["units"] = "flux quantum"
    dataset.logical_energy_ghz.attrs["units"] = "GHz"
    dataset.projected_hamiltonian_real_ghz.attrs["units"] = "GHz"
    dataset.projected_hamiltonian_imag_ghz.attrs["units"] = "GHz"
    dataset.effective_hamiltonian_real_ghz.attrs["units"] = "GHz"
    dataset.effective_hamiltonian_imag_ghz.attrs["units"] = "GHz"
    dataset.pauli_coefficient_real_mhz.attrs["units"] = "MHz"
    dataset.pauli_coefficient_imag_mhz.attrs["units"] = "MHz"
    dataset.estimated_iswap_time_ns.attrs["units"] = "ns"
    dataset.estimated_cz_time_ns.attrs["units"] = "ns"
    dataset.iswap_static_coherent_time_ns.attrs["units"] = "ns"
    dataset.cz_static_coherent_time_ns.attrs["units"] = "ns"
    dataset.iswap_static_virtual_z_rad.attrs["units"] = "rad"
    dataset.cz_static_virtual_z_rad.attrs["units"] = "rad"
    return dataset


def dualrail_idle_score(
    landscape: xr.Dataset,
    *,
    exchange_weight: float = 1.0,
    zz_weight: float = 1.0,
) -> xr.DataArray:
    """Return a static residual-coupling score for every landscape point.

    The exchange and ZZ terms are matrix elements of a Hermitian effective
    logical Hamiltonian reconstructed from the assigned dressed eigenstates.
    This includes virtual coupling and remains meaningful away from resonance.
    Direct ``P H P`` terms and dressed-spectrum diagnostics remain separate in
    the source dataset; the score is only a ranking aid for choosing an idle.
    """

    required = ("effective_exchange_mhz", "effective_zz_mhz")
    missing = [name for name in required if name not in landscape]
    if missing:
        raise ValueError(f"landscape is missing required variables: {missing}")
    weights = np.asarray([exchange_weight, zz_weight], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights):
        raise ValueError(
            "idle-score weights must be finite, nonnegative, and not both zero"
        )

    score = np.hypot(
        exchange_weight * np.abs(landscape.effective_exchange_mhz),
        zz_weight * np.abs(landscape.effective_zz_mhz),
    ).rename("idle_residual_coupling_mhz")
    score.attrs.update(
        {
            "description": (
                "Weighted norm of effective logical exchange and effective logical ZZ"
            ),
            "units": "MHz",
            "exchange_weight": float(exchange_weight),
            "zz_weight": float(zz_weight),
        }
    )
    return score


def select_dualrail_idle_point(
    landscape: xr.Dataset,
    *,
    exchange_weight: float = 1.0,
    zz_weight: float = 1.0,
    require_qualified: bool = True,
) -> xr.Dataset:
    """Select the minimum-residual static idle point from a landscape.

    Qualified points must pass both the logical-assignment and spectral-gap
    gates already stored by :func:`run_static_interaction_landscape`.
    ``require_qualified=False`` is intended only for explicit diagnostics when
    a coarse grid contains no valid point.
    """

    score = dualrail_idle_score(
        landscape,
        exchange_weight=exchange_weight,
        zz_weight=zz_weight,
    )
    search_dimensions = (
        "coupler_flux",
        "logical_frequency_a_mhz",
        "logical_frequency_b_mhz",
    )
    if score.dims != search_dimensions:
        raise ValueError(
            "idle score must have dimensions "
            f"{search_dimensions}, received {score.dims}"
        )
    eligible = np.isfinite(np.asarray(score, dtype=float))
    if require_qualified:
        flags = ("logical_assignment_valid", "spectrally_isolated")
        missing = [name for name in flags if name not in landscape]
        if missing:
            raise ValueError(f"landscape is missing qualification flags: {missing}")
        eligible &= np.asarray(landscape.logical_assignment_valid, dtype=bool)
        eligible &= np.asarray(landscape.spectrally_isolated, dtype=bool)
    if not np.any(eligible):
        qualifier = " qualified" if require_qualified else " finite"
        raise ValueError(f"static landscape contains no{qualifier} idle candidate")

    ranked = np.where(eligible, np.asarray(score, dtype=float), np.inf)
    flat_index = int(np.argmin(ranked))
    indices = np.unravel_index(flat_index, ranked.shape)
    indexers = dict(zip(search_dimensions, indices, strict=True))
    selected = landscape.isel(indexers).copy(deep=True)
    selected[score.name] = score.isel(indexers)
    selected.attrs.update(
        {
            "idle_selection_method": "minimum weighted residual coupling",
            "idle_selection_qualified_required": bool(require_qualified),
            "idle_exchange_weight": float(exchange_weight),
            "idle_zz_weight": float(zz_weight),
        }
    )
    return selected


def _pareto_ranks(objectives: np.ndarray) -> np.ndarray:
    """Return zero-based non-dominated sorting ranks for maximized objectives."""

    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2:
        raise ValueError("objectives must be a two-dimensional array")
    ranks = np.full(objectives.shape[0], -1, dtype=int)
    remaining = np.arange(objectives.shape[0])
    rank = 0
    while remaining.size:
        values = objectives[remaining]
        front = np.ones(remaining.size, dtype=bool)
        for index, value in enumerate(values):
            dominates = np.all(values >= value, axis=1) & np.any(
                values > value, axis=1
            )
            if np.any(dominates):
                front[index] = False
        if not np.any(front):
            raise RuntimeError("failed to identify a finite Pareto front")
        ranks[remaining[front]] = rank
        remaining = remaining[~front]
        rank += 1
    return ranks


def rank_static_gate_candidates(
    landscape: xr.Dataset,
    gate_type: Literal["iswap", "cz"],
    *,
    minimum_coherent_ceiling: float = 0.995,
    maximum_candidates: int = 20,
    minimum_sigma_min: float | None = None,
    minimum_leakage_gap_mhz: float | None = None,
    maximum_cz_offdiagonal_to_zz_ratio: float = 0.25,
) -> xr.Dataset:
    """Retain a Pauli-aware Pareto set after hard static acceptance gates.

    No weighted aggregate physics score is produced.  ``pareto_rank`` records
    successive non-dominated fronts while all objective components remain as
    independent variables in the returned dataset.
    """

    if gate_type not in ("iswap", "cz"):
        raise ValueError("gate_type must be 'iswap' or 'cz'")
    if (
        not np.isfinite(minimum_coherent_ceiling)
        or not 0 <= minimum_coherent_ceiling <= 1
    ):
        raise ValueError("minimum_coherent_ceiling must lie in [0, 1]")
    if (
        not isinstance(maximum_candidates, (int, np.integer))
        or isinstance(maximum_candidates, bool)
        or not 1 <= maximum_candidates <= 30
    ):
        raise ValueError("maximum_candidates must be an integer from 1 to 30")
    if (
        not np.isfinite(maximum_cz_offdiagonal_to_zz_ratio)
        or maximum_cz_offdiagonal_to_zz_ratio <= 0
    ):
        raise ValueError(
            "maximum_cz_offdiagonal_to_zz_ratio must be positive and finite"
        )

    common = (
        "logical_assignment_valid",
        "spectrally_isolated",
        "logical_subspace_sigma_min",
        "nearest_leakage_gap_mhz",
        "nearest_leakage_branch_label",
        "maximum_coupler_participation",
        "model_logical_detuning_mhz",
        "effective_exchange_mhz",
        "effective_zz_mhz",
    )
    gate_variables = (
        (
            "iswap_exchange_component_mhz",
            "iswap_pair_creation_component_mhz",
            "iswap_nuisance_pauli_norm_mhz",
            "iswap_static_coherent_ceiling",
        )
        if gate_type == "iswap"
        else (
            "cz_offdiagonal_pauli_norm_mhz",
            "cz_offdiagonal_to_zz_ratio",
            "cz_static_coherent_ceiling",
        )
    )
    missing = [name for name in (*common, *gate_variables) if name not in landscape]
    if missing:
        raise ValueError(f"landscape is missing required variables: {missing}")

    ceiling_name = f"{gate_type}_static_coherent_ceiling"
    point_dimensions = landscape[ceiling_name].dims
    if not point_dimensions:
        raise ValueError("static coherent ceiling must have at least one dimension")
    if len(point_dimensions) == 1:
        source_dimension = point_dimensions[0]
        flattened = landscape
    else:
        source_dimension = "_atlas_point"
        flattened = landscape.stack({source_dimension: point_dimensions})

    def values(name: str, *, dtype=float) -> np.ndarray:
        data = np.asarray(flattened[name], dtype=dtype)
        if data.shape != (flattened.sizes[source_dimension],):
            raise ValueError(f"{name} must vary only over atlas point dimensions")
        return data

    eligible = values("logical_assignment_valid", dtype=bool).copy()
    eligible &= values("spectrally_isolated", dtype=bool)
    ceiling = values(ceiling_name)
    eligible &= np.isfinite(ceiling) & (ceiling >= minimum_coherent_ceiling)
    leakage_label = values("nearest_leakage_branch_label", dtype=str)
    eligible &= np.char.str_len(leakage_label) > 0
    if minimum_sigma_min is not None:
        if not np.isfinite(minimum_sigma_min) or not 0 <= minimum_sigma_min <= 1:
            raise ValueError("minimum_sigma_min must lie in [0, 1]")
        eligible &= values("logical_subspace_sigma_min") >= minimum_sigma_min
    if minimum_leakage_gap_mhz is not None:
        if not np.isfinite(minimum_leakage_gap_mhz) or minimum_leakage_gap_mhz < 0:
            raise ValueError("minimum_leakage_gap_mhz must be finite and nonnegative")
        eligible &= values("nearest_leakage_gap_mhz") >= minimum_leakage_gap_mhz

    if gate_type == "iswap":
        exchange = np.abs(values("iswap_exchange_component_mhz"))
        eligible &= exchange > np.finfo(float).eps
        objectives = np.column_stack(
            (
                exchange,
                -np.abs(values("iswap_pair_creation_component_mhz")),
                -values("iswap_nuisance_pauli_norm_mhz"),
                -np.abs(values("model_logical_detuning_mhz")),
                -np.abs(values("effective_zz_mhz")),
                values("logical_subspace_sigma_min"),
                values("nearest_leakage_gap_mhz"),
                -values("maximum_coupler_participation"),
                ceiling,
            )
        )
        mechanism = "exchange"
    else:
        zz = np.abs(values("effective_zz_mhz"))
        ratio = values("cz_offdiagonal_to_zz_ratio")
        eligible &= zz > np.finfo(float).eps
        eligible &= ratio <= maximum_cz_offdiagonal_to_zz_ratio
        objectives = np.column_stack(
            (
                zz,
                -np.abs(values("effective_exchange_mhz")),
                -values("cz_offdiagonal_pauli_norm_mhz"),
                values("logical_subspace_sigma_min"),
                values("nearest_leakage_gap_mhz"),
                -values("maximum_coupler_participation"),
                ceiling,
            )
        )
        mechanism = "dispersive_zz"

    eligible &= np.all(np.isfinite(objectives), axis=1)
    source_indices = np.flatnonzero(eligible)
    if source_indices.size:
        ranks = _pareto_ranks(objectives[source_indices])
        front = ranks == 0
        source_indices = source_indices[front]
        order = np.argsort(-ceiling[source_indices], kind="stable")
        source_indices = source_indices[order[:maximum_candidates]]
        ranks = np.zeros(source_indices.size, dtype=int)
    else:
        ranks = np.empty(0, dtype=int)

    source_coordinate = np.asarray(flattened[source_dimension])[source_indices]
    selected = flattened.isel({source_dimension: source_indices})
    if source_dimension != "candidate":
        selected = selected.rename({source_dimension: "candidate"})
    if "candidate" in selected.indexes:
        selected = selected.reset_index("candidate")
    selected = selected.assign_coords(candidate=np.arange(source_indices.size))
    if len(point_dimensions) == 1:
        source_name = (
            point_dimensions[0]
            if point_dimensions[0] != "candidate"
            else "source_candidate"
        )
        selected[source_name] = ("candidate", source_coordinate)
    selected["pareto_rank"] = ("candidate", ranks)
    selected["gate_mechanism"] = (
        "candidate",
        np.full(source_indices.size, mechanism, dtype="U16"),
    )
    selected["limiting_leakage_branch"] = (
        "candidate",
        leakage_label[source_indices],
    )
    selected["static_coherent_ceiling"] = (
        "candidate",
        ceiling[source_indices],
    )
    selected.attrs.update(
        {
            "gate_type": gate_type,
            "selection_method": "hard acceptance gates and non-dominated sorting",
            "minimum_coherent_ceiling": float(minimum_coherent_ceiling),
            "maximum_candidates": int(maximum_candidates),
            "retained_candidates": int(source_indices.size),
        }
    )
    return selected


def _local_refinement_axis(values: np.ndarray, center: float) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=float))
    if values.size == 1:
        return np.asarray([center])
    index = int(np.argmin(np.abs(values - center)))
    refined = [float(center)]
    if index > 0:
        refined.append(float(0.5 * (values[index - 1] + center)))
    if index + 1 < values.size:
        refined.append(float(0.5 * (center + values[index + 1])))
    return np.asarray(sorted(set(refined)))


def refine_static_gate_candidates(
    config: FiveTransmonConfig,
    contour_a: LogicalContour,
    contour_b: LogicalContour,
    coarse_landscape: xr.Dataset,
    gate_type: Literal["iswap", "cz"],
    *,
    seed_candidates: int = 3,
    cpus: int = 1,
    minimum_coherent_ceiling: float = 0.995,
    maximum_candidates: int = 20,
    minimum_sigma_min: float | None = None,
    minimum_leakage_gap_mhz: float | None = None,
    maximum_cz_offdiagonal_to_zz_ratio: float = 0.25,
    coherent_ceiling_max_time_ns: float | None = None,
    coherent_ceiling_time_samples: int | None = None,
    target_cphase_rad: float | None = None,
) -> xr.Dataset:
    """Adaptively rescan half-grid neighborhoods around coarse Pareto points."""

    if (
        not isinstance(seed_candidates, (int, np.integer))
        or isinstance(seed_candidates, bool)
        or seed_candidates < 0
    ):
        raise ValueError("seed_candidates must be a nonnegative integer")
    selection = {
        "minimum_coherent_ceiling": minimum_coherent_ceiling,
        "maximum_candidates": maximum_candidates,
        "minimum_sigma_min": minimum_sigma_min,
        "minimum_leakage_gap_mhz": minimum_leakage_gap_mhz,
        "maximum_cz_offdiagonal_to_zz_ratio": (
            maximum_cz_offdiagonal_to_zz_ratio
        ),
    }
    coarse = rank_static_gate_candidates(coarse_landscape, gate_type, **selection)
    seed_selection = {
        **selection,
        "minimum_coherent_ceiling": 0.0,
        "maximum_candidates": min(max(int(seed_candidates), 1), 30),
    }
    seeds = rank_static_gate_candidates(
        coarse_landscape,
        gate_type,
        **seed_selection,
    )
    if seed_candidates == 0 or seeds.sizes["candidate"] == 0:
        coarse.attrs["adaptive_seed_candidates"] = int(seed_candidates)
        coarse.attrs["adaptive_refined_regions"] = 0
        return coarse

    dimensions = (
        "coupler_flux",
        "logical_frequency_a_mhz",
        "logical_frequency_b_mhz",
    )
    missing = [name for name in dimensions if name not in coarse_landscape.coords]
    if missing:
        raise ValueError(f"coarse landscape is missing grid coordinates: {missing}")
    ceiling_time = float(
        coarse_landscape.attrs.get("coherent_ceiling_max_time_ns", 600.0)
        if coherent_ceiling_max_time_ns is None
        else coherent_ceiling_max_time_ns
    )
    ceiling_samples = int(
        coarse_landscape.attrs.get("coherent_ceiling_time_samples", 61)
        if coherent_ceiling_time_samples is None
        else coherent_ceiling_time_samples
    )
    cphase = float(
        coarse_landscape.attrs.get("target_cphase_rad", np.pi)
        if target_cphase_rad is None
        else target_cphase_rad
    )

    reports = [coarse]
    seed_count = min(int(seed_candidates), seeds.sizes["candidate"])
    for seed in range(seed_count):
        frequency_a = float(seeds.logical_frequency_a_mhz.isel(candidate=seed))
        frequency_b = float(seeds.logical_frequency_b_mhz.isel(candidate=seed))
        coupler_flux = float(seeds.coupler_flux.isel(candidate=seed))
        refined = run_static_interaction_landscape(
            config,
            contour_a,
            contour_b,
            _local_refinement_axis(
                coarse_landscape.logical_frequency_a_mhz.values, frequency_a
            ),
            _local_refinement_axis(
                coarse_landscape.logical_frequency_b_mhz.values, frequency_b
            ),
            _local_refinement_axis(coarse_landscape.coupler_flux.values, coupler_flux),
            cpus=cpus,
            coherent_ceiling_max_time_ns=ceiling_time,
            coherent_ceiling_time_samples=ceiling_samples,
            target_cphase_rad=cphase,
        )
        reports.append(rank_static_gate_candidates(refined, gate_type, **selection))

    combined = xr.concat(
        reports,
        dim="candidate",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="outer",
    )
    coordinates = np.column_stack(
        [np.asarray(combined[name], dtype=float) for name in dimensions]
    )
    _, unique_index = np.unique(coordinates, axis=0, return_index=True)
    combined = combined.isel(candidate=np.sort(unique_index))
    result = rank_static_gate_candidates(combined, gate_type, **selection)
    result.attrs["adaptive_seed_candidates"] = int(seed_candidates)
    result.attrs["adaptive_refined_regions"] = seed_count
    result.attrs["adaptive_refinement"] = "half-step axis neighborhoods"
    return result


def run_static_architecture_loop(
    architecture_cases: Mapping[
        str, tuple[FiveTransmonConfig, LogicalContour, LogicalContour]
    ],
    logical_frequency_a_mhz,
    logical_frequency_b_mhz,
    coupler_flux_values,
    *,
    cpus: int = 1,
    allow_parasitic_couplings: bool = False,
    coherent_ceiling_max_time_ns: float = 600.0,
    coherent_ceiling_time_samples: int = 61,
    target_cphase_rad: float = np.pi,
) -> xr.Dataset:
    """Evaluate an outer architecture loop with separately calibrated contours.

    Each named case supplies its own device configuration and pair contours, so
    callers can vary internal pair coupling, coupler EJ/EC, rail-coupler
    strengths, and intentional left/right asymmetry without reusing invalid
    contour calibrations.  Parasitic paths remain prohibited by default.
    """

    if not architecture_cases:
        raise ValueError("architecture_cases must contain at least one named case")
    landscapes = []
    fingerprints = []
    parasitic_names = ("a2_b1", "a1_c", "c_b2", "a1_b1", "a1_b2", "a2_b2")
    for name, case in architecture_cases.items():
        if not isinstance(name, str) or not name:
            raise ValueError("architecture case names must be nonempty strings")
        if len(case) != 3:
            raise ValueError(
                "each architecture case must contain config, contour_a, contour_b"
            )
        config, contour_a, contour_b = case
        if not allow_parasitic_couplings:
            enabled = [
                coupling_name
                for coupling_name in parasitic_names
                if abs(getattr(config.coupling, coupling_name))
                > np.finfo(float).eps
            ]
            if enabled:
                raise ValueError(
                    "parasitic couplings must remain disabled during the baseline "
                    f"architecture loop: {enabled}"
                )
        landscape = run_static_interaction_landscape(
            config,
            contour_a,
            contour_b,
            logical_frequency_a_mhz,
            logical_frequency_b_mhz,
            coupler_flux_values,
            cpus=cpus,
            coherent_ceiling_max_time_ns=coherent_ceiling_max_time_ns,
            coherent_ceiling_time_samples=coherent_ceiling_time_samples,
            target_cphase_rad=target_cphase_rad,
        )
        fingerprints.append(str(landscape.attrs["fingerprint"]))
        landscapes.append(landscape.expand_dims(architecture=[name]))
    result = xr.concat(
        landscapes,
        dim="architecture",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="exact",
    )
    result["architecture_fingerprint"] = (
        "architecture",
        np.asarray(fingerprints, dtype="U64"),
    )
    result.attrs.update(
        {
            "description": "Outer architecture loop over Pauli-aware static atlases.",
            "architecture_parameters": (
                "internal pair coupling, coupler EJ/EC, rail-coupler coupling, "
                "and intentional coupling asymmetry"
            ),
            "parasitic_couplings_enabled": int(allow_parasitic_couplings),
        }
    )
    return result


def load_or_compute_static_interaction_landscape(
    config: FiveTransmonConfig | None = None,
    contour_a: LogicalContour | None = None,
    contour_b: LogicalContour | None = None,
    logical_frequency_a_mhz=None,
    logical_frequency_b_mhz=None,
    coupler_flux_values=None,
    *,
    path=DATA_DIR / "dualrail_fsim_static_landscape.nc",
    force: bool = False,
    cpus: int = 1,
    coherent_ceiling_max_time_ns: float = 600.0,
    coherent_ceiling_time_samples: int = 61,
    target_cphase_rad: float = np.pi,
) -> tuple[xr.Dataset, Path]:
    """Load a fingerprint-validated landscape or compute and save it."""

    path = Path(path)
    inputs = (
        config,
        contour_a,
        contour_b,
        logical_frequency_a_mhz,
        logical_frequency_b_mhz,
        coupler_flux_values,
    )
    supplied = [item is not None for item in inputs]
    if any(supplied) and not all(supplied):
        raise ValueError(
            "all model, contour, and grid inputs must be supplied together"
        )

    if path.exists() and not force:
        dataset = xr.load_dataset(path)
        if (
            dataset.attrs.get("static_dualrail_fsim_schema")
            != STATIC_DUALRAIL_FSIM_SCHEMA
        ):
            dataset.close()
            raise ValueError(
                f"stale static landscape cache {path}; rerun with force=True"
            )
        if all(supplied):
            frequency_a = _finite_1d(logical_frequency_a_mhz, "logical_frequency_a_mhz")
            frequency_b = _finite_1d(logical_frequency_b_mhz, "logical_frequency_b_mhz")
            coupler_flux = _finite_1d(coupler_flux_values, "coupler_flux_values")
            expected = _landscape_fingerprint(
                config,
                contour_a,
                contour_b,
                frequency_a,
                frequency_b,
                coupler_flux,
                float(coherent_ceiling_max_time_ns),
                int(coherent_ceiling_time_samples),
                float(target_cphase_rad),
            )
            if dataset.attrs.get("fingerprint") != expected:
                dataset.close()
                raise ValueError(
                    f"static landscape cache {path} has different model, contours, "
                    "or grids; rerun with force=True"
                )
        return dataset, path

    if not all(supplied):
        raise ValueError(
            "model, contours, and grids are required to compute a landscape"
        )
    dataset = run_static_interaction_landscape(
        config,
        contour_a,
        contour_b,
        logical_frequency_a_mhz,
        logical_frequency_b_mhz,
        coupler_flux_values,
        cpus=cpus,
        coherent_ceiling_max_time_ns=coherent_ceiling_max_time_ns,
        coherent_ceiling_time_samples=coherent_ceiling_time_samples,
        target_cphase_rad=target_cphase_rad,
    )
    return dataset, save_dataset(dataset, path)


__all__ = [
    "DYNAMIC_DUALRAIL_FSIM_SCHEMA",
    "ISOLATED_LOGICAL_CONTOUR_SCHEMA",
    "LOGICAL_MODE_LABELS",
    "LOGICAL_STATE_LABELS",
    "MODE_CHARACTER_LABELS",
    "POPULATION_TRAJECTORY_SCHEMA",
    "STATIC_DUALRAIL_FSIM_SCHEMA",
    "DualRailCouplingConfig",
    "DualRailFSimModel",
    "DualRailPopulationTrajectory",
    "DualRailPulse",
    "DualRailPulseConstraints",
    "DualRailPulseResult",
    "FiveTransmonConfig",
    "IsolatedPairModes",
    "LogicalContour",
    "LogicalBranchTrackingResult",
    "LogicalModeMapping",
    "ReducedPairCalibration",
    "StaticInteractionResult",
    "StaticCoherentCeilingResult",
    "StaticDynamicRateComparison",
    "average_gate_fidelity",
    "assign_dressed_logical_states",
    "build_dualrail_fsim_model",
    "build_logical_contour",
    "calibrate_reduced_pair",
    "calibrate_reduced_pair_multi_point",
    "closest_unitary",
    "compare_static_and_dynamic_rates",
    "dualrail_idle_score",
    "extract_fsim_angles",
    "evaluate_static_interaction",
    "evaluate_static_coherent_ceiling",
    "local_z_optimized_fidelity",
    "load_or_compute_static_interaction_landscape",
    "run_static_interaction_landscape",
    "run_static_architecture_loop",
    "rank_static_gate_candidates",
    "refine_static_gate_candidates",
    "select_dualrail_idle_point",
    "simulate_dualrail_population_trajectory",
    "simulate_dualrail_pulse",
    "track_logical_branches",
    "track_dualrail_model_path",
    "validate_fixed_basis_along_path",
]
