"""One device definition for reference circuits and calibrated logical gates."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from hashlib import sha256

import numpy as np
import scqubits as scq
import xarray as xr

from dtc_dualrail.circuits import gen_five_transmon_dual_rail, gen_tdt
from dtc_dualrail.dualrail_fsim import (
    DualRailCouplingConfig,
    FiveTransmonConfig,
    LogicalContour,
    LogicalModeMapping,
    ReducedPairCalibration,
    calibrate_reduced_pair_multi_point,
)
from dtc_dualrail.fsim import TunableTransmonConfig

SINGLE_TRANSMON_SWEEP_SCHEMA = "single-transmon-spectrum-v1"


def run_single_transmon_sweep(
    config: TunableTransmonConfig,
    flux_values,
    *,
    ncut_values=(15, 20, 25, 30),
    convergence_tolerance_khz: float = 50.0,
) -> xr.Dataset:
    """Evaluate one transmon spectrum and its charge-basis convergence.

    Frequencies are returned in GHz and adjacent-``ncut`` changes in kHz.  The
    convergence gate covers both transitions at every requested flux point.
    """

    flux = np.asarray(flux_values, dtype=float)
    ncut = np.asarray(ncut_values)
    if flux.ndim != 1 or flux.size == 0 or np.any(~np.isfinite(flux)):
        raise ValueError("flux_values must be a nonempty finite one-dimensional array")
    if (
        ncut.ndim != 1
        or ncut.size < 2
        or np.issubdtype(ncut.dtype, np.bool_)
        or np.any(ncut != ncut.astype(int))
        or np.any(ncut < 2)
    ):
        raise ValueError("ncut_values must contain at least two integers of at least 2")
    ncut = ncut.astype(int)
    if np.any(np.diff(ncut) <= 0):
        raise ValueError("ncut_values must be strictly increasing")
    if (
        not np.isfinite(convergence_tolerance_khz)
        or convergence_tolerance_khz <= 0
    ):
        raise ValueError("convergence_tolerance_khz must be positive and finite")

    shape = (ncut.size, flux.size)
    f01 = np.empty(shape)
    f12 = np.empty(shape)
    n01 = np.empty(shape)
    n12 = np.empty(shape)
    for ncut_index, cutoff in enumerate(ncut):
        for flux_index, flux_value in enumerate(flux):
            transmon = scq.TunableTransmon(
                EJmax=config.EJmax,
                EC=config.EC,
                d=config.d,
                flux=float(flux_value),
                ng=config.ng,
                ncut=int(cutoff),
                truncated_dim=max(config.truncated_dim, 3),
                id_str="single_transmon_validation",
            )
            energies, states = transmon.eigensys(evals_count=3)
            energies = np.asarray(energies, dtype=float)
            charge = np.asarray(transmon.n_operator(), dtype=complex)
            states = np.asarray(states, dtype=complex)
            charge_energy_basis = states.conj().T @ charge @ states
            f01[ncut_index, flux_index] = energies[1] - energies[0]
            f12[ncut_index, flux_index] = energies[2] - energies[1]
            n01[ncut_index, flux_index] = abs(charge_energy_basis[0, 1])
            n12[ncut_index, flux_index] = abs(charge_energy_basis[1, 2])

    transition_frequency = np.stack((f01, f12), axis=-1)
    ncut_delta_khz = 1e6 * np.abs(np.diff(transition_frequency, axis=0))
    maximum_delta_khz = float(np.max(ncut_delta_khz[-1]))
    dataset = xr.Dataset(
        data_vars={
            "f01_ghz": (("ncut", "flux"), f01),
            "f12_ghz": (("ncut", "flux"), f12),
            "anharmonicity_ghz": (("ncut", "flux"), f12 - f01),
            "n01": (("ncut", "flux"), n01),
            "n12": (("ncut", "flux"), n12),
            "ncut_delta_khz": (
                ("ncut_step", "flux", "transition"),
                ncut_delta_khz,
            ),
        },
        coords={
            "ncut": ncut,
            "ncut_step": ncut[1:],
            "flux": flux,
            "transition": ["f01", "f12"],
        },
        attrs={
            "single_transmon_sweep_schema": SINGLE_TRANSMON_SWEEP_SCHEMA,
            "convergence_tolerance_khz": float(convergence_tolerance_khz),
            "maximum_final_ncut_delta_khz": maximum_delta_khz,
            "ncut_converged": int(maximum_delta_khz < convergence_tolerance_khz),
            "transmon_config": json.dumps(asdict(config), sort_keys=True),
        },
    )
    for name in ("f01_ghz", "f12_ghz", "anharmonicity_ghz"):
        dataset[name].attrs["units"] = "GHz"
    dataset.ncut_delta_khz.attrs["units"] = "kHz"
    dataset.flux.attrs["units"] = "flux quantum"
    return dataset


@dataclass(frozen=True)
class TDTPairConfig:
    """Physical inputs for an isolated tunable dual rail and its resonator.

    Names and units match ``gen_tdt``: Josephson energies are in GHz;
    ``C1`` and ``C2`` are shunt *charging energies in MHz*, not capacitances.
    All other ``C*`` fields are capacitances in fF; ``Lr`` is in nH.
    Reduced rail EJmax and asymmetry are derived from the two junctions.
    Reduced EC and charge-charge coupling must be fitted to a reference contour.
    """

    Ej1: float
    Cj1: float
    Ej1_2: float
    Cj1_2: float
    C1: float
    Ej2: float
    Cj2: float
    Ej2_2: float
    Cj2_2: float
    C2: float
    C12: float
    C1r: float
    C2r: float
    Cr: float
    Lr: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("C1", "C2", "Cr", "Lr"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if any(value == 0 for value in self.ejmax_ghz):
            raise ValueError("each rail must have positive total Josephson energy")

    @property
    def ejmax_ghz(self) -> tuple[float, float]:
        return self.Ej1 + self.Ej1_2, self.Ej2 + self.Ej2_2

    @property
    def asymmetry(self) -> tuple[float, float]:
        return (
            abs(self.Ej1 - self.Ej1_2) / self.ejmax_ghz[0],
            abs(self.Ej2 - self.Ej2_2) / self.ejmax_ghz[1],
        )

    def build_circuit(self):
        """Build the reference circuit using the existing hierarchical basis."""
        return gen_tdt(**asdict(self))

    def fingerprint(self) -> str:
        """Identify physical inputs for reference-cache filenames."""
        payload = {name: float(value) for name, value in asdict(self).items()}
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class LogicalFSimDeviceConfig:
    """Authoritative inputs for the circuit-to-five-transmon workflow.

    Pair junction values supply both reference circuits and reduced rail physics.
    The coupler links are specified in both reduced-model units (GHz) and direct
    reference-circuit capacitances (fF). Rail EC and reduced intra-pair links are
    calibration outputs. ``rail_ncut`` and ``rail_truncated_dim`` control only
    the reduced basis.
    """

    pair_a: TDTPairConfig
    pair_b: TDTPairConfig
    coupler: TunableTransmonConfig
    a2_c_ghz: float = 0.1
    c_b1_ghz: float = 0.1
    a2_c_ff: float = 3.0
    c_b1_ff: float = 3.0
    coupler_junction_capacitance_ff: tuple[float, float] = (0.2, 0.2)
    a2_b1_ff: float = 0.0
    anchor_logical_frequency_mhz: float = 125.0
    rail_ncut: int = 15
    rail_truncated_dim: int = 3
    logical_mapping: LogicalModeMapping = field(default_factory=LogicalModeMapping)
    min_assignment_probability: float = 0.5
    min_leakage_gap_mhz: float = 1.0
    calibration_residual_tolerance_fraction: float = 0.1

    def __post_init__(self) -> None:
        for name in ("a2_c_ghz", "c_b1_ghz"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("a2_c_ff", "c_b1_ff"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.a2_b1_ff) or self.a2_b1_ff < 0:
            raise ValueError("a2_b1_ff must be nonnegative and finite")
        if (
            len(self.coupler_junction_capacitance_ff) != 2
            or any(
                not math.isfinite(value) or value <= 0
                for value in self.coupler_junction_capacitance_ff
            )
        ):
            raise ValueError(
                "coupler_junction_capacitance_ff must contain two positive finite values"
            )
        if (
            not math.isfinite(self.anchor_logical_frequency_mhz)
            or self.anchor_logical_frequency_mhz <= 0
        ):
            raise ValueError("anchor_logical_frequency_mhz must be positive and finite")
        for name, minimum in (("rail_ncut", 1), ("rail_truncated_dim", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if self.rail_truncated_dim > 2 * self.rail_ncut + 1:
            raise ValueError("rail_truncated_dim exceeds the charge basis dimension")
        self.logical_mapping.validate()
        if (
            not math.isfinite(self.calibration_residual_tolerance_fraction)
            or not 0 < self.calibration_residual_tolerance_fraction < 1
        ):
            raise ValueError(
                "calibration_residual_tolerance_fraction must lie between 0 and 1"
            )

    def calibrate(
        self, contour_a: LogicalContour, contour_b: LogicalContour
    ) -> tuple[FiveTransmonConfig, ReducedPairCalibration, ReducedPairCalibration]:
        """Return the derived simulation config and two full-contour fit reports.

        Supply contours computed from this definition's reference circuits.
        Reference circuits use zero offset charge, as do the reduced rails.
        """
        calibrations = [
            calibrate_reduced_pair_multi_point(
                contour,
                anchor_logical_frequency_mhz=self.anchor_logical_frequency_mhz,
                ejmax_ghz=pair.ejmax_ghz,
                asymmetry=pair.asymmetry,
                ncut=self.rail_ncut,
                truncated_dim=self.rail_truncated_dim,
                residual_tolerance_fraction=(
                    self.calibration_residual_tolerance_fraction
                ),
            )
            for pair, contour in ((self.pair_a, contour_a), (self.pair_b, contour_b))
        ]
        calibration_a, calibration_b = calibrations
        config = FiveTransmonConfig(
            rail_a1=calibration_a.rail_1,
            rail_a2=calibration_a.rail_2,
            coupler=self.coupler,
            rail_b1=calibration_b.rail_1,
            rail_b2=calibration_b.rail_2,
            coupling=DualRailCouplingConfig(
                a1_a2=calibration_a.coupling_ghz,
                a2_c=self.a2_c_ghz,
                c_b1=self.c_b1_ghz,
                b1_b2=calibration_b.coupling_ghz,
            ),
            logical_mapping=self.logical_mapping,
            parameter_source=json.dumps(asdict(self), sort_keys=True),
            min_assignment_probability=self.min_assignment_probability,
            min_leakage_gap_mhz=self.min_leakage_gap_mhz,
        )
        return config, calibration_a, calibration_b

    @staticmethod
    def requires_direct_reference(
        calibration_a: ReducedPairCalibration,
        calibration_b: ReducedPairCalibration,
    ) -> bool:
        """Return whether either reduced pair missed its credibility gate."""

        return not (
            calibration_a.residual_within_tolerance
            and calibration_b.residual_within_tolerance
        )

    def build_reference_circuit(self):
        """Build the direct five-transmon static source-of-truth circuit."""

        pairs = (self.pair_a, self.pair_b)
        rail_ej = (
            (pairs[0].Ej1, pairs[0].Ej1_2),
            (pairs[0].Ej2, pairs[0].Ej2_2),
            (pairs[1].Ej1, pairs[1].Ej1_2),
            (pairs[1].Ej2, pairs[1].Ej2_2),
        )
        rail_cj = (
            (pairs[0].Cj1, pairs[0].Cj1_2),
            (pairs[0].Cj2, pairs[0].Cj2_2),
            (pairs[1].Cj1, pairs[1].Cj1_2),
            (pairs[1].Cj2, pairs[1].Cj2_2),
        )
        coupler_ej = (
            0.5 * self.coupler.EJmax * (1 + self.coupler.d),
            0.5 * self.coupler.EJmax * (1 - self.coupler.d),
        )
        parasitic = {"A2_B1": self.a2_b1_ff} if self.a2_b1_ff > 0 else None
        return gen_five_transmon_dual_rail(
            rail_ej_pairs_ghz=rail_ej,
            rail_cj_pairs_ff=rail_cj,
            rail_charging_energy_ghz=(
                self.pair_a.C1 / 1e3,
                self.pair_a.C2 / 1e3,
                self.pair_b.C1 / 1e3,
                self.pair_b.C2 / 1e3,
            ),
            coupler_ej_pair_ghz=coupler_ej,
            coupler_cj_pair_ff=self.coupler_junction_capacitance_ff,
            coupler_charging_energy_ghz=self.coupler.EC,
            coupling_capacitance_ff={
                "A1_A2": self.pair_a.C12,
                "A2_C": self.a2_c_ff,
                "C_B1": self.c_b1_ff,
                "B1_B2": self.pair_b.C12,
            },
            parasitic_capacitance_ff=parasitic,
        )


__all__ = [
    "SINGLE_TRANSMON_SWEEP_SCHEMA",
    "LogicalFSimDeviceConfig",
    "TDTPairConfig",
    "run_single_transmon_sweep",
]
