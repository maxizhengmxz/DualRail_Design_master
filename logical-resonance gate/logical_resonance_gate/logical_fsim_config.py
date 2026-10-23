"""Versioned configuration for the five-transmon logical-fSim workflow."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .device import LogicalFSimDeviceConfig, TDTPairConfig
from .dualrail_fsim import LogicalModeMapping
from .fsim import SolverConfig, TunableTransmonConfig

LOGICAL_FSIM_WORKFLOW_SCHEMA = "logical-fsim-workflow-v3"


def _canonical(value: Any) -> Any:
    """Return JSON-serializable values with deterministic container ordering."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_fingerprint(payload: Any) -> str:
    """Hash a configuration or cache payload using one canonical encoding."""

    encoded = json.dumps(
        _canonical(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LogicalFSimWorkflowConfig:
    """Validated TOML-backed inputs for every logical-fSim notebook stage."""

    source: Path
    values: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> LogicalFSimWorkflowConfig:
        source = Path(path)
        with source.open("rb") as handle:
            values = tomllib.load(handle)
        config = cls(source=source, values=values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.values.get("schema") != LOGICAL_FSIM_WORKFLOW_SCHEMA:
            raise ValueError(
                f"workflow config must use schema {LOGICAL_FSIM_WORKFLOW_SCHEMA!r}"
            )
        required = {
            "execution",
            "thresholds",
            "pair_a",
            "pair_b",
            "coupler",
            "model",
            "reference",
            "contours",
            "static",
            "idle",
            "gates",
            "pulse",
            "trajectory",
            "phase1",
            "phase2",
        }
        missing = sorted(required - self.values.keys())
        if missing:
            raise ValueError(f"workflow config is missing sections {missing}")
        if self.values["reference"].get("mode") not in {"computed", "configured"}:
            raise ValueError("reference.mode must be 'computed' or 'configured'")
        self.device_config()
        self.solver_config().validate()
        for gate in ("iswap", "cz"):
            if gate not in self.values["gates"]:
                raise ValueError(f"workflow config is missing gates.{gate}")
        phase1 = self.section("phase1")
        ncut_values = np.asarray(phase1["ncut_values"])
        if ncut_values.shape != (4,) or np.any(
            ncut_values != np.asarray([15, 20, 25, 30])
        ):
            raise ValueError("phase1.ncut_values must be [15, 20, 25, 30]")
        for name in ("rail_flux_linspace", "coupler_flux_linspace"):
            values = phase1[name]
            if len(values) != 3 or int(values[2]) < 1:
                raise ValueError(f"phase1.{name} must be [start, stop, count]")
        pulse_coupler_dim = phase1["pulse_coupler_truncated_dim"]
        static_coupler_dim = self.section("coupler", "truncated_dim")
        if (
            isinstance(pulse_coupler_dim, bool)
            or not isinstance(pulse_coupler_dim, int)
            or pulse_coupler_dim < static_coupler_dim
        ):
            raise ValueError(
                "phase1.pulse_coupler_truncated_dim must be an integer no smaller "
                "than coupler.truncated_dim"
            )
        phase2 = self.section("phase2")
        if (
            not np.isfinite(phase2["coherent_ceiling_max_time_ns"])
            or phase2["coherent_ceiling_max_time_ns"] <= 0
        ):
            raise ValueError(
                "phase2.coherent_ceiling_max_time_ns must be positive and finite"
            )
        samples = phase2["coherent_ceiling_time_samples"]
        if (
            isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples < 3
        ):
            raise ValueError(
                "phase2.coherent_ceiling_time_samples must be an integer of at least 3"
            )
        candidate_count = phase2["candidate_count"]
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or not 1 <= candidate_count <= 30
        ):
            raise ValueError("phase2.candidate_count must be an integer from 1 to 30")
        adaptive_seeds = phase2["adaptive_seed_candidates"]
        if (
            isinstance(adaptive_seeds, bool)
            or not isinstance(adaptive_seeds, int)
            or adaptive_seeds < 0
        ):
            raise ValueError(
                "phase2.adaptive_seed_candidates must be a nonnegative integer"
            )
        ceiling = phase2["minimum_coherent_ceiling"]
        if not np.isfinite(ceiling) or not 0 <= ceiling <= 1:
            raise ValueError("phase2.minimum_coherent_ceiling must lie in [0, 1]")
        ratio = phase2["maximum_cz_offdiagonal_to_zz_ratio"]
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError(
                "phase2.maximum_cz_offdiagonal_to_zz_ratio must be positive and finite"
            )
        if not isinstance(phase2["enforce_exit_gate"], bool):
            raise ValueError("phase2.enforce_exit_gate must be a boolean")

    def section(self, *path: str) -> Any:
        value: Any = self.values
        for name in path:
            try:
                value = value[name]
            except (KeyError, TypeError) as error:
                dotted = ".".join(path)
                raise KeyError(f"workflow config has no {dotted!r} value") from error
        return value

    def fingerprint(self, *sections: str, extra: Any | None = None) -> str:
        """Fingerprint named top-level sections plus derived artifact inputs."""

        unknown = sorted(set(sections) - self.values.keys())
        if unknown:
            raise KeyError(f"workflow config has no sections {unknown}")
        payload = {
            "schema": LOGICAL_FSIM_WORKFLOW_SCHEMA,
            "sections": {name: self.values[name] for name in sorted(sections)},
            "extra": extra,
        }
        return stable_fingerprint(payload)

    def device_config(self) -> LogicalFSimDeviceConfig:
        model = self.section("model")
        mapping = LogicalModeMapping(**model.get("logical_mapping", {}))
        return LogicalFSimDeviceConfig(
            pair_a=TDTPairConfig(**self.section("pair_a")),
            pair_b=TDTPairConfig(**self.section("pair_b")),
            coupler=TunableTransmonConfig(**self.section("coupler")),
            a2_c_ghz=float(model["a2_c_ghz"]),
            c_b1_ghz=float(model["c_b1_ghz"]),
            a2_c_ff=float(model.get("a2_c_ff", 3.0)),
            c_b1_ff=float(model.get("c_b1_ff", 3.0)),
            coupler_junction_capacitance_ff=tuple(
                map(float, model.get("coupler_junction_capacitance_ff", (0.2, 0.2)))
            ),
            a2_b1_ff=float(model.get("a2_b1_ff", 0.0)),
            anchor_logical_frequency_mhz=float(model["anchor_logical_frequency_mhz"]),
            rail_ncut=int(model["rail_ncut"]),
            rail_truncated_dim=int(model["rail_truncated_dim"]),
            logical_mapping=mapping,
            min_assignment_probability=float(model["min_assignment_probability"]),
            min_leakage_gap_mhz=float(model["min_leakage_gap_mhz"]),
            calibration_residual_tolerance_fraction=float(
                self.section("phase1", "calibration_tolerance_fraction")
            ),
        )

    def solver_config(self) -> SolverConfig:
        return SolverConfig(**self.section("pulse", "solver"))

    def reference_grid(self) -> dict[str, Any]:
        reference = self.section("reference")
        return {
            "phi_1_values": np.linspace(*reference["phi_1_linspace"]),
            "phi_2_values": np.linspace(*reference["phi_2_linspace"]),
            "evals_count": int(reference["evals_count"]),
        }

    def static_grids(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        static = self.section("static")
        return (
            np.asarray(static["frequency_a_mhz"], dtype=float),
            np.asarray(static["frequency_b_mhz"], dtype=float),
            np.linspace(*static["coupler_flux_linspace"]),
        )

    def phase1_settings(self) -> dict[str, Any]:
        """Return validated numerical settings for the Phase 1 gates."""

        return dict(self.section("phase1"))

    def phase2_settings(self) -> dict[str, Any]:
        """Return validated Pauli-aware static-atlas settings."""

        return dict(self.section("phase2"))

    def configured_operating_points(self, pair: str) -> xr.Dataset:
        """Build deterministic reference points used by the smoke configuration."""

        if self.section("reference", "mode") != "configured":
            raise ValueError("configured operating points require reference.mode")
        values = self.section("configured_contours", pair)
        frequency = np.asarray(values["logical_frequency_mhz"], dtype=float)
        count = frequency.size
        if count < 2:
            raise ValueError("a configured contour needs at least two points")
        arrays = {
            name: np.asarray(values[name], dtype=float)
            for name in ("phi_1", "phi_2", "carrier_frequency_ghz")
        }
        if any(item.shape != frequency.shape for item in arrays.values()):
            raise ValueError("configured contour arrays must have equal lengths")
        data_vars = {
            "phi_1": ("operating_point", arrays["phi_1"]),
            "phi_2": ("operating_point", arrays["phi_2"]),
            "splitting_ghz": ("operating_point", frequency / 1e3),
            "operating_frequency_ghz": (
                "operating_point",
                arrays["carrier_frequency_ghz"],
            ),
            "hybridization": (
                "operating_point",
                np.asarray(values.get("hybridization", [0.99] * count)),
            ),
            "polarization": (
                "operating_point",
                np.asarray(values.get("polarization", [0.0] * count)),
            ),
        }
        if "two_excitation_frequency_ghz" in values:
            data_vars["two_excitation_frequency_ghz"] = (
                ("operating_point", "two_excitation_state"),
                np.asarray(values["two_excitation_frequency_ghz"], dtype=float),
            )
        if "charge_matrix_element" in values:
            data_vars["charge_matrix_element"] = (
                ("operating_point", "physical_rail", "rail_state"),
                np.asarray(values["charge_matrix_element"], dtype=float),
            )
        return xr.Dataset(
            data_vars=data_vars,
            attrs={"flux_param_1": "Phi_1", "flux_param_2": "Phi_2"},
        )


def load_logical_fsim_workflow_config(
    path: str | Path,
) -> LogicalFSimWorkflowConfig:
    """Load and validate a versioned logical-fSim workflow configuration."""

    return LogicalFSimWorkflowConfig.load(path)


__all__ = [
    "LOGICAL_FSIM_WORKFLOW_SCHEMA",
    "LogicalFSimWorkflowConfig",
    "load_logical_fsim_workflow_config",
    "stable_fingerprint",
]
