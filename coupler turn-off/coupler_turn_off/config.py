"""Validate explicit user inputs before constructing an expensive Hamiltonian."""

import hashlib
import json
import math
import re
from pathlib import Path

OBJECTIVES = ("physical", "logical", "combined")
PAIR_KEYS = (
    "ej1_ghz",
    "ej2_ghz",
    "loaded_ec1_mhz",
    "loaded_ec2_mhz",
    "splitting_ghz",
    "chi_m_ghz",
    "chi_p_ghz",
)
DEVICE_KEYS = (
    "C1_shunt_fF",
    "C2_shunt_fF",
    "C12_fF",
    "C1r_fF",
    "C2r_fF",
    "Cr_fF",
    "Lr_nH",
)
SCREEN_KEYS = (
    "chi_ratio_max",
    "abs_mean_chi_min_mhz",
    "hybridization_min",
    "dispersive_max",
    "logical_frequency_bounds_mhz",
    "operation_frequency_bounds_ghz",
    "tracking_min",
    "logical_subspace_min",
    "readout_subspace_min",
    "coupler_frequency_bounds_ghz",
    "rail_ec_bounds_mhz",
    "rail_ej_max_ghz",
    "rail_ej_over_ec_min",
    "coupler_ec_max_mhz",
    "coupler_ej_max_ghz",
)


def fingerprint(config):
    """Cache identity includes the complete supplied configuration."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:16]


def validate(config):
    """Return actionable input errors; never infer a prior case's parameters."""
    errors = []
    if not isinstance(config, dict):
        return ["Configuration must be a JSON object"]
    for section in ("direct_a2_b1", "screens"):
        if section in config and not isinstance(config[section], dict):
            errors.append(f"{section} must be a JSON object")
    pair = config.get("pair")
    if isinstance(pair, dict) and not isinstance(pair.get("device", {}), dict):
        errors.append("pair.device must be a JSON object")
    candidates = config.get("coupler_candidates")
    if candidates is not None and (
        not isinstance(candidates, list)
        or not all(isinstance(c, dict) for c in candidates)
    ):
        errors.append("coupler_candidates must be a list of objects")
    for key in (
        "ej_search_bounds_ghz",
        "basis",
        "audit_basis",
        "readout_window_ghz",
        "resonator_inductances_nh",
    ):
        if config.get(key) is not None and not isinstance(config[key], list):
            errors.append(f"{key} must be a list")
    if errors:
        return errors

    def need(mapping, key, path, nullable=False):
        if key not in mapping or (mapping[key] is None and not nullable):
            errors.append(f"Supply {path}.{key}")
            return None
        return mapping[key]

    if config.get("schema") != "coupler-off-v1":
        errors.append("schema must be coupler-off-v1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(config.get("case_id", ""))):
        errors.append("case_id must contain only letters, digits, underscore or hyphen")
    pair = config.get("pair")
    if not isinstance(pair, dict):
        errors.append(
            "Request pair: the externally readout-optimized pair, including its device and readout baseline"
        )
    else:
        for key in PAIR_KEYS:
            need(pair, key, "pair")
        for key in DEVICE_KEYS:
            need(pair.get("device", {}), key, "pair.device")
    direct = config.get("direct_a2_b1", {})
    supplied = [
        key
        for key in ("capacitance_ff", "charge_coupling_ghz", "j01_mhz", "bare_exchange_mhz")
        if direct.get(key) is not None
    ]
    if len(supplied) != 1:
        errors.append(
            "Request exactly one direct_a2_b1.capacitance_ff, .charge_coupling_ghz (J), or .j01_mhz (local exchange); .bare_exchange_mhz is a legacy J01 alias"
        )
    elif not isinstance(direct[supplied[0]], (int, float)) or direct[supplied[0]] < 0:
        errors.append(
            "Direct grounded capacitive coupling must be a nonnegative magnitude"
        )
    candidates = config.get("coupler_candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append(
            "Supply coupler_candidates: one or more EC/link-capacitor geometries to test"
        )
    else:
        ids = []
        for i, candidate in enumerate(candidates):
            ids.append(candidate.get("id"))
            for key in ("id", "ec_ghz", "links_ff"):
                need(candidate, key, f"coupler_candidates[{i}]")
            if (
                not isinstance(candidate.get("links_ff"), list)
                or len(candidate["links_ff"]) != 2
            ):
                errors.append(f"coupler_candidates[{i}].links_ff must be [A2-C, C-B1]")
        if None in ids or len(set(ids)) != len(ids):
            errors.append("Candidate ids must be present and unique")
    for key in SCREEN_KEYS:
        need(
            config.get("screens", {}),
            key,
            "screens",
            nullable=key == "coupler_ej_max_ghz",
        )
    for key in (
        "resonator_inductances_nh",
        "ej_search_bounds_ghz",
        "basis",
        "audit_basis",
        "readout_window_ghz",
    ):
        need(config, key, "config")
    for key, length in (
        ("resonator_inductances_nh", 2),
        ("ej_search_bounds_ghz", 5),
        ("basis", 4),
        ("audit_basis", 4),
        ("readout_window_ghz", 2),
    ):
        value = config.get(key)
        if value is not None and (not isinstance(value, list) or len(value) != length):
            errors.append(f"{key} must contain {length} entries")

    def numeric_tree(value, path):
        if isinstance(value, dict):
            for k, v in value.items():
                numeric_tree(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                numeric_tree(v, f"{path}[{i}]")
        elif isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{path} must be finite")

    numeric_tree(config, "config")

    # Shape/type checks here prevent opaque SciPy failures after a long run.
    def number(value, path, positive=True):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or (positive and value <= 0)
        ):
            errors.append(
                f"{path} must be a finite {'positive ' if positive else ''}number"
            )
            return False
        return True

    def interval(value, path):
        if not isinstance(value, list) or len(value) != 2:
            errors.append(f"{path} must be [lower, upper]")
        elif all(number(v, path) for v in value) and value[0] >= value[1]:
            errors.append(f"{path}: lower must be smaller than upper")

    if isinstance(pair, dict):
        for key in PAIR_KEYS:
            if pair.get(key) is not None:
                number(pair[key], "pair." + key, positive=not key.startswith("chi_"))
        for key in DEVICE_KEYS:
            value = pair.get("device", {}).get(key)
            if value is not None:
                number(
                    value,
                    "pair.device." + key,
                    positive=key in ("C1_shunt_fF", "C2_shunt_fF", "Cr_fF", "Lr_nH"),
                )
                if isinstance(value, (int, float)) and value < 0:
                    errors.append(f"pair.device.{key} cannot be negative")
    for candidate in candidates or []:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(candidate.get("id", ""))):
            errors.append(
                "Candidate ids must contain only letters, digits, underscore or hyphen"
            )
        if candidate.get("ec_ghz") is not None:
            number(candidate["ec_ghz"], "candidate.ec_ghz")
        for link in candidate.get("links_ff") or []:
            number(link, "candidate.links_ff")
    for key, value in config.get("screens", {}).items():
        if key not in SCREEN_KEYS:
            errors.append(f"Unknown screen: {key}")
            continue
        if value is None:
            continue
        if key.endswith("_bounds_mhz") or key.endswith("_bounds_ghz"):
            interval(value, "screens." + key)
        else:
            number(value, "screens." + key)
            if (
                key
                in (
                    "chi_ratio_max",
                    "hybridization_min",
                    "tracking_min",
                    "logical_subspace_min",
                    "readout_subspace_min",
                )
                and isinstance(value, (int, float))
                and value > 1
            ):
                errors.append(f"screens.{key} is a fraction, not a percentage")
    for bound in config.get("ej_search_bounds_ghz") or []:
        interval(bound, "ej_search_bounds_ghz")
    if config.get("readout_window_ghz") is not None:
        interval(config["readout_window_ghz"], "readout_window_ghz")
    for key in ("basis", "audit_basis"):
        for v in config.get(key) or []:
            if not isinstance(v, int) or isinstance(v, bool) or v < 3:
                errors.append(f"{key} entries must be integer dimensions >=3")
        value = config.get(key)
        if (
            isinstance(value, list)
            and len(value) == 4
            and all(isinstance(v, int) for v in value)
        ):
            if value[2] > value[0] ** 2 * value[1]:
                errors.append(
                    f"{key}: retained pair count exceeds the pair Hilbert dimension"
                )
    for value in config.get("resonator_inductances_nh") or []:
        number(value, "resonator_inductances_nh")
    ceiling = config.get("spectral_ceiling_ghz")
    if ceiling is not None:
        number(ceiling, "spectral_ceiling_ghz")
    spec = config.get("sweep")
    if spec is not None and not isinstance(spec, dict):
        errors.append("sweep must be null or an object with explicit axes")
    if isinstance(spec, dict):
        for key in ("fa_bare_ghz", "fc_bare_ghz"):
            v = spec.get(key)
            if not isinstance(v, list) or len(v) != 3:
                errors.append(f"sweep.{key} must be [low, high, count]")
            else:
                interval(v[:2], "sweep." + key)
                if not isinstance(v[2], int) or isinstance(v[2], bool) or v[2] < 2:
                    errors.append(f"sweep.{key}: count must be integer >=2")
        number(
            spec.get("a1_ej_search_half_width_ghz"), "sweep.a1_ej_search_half_width_ghz"
        )
    return errors


def load(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate(config)
    if errors:
        raise ValueError("Missing/invalid inputs:\n- " + "\n- ".join(errors))
    return config
