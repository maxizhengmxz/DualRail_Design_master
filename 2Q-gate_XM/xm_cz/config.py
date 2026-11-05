"""Validate the public physics inputs before launching numerical workers."""

import math

from .channel_policy import LOGICAL_ORDER

LABELS = ("A1", "A2", "B1", "B2")


def validate_payload(payload: dict) -> None:
    """Reject malformed inputs; physical infeasibility is a separate outcome."""
    ec = payload.get("ec_GHz", [])
    if len(ec) != 4 or not all(math.isfinite(x) and 0 < x < 0.400 for x in ec):
        raise ValueError("ec_GHz requires ordered A1,A2,B1,B2 values in (0,0.400) GHz")
    splits = payload.get("logical_splittings_GHz", [0.150, 0.090])
    if len(splits) != 2 or not all(math.isfinite(x) and x > 0 for x in splits):
        raise ValueError("logical_splittings_GHz requires two positive targets: A,B")
    g = payload.get("g_GHz", 0.005)
    if not math.isfinite(g) or g <= 0:
        raise ValueError("g_GHz must be positive and finite")
    if payload.get("resonance_channel", "0020") not in ("0020", "0200"):
        raise ValueError("resonance_channel must be 0020 or 0200")
    sources = payload.get("logical_sources", list(LOGICAL_ORDER))
    if (
        not isinstance(sources, list)
        or not sources
        or len(set(sources)) != len(sources)
        or any(s not in LOGICAL_ORDER for s in sources)
    ):
        raise ValueError(
            "logical_sources must be a nonempty unique list of logical labels in A,B order"
        )
    fixed_a = payload.get("fixed_A")
    if fixed_a is not None:
        required = {"EJ_GHz", "J_GHz", "operating_frequency_GHz"}
        if not isinstance(fixed_a, dict) or not required <= fixed_a.keys():
            raise ValueError(
                "fixed_A requires EJ_GHz, J_GHz and operating_frequency_GHz"
            )
        if len(fixed_a["EJ_GHz"]) != 2 or not all(
            math.isfinite(x) and 0 < x < 30 for x in fixed_a["EJ_GHz"]
        ):
            raise ValueError("fixed_A.EJ_GHz requires two finite values in (0,30)")
        if not math.isfinite(fixed_a["J_GHz"]) or not 0 < fixed_a["J_GHz"] <= 0.25:
            raise ValueError("fixed_A.J_GHz must be in (0,0.25]")
        if (
            not math.isfinite(fixed_a["operating_frequency_GHz"])
            or fixed_a["operating_frequency_GHz"] <= 0
        ):
            raise ValueError(
                "fixed_A.operating_frequency_GHz must be finite and positive"
            )
        interval = fixed_a.get("logical_frequency_bounds_GHz", [0.14, 0.16])
        if (
            len(interval) != 2
            or not all(math.isfinite(x) for x in interval)
            or not 0 < interval[0] < interval[1]
        ):
            raise ValueError(
                "fixed_A logical-frequency bounds must be positive and increasing"
            )
    bounds = payload.get("bare_f01_bounds_GHz", {k: [4.0, 5.0] for k in LABELS})
    if set(bounds) != set(LABELS):
        raise ValueError("bare_f01_bounds_GHz must specify all four transmons")
    for label, interval in bounds.items():
        if (
            len(interval) != 2
            or not all(math.isfinite(x) for x in interval)
            or not 0 < interval[0] < interval[1]
        ):
            raise ValueError(f"Invalid bare f01 interval for {label}")
    for pair in (LABELS[:2], LABELS[2:]):
        if fixed_a is not None and pair == LABELS[:2]:
            continue
        if max(bounds[k][0] for k in pair) > min(bounds[k][1] for k in pair):
            raise ValueError(f"Empty frequency intersection for {pair}")
    scan = payload.get("frequency_GHz")
    if scan is not None:
        if len(scan) != 3 or not all(math.isfinite(x) for x in scan):
            raise ValueError("frequency_GHz must contain finite start, stop, step")
        low = max(bounds[k][0] for k in LABELS[:2])
        high = min(bounds[k][1] for k in LABELS[:2])
        if (
            scan[0] > scan[1]
            or scan[2] <= 0
            or (fixed_a is None and not low <= scan[0] <= scan[1] <= high)
        ):
            raise ValueError("A scan must lie inside both A transmon bounds")
    duration = payload.get("duration_ns")
    if duration is not None and (
        len(duration) != 2
        or not all(math.isfinite(x) for x in duration)
        or not 0 < duration[0] < duration[1]
    ):
        raise ValueError("duration_ns requires a positive increasing time window")
