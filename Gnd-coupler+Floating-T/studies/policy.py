"""Shared archive/source identity and retry policy for the opt-in study."""

from pathlib import Path
import hashlib
from floating_coupler.batch import source_signature, digest

HERE = Path(__file__).resolve().parents[1]


def campaign_source_signature():
    """Include readout recovery/model policy as well as coupler numerical source."""
    root = HERE.parent
    return digest(
        dict(
            coupler=source_signature(),
            readout=[
                (str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
                for p in sorted((root / "readout_workflow").glob("*.py"))
            ],
        )
    )


def failed_reasons(row):
    reasons = []
    if row.get("status") != "evaluated":
        reasons.append("calculation")
    for pair in ["A", "B"]:
        if not row.get("readout", {}).get(pair, {}).get("passed", False):
            reasons.append("readout_" + pair)
    if not row.get("off", {}).get("found", False):
        reasons.append("off")
    if not row.get("gate", {}).get("found", False):
        reasons.append("gate")
    elif not row["gate"].get("converged", False):
        reasons.append("gate_precision")
    return reasons


def retry_task(base, row):
    from copy import deepcopy

    local = deepcopy(base)
    if row.get("off", {}).get("found"):
        local["off_ghz"] = row["off"]["fc_ghz"]
    if row.get("gate", {}).get("found"):
        local["gate_seed"] = deepcopy(row["gate"]["best"])
        local["gate_seed"]["fb_ghz"] -= (
            sum(row.get("idle_shifts_ghz", {}).get(n, 0.0) for n in ["B1", "B2"]) / 2
        )
    return dict(
        id=row["id"],
        kind=row["kind"],
        errors=row["errors"],
        base=local,
        representative=True,
        retained_readouts={
            p: r for p, r in row.get("readout", {}).items() if r.get("passed", False)
        },
    )
