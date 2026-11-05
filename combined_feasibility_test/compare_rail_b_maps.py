"""Compare provisional and official direct-EC rail-B campaigns."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from readout_workflow.storage import atomic_json

DATA = HERE / "data"


def keyed(path):
    rows = json.loads(path.read_text())
    return {(r["ec1_target_mhz"], r["ec2_target_mhz"]): r for r in rows}


def main():
    provisional = keyed(DATA / "rail_b_readout_map/results.json")
    official = keyed(DATA / "rail_b_readout_map_official/results.json")
    fields = (
        "splitting_ghz",
        "operating_frequency_ghz",
        "J12_ghz",
        "ej1_ghz",
        "ej2_ghz",
    )
    common = [ec for ec in official if official[ec]["point"] and provisional[ec]["point"]]
    maxima = {}
    for field in fields:
        differences = [
            (abs(official[ec]["point"][field] - provisional[ec]["point"][field]), ec)
            for ec in common
        ]
        value, ec = max(differences)
        maxima[field] = {"absolute_difference": value, "ec_mhz": ec}
    official_only = [
        ec
        for ec in official
        if official[ec]["status"] == "validated_pass"
        and provisional[ec]["status"] != "validated_pass"
    ]
    summary = {
        "targets": len(official),
        "provisional_validated": sum(r["status"] == "validated_pass" for r in provisional.values()),
        "official_validated": sum(r["status"] == "validated_pass" for r in official.values()),
        "status_disagreements": official_only,
        "common_validated_targets": len(common),
        "maximum_witness_differences": maxima,
        "interpretation": "Different validated witnesses are allowed; the campaign is first-feasible rather than a unique-solution or global optimization search.",
    }
    atomic_json(DATA / "rail_b_map_comparison.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
