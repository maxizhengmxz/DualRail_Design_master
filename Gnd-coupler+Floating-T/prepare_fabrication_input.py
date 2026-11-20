"""Package a completed nominal design for the opt-in robustness workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def package(refined_five, readout_a, readout_b, gate, output):
    refined = load(refined_five)
    result = refined.get("result", refined)
    physical = result["physical"]
    readouts = {"A": load(readout_a), "B": load(readout_b)}
    gate_record = load(gate)
    best = gate_record.get("best", gate_record)
    selection = best.get("channel_selection", physical.get("channel_selection"))
    if selection is None:
        raise ValueError("Gate record does not contain the selected CZ channel")
    seed = {key: float(best[key]) for key in ("fb_ghz", "fc_ghz", "time_ns")}
    nominal = {
        pair: list(map(float, record["coordinates"][:2]))
        for pair, record in readouts.items()
    }
    payload = {
        "schema": "coupler-fabrication-input-v1",
        "five_physical": physical,
        "readouts": readouts,
        "nominal_readout_positions": nominal,
        "selection": selection,
        "off_ghz": float(result["x"][5]),
        "gate_seed": seed,
        "provenance": {
            "refined_five": str(Path(refined_five).resolve()),
            "readout_A": str(Path(readout_a).resolve()),
            "readout_B": str(Path(readout_b).resolve()),
            "gate": str(Path(gate).resolve()),
        },
    }
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-five", required=True)
    parser.add_argument("--readout-A", required=True)
    parser.add_argument("--readout-B", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    package(args.refined_five, args.readout_A, args.readout_B, args.gate, args.output)


if __name__ == "__main__":
    main()
