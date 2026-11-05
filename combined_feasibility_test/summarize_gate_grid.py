#!/usr/bin/env python3
"""Aggregate any configured logical-frequency grid into a joint gate map."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from readout_workflow.storage import atomic_json


def logical_pair_mhz(payload):
    return tuple(round(1000*value,9) for value in payload["logical_splittings_GHz"])


def ec_b_mhz(payload):
    return tuple(round(1000*value) for value in payload["ec_GHz"][2:])


def summarize(config,points_directory,raw_threshold=.95,post_threshold=.999):
    expected={tuple(map(float,pair)) for pair in itertools.product(
        config["logical_frequency_grid_mhz"]["rail_a"],
        config["logical_frequency_grid_mhz"]["rail_b"],
    )}
    expected_ec={ec_b_mhz(task) for task in config["tasks"]}
    grouped=defaultdict(dict)
    for path in points_directory.glob("*.json"):
        record=json.loads(path.read_text())
        grouped[ec_b_mhz(record["payload"])][logical_pair_mhz(record["payload"])]=record
    rows=[]
    for ec in sorted(expected_ec):
        records=grouped[ec];samples=[]
        for pair in sorted(expected):
            record=records.get(pair);final=None if record is None or record.get("status")!="ok" else (record.get("result",{}).get("final") or None)
            samples.append({
                "logical_frequency_mhz":pair,
                "status":"missing" if record is None else record.get("result",{}).get("status",record.get("status")),
                "raw_fidelity":None if final is None else final.get("raw_fidelity"),
                "postselected_fidelity":None if final is None else final.get("postselected_fidelity"),
            })
        calibrated=[sample for sample in samples if sample["raw_fidelity"] is not None and sample["postselected_fidelity"] is not None]
        worst_raw=min(calibrated,key=lambda sample:sample["raw_fidelity"]) if calibrated else None
        worst_post=min(calibrated,key=lambda sample:sample["postselected_fidelity"]) if calibrated else None
        complete=len(calibrated)==len(expected)
        rows.append({
            "ec_b_mhz":ec,
            "expected_logical_samples":len(expected),
            "calibrated_logical_samples":len(calibrated),
            "complete":complete,
            "minimum_raw_fidelity":None if worst_raw is None else worst_raw["raw_fidelity"],
            "minimum_raw_at_logical_frequency_mhz":None if worst_raw is None else worst_raw["logical_frequency_mhz"],
            "minimum_postselected_fidelity":None if worst_post is None else worst_post["postselected_fidelity"],
            "minimum_postselected_at_logical_frequency_mhz":None if worst_post is None else worst_post["logical_frequency_mhz"],
            "joint_gate_pass":bool(complete and worst_raw["raw_fidelity"]>raw_threshold and worst_post["postselected_fidelity"]>post_threshold),
            "samples":samples,
        })
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--points",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--raw-threshold",type=float,default=.95)
    parser.add_argument("--post-threshold",type=float,default=.999)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    rows=summarize(config,args.points,args.raw_threshold,args.post_threshold)
    grid=config["logical_frequency_grid_mhz"]
    result={
        "logical_frequency_grid_mhz":grid,
        "raw_fidelity_threshold_strict":args.raw_threshold,
        "postselected_fidelity_threshold_strict":args.post_threshold,
        "ordered_ec_points":len(rows),
        "complete_points":sum(row["complete"] for row in rows),
        "joint_gate_pass_points":sum(row["joint_gate_pass"] for row in rows),
        "results":rows,
    }
    atomic_json(args.output.resolve(),result)
    print(json.dumps({key:result[key] for key in ("ordered_ec_points","complete_points","joint_gate_pass_points")}))


if __name__=="__main__":main()
