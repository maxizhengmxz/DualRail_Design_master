"""Create an ordered-EC gate campaign with a configurable logical grid."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from readout_workflow.storage import atomic_json

DATA = HERE / "data"
WEEKEND_READOUT = HERE.parent / "Ec_tuple search workflow" / "readout"
A_EC_MHZ = (165, 240)

def frequency_axis(values, bounds, name):
    axis=tuple(float(value) for value in values)
    if not axis or len(axis)!=len(set(axis)) or any(not bounds[0]<=value<=bounds[1] for value in axis):
        raise ValueError(f"{name} values must be unique and inside {bounds[0]}-{bounds[1]} MHz")
    return axis

def build_tasks(a_ec_mhz,b_rows,logical_a_mhz,logical_b_mhz):
    pairs=list(itertools.product(logical_a_mhz,logical_b_mhz))
    tasks=[{
        "ec_GHz":[a_ec_mhz[0]/1000,a_ec_mhz[1]/1000,row["ec1_target_mhz"]/1000,row["ec2_target_mhz"]/1000],
        "logical_splittings_GHz":[logical_a/1000,logical_b/1000],
    } for logical_a,logical_b in pairs for row in b_rows]
    identities=[(*task["ec_GHz"],*task["logical_splittings_GHz"]) for task in tasks]
    if len(identities)!=len(set(identities)):raise RuntimeError("Duplicate ordered EC/logical-frequency gate tasks")
    return tasks

def select_b_rows(rows,count=None,readout_feasible_only=False):
    """Select the independent gate domain; filtering is diagnostic-only."""
    domain=[row for row in rows if row["status"]=="validated_pass"] if readout_feasible_only else list(rows)
    if count is None or count>=len(domain):return domain
    if count<1:raise ValueError("benchmark-b-points must be positive")
    coordinates=[(row["ec1_target_mhz"],row["ec2_target_mhz"]) for row in domain]
    center=tuple(sum(p[i] for p in coordinates)/len(coordinates) for i in (0,1))
    chosen=[min(range(len(domain)),key=lambda i:sum((coordinates[i][j]-center[j])**2 for j in (0,1)))]
    while len(chosen)<count:
        remaining=(i for i in range(len(domain)) if i not in chosen)
        chosen.append(max(remaining,key=lambda i:min(math.dist(coordinates[i],coordinates[j]) for j in chosen)))
    return [domain[i] for i in sorted(chosen)]

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rail-a-results",type=Path,default=WEEKEND_READOUT/"direct_ec/rail_a_step5/results.json")
    parser.add_argument("--rail-b-results",type=Path,default=WEEKEND_READOUT/"direct_ec/rail_b_step2/results.json")
    parser.add_argument("--a-ec-mhz",type=float,nargs=2,default=A_EC_MHZ,metavar=("EC_A1","EC_A2"))
    parser.add_argument("--logical-a-mhz",type=float,nargs="+",default=[150.0])
    parser.add_argument("--logical-b-mhz",type=float,nargs="+",default=[90.0])
    parser.add_argument("--config-output",type=Path,default=DATA/"gate_logical_grid.json")
    parser.add_argument("--gate-output-directory",default="gate_logical_grid_output")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--benchmark-b-points",type=int,help="Use a deterministic spatial subset of the selected independent gate domain.")
    parser.add_argument("--readout-feasible-only",action="store_true",help="Diagnostic shortcut only; production gate maps must omit this flag and use the full independent EC domain.")
    args=parser.parse_args()
    logical_a=frequency_axis(args.logical_a_mhz,(140,160),"rail-A logical frequency")
    logical_b=frequency_axis(args.logical_b_mhz,(80,100),"rail-B logical frequency")
    a_ec=tuple(args.a_ec_mhz)
    a_rows = json.loads(args.rail_a_results.read_text())
    b_rows = select_b_rows(json.loads(args.rail_b_results.read_text()),args.benchmark_b_points,args.readout_feasible_only)
    a_row = next(
        row
        for row in a_rows
        if (row["ec1_target_mhz"], row["ec2_target_mhz"]) == a_ec
    )
    if a_row["status"] != "validated_pass":
        raise RuntimeError("Selected rail-A point is not readout validated")
    atomic_json(DATA / "selected_rail_a_readout_witness.json", a_row)
    defaults = {
        "bare_f01_bounds_GHz": {label: [4.0, 5.0] for label in ("A1", "A2", "B1", "B2")},
        "frequency_GHz": [4.0, 5.0, 0.050],
        "resonance_channel": "0020",
        "g_GHz": 0.005,
        "point_budget_seconds": 180,
    }
    tasks=build_tasks(a_ec,b_rows,logical_a,logical_b)
    config = {
        "schema_version": 1,
        "handler": "xm_cz.evaluator:evaluate_ec",
        "handler_revision": 1,
        "workers": args.workers,
        "progress_interval_seconds": 10,
        "task_timeout_seconds": 450,
        "output_directory": args.gate_output_directory,
        "task_defaults": defaults,
        "logical_frequency_grid_mhz":{"rail_a":list(logical_a),"rail_b":list(logical_b),"aggregation":"pointwise intersection; minimum raw and post-selected fidelity"},
        "gate_domain_policy":"readout-feasible diagnostic subset" if args.readout_feasible_only else "all ordered EC targets; independent of readout status",
        "ec_orientation_policy": "ordered; evaluate every (B1,B2) pair explicitly; no readout-symmetry mirroring",
        "tasks": tasks,
    }
    atomic_json(args.config_output.resolve(), config)
    print(json.dumps({"selected_a_ec_mhz":a_ec,"logical_a_mhz":logical_a,"logical_b_mhz":logical_b,"independent_gate_ec_targets":len(b_rows),"gate_tasks":len(tasks),"benchmark_subset":args.benchmark_b_points is not None,"readout_prefiltered":args.readout_feasible_only,"config":str(args.config_output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
