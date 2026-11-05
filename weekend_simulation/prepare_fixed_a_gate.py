#!/usr/bin/env python3
"""Prepare an independent ordered rail-B gate map around one fixed rail-A witness."""
from __future__ import annotations

import argparse,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"2Q-gate_XM"),str(ROOT)]
from readout_workflow.storage import atomic_json
from xm_cz.model import fixed_pair_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-results",type=Path,required=True)
    parser.add_argument("--rail-b-results",type=Path,default=HERE/"readout/direct_ec/rail_b_step2/results.json")
    parser.add_argument("--frequency-ghz",type=float,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    rows=json.loads(args.frequency_results.read_text())
    row=min((r for r in rows if r["status"]=="validated_pass"),key=lambda r:abs(r["target_frequency_ghz"]-args.frequency_ghz))
    if abs(row["target_frequency_ghz"]-args.frequency_ghz)>1e-7: raise RuntimeError("Requested frequency is not an exact validated scan point")
    point=row["point"]; ec=[point["loaded_ec1_mhz"]/1000,point["loaded_ec2_mhz"]/1000]
    fixed={"EJ_GHz":[point["ej1_ghz"],point["ej2_ghz"]],"J_GHz":point["J12_ghz"],
           "operating_frequency_GHz":point["operating_frequency_ghz"],"logical_frequency_bounds_GHz":[.14,.16],
           "readout_C12_fF":point["device"]["C12_fF"],"readout_Cg_fF":point["device"]["C1r_fF"]}
    model=fixed_pair_model(*ec,*fixed["EJ_GHz"],fixed["J_GHz"],8); logical_a=model["logical_splitting_GHz"]
    if not .14<=logical_a<=.16: raise RuntimeError("Fixed witness leaves the allowed logical-A interval in the gate model")
    b_rows=json.loads(args.rail_b_results.read_text())
    defaults={"bare_f01_bounds_GHz":{name:[4.,5.] for name in ("A1","A2","B1","B2")},
              "frequency_GHz":[4.,5.,.05],"logical_splittings_GHz":[logical_a,.08],
              "resonance_channel":"0020","g_GHz":.005,"point_budget_seconds":300,"fixed_A":fixed}
    tasks=[{"ec_GHz":[*ec,r["ec1_target_mhz"]/1000,r["ec2_target_mhz"]/1000]} for r in b_rows]
    config={"schema_version":1,"handler":"xm_cz.evaluator:evaluate_ec","handler_revision":2,
            "workers":args.workers,"progress_interval_seconds":10,"task_timeout_seconds":450,
            "output_directory":"gate_output","task_defaults":defaults,
            "logical_frequency_grid_mhz":{"rail_a":[1000*logical_a],"rail_b":[80.],"aggregation":"one fixed-A witness"},
            "gate_domain_policy":"all ordered EC targets; independent of readout status",
            "ec_orientation_policy":"ordered; no symmetry reduction","fixed_A_source":str(args.frequency_results.resolve()),"tasks":tasks}
    atomic_json(output/"fixed_A_witness.json",row);atomic_json(output/"config.json",config)
    print(json.dumps({"output":str(output),"tasks":len(tasks),"fixed_A_frequency_GHz":args.frequency_ghz,"gate_logical_A_MHz":1000*logical_a},indent=2))


if __name__=="__main__":main()
