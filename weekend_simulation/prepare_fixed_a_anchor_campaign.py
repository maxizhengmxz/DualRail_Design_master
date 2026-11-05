#!/usr/bin/env python3
"""Prepare an A25-style fixed-rail-A anchor-selection campaign."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"2Q-gate_XM"),str(ROOT)]
from readout_workflow.storage import atomic_json
from xm_cz.model import fixed_pair_model


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--frequency-results",type=Path,required=True);p.add_argument("--rail-b-results",type=Path,default=HERE/"readout/direct_ec/rail_b_step2/results.json");p.add_argument("--frequencies-ghz",type=float,nargs="+",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=4);p.add_argument("--maximum-s1",type=float,default=1e99);p.add_argument("--maximum-source-score",type=float,default=1e99);p.add_argument("--fidelity-fallback-count",type=int,default=5);p.add_argument("--rail-b-bare-frequency-ghz",type=float,nargs=2,default=(4.,5.),metavar=("LOW","HIGH"));a=p.parse_args()
    if not 0<a.rail_b_bare_frequency_ghz[0]<a.rail_b_bare_frequency_ghz[1]:raise ValueError("Invalid rail-B bare-frequency interval")
    source=json.loads(a.frequency_results.read_text());anchors=[]
    for requested in a.frequencies_ghz:
        row=min((r for r in source if r["status"]=="validated_pass"),key=lambda r:abs(r["target_frequency_ghz"]-requested))
        if abs(row["target_frequency_ghz"]-requested)>1e-7:raise RuntimeError(f"No exact validated witness at {requested:g} GHz")
        q=row["point"];ec=[q["loaded_ec1_mhz"]/1000,q["loaded_ec2_mhz"]/1000]
        fixed={"EJ_GHz":[q["ej1_ghz"],q["ej2_ghz"]],"J_GHz":q["J12_ghz"],"operating_frequency_GHz":q["operating_frequency_ghz"],"logical_frequency_bounds_GHz":[.14,.16],"readout_C12_fF":q["device"]["C12_fF"],"readout_Cg_fF":q["device"]["C1r_fF"]}
        gate=fixed_pair_model(*ec,*fixed["EJ_GHz"],fixed["J_GHz"],8)["logical_splitting_GHz"]
        record={"anchor_id":f"fA_{requested:.3f}","readout_operating_frequency_GHz":q["operating_frequency_ghz"],"readout_logical_frequency_GHz":q["splitting_ghz"],"gate_logical_frequency_GHz":gate,"bare_f01_GHz":[q["bare_f1_ghz"],q["bare_f2_ghz"]],"fixed_A":fixed,"complete_readout_witness":q}
        record["anchor_hash"]=digest(record);anchors.append(record)
    if len({x["anchor_hash"] for x in anchors})!=len(anchors):raise RuntimeError("Duplicate anchors")
    ec=[anchors[0]["complete_readout_witness"]["loaded_ec1_mhz"]/1000,anchors[0]["complete_readout_witness"]["loaded_ec2_mhz"]/1000]
    lightweight=[{k:v for k,v in anchor.items() if k!="complete_readout_witness"} for anchor in anchors]
    b=json.loads(a.rail_b_results.read_text());defaults={"bare_f01_bounds_GHz":{"A1":[4.,5.],"A2":[4.,5.],"B1":list(a.rail_b_bare_frequency_ghz),"B2":list(a.rail_b_bare_frequency_ghz)},"frequency_GHz":[4.,5.,.05],"logical_splittings_GHz":[anchors[0]["gate_logical_frequency_GHz"],.08],"resonance_channel":"0020","g_GHz":.005,"point_budget_seconds":300,"rail_A_anchors":lightweight,"maximum_S1":a.maximum_s1,"maximum_source_score":a.maximum_source_score,"fidelity_fallback_count":a.fidelity_fallback_count}
    tasks=[{"ec_GHz":[*ec,r["ec1_target_mhz"]/1000,r["ec2_target_mhz"]/1000]} for r in b]
    config={"schema_version":1,"handler":"xm_cz.anchor_evaluator:evaluate_anchor_set","handler_revision":1,"workers":a.workers,"progress_interval_seconds":10,"task_timeout_seconds":900,"output_directory":"gate_output","task_defaults":defaults,"gate_domain_policy":"all ordered rail-B EC targets, independent of readout","anchor_selection_policy":"screen all anchors; rank by S1 then source score; fidelity fallback until thresholds pass","tasks":tasks}
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);atomic_json(out/"rail_A_anchors.json",anchors);atomic_json(out/"config.json",config)
    print(json.dumps({"output":str(out),"tasks":len(tasks),"rail_B_bare_frequency_GHz":a.rail_b_bare_frequency_ghz,"anchors":[{"id":x["anchor_id"],"gate_logical_MHz":1000*x["gate_logical_frequency_GHz"]} for x in anchors]},indent=2))


if __name__=="__main__":main()
