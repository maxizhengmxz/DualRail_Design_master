#!/usr/bin/env python3
"""Reconcile rail B while auditing the immutable selected rail-A anchor."""
from __future__ import annotations
import argparse,json,sqlite3,sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from readout_workflow.storage import atomic_json
HERE=Path(__file__).resolve().parent;B_RESULTS=HERE/"readout/direct_ec/rail_b_step2/results.json"

def one(task):
    record,bpoint,path=task;result=record["result"];anchor=result["selected_anchor"];final=result["final"]
    expected=anchor["fixed_A"]["EJ_GHz"];actual=final["EJ_GHz"][:2];ej_error=max(abs(a-b) for a,b in zip(actual,expected));j_error=abs(final["couplings_GHz"]["A1_A2"]-anchor["fixed_A"]["J_GHz"]);a_pass=ej_error<=1e-8 and j_error<=1e-10
    from combined_feasibility_test.reconcile_and_plot import reconcile_rail
    ec=[1000*x for x in record["payload"]["ec_GHz"][2:]];b=reconcile_rail(ec,final["couplings_GHz"]["B1_B2"],bpoint,(.08,.10));passed=a_pass and b["status"]=="validated_pass"
    out={"ec_b_mhz":ec,"selected_anchor":anchor,"gate":{"raw_fidelity":final["raw_fidelity"],"postselected_fidelity":final["postselected_fidelity"],"EJ_GHz":final["EJ_GHz"],"couplings_GHz":final["couplings_GHz"],"fB_GHz":final["fB_GHz"]},"rail_a_fixed_audit":{"passed":a_pass,"maximum_EJ_error_GHz":ej_error,"J_error_GHz":j_error},"rail_b":b,"fixed_hardware_pass":passed};atomic_json(Path(path),out);return out

def run(campaign,workers):
    manifest=json.loads((campaign/"gate_output/manifest.json").read_text());
    if manifest["status"]!="complete":raise RuntimeError("Gate campaign is not complete")
    b_rows=json.loads(B_RESULTS.read_text());b_valid={(r["ec1_target_mhz"],r["ec2_target_mhz"]):r for r in b_rows if r["status"]=="validated_pass"};out=campaign/"fixed_hardware_reconciliation";points=out/"points";points.mkdir(parents=True,exist_ok=True);results=[];tasks=[]
    for p in (campaign/"gate_output/points").glob("*.json"):
        record=json.loads(p.read_text());r=record["result"];f=r.get("final") or {};ec=tuple(round(1000*x) for x in record["payload"]["ec_GHz"][2:])
        if ec not in b_valid or r.get("status")!="calibrated" or f.get("raw_fidelity",0)<=.95 or f.get("postselected_fidelity",0)<=.999 or not r.get("selected_anchor"):continue
        path=points/f"ec_{ec[0]}_{ec[1]}.json"
        if path.exists():results.append(json.loads(path.read_text()))
        else:tasks.append((record,b_valid[ec]["point"],str(path)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):results.append(f.result());
    results.sort(key=lambda r:r["ec_b_mhz"]);db=out/"fixed_hardware_results.sqlite";con=sqlite3.connect(db);con.execute("DROP TABLE IF EXISTS reconciliation");con.execute("CREATE TABLE reconciliation(ec_b1_mhz INTEGER,ec_b2_mhz INTEGER,reconciled_pass INTEGER,details_json TEXT,PRIMARY KEY(ec_b1_mhz,ec_b2_mhz))");con.executemany("INSERT INTO reconciliation VALUES(?,?,?,?)",[(round(r["ec_b_mhz"][0]),round(r["ec_b_mhz"][1]),int(r["fixed_hardware_pass"]),json.dumps(r,separators=(",",":"))) for r in results]);con.commit();con.close();summary={"overlay_points":len(results),"reconciled_pass":sum(r["fixed_hardware_pass"] for r in results),"reconciled_fail":sum(not r["fixed_hardware_pass"] for r in results),"rail_b_failure_counts":dict(Counter(x for r in results for x in r["rail_b"]["failures"])),"rail_a_audit_failures":sum(not r["rail_a_fixed_audit"]["passed"] for r in results)};atomic_json(out/"summary.json",summary);return summary

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("campaign",type=Path);p.add_argument("--workers",type=int,default=4);a=p.parse_args();print(json.dumps(run(a.campaign.resolve(),a.workers),indent=2))
