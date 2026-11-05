#!/usr/bin/env python3
"""Pool reconciled candidates from nested frequency searches without displacement."""
from __future__ import annotations
import argparse,json,sqlite3
from collections import Counter,defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
B_RESULTS=HERE/"readout/direct_ec/rail_b_step2/results.json"

def load(campaign,label):
    path=campaign/"fixed_hardware_reconciliation/fixed_hardware_results.sqlite";con=sqlite3.connect(path);rows=[]
    for b1,b2,passed,details in con.execute("SELECT ec_b1_mhz,ec_b2_mhz,reconciled_pass,details_json FROM reconciliation"):
        item=json.loads(details);idle=item["rail_b"].get("point") or {};gate=item["gate"]["EJ_GHz"][2:]
        metric=sum(abs(gate[i]-idle[f"ej{i+1}_ghz"]) for i in (0,1)) if idle else float("inf")
        rows.append(((b1,b2),{"source":label,"source_campaign":str(campaign),"passed":bool(passed),"rail_b_ej_shift_l1_ghz":metric,"details":item}))
    con.close();return rows

def run(output,sources):
    pool=defaultdict(list)
    for label,campaign in sources:
        for ec,item in load(campaign,label):pool[ec].append(item)
    results=[]
    for ec,candidates in sorted(pool.items()):
        passing=[c for c in candidates if c["passed"]]
        selected=min(passing or candidates,key=lambda c:(c["rail_b_ej_shift_l1_ghz"],len(c["details"]["rail_b"].get("failures",[]))))
        results.append({"ec_b_mhz":list(ec),"candidate_count":len(candidates),"passing_candidate_count":len(passing),"fixed_hardware_pass":bool(passing),"selected_source":selected["source"],"selected_metric_rail_b_ej_shift_l1_ghz":selected["rail_b_ej_shift_l1_ghz"],"selected":selected["details"],"candidates":[{"source":c["source"],"passed":c["passed"],"rail_b_ej_shift_l1_ghz":c["rail_b_ej_shift_l1_ghz"],"rail_b_failures":c["details"]["rail_b"].get("failures",[])} for c in candidates]})
    output.mkdir(parents=True,exist_ok=True);db=output/"fixed_hardware_results.sqlite";con=sqlite3.connect(db);con.execute("DROP TABLE IF EXISTS reconciliation");con.execute("CREATE TABLE reconciliation(ec_b1_mhz INTEGER,ec_b2_mhz INTEGER,reconciled_pass INTEGER,selected_source TEXT,details_json TEXT,PRIMARY KEY(ec_b1_mhz,ec_b2_mhz))");con.executemany("INSERT INTO reconciliation VALUES(?,?,?,?,?)",[(r["ec_b_mhz"][0],r["ec_b_mhz"][1],int(r["fixed_hardware_pass"]),r["selected_source"],json.dumps(r,separators=(",",":"))) for r in results]);con.commit();con.close()
    all_points=[(r["ec1_target_mhz"],r["ec2_target_mhz"]) for r in json.loads(B_RESULTS.read_text())];values={p:0 for p in all_points}
    for r in results:values[tuple(r["ec_b_mhz"])]=2 if r["fixed_hardware_pass"] else 1
    plots=output/"plots";plots.mkdir(exist_ok=True);colors=ListedColormap(["#dedede","#f4a261","#2a9d8f"]);norm=BoundaryNorm([-.5,.5,1.5,2.5],3);fig,ax=plt.subplots(figsize=(7.4,6.4),layout="constrained");points=sorted(values);ax.scatter([p[0] for p in points],[p[1] for p in points],c=[values[p] for p in points],s=12,marker="s",cmap=colors,norm=norm);ax.set(title="Candidate-preserving fixed-hardware reconciliation",xlabel=r"$E_{C,B1}/h$ (MHz)",ylabel=r"$E_{C,B2}/h$ (MHz)");ax.set_aspect("equal");ax.legend(handles=[Patch(color=colors(i),label=x) for i,x in enumerate(("Outside candidate overlay","All candidates failed","At least one candidate passed"))],fontsize=8);fig.savefig(plots/"fixed_hardware_reconciliation.png",dpi=220);fig.savefig(plots/"fixed_hardware_reconciliation.pdf");plt.close(fig)
    feasible={point:int(values[point]==2) for point in points};binary=ListedColormap(["#dedede","#2a9d8f"]);binary_norm=BoundaryNorm([-.5,.5,1.5],2);fig,ax=plt.subplots(figsize=(7.4,6.4),layout="constrained");ax.scatter([p[0] for p in points],[p[1] for p in points],c=[feasible[p] for p in points],s=12,marker="s",cmap=binary,norm=binary_norm);ax.set(title="Final candidate-preserving feasible region",xlabel=r"$E_{C,B1}/h$ (MHz)",ylabel=r"$E_{C,B2}/h$ (MHz)");ax.set_aspect("equal");ax.legend(handles=[Patch(color="#dedede",label="Not finally feasible"),Patch(color="#2a9d8f",label="Fixed-hardware feasible")],fontsize=8);fig.savefig(plots/"final_feasibility_map.png",dpi=220);fig.savefig(plots/"final_feasibility_map.pdf");plt.close(fig)
    passing=[r for r in results if r["fixed_hardware_pass"]];summary={"candidate_sources":{label:str(path) for label,path in sources},"points_with_candidates":len(results),"reconciled_pass":len(passing),"reconciled_fail":len(results)-len(passing),"selected_source_counts":dict(Counter(r["selected_source"] for r in passing)),"points_with_multiple_candidates":sum(r["candidate_count"]>1 for r in results),"points_recovered_by_expanded_source":sum(r["fixed_hardware_pass"] and r["selected_source"]=="expanded_4_5p5" and not any(c["source"]=="baseline_4_5" and c["passed"] for c in r["candidates"]) for r in results),"selection_metric":"minimum sum(abs(EJ_B_gate-EJ_B_idle)) among reconciliation-passing candidates"};(output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");return summary

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output",type=Path);p.add_argument("--baseline",type=Path,required=True);p.add_argument("--expanded",type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.output.resolve(),[("baseline_4_5",a.baseline.resolve()),("expanded_4_5p5",a.expanded.resolve())]),indent=2))
