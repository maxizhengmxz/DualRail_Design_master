#!/usr/bin/env python3
"""Compare two completed fixed-anchor campaigns with different B bounds."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.patches import Patch

def load(campaign):
    rows={}
    for p in (campaign/"gate_output/points").glob("*.json"):
        record=json.loads(p.read_text());ec=tuple(round(1000*x) for x in record["payload"]["ec_GHz"][2:]);r=record["result"];f=r.get("final") or {};rows[ec]={"pass":r.get("status")=="calibrated" and f.get("raw_fidelity",0)>.95 and f.get("postselected_fidelity",0)>.999,"status":r.get("workflow_status") or r.get("status"),"fB":f.get("fB_GHz"),"anchor":(r.get("selected_anchor") or {}).get("anchor_id")}
    return rows

def run(old,new):
    a,b=load(old),load(new)
    if set(a)!=set(b):raise RuntimeError("Campaign domains differ")
    categories={};counts=Counter();recovered_by_old_status=Counter()
    for ec in a:
        category=3 if a[ec]["pass"] and b[ec]["pass"] else 2 if not a[ec]["pass"] and b[ec]["pass"] else 1 if a[ec]["pass"] and not b[ec]["pass"] else 0
        name=("fail_both","lost_with_expansion","recovered_with_expansion","pass_both")[category];categories[ec]=category;counts[name]+=1
        if category==2:recovered_by_old_status[a[ec]["status"]]+=1
    output=new/"frequency_bound_comparison";output.mkdir(exist_ok=True);colors=["#dedede","#d73027","#1a9850","#4575b4"];labels=["Fails in both ranges","Passes at 4–5 GHz only","Recovered by 4–5.5 GHz","Passes in both ranges"];cmap=ListedColormap(colors);norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4);points=sorted(categories);fig,ax=plt.subplots(figsize=(7.5,6.5),layout="constrained");ax.scatter([p[0] for p in points],[p[1] for p in points],c=[categories[p] for p in points],s=12,marker="s",cmap=cmap,norm=norm);ax.set(title="Effect of expanding rail-B bare-frequency bound",xlabel=r"$E_{C,B1}/h$ (MHz)",ylabel=r"$E_{C,B2}/h$ (MHz)");ax.set_aspect("equal");ax.legend(handles=[Patch(color=color,label=label) for color,label in zip(colors,labels)],fontsize=8,loc="best");fig.savefig(output/"gate_feasibility_comparison.png",dpi=220);fig.savefig(output/"gate_feasibility_comparison.pdf");plt.close(fig)
    oldfb=[r["fB"] for r in a.values() if r["fB"] is not None];newfb=[r["fB"] for r in b.values() if r["fB"] is not None];summary={"old":str(old),"new":str(new),"counts":dict(counts),"recovered_by_old_status":dict(recovered_by_old_status),"old_gate_pass":sum(r["pass"] for r in a.values()),"new_gate_pass":sum(r["pass"] for r in b.values()),"old_fB_range_GHz":[min(oldfb),max(oldfb)],"new_fB_range_GHz":[min(newfb),max(newfb)],"new_points_above_5GHz":sum(x>5 for x in newfb),"selected_anchor_counts_old":dict(Counter(r["anchor"] for r in a.values() if r["pass"])),"selected_anchor_counts_new":dict(Counter(r["anchor"] for r in b.values() if r["pass"]))};(output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");return summary

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("old",type=Path);p.add_argument("new",type=Path);a=p.parse_args();print(json.dumps(run(a.old.resolve(),a.new.resolve()),indent=2))
