#!/usr/bin/env python3
"""Create fixed-anchor maps and an independent rail-B readout overlay."""
from __future__ import annotations
import argparse,json,sqlite3
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
B_RESULTS=HERE/"readout/direct_ec/rail_b_step2/results.json"

def load(campaign):
    manifest=json.loads((campaign/"gate_output/manifest.json").read_text())
    if manifest["status"]!="complete":raise RuntimeError("Gate campaign is not complete")
    records=[json.loads(p.read_text()) for p in sorted((campaign/"gate_output/points").glob("*.json"))]
    if len(records)!=manifest["total"]:raise RuntimeError("Checkpoint count mismatch")
    rows=[]
    for r in records:
        result=r["result"];final=result.get("final") or {};ec=r["payload"]["ec_GHz"]
        rows.append({"task_id":r["task_id"],"ec_b1_mhz":1000*ec[2],"ec_b2_mhz":1000*ec[3],"status":result.get("status"),"workflow_status":result.get("workflow_status"),"raw":final.get("raw_fidelity"),"post":final.get("postselected_fidelity"),"selected_anchor_id":(result.get("selected_anchor") or {}).get("anchor_id"),"selected_anchor_rank":result.get("selected_anchor_rank"),"screening":result.get("anchor_screening",[]),"record":r})
    return rows

def scatter(ax,rows,key,title,**kwargs):
    values=[row.get(key,np.nan) if row.get(key) is not None else np.nan for row in rows]
    im=ax.scatter([r["ec_b1_mhz"] for r in rows],[r["ec_b2_mhz"] for r in rows],c=values,s=12,marker="s",**kwargs);ax.set(title=title,xlabel=r"$E_{C,B1}/h$ (MHz)",ylabel=r"$E_{C,B2}/h$ (MHz)");ax.set_aspect("equal");plt.colorbar(im,ax=ax)

def run(campaign):
    rows=load(campaign);output=campaign/"gate_output";plots=output/"plots";plots.mkdir(exist_ok=True)
    anchors=json.loads((campaign/"rail_A_anchors.json").read_text());ids=[a["anchor_id"] for a in anchors]
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout="constrained")
    for ax,anchor_id in zip(axes.flat,ids):
        local=[]
        for row in rows:
            item=next((x for x in row["screening"] if x["anchor_id"]==anchor_id),None);local.append(row|{"value":item.get("S1") if item else None})
        scatter(ax,local,"value",f"{anchor_id}: screening $S_1$",cmap="viridis")
    axes.flat[-1].set_axis_off();fig.savefig(plots/"anchor_screening_s1.png",dpi=220);fig.savefig(plots/"anchor_screening_s1.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout="constrained");scatter(axes[0],[r|{"value":100*r["raw"] if r["raw"] is not None else None} for r in rows],"value","Selected-anchor raw CZ fidelity (%)",cmap="viridis",vmin=95,vmax=100);scatter(axes[1],[r|{"value":100*r["post"] if r["post"] is not None else None} for r in rows],"value","Selected-anchor post-selected CZ fidelity (%)",cmap="viridis",vmin=99.9,vmax=99.99);index={name:i for i,name in enumerate(ids)};scatter(axes[2],[r|{"value":index.get(r["selected_anchor_id"],np.nan)} for r in rows],"value","Selected rail-A anchor",cmap="tab10",vmin=0,vmax=max(1,len(ids)-1));fig.savefig(plots/"selected_anchor_gate_maps.png",dpi=220);fig.savefig(plots/"selected_anchor_gate_maps.pdf");plt.close(fig)
    readout_rows=json.loads(B_RESULTS.read_text());readout={(r["ec1_target_mhz"],r["ec2_target_mhz"]):r["status"]=="validated_pass" for r in readout_rows};categories={};counts=Counter()
    for r in rows:
        ec=(round(r["ec_b1_mhz"]),round(r["ec_b2_mhz"]));gate=r["raw"] is not None and r["raw"]>.95 and r["post"]>.999;ro=readout[ec];category=3 if gate and ro else 2 if gate else 1 if ro else 0;categories[ec]=category;counts[("neither","readout_only","gate_only","overlay")[category]]+=1
    cmap=ListedColormap(["#dedede","#8ecae6","#f4a261","#2a9d8f"]);norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4);points=sorted(categories);fig,ax=plt.subplots(figsize=(7.4,6.4),layout="constrained");ax.scatter([p[0] for p in points],[p[1] for p in points],c=[categories[p] for p in points],s=12,marker="s",cmap=cmap,norm=norm);ax.set(title="Rail-B readout and selected-anchor gate overlay",xlabel=r"$E_{C,B1}/h$ (MHz)",ylabel=r"$E_{C,B2}/h$ (MHz)");ax.set_aspect("equal");ax.legend(handles=[Patch(color=cmap(i),label=x) for i,x in enumerate(("Neither","Readout only","Gate only","Overlay"))],fontsize=8);fig.savefig(plots/"readout_gate_overlay.png",dpi=220);fig.savefig(plots/"readout_gate_overlay.pdf");plt.close(fig)
    summary={"tasks":len(rows),"status_counts":dict(Counter(r["workflow_status"] or r["status"] for r in rows)),"selected_anchor_counts":dict(Counter(r["selected_anchor_id"] for r in rows if r["selected_anchor_id"])),"gate_pass":sum(r["raw"] is not None and r["raw"]>.95 and r["post"]>.999 for r in rows),"overlay_counts":dict(counts)};(output/"anchor_campaign_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return summary

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("campaign",type=Path);a=p.parse_args();print(json.dumps(run(a.campaign.resolve()),indent=2))
