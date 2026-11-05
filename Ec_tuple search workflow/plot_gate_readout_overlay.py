#!/usr/bin/env python3
"""Overlay one compact independent gate map with the stored rail-B readout map."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
B_RESULTS=HERE/"readout"/"direct_ec"/"rail_b_step2"/"results.json"


def create_overlay(campaign):
    output=campaign/"gate_output";database=output/"compact_results.sqlite"
    connection=sqlite3.connect(database);gate={(round(a),round(b)):(raw,post,status) for a,b,raw,post,status in connection.execute("SELECT ec_b1_mhz,ec_b2_mhz,raw_fidelity,postselected_fidelity,result_status FROM gate_results")};connection.close()
    readout_rows=json.loads(B_RESULTS.read_text());readout={(row["ec1_target_mhz"],row["ec2_target_mhz"]):row["status"]=="validated_pass" for row in readout_rows}
    if set(gate)!=set(readout):raise RuntimeError("Independent gate and readout domains differ")
    categories={};counts={"neither":0,"readout_only":0,"gate_only":0,"overlay":0}
    for ec in sorted(readout):
        raw,post,status=gate[ec];gate_pass=status=="calibrated" and raw is not None and post is not None and raw>.95 and post>.999;readout_pass=readout[ec]
        category=3 if readout_pass and gate_pass else 2 if gate_pass else 1 if readout_pass else 0
        name=("neither","readout_only","gate_only","overlay")[category];counts[name]+=1;categories[ec]=category
    plots=output/"plots";plots.mkdir(exist_ok=True);cmap=ListedColormap(["#dedede","#8ecae6","#f4a261","#2a9d8f"]);norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
    fig,ax=plt.subplots(figsize=(7.4,6.4),layout="constrained");points=sorted(categories);image=ax.scatter([p[0] for p in points],[p[1] for p in points],c=[categories[p] for p in points],s=12,marker="s",cmap=cmap,norm=norm)
    ax.set(title="Independent rail-B readout and gate overlay",xlabel=r"$E_{C,B1}/h$ (MHz)",ylabel=r"$E_{C,B2}/h$ (MHz)");ax.set_aspect("equal");ax.legend(handles=[Patch(color=cmap(i),label=label) for i,label in enumerate(("Neither passes","Readout only","Gate only","Readout + gate overlay"))],fontsize=8,loc="upper right")
    fig.savefig(plots/"readout_gate_overlay.png",dpi=220);fig.savefig(plots/"readout_gate_overlay.pdf");plt.close(fig)
    result={"gate_thresholds":{"raw_strictly_greater_than":.95,"postselected_strictly_greater_than":.999},"domain_points":len(categories),"counts":counts,"overlay_points":counts["overlay"],"readout_results":str(B_RESULTS),"compact_gate_database":str(database)}
    (output/"overlay_summary.json").write_text(json.dumps(result,indent=2)+"\n");return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("campaign",type=Path);args=parser.parse_args();print(json.dumps(create_overlay(args.campaign.resolve()),indent=2))


if __name__=="__main__":main()
