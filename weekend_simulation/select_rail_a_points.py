#!/usr/bin/env python3
"""Verify and plot the rail-A EC points selected for weekend repetitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from readout_workflow.storage import atomic_json

HERE=Path(__file__).resolve().parent
SOURCE=HERE/"readout"/"direct_ec"/"rail_a_step5"/"results.json"
OUTPUT=HERE/"selections"/"rail_a_ec_selections.json"
PLOTS=HERE/"readout"/"plots"
BASE_SELECTION=((160,240),(155,215),(155,260),(170,250),(160,225),(160,250),(170,240),(165,245),(165,235),(165,255),(155,245),(155,235))
BOUNDARY_ADDITIONS=((150,270),(150,195),(165,215),(175,235),(175,255),(165,265),(155,270),(170,225))
INTERIOR_ADDITIONS=((170,255),(170,245),(170,235),(165,260),(165,250),(165,240),(165,230),(165,225),(160,260),(160,255),(160,245),(160,235),(160,230),(160,220))
ALL_SELECTION=BASE_SELECTION+BOUNDARY_ADDITIONS+INTERIOR_ADDITIONS


def main():
    rows=json.loads(SOURCE.read_text());by={(r["ec1_target_mhz"],r["ec2_target_mhz"]):r for r in rows}
    xs=sorted({p[0] for p in by});ys=sorted({p[1] for p in by});mask=np.zeros((len(ys),len(xs)),bool)
    for (x,y),row in by.items():mask[ys.index(y),xs.index(x)]=row["status"]=="validated_pass"
    margin=distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]*5
    selected=[]
    for index,base in enumerate(ALL_SELECTION,1):
        swapped=(base[1],base[0])
        for orientation,ec in (("base",base),("swapped",swapped)):
            row=by.get(ec)
            if row is None or row["status"]!="validated_pass":raise RuntimeError(f"Selected point is not readout validated: {ec}")
            point=row["point"]
            group="original" if base in BASE_SELECTION else "boundary_addition" if base in BOUNDARY_ADDITIONS else "interior_addition"
            selected.append({"selection_id":f"A{index}{'s' if orientation=='swapped' else ''}","pair_id":f"A{index}","orientation":orientation,"selection_group":group,"ec_a_mhz":list(ec),"readout_status":row["status"],"readout_logical_frequency_mhz":1000*point["splitting_ghz"],"readout_operating_frequency_ghz":point["operating_frequency_ghz"],"idle_ej_ghz":[point["ej1_ghz"],point["ej2_ghz"]],"interior_margin_mhz":float(margin[ys.index(ec[1]),xs.index(ec[0])]),"source_checkpoint":f"ec_{ec[0]}_{ec[1]}.json"})
    manifest={"source_results":str(SOURCE),"selection_policy":"Original 12 maximin interior points plus 8 validated boundary points and 14 gap-filling interior points; each exact label-swapped ordering is a separate gate repetition.","runtime_target_hours":24,"runtime_basis":{"completed_ordered_repetitions":24,"observed_wall_hours":8.43,"additional_ordered_repetitions":44,"projected_total_wall_hours":23.89},"base_lobe_centroid_mhz":[159.6969696969697,237.87878787878788],"original_base_pairs":len(BASE_SELECTION),"boundary_additions":len(BOUNDARY_ADDITIONS),"interior_additions":len(INTERIOR_ADDITIONS),"base_pairs":len(ALL_SELECTION),"ordered_repetitions":len(selected),"selections":selected}
    atomic_json(OUTPUT,manifest)

    grid=np.full((len(ys),len(xs)),np.nan)
    for (x,y),row in by.items():grid[ys.index(y),xs.index(x)]=1 if row["status"]=="validated_pass" else 0
    cmap=ListedColormap(["#d8dee3","#2a9d8f"]);norm=BoundaryNorm([-.5,.5,1.5],2)
    fig,ax=plt.subplots(figsize=(8.4,7.2),layout="constrained")
    ax.imshow(grid,origin="lower",interpolation="nearest",extent=[min(xs)-2.5,max(xs)+2.5,min(ys)-2.5,max(ys)+2.5],aspect="equal",cmap=cmap,norm=norm)
    for index,base in enumerate(ALL_SELECTION,1):
        swapped=(base[1],base[0]);ax.plot([base[0],swapped[0]],[base[1],swapped[1]],color="#455a64",lw=.7,ls=":",alpha=.65)
        color="#1565c0" if base in BASE_SELECTION else "#212121" if base in BOUNDARY_ADDITIONS else "#ef6c00"
        marker="*" if base in BOUNDARY_ADDITIONS else "o"
        ax.scatter(*base,s=85 if marker=="*" else 60,c=color,marker=marker,edgecolors="white",linewidths=.7,zorder=3);ax.annotate(f"A{index}",base,xytext=(4,4),textcoords="offset points",fontsize=6.5,fontweight="bold")
        ax.scatter(*swapped,s=70,c=color,marker="D",edgecolors="white",linewidths=.7,zorder=3);ax.annotate(f"A{index}s",swapped,xytext=(4,4),textcoords="offset points",fontsize=6.5,fontweight="bold")
    ax.legend(handles=[Patch(color=cmap(1),label="Validated readout"),Patch(color=cmap(0),label="No validated witness found"),plt.Line2D([],[],marker="o",ls="",color="#1565c0",label="Original selection"),plt.Line2D([],[],marker="*",ls="",color="#212121",label="Added boundary point"),plt.Line2D([],[],marker="o",ls="",color="#ef6c00",label="Added interior point"),plt.Line2D([],[],marker="D",ls="",color="#777777",label="Swapped ordering")],loc="center",fontsize=8)
    ax.set(title="Selected rail-A EC repetitions on the independent readout map",xlabel=r"$E_{C,A1}/h$ (MHz)",ylabel=r"$E_{C,A2}/h$ (MHz)")
    PLOTS.mkdir(parents=True,exist_ok=True);fig.savefig(PLOTS/"rail_a_selected_ec_repetitions.png",dpi=220);fig.savefig(PLOTS/"rail_a_selected_ec_repetitions.pdf");plt.close(fig)
    print(json.dumps({"manifest":str(OUTPUT),"ordered_repetitions":len(selected),"new_ordered_repetitions":2*(len(BOUNDARY_ADDITIONS)+len(INTERIOR_ADDITIONS)),"base_pairs":[list(p) for p in ALL_SELECTION]},indent=2))


if __name__=="__main__":main()
