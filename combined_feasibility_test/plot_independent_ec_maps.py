#!/usr/bin/env python3
"""Plot completed full-orientation direct-EC readout maps for both rails."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from readout_workflow.storage import atomic_json

DATA=HERE.parent/"weekend_simulation"/"readout";PLOTS=DATA/"plots"
CAMPAIGNS=(
    ("rail_a","Rail A readout: 140-160 MHz",DATA/"direct_ec"/"rail_a_step5",5),
    ("rail_b","Rail B readout: 80-100 MHz",DATA/"direct_ec"/"rail_b_step2",2),
)


def main():
    PLOTS.mkdir(parents=True,exist_ok=True);summary={}
    for key,title,directory,step in CAMPAIGNS:
        progress=json.loads((directory/"progress.json").read_text())
        if progress["phase"]!="complete":raise RuntimeError(f"{key} campaign is not complete")
        rows=json.loads((directory/"results.json").read_text())
        xs=sorted({row["ec1_target_mhz"] for row in rows});ys=sorted({row["ec2_target_mhz"] for row in rows})
        xi={value:index for index,value in enumerate(xs)};yi={value:index for index,value in enumerate(ys)}
        status=np.full((len(ys),len(xs)),np.nan);logical=np.full_like(status,np.nan)
        for row in rows:
            i,j=yi[row["ec2_target_mhz"]],xi[row["ec1_target_mhz"]]
            passed=row["status"]=="validated_pass";status[i,j]=1 if passed else 0
            if passed:logical[i,j]=1000*row["point"]["splitting_ghz"]
        passed=[row for row in rows if row["status"]=="validated_pass"]
        frequencies=[1000*row["point"]["splitting_ghz"] for row in passed]
        summary[key]={"logical_frequency_mhz":json.loads((directory/"settings.json").read_text())["logical_frequency_mhz"],"ec_step_mhz":step,"targets":len(rows),"validated":len(passed),"validation_fraction":len(passed)/len(rows),"logical_frequency_found_range_mhz":[min(frequencies),max(frequencies)],"ec1_range_mhz":[min(xs),max(xs)],"ec2_range_mhz":[min(ys),max(ys)]}
        extent=[min(xs)-step/2,max(xs)+step/2,min(ys)-step/2,max(ys)+step/2]
        fig,axes=plt.subplots(1,2,figsize=(13.2,6.1),constrained_layout=True)
        cmap=ListedColormap(["#d8dee3","#2a9d8f"]);norm=BoundaryNorm([-.5,.5,1.5],2)
        axes[0].imshow(status,origin="lower",interpolation="nearest",extent=extent,aspect="equal",cmap=cmap,norm=norm)
        axes[0].legend(handles=[Patch(color=cmap(0),label="No validated witness found"),Patch(color=cmap(1),label="Validated readout")],loc="lower right",fontsize=8)
        axes[0].set_title(f"Feasibility ({len(passed)}/{len(rows)} validated)")
        image=axes[1].imshow(logical,origin="lower",interpolation="nearest",extent=extent,aspect="equal",cmap="viridis")
        axes[1].set_title(f"Selected-witness logical frequency\n{min(frequencies):.2f}-{max(frequencies):.2f} MHz")
        fig.colorbar(image,ax=axes[1],label="Logical frequency (MHz)",shrink=.82)
        for ax in axes:ax.set(xlabel=r"$E_{C1}/h$ (MHz)",ylabel=r"$E_{C2}/h$ (MHz)")
        fig.suptitle(f"{title} - {step} MHz EC lattice, 1.5 scaled two-orientation hull",fontsize=15,fontweight="bold")
        fig.savefig(PLOTS/f"{key}_independent_ec_readout_map.png",dpi=220)
        fig.savefig(PLOTS/f"{key}_independent_ec_readout_map.pdf")
        plt.close(fig)
    atomic_json(DATA/"summaries"/"independent_ec_map_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
