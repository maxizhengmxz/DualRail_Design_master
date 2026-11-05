#!/usr/bin/env python3
"""Plot the two reusable frequency-specific fine C1-C2 readout maps."""

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

DATA=HERE.parent/"Ec_tuple search workflow"/"readout"
PLOTS=DATA/"plots"
CAMPAIGNS=(
    ("Rail A: 140-160 MHz",DATA/"fine_c1_c2"/"rail_a_140_160"/"results.json"),
    ("Rail B: 80-100 MHz",DATA/"fine_c1_c2"/"rail_b_80_100"/"results.json"),
)


def arrays(rows,mirror=False):
    low,high=36,128
    shape=(high-low+1,high-low+1)
    status=np.full(shape,np.nan);logical=np.full(shape,np.nan)
    for row in rows:
        c1,c2=int(round(row["c1"])),int(round(row["c2"]))
        viable=row["status"] in ("validated_pass","recovered_viable")
        value=2 if row["status"]=="recovered_viable" else 1 if viable else 0
        status[c2-low,c1-low]=value
        if viable:logical[c2-low,c1-low]=1000*row["point"]["splitting_ghz"]
        if mirror:
            status[c1-low,c2-low]=value
            if viable:logical[c1-low,c2-low]=logical[c2-low,c1-low]
    return status,logical


def main():
    PLOTS.mkdir(parents=True,exist_ok=True)
    loaded=[];summary={}
    for title,path in CAMPAIGNS:
        rows=json.loads(path.read_text());loaded.append((title,rows))
        viable=[row for row in rows if row["status"] in ("validated_pass","recovered_viable")]
        logical=[1000*row["point"]["splitting_ghz"] for row in viable]
        summary[title]={
            "calculated_triangular_points":len(rows),
            "validated_pass":sum(row["status"]=="validated_pass" for row in rows),
            "diagnostically_recovered":sum(row["status"]=="recovered_viable" for row in rows),
            "viable_total":len(viable),
            "logical_frequency_mhz_range":[min(logical),max(logical)],
            "logical_frequency_mhz_quantiles":dict(zip(("q05","q25","q50","q75","q95"),map(float,np.quantile(logical,[.05,.25,.5,.75,.95])))),
        }
    cmap=ListedColormap(["#e8edf1","#2a9d8f","#f4a261"])
    norm=BoundaryNorm([-.5,.5,1.5,2.5],cmap.N)
    fig,axes=plt.subplots(2,2,figsize=(12.8,10.2),constrained_layout=True)
    for column,(title,rows) in enumerate(loaded):
        status,logical=arrays(rows)
        ax=axes[0,column]
        ax.imshow(status,origin="lower",interpolation="nearest",extent=[35.5,128.5,35.5,128.5],cmap=cmap,norm=norm,aspect="equal")
        ax.plot([36,128],[36,128],color="#67727c",lw=.7,ls="--")
        ax.set_title(f"{title}\ncomputed triangular campaign")
        viable=summary[title]["viable_total"]
        ax.text(.03,.97,f"{viable}/4278 viable",transform=ax.transAxes,va="top",ha="left",bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
        ax.legend(handles=[Patch(color=cmap(0),label="No validated witness"),Patch(color=cmap(1),label="Validated"),Patch(color=cmap(2),label="Recovered by diagnosis")],loc="lower right",fontsize=8)
        ax=axes[1,column]
        image=ax.imshow(logical,origin="lower",interpolation="nearest",extent=[35.5,128.5,35.5,128.5],cmap="viridis",aspect="equal")
        ax.plot([36,128],[36,128],color="white",lw=.7,ls="--",alpha=.8)
        q=summary[title]["logical_frequency_mhz_quantiles"]
        ax.set_title(f"Selected-witness logical frequency\nmedian {q['q50']:.2f} MHz; 5-95% {q['q05']:.2f}-{q['q95']:.2f} MHz")
        fig.colorbar(image,ax=ax,label="Logical frequency (MHz)",shrink=.82)
    for ax in axes.flat:ax.set(xlabel=r"$C_1$ shunt (fF)",ylabel=r"$C_2$ shunt (fF)",xlim=(35.5,128.5),ylim=(35.5,128.5))
    fig.suptitle("Independent fine C1-C2 readout campaigns",fontsize=16,fontweight="bold")
    fig.savefig(PLOTS/"fine_c1_c2_independent_maps.png",dpi=220)
    fig.savefig(PLOTS/"fine_c1_c2_independent_maps.pdf")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12.8,6.2),constrained_layout=True)
    for ax,(title,rows) in zip(axes,loaded):
        status,_=arrays(rows,mirror=True)
        ax.imshow(status,origin="lower",interpolation="nearest",extent=[35.5,128.5,35.5,128.5],cmap=cmap,norm=norm,aspect="equal")
        ax.plot([36,128],[36,128],color="#67727c",lw=.7,ls="--")
        ax.set(title=f"{title}\nfull display by exact readout label swap",xlabel=r"$C_1$ shunt (fF)",ylabel=r"$C_2$ shunt (fF)")
    axes[1].legend(handles=[Patch(color=cmap(0),label="No validated witness"),Patch(color=cmap(1),label="Validated"),Patch(color=cmap(2),label="Recovered by diagnosis")],loc="lower right",fontsize=8)
    fig.suptitle("Mirrored readout display (no duplicated calculation)",fontsize=16,fontweight="bold")
    fig.savefig(PLOTS/"fine_c1_c2_mirrored_maps.png",dpi=220)
    fig.savefig(PLOTS/"fine_c1_c2_mirrored_maps.pdf")
    plt.close(fig)
    atomic_json(DATA/"summaries"/"fine_c1_c2_map_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
