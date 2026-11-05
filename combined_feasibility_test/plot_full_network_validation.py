"""Summarize resonator-inclusive full-network endpoint validation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    rows = json.loads((DATA / "full_network_validation/results.json").read_text())
    targets = json.loads((DATA / "rail_b_readout_map_official/results.json").read_text())
    xs = sorted({r["ec1_target_mhz"] for r in targets});ys = sorted({r["ec2_target_mhz"] for r in targets})
    xi={x:i for i,x in enumerate(xs)};yi={y:i for i,y in enumerate(ys)}
    metrics={name:np.full((len(ys),len(xs)),np.nan) for name in ("raw","post","mismatch","dispersive","s1")}
    category={name:0 for name in ("hardware","readout","dispersive","s1","gate","all")}
    for row in rows:
        x,y=row["ec_b_mhz"];result=row["larger_basis"]
        readout_ok=all(r["passed"] for r in result["readout"])
        dispersive_ok=all(r["dispersive_ratio"]<=.1 for r in result["dispersive"])
        s1_ok=result["updated_s1"]["maximum_ratio"]<.2
        gate_ok=result["gate"]["raw_fidelity"]>.95 and result["gate"]["postselected_fidelity"]>.999
        category["hardware"]+=result["hardware"]["success"];category["readout"]+=readout_ok
        category["dispersive"]+=dispersive_ok;category["s1"]+=s1_ok;category["gate"]+=gate_ok
        category["all"]+=readout_ok and dispersive_ok and s1_ok and gate_ok
        metrics["raw"][yi[y],xi[x]]=100*result["gate"]["raw_fidelity"]
        metrics["post"][yi[y],xi[x]]=100*result["gate"]["postselected_fidelity"]
        metrics["mismatch"][yi[y],xi[x]]=100*max(r["chi_mismatch"] for r in result["readout"])
        metrics["dispersive"][yi[y],xi[x]]=max(r["dispersive_ratio"] for r in result["dispersive"])
        metrics["s1"][yi[y],xi[x]]=result["updated_s1"]["maximum_ratio"]
    panels=[
        ("raw","Full-network raw CZ fidelity","%","viridis"),
        ("post","Full-network post-selected CZ fidelity","%","viridis"),
        ("mismatch","Worst conditional readout mismatch","%","magma"),
        ("dispersive","Worst resonator dispersive ratio","|g/Δ|","magma"),
        ("s1","Updated maximum S1 channel ratio","|g/Δ|","magma"),
    ]
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True);axes=axes.ravel()
    extent=[min(xs)-2.5,max(xs)+2.5,min(ys)-2.5,max(ys)+2.5]
    for ax,(key,title,label,cmap) in zip(axes,panels,strict=False):
        image=ax.imshow(metrics[key],origin="lower",interpolation="nearest",extent=extent,aspect="auto",cmap=cmap)
        values=metrics[key][np.isfinite(metrics[key])]
        ax.set(xlabel=r"$E_{C,B1}$ (MHz)",ylabel=r"$E_{C,B2}$ (MHz)",title=f"{title}\n{values.min():.4g}–{values.max():.4g}")
        fig.colorbar(image,ax=ax,shrink=.86,label=label)
    axes[-1].axis("off")
    text="\n".join(["Larger-basis category passes (of 16)",*(f"{name}: {value}" for name,value in category.items())])
    axes[-1].text(.05,.95,text,va="top",fontsize=13,family="monospace")
    fig.suptitle("Resonator-inclusive six-node endpoint validation",fontsize=16)
    plots=DATA/"plots";plots.mkdir(exist_ok=True)
    fig.savefig(plots/"full_network_validation.png",dpi=220);fig.savefig(plots/"full_network_validation.pdf");plt.close(fig)
    convergence={
        "maximum_raw_fidelity_abs_change":max(r["convergence"]["raw_fidelity_abs_change"] for r in rows),
        "maximum_postselected_fidelity_abs_change":max(r["convergence"]["postselected_fidelity_abs_change"] for r in rows),
        "maximum_S1_abs_change":max(r["convergence"]["S1_abs_change"] for r in rows),
    }
    summary={"category_pass_counts":category,"convergence":convergence,
             "larger_basis_raw_fidelity_range":[float(np.nanmin(metrics["raw"])/100),float(np.nanmax(metrics["raw"])/100)],
             "larger_basis_postselected_fidelity_range":[float(np.nanmin(metrics["post"])/100),float(np.nanmax(metrics["post"])/100)],
             "interpretation":"Endpoint validation at inherited idle/gate controls; failure is not a global infeasibility proof."}
    (DATA/"full_network_validation/analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
