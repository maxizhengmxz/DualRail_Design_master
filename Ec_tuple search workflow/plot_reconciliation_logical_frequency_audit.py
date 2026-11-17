"""Plot rail-B readout logical frequency and J mismatch on one corrected overlay."""
from __future__ import annotations
import argparse,json,sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.patches import Patch

def main():
    p=argparse.ArgumentParser();p.add_argument("campaign",type=Path);p.add_argument("--rail-b",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    brows=json.loads(a.rail_b.read_text());all_points={(round(r["ec1_target_mhz"]),round(r["ec2_target_mhz"])):r for r in brows};readout={q:r for q,r in all_points.items() if r["status"]=="validated_pass"}
    con=sqlite3.connect(a.campaign/"fixed_hardware_reconciliation/fixed_hardware_results.sqlite");recon={}
    for passed,text in con.execute("select reconciled_pass,details_json from reconciliation"):
        item=json.loads(text);s=item.get("selected",item);q=tuple(round(v) for v in s["ec_b_mhz"]);recon[q]=(bool(passed),s)
    con.close();overlay=set(recon);final={q for q,(passed,_) in recon.items() if passed};points=set(all_points)
    categories={q:(2 if q in final else 1 if q in overlay else 0) for q in points};cmap=ListedColormap(["#dedede","#f4a261","#2a9d8f"]);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    fig,axes=plt.subplots(1,3,figsize=(17,5.5),layout="constrained",sharex=True,sharey=True)
    ordered=sorted(points);axes[0].scatter([q[0] for q in ordered],[q[1] for q in ordered],c=[categories[q] for q in ordered],s=12,marker="s",linewidths=0,cmap=cmap,norm=norm,rasterized=True);axes[0].set_title(f"Gate/readout overlay and reconciliation\n{len(overlay):,} overlay; {len(final):,} reconciled")
    q=sorted(overlay);logical=[1000*readout[x]["point"]["splitting_ghz"] for x in q];im=axes[1].scatter([x[0] for x in q],[x[1] for x in q],c=logical,s=15,marker="s",linewidths=0,cmap="viridis",vmin=80,vmax=85,rasterized=True);axes[1].scatter([x[0] for x in final],[x[1] for x in final],s=22,facecolors="none",edgecolors="black",linewidths=.35);fig.colorbar(im,ax=axes[1],label="Independent readout logical frequency (MHz)");axes[1].set_title("Rail-B logical frequency on overlay\nblack outline = reconciled pass")
    delta=[1000*(recon[x][1]["gate"]["couplings_GHz"]["B1_B2"]-readout[x]["point"]["J12_ghz"]) for x in q];lim=max(abs(min(delta)),abs(max(delta)));im=axes[2].scatter([x[0] for x in q],[x[1] for x in q],c=delta,s=15,marker="s",linewidths=0,cmap="coolwarm",vmin=-lim,vmax=lim,rasterized=True);axes[2].scatter([x[0] for x in final],[x[1] for x in final],s=22,facecolors="none",edgecolors="black",linewidths=.35);fig.colorbar(im,ax=axes[2],label=r"$J_B^{gate}-J_B^{readout}$ (MHz)");axes[2].set_title("Coupling change demanded by gate\nblack outline = reconciled pass")
    axes[0].legend(handles=[Patch(color=cmap(i),label=x) for i,x in enumerate(("Outside overlay","Overlay, reconciliation failed","Reconciled pass"))],fontsize=8)
    for ax in axes:ax.set_aspect("equal");ax.set_xlabel(r"$E_{C,B1}/h$ (MHz)");ax.set_ylabel(r"$E_{C,B2}/h$ (MHz)")
    fig.suptitle(r"Corrected $E_{C,A}=(156,254)$ MHz reconciliation audit",fontsize=15);a.output.mkdir(parents=True,exist_ok=True);png=a.output/"rail_b_overlay_logical_frequency_and_coupling_audit.png";pdf=a.output/"rail_b_overlay_logical_frequency_and_coupling_audit.pdf";fig.savefig(png,dpi=220);fig.savefig(pdf);plt.close(fig);print(png)
if __name__=="__main__":main()
