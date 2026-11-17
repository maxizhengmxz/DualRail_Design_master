"""Compare one fixed-A campaign before and after cavity-frequency correction."""
from __future__ import annotations
import argparse,json,sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.patches import Patch

def gate_pass(directory):
    con=sqlite3.connect(directory/"gate_output/compact_anchor_results.sqlite")
    rows=con.execute("SELECT ec_b1_mhz,ec_b2_mhz,status,raw_fidelity,postselected_fidelity FROM gate_results").fetchall();con.close()
    return {(round(a),round(b)) for a,b,status,raw,post in rows if status=="calibrated" and raw>.95 and post>.999}

def reconciled_pass(directory):
    con=sqlite3.connect(directory/"fixed_hardware_reconciliation/fixed_hardware_results.sqlite")
    values={(a,b) for a,b,p in con.execute("SELECT ec_b1_mhz,ec_b2_mhz,reconciled_pass FROM reconciliation") if p};con.close();return values

def scatter(ax,points,values,cmap,norm,title):
    ordered=sorted(points);ax.scatter([p[0] for p in ordered],[p[1] for p in ordered],c=[values[p] for p in ordered],s=12,marker="s",linewidths=0,cmap=cmap,norm=norm,rasterized=True)
    ax.set(title=title,xlabel=r"$E_{C,B1}/h$ (MHz)",ylabel=r"$E_{C,B2}/h$ (MHz)");ax.set_aspect("equal")

def main():
    p=argparse.ArgumentParser();p.add_argument("--old",type=Path,required=True);p.add_argument("--new",type=Path,required=True);p.add_argument("--rail-b",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    b=json.loads(a.rail_b.read_text());points={(round(r["ec1_target_mhz"]),round(r["ec2_target_mhz"])) for r in b};readout={(round(r["ec1_target_mhz"]),round(r["ec2_target_mhz"])) for r in b if r["status"]=="validated_pass"}
    go,gn=gate_pass(a.old),gate_pass(a.new);fo,fn=reconciled_pass(a.old),reconciled_pass(a.new)
    overlay_cmap=ListedColormap(["#dedede","#8ecae6","#f4a261","#2a9d8f"]);overlay_norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
    def overlay(g):return {q:(3 if q in readout and q in g else 2 if q in g else 1 if q in readout else 0) for q in points}
    difference_cmap=ListedColormap(["#e5e5e5","#4575b4","#d73027","#1a9850"]);difference_norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
    def difference(old,new):return {q:(1 if q in old and q in new else 2 if q in old else 3 if q in new else 0) for q in points}
    final_cmap=ListedColormap(["#dedede","#2a9d8f"]);final_norm=BoundaryNorm([-.5,.5,1.5],2)
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout="constrained",sharex=True,sharey=True)
    scatter(axes[0,0],points,overlay(go),overlay_cmap,overlay_norm,f"(a) Before: gate/readout overlay\n{len(go & readout):,} overlay points")
    scatter(axes[0,1],points,overlay(gn),overlay_cmap,overlay_norm,f"(b) Corrected: gate/readout overlay\n{len(gn & readout):,} overlay points")
    scatter(axes[0,2],points,difference(go & readout,gn & readout),difference_cmap,difference_norm,f"(c) Overlay change\nshared {len((go&readout)&(gn&readout)):,}; old only {len((go&readout)-(gn&readout)):,}; new only {len((gn&readout)-(go&readout)):,}")
    scatter(axes[1,0],points,{q:int(q in fo) for q in points},final_cmap,final_norm,f"(d) Before: reconciled feasible\n{len(fo):,} points")
    scatter(axes[1,1],points,{q:int(q in fn) for q in points},final_cmap,final_norm,f"(e) Corrected: reconciled feasible\n{len(fn):,} points")
    scatter(axes[1,2],points,difference(fo,fn),difference_cmap,difference_norm,f"(f) Reconciled-region change\nshared {len(fo&fn):,}; old only {len(fo-fn):,}; new only {len(fn-fo):,}")
    axes[0,0].legend(handles=[Patch(color=overlay_cmap(i),label=x) for i,x in enumerate(("Neither","Readout only","Gate only","Gate + readout"))],fontsize=8)
    axes[0,2].legend(handles=[Patch(color=difference_cmap(i),label=x) for i,x in enumerate(("Neither","Both","Before only","Corrected only"))],fontsize=8)
    fig.suptitle(r"$E_{C,A}=(156,254)$ MHz: effect of enforcing 6.05/5.95 GHz cavity provenance",fontsize=15)
    a.output.mkdir(parents=True,exist_ok=True);png=a.output/"A156_254_before_after_cavity_correction.png";pdf=a.output/"A156_254_before_after_cavity_correction.pdf";fig.savefig(png,dpi=220);fig.savefig(pdf);plt.close(fig)
    summary={"gate_readout_overlay":{"before":len(go&readout),"corrected":len(gn&readout),"shared":len((go&readout)&(gn&readout)),"before_only":len((go&readout)-(gn&readout)),"corrected_only":len((gn&readout)-(go&readout))},"reconciled":{"before":len(fo),"corrected":len(fn),"shared":len(fo&fn),"before_only":len(fo-fn),"corrected_only":len(fn-fo)}};(a.output/"A156_254_before_after_cavity_correction.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));print(png)
if __name__=="__main__":main()
