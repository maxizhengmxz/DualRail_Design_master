#!/usr/bin/env python3
"""Plot tuning metrics on the reconciliation-passing fixed-anchor region."""
from __future__ import annotations
import argparse,json,sqlite3,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"2Q-gate_XM"));sys.path.insert(0,str(ROOT))
from xm_cz.model import fixed_pair_model,local_model_from_ej

HERE=Path(__file__).resolve().parent;B_RESULTS=HERE/"readout/direct_ec/rail_b_step2/results.json"

def panel_pair(rows,all_points,keys,titles,output,name,label,limits=None):
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout="constrained",sharex=True,sharey=True)
    for ax,key,title in zip(axes,keys,titles):
        ax.scatter([p[0] for p in all_points],[p[1] for p in all_points],c="#e4e4e4",s=12,marker="s")
        kwargs={"cmap":"coolwarm" if (limits and limits[0]<0) else "viridis"}
        if limits:kwargs|={"vmin":limits[0],"vmax":limits[1]}
        image=ax.scatter([r["ec_b1_mhz"] for r in rows],[r["ec_b2_mhz"] for r in rows],c=[r[key] for r in rows],s=13,marker="s",**kwargs)
        ax.set(title=title,xlabel=r"$E_{C,B1}/h$ (MHz)",ylabel=r"$E_{C,B2}/h$ (MHz)");ax.set_aspect("equal")
    fig.colorbar(image,ax=axes,label=label,shrink=.88);fig.savefig(output/f"{name}.png",dpi=220);fig.savefig(output/f"{name}.pdf");plt.close(fig)

def run(campaign,reconciliation_directory="fixed_hardware_reconciliation"):
    db=campaign/reconciliation_directory/"fixed_hardware_results.sqlite";con=sqlite3.connect(db);stored=[json.loads(x[0]) for x in con.execute("SELECT details_json FROM reconciliation WHERE reconciled_pass=1")];con.close();items=[item["selected"] if "selected" in item else item for item in stored]
    rows=[]
    for item in items:
        ec=[x/1000 for x in item["ec_b_mhz"]];idle=item["rail_b"]["point"];gate_ej=item["gate"]["EJ_GHz"][2:]
        gate_bare=[float(local_model_from_ej(ec[i],gate_ej[i])[1][1]) for i in (0,1)]
        gate_pair=fixed_pair_model(ec[0],ec[1],gate_ej[0],gate_ej[1],item["gate"]["couplings_GHz"]["B1_B2"],8)
        gate_operating=float(np.mean([gate_pair["energies"][mode] for mode in gate_pair["modes"]]))
        anchor=item["selected_anchor"]
        rows.append({"ec_b1_mhz":item["ec_b_mhz"][0],"ec_b2_mhz":item["ec_b_mhz"][1],
                     "delta_ej_b1_ghz":gate_ej[0]-idle["ej1_ghz"],"delta_ej_b2_ghz":gate_ej[1]-idle["ej2_ghz"],
                     "delta_bare_b1_mhz":1000*(gate_bare[0]-idle["bare_f1_ghz"]),"delta_bare_b2_mhz":1000*(gate_bare[1]-idle["bare_f2_ghz"]),
                     "idle_operating_b_ghz":idle["operating_frequency_ghz"],"gate_operating_b_ghz":gate_operating,
                     "readout_logical_a_mhz":1000*anchor["readout_logical_frequency_GHz"],"gate_logical_a_mhz":1000*anchor["gate_logical_frequency_GHz"],"anchor_id":anchor["anchor_id"]})
    all_points=[(r["ec1_target_mhz"],r["ec2_target_mhz"]) for r in json.loads(B_RESULTS.read_text())];output=campaign/reconciliation_directory/"plots";output.mkdir(exist_ok=True)
    ejmax=max(abs(r[k]) for r in rows for k in ("delta_ej_b1_ghz","delta_ej_b2_ghz"));fmax=max(abs(r[k]) for r in rows for k in ("delta_bare_b1_mhz","delta_bare_b2_mhz"))
    panel_pair(rows,all_points,("delta_ej_b1_ghz","delta_ej_b2_ghz"),(r"Rail B1: $E_J$ shift",r"Rail B2: $E_J$ shift"),output,"rail_b_ej_shifts","Gate minus idle $E_J/h$ (GHz)",(-ejmax,ejmax))
    panel_pair(rows,all_points,("delta_bare_b1_mhz","delta_bare_b2_mhz"),(r"Rail B1: bare $f_{01}$ shift",r"Rail B2: bare $f_{01}$ shift"),output,"rail_b_bare_frequency_shifts","Gate minus idle bare $f_{01}$ (MHz)",(-fmax,fmax))
    panel_pair(rows,all_points,("readout_logical_a_mhz","gate_logical_a_mhz"),("Selected rail-A readout logical frequency","Selected rail-A gate-model logical frequency"),output,"rail_a_logical_frequencies","Logical frequency (MHz)",(140,160))
    panel_pair(rows,all_points,("idle_operating_b_ghz","gate_operating_b_ghz"),("Rail B: idle readout operating frequency","Rail B: gate operating frequency"),output,"rail_b_operating_frequencies","Operating frequency (GHz)",(4,6))
    summary={"points":len(rows),"shared_ej_shift_scale_ghz":[-ejmax,ejmax],"shared_bare_frequency_shift_scale_mhz":[-fmax,fmax],"logical_frequency_scale_mhz":[140,160],"data":rows,"plots":[p.name for p in sorted(output.iterdir())]};(campaign/reconciliation_directory/"metric_plot_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return {k:v for k,v in summary.items() if k!="data"}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("campaign",type=Path);p.add_argument("--reconciliation-directory",default="fixed_hardware_reconciliation");a=p.parse_args();print(json.dumps(run(a.campaign.resolve(),a.reconciliation_directory),indent=2))
