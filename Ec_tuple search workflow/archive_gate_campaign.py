#!/usr/bin/env python3
"""Plot, compact, verify, and optionally prune a completed gate campaign."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter,defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compact_result(record):
    result=record.get("result",{});final=result.get("final") or {}
    return {
        "executor_status":record.get("status"),"result_status":result.get("status"),
        "duration_seconds":record.get("duration_seconds"),"calculation_seconds":result.get("calculation_seconds"),
        "frequency_boundary":result.get("frequency_boundary"),"calibration_time_boundary":result.get("calibration_time_boundary"),
        "calibration_detuning_boundary":result.get("calibration_detuning_boundary"),"fA_GHz":result.get("fA_GHz"),
        "fB_GHz":final.get("fB_GHz"),"time_ns":final.get("time_ns"),"B_frequency_offset_MHz":final.get("B_frequency_offset_MHz"),
        "raw_fidelity":final.get("raw_fidelity"),"postselected_fidelity":final.get("postselected_fidelity"),"acceptance":final.get("acceptance"),
        "leakage_by_input":final.get("leakage_by_input"),"return_by_input":final.get("return_by_input"),"phase_error_rad":final.get("phase_error_rad"),
        "EJ_GHz":final.get("EJ_GHz"),"couplings_GHz":final.get("couplings_GHz"),"auxiliary_bare_weight":final.get("auxiliary_bare_weight"),
        "S1":final.get("S1"),"S2":final.get("S2"),"S1_by_input":final.get("S1_by_input"),
        "maximum_logical_ratio":final.get("maximum_logical_ratio"),"maximum_auxiliary_ratio":final.get("maximum_auxiliary_ratio"),
        "failure_counts":result.get("failures"),"error_type":record.get("error_type"),"error":record.get("error"),
    }


def scatter(ax,rows,key,title,cmap="viridis",vmin=None,vmax=None,scale=1):
    values=[row[key]*scale if row.get(key) is not None else np.nan for row in rows]
    image=ax.scatter([row["ec_b1_mhz"] for row in rows],[row["ec_b2_mhz"] for row in rows],c=values,s=28,marker="s",cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set(title=title,xlabel=r"$E_{C,B1}/h$ (MHz)",ylabel=r"$E_{C,B2}/h$ (MHz)");ax.set_aspect("equal");plt.colorbar(image,ax=ax)


def archive(campaign,prune=False):
    config=json.loads((campaign/"config.json").read_text());output=campaign/config["output_directory"]
    manifest=json.loads((output/"manifest.json").read_text())
    if manifest.get("status")!="complete":raise RuntimeError("Campaign must be complete before archival")
    paths=sorted((output/"points").glob("*.json"))
    if len(paths)!=manifest["total"]:raise RuntimeError("Checkpoint count does not match manifest")
    records=[json.loads(path.read_text()) for path in paths]
    compact=[]
    for record in records:
        payload=record["payload"];metrics=compact_result(record);ec=[round(1000*v,9) for v in payload["ec_GHz"]];logical=[round(1000*v,9) for v in payload["logical_splittings_GHz"]]
        compact.append({"task_id":record["task_id"],"ec_a1_mhz":ec[0],"ec_a2_mhz":ec[1],"ec_b1_mhz":ec[2],"ec_b2_mhz":ec[3],"logical_a_mhz":logical[0],"logical_b_mhz":logical[1],**metrics})
    database=output/"compact_results.sqlite";connection=sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL");connection.execute("DROP TABLE IF EXISTS gate_results")
    connection.execute("CREATE TABLE gate_results(task_id TEXT PRIMARY KEY,ec_a1_mhz REAL,ec_a2_mhz REAL,ec_b1_mhz REAL,ec_b2_mhz REAL,logical_a_mhz REAL,logical_b_mhz REAL,executor_status TEXT,result_status TEXT,duration_seconds REAL,raw_fidelity REAL,postselected_fidelity REAL,details_json TEXT)")
    connection.executemany("INSERT INTO gate_results VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",[(r["task_id"],r["ec_a1_mhz"],r["ec_a2_mhz"],r["ec_b1_mhz"],r["ec_b2_mhz"],r["logical_a_mhz"],r["logical_b_mhz"],r["executor_status"],r["result_status"],r["duration_seconds"],r["raw_fidelity"],r["postselected_fidelity"],json.dumps({k:v for k,v in r.items() if k not in {"task_id","ec_a1_mhz","ec_a2_mhz","ec_b1_mhz","ec_b2_mhz","logical_a_mhz","logical_b_mhz","executor_status","result_status","duration_seconds","raw_fidelity","postselected_fidelity"}},separators=(",",":"))) for r in compact])
    connection.commit();connection.execute("PRAGMA wal_checkpoint(TRUNCATE)");count=connection.execute("SELECT count(*) FROM gate_results").fetchone()[0];connection.close()
    if count!=len(records):raise RuntimeError("Compact database verification failed")
    from plot_gate_readout_overlay import create_overlay
    create_overlay(campaign)

    grouped=defaultdict(list)
    for row in compact:grouped[(row["ec_b1_mhz"],row["ec_b2_mhz"])].append(row)
    summaries=[]
    expected=len(config["logical_frequency_grid_mhz"]["rail_a"])*len(config["logical_frequency_grid_mhz"]["rail_b"])
    for (b1,b2),items in sorted(grouped.items()):
        calibrated=[r for r in items if r["raw_fidelity"] is not None and r["postselected_fidelity"] is not None]
        complete=len(calibrated)==expected
        summaries.append({"ec_b1_mhz":b1,"ec_b2_mhz":b2,"calibrated_fraction":len(calibrated)/expected,"minimum_raw":min(r["raw_fidelity"] for r in calibrated) if complete else None,"minimum_post":min(r["postselected_fidelity"] for r in calibrated) if complete else None,"raw_range":np.ptp([r["raw_fidelity"] for r in calibrated]) if complete else None,"post_range":np.ptp([r["postselected_fidelity"] for r in calibrated]) if complete else None,"joint_pass":bool(complete and min(r["raw_fidelity"] for r in calibrated)>.95 and min(r["postselected_fidelity"] for r in calibrated)>.999)})
    plots=output/"plots";plots.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout="constrained");scatter(axes[0],summaries,"minimum_raw","Minimum raw fidelity (%)",vmin=95,vmax=100,scale=100);scatter(axes[1],summaries,"minimum_post","Minimum postselected fidelity (%)",vmin=99.9,vmax=100,scale=100);scatter(axes[2],summaries,"calibrated_fraction","Calibrated logical-grid fraction",vmin=0,vmax=1);fig.savefig(plots/"joint_gate_maps.png",dpi=200);fig.savefig(plots/"joint_gate_maps.pdf");plt.close(fig)
    complete=[r for r in summaries if r["raw_range"] is not None]
    if complete:
        fig,axes=plt.subplots(1,2,figsize=(11,5),layout="constrained");scatter(axes[0],complete,"raw_range","Raw-fidelity range (percentage points)",cmap="magma",scale=100);scatter(axes[1],complete,"post_range","Postselected-fidelity range (percentage points)",cmap="magma",scale=100);fig.savefig(plots/"logical_grid_sensitivity.png",dpi=200);fig.savefig(plots/"logical_grid_sensitivity.pdf");plt.close(fig)
    logical_pairs=sorted({(r["logical_a_mhz"],r["logical_b_mhz"]) for r in compact});a_values=sorted({p[0] for p in logical_pairs});b_values=sorted({p[1] for p in logical_pairs});calibrated=np.zeros((len(b_values),len(a_values)));passing=np.zeros_like(calibrated);runtime=np.zeros_like(calibrated)
    for ia,a in enumerate(a_values):
        for ib,b in enumerate(b_values):
            subset=[r for r in compact if r["logical_a_mhz"]==a and r["logical_b_mhz"]==b];calibrated[ib,ia]=sum(r["raw_fidelity"] is not None for r in subset);passing[ib,ia]=sum(r["raw_fidelity"] is not None and r["raw_fidelity"]>.95 and r["postselected_fidelity"]>.999 for r in subset);runtime[ib,ia]=np.mean([r["duration_seconds"] for r in subset])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout="constrained")
    for ax,array,title in zip(axes,(calibrated,passing,runtime),("Calibrated EC points","Points above both thresholds","Mean runtime per task (s)")):
        image=ax.imshow(array,origin="lower",aspect="auto",extent=[min(a_values)-.5,max(a_values)+.5,min(b_values)-.5,max(b_values)+.5],cmap="viridis");ax.set(title=title,xlabel="Logical A (MHz)",ylabel="Logical B (MHz)");plt.colorbar(image,ax=ax)
    fig.savefig(plots/"logical_grid_summary.png",dpi=200);fig.savefig(plots/"logical_grid_summary.pdf");plt.close(fig)
    archive_summary={"tasks":len(records),"compact_database":str(database),"compact_database_bytes":database.stat().st_size,"original_checkpoint_bytes":sum(path.stat().st_size for path in paths),"result_status_counts":Counter(r["result_status"] for r in compact),"joint_pass_points":sum(r["joint_pass"] for r in summaries),"plots":[p.name for p in sorted(plots.iterdir())]}
    (output/"archive_summary.json").write_text(json.dumps(archive_summary,indent=2,default=dict)+"\n")
    if prune:
        for path in paths:path.unlink()
        (output/"points").rmdir()
    return archive_summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("campaign",type=Path);parser.add_argument("--prune",action="store_true");parser.add_argument("--skip-reconciliation",action="store_true");parser.add_argument("--reconciliation-workers",type=int,default=4);args=parser.parse_args()
    campaign=args.campaign.resolve();result=archive(campaign,args.prune)
    if not args.skip_reconciliation:
        from reconcile_overlay import reconcile
        result["fixed_hardware_reconciliation"]=reconcile(campaign,args.reconciliation_workers,args.prune)
    print(json.dumps(result,indent=2,default=dict))


if __name__=="__main__":main()
