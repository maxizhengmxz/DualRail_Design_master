#!/usr/bin/env python3
"""Compact verified anchor checkpoints and remove duplicated per-task JSON."""
from __future__ import annotations
import argparse,json,sqlite3
from pathlib import Path

def run(campaign,prune):
    output=campaign/"gate_output";manifest=json.loads((output/"manifest.json").read_text())
    if manifest["status"]!="complete":raise RuntimeError("Gate campaign is incomplete")
    paths=sorted((output/"points").glob("*.json"));
    if len(paths)!=manifest["total"]:raise RuntimeError("Gate checkpoint count mismatch")
    db=output/"compact_anchor_results.sqlite";con=sqlite3.connect(db);con.execute("DROP TABLE IF EXISTS gate_results");con.execute("CREATE TABLE gate_results(task_id TEXT PRIMARY KEY,ec_b1_mhz REAL,ec_b2_mhz REAL,status TEXT,raw_fidelity REAL,postselected_fidelity REAL,selected_anchor_id TEXT,result_json TEXT)")
    rows=[]
    for path in paths:
        record=json.loads(path.read_text());result=record["result"];final=result.get("final") or {};ec=record["payload"]["ec_GHz"]
        rows.append((record["task_id"],1000*ec[2],1000*ec[3],result.get("workflow_status") or result.get("status"),final.get("raw_fidelity"),final.get("postselected_fidelity"),(result.get("selected_anchor") or {}).get("anchor_id"),json.dumps(result,separators=(",",":"))))
    con.executemany("INSERT INTO gate_results VALUES(?,?,?,?,?,?,?,?)",rows);con.commit();count=con.execute("SELECT count(*) FROM gate_results").fetchone()[0];con.close()
    if count!=manifest["total"]:raise RuntimeError("Compact gate database verification failed")
    removed=0
    if prune:
        for path in paths:removed+=path.stat().st_size;path.unlink()
        (output/"points").rmdir()
        recon=campaign/"fixed_hardware_reconciliation/points"
        if recon.exists():
            for path in recon.glob("*.json"):removed+=path.stat().st_size;path.unlink()
            recon.rmdir()
    summary={"tasks":count,"database":str(db),"database_bytes":db.stat().st_size,"pruned":prune,"removed_checkpoint_bytes":removed};(output/"compaction_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return summary

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("campaign",type=Path);p.add_argument("--prune",action="store_true");a=p.parse_args();print(json.dumps(run(a.campaign.resolve(),a.prune),indent=2))
