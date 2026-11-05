#!/usr/bin/env python3
"""Run the first twelve ordered configurations from the prior weekend list."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from datetime import UTC,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;SELECTIONS=HERE/"selections/rail_a_ec_selections.json"

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=HERE/"sampled_runs/tonight_first12_B4_5p5");p.add_argument("--parallel",type=int,default=4);p.add_argument("--workers",type=int,default=4);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);selections=json.loads(SELECTIONS.read_text())["selections"][:12];jobs=[];active={}
    for s in selections:
        ec=s["ec_a_mhz"];directory=out/f"{s['selection_id']}_ecA_{ec[0]}_{ec[1]}";jobs.append({"id":s["selection_id"],"ec":ec,"directory":directory,"state":"complete" if (directory/"gate_output/compaction_summary.json").exists() else "pending"})
    def save():
        (out/"campaign_status.json").write_text(json.dumps({"updated_utc":datetime.now(UTC).isoformat(),"parallel_repetitions":a.parallel,"workers_per_repetition":a.workers,"jobs":[{k:j[k] if k!="directory" else str(j[k]) for k in ("id","ec","directory","state")} for j in jobs]},indent=2)+"\n")
    save()
    while any(j["state"] not in ("complete","failed") for j in jobs):
        for j in jobs:
            if len(active)>=a.parallel:break
            if j["state"]!="pending":continue
            j["directory"].mkdir(parents=True,exist_ok=True);log=(j["directory"]/"repetition.log").open("a");cmd=[sys.executable,str(HERE/"run_sampled_anchor_repetition.py"),"--selection-id",j["id"],"--ec-a-mhz",*map(str,j["ec"]),"--output",str(j["directory"]),"--workers",str(a.workers)];process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);j.update(state="running",process=process,log=log);active[process.pid]=j;save()
        time.sleep(5)
        for pid,j in list(active.items()):
            code=j["process"].poll()
            if code is None:continue
            j["log"].close();active.pop(pid);j["state"]="complete" if code==0 and (j["directory"]/"gate_output/compaction_summary.json").exists() else "failed";save()
    return 1 if any(j["state"]=="failed" for j in jobs) else 0

if __name__=="__main__":raise SystemExit(main())
