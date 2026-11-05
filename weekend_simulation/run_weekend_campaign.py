#!/usr/bin/env python3
"""Prepare, run, archive, and resume the selected weekend repetitions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
SELECTIONS=HERE/"selections"/"rail_a_ec_selections.json"
CAMPAIGN=HERE/"combined_runs"/"production_logical_A140_B80"
STATUS=CAMPAIGN/"campaign_status.json"


def utc():return datetime.now(UTC).isoformat()


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+f".{os.getpid()}.tmp");temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def prepare(selection,workers):
    ec=selection["ec_a_mhz"];directory=CAMPAIGN/f"{selection['selection_id']}_ecA_{ec[0]}_{ec[1]}";directory.mkdir(parents=True,exist_ok=True)
    config=directory/"config.json"
    if not config.exists():
        subprocess.run([sys.executable,str(ROOT/"combined_feasibility_test"/"prepare_gate.py"),"--a-ec-mhz",*map(str,ec),"--logical-a-mhz","140","--logical-b-mhz","80","--config-output",str(config),"--gate-output-directory","gate_output","--workers",str(workers)],cwd=ROOT,check=True,stdout=(directory/"prepare.log").open("w"),stderr=subprocess.STDOUT)
    saved=json.loads(config.read_text())
    if saved.get("gate_domain_policy")!="all ordered EC targets; independent of readout status" or len(saved["tasks"])!=4955:raise RuntimeError(f"Invalid production gate domain in {config}")
    return directory,config


def gate_archived(directory):
    summary=directory/"gate_output"/"archive_summary.json";database=directory/"gate_output"/"compact_results.sqlite"
    return summary.exists() and database.exists() and not (directory/"gate_output"/"points").exists()


def reconciled(directory):
    summary=directory/"fixed_hardware_reconciliation"/"summary.json";database=directory/"fixed_hardware_reconciliation"/"fixed_hardware_results.sqlite"
    return summary.exists() and database.exists()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--parallel-repetitions",type=int,default=2);parser.add_argument("--workers-per-repetition",type=int,default=4);parser.add_argument("--background",action="store_true");args=parser.parse_args()
    if args.parallel_repetitions<1 or args.workers_per_repetition<1:raise ValueError("worker counts must be positive")
    if args.background:
        CAMPAIGN.mkdir(parents=True,exist_ok=True);launch=CAMPAIGN/"launch.json"
        if launch.exists():
            previous=json.loads(launch.read_text())
            try:os.kill(previous["pid"],0);raise RuntimeError("Weekend supervisor is already running")
            except ProcessLookupError:pass
        command=[sys.executable,str(Path(__file__).resolve()),"--parallel-repetitions",str(args.parallel_repetitions),"--workers-per-repetition",str(args.workers_per_repetition)]
        with (CAMPAIGN/"supervisor.log").open("a") as log:process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(launch,{"pid":process.pid,"started_utc":utc(),"command":command});print(json.dumps({"pid":process.pid,"campaign":str(CAMPAIGN),"logical_frequency_mhz":{"rail_a":140,"rail_b":80}}));return 0
    CAMPAIGN.mkdir(parents=True,exist_ok=True);manifest=json.loads(SELECTIONS.read_text());jobs=[]
    for selection in manifest["selections"]:
        directory,config=prepare(selection,args.workers_per_repetition)
        state="completed" if gate_archived(directory) and reconciled(directory) else "reconcile_pending" if gate_archived(directory) else "pending"
        jobs.append({"selection":selection,"directory":directory,"config":config,"state":state})
    active={};started=time.monotonic()
    def snapshot():
        write_json(STATUS,{"updated_utc":utc(),"logical_frequency_mhz":{"rail_a":140,"rail_b":80},"parallel_repetitions":args.parallel_repetitions,"workers_per_repetition":args.workers_per_repetition,"elapsed_session_seconds":time.monotonic()-started,"jobs":[{"selection_id":j["selection"]["selection_id"],"ec_a_mhz":j["selection"]["ec_a_mhz"],"state":j["state"],"directory":str(j["directory"])} for j in jobs]})
    snapshot()
    while any(j["state"] not in ("completed","failed") for j in jobs):
        for job in jobs:
            if len(active)>=args.parallel_repetitions:break
            if job["state"]=="reconcile_pending":
                log=(job["directory"]/"reconciliation.log").open("a")
                process=subprocess.Popen([sys.executable,str(HERE/"reconcile_overlay.py"),str(job["directory"]),"--workers",str(args.workers_per_repetition),"--prune-checkpoints"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job.update(state="reconciling",process=process,log=log,process_kind="reconciliation");active[process.pid]=job;snapshot();continue
            if job["state"]!="pending":continue
            log=(job["directory"]/"gate_runner.log").open("a")
            process=subprocess.Popen([sys.executable,"-m","xm_cz.runner",str(job["config"])],cwd=ROOT/"2Q-gate_XM",stdout=log,stderr=subprocess.STDOUT)
            job.update(state="running",process=process,log=log,process_kind="gate");active[process.pid]=job;snapshot()
        if not active:break
        time.sleep(5)
        for pid,job in list(active.items()):
            code=job["process"].poll()
            if code is None:continue
            job["log"].close();active.pop(pid)
            if code!=0:
                job["state"]="failed";snapshot();continue
            if job["process_kind"]=="reconciliation":
                job["state"]="completed" if reconciled(job["directory"]) else "failed";snapshot();continue
            job["state"]="archiving";snapshot()
            with (job["directory"]/"archive.log").open("a") as log:
                archived_run=subprocess.run([sys.executable,str(HERE/"archive_gate_campaign.py"),str(job["directory"]),"--prune"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            job["state"]="completed" if archived_run.returncode==0 and gate_archived(job["directory"]) and reconciled(job["directory"]) else "failed";snapshot()
    snapshot();failed=[j for j in jobs if j["state"]=="failed"]
    return 1 if failed else 0


if __name__=="__main__":raise SystemExit(main())
