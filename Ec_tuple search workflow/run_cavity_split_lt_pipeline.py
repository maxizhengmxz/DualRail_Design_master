#!/usr/bin/env python3
"""Continue the A=6.05/B=5.95 readout rebuild into the LT gate campaign."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
def utc():return datetime.now(UTC).isoformat()
def atomic(path,value):
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(path)
def wait_for(path):
    while not path.exists():time.sleep(30)

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--background",action="store_true");p.add_argument("--direct-workers-per-map",type=int,default=2);p.add_argument("--parallel-repetitions",type=int,default=4);p.add_argument("--gate-workers",type=int,default=4);a=p.parse_args();base=a.root.resolve();base.mkdir(parents=True,exist_ok=True)
    if a.background:
        command=[sys.executable,str(Path(__file__).resolve()),"--root",str(base),"--direct-workers-per-map",str(a.direct_workers_per_map),"--parallel-repetitions",str(a.parallel_repetitions),"--gate-workers",str(a.gate_workers)]
        with (base/"pipeline.log").open("a") as log:process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(base/"pipeline_launch.json",{"pid":process.pid,"started_utc":utc(),"command":command});print(json.dumps({"pid":process.pid,"root":str(base)},indent=2));return
    fine=base/"readout/fine_c1_c2";direct=base/"readout/direct_ec";direct.mkdir(parents=True,exist_ok=True)
    specs=[("rail_a",fine/"rail_a_140_160_fr6p05/results.json",direct/"rail_a_step2_fr6p05",(140,160),6.05),("rail_b",fine/"rail_b_80_100_fr5p95/results.json",direct/"rail_b_step2_fr5p95",(80,100),5.95)]
    atomic(base/"pipeline_status.json",{"phase":"waiting_for_fine_maps","updated_utc":utc()})
    for _,source,_,_,_ in specs:wait_for(source)
    processes=[]
    for name,source,output,logical,frequency in specs:
        if (output/"results.json").exists():continue
        output.mkdir(parents=True,exist_ok=True);log=(output/"run.log").open("a")
        command=[sys.executable,"-m","readout_workflow.direct_ec_feasibility","--run","--source",str(source),"--output",str(output),"--workers",str(a.direct_workers_per_map),"--logical-frequency-mhz",*map(str,logical),"--bare-resonator-ghz",str(frequency),"--ec-step-mhz","2","--domain","hull","--orientation-domain","full","--hull-scale","1.5"]
        processes.append((name,subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log))
    atomic(base/"pipeline_status.json",{"phase":"direct_ec_maps","updated_utc":utc(),"running":[x[0] for x in processes]})
    codes=[]
    for name,process,log in processes:codes.append((name,process.wait()));log.close()
    if any(code for _,code in codes):atomic(base/"pipeline_status.json",{"phase":"failed_direct_ec","updated_utc":utc(),"return_codes":codes});raise RuntimeError(codes)
    a_results=direct/"rail_a_step2_fr6p05/results.json";b_results=direct/"rail_b_step2_fr5p95/results.json";campaign=base/"sampled_runs/rail_a_lt_partition"
    command=[sys.executable,str(HERE/"run_readout_partition_campaign.py"),"--side","lt","--output",str(campaign),"--parallel",str(a.parallel_repetitions),"--workers",str(a.gate_workers),"--rail-a-results",str(a_results),"--rail-b-results",str(b_results),"--rail-a-bare-resonator-ghz","6.05"]
    atomic(base/"pipeline_status.json",{"phase":"lt_partition_campaign","updated_utc":utc(),"command":command})
    code=subprocess.run(command,cwd=ROOT).returncode
    atomic(base/"pipeline_status.json",{"phase":"complete" if code==0 else "failed_lt_campaign","updated_utc":utc(),"return_code":code})
    raise SystemExit(code)

if __name__=="__main__":main()
