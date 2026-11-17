#!/usr/bin/env python3
"""Run and plot coarse rail-A C1-C2 maps at two bare cavity frequencies."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PASS={"validated_pass","recovered_viable"}

def utc(): return datetime.now(UTC).isoformat()
def atomic(path,value):
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(path)

def plot_map(root: Path, frequencies: list[float]):
    datasets=[]
    for frequency in frequencies:
        directory=root/f"bare_cavity_{frequency:.2f}GHz"
        rows=json.loads((directory/"results.json").read_text());passed=[r for r in rows if r["status"] in PASS]
        datasets.append((frequency,rows,passed))
    fig,axes=plt.subplots(2,len(datasets),figsize=(12,10),sharex=True,sharey=True,layout="constrained")
    scatter=None
    for col,(frequency,rows,passed) in enumerate(datasets):
        ax=axes[0,col];ax.scatter([r["c1"] for r in rows],[r["c2"] for r in rows],s=12,marker="s",color="#dedede",linewidths=0)
        ax.scatter([r["c1"] for r in passed],[r["c2"] for r in passed],s=18,marker="s",color="#16865c",linewidths=0)
        ax.set_title(f"Bare cavity {frequency:.2f} GHz\n{len(passed):,}/{len(rows):,} feasible")
        ax=axes[1,col];ax.scatter([r["c1"] for r in rows],[r["c2"] for r in rows],s=12,marker="s",color="#dedede",linewidths=0)
        if passed:
            scatter=ax.scatter([r["c1"] for r in passed],[r["c2"] for r in passed],s=20,marker="s",c=[1000*r["point"]["splitting_ghz"] for r in passed],cmap="viridis",vmin=140,vmax=160,linewidths=0)
        ax.set_title("Logical frequency on feasible region")
    for ax in axes.flat:
        ax.set(xlabel=r"$C_1$ (fF)",ylabel=r"$C_2$ (fF)");ax.set_aspect("equal");ax.plot([36,128],[36,128],color="0.7",lw=.8)
    if scatter is not None: fig.colorbar(scatter,ax=axes[1,:],label="Logical frequency (MHz)",shrink=.86)
    fig.suptitle("Rail-A coarse C1-C2 readout maps (logical frequency 140-160 MHz)",fontsize=15)
    fig.savefig(root/"rail_a_c1_c2_bare_cavity_comparison.png",dpi=220);fig.savefig(root/"rail_a_c1_c2_bare_cavity_comparison.pdf");plt.close(fig)
    sets=[{(r["c1"],r["c2"]) for r in passed} for _,_,passed in datasets]
    summary={"frequencies_ghz":frequencies,"grid_step_ff":2,"counts":{f"{f:.2f}":len(p) for f,_,p in datasets},"joint_feasible":len(set.intersection(*sets)),"only_lower_frequency":len(sets[0]-sets[1]),"only_higher_frequency":len(sets[1]-sets[0])}
    atomic(root/"comparison_summary.json",summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--workers-per-map",type=int,default=2);p.add_argument("--background",action="store_true");a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);frequencies=[5.95,6.05]
    if a.background:
        command=[sys.executable,str(Path(__file__).resolve()),"--output",str(root),"--workers-per-map",str(a.workers_per_map)]
        with (root/"supervisor.log").open("a") as log: process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(root/"launch.json",{"pid":process.pid,"started_utc":utc(),"command":command});print(json.dumps({"pid":process.pid,"output":str(root)},indent=2));return
    processes=[]
    for frequency in frequencies:
        directory=root/f"bare_cavity_{frequency:.2f}GHz";directory.mkdir(exist_ok=True);log=(directory/"run.log").open("a")
        command=[sys.executable,"-m","readout_workflow.fine_readout_feasibility","--run","--output",str(directory),"--workers",str(a.workers_per_map),"--logical-frequency-mhz","140","160","--abs-mean-chi-min-mhz","1","--bare-resonator-ghz",str(frequency),"--capacitance-step-ff","2"]
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);processes.append((frequency,process,log))
    atomic(root/"comparison_status.json",{"state":"running","started_utc":utc(),"frequencies_ghz":frequencies,"workers_per_map":a.workers_per_map})
    codes=[]
    for frequency,process,log in processes: codes.append((frequency,process.wait()));log.close()
    if any(code for _,code in codes):
        atomic(root/"comparison_status.json",{"state":"failed","updated_utc":utc(),"return_codes":codes})
        raise RuntimeError(f"Map run failed: {codes}")
    summary=plot_map(root,frequencies)
    atomic(root/"comparison_status.json",{"state":"complete","updated_utc":utc(),"return_codes":codes,"summary":summary})
    print(json.dumps(summary,indent=2))
    summary=plot_map(root,frequencies);atomic(root/"comparison_status.json",{"state":"complete","updated_utc":utc(),"return_codes":codes,"summary":summary});print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
