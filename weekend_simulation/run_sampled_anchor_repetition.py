#!/usr/bin/env python3
"""Run one EC-A repetition from readout frequency scan through compaction."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
DIRECT=HERE/"readout/direct_ec/rail_a_step2/results.json"

def command(args,log,cwd=ROOT):
    with log.open("a") as stream:subprocess.run(args,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,check=True)

def anchors(rows,step=.025,fractions=(.1,.3,.5,.7,.9)):
    good=sorted((r for r in rows if r["status"]=="validated_pass"),key=lambda r:r["target_frequency_ghz"]);groups=[];group=[]
    for row in good:
        if group and row["target_frequency_ghz"]-group[-1]["target_frequency_ghz"]>step*1.5:groups.append(group);group=[]
        group.append(row)
    if group:groups.append(group)
    if not groups:raise RuntimeError("No connected readout-frequency component")
    group=max(groups,key=lambda g:(g[-1]["target_frequency_ghz"]-g[0]["target_frequency_ghz"],len(g)))
    low,high=group[0]["target_frequency_ghz"],group[-1]["target_frequency_ghz"]
    chosen=[]
    for fraction in fractions:
        target=low+fraction*(high-low);row=min(group,key=lambda r:abs(r["target_frequency_ghz"]-target))
        if row not in chosen:chosen.append(row)
    if len(chosen)<len(fractions):
        for row in group:
            if row not in chosen:chosen.append(row)
            if len(chosen)==len(fractions):break
    if len(chosen)<len(fractions):raise RuntimeError("Largest component has fewer than five validated samples")
    return [r["target_frequency_ghz"] for r in sorted(chosen,key=lambda r:r["target_frequency_ghz"])],(low,high),len(group)

def main():
    p=argparse.ArgumentParser();p.add_argument("--selection-id",required=True);p.add_argument("--ec-a-mhz",type=float,nargs=2,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=4);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    scan=out/"readout_frequency_scan";results=scan/"results.json"
    if not results.exists():command([sys.executable,"-m","readout_workflow.physical_frequency_bounds","--ec-mhz",*map(str,a.ec_a_mhz),"--frequency-ghz","4","8","0.025","--logical-frequency-mhz","140","160","--warm-results",str(DIRECT),"--output",str(scan)],out/"readout_scan.log")
    frequencies,component,count=anchors(json.loads(results.read_text()));(out/"anchor_selection.json").write_text(json.dumps({"selection_id":a.selection_id,"ec_a_mhz":a.ec_a_mhz,"largest_sampled_component_ghz":component,"component_sample_count":count,"sampling_fractions":[.1,.3,.5,.7,.9],"selected_validated_frequencies_ghz":frequencies},indent=2)+"\n")
    config=out/"config.json"
    if not config.exists():command([sys.executable,str(HERE/"prepare_fixed_a_anchor_campaign.py"),"--frequency-results",str(results),"--frequencies-ghz",*map(str,frequencies),"--rail-b-bare-frequency-ghz","4","5.5","--output",str(out),"--workers",str(a.workers)],out/"prepare.log")
    manifest=out/"gate_output/manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("status")!="complete":command([sys.executable,"-m","xm_cz.runner",str(config)],out/"gate.log",ROOT/"2Q-gate_XM")
    if not (out/"gate_output/compaction_summary.json").exists():command([sys.executable,str(HERE/"run_anchor_post_pipeline.py"),str(out),"--workers",str(a.workers),"--compact-and-prune"],out/"post.log")

if __name__=="__main__":main()
