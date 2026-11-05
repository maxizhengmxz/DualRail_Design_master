#!/usr/bin/env python3
"""Wait for a gate campaign, then postprocess and reconcile rail B."""
import argparse,json,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument("campaign",type=Path);p.add_argument("--workers",type=int,default=4);p.add_argument("--compact-and-prune",action="store_true");a=p.parse_args();campaign=a.campaign.resolve();manifest=campaign/"gate_output/manifest.json"
while True:
    status=json.loads(manifest.read_text()).get("status") if manifest.exists() else None
    if status=="complete":break
    if status not in (None,"running"):raise RuntimeError(f"Gate campaign ended with status {status}")
    time.sleep(30)
root=Path(__file__).resolve().parents[1]
for script,extra in (("postprocess_anchor_campaign.py",[]),("reconcile_anchor_campaign.py",["--workers",str(a.workers)]),("plot_anchor_reconciled_metrics.py",[])):
    subprocess.run([sys.executable,str(root/"Ec_tuple search workflow"/script),str(campaign),*extra],cwd=root,check=True)
if a.compact_and_prune:
    subprocess.run([sys.executable,str(root/"Ec_tuple search workflow/compact_anchor_campaign.py"),str(campaign),"--prune"],cwd=root,check=True)
