"""Existential readout feasibility versus physical operating frequency.

For one fixed loaded-(EC1, EC2) pair, this driver asks whether *some* allowed
logical splitting, C12 and equal readout coupling Cg satisfies the established
readout policy at each requested dressed operating frequency.  The complete
Boolean scan is retained and disconnected feasible intervals are labelled.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from readout_workflow import calculation as s
from readout_workflow.direct_ec_feasibility import capacitances, point
from readout_workflow.storage import atomic_json, utc
from readout_workflow.validation import POLICY, audit_point


def connected_intervals(rows):
    """Label runs of adjacent passing samples; never bridge a failed sample."""
    ordered = sorted(rows, key=lambda row: row["target_frequency_ghz"])
    intervals = []
    current = []
    for row in ordered:
        if row["status"] == "validated_pass":
            current.append(row)
        elif current:
            intervals.append(current); current = []
    if current:
        intervals.append(current)
    for component, group in enumerate(intervals, 1):
        for row in group:
            row["component"] = component
    return [dict(component=i, sample_count=len(group),
                 sampled_lower_ghz=group[0]["target_frequency_ghz"],
                 sampled_upper_ghz=group[-1]["target_frequency_ghz"])
            for i, group in enumerate(intervals, 1)]


def plot_results(rows, output):
    ordered=sorted(rows,key=lambda row:row["target_frequency_ghz"])
    frequency=np.array([row["target_frequency_ghz"] for row in ordered])
    passed=np.array([row["status"]=="validated_pass" for row in ordered])
    logical=np.array([1000*row["point"]["splitting_ghz"] if row.get("point") else np.nan for row in ordered])
    figure,axes=plt.subplots(2,1,figsize=(9,6.5),sharex=True,layout="constrained")
    axes[0].step(frequency,passed.astype(int),where="mid",color="tab:blue")
    axes[0].set_yticks([0,1],["fail","pass"]); axes[0].set_ylabel("Readout feasibility")
    scatter=axes[1].scatter(frequency[passed],logical[passed],c=[row.get("component") for row in ordered if row["status"]=="validated_pass"],cmap="tab10",s=28)
    axes[1].axhspan(140,160,color="0.9",zorder=-1); axes[1].set_ylim(139,161)
    axes[1].set_xlabel("Dressed physical operating frequency (GHz)"); axes[1].set_ylabel("Logical frequency (MHz)")
    figure.savefig(output/"physical_frequency_feasibility.png",dpi=220)
    figure.savefig(output/"physical_frequency_feasibility.pdf")
    plt.close(figure)


def initialize(logical_mhz):
    bounds = tuple(float(value) / 1000 for value in logical_mhz)
    s.CONDITIONS = replace(s.CONDITIONS, logical_frequency_ghz=bounds)
    s.BASELINE = dict(s.BASELINE, logical_frequency=bounds)


def solve_frequency(ec_mhz, target_ghz, warm, logical_mhz, max_nfev=55):
    initialize(logical_mhz)
    ec1, ec2 = map(float, ec_mhz)
    lower, upper = map(np.asarray, s.nominal_search_bounds())
    cache = {}
    started = time.monotonic()

    def evaluate(x):
        key = tuple(map(float, x))
        if key not in cache:
            cache[key] = point(ec1, ec2, x, s.SCREENING_BASIS)
        return cache[key]

    def residual(x):
        candidate = evaluate(x)
        device = candidate["device"]
        return np.r_[s.violations(candidate, margin=.3),
                     (candidate["operating_frequency_ghz"] - target_ghz) / .001,
                     max(0, 36-device["C1_shunt_fF"])/10,
                     max(0, device["C1_shunt_fF"]-128)/10,
                     max(0, 36-device["C2_shunt_fF"])/10,
                     max(0, device["C2_shunt_fF"]-128)/10]

    seeds = []
    for source in warm:
        x = np.asarray(source.get("x", s.x_from_point(source)), float).copy()
        x[0] += target_ghz - source.get("operating_frequency_ghz", x[0])
        x = np.clip(x, lower+1e-7, upper-1e-7)
        if not any(np.linalg.norm(x-prior) < 1e-6 for prior in seeds): seeds.append(x)
    attempts=[]; chosen=None
    with threadpool_limits(limits=1):
        for seed in seeds[:4]:
            fit=least_squares(residual, seed, bounds=(lower,upper),
                              x_scale=[1,1,2,5], diff_step=2e-5,
                              ftol=1e-9,xtol=1e-9,gtol=1e-9,max_nfev=max_nfev)
            candidate=audit_point(evaluate(fit.x))
            frequency_error=abs(candidate["operating_frequency_ghz"]-target_ghz)
            failures=s.failures(candidate,require_sign_check=True)
            if frequency_error>s.CONDITIONS.fixed_frequency_tolerance_ghz:
                failures.append("fixed_operation_frequency")
            attempts.append(dict(nfev=int(fit.nfev),cost=float(fit.cost),
                                 frequency_error_ghz=frequency_error,failures=failures))
            if not failures:
                chosen=candidate; break
    return dict(target_frequency_ghz=float(target_ghz),
                status="validated_pass" if chosen else "no_validated_solution_found",
                point=chosen,attempts=attempts,seconds=time.monotonic()-started,
                completed_utc=utc(),component=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ec-mhz",nargs=2,type=float,required=True,metavar=("EC1","EC2"))
    parser.add_argument("--frequency-ghz",nargs=3,type=float,default=(4,8,.025),metavar=("LOW","HIGH","STEP"))
    parser.add_argument("--logical-frequency-mhz",nargs=2,type=float,default=(140,160))
    parser.add_argument("--warm-results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-nfev",type=int,default=55)
    args=parser.parse_args(); initialize(args.logical_frequency_mhz)
    low,high,step=args.frequency_ghz
    if not 4<=low<high<=8 or step<=0: raise ValueError("invalid frequency grid")
    rows=json.loads(args.warm_results.read_text())
    warm=[row["point"] for row in rows if row.get("status")=="validated_pass" and
          (row.get("ec1_target_mhz"),row.get("ec2_target_mhz"))==tuple(args.ec_mhz)]
    if not warm: raise RuntimeError("No validated warm witness for the requested ordered EC pair")
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    settings=dict(ec_mhz=args.ec_mhz,frequency_grid_ghz=[low,high,step],
                  logical_frequency_mhz=args.logical_frequency_mhz,
                  C12_bounds_ff=s.CONDITIONS.c12_bounds_ff,
                  Cg_bounds_ff=s.CONDITIONS.readout_cap_bounds_ff,
                  validation_policy=POLICY,conditions=asdict(s.CONDITIONS))
    atomic_json(output/"settings.json",settings)
    targets=np.arange(low,high+step/2,step)
    results=[]
    for index,target in enumerate(targets,1):
        path=output/f"frequency_{target:.6f}.json"
        row=json.loads(path.read_text()) if path.exists() else solve_frequency(args.ec_mhz,target,warm,args.logical_frequency_mhz,args.max_nfev)
        if not path.exists(): atomic_json(path,row)
        results.append(row)
        atomic_json(output/"progress.json",dict(total=len(targets),completed=index,passed=sum(r["status"]=="validated_pass" for r in results),updated_utc=utc()))
        print(json.dumps(dict(completed=index,total=len(targets),frequency_ghz=target,status=row["status"],seconds=row["seconds"])),flush=True)
    intervals=connected_intervals(results)
    atomic_json(output/"results.json",results)
    atomic_json(output/"summary.json",dict(settings=settings,intervals=intervals,total=len(results),passed=sum(r["status"]=="validated_pass" for r in results),runtime_seconds=sum(r["seconds"] for r in results)))
    plot_results(results,output)


if __name__=="__main__": main()
