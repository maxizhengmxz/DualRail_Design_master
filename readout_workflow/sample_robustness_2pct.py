"""300 paired +/-2% capacitance scenarios, with EJ recalibration each time."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import combinations, product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc, spearmanr
from threadpoolctl import threadpool_limits

from model import Basis, Device, ReadoutModel
from search_equal_g import annotate
from study_capacitance_robustness import CarrierSlice, CAPS

OUT=Path(__file__).resolve().parent/"data"/"equal_g"/"robustness"/"samples_2pct"
SEED=20261014


def scenarios():
    # 300 distinct variation vectors per design. Interior samples are separated
    # from stress corners and controlled one-at-a-time endpoint experiments.
    rows=[]
    for values in 2*qmc.LatinHypercube(d=5,seed=SEED).random(258)-1:
        rows.append({"kind":"interior","errors":(.02*values).tolist()})
    for values in product((-1,1),repeat=5):
        rows.append({"kind":"corner","errors":(.02*np.array(values)).tolist()})
    for i in range(5):
        for sign in (-1,1):
            values=np.zeros(5);values[i]=sign*.02
            rows.append({"kind":"one_at_a_time","parameter":CAPS[i],"errors":values.tolist()})
    return [dict(row,id=i) for i,row in enumerate(rows)]


def failure_reasons(p,limit=.009):
    if p is None: return ["no_matching_candidate_found"]
    reasons=[]
    tests={"hybridization":p["hybridization"]>=.95,
        "chi_matching":max(p["ratio_m"],p["abs_delta_chi_over_chi_p"])<=limit+1e-7,
        "dispersive":p["lambda_max"]<=.1,"assignment_overlap":p["min_assignment_overlap"]>=.95,
        "logical_gap":.095<=p["splitting_ghz"]<=.105,
        "EC_range":150<=min(p["loaded_ec1_mhz"],p["loaded_ec2_mhz"])
                     and max(p["loaded_ec1_mhz"],p["loaded_ec2_mhz"])<=350,
        "EJ_over_EC":min(p["ej1_over_ec1"],p["ej2_over_ec2"])>50,
        "EJ_limit":max(p["ej1_ghz"],p["ej2_ghz"])<25}
    for key,ok in tests.items():
        if not ok: reasons.append(key)
    return reasons


def evaluate(task):
    label,reference,scenario,limit=task
    with threadpool_limits(limits=1):
        d=replace(Device(**reference["device"]),**{
            key:reference["device"][key]*(1+error) for key,error in zip(CAPS,scenario["errors"])})
        m=ReadoutModel(d)
        p=CarrierSlice(m,reference).constrained(limit=limit)
        if p is not None: p=annotate(m,p)
        reasons=failure_reasons(p,limit)
    return {"design":label,**scenario,"matching_limit":limit,"point":p,"failure_reasons":reasons,"passes":not reasons}


def validate(rows):
    checks=[]
    with threadpool_limits(limits=1):
        for label in ("A","B"):
            group=[r for r in rows if r["design"]==label and r["point"] is not None]
            passing=sorted([r for r in group if r["passes"]],key=lambda r:r["point"]["hybridization"])
            failing=sorted([r for r in group if not r["passes"]],key=lambda r:abs(r["point"]["hybridization"]-.95))
            selected=passing[:3]+failing[:3]+[min(group,key=lambda r:r["point"]["hybridization"])]
            for row in selected:
                p=row["point"]
                m=ReadoutModel(Device(**p["device"]),Basis(20,9,8,30))
                q=annotate(m,m.evaluate(p["ej1_ghz"],p["ej2_ghz"]))
                checks.append({"design":label,"id":row["id"],"small_basis_pass":row["passes"],
                    "large_basis_pass":not failure_reasons(q,row.get("matching_limit",.009)),"point":q,
                    "H_difference":q["hybridization"]-p["hybridization"]})
    (OUT/"validation.json").write_text(json.dumps(checks,indent=2))
    return checks


def render(rows,nominals,checks):
    limit=rows[0].get("matching_limit",.009)
    analyses={}
    for label in ("A","B"):
        group=[r for r in rows if r["design"]==label]
        interior=[r for r in group if r["kind"]=="interior" and r["point"] is not None]
        x=np.array([r["errors"] for r in interior])/.02
        y=np.array([r["point"]["hybridization"] for r in interior])
        pairs=list(combinations(range(5),2))
        features=np.column_stack([np.ones(len(x)),x]+[x[:,i]*x[:,j] for i,j in pairs])
        coeff=np.linalg.lstsq(features,y,rcond=None)[0]
        fitted=features@coeff
        r2=1-sum((y-fitted)**2)/sum((y-y.mean())**2)
        oat=[]
        for i,key in enumerate(CAPS):
            endpoints=sorted([r for r in group if r["kind"]=="one_at_a_time" and r["parameter"]==key],key=lambda r:r["errors"][i])
            oat.append({"parameter":key,"H_minus":endpoints[0]["point"]["hybridization"],
                "H_plus":endpoints[1]["point"]["hybridization"],
                "both_pass":all(r["passes"] for r in endpoints),
                "worst_H_loss":nominals[label]["hybridization"]-min(r["point"]["hybridization"] for r in endpoints)})
        reasons={}
        for row in group:
            for reason in row["failure_reasons"]: reasons[reason]=reasons.get(reason,0)+1
        corr=[float(spearmanr(x[:,i],y).statistic) for i in range(5)]
        readout_diff=x[:,3]-x[:,4]
        analysis={"counts":{kind:{"passes":sum(r["passes"] for r in group if r["kind"]==kind),
            "total":sum(r["kind"]==kind for r in group)} for kind in ("interior","corner","one_at_a_time")},
            "all_passes":sum(r["passes"] for r in group),"failure_reasons":reasons,
            "nominal":nominals[label],"oat":oat,"spearman_H":dict(zip(CAPS,corr)),
            "linear_H_coefficients":dict(zip(CAPS,coeff[1:6].tolist())),
            "interaction_H_coefficients":{CAPS[i]+" x "+CAPS[j]:float(v) for (i,j),v in zip(pairs,coeff[6:])},
            "regression_r2":float(r2),
            "spearman_readout_difference_H":float(spearmanr(readout_diff,y).statistic),
            "g_asymmetry_range":[min(r["point"]["g_relative_difference"] for r in interior),max(r["point"]["g_relative_difference"] for r in interior)]}
        analysis["readout_error_quadrants"]={}
        for sign1,sign2 in product((-1,1),repeat=2):
            subset=[r for r in interior if r["errors"][3]*sign1>0 and r["errors"][4]*sign2>0]
            analysis["readout_error_quadrants"][f"C1r{sign1:+d}, C2r{sign2:+d}"]={
                "passes":sum(r["passes"] for r in subset),"total":len(subset),
                "mean_H":float(np.mean([r["point"]["hybridization"] for r in subset]))}
        analyses[label]=analysis
    (OUT/"analysis.json").write_text(json.dumps(analyses,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    names=["C1 shunt","C2 shunt","C12","C1r","C2r"]
    for color,(label,a) in zip(("tab:blue","tab:orange"),analyses.items()):
        interior=[r for r in rows if r["design"]==label and r["kind"]=="interior" and r["point"] is not None]
        axes[0,0].scatter([100*(r["errors"][3]-r["errors"][4]) for r in interior],
            [r["point"]["hybridization"] for r in interior],s=12,alpha=.55,label=label,color=color)
        offset=-.18 if label=="A" else .18
        axes[0,1].bar(np.arange(5)+offset,[100*d["worst_H_loss"] for d in a["oat"]],width=.36,label=label,color=color)
        axes[1,0].scatter([r["point"]["splitting_ghz"]*1000 for r in interior],
            [r["point"]["lambda_max"] for r in interior],s=12,alpha=.55,label=label,color=color)
        axes[1,1].bar(np.arange(5)+offset,[a["linear_H_coefficients"][key]*100 for key in CAPS],width=.36,label=label,color=color)
    axes[0,0].axhline(.95,color="black",ls="--")
    axes[0,0].set_xlabel("Relative C1r error minus C2r error (percentage points)")
    axes[0,0].set_ylabel("Recalibrated hybridization score")
    axes[0,1].set_ylabel("Worst isolated +/-2% loss in H (percentage points)")
    axes[1,0].axhline(.1,color="black",ls="--")
    for gap in (95,105): axes[1,0].axvline(gap,color="gray",ls=":")
    axes[1,0].set_xlabel("Recalibrated logical splitting (MHz)")
    axes[1,0].set_ylabel("Maximum dispersive ratio")
    axes[1,1].set_ylabel("Fitted change in H for +2% error (percentage points)")
    for ax in (axes[0,1],axes[1,1]): ax.set_xticks(np.arange(5),names)
    for ax in axes.flat: ax.grid(alpha=.2);ax.legend()
    fig.suptitle("Independent +/-2% capacitance variation; both EJs recalibrated at 4.4 GHz")
    fig.savefig(OUT/"sensitivity.png",dpi=180)
    fig.savefig(OUT/"sensitivity.pdf")
    lines=["# 300-scenario +/-2% capacitance robustness study","",
        "Each of C1_shunt, C2_shunt, C12, C1r and C2r is independently varied between 0.98 and 1.02 times its nominal value. Cr and Lr are held fixed (6 GHz bare resonator, 50 ohm equivalent-mode impedance). Both EJs are recalibrated for every sample, permitting detuning from maximum hybridization, at a fixed dressed carrier of 4.4 GHz. Exact equal g is not imposed after fabrication.","",
        f"There are 300 perturbation vectors per design: 258 Latin-hypercube interior vectors (seed {SEED}), all 32 extreme corners, and 10 isolated single-parameter +/-2% endpoints. The same 300 vectors are applied to A and B for paired comparison: 600 circuit recalibrations total. Interior sampling assumes independent uniform marginal errors as a sampling convention, not a measured fabrication distribution. Stress corners and isolated controls are reported separately, so the combined count is not a yield estimate.","",
        f"Recalibration maximizes H within the mismatch <={limit:g} set, then checks H>=0.95, max multilevel dispersive ratio<=0.1, gap 95-105 MHz, both EJ/h<25 GHz, both loaded EC/h=150-350 MHz, both EJ/EC>50, and branch overlap>=0.95. Numerical ratio comparisons allow 1e-7 absolute tolerance. The EJ search is first recentered using the perturbed ECs and numerical local transmon frequencies; EJ2 is then searched within +/-0.18 GHz of that updated reference, and EJ1 adjusts to keep the carrier fixed. Reported failures are failures of this bounded calibration procedure, not global infeasibility over every carrier.","",
        "Correction to the earlier corner study: its search interval was centered on the old EJ values, so an EC change could move the true hybridization maximum outside the interval. An interval edge could then be treated as the maximum and accepted before finding a better detuned solution. This study corrects that issue and supersedes the earlier +/-2% corner counts. The earlier +/-1% and +/-5% counts have not been recalculated here.","",
        "| Design | Interior passes | Corner passes | Isolated endpoint passes | Combined passes |",
        "|---|---:|---:|---:|---:|"]
    for label,a in analyses.items():
        counts=a["counts"]
        lines.append(f"| {label} | {counts['interior']['passes']}/258 | {counts['corner']['passes']}/32 | {counts['one_at_a_time']['passes']}/10 | {a['all_passes']}/300 |")
    lines += ["","## Parameter sensitivity after EJ recalibration","",
        "Worst H loss is the nominal recalibrated H minus the lower H at the two isolated +/-2% endpoints. A larger value means less robust in this controlled test; negative means both endpoints improved H.","",
        "| Parameter | A: H at -2%, +2% | A: worst H loss | B: H at -2%, +2% | B: worst H loss |",
        "|---|---:|---:|---:|---:|"]
    for i,key in enumerate(CAPS):
        a,b=analyses["A"]["oat"][i],analyses["B"]["oat"][i]
        lines.append(f"| {key} | {a['H_minus']:.6f}, {a['H_plus']:.6f} | {a['worst_H_loss']:.6f} | {b['H_minus']:.6f}, {b['H_plus']:.6f} | {b['worst_H_loss']:.6f} |")
    lines += ["","## Combinations and associations","",
        "The input errors were sampled independently. Correlations below describe their association with the recalibrated output, not an assumed correlation in fabrication. A negative Spearman coefficient means a positive error tends to reduce H. Linear-plus-pairwise-interaction fits use errors normalized by 2%, and are descriptive sensitivity models, not substitutes for the Hamiltonian calculation."]
    for label,a in analyses.items():
        lines += ["",f"### Design {label}","",f"Failure reasons (can overlap): {a['failure_reasons']}.","",
            f"Spearman correlation of differential readout-cap error (epsilon_C1r - epsilon_C2r) with H: {a['spearman_readout_difference_H']:.4f}. Regression R-squared: {a['regression_r2']:.5f}.","",
            "| Parameter | Spearman with H | Fitted linear H change for +2% |","|---|---:|---:|"]
        for key in CAPS: lines.append(f"| {key} | {a['spearman_H'][key]:.4f} | {a['linear_H_coefficients'][key]:.6f} |")
        ranking=sorted(a["oat"],key=lambda item:item["worst_H_loss"],reverse=True)
        lines += ["","Isolated-endpoint sensitivity ranking, most to least loss of H: "+
            " > ".join(item["parameter"] for item in ranking)+". This ranking is specific to this nominal circuit and recalibration protocol."]
        lines += ["","Readout-cap error sign combinations, using interior samples only:","",
            "| Error signs | Passing / sampled | Mean recalibrated H |","|---|---:|---:|"]
        for key,values in a["readout_error_quadrants"].items():
            lines.append(f"| {key} | {values['passes']}/{values['total']} | {values['mean_H']:.6f} |")
        lines += ["","Largest fitted pairwise interaction terms:",""]
        for key,value in sorted(a["interaction_H_coefficients"].items(),key=lambda item:abs(item[1]),reverse=True)[:3]:
            lines.append(f"- {key}: {value:.6g} in H at normalized product 1.")
        failed=[r for r in rows if r["design"]==label and not r["passes"] and r["kind"]=="interior"]
        if failed:
            mean=np.mean([r["errors"] for r in failed],axis=0)*100
            lines += ["","Mean relative errors among failed interior samples (percent): "+", ".join(f"{k}={v:+.3f}%" for k,v in zip(CAPS,mean))+"."]
    changed=[c for c in checks if c["small_basis_pass"]!=c["large_basis_pass"]]
    lines += ["","## Numerical checks","",
        f"{len(checks)} selected points, including near-threshold passes/failures and worst-H cases, were recomputed at unchanged physical parameters in basis 20/9/8/30 versus screening basis 15/6/5/20. Pass/fail changes: {len(changed)}. All samples and retuned parameters are in samples.json; validation.json records the larger-basis checks. Finite-drive and frequency-noise performance are not simulated.","",
        "Reproduce with `.\\.venv\\Scripts\\python.exe readout_workflow/sample_robustness_2pct.py`. Use `--report-only` to regenerate figures and analysis from saved results."]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({label:{"counts":a["counts"],"failure_reasons":a["failure_reasons"]}
                      for label,a in analyses.items()},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--matching-limit",type=float,default=.009)
    args=parser.parse_args()
    global OUT
    root=OUT.parent
    if args.matching_limit != .009:
        OUT=root/f"samples_2pct_limit_{args.matching_limit:g}"
    OUT.mkdir(parents=True,exist_ok=True)
    entries=json.loads((root/"tradeoff.json").read_text())
    nominals={entry["label"]:entry["constrained_maximum"] for entry in entries}
    if args.matching_limit != .009:
        with threadpool_limits(limits=1):
            for label,p in list(nominals.items()):
                m=ReadoutModel(Device(**p["device"]))
                nominals[label]=annotate(m,CarrierSlice(m,p).constrained(args.matching_limit))
    if args.report_only:
        rows=json.loads((OUT/"samples.json").read_text())
        checks=json.loads((OUT/"validation.json").read_text())
    else:
        vectors=scenarios()
        (OUT/"sampling.json").write_text(json.dumps({"seed":SEED,"parameters":CAPS,"scenarios":vectors},indent=2))
        rows=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending=[pool.submit(evaluate,(label,reference,scenario,args.matching_limit))
                     for label,reference in nominals.items() for scenario in vectors]
            for future in as_completed(pending):
                rows.append(future.result())
                if len(rows)%25==0:
                    print(f"Completed {len(rows)}/600 recalibrations",flush=True)
                    (OUT/"samples.json").write_text(json.dumps(sorted(rows,key=lambda r:(r["design"],r["id"])),indent=2))
        rows.sort(key=lambda r:(r["design"],r["id"]))
        (OUT/"samples.json").write_text(json.dumps(rows,indent=2))
        checks=validate(rows)
    render(rows,nominals,checks)


if __name__=="__main__": main()
