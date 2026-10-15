"""Compare best achievable hybridization at 0.9% and 1% chi ceilings."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent/"data"/"equal_g"/"robustness"
OUT=ROOT/"samples_2pct_limit_0.01"


def main():
    old=json.loads((ROOT/"samples_2pct"/"samples.json").read_text())
    new=json.loads((OUT/"samples.json").read_text())
    asym=json.loads((ROOT/"asymmetry_1pct.json").read_text())
    checks=json.loads((OUT/"validation.json").read_text())
    oldmap={(r["design"],r["id"]):r for r in old}
    stats={}
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True,layout="constrained")
    for ax,label in zip(axes,("A","B")):
        group=[r for r in new if r["design"]==label and r["kind"]=="interior"]
        h1=np.array([r["point"]["hybridization"] for r in group])
        h0=np.array([oldmap[label,r["id"]]["point"]["hybridization"] for r in group])
        for values,color,name in ((h0,"tab:gray","Mismatch <=0.9%"),(h1,"tab:blue","Mismatch <=1.0%")):
            h=np.sort(values)
            ax.step(np.r_[h[0]-.001,h],np.r_[0,np.arange(1,len(h)+1)/len(h)],where="post",color=color,label=name)
        fail=np.mean(h1<.95)
        ax.axvspan(.93,.95,color="tab:red",alpha=.07)
        ax.axvline(.95,color="black",ls="--",label="Required H=0.95")
        ax.plot([.93,.95],[fail,fail],color="tab:blue",ls=":")
        ax.scatter([.95],[fail],color="tab:blue",zorder=5)
        ax.set_title(f"Design {label}: {sum(h1>=.95)}/{len(h1)} reach H>=0.95")
        ax.set_xlabel("Highest H achievable within mismatch ceiling")
        ax.set_xlim(.93,.98);ax.set_ylim(0,1.02);ax.grid(alpha=.25);ax.legend(loc="lower right",fontsize=8)
        other_failures={}
        for r in group:
            for reason in r["failure_reasons"]:
                if reason!="hybridization": other_failures[reason]=other_failures.get(reason,0)+1
        stats[label]={"interior_old_pass":int(sum(h0>=.95)),"interior_new_pass":int(sum(h1>=.95)),
            "interior_total":len(group),"cdf_at_0.95":float(fail),
            "H_quantiles_1pct":dict(zip(("min","p05","median","p95","max"),np.quantile(h1,[0,.05,.5,.95,1]).tolist())),
            "mean_H_gain":float(np.mean(h1-h0)),"minimum_H_gain":float(min(h1-h0)),
            "nonmonotone_pairs":int(sum(h1<h0-1e-6)),"other_failures":other_failures,
            "counts":{kind:{"passes":sum(r["passes"] for r in new if r["design"]==label and r["kind"]==kind),
                           "total":sum(r["design"]==label and r["kind"]==kind for r in new)}
                      for kind in ("interior","corner","one_at_a_time")}}
    axes[0].set_ylabel("Empirical CDF: fraction with best H <= horizontal value")
    fig.suptitle("Independent +/-2% capacitance errors: both EJs recalibrated at 4.4 GHz\nCDF uses 258 interior samples per design; corners and isolated tests excluded")
    fig.savefig(OUT/"matching_cdf.png",dpi=180)
    fig.savefig(OUT/"matching_cdf.pdf")
    (OUT/"cdf_statistics.json").write_text(json.dumps(stats,indent=2))
    lines=["# One-percent matching ceiling: robustness CDF and rail asymmetry","",
        "Same 300 capacitance-error vectors per design as the 0.9% study: 258 Latin-hypercube interior samples, 32 corners and 10 isolated endpoints. Each physical capacitor independently lies within +/-2% of nominal. Only the matching ceiling changes to <=1%. Both EJs are recalibrated, allowing non-maximal hybridization, with dressed carrier fixed at 4.4 GHz. The bare resonator stays 6 GHz / 50 ohm equivalent-mode impedance. All previous energy, dispersive, overlap and 95-105 MHz gap requirements remain.","",
        "## Interpreting the CDF","",
        r"For each fabricated circuit, define $H_*(r)=\max H$ over the searched EJ settings with $R=\max_i|\Delta\chi/\chi_i|\le r$ at the fixed carrier. The plotted empirical CDF is $F_r(h)=\#\{H_*(r)\le h\}/N$. At h=0.95, the CDF gives approximately the fraction unable to retain H>=0.95; one minus this value is the fraction that can. Other constraints are checked separately. Only the 258 interior samples enter the CDF, so deliberate corners are not mixed into a putative fabrication distribution. Sampling assumes independent uniform errors for coverage; actual fabrication yield requires a measured error distribution.","",
        "Increasing the permitted mismatch enlarges the allowed EJ set, so the best achievable H cannot decrease. This does not imply a globally monotonic relationship between H and actual mismatch along an arbitrary EJ sweep. In these nominal designs, moving toward the hybridization maximum from the matched point makes the mismatch worse (larger), rather than improving matching.","",
        "![CDF](matching_cdf.png)","",
        "| Design | Interior passes at 0.9% | Interior passes at 1% | 1% corner passes | 1% isolated endpoint passes | Mean gain in H |",
        "|---|---:|---:|---:|---:|---:|"]
    for label,s in stats.items():
        c=s["counts"]
        lines.append(f"| {label} | {s['interior_old_pass']}/258 | {s['interior_new_pass']}/258 | {c['corner']['passes']}/32 | {c['one_at_a_time']['passes']}/10 | {s['mean_H_gain']:.6f} |")
    lines += ["","## Why C1 and C2 have different sensitivities","",
        "There is no fundamental sensitivity associated with the rail number. The nominal circuit is asymmetric: A has shunts 118 and 78 fF, loaded EC/h about 153 and 227 MHz, unequal readout capacitors, and different EJs. B is likewise asymmetric. A shunt change alters both the inverse-capacitance charging term and the capacitive charge couplings, as well as the anharmonicity and matrix elements. Restoring the local transition frequency through EJ tuning therefore does not restore the original Hamiltonian or dispersive pulls.","",
        "Controlled diagnostics below first restore the two nominal local transition frequencies using EJ recalibration, then solve the 1% matching constraint at the carrier. The columns distinguish frequency restoration from the additional detuning required to regain matching. The rail-label exchange check swaps both shunts, both readout capacitors, both EJs and the reference states, then recalibrates again. This tests whether the effect follows the physical circuit or an arbitrary label/search direction.","",
        "| Design | Perturbation | EC1, EC2 (MHz) | g1, g2 after local-frequency restoration (MHz) | Matched local f1-f2 (MHz) | Matched H | H change after complete label exchange |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in asym:
        p,q=r["matched"],r["local_frequencies_restored"]
        desc=r["parameter"]+(f" {100*r['error']:+g}%" if r["error"] else "")
        lines.append(f"| {r['design']} | {desc} | {p['loaded_ec1_mhz']:.3f}, {p['loaded_ec2_mhz']:.3f} | {q['g1r_over_2pi_mhz']:.4f}, {q['g2r_over_2pi_mhz']:.4f} | {r['matched_bare_detuning_mhz']:.5f} | {p['hybridization']:.7f} | {r['label_swap_H_difference']:.3g} |")
    maxswap=max(abs(r["label_swap_H_difference"]) for r in asym)
    lines += ["","### Relative-percent versus absolute-fF tolerance","",
        "A major part of the apparent C1/C2 asymmetry is the tolerance convention: the same +/-2% means a larger absolute fF excursion for the larger shunt. Central finite-difference sensitivities below use the fully EJ-recalibrated endpoints, not fixed-EJ derivatives.","",
        "| Design | C1, C2 nominal (fF) | +/-2% absolute excursions (fF) | dH/dC1 (per fF) | dH/dC2 (per fF) | Magnitude ratio |",
        "|---|---:|---:|---:|---:|---:|"]
    for label in ("A","B"):
        nominal=next(r for r in asym if r["design"]==label and r["parameter"]=="nominal")["matched"]
        caps=[nominal["device"][key] for key in ("C1_shunt_fF","C2_shunt_fF")]
        derivatives=[]
        for key,c in zip(("C1_shunt_fF","C2_shunt_fF"),caps):
            endpoints=sorted([r for r in asym if r["design"]==label and r["parameter"]==key],key=lambda r:r["error"])
            derivatives.append((endpoints[1]["matched"]["hybridization"]-endpoints[0]["matched"]["hybridization"])/(.04*c))
        lines.append(f"| {label} | {caps[0]:g}, {caps[1]:g} | {caps[0]*.02:.3f}, {caps[1]*.02:.3f} | {derivatives[0]:.7f} | {derivatives[1]:.7f} | {abs(derivatives[0]/derivatives[1]):.4f} |")
    lines += ["","Thus C1 is substantially more sensitive under equal percentage tolerances, but the two shunts have nearly equal sensitivity magnitudes per fF in these circuits. They act in opposite directions. The remaining difference includes nonlinear response and the asymmetric charging/coupling parameters; this is not an intrinsic weakness of transmon 1."]
    lines += ["",f"Maximum H difference after full rail-label exchange: {maxswap:.3g}. The sensitivity follows the physical rail parameters, not the name C1. Its magnitude is a combined sensitivity of charging energy, coupling and matching-required detuning; the individual charging energies alone do not establish the ranking.","",
        "In the ideal two-level pair H=1-|delta|/sqrt(delta^2+4J^2), where delta is the effective rail detuning and J the exchange coupling. This explains why a larger matching-required detuning costs H. The tabulated local f1-f2 is a useful diagnostic but is not exactly the effective detuning after resonator dressing; the actual H is always calculated from full-circuit transition strengths.","",
        f"All paired samples were checked for nondecreasing H when relaxing the matching ceiling: {sum(s['nonmonotone_pairs'] for s in stats.values())} violations above 1e-6. Larger-basis checks at 14 selected points changed {sum(c['small_basis_pass']!=c['large_basis_pass'] for c in checks)} classifications. See the accompanying report.md for the full sensitivity analysis. Finite-drive and noise performance remain outside this calculation."]
    (OUT/"overview.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(stats,indent=2))

if __name__=="__main__": main()
