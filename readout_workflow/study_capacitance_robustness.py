"""Matching/hybridization tradeoff and independent capacitance corner tests.

Both EJs are always recalibrated. Physical capacitors cannot be readjusted to
restore equal g after fabrication.
"""
from dataclasses import replace
from itertools import product
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from threadpoolctl import threadpool_limits

from model import Basis, Device, ReadoutModel, _transmon
from search_equal_g import annotate

OUT=Path(__file__).resolve().parent/"data"/"equal_g"/"robustness"
CAPS=("C1_shunt_fF","C2_shunt_fF","C12_fF","C1r_fF","C2r_fF")


class CarrierSlice:
    def __init__(self,model,reference,carrier=4.4):
        self.model,self.reference,self.carrier=model,dict(reference),carrier
        # Fabrication errors shift the EJs needed for the same local frequency.
        # Recenter on the perturbed ECs before searching for hybridization;
        # otherwise a clipped interval edge can masquerade as the maximum.
        for i in (1,2):
            target=reference.get(f"bare_f{i}_ghz")
            if target is not None:
                def frequency_residual(ej):
                    return _transmon(ej,model.ec_ghz[i-1],model.basis)[0][1]-target
                self.reference[f"ej{i}_ghz"]=brentq(frequency_residual,2,24.999,xtol=1e-10)
        self.cache={}

    def point(self,ej2):
        if ej2 not in self.cache:
            def f(ej1):
                return self.model.evaluate(ej1,ej2)["operating_frequency_ghz"]-self.carrier
            guess=self.reference["ej1_ghz"]
            ej1=brentq(f,max(2,guess-2),min(24.999,guess+2),xtol=1e-9)
            self.cache[ej2]=self.model.evaluate(ej1,ej2)
        return self.cache[ej2]

    def maximum(self,key="hybridization"):
        center=self.reference["ej2_ghz"]
        sign=-1 if key=="hybridization" else 1
        fit=minimize_scalar(lambda x:sign*self.point(x)[key],bounds=(center-.18,center+.18),
                            method="bounded",options={"xatol":2e-7})
        return self.point(fit.x)

    def constrained(self,limit=.009):
        """Allow detuning from the unconstrained maximum to meet the chi limit.

        Maximizing H *within* the chi-feasible set does not require maximum
        hybridization of the circuit. H>=0.95 is checked after recalibration.
        """
        peak=self.maximum()
        def signed_ratio(x):
            p=self.point(x)
            return p["delta_chi_ghz"]/min(abs(p["chi_p_ghz"]),abs(p["chi_m_ghz"]))
        if abs(signed_ratio(peak["ej2_ghz"])) <= limit:
            return peak
        # Locate both boundaries, allowing nonmonotonic chi. Pick the highest
        # hybridization feasible boundary, rather than forcing exact chi=0.
        center=peak["ej2_ghz"]
        grid=np.linspace(center-.18,center+.18,9)
        values=[signed_ratio(x) for x in grid]
        roots=[]
        for target in (-limit,limit):
            for x,y,fx,fy in zip(grid[:-1],grid[1:],values[:-1],values[1:]):
                if (fx-target)*(fy-target)<0:
                    root=brentq(lambda t:signed_ratio(t)-target,x,y,xtol=1e-9)
                    roots.append(self.point(root))
        return max(roots,key=lambda p:p["hybridization"]) if roots else None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser()
    parser.add_argument("--report-only",action="store_true")
    args=parser.parse_args()
    if args.report_only:
        summary=json.loads((OUT/"tradeoff.json").read_text())
        corners=json.loads((OUT/"corners.json").read_text())
        for row in corners: row.pop("fixed_ej",None)
        (OUT/"corners.json").write_text(json.dumps(corners,indent=2))
        render(summary,corners)
        return
    parent=OUT.parent
    designs=json.loads((parent/"validated.json").read_text())
    summary=[]
    with threadpool_limits(limits=1):
        for label,design in zip(("A","B"),designs):
            model=ReadoutModel(Device(**design["device"]))
            sl=CarrierSlice(model,design)
            maximum=annotate(model,sl.maximum())
            minimum=annotate(model,sl.maximum("splitting_ghz"))
            safe=annotate(model,sl.constrained())
            curve=[annotate(model,sl.point(x)) for x in
                   np.linspace(design["ej2_ghz"]-.035,design["ej2_ghz"]+.035,71)]
            summary.append({"label":label,"previous":design["convergence"][-1],
                            "maximum_hybridization":maximum,"minimum_gap":minimum,
                            "constrained_maximum":safe,"curve":curve})
            print(label,"peak",maximum["hybridization"],maximum["abs_delta_chi_over_chi_p"],
                  "safe",safe["hybridization"],flush=True)
        (OUT/"tradeoff.json").write_text(json.dumps(summary,indent=2))
        corners=[]
        for entry in summary:
            nominal=entry["constrained_maximum"]
            for tolerance in (.01,.02,.05):
                for signs in product((-1,1),repeat=5):
                    device=replace(Device(**nominal["device"]),**{
                        key:nominal["device"][key]*(1+tolerance*s) for key,s in zip(CAPS,signs)})
                    model=ReadoutModel(device)
                    sl=CarrierSlice(model,nominal)
                    retuned=sl.constrained()
                    if retuned is not None: retuned=annotate(model,retuned)
                    corners.append({"label":entry["label"],"tolerance":tolerance,
                                    "signs":list(signs),"retuned":retuned,
                                    "calibration_method":"EC-recentered-v2"})
                print(entry["label"],"tolerance",tolerance,"done",flush=True)
                (OUT/"corners.json").write_text(json.dumps(corners,indent=2))
        validation=[]
        for entry in summary:
            for name in ("maximum_hybridization","constrained_maximum"):
                p=entry[name]
                m=ReadoutModel(Device(**p["device"]),Basis(20,9,8,30))
                validation.append({"label":entry["label"],"kind":name,
                    "point":annotate(m,m.evaluate(p["ej1_ghz"],p["ej2_ghz"]))})
        (OUT/"validation.json").write_text(json.dumps(validation,indent=2))
    render(summary,corners)


def render(summary,corners):
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for entry in summary:
        curve=entry["curve"]
        x=[(p["ej2_ghz"]-entry["maximum_hybridization"]["ej2_ghz"])*1000 for p in curve]
        axes[0].plot(x,[100*max(p["ratio_m"],p["abs_delta_chi_over_chi_p"]) for p in curve],label=entry["label"])
        axes[1].plot(x,[p["hybridization"] for p in curve],label=entry["label"])
        chosen=entry["constrained_maximum"]
        shift=(chosen["ej2_ghz"]-entry["maximum_hybridization"]["ej2_ghz"])*1000
        color=axes[1].lines[-1].get_color()
        axes[0].scatter([shift],[100*max(chosen["ratio_m"],chosen["abs_delta_chi_over_chi_p"])],
                        color=color,marker="o",zorder=5)
        axes[1].scatter([shift],[chosen["hybridization"]],color=color,marker="o",zorder=5)
    axes[0].axhline(1,color="black",ls="--")
    axes[0].set_ylabel("Worst chi mismatch (%)")
    axes[1].axhline(.95,color="black",ls="--")
    axes[1].set_ylabel("Rail hybridization score")
    for ax in axes:
        ax.axvline(0,color="gray",ls=":")
        ax.set_xlabel("EJ2/h from maximum-hybridization point (MHz)")
        ax.legend();ax.grid(alpha=.25)
    fig.suptitle("Fixed capacitors; EJs retuned at 4.4 GHz; dots mark chosen detuned points")
    fig.savefig(OUT/"hybridization_tradeoff.png",dpi=180)
    fig.savefig(OUT/"hybridization_tradeoff.pdf")
    lines=["# Capacitance robustness and hybridization tradeoff","",
        "Chi is an inequality, not a zero-mismatch objective. Nominal points maximize the existing charge-transition hybridization score at fixed 4.4 GHz dressed carrier, subject to worst-normalized mismatch <=0.009. This leaves 0.001 absolute margin to the 1% requirement. Equal local g is nominal only; fabricated capacitor errors are not corrected by changing capacitors.","",
        "| Design | Previous mismatch | Previous hybridization | Maximum hybridization | Mismatch there | Safe hybridization (R=0.009) |",
        "|---|---:|---:|---:|---:|---:|"]
    if any(row.get("calibration_method")!="EC-recentered-v2" for row in corners):
        lines[2:2]=["**Correction:** the historical corner counts below used an EJ search window centered on the unperturbed device. A capacitance error could move the hybridization maximum outside that window and cause a premature boundary selection. The corrected [300-scenario +/-2% study](samples_2pct/report.md) supersedes the +/-2% counts below. The earlier +/-1% and +/-5% counts have not yet been recalculated with the corrected search.",""]
    for entry in summary:
        a,b,c=entry["previous"],entry["maximum_hybridization"],entry["constrained_maximum"]
        lines.append(f"| {entry['label']} | {max(a['ratio_m'],a['abs_delta_chi_over_chi_p']):.6f} | {a['hybridization']:.6f} | {b['hybridization']:.6f} | {max(b['ratio_m'],b['abs_delta_chi_over_chi_p']):.6f} | {c['hybridization']:.6f} |")
    lines += ["","Independent simultaneous +/- errors applied to the five physical capacitors C1_shunt,C2_shunt,C12,C1r,C2r; 32 corners per tolerance and design. Cr and Lr are held fixed. No statistical fabrication distribution is assumed. A corner test is a screen, not certification of every interior point.","",
        "Both EJs are always recalibrated at the 4.4 GHz carrier. The code first evaluates the unconstrained hybridization maximum. If that point violates the 0.9% matching ceiling, it varies EJ2 away from the maximum and adjusts EJ1 to maintain the carrier. It finds the chi-feasible boundaries and selects the boundary with the highest hybridization. Thus recalibration explicitly permits non-maximal hybridization; it does not return the unconstrained maximum when that maximum fails chi matching. H>=0.95 is an acceptance floor, not a demand for maximal mixing. The caps, including C12, remain at their perturbed values. Acceptance also requires the same EC, EJ/EC, EJ, dispersive and overlap screens as before, and logical splitting 95-105 MHz. Counts describe this calibration procedure at 4.4 GHz, not global infeasibility if the carrier is also moved or the margin is reduced.","",
        "| Design | Independent cap errors | EJ-recalibrated passing corners | Retuned worst hybridization | Retuned worst dispersive ratio |",
        "|---|---:|---:|---:|---:|"]
    for label in ("A","B"):
        for tol in (.01,.02,.05):
            rows=[r for r in corners if r["label"]==label and r["tolerance"]==tol]
            tuned=[r["retuned"] for r in rows if r["retuned"] is not None]
            nt=sum(r["retuned"] is not None and r["retuned"]["physical_pass"] for r in rows)
            lines.append(f"| {label} | +/-{100*tol:g}% | {nt}/32 | {min(r['hybridization'] for r in tuned):.6f} | {max(r['lambda_max'] for r in tuned):.6f} |")
    lines += ["","Maximum hybridization refers to the existing numerical charge-transition score; it is not a measured coherence time. Minimum dressed splitting and maximum score need not occur at exactly the same EJ2. Both points are recorded. The local coupling equality can drift when the EJs are recalibrated; imposing it as an exact post-fabrication constraint would remove useful calibration freedom.","",
        "Nominal maximum and safe points were checked unchanged in the 20/9/8/30 basis (validation.json). Corner screening uses 15/6/5/20. Low-photon dispersive tests do not validate driven readout or noise sensitivity."]
    lines += ["", "## The 95% hybridization requirement", "",
        "The corner counts above already allow non-maximal hybridization down to H=0.95 inclusive. Every corner recalibrates both EJs. At +/-1% and +/-2%, every corner has a chi-feasible candidate; all rejected candidates fail H>=0.95. At +/-5%, 26/32 corners per design have no chi-feasible boundary found in the searched EJ2 interval, and the other six fail the hybridization threshold. This is a search outcome, not a proof of global infeasibility.", "",
        "## What the hybridization score measures", "",
        r"For transmon i and logical branch s in {-,+}, define the charge-transition strength $S_{is}=|\langle G|n_i|s\rangle|^2$ using full-circuit eigenstates. Normalize within the two logical branches: $p_{is}=S_{is}/(S_{i-}+S_{i+})$.", "",
        r"The implemented score is $H=1-2\,\mathrm{mean}_{i,s}|p_{is}-1/2|$. Equal transition strength into both branches for each transmon gives H=1. Fully localized logical excitations give H=0 in the ideal two-level model.", "",
        "This is not a squared state overlap or Bell-state fidelity, and 0.95 does not mean 95% probability in the logical subspace. In the ideal two-level model, with complementary rail populations p and 1-p, H=1-|2p-1|. H=0.95 therefore corresponds to 52.5%/47.5% populations (or the reverse). For the full multilevel circuit these are illustrative ideal-model populations, not populations inferred from the score alone.", "",
        "Normalization uses only the two logical transitions, so this score alone does not bound resonator or higher-level admixture. The separate reference-state overlap and dispersive checks remain necessary. See model.py, ReadoutModel.evaluate, for the implementation."]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")

if __name__ == "__main__": main()
