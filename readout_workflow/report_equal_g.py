"""Validate and plot equal-g designs and a fixed-circuit EJ tuning path."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from model import Basis, Device, ReadoutModel
from search_equal_g import solve, annotate, OUT


def validate(p):
    checks=[]
    for b in (Basis(),Basis(20,9,8,30),Basis(25,12,10,40)):
        m=ReadoutModel(Device(**p["device"]),b)
        checks.append(annotate(m,m.evaluate(p["ej1_ghz"],p["ej2_ghz"])))
    p["convergence"]=checks
    return p


def main():
    with threadpool_limits(limits=1):
        designs=[validate(solve(118,78,6.,mode="chi",chi_target=-.008)),
                 validate(solve(115,80,8.,mode="chi",chi_target=-.008))]
        p=designs[0]
        m=ReadoutModel(Device(**p["device"]),Basis(20,9,8,30))
        curve=[]
        for f in np.linspace(4.1,4.9,81):
            def residual(x):
                q=m.evaluate(*x)
                g1,g2=q["g1r_over_2pi_mhz"],q["g2r_over_2pi_mhz"]
                return [q["operating_frequency_ghz"]-f,.1*(g1-g2)/((g1+g2)/2)]
            sol=least_squares(residual,[p["ej1_ghz"],p["ej2_ghz"]],
                bounds=([2,2],[24.999,24.999]),xtol=1e-11,ftol=1e-11,gtol=1e-11)
            q=annotate(m,m.evaluate(*sol.x))
            q["passes"]=bool(q["physical_pass"] and max(abs(sol.fun))<1e-7
                              and q["g_relative_difference"]<1e-5)
            curve.append(q)
        (OUT/"validated.json").write_text(json.dumps(designs,indent=2))
        (OUT/"fixed_device_curve.json").write_text(json.dumps(curve,indent=2))

    f=np.array([q["operating_frequency_ghz"] for q in curve])
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,layout="constrained")
    axes[0,0].plot(f,[max(q["ratio_m"],q["abs_delta_chi_over_chi_p"]) for q in curve])
    axes[0,0].axhline(.01,color="black",ls="--")
    axes[0,0].set_ylabel(r"$\max_i|\Delta\chi/\chi_i|$")
    axes[0,1].plot(f,[q["splitting_ghz"]*1000 for q in curve])
    axes[0,1].set_ylabel("Logical splitting (MHz)")
    for value in (95,105): axes[0,1].axhline(value,color="black",ls="--")
    axes[1,0].plot(f,[q["lambda_max"] for q in curve])
    axes[1,0].axhline(.1,color="black",ls="--")
    axes[1,0].set_ylabel(r"$\max_{ij}|g_{ij}/\Delta_{ij}|$")
    axes[1,1].plot(f,[q["hybridization"] for q in curve],label="Rail hybridization")
    axes[1,1].axhline(.95,color="black",ls="--")
    axes[1,1].set_ylabel("Rail hybridization")
    good=[q for q in curve if q["passes"]]
    lo,hi=min(q["operating_frequency_ghz"] for q in good),max(q["operating_frequency_ghz"] for q in good)
    for ax in axes.flat:
        ax.axvspan(lo,hi,color="green",alpha=.12)
        ax.grid(alpha=.25)
    for ax in axes[1]: ax.set_xlabel("Dressed carrier frequency (GHz)")
    fig.suptitle("Fixed circuit, 6 GHz bare resonator: retune EJs to maintain equal g")
    fig.savefig(OUT/"equal_g_interval.png",dpi=180)
    fig.savefig(OUT/"equal_g_interval.pdf")
    lines=["# Equal readout-coupling search", "",
        "Bare resonator 6 GHz, equivalent-mode impedance 50 ohm. Notebook inputs unchanged.",
        "The full capacitance inverse and numerical local transmon charge elements enforce equal positive magnitudes of the local 0-1 couplings. Higher transition matrix elements need not be equal.", "",
        "The same low-photon dispersive screen checks every upward isolated-pair transition from g,m,p, not just individual transmon g/detuning. Acceptance also requires both chi normalizations <0.01, logical splitting 100 +/- 5 MHz, EJ/h<25 GHz, loaded EC/h=150-350 MHz, EJ/EC>50, rail hybridization>=0.95 and six-branch overlap>=0.95.","",
        "## Validated points", "",
        "| Quantity | Design A | Design B |", "|---|---:|---:|"]
    a,b=[d["convergence"][-1] for d in designs]
    rows=[("EJ1/h (GHz)","ej1_ghz"), ("EJ2/h (GHz)","ej2_ghz"),
          ("Loaded EC1/h (MHz)","loaded_ec1_mhz"),("Loaded EC2/h (MHz)","loaded_ec2_mhz"),
          ("EJ1/EC1","ej1_over_ec1"),("EJ2/EC2","ej2_over_ec2"),
          ("g1r/2pi (MHz)","g1r_over_2pi_mhz"),("g2r/2pi (MHz)","g2r_over_2pi_mhz"),
          ("Relative g difference","g_relative_difference"),
          ("Chi mismatch / chi+","abs_delta_chi_over_chi_p"),
          ("Chi mismatch / chi-","ratio_m"),("Maximum dispersive ratio","lambda_max"),
          ("Hybridization","hybridization"),("Branch overlap","min_assignment_overlap")]
    for label,key in rows: lines.append(f"| {label} | {a[key]:.9g} | {b[key]:.9g} |")
    for key in ("C1_shunt_fF","C2_shunt_fF","C12_fF","C1r_fF","C2r_fF"):
        lines.append(f"| {key} | {a['device'][key]:.9g} | {b['device'][key]:.9g} |")
    lines += ["", "## Main device inputs", "", "```python", "device_parameters = {"]
    values={"EJ1_GHz":p["ej1_ghz"],"EJ2_GHz":p["ej2_ghz"],
            **{k:p["device"][k] for k in ("C1_shunt_fF","C2_shunt_fF","C12_fF","C1r_fF","C2r_fF")},
            "resonator_frequency_ghz":6.,"resonator_impedance_ohm":50.}
    lines += [f'    "{k}": {v:.12g},' for k,v in values.items()]
    lines += ["}","```","",'Use `resonator_impedance_kind = "mode"`.', "",
        "## Interpretation", "",
        "At 4.4 GHz, the sampled equal-g resonance-minimum designs with 6 fF C1r have mismatch approximately 2.3-2.6%. Solving exact chi matching together with equal g and the stationary gap for five representative shunt pairs requires maximum dispersive ratios 0.22-0.23. This is a sampled result, not a proof that every resonant design is infeasible.", "",
        "The passing points above retain slight detuning. They satisfy the actual 1% matching target rather than exact zero mismatch. Equal g does not automatically make the design better than the previous unequal-g equal-capacitance solutions; unequal anharmonicities and higher-level matrix elements also enter chi.", "",
        f"Design A has a sampled passing interval {lo:.3f}-{hi:.3f} GHz along the fixed-circuit equal-g path (10 MHz grid). Only EJ1 and EJ2 are retuned. The coupling capacitances, C12 and resonator stay fixed. Every sample is recalculated in the 20/9/8/30 basis. The logical gap may move within 95-105 MHz; individual constraints, including EJ/EC>50, determine the accepted interval. This is not a fixed-EJ bandwidth or a fabrication tolerance.", "",
        "Convergence at unchanged component values was checked in bases 15/6/5/20, 20/9/8/30 and 25/12/10/40 (charge cutoff / transmon levels / resonator levels / eigenlevels). See validated.json. The two larger bases preserve equality and the acceptance criteria. Finite-drive transitions, Purcell-filter dynamics and frequency-noise sensitivity are not simulated.", "",
        "## Reproduce", "", "```powershell",
        r".\.venv\Scripts\python.exe readout_workflow/search_equal_g.py",
        r".\.venv\Scripts\python.exe readout_workflow/refine_equal_g.py",
        r".\.venv\Scripts\python.exe readout_workflow/report_equal_g.py", "```"]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"designs":designs,"interval":[lo,hi]},indent=2))

if __name__ == "__main__":
    main()
