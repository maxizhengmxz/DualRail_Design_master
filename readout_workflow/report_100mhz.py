"""Render the validated 6 GHz / 100 MHz design and its operating interval."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parent/"data"/"search_100mhz"
    designs=json.loads((root/"validated.json").read_text())
    curve=json.loads((root/"fixed_device_curve.json").read_text())
    p=designs[0]
    good=[q for q in curve if q["passes"]]
    freq=np.array([q["operating_frequency_ghz"] for q in curve])
    lo,hi=min(q["operating_frequency_ghz"] for q in good),max(q["operating_frequency_ghz"] for q in good)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,layout="constrained")
    ratio=np.array([max(q["ratio_m"],q["abs_delta_chi_over_chi_p"]) for q in curve])
    axes[0,0].semilogy(freq,np.maximum(ratio,1e-12))
    axes[0,0].axhline(.01,color="black",ls="--")
    axes[0,0].set_ylabel(r"$\max_i|\Delta\chi/\chi_i|$")
    axes[0,0].set_ylim(1e-12,.1)
    axes[0,0].set_title("Reoptimized chi ratio: numerical solver residual",fontsize=10)
    axes[0,1].plot(freq,[q["splitting_ghz"]*1000 for q in curve])
    for value in (95,105):
        axes[0,1].axhline(value,color="black",ls="--")
    axes[0,1].set_ylabel("Logical splitting (MHz)")
    axes[1,0].plot(freq,[q["lambda_max"] for q in curve])
    axes[1,0].axhline(.1,color="black",ls="--")
    axes[1,0].set_ylabel(r"$\max_{ij}|g_{ij}/\Delta_{ij}|$")
    axes[1,1].plot(freq,[q["ej1_over_ec1"] for q in curve],label="T1")
    axes[1,1].plot(freq,[q["ej2_over_ec2"] for q in curve],label="T2")
    axes[1,1].axhline(50,color="black",ls="--")
    axes[1,1].set_ylabel(r"$E_J/E_C$")
    axes[1,1].legend()
    for ax in axes.flat:
        ax.axvspan(lo,hi,color="green",alpha=.1)
        ax.grid(alpha=.25)
    for ax in axes[1]:
        ax.set_xlabel("Dressed carrier frequency (GHz)")
    fig.suptitle("Fixed circuit, 6 GHz bare resonator: retune only EJ1 and EJ2 to match chi")
    fig.savefig(root/"feasible_interval.png",dpi=180)
    fig.savefig(root/"feasible_interval.pdf")

    lines=["# Feasible 6 GHz resonator / 100 MHz logical-frequency design", "",
           "The original notebook's manual inputs were not changed. Frequencies quoted as EJ/h and EC/h are in GHz unless noted. The resonator input is the bare single-mode frequency, with 50 ohm equivalent-mode impedance; it is not a CPW characteristic-impedance specification.", "",
           "## Copyable main device inputs", "", "```python", "device_parameters = {"]
    values={"EJ1_GHz":round(p["ej1_ghz"],6),"EJ2_GHz":round(p["ej2_ghz"],6),
            **{k:p["device"][k] for k in ("C1_shunt_fF","C2_shunt_fF")},
            "C12_fF":round(p["device"]["C12_fF"],6),"C1r_fF":6.,"C2r_fF":6.,
            "resonator_frequency_ghz":6.,"resonator_impedance_ohm":50.}
    lines += [f'    "{k}": {v},' for k,v in values.items()]
    lines += ["}","```","", "Use `resonator_impedance_kind = \"mode\"`.", "",
              "## Checked operating points", "",
              "| Readout caps (fF) | Shunts (fF) | C12 (fF) | Carrier (GHz) | EJ1/h, EJ2/h (GHz) | Loaded EC1/h, EC2/h (MHz) | EJ/EC ratios | Logical splitting (MHz) | Larger-basis chi ratio | Maximum dispersive ratio | Hybridization |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in designs:
        q=d["convergence"][-1]
        lines.append(f"| {d['device']['C1r_fF']:g}, {d['device']['C2r_fF']:g} | {d['device']['C1_shunt_fF']:g}, {d['device']['C2_shunt_fF']:g} | {d['device']['C12_fF']:.6f} | {q['operating_frequency_ghz']:.6f} | {d['ej1_ghz']:.6f}, {d['ej2_ghz']:.6f} | {d['loaded_ec1_mhz']:.3f}, {d['loaded_ec2_mhz']:.3f} | {d['ej1_over_ec1']:.3f}, {d['ej2_over_ec2']:.3f} | {q['splitting_ghz']*1000:.6f} | {q['abs_delta_chi_over_chi_p']:.4g} | {q['lambda_max']:.5f} | {q['hybridization']:.5f} |")
    q=p["convergence"][-1]
    lines += ["", "## Interpretation and convergence", "",
              "The operation point is slightly detuned from the exact minimum-splitting point. Chi is matched by tuning EJ1/EJ2 while C12 sets the logical splitting. Strong rail hybridization >=0.95 is retained, but exact maximal mixing is not imposed. A search that only returns minimum-splitting points will generally miss these chi-matched operating points.", "",
              f"Main design: chi+ and chi- are approximately {q['chi_p_ghz']*1e3:.6f} and {q['chi_m_ghz']*1e3:.6f} MHz. Minimum six-branch reference overlap is {q['min_assignment_overlap']:.6f}. The dressed ground-state resonator transition is {q['dressed_r_ghz']:.6f} GHz; the specified bare frequency is 6 GHz.", "",
              "Loaded EC is defined by e^2(C^-1)ii/2. The search range was EC/h=150–350 MHz. The dispersive screen includes upward isolated-pair transitions from g,m,p to higher states; |gij/Deltaij|<0.1 is a low-photon criterion. This does not certify finite-photon measurement performance.", "",
              "Numerical bases checked at identical physical parameters: ncut/transmon levels/resonator levels/eigenlevels = 15/6/5/20, 20/9/8/30, and 25/12/10/40. The tiny optimized ratio in the smallest basis is not quoted as converged precision; the largest-basis result is reported above.", "",
              "## Feasible parameter region", "",
              "At a 4.4 GHz carrier with equal 6 fF readout caps, the sampled rectangle C1_shunt=115–118 fF and C2_shunt=78–85 fF has feasible chi-matched designs when EJ1, EJ2, and C12 are jointly retuned. Loaded EC/h values in the checked grid are approximately 153–157 MHz (T1) and 207–224 MHz (T2). This is a tuned family, not an independent fabrication-tolerance box. See feasible_grid.json for individual sampled values.", "",
              "## Finite tuning intervals", "",
              f"For the main fixed circuit, retuning only EJ1/EJ2 along chi matching gives a sampled contiguous carrier interval {lo:.3f}–{hi:.3f} GHz, using a 10 MHz carrier grid. Logical splitting stays within 100 +/- 5 MHz; both EJ<25 GHz, both EJ/EC>50, max dispersive ratio<0.1, hybridization>=0.95 and branch overlap>=0.95 hold at all accepted samples. The endpoints are sampled estimates, not certified continuous boundaries. The tiny chi-ratio curve is the numerical residual of reoptimization in the 20/9/8/30 basis, not a claim of physical matching accuracy or a fixed-EJ bandwidth. Largest-basis convergence was checked at the representative points, not every curve sample.", ""]
    band=p["strong_hybridization_band"]
    lines.append(f"At fixed EJ1={p['ej1_ghz']:.9f} GHz and all component values fixed, the sampled/endpoint-checked EJ2 interval is approximately {band['ej2_lower_ghz']:.9f}–{band['ej2_upper_ghz']:.9f} GHz (width {band['width_mhz']:.3f} MHz), also retaining hybridization>=0.95. Strict inequalities exclude boundary points.")
    lines += ["", "## Reproduce a matched point", "", "```python", "from threadpoolctl import threadpool_limits", "from search_100mhz import solve_point", "with threadpool_limits(limits=1):", "    point = solve_point(115, 80, cap=6, carrier=4.4, match_chi=True)", "```", "",
              "No loss, fabrication disorder, or drive-induced transitions are simulated. The broad operating interval requires retuning EJs; it does not assert a fixed EJ setting works across all frequencies."]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(root/"report.md")
    print(root/"feasible_interval.png")


if __name__ == "__main__":
    main()
