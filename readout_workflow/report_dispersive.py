"""Produce figures and a readable summary from the saved dispersive study."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent/"data"/"dispersive_study"
    baseline = json.loads((root/"scan.json").read_text())
    baseline = [c for c in baseline if c["fr_ghz"] == 5.99 and c["c12_fF"] == 2.3 and c["asymmetry"] == 1.]
    refined = json.loads((root/"refined.json").read_text())
    targets = json.loads((root/"at4p4.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout="constrained")
    for col, cases in enumerate((baseline, refined)):
        for case in cases:
            p = case["points"]
            f = np.array([q["operating_frequency_ghz"] for q in p])
            ratio = np.array([max(q["ratio_m"],q["abs_delta_chi_over_chi_p"]) for q in p])
            lam = np.array([q["lambda_max"] for q in p])
            d = case["device"]
            z = np.sqrt(d["Lr_nH"]*1e-9/(d["Cr_fF"]*1e-15))
            label = f"Cg={case['cap_fF']:g} fF"
            if col:
                label += f", C12={case['c12_fF']:g} fF, fr={case['fr_ghz']:g} GHz, Z={z:g} ohm"
            axes[0,col].plot(f,ratio,label=label)
            axes[1,col].plot(f,lam,label=label)
        axes[0,col].set_yscale("log")
        axes[0,col].set_ylim(1e-5,1)
        axes[0,col].axhline(.01,color="black",ls="--",lw=1)
        axes[1,col].axhline(.1,color="black",ls="--",lw=1)
        axes[1,col].set_ylim(0,.4)
        axes[0,col].legend(fontsize=7,loc="upper left")
        for ax in axes[:,col]:
            ax.axvline(4.4,color="gray",ls=":",lw=1)
            ax.set_xlim(3.3,5.3)
            ax.grid(alpha=.25)
        axes[1,col].set_xlabel("Dressed operating frequency (GHz)")
    axes[0,0].set_title("Change readout coupling only: C12=2.3 fF, fr=5.99 GHz")
    axes[0,1].set_title("Retune C12 and resonator parameters")
    axes[0,0].set_ylabel(r"$\max_i|\Delta\chi/\chi_i|$")
    axes[1,0].set_ylabel(r"$\lambda_{\max}=\max_{ij}|g_{ij}/\Delta_{ij}|$")
    fig.suptitle("Low-power matching: ratio < 0.01, lambda <= 0.1, branch overlap >= 0.95",fontsize=12)
    fig.savefig(root/"comparison.png",dpi=180)
    fig.savefig(root/"comparison.pdf")

    lines = ["# Readout-coupling and dispersive study", "",
             "Notebook inputs were not changed. Baseline shunts: 64 and 54 fF; equal readout couplings. All impedances are equivalent-mode impedances, not CPW characteristic impedances.", "",
             "The screen uses max |gij/Deltaij| <= 0.1 for upward isolated-pair transitions from g,m,p, branch assignment overlap >= 0.95, and hybridization >= 0.95. Both chi normalizations must be below 0.01. This is a low-power static screen, not a finite-photon measurement validation.", "",
             "## At exactly 4.4 GHz", "",
             "| Cg (fF) | C12 (fF) | Bare fr (GHz) | Zmode (ohm) | EJ1/h (GHz) | EJ2/h (GHz) | R+ | lambda max | chi+ (MHz) | Logical splitting (MHz) | Larger-basis R+ |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in targets:
        lines.append(f"| {p['cap']:g} | {p['c12']:g} | {p['fr']:g} | {p['z']:g} | {p['ej1_ghz']:.6f} | {p['ej2_ghz']:.6f} | {p['abs_delta_chi_over_chi_p']:.6g} | {p['lambda_max']:.4f} | {p['chi_p_ghz']*1e3:.4f} | {p['splitting_ghz']*1e3:.3f} | {p['larger_basis_ratio']:.6g} |")
    lines += ["", "## Sampled contiguous qualifying intervals", "",
              "Endpoints are sampled estimates, not analytically certified boundaries; refined grids have 201 EJ1 values over 6–16 GHz.", ""]
    for c in refined:
        d=c["device"]
        z=np.sqrt(d["Lr_nH"]*1e-9/(d["Cr_fF"]*1e-15))
        for interval in c["good_intervals"]:
            good=[p for p in c["points"] if p["good"]]
            splits=[p["splitting_ghz"]*1e3 for p in good]
            lines.append(f"- Cg={c['cap_fF']:g} fF, C12={c['c12_fF']:g} fF, fr={c['fr_ghz']:g} GHz, Zmode={z:g} ohm: {interval['lower_ghz']:.3f}–{interval['upper_ghz']:.3f} GHz ({interval['width_mhz']:.0f} MHz), both |chi| >= {interval['min_abs_chi_mhz']:.3f} MHz; logical splitting {min(splits):.1f}–{max(splits):.1f} MHz.")
    lines += ["", "## Scope", "",
              "Coarse study: equal coupling 6/10/15 fF; bare fr 5.5/5.99/6.5/7 GHz; C12 1/2.3/4 fF. Additional studies varied C12 down to 0.3 fF, coupling ratios 0.8/0.9/1.1/1.2, T2 shunt 44/64/74 fF, fr up to 7.5 GHz, and mode impedance down to 25 ohm. These are examples within the explored parameter sets, not global optima.", "",
              "Larger-basis checks use ncut=20, 9 transmon levels, 8 resonator levels and 30 eigenlevels versus production 15/6/5/20. They evaluate the same optimized physical points, without reoptimizing. Endpoint convergence was not checked individually.", "",
              "Perturbative background: [Koch et al. (2007), Section III](https://schusterlab.stanford.edu/static/pdfs/Koch2007.pdf)."]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(root/"report.md")
    print(root/"comparison.png")


if __name__ == "__main__":
    main()
