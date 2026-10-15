"""Validate sampled feasible physical shunt pairs and report their witnesses."""
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from model import Basis, Device, ReadoutModel
from map_shunt_pairs import OUT, GRID, CHARGE_CONSTANT, classify
from search_equal_g import annotate


def save_pdf(figure,path):
    for version in range(100):
        suffix="" if version==0 else ("_updated" if version==1 else f"_updated_{version}")
        candidate=path.with_stem(path.stem+suffix)
        try:
            figure.savefig(candidate)
            return candidate
        except PermissionError:
            if version==99: raise


def main():
    records={}
    for name in ("grid.json","edge_coupling.json","refined.json"):
        path=OUT/name
        if path.exists():
            for r in json.loads(path.read_text()): records[r["c1"],r["c2"]]=r
    checked=[]
    with threadpool_limits(limits=1):
        for r in records.values():
            if not r["found"]: continue
            p=r["best"]
            m=ReadoutModel(Device(**p["device"]),Basis(20,9,8,30))
            q=annotate(m,m.evaluate(p["ej1_ghz"],p["ej2_ghz"]))
            q.update(participation1=r["c1"]*1e-15*m.cinv[0,0],
                     participation2=r["c2"]*1e-15*m.cinv[1,1])
            q["failures"]=classify(q);q["passes"]=not q["failures"]
            checked.append({"c1":r["c1"],"c2":r["c2"],"small_basis":p,"large_basis":q})
    (OUT/"validated.json").write_text(json.dumps(checked,indent=2))
    good=[r for r in checked if r["large_basis"]["passes"]]
    large={(r["c1"],r["c2"]):r["large_basis"] for r in good}
    shape=(len(GRID),len(GRID))
    maps=[np.full(shape,np.nan) for _ in range(5)]
    for i,c1 in enumerate(GRID):
        for j,c2 in enumerate(GRID):
            key=(max(c1,c2),min(c1,c2))
            p=large.get(key)
            if p:
                # Mirrored cells exchange the complete rails, including EC.
                ec1=p["loaded_ec1_mhz"] if c1>=c2 else p["loaded_ec2_mhz"]
                ec2=p["loaded_ec2_mhz"] if c1>=c2 else p["loaded_ec1_mhz"]
                for matrix,value in zip(maps,(p["hybridization"],p["operating_frequency_ghz"],p["device"]["C1r_fF"],ec1,ec2)):
                    matrix[j,i]=value
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharex=True,sharey=True,layout="constrained")
    titles=("Feasible witness: hybridization", "Operation frequency (GHz)","Equal readout cap Cg (fF)")
    for ax,matrix,title in zip(axes,maps[:3],titles):
        cmap=plt.colormaps["viridis"].copy();cmap.set_bad("#eeeeee")
        mesh=ax.pcolormesh(GRID,GRID,np.ma.masked_invalid(matrix),shading="nearest",cmap=cmap)
        fig.colorbar(mesh,ax=ax)
        ax.plot([50,116.3],[50,116.3],color="black",ls=":",lw=1)
        for r in good:
            if r["c1"] not in GRID or r["c2"] not in GRID:
                ax.scatter([r["c1"],r["c2"]],[r["c2"],r["c1"]],s=8,color="black")
        ax.set_title(title,fontsize=10);ax.set_xlabel("C1 (fF)")
    axes[0].set_ylabel("C2 (fF)")
    fig.suptitle("6 GHz bare resonator: equal readout capacitors; carrier searched from 4 to 8 GHz\nGray = no passing design found in this bounded search; black dots = refined passing pairs")
    fig.savefig(OUT/"feasible_pairs.png",dpi=180)
    feasible_pdf=save_pdf(fig,OUT/"feasible_pairs.pdf")
    plt.close(fig)

    ec_values=[r["large_basis"][key] for r in good for key in ("loaded_ec1_mhz","loaded_ec2_mhz")]
    vmin,vmax=min(ec_values),max(ec_values)
    ecfig,ecaxes=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    ecaxes[1].sharex(ecaxes[0]);ecaxes[1].sharey(ecaxes[0])
    for index,(ax,matrix) in enumerate(zip(ecaxes[:2],maps[3:]),start=1):
        cmap=plt.colormaps["viridis"].copy();cmap.set_bad("#eeeeee")
        mesh=ax.pcolormesh(GRID,GRID,np.ma.masked_invalid(matrix),shading="nearest",cmap=cmap,vmin=vmin,vmax=vmax)
        ax.plot([50,116.3],[50,116.3],color="black",ls=":",lw=1)
        for r in good:
            if r["c1"] not in GRID or r["c2"] not in GRID:
                p=r["large_basis"]
                vals=(p[f"loaded_ec{index}_mhz"],p[f"loaded_ec{3-index}_mhz"])
                ax.scatter([r["c1"],r["c2"]],[r["c2"],r["c1"]],c=vals,
                           cmap=cmap,vmin=vmin,vmax=vmax,s=24,edgecolors="black",linewidths=.5)
        ax.set_xlabel("C1 (fF)")
        ax.set_title(rf"Loaded $E_{{C{index}}}/h$ (MHz)")
    ecaxes[0].set_ylabel("C2 (fF)")
    ecfig.colorbar(mesh,ax=ecaxes[:2],label=r"Loaded $E_C/h$ (MHz)")
    cdf_ax=ecaxes[2]
    for index in (1,2):
        values=np.sort([r["large_basis"][f"loaded_ec{index}_mhz"] for r in good])
        fractions=np.arange(1,len(values)+1)/len(values)
        cdf_ax.step(np.r_[vmin-5,values,vmax+5],np.r_[0,fractions,1],where="post",
                    label=rf"Transmon {index}: $E_{{C{index}}}/h$")
    cdf_ax.set_title(rf"CDF: {len(good)} selected circuits ($C_1\geq C_2$)")
    cdf_ax.set_xlabel(r"Loaded $E_C/h$ (MHz)")
    cdf_ax.set_ylabel("Cumulative fraction")
    cdf_ax.set_xlim(vmin-5,vmax+5);cdf_ax.set_ylim(0,1.02)
    cdf_ax.grid(alpha=.25);cdf_ax.legend(fontsize=8)
    ecfig.suptitle("Total capacitance-loaded charging energies and their empirical CDFs\nMaps share a color scale; CDF counts each selected circuit once, without mirrored duplicates")
    ecfig.savefig(OUT/"loaded_charging_energies.png",dpi=180)
    ec_pdf=save_pdf(ecfig,OUT/"loaded_charging_energies.pdf")
    plt.close(ecfig)

    bounds=[]
    for c1 in GRID:
        row=sorted([r for r in good if r["c1"]==c1],key=lambda r:r["c2"])
        if not row: continue
        allrow=sorted([r for r in records.values() if r["c1"]==c1],key=lambda r:r["c2"])
        lower=max((r["c2"] for r in allrow if r["c2"]<row[0]["c2"] and not r["found"]),default=None)
        upper=min((r["c2"] for r in allrow if r["c2"]>row[-1]["c2"] and not r["found"]),default=None)
        bounds.append({"larger_shunt_fF":float(c1),"smallest_found_smaller_shunt_fF":row[0]["c2"],
            "largest_found_smaller_shunt_fF":row[-1]["c2"],"lower_no_solution_sample_fF":lower,
            "upper_no_solution_sample_fF":upper,"sampled_values_fF":[r["c2"] for r in row]})
    (OUT/"ranges.json").write_text(json.dumps(bounds,indent=2))
    witnesses=[]
    for c1 in GRID:
        row=[r for r in good if r["c1"]==c1]
        if row: witnesses.append(max(row,key=lambda r:r["large_basis"]["hybridization"]))
    lines=["# Acceptable physical shunt-capacitance pairs", "",
        "This is an existence map: at each C1,C2 pair, EJ1, EJ2, C12, and the equal readout capacitance C1r=C2r=Cg may be redesigned. It is not a fabrication tolerance map with C12 and Cg held fixed. Both transmon carrier frequencies may lie anywhere in the requested 4-8 GHz range. The bare resonator is fixed at 6 GHz and equivalent-mode impedance 50 ohm.", "",
        f"With 90% participation, EC/h [GHz] is approximately {CHARGE_CONSTANT*.9:.6f}/C_shunt[fF]. Taking the previous typical range 150-350 MHz gives shunt bounds {CHARGE_CONSTANT*.9/.35:.3f}-{CHARGE_CONSTANT*.9/.15:.3f} fF. Ninety percent sets the search bounds and approximate EC labels; it is not imposed as an exact physical constraint. The complete inverse capacitance matrix provides all final ECs and couplings. Exact participation cannot generally stay fixed while C12 and Cg are independently changed.", "",
        "Acceptance: worst of both chi normalizations <=0.01 (1e-7 comparison tolerance); numerical multilevel dispersive maximum<=0.1; logical gap 95-105 MHz (solver targets 100 MHz); EJ/h<25 GHz and EJ/EC>50 for each rail; each actual loaded EC/h=150-350 MHz; hybridization>=0.95; reference-state overlap>=0.95. Equal capacitors do not imply equal g. The chi-target branches use 0.98% to leave margin to the final 1% check. Non-maximal hybridization is permitted.", "",
        "## Found ranges", "",
        "Order the shunts as C_L>=C_S. Either physical transmon may be the larger one: swapping all rail parameters produces the mirrored half of the map. Bounds below are the minimum and maximum *sampled passing* C_S for each C_L; they do not certify every interior value or a sharp continuous boundary.", "",
        "| C_L (fF) | Approx EC_L/h at 90% (MHz) | Sampled passing C_S range (fF) | Adjacent lower / upper no-solution-found samples (fF) |",
        "|---:|---:|---:|---:|"]
    for r in bounds:
        lines.append(f"| {r['larger_shunt_fF']:.3f} | {900*CHARGE_CONSTANT/r['larger_shunt_fF']:.1f} | {r['smallest_found_smaller_shunt_fF']:.3f}-{r['largest_found_smaller_shunt_fF']:.3f} | {r['lower_no_solution_sample_fF']} / {r['upper_no_solution_sample_fF']} |")
    lines += ["",f"[Feasible-pair PDF]({feasible_pdf.name})", "", "![Feasible pairs](feasible_pairs.png)", "", "## Loaded charging-energy maps", "",
        r"Each map shows the selected solution's total capacitance-loaded $E_{Ci}/h=e^2(C^{-1})_{ii}/(2h)$ in MHz, including all physical capacitances. These values are not the approximate 90%-participation labels. Mirroring a circuit across C1=C2 exchanges EC1 and EC2. Both maps use the same color scale; outlined points show refined solutions. The third panel shows separate empirical CDFs of EC1/h and EC2/h over the unique selected circuits, including refined pairs, with C1>=C2. Each circuit receives equal weight; mirrored copies are excluded. Pooling both mirror orientations would make the two CDFs identical by label symmetry. These CDFs describe the sampled solutions, not fabrication probabilities or a continuous-area-weighted distribution.", "",
        f"[Loaded charging-energy PDF]({ec_pdf.name})", "", "![Loaded charging energies](loaded_charging_energies.png)", "", "## Reproducible example circuits", "",
        "Every row is a separate redesigned circuit. These examples were recomputed at identical components in the larger 20/9/8/30 numerical basis. Carrier, C12 and Cg shown are witnesses, not claimed global optima.", "",
        "| C_L,C_S (fF) | EJ_L,EJ_S (GHz) | C12 (fF) | Equal Cg (fF) | Carrier (GHz) | EC_L,EC_S (MHz) | EJ/EC | H | Chi ratio | Dispersive maximum |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|"]
    for r in witnesses:
        p=r["large_basis"]
        lines.append(f"| {r['c1']:.3f},{r['c2']:.3f} | {p['ej1_ghz']:.6f},{p['ej2_ghz']:.6f} | {p['device']['C12_fF']:.6f} | {p['device']['C1r_fF']:.6f} | {p['operating_frequency_ghz']:.6f} | {p['loaded_ec1_mhz']:.2f},{p['loaded_ec2_mhz']:.2f} | {p['ej1_over_ec1']:.2f},{p['ej2_over_ec2']:.2f} | {p['hybridization']:.6f} | {max(p['ratio_m'],p['abs_delta_chi_over_chi_p']):.6f} | {p['lambda_max']:.6f} |")
    participation=[p["large_basis"][key] for p in good for key in ("participation1","participation2")]
    carriers=[p["large_basis"]["operating_frequency_ghz"] for p in good]
    changed=[r for r in checked if not r["large_basis"]["passes"]]
    lines += ["", "## Scope and numerical checks", "",
        f"{len(records)} distinct ordered shunt pairs were searched; {len(checked)} produced small-basis passing witnesses; {len(good)} retained all checks in the larger basis. Larger-basis rejections: {len(changed)}. Actual participation among passing witnesses spans {100*min(participation):.2f}-{100*max(participation):.2f}%. Selected witness carriers span {min(carriers):.3f}-{max(carriers):.3f} GHz, although 4-8 GHz was searched.", "",
        "A no-solution-found result is not proof of physical impossibility. The search samples carrier candidates on a 0.5 GHz grid plus 4.4 GHz and energy-window-dependent candidates. It samples readout caps estimated for dispersive ratios near 0.05, 0.085, and 0.099, with near-cutoff adjustment, and solves local nonlinear branches for the EJs and C12. Equal readout caps are bounded between 0.3 and 18 fF; C12 between 0.01 and 10 fF; EJs between 2 and 24.999 GHz. Narrow or disconnected unsampled solutions may remain. Original grid search uses 5 fF shunt steps; refined points are recorded separately.", "",
        "No minimum absolute dispersive pull, measurement speed or drive-photon requirement was imposed. The passing witnesses do not establish driven-readout fidelity or fabrication robustness. The specified 50 ohm impedance is the equivalent LC-mode impedance, not automatically the CPW characteristic impedance.", "",
        "Files: grid.json, edge_coupling.json and refined.json preserve all attempted points; validated.json contains larger-basis checks; ranges.json contains sampled C2 values and neighboring failed-search samples.", "",
        "Reproduce using `map_shunt_pairs.py`, optionally `--edge-coupling` and `--refine`, then `report_shunt_pairs.py`."]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"searched_pairs":len(records),"passing_pairs":len(good),
                      "feasible_pdf":str(feasible_pdf),"charging_energy_pdf":str(ec_pdf)},indent=2))


if __name__=="__main__": main()
