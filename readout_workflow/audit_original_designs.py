"""Distinguish original tdt presets from later optimized A/B designs."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from model import Basis, Device, ReadoutModel, _transmon
from reference_devices import original_preset
from search_equal_g import annotate
from map_shunt_pairs import search_pair, classify, CHARGE_CONSTANT

OUT=Path(__file__).resolve().parent/"data"/"original_design_audit"


def audit(name):
    preset=original_preset(name)
    output={"name":name,"preset":preset,"points":[]}
    with threadpool_limits(limits=1):
        for resonator in ("original","6_GHz"):
            f=preset["resonator_frequency_ghz"] if resonator=="original" else 6.
            z=preset["resonator_mode_impedance_ohm"] if resonator=="original" else 50.
            omega=2*np.pi*f*1e9
            d=Device(**{key:preset[key] for key in ("C1_shunt_fF","C2_shunt_fF","C12_fF","C1r_fF","C2r_fF")},
                     Cr_fF=1/(omega*z)*1e15,Lr_nH=z/omega*1e9)
            m=ReadoutModel(d,Basis(20,9,8,30))
            p=annotate(m,m.evaluate(preset["ej1_4p4_ghz"],preset["ej2_4p4_ghz"]))
            p.update(resonator_case=resonator,bare_resonator_ghz=f,failures=classify(p))
            # Local frequency bounds required by EJ/EC>50 and EJ<25.
            p["allowed_local_frequency_ranges_ghz"]=[
                [_transmon(50*ec,ec,m.basis)[0][1],_transmon(25,ec,m.basis)[0][1]] for ec in m.ec_ghz[:2]]
            output["points"].append(p)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--report-only",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        original=json.loads((OUT/"operating_points.json").read_text())
        results=json.loads((OUT/"retuned_shunt_search.json").read_text())
        render(original,results)
        return
    original=[audit(name) for name in ("tdt1","tdt2")]
    (OUT/"operating_points.json").write_text(json.dumps(original,indent=2))
    for row in original:
        for p in row["points"]:
            print(row["name"],p["resonator_case"],
                "EC",p["loaded_ec1_mhz"],p["loaded_ec2_mhz"],
                "EJ/EC",p["ej1_over_ec1"],p["ej2_over_ec2"],
                "gapMHz",p["splitting_ghz"]*1000,
                "chiR",max(p["ratio_m"],p["abs_delta_chi_over_chi_p"]),
                "lambda",p["lambda_max"],"fail",p["failures"],flush=True)
    # This second question allows redesigning C12 and the common readout cap,
    # unlike the direct audit of original circuits above.
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(search_pair,[(r["preset"]["C1_shunt_fF"],r["preset"]["C2_shunt_fF"]) for r in original]))
    for r,record in zip(results,original): r["name"]=record["name"]
    (OUT/"retuned_shunt_search.json").write_text(json.dumps(results,indent=2))
    for row in results:
        print(row["name"],"retuned readout-cap/C12/EJ search:",row["found"],"attempts",len(row["attempts"]),flush=True)
    render(original,results)


def render(original,results):
    lines=["# Original tdt1/tdt2 versus the later design requirements","",
        "The original presets are not the later optimized designs A/B. This audit preserves the original physical shunts, C12 and readout capacitors, and uses the equivalent single-junction EJs that reproduce the original operating point near 4.4 GHz. The original two junction capacitors are absorbed into each shunt as in reference_devices.py. No notebook parameters were modified.","",
        "## Direct audit with a 6 GHz bare resonator","",
        "The table uses a 6 GHz bare resonator and 50 ohm equivalent-mode impedance for direct comparison with the requested designs. Original-resonator results are also saved in operating_points.json; the same main failures hold there.","",
        "| Quantity | Original tdt1 | Original tdt2 | Requirement |","|---|---:|---:|---:|"]
    a,b=[next(p for p in row["points"] if p["resonator_case"]=="6_GHz") for row in original]
    rows=[("C1,C2 shunts (fF)",*[f"{p['device']['C1_shunt_fF']:.3f},{p['device']['C2_shunt_fF']:.3f}" for p in (a,b)],"fixed in direct audit"),
        ("Equal readout caps (fF)",a['device']['C1r_fF'],b['device']['C1r_fF'],"original values"),
        ("Loaded EC1,EC2/h (MHz)",*[f"{p['loaded_ec1_mhz']:.3f},{p['loaded_ec2_mhz']:.3f}" for p in (a,b)],"150-350 each"),
        ("Effective EJ1,EJ2/h (GHz)",*[f"{p['ej1_ghz']:.5f},{p['ej2_ghz']:.5f}" for p in (a,b)],"<25 each"),
        ("EJ1/EC1,EJ2/EC2",*[f"{p['ej1_over_ec1']:.3f},{p['ej2_over_ec2']:.3f}" for p in (a,b)],">50 each"),
        ("Logical gap (MHz)",*[f"{p['splitting_ghz']*1000:.3f}" for p in (a,b)],"95-105"),
        ("Worst chi ratio",*[f"{max(p['ratio_m'],p['abs_delta_chi_over_chi_p']):.6f}" for p in (a,b)],"<=0.01"),
        ("Multilevel dispersive maximum",*[f"{p['lambda_max']:.6f}" for p in (a,b)],"<=0.1"),
        ("Individual local g/Delta",*[f"{p['local_lambda1']:.4f},{p['local_lambda2']:.4f}" for p in (a,b)],"also exceeds 0.1"),
        ("Hybridization",*[f"{p['hybridization']:.6f}" for p in (a,b)],">=0.95"),
        ("Reference-state overlap",*[f"{p['min_assignment_overlap']:.6f}" for p in (a,b)],">=0.95 numerical screen")]
    for label,x,y,requirement in rows: lines.append(f"| {label} | {x} | {y} | {requirement} |")
    lines += ["","## Why larger shunts appeared in the later map","",
        "Good chi matching was never the only new requirement: the original operating points violate EJ/EC>50 and the requested dispersive cutoff. At the same approximately 4.4 GHz carrier, lowering EC makes EJ/EC>50 possible. Larger shunts reduce EC, while smaller readout capacitors reduce resonator coupling. These are simultaneous redesign choices, not a simple conversion between readout-capacitance values.","",
        "The original shunt participation is also below the later approximate 90% assumption:","",
        "| Circuit | Shunt participation in EC1 | Shunt participation in EC2 |","|---|---:|---:|"]
    for row,p in zip(original,(a,b)):
        participation=[p['device'][key]*p[ec]*1e-3/CHARGE_CONSTANT for key,ec in
                       (("C1_shunt_fF","loaded_ec1_mhz"),("C2_shunt_fF","loaded_ec2_mhz"))]
        lines.append(f"| {row['name']} | {100*participation[0]:.2f}% | {100*participation[1]:.2f}% |")
    lines += ["","Reducing C1r/C2r with shunts held fixed removes capacitive loading and raises the loaded ECs. This can make EJ/EC>50 harder at a fixed carrier even while the dispersive condition improves. Raising the carrier to restore EJ/EC then changes its detuning from the 6 GHz resonator.","",
        "## Retuning at the exact original shunt values","",
        "A separate bounded search allowed EJ1, EJ2, C12, equal readout capacitance and carrier (4-8 GHz) to change, preserving only each original shunt pair. This is a redesign question, distinct from whether the original circuits at their reported operating points pass.","",
        "| Original shunt pair | Passing redesigned witness found? | Nonlinear branch attempts |","|---|---|---:|"]
    for r in results: lines.append(f"| {r['name']} ({r['c1']:.3f},{r['c2']:.3f} fF) | {r['found']} | {len(r['attempts'])} |")
    lines += ["","No passing witness in this sampled search is not a proof of infeasibility. In particular, the shunt-pair figure is a map of discovered passing examples, not a rigorous physical boundary. Its gray cells must not be interpreted as an established exclusion region. Full original-point checks here use basis 20/9/8/30. Retuned searches use the smaller screening basis and preserve the bounded search assumptions documented in shunt_pair_map/report.md."]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__": main()
