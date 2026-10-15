"""6 GHz / 100 MHz search with equal numerical local 0-1 readout couplings.

Notebook inputs are untouched. Equality is enforced using the complete inverse
capacitance matrix and numerical charge elements, not a fourth-root estimate.
The resonance condition is a stationary dressed gap versus EJ2 at fixed circuit.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from model import Basis, Device, ReadoutModel
from study_dispersive import diagnostics

OUT = Path(__file__).resolve().parent / "data" / "equal_g"


def make_model(c1, c2, c12, cap1, cap2, basis=Basis()):
    omega = 2*np.pi*6e9
    return ReadoutModel(Device(C1_shunt_fF=c1, C2_shunt_fF=c2,
        C12_fF=c12, C1r_fF=cap1, C2r_fF=cap2,
        Cr_fF=1/(omega*50)*1e15, Lr_nH=50/omega*1e9), basis)


def annotate(model, point):
    point.update(diagnostics(model, point))
    ec = model.ec_ghz[:2]
    g1, g2 = point["g1r_over_2pi_mhz"], point["g2r_over_2pi_mhz"]
    point.update(device=asdict(model.device), basis=asdict(model.basis),
        loaded_ec1_mhz=float(ec[0]*1000), loaded_ec2_mhz=float(ec[1]*1000),
        ej1_over_ec1=float(point["ej1_ghz"]/ec[0]),
        ej2_over_ec2=float(point["ej2_ghz"]/ec[1]),
        g_relative_difference=float(abs(g1-g2)/((g1+g2)/2)))
    point["physical_pass"] = bool(
        max(point["ratio_m"], point["abs_delta_chi_over_chi_p"]) < .01
        and point["lambda_max"] <= .1 and point["hybridization"] >= .95
        and point["min_assignment_overlap"] >= .95
        and np.all((ec >= .15) & (ec <= .35))
        and min(point["ej1_over_ec1"], point["ej2_over_ec2"]) > 50
        and max(point["ej1_ghz"], point["ej2_ghz"]) < 25
        and .095 <= point["splitting_ghz"] <= .105)
    return point


def gap_derivative(m, ej1, ej2):
    step = 2e-4
    return (m.evaluate(ej1, ej2+step)["splitting_ghz"]
            -m.evaluate(ej1, ej2-step)["splitting_ghz"])/(2*step)


def solve(c1, c2, cap=6., carrier=4.4, initial=None, mode="resonance", basis=Basis(), chi_target=0.):
    # resonance: solve EJ1,EJ2,C12,C2r for carrier,gap,resonance,equal g.
    # both: also solve C1r to impose chi matching at the resonance minimum.
    # chi: substitute chi matching for resonance (comparison with old search).
    if initial is None:
        m = make_model(c1,c2,2.5,cap,cap,basis)
        initial = [(carrier+ec)**2/(8*ec) for ec in m.ec_ghz[:2]]+[2.5,cap]
        if mode == "both":
            initial += [cap]
    def unpack(x):
        return make_model(c1,c2,x[2],x[4] if mode == "both" else cap,x[3],basis)
    def residual(x):
        m = unpack(x)
        p = m.evaluate(x[0],x[1])
        g1,g2=p["g1r_over_2pi_mhz"],p["g2r_over_2pi_mhz"]
        result=[p["operating_frequency_ghz"]-carrier,p["splitting_ghz"]-.1,
                .1*(g1-g2)/((g1+g2)/2)]
        chi=.1*(p["delta_chi_ghz"]/max(abs(p["chi_p_ghz"]),abs(p["chi_m_ghz"]),1e-8)-chi_target)
        result += [chi if mode == "chi" else .1*gap_derivative(m,x[0],x[1])]
        if mode == "both":
            result += [chi]
        return result
    lower,upper=[2,2,.01,.2],[24.999,24.999,10,20]
    if mode == "both":
        lower += [.2]; upper += [20]
    sol=least_squares(residual,initial,bounds=(lower,upper),diff_step=3e-5,
        xtol=2e-10,ftol=2e-10,gtol=2e-10,max_nfev=100)
    m=unpack(sol.x)
    p=annotate(m,m.evaluate(sol.x[0],sol.x[1]))
    p.update(mode=mode, carrier_target=carrier, chi_target=chi_target, solver_residual=float(max(abs(sol.fun))),
        solver_success=bool(sol.success), solution=sol.x.tolist(),
        gap_derivative=gap_derivative(m,sol.x[0],sol.x[1]))
    p["passes"] = bool(p["physical_pass"] and p["solver_residual"] < 1e-6
                       and p["g_relative_difference"] < 1e-5)
    return p


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    with threadpool_limits(limits=1):
        for cap in (6.,10.):
            for c1,c2 in ((64,54),(80,80),(95,80),(115,80),(105,105),(115,100)):
                p=solve(c1,c2,cap)
                results.append(p)
                print(f"C={c1},{c2} Cr1={cap}: R={p['abs_delta_chi_over_chi_p']:.5g} "
                      f"lambda={p['lambda_max']:.4f} g={p['g1r_over_2pi_mhz']:.3f},"
                      f"{p['g2r_over_2pi_mhz']:.3f} PASS={p['passes']}",flush=True)
                (OUT/"resonance_grid.json").write_text(json.dumps(results,indent=2))
        matched=[]
        for c1,c2 in ((80,80),(95,80),(115,80),(105,105),(115,100)):
            p=solve(c1,c2,6.,mode="both")
            matched.append(p)
            print(f"BOTH C={c1},{c2}: R={p['abs_delta_chi_over_chi_p']:.5g} "
                  f"lambda={p['lambda_max']:.4f} caps={p['device']['C1r_fF']:.4f},"
                  f"{p['device']['C2r_fF']:.4f} PASS={p['passes']}",flush=True)
            (OUT/"matched_designs.json").write_text(json.dumps(matched,indent=2))


if __name__ == "__main__":
    main()
