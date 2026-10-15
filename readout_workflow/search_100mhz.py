"""Search loaded EC values with a 6 GHz resonator and 100 MHz logical splitting.

This leaves notebook inputs untouched. Local EJs and C12 are solved together
for a specified dressed carrier and a stationary splitting versus EJ2.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from model import Basis, Device, ReadoutModel
from study_dispersive import diagnostics


def solve_point(c1, c2, cap=6., carrier=4.4, asymmetry=1., initial=None, basis=Basis(), match_chi=False):
    w,z = 2*np.pi*6e9,50.
    def model(c12):
        return ReadoutModel(Device(C1_shunt_fF=c1,C2_shunt_fF=c2,C12_fF=c12,
                                  C1r_fF=cap,C2r_fF=cap*asymmetry,
                                  Cr_fF=1/(w*z)*1e15,Lr_nH=z/w*1e9),basis)
    if initial is None:
        m=model(2.5)
        initial=[(carrier+ec)**2/(8*ec) for ec in m.ec_ghz[:2]]+[2.5]
    def residual(x):
        ej1,ej2,c12=x
        m=model(c12)
        p=m.evaluate(ej1,ej2)
        if match_chi:
            scale=max(abs(p["chi_p_ghz"]),abs(p["chi_m_ghz"]),1e-8)
            return [p["operating_frequency_ghz"]-carrier,p["splitting_ghz"]-.1,
                    .1*p["delta_chi_ghz"]/scale]
        delta=2e-4
        derivative=(m.evaluate(ej1,ej2+delta)["splitting_ghz"]-m.evaluate(ej1,ej2-delta)["splitting_ghz"])/(2*delta)
        return [p["operating_frequency_ghz"]-carrier,p["splitting_ghz"]-.1,.1*derivative]
    sol=least_squares(residual,initial,bounds=([2,2,.01],[24.999,24.999,10]),
                      xtol=1e-10,ftol=1e-10,gtol=1e-10,diff_step=1e-4,max_nfev=70)
    m=model(sol.x[2])
    p=m.evaluate(sol.x[0],sol.x[1])
    p.update(diagnostics(m,p))
    ec=m.ec_ghz[:2]
    p.update({"loaded_ec1_mhz":ec[0]*1e3,"loaded_ec2_mhz":ec[1]*1e3,
              "ej1_over_ec1":sol.x[0]/ec[0],"ej2_over_ec2":sol.x[1]/ec[1],
              "device":asdict(m.device),"carrier_target":carrier,
              "solver_residual":float(np.max(np.abs(sol.fun))),
              "solver_success":bool(sol.success)})
    p["search_objective"] = "chi matching" if match_chi else "stationary splitting"
    p["passes"] = bool(sol.success and p["solver_residual"]<1e-6
                       and max(p["abs_delta_chi_over_chi_p"],p["ratio_m"])<.01
                       and p["lambda_max"]<.1 and p["min_assignment_overlap"]>=.95
                       and p["hybridization"]>=.95
                       and np.all(ec>=.15) and np.all(ec<=.35)
                       and min(p["ej1_over_ec1"],p["ej2_over_ec2"])>50)
    return p


def main():
    out=Path(__file__).resolve().parent/"data"/"search_100mhz"
    out.mkdir(parents=True,exist_ok=True)
    points=[]
    with threadpool_limits(limits=1):
        for cap in (6.,10.):
            for c1 in (75.,85.,95.,105.):
                for c2 in (75.,85.,95.,105.):
                    p=solve_point(c1,c2,cap)
                    points.append(p)
                    print(f"Cg={cap:g} C1={c1:g} C2={c2:g}: R={p['abs_delta_chi_over_chi_p']:.5g} lambda={p['lambda_max']:.4f} EC={p['loaded_ec1_mhz']:.1f},{p['loaded_ec2_mhz']:.1f} ratioE={p['ej1_over_ec1']:.1f},{p['ej2_over_ec2']:.1f} C12={p['device']['C12_fF']:.4f} PASS={p['passes']}",flush=True)
                    (out/"grid.json").write_text(json.dumps(points,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
