"""Search physical shunt pairs, equal readout capacitors, carrier 4-8 GHz.

90% participation labels the search bounds only; final charging energies and
all couplings use the physical capacitance matrix. No notebook edits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge as e, h
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from model import Basis, Device, ReadoutModel
from search_equal_g import annotate

OUT=Path(__file__).resolve().parent/"data"/"shunt_pair_map"
CHARGE_CONSTANT=e**2/(2*h)*1e6  # EC[GHz] = constant / C[fF].
MAX_SHUNT=.9*CHARGE_CONSTANT/.15
GRID=np.unique(np.r_[np.arange(50.,116.,5.),MAX_SHUNT])


def make_model(c1,c2,c12,cg,basis=Basis()):
    omega=2*np.pi*6e9
    return ReadoutModel(Device(C1_shunt_fF=c1,C2_shunt_fF=c2,C12_fF=c12,
        C1r_fF=cg,C2r_fF=cg,Cr_fF=1/(omega*50)*1e15,Lr_nH=50/omega*1e9),basis)


def classify(p):
    checks={"chi":max(p["ratio_m"],p["abs_delta_chi_over_chi_p"])<=.0100001,
        "hybridization":p["hybridization"]>=.95,
        "dispersive":p["lambda_max"]<=.1,
        "overlap":p["min_assignment_overlap"]>=.95,
        "ec":150<=min(p["loaded_ec1_mhz"],p["loaded_ec2_mhz"])
             and max(p["loaded_ec1_mhz"],p["loaded_ec2_mhz"])<=350,
        "ej_ec":min(p["ej1_over_ec1"],p["ej2_over_ec2"])>50,
        "ej":max(p["ej1_ghz"],p["ej2_ghz"])<25,
        "carrier":4-1e-6<=p["operating_frequency_ghz"]<=8+1e-6,
        "gap":.095<=p["splitting_ghz"]<=.105,
        "solver":p.get("solver_residual",0)<2e-6}
    return [key for key,ok in checks.items() if not ok]


def solve(c1,c2,cg,f,mode="resonance",initial=None,chi_target=0.,basis=Basis()):
    if initial is None:
        m=make_model(c1,c2,2.5,cg,basis)
        initial=[np.clip((f+ec)**2/(8*ec),2.001,24.998) for ec in m.ec_ghz[:2]]+[2.5]
    def residual(x):
        m=make_model(c1,c2,x[2],cg,basis)
        p=m.evaluate(x[0],x[1])
        if mode=="resonance":
            step=2e-4
            third=.1*(m.evaluate(x[0],x[1]+step)["splitting_ghz"]
                      -m.evaluate(x[0],x[1]-step)["splitting_ghz"])/(2*step)
        else:
            third=.1*(p["delta_chi_ghz"]/min(abs(p["chi_p_ghz"]),abs(p["chi_m_ghz"]))-chi_target)
        return [p["operating_frequency_ghz"]-f,p["splitting_ghz"]-.1,third]
    sol=least_squares(residual,initial,bounds=([2,2,.01],[24.999,24.999,10]),
        diff_step=3e-5,xtol=1e-9,ftol=1e-9,gtol=1e-9,max_nfev=45)
    m=make_model(c1,c2,sol.x[2],cg,basis)
    p=annotate(m,m.evaluate(sol.x[0],sol.x[1]))
    p.update(solver_residual=float(max(abs(sol.fun))),solution=sol.x.tolist(),mode=mode,
        chi_target=chi_target,carrier_target=f,
        participation1=c1*1e-15*m.cinv[0,0],participation2=c2*1e-15*m.cinv[1,1])
    p["failures"]=classify(p);p["passes"]=not p["failures"]
    return p


def carrier_candidates(c1,c2):
    # Cover all 4-8 GHz, plus near energy-boundary and midpoint frequencies.
    # The estimates only prioritize trials; numerical ECs decide acceptance.
    approx=.9*CHARGE_CONSTANT/np.array([c1,c2])
    lo=max(4.,19*max(approx));hi=min(8.,np.sqrt(200*min(approx))-min(approx))
    extras=[4.4]
    if lo<hi: extras += [lo+.08,hi-.08,(lo+hi)/2]
    # Near an EJ/EC or EJ limit, a 90% capacitance estimate can shift a narrow
    # allowed carrier window. Also sample windows from actual physical ECs.
    for cg in (3.,8.):
        actual=make_model(c1,c2,2.5,cg).ec_ghz[:2]
        low=max(4.,19*max(actual));high=min(8.,np.sqrt(200*min(actual))-min(actual))
        if low<high: extras += [low+.03,high-.03,(low+high)/2]
    candidates=np.unique(np.round(np.r_[np.arange(4.,8.01,.5),extras],5))
    candidates=candidates[(candidates>=4)&(candidates<=8)&(abs(candidates-6)>.12)]
    # 4.4 first, then prioritized physically allowed frequencies, then others.
    return sorted(candidates,key=lambda f:(not (lo-.3<=f<=hi+.3),abs(f-4.4)))


def search_pair(task):
    c1,c2=task
    attempts=[]
    with threadpool_limits(limits=1):
        for f in carrier_candidates(c1,c2):
            # Estimate a useful cap from the multilevel dispersive ratio at
            # the loaded harmonic resonance; iterate below with exact screens.
            m=make_model(c1,c2,2.5,5.)
            ec=m.ec_ghz[:2]
            guess=[np.clip((f+x)**2/(8*x),2.001,24.998) for x in ec]
            if any((f+x)**2/(8*x)>26 or (f+x)**2/(8*x)/x<45 for x in ec):
                continue
            probe=annotate(m,m.evaluate(*guess))
            cap_est=np.clip(5*.085/max(probe["lambda_max"],.005),.3,18.)
            caps=np.unique(np.clip([cap_est,cap_est*.6,cap_est*1.17],.3,18.))
            # Try middle-strength first, weaker then closer to cutoff.
            caps=sorted(caps,key=lambda cap:abs(cap-cap_est))
            for cg in caps:
                r=solve(c1,c2,cg,f)
                attempts.append(r)
                # Explicitly approach the allowed dispersive edge; do not lose
                # a design merely because the first cap estimate overshoots.
                for _ in range(2):
                    if r["solver_residual"]<2e-6 and r["lambda_max"]>.0995:
                        cg*=.099/r["lambda_max"]
                        r=solve(c1,c2,cg,f)
                        attempts.append(r)
                if r["passes"]:
                    return {"c1":c1,"c2":c2,"found":True,"best":r,"attempts":attempts}
                if r["solver_residual"]>2e-6 or r["lambda_max"]>.115:
                    continue
                if any(key in r["failures"] for key in ("ec","ej_ec","ej","carrier")):
                    continue
                # Allow non-maximal mixing. Test both signs at a 0.98% target,
                # leaving numerical/convergence margin to the 1% ceiling.
                choices=[]
                for target in (-.0098,.0098):
                    q=solve(c1,c2,cg,f,mode="chi",initial=r["solution"],chi_target=target)
                    attempts.append(q)
                    if q["passes"]: choices.append(q)
                if choices:
                    best=max(choices,key=lambda p:p["hybridization"])
                    return {"c1":c1,"c2":c2,"found":True,"best":best,"attempts":attempts}
    # Report closest attempt for diagnostics, without declaring infeasibility.
    best=min(attempts,key=lambda p:(len(p["failures"]),max(p["ratio_m"],p["abs_delta_chi_over_chi_p"]))) if attempts else None
    return {"c1":c1,"c2":c2,"found":False,"best":best,"attempts":attempts}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--refine",action="store_true")
    parser.add_argument("--edge-coupling",action="store_true")
    parser.add_argument("--retry-failed",action="store_true")
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    grid=GRID if not args.smoke else np.array([60.,80.,100.,115.])
    tasks=[(float(c1),float(c2)) for c1 in grid for c2 in grid if c2<=c1]
    path=OUT/("smoke.json" if args.smoke else "grid.json")
    results=json.loads(path.read_text()) if args.resume and path.exists() else []
    completed={(r["c1"],r["c2"]) for r in results}
    tasks=[task for task in tasks if task not in completed]
    if args.refine:
        path=OUT/"refined.json"
        results=json.loads(path.read_text()) if path.exists() else []
        combined=json.loads((OUT/"grid.json").read_text())
        if (OUT/"edge_coupling.json").exists():
            combined += json.loads((OUT/"edge_coupling.json").read_text())
        combined += results
        combined=list({(r["c1"],r["c2"]):r for r in combined}.values())
        tasks=[]
        for c1 in GRID:
            row=sorted([r for r in combined if r["c1"]==c1],key=lambda r:r["c2"])
            for a,b in zip(row[:-1],row[1:]):
                if a["found"]!=b["found"]:
                    tasks.append((float(c1),(a["c2"]+b["c2"])/2))
        tasks=list(dict.fromkeys(tasks))
    if args.edge_coupling:
        path=OUT/"edge_coupling.json"
        results=json.loads(path.read_text()) if path.exists() else []
        coarse=json.loads((OUT/"grid.json").read_text())
        good=[r for r in coarse if r["found"]]
        completed={(r["c1"],r["c2"]) for r in results if r["found"] or not args.retry_failed}
        tasks=[(r["c1"],r["c2"]) for r in coarse if not r["found"]
               and (r["c1"],r["c2"]) not in completed
               and any(abs(r["c1"]-g["c1"])<=5.01 and abs(r["c2"]-g["c2"])<=5.01 for g in good)]
    (OUT/"settings.json").write_text(json.dumps({"participation_for_bounds":.9,
        "EC_range_GHz":[.15,.35],"shunt_bounds_fF":[.9*CHARGE_CONSTANT/.35,MAX_SHUNT],
        "grid_fF":grid.tolist(),"carrier_range_GHz":[4,8],"readout_caps":"equal, variable",
        "C12":"variable","impedance_mode_ohm":50,"bare_resonator_GHz":6,
        "chi_search_target":.0098,"chi_acceptance":.01},indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(search_pair,task) for task in tasks]
        for future in as_completed(pending):
            result=future.result();results.append(result)
            results=list({(r["c1"],r["c2"]):r for r in results}.values())
            print(f"{len(results)} pairs: C1={result['c1']:.3f}, C2={result['c2']:.3f}, found={result['found']}, attempts={len(result['attempts'])}",flush=True)
            path.write_text(json.dumps(sorted(results,key=lambda r:(r["c1"],r["c2"])),indent=2))


if __name__=="__main__": main()
