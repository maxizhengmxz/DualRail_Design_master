"""Joint inequality search for the refined, symmetry-restricted design map.

All energies in GHz (H/h), capacitances in fF. The logical gap is a free
inequality, not an equality target. Local frequencies parameterize EJ tuning
and are inverted numerically using each candidate's actual loaded EC.
"""
from dataclasses import asdict
import time

import numpy as np
from scipy.optimize import brentq, least_squares
from threadpoolctl import threadpool_limits

from model import Basis, ReadoutModel, _transmon
from map_shunt_pairs import make_model, CHARGE_CONSTANT
from study_dispersive import diagnostics

SETTINGS={
    "schema":"refined-inequality-map-v1", "strict_triangle":"C1>C2",
    "bare_resonator_ghz":6.,"mode_impedance_ohm":50.,
    "carrier_ghz":[4.,8.],"gap_ghz":[.09,.11],"ec_ghz":[.15,.4],
    "ec_upper_strict":True,"ej_upper_ghz_strict":25.,"ej_over_ec_lower_strict":50.,
    "chi_max":.01,"lambda_max":.1,"hybridization_min":.95,"overlap_min":.95,
    "equal_readout_caps":True,"cg_bounds_fF":[.1,25.],"c12_bounds_fF":[.005,15.],
    "initial_participation_estimate":.9,"participation_enforced":False,
    "screening_basis":asdict(Basis()),
    "validation_basis":asdict(Basis(20,9,8,40)),
    "no_minimum_chi_or_readout_speed":True,
}


def inverse_frequency(frequency,ec,basis):
    return brentq(lambda ej:_transmon(ej,ec,basis)[0][1]-frequency,
                  .02,100.,xtol=2e-10)


def point_from_x(c1,c2,x,basis=Basis()):
    fmean,detune_units,c12,cg=map(float,x)
    m=make_model(c1,c2,c12,cg,basis)
    detune=.01*detune_units
    ej1=inverse_frequency(fmean+detune/2,m.ec_ghz[0],basis)
    ej2=inverse_frequency(fmean-detune/2,m.ec_ghz[1],basis)
    p=m.evaluate(ej1,ej2)
    p.update(diagnostics(m,p))
    ec=m.ec_ghz[:2]
    p.update(device=asdict(m.device),basis=asdict(basis),x=list(map(float,x)),
        loaded_ec1_mhz=float(ec[0]*1000),loaded_ec2_mhz=float(ec[1]*1000),
        ej1_over_ec1=float(ej1/ec[0]),ej2_over_ec2=float(ej2/ec[1]),
        participation1=float(c1*1e-15*m.cinv[0,0]),
        participation2=float(c2*1e-15*m.cinv[1,1]))
    p["ratio"]=float(max(p["ratio_m"],p["abs_delta_chi_over_chi_p"]))
    return p


def x_from_point(p):
    return [(p["bare_f1_ghz"]+p["bare_f2_ghz"])/2,
            (p["bare_f1_ghz"]-p["bare_f2_ghz"])/.01,
            p["device"]["C12_fF"],p["device"]["C1r_fF"]]


def violations(p,margin=0.):
    ec1,ec2=p["loaded_ec1_mhz"]/1000,p["loaded_ec2_mhz"]/1000
    # Every condition participates in optimization. Small optional margins
    # improve numerical stability but are not additional acceptance criteria.
    return np.array([
        max(0.,p["ratio"]-(.01-margin*.001))/.01,
        max(0.,(.95+margin*.002)-p["hybridization"])/.05,
        max(0.,p["lambda_max"]-(.1-margin*.002))/.1,
        max(0.,(.95+margin*.002)-p["min_assignment_overlap"])/.05,
        max(0.,(.09+margin*.0002)-p["splitting_ghz"])/.01,
        max(0.,p["splitting_ghz"]-(.11-margin*.0002))/.01,
        max(0.,4.+margin*.001-p["operating_frequency_ghz"]),
        max(0.,p["operating_frequency_ghz"]-(8.-margin*.001)),
        max(0.,.15+margin*.0001-ec1)/.1,
        max(0.,.15+margin*.0001-ec2)/.1,
        max(0.,ec1-(.4-margin*.0001))/.1,
        max(0.,ec2-(.4-margin*.0001))/.1,
        max(0.,p["ej1_ghz"]-(25.-margin*.01))/10,
        max(0.,p["ej2_ghz"]-(25.-margin*.01))/10,
        max(0.,50.+margin*.02-p["ej1_over_ec1"])/50,
        max(0.,50.+margin*.02-p["ej2_over_ec2"])/50,
    ])


def failures(p):
    checks={"chi":p["ratio"]<=.01+1e-8,
        "H":p["hybridization"]>=.95-1e-8,
        "lambda":p["lambda_max"]<=.1+1e-9,
        "overlap":p["min_assignment_overlap"]>=.95-1e-8,
        "gap":.09-1e-8<=p["splitting_ghz"]<=.11+1e-8,
        "carrier":4-1e-8<=p["operating_frequency_ghz"]<=8+1e-8,
        "EC":150-1e-7<=min(p["loaded_ec1_mhz"],p["loaded_ec2_mhz"])
              and max(p["loaded_ec1_mhz"],p["loaded_ec2_mhz"])<400,
        "EJ":max(p["ej1_ghz"],p["ej2_ghz"])<25,
        "EJ_EC":min(p["ej1_over_ec1"],p["ej2_over_ec2"])>50}
    return [key for key,ok in checks.items() if not ok]


def quality(p):
    """Prefer a passing witness with margin, not an artificial equality."""
    if failures(p): return (-1.,-float(np.linalg.norm(violations(p))),p["hybridization"])
    margins=[(.01-p["ratio"])/.01,(p["hybridization"]-.95)/.05,
             (.1-p["lambda_max"])/.1,(p["splitting_ghz"]-.09)/.02,
             (.11-p["splitting_ghz"])/.02]
    return (1.,float(min(margins)),p["hybridization"])


def seed_candidates(c1,c2,warm=()):
    seeds=[]
    for p in warm:
        if p is not None: seeds.append(np.array(p.get("x",x_from_point(p)),float))
    # Prioritize both sides of the resonator; actual energy windows provide
    # additional seeds rather than eliminating points on approximate bounds.
    windows=[]
    for cg in (1.,4.,8.):
        m=make_model(c1,c2,2.,cg)
        ec=m.ec_ghz[:2]
        lower=max(4.02,19.05*max(ec))
        upper=min(7.98,np.sqrt(199.5*min(ec))-min(ec))
        if lower<upper:
            windows.extend((lower+.1*(upper-lower),(lower+upper)/2,upper-.1*(upper-lower)))
    freqs=list(dict.fromkeys(round(f,5) for f in ([4.4,7.5,4.05,7.95,5.,6.8]+windows)))
    for f in freqs:
        m=make_model(c1,c2,2.,4.)
        ec=m.ec_ghz[:2]
        if any((f+v)**2/(8*v)>29 or (f+v)**2/(8*v)/v<40 for v in ec): continue
        ej=[(f+v)**2/(8*v) for v in ec]
        p=m.evaluate(*ej);p.update(diagnostics(m,p))
        cg=float(np.clip(4*.075/max(p["lambda_max"],.005),.15,18.))
        ceff=np.sqrt((CHARGE_CONSTANT/ec[0])*(CHARGE_CONSTANT/ec[1]))
        c12=float(np.clip(.1*ceff/f-cg*cg/530,.15,5.))
        # Include differential detuning in both directions; it is not fixed to
        # the resonance minimum or to exact chi matching.
        for detune in (0.,.5,-.5): seeds.append(np.array([f,detune,c12,cg]))
    unique=[]
    for seed in seeds:
        seed=np.clip(seed,[3.8,-15,.005,.1],[8.2,15,15,25])
        if not any(np.linalg.norm(seed-prior)<1e-5 for prior in unique): unique.append(seed)
    return unique


def search(task):
    c1,c2=map(float,task["pair"])
    if not c1>c2: raise ValueError("Search is restricted to strict C1>C2")
    level=task.get("level","normal")
    basis=Basis(**task["basis"]) if "basis" in task else Basis()
    limits={"fast":(3,35,25.),"normal":(9,65,90.),"deep":(24,110,240.)}
    nstarts,max_nfev,max_seconds=limits[level]
    started=time.monotonic();attempts=[];best=None;best_pass=None;evals=0
    with threadpool_limits(limits=1):
        seeds=seed_candidates(c1,c2,task.get("warm",()))
        # Cold and nearby-solution seeds are interleaved so upper-band starts
        # are tested before spending all trials in one basin.
        if len(task.get("warm",())) and len(seeds)>len(task["warm"])+3:
            nw=min(len(task["warm"]),len(seeds))
            ordered=[]
            for i in range(max(nw,len(seeds)-nw)):
                if i<nw: ordered.append(seeds[i])
                if nw+i<len(seeds): ordered.append(seeds[nw+i])
            seeds=ordered
        cache={}
        def evaluate(x):
            nonlocal best,best_pass,evals
            key=tuple(map(float,x))
            if key not in cache:
                p=point_from_x(c1,c2,x,basis);evals+=1
                cache[key]=p
                if best is None or quality(p)>quality(best): best=p
                if not failures(p) and (best_pass is None or quality(p)>quality(best_pass)): best_pass=p
            return cache[key]
        for index,seed in enumerate(seeds[:nstarts]):
            if index and time.monotonic()-started>max_seconds: break
            margin=.5 if level!="deep" else (.2 if index%2==0 else 0.)
            try:
                sol=least_squares(lambda x:np.nan_to_num(violations(evaluate(x),margin),nan=1e3,posinf=1e3),
                    seed,bounds=([3.8,-15,.005,.1],[8.2,15,15,25]),
                    x_scale=[1.,1.,2.,5.],diff_step=2e-5,ftol=1e-8,xtol=1e-8,gtol=1e-8,
                    max_nfev=max_nfev)
                p=evaluate(sol.x)
                attempts.append({"seed":seed.tolist(),"x":sol.x.tolist(),"cost":float(sol.cost),
                    "nfev":int(sol.nfev),"failures":failures(p),"ratio":p["ratio"],
                    "H":p["hybridization"],"lambda":p["lambda_max"],"gap":p["splitting_ghz"]})
            except (ValueError,FloatingPointError) as exc:
                attempts.append({"seed":seed.tolist(),"error":str(exc)})
            if best_pass is not None: break
    selected=best_pass or best
    return {"c1":c1,"c2":c2,"level":level,"status":"screened_pass" if best_pass else "no_witness_found",
        "point":selected,"failures":failures(selected) if selected else ["no_evaluation"],
        "attempts":attempts,"evaluations":evals,"elapsed_seconds":time.monotonic()-started}


def validate(result,basis=Basis(20,9,8,40)):
    p=result["point"]
    with threadpool_limits(limits=1):
        m=ReadoutModel(__import__("model").Device(**p["device"]),basis)
        q=m.evaluate(p["ej1_ghz"],p["ej2_ghz"]);q.update(diagnostics(m,q))
        ec=m.ec_ghz[:2]
        q.update(device=p["device"],basis=asdict(basis),x=p["x"],
            loaded_ec1_mhz=float(ec[0]*1000),loaded_ec2_mhz=float(ec[1]*1000),
            ej1_over_ec1=float(p["ej1_ghz"]/ec[0]),ej2_over_ec2=float(p["ej2_ghz"]/ec[1]),
            participation1=p["participation1"],participation2=p["participation2"])
        q["ratio"]=float(max(q["ratio_m"],q["abs_delta_chi_over_chi_p"]))
    out=dict(result);out["screening_point"]=p;out["point"]=q
    out["failures"]=failures(q);out["status"]="validated_pass" if not out["failures"] else "validation_failed"
    return out
