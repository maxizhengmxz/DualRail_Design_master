"""Direct loaded-EC feasibility search at fixed EC1 and EC2.

Capacitances are solved analytically from the Schur complement of the full
three-node C matrix at every trial. This preserves the requested loaded ECs
as C12 and equal readout coupling capacitors change. No readout optimization,
failed-target retries, contour refinement, or failure diagnosis is performed.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,time
from pathlib import Path
from dataclasses import replace,asdict
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import ConvexHull,Delaunay
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
# Also support an explicit direct-file invocation from an IDE.
if not __package__:
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from readout_workflow import calculation as s
from readout_workflow.validation import audit_point
from readout_workflow.storage import atomic_json,utc
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'readout_workflow/data/direct_ec_step5'
SOURCE=ROOT/'readout_workflow/data/feasibility_150mhz_step1_diag2/results.json'

def initialize():
    s.CONDITIONS=replace(s.CONDITIONS,logical_frequency_ghz=(.14,.16))
    s.BASELINE=dict(s.BASELINE,logical_frequency=(.14,.16))

def capacitances(ec1,ec2,c12,cg):
    """Physical C1,C2 (fF) with EXACT prescribed inverse-matrix diagonals.

    Eliminate resonator: A=[[X,-b],[-b,Y]], b=C12+Cg^2/R,
    R=Cr+2Cg. If diag(A^-1)=(d1,d2), determinant D satisfies
    d1*d2*D^2-D-b^2=0. X=d2*D, Y=d1*D; subtract attached C.
    """
    cr=s.Device().Cr_fF;r=cr+2*cg
    d1=(ec1/1000)/s.CHARGE_CONSTANT;d2=(ec2/1000)/s.CHARGE_CONSTANT
    b=c12+cg*cg/r
    determinant=(1+np.sqrt(1+4*d1*d2*b*b))/(2*d1*d2)
    correction=c12+cg-cg*cg/r
    return d2*determinant-correction,d1*determinant-correction

def point(ec1,ec2,x,basis):
    c1,c2=capacitances(ec1,ec2,x[2],x[3])
    return s.point_from_x(c1,c2,x,basis)

def solve(task):
    initialize();ec1,ec2=task['ec'];started=time.monotonic();attempts=[];chosen=None;closest=None
    lower,upper=map(np.array,s.nominal_search_bounds());cache={}
    def evaluate(x):
        k=tuple(map(float,x))
        if k not in cache:cache[k]=point(ec1,ec2,x,s.SCREENING_BASIS)
        return cache[k]
    def residual(x):
        p=evaluate(x);d=p['device']
        return np.r_[s.violations(p,margin=.3),max(0,36-d['C1_shunt_fF'])/10,max(0,d['C1_shunt_fF']-128)/10,max(0,36-d['C2_shunt_fF'])/10,max(0,d['C2_shunt_fF']-128)/10]
    seeds=[]
    for p in task['warm']:
        x=np.clip(s.x_from_point(p),lower+1e-6,upper-1e-6)
        if not any(np.linalg.norm(x-y)<1e-5 for y in seeds):seeds.append(x)
    with threadpool_limits(limits=1):
        for seed in seeds[:3]:
            if attempts and time.monotonic()-started>25:break
            fit=least_squares(residual,seed,bounds=(lower,upper),x_scale=[1,1,2,5],diff_step=2e-5,ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=35)
            p=evaluate(fit.x);attempts.append(dict(seed=seed.tolist(),x=fit.x.tolist(),nfev=int(fit.nfev),cost=float(fit.cost),failures=s.failures(p),endpoint=p))
            if closest is None or np.linalg.norm(s.violations(p))<np.linalg.norm(s.violations(closest)):closest=p
            candidates=[q for q in cache.values() if not s.failures(q)]
            if candidates:
                # First acceptable numerical witness; do not rank by readout.
                q=audit_point(candidates[0]);d=q['device']
                errs=[abs(q['loaded_ec1_mhz']-ec1),abs(q['loaded_ec2_mhz']-ec2)]
                failed=s.failures(q,require_sign_check=True)
                if any(v>1e-6 for v in errs):failed.append('loaded_EC_target')
                if not all(36<=d[k]<=128 for k in ('C1_shunt_fF','C2_shunt_fF')):failed.append('capacitance_search_range')
                attempts[-1]['audit_failures']=failed
                if not failed:chosen=q;break
    return dict(ec1_target_mhz=ec1,ec2_target_mhz=ec2,status='validated_pass' if chosen else 'no_validated_solution_found',point=chosen,closest_screening_point=closest,attempts=attempts,seconds=time.monotonic()-started,completed_utc=utc(),readout_optimized=False)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--output',type=Path,default=OUT);a=parser.parse_args();OUT=a.output.resolve()
    if not a.run:print('Use --run to calculate; no automatic calculation on import.');return
    initialize();OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads(SOURCE.read_text());good=[r['point'] for r in old if r['status'] in ('validated_pass','recovered_viable')];xy=np.array([[p['loaded_ec1_mhz'],p['loaded_ec2_mhz']] for p in good]);hull=ConvexHull(xy);tri=Delaunay(xy[hull.vertices])
    xs=np.arange(np.ceil(xy[:,0].min()/5)*5,xy[:,0].max()+1e-8,5);ys=np.arange(np.ceil(xy[:,1].min()/5)*5,xy[:,1].max()+1e-8,5);targets=[(int(x),int(y)) for x in xs for y in ys if tri.find_simplex([[x,y]])[0]>=0]
    assert len(targets)==48
    policy=dict(conditions=asdict(s.CONDITIONS),ec_step_mhz=5,targets=targets,contour='convex hull of 339 prior validated witnesses; membership is not assumed feasible',source_sha256=sha256(SOURCE.read_bytes()).hexdigest(),capacitance_range_ff=[36,128],budget=dict(starts=3,max_nfev=35,screening_soft_seconds=25),optimize_readout=False,retry_failed=False)
    from readout_workflow.validation import POLICY
    files=[Path(__file__),*[ROOT/'readout_workflow'/name for name in ('calculation.py','model.py','validation.py')],*sorted((ROOT/'core').glob('*.py'))]
    policy['validation_policy']=POLICY
    policy['source_hashes']={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}
    settings_path=OUT/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=json.loads(json.dumps(policy)):
        raise RuntimeError('Saved settings/source/validation policy differ. Use --output with a new directory; existing results are preserved.')
    atomic_json(settings_path,policy)
    # Verify exact capacitor reconstruction on every previous physical witness.
    worst=0
    for p in good:
        d=p['device'];c1,c2=capacitances(p['loaded_ec1_mhz'],p['loaded_ec2_mhz'],d['C12_fF'],d['C1r_fF']);worst=max(worst,abs(c1-d['C1_shunt_fF']),abs(c2-d['C2_shunt_fF']))
    assert worst<1e-8
    atomic_json(OUT/'capacitance_reconstruction_check.json',dict(witnesses_checked=len(good),max_capacitance_error_ff=worst))
    rows=[]
    for e in targets:
        f=OUT/f'ec_{e[0]}_{e[1]}.json'
        if f.exists():rows.append(json.loads(f.read_text()))
    done={(r['ec1_target_mhz'],r['ec2_target_mhz']) for r in rows}
    def progress(phase='running'):atomic_json(OUT/'progress.json',dict(phase=phase,total=48,completed=len(rows),passed=sum(r['status']=='validated_pass' for r in rows),updated_utc=utc(),pid=os.getpid()))
    progress()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={}
        for e in targets:
            if e in done:continue
            near=np.argsort(np.sum((xy-np.array(e))**2,axis=1))[:3]
            futures[pool.submit(solve,dict(ec=e,warm=[good[i] for i in near]))]=e
        for future in as_completed(futures):
            r=future.result();rows.append(r);atomic_json(OUT/f"ec_{r['ec1_target_mhz']}_{r['ec2_target_mhz']}.json",r);progress();print(json.dumps(dict(ec=futures[future],status=r['status'],seconds=r['seconds'],completed=len(rows))),flush=True)
    rows.sort(key=lambda r:(r['ec1_target_mhz'],r['ec2_target_mhz']));atomic_json(OUT/'results.json',rows);progress('complete');print('COMPLETE',flush=True)
if __name__=='__main__':main()
