"""Attainable frequency endpoints at fixed loaded EC pairs.

Four constrained searches per saved feasible pair: logical min/max and dressed
operation min/max. The logical window remains 90-110 MHz for every search.
Only the selected frequency is optimized; chi strength is a constraint, not an
objective. Results are best-found validated endpoints, not global certificates.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,subprocess
from pathlib import Path
from dataclasses import asdict,replace
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from readout_workflow import calculation as s
from readout_workflow.direct_ec_feasibility import point
from readout_workflow.validation import audit_point,FAST,INTERMEDIATE,POLICY
from readout_workflow.storage import atomic_json,utc
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'readout_workflow/data/direct_ec_100mhz_step2/results.json'
OUT=ROOT/'readout_workflow/data/frequency_limits_100mhz_ec_step2'
OBJECTIVES=('logical_min','logical_max','operation_min','operation_max')

def configure():
    s.CONDITIONS=replace(s.CONDITIONS,logical_frequency_ghz=(.09,.11))
    s.BASELINE=dict(s.BASELINE,logical_frequency=(.09,.11))

def solve(task):
    configure();start=time.perf_counter();ec1,ec2=task['ec'];baseline=task['point'];goal=task['objective'];key='splitting_ghz' if goal.startswith('logical') else 'operating_frequency_ghz';direction=1 if goal.endswith('min') else -1;scale=10 if key=='splitting_ghz' else .2
    lower,upper=map(np.array,s.nominal_search_bounds());cache={};attempts=[]
    def evaluate(x,basis):
        identifier=(basis,tuple(map(float,x)))
        if identifier not in cache:cache[identifier]=point(ec1,ec2,x,basis)
        return cache[identifier]
    def constraints(x,basis):
        p=evaluate(x,basis);d=p['device'];low,high=s.CONDITIONS.logical_frequency_ghz
        # 1 kHz inward frequency guards protect endpoints against numerical
        # truncation. Physical acceptance retains the original 90-110 window.
        return np.r_[s.slacks(p,s.BASELINE,include_H=True,margin=1),
            (p['splitting_ghz']-low-1e-6)/.01,(high-1e-6-p['splitting_ghz'])/.01,
            p['operating_frequency_ghz']-4-1e-6,8-1e-6-p['operating_frequency_ghz'],
            (d['C1_shunt_fF']-36-1e-6)/10,(128-1e-6-d['C1_shunt_fF'])/10,
            (d['C2_shunt_fF']-36-1e-6)/10,(128-1e-6-d['C2_shunt_fF'])/10]
    x=np.clip(np.array(s.x_from_point(baseline)),lower+1e-6,upper-1e-6)
    with threadpool_limits(limits=1):
        for basis,maxiter in ((FAST,45),(INTERMEDIATE,30)):
            try:
                evaluate(x,basis)
                fit=minimize(lambda z:direction*scale*evaluate(z,basis)[key],x,method='SLSQP',bounds=list(zip(lower,upper)),constraints=[dict(type='ineq',fun=lambda z,basis=basis:constraints(z,basis))],options=dict(maxiter=maxiter,ftol=1e-9,eps=2e-6))
                x=fit.x;attempts.append(dict(basis=asdict(basis),success=bool(fit.success),iterations=int(fit.nit),message=str(fit.message),endpoint=evaluate(x,basis)))
            except (ValueError,FloatingPointError) as error:
                attempts.append(dict(basis=asdict(basis),error=repr(error)))
        candidates=[p for (b,_),p in cache.items() if b==INTERMEDIATE and not s.failures(p)]
        candidates.sort(key=lambda p:direction*p[key]);chosen=None;rejections=[]
        for candidate in candidates[:3]:
            q=audit_point(candidate);d=q['device'];failed=s.failures(q,require_sign_check=True)
            if max(abs(q['loaded_ec1_mhz']-ec1),abs(q['loaded_ec2_mhz']-ec2))>1e-6:failed.append('fixed_EC')
            if not all(36<=d[k]<=128 for k in ('C1_shunt_fF','C2_shunt_fF')):failed.append('C_search_domain')
            if failed:rejections.append(failed)
            else:chosen=q;break
        if chosen is None or direction*chosen[key]>direction*baseline[key]:
            # A feasible stored baseline always supplies a valid attainable
            # endpoint. Re-audit it rather than reporting an unvalidated fit.
            q=audit_point(baseline)
            if not s.failures(q,require_sign_check=True):
                chosen=q;status='baseline_only'
            else:return dict(ec=task['ec'],objective=goal,status='no_validated_endpoint',failures=s.failures(q,require_sign_check=True),attempts=attempts,rejections=rejections,seconds=time.perf_counter()-start,completed_utc=utc())
        else:status='validated_endpoint'
    return dict(ec=task['ec'],objective=goal,status=status,metric=key,value_ghz=chosen[key],baseline_value_ghz=baseline[key],point=chosen,attempts=attempts,rejections=rejections,seconds=time.perf_counter()-start,completed_utc=utc(),global_optimality_proven=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--background',action='store_true');parser.add_argument('--pilot',type=int);a=parser.parse_args()
    if not a.run:print('Use --run to calculate; no work starts on import.');return
    OUT.mkdir(parents=True,exist_ok=True)
    if a.background:
        from Readout_ParamSpace_Robustness import process_alive
        launch=OUT/'launch.json'
        if launch.exists() and process_alive(json.loads(launch.read_text())['pid']):raise RuntimeError('Already running')
        command=[str(ROOT/'.venv/Scripts/pythonw.exe'),'-m','readout_workflow.frequency_limits','--run']
        if a.pilot:command+=['--pilot',str(a.pilot)]
        with (OUT/'run.log').open('ab') as log:p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS|subprocess.CREATE_BREAKAWAY_FROM_JOB)
        r=dict(pid=p.pid,started_utc=utc());atomic_json(launch,r);print(json.dumps(r));return
    configure();allrows=json.loads(SOURCE.read_text());rows=[r for r in allrows if r['status']=='validated_pass'];assert len(rows)==1251
    files=[Path(__file__),*[ROOT/'readout_workflow'/name for name in ('calculation.py','model.py','validation.py','direct_ec_feasibility.py')],*sorted((ROOT/'core').glob('*.py'))]
    policy=dict(source_sha256=sha256(SOURCE.read_bytes()).hexdigest(),conditions=asdict(s.CONDITIONS),fixed_EC_pairs=[r['ec1_target_mhz:ec2_target_mhz'] if 'ec1_target_mhz:ec2_target_mhz' in r else [r['ec1_target_mhz'],r['ec2_target_mhz']] for r in rows],objectives=OBJECTIVES,inward_frequency_guard_ghz=1e-6,readout_optimized=False,global_optimality_proven=False,validation_policy=POLICY,source_hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files})
    settings=OUT/'settings.json'
    if settings.exists() and json.loads(settings.read_text())!=json.loads(json.dumps(policy)):raise RuntimeError('Source/settings changed; existing endpoint data are preserved')
    atomic_json(settings,policy)
    def path(ec,goal):return OUT/f'ec_{ec[0]:g}_{ec[1]:g}_{goal}.json'
    selected=rows if not a.pilot else [rows[i] for i in np.linspace(0,len(rows)-1,min(a.pilot,len(rows)),dtype=int)]
    jobs=[dict(ec=[r['ec1_target_mhz'],r['ec2_target_mhz']],objective=goal,point=r['point']) for r in selected for goal in OBJECTIVES]
    existing=[json.loads(p.read_text()) for p in OUT.glob('ec_*.json')]
    def progress(phase='running'):atomic_json(OUT/'progress.json',dict(phase=phase,completed=len(existing),total=4*len(rows),sampled_pairs=len(selected),pilot=bool(a.pilot),updated_utc=utc(),pid=os.getpid()))
    progress();started=time.perf_counter();finished=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(solve,j):j for j in jobs if not path(j['ec'],j['objective']).exists()}
        for f in as_completed(futures):
            result=f.result();atomic_json(path(result['ec'],result['objective']),result);existing.append(result);finished.append(result);progress();print(json.dumps(dict(ec=result['ec'],objective=result['objective'],status=result['status'],value_ghz=result.get('value_ghz'),seconds=result['seconds'])),flush=True)
    if not a.pilot:atomic_json(OUT/'results.json',existing)
    progress('pilot_complete' if a.pilot else 'complete');print(json.dumps(dict(finished=len(finished),wall_seconds=time.perf_counter()-started,mean_worker_seconds=float(np.mean([r['seconds'] for r in finished])) if finished else 0)),flush=True)
if __name__=='__main__':main()
