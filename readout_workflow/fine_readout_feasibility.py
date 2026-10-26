"""Fine 140-160 MHz feasibility map plus 2 fF failure diagnosis.

No readout-strength maximization or branch continuation. Feasibility stops
at its first fully audited witness. Diagnostics minimize normalized constraint
deficits under chi matching, not readout strength. All attempts are retained.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,subprocess,time
from pathlib import Path
from dataclasses import asdict,replace
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor,FIRST_COMPLETED,wait
# Also support an explicit direct-file invocation from an IDE.
if not __package__:
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from readout_workflow import calculation as s, diagnostics as diag
from readout_workflow.storage import Ledger,atomic_json,utc,status,key
from readout_workflow.validation import audit_point
from Readout_ParamSpace_Robustness import process_alive
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'readout_workflow/data/feasibility_150mhz_step1_diag2'

def initialize():
    s.CONDITIONS=replace(s.CONDITIONS,logical_frequency_ghz=(.14,.16))
    s.BASELINE=dict(s.BASELINE,logical_frequency=(.14,.16))
    original=s.seed_candidates
    def seeds(c1,c2,warm=()):
        # Original analytic C12 guesses target 100 MHz: scale only cold seeds.
        cold=original(c1,c2,())
        diverse=[]
        for x in cold:
            if not any(abs(x[0]-old[0])<.1 for old in diverse):
                y=x.copy();y[2]=min(s.CONDITIONS.c12_bounds_ff[1]-.0001,y[2]*1.5);diverse.append(y)
        warmed=[__import__('numpy').array(p.get('x',s.x_from_point(p)),float) for p in warm]
        return warmed+diverse
    s.seed_candidates=seeds

def calculate(task):
    started=time.monotonic()
    with threadpool_limits(limits=1):
        r=s.search(dict(pair=task['pair'],level='fast',warm=task['warm']))
        if r['status']=='screened_pass':
            p=audit_point(r['point'])
            r['screening_point']=r['point'];r['point']=p
            r['failures']=s.failures(p,require_sign_check=True)
            r['status']='validated_pass' if not r['failures'] else 'validation_failed'
        else:
            r['status']='no_witness_found'
    r.update(seconds=time.monotonic()-started,mode='quick_feasibility',readout_optimized=False)
    return r

def diagnose(task):
    with threadpool_limits(limits=1):
        r=diag.diagnose(task)
        if r['point'] is not None and r['status'] in ('recovered_viable','matched_reliable'):
            p=audit_point(r['point']);r['diagnostic_validation_point']=r['point'];r['point']=p
            failed=s.failures(p,require_sign_check=True);r['failures']=failed;r['normalized_deficits']=diag.deficits(p)
            if not failed:r['status']='recovered_viable';r['dominant']='recovered_viable'
            elif p['ratio']>.01 or p['min_assignment_overlap']<.95:r['status']='no_reliable_chi_match';r['dominant']='no_reliable_chi_match'
            else:
                r['status']='matched_reliable'
                names=[name for name in failed if name in r['normalized_deficits']]
                r['dominant']=max(names,key=lambda n:r['normalized_deficits'][n]) if names else 'chi_sign_stability'
            r['audit_basis']=p['basis']
            r['probe_basis']=p.get('chi_sign_probe_basis',p['basis'])
        r['readout_optimized']=False
        return r

def run(output,workers):
    initialize()
    pairs=[(c1,c2) for c2 in range(36,129) for c1 in range(c2+1,129)]
    # Self-contained fresh runs: algorithmic seeds and newly accepted neighbors.
    # Reusing an old ledger still requires its exact settings/source signature.
    files=[Path(__file__),*sorted((ROOT/'core').glob('*.py')),*[ROOT/'readout_workflow'/f for f in ('calculation.py','model.py','validation.py','storage.py','diagnostics.py')]]
    policy=dict(mode='feasibility_plus_diagnosis',conditions=asdict(s.CONDITIONS),grid_ff=dict(low=36,high=128,step=1,strict_c1_gt_c2=True),diagnostic_step_ff=2,feasibility_budget=s.SEARCH_EFFORT['fast'],diagnostic_budget=diag.DIAGNOSTIC_BUDGET,diagnostic_scales=diag.SCALES,optimize_readout=False,continuation=False,source_hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files})
    ledger=Ledger(output,policy);atomic_json(output/'settings.json',policy)
    active={r['pair_key']:r for r in ledger.rows('feasibility_normal')}
    recovered={r['pair_key']:r for r in ledger.rows('diagnostic') if r['status']=='recovered_viable'}
    diagnostic_count=len(ledger.rows('diagnostic'))
    def progress():ledger.progress(pairs_total=len(pairs),feasibility_completed=len(active),diagnostics_completed=diagnostic_count,viable_pairs=sum(r['status']=='validated_pass' for r in active.values())+len(recovered),optimize_readout=False,continuation=False)
    try:
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
            for phase,worker in (('feasibility_normal',calculate),('diagnostic',diagnose)):
                ledger.phase=phase;progress()
                targets=[p for p in pairs if phase!='diagnostic' or (active[key(p)]['status']!='validated_pass' and (p[0]-36)%2==0 and (p[1]-36)%2==0)]
                if phase=='diagnostic':atomic_json(output/'diagnostic_batch.json',dict(step_ff=2,targets=targets,policy_hash=ledger.policy_hash))
                iterator=iter(p for p in targets if not ledger.done(phase,p));pending={}
                def submit():
                    pair=next(iterator,None)
                    if pair is None:return
                    good=[r for r in active.values() if r['status']=='validated_pass']+list(recovered.values())
                    near=sorted(good,key=lambda r:(r['c1']-pair[0])**2+(r['c2']-pair[1])**2)[:3 if phase=='diagnostic' else 1]
                    if phase=='diagnostic' and active[key(pair)].get('point'):near=[active[key(pair)]]+near
                    task=dict(pair=pair,warm=[r['point'] for r in near],seed_source_ids=[r['sample_id'] for r in near])
                    pending[pool.submit(worker,task)]=pair
                for _ in range(workers):submit()
                while pending:
                    finished,_=wait(pending,timeout=45,return_when=FIRST_COMPLETED)
                    for future in finished:
                        pair=pending.pop(future);r=future.result();record=ledger.save(phase,pair,r)
                        if phase=='feasibility_normal':active[key(pair)]=record
                        else:
                            diagnostic_count+=1
                            if r['status']=='recovered_viable':recovered[key(pair)]=record
                        progress();print(json.dumps(dict(phase=phase,pair=pair,status=r['status'],feasibility=len(active),diagnostics=diagnostic_count)),flush=True);submit()
                if phase=='feasibility_normal':atomic_json(output/'feasibility_results.json',list(active.values()))
        diagnostics=ledger.rows('diagnostic');final=[recovered.get(key(p),active[key(p)]) for p in pairs]
        atomic_json(output/'diagnostic_results.json',diagnostics);atomic_json(output/'results.json',final)
        assert len(final)==4278
        assert len(diagnostics)==sum(active[key(p)]['status']!='validated_pass' and (p[0]-36)%2==0 and (p[1]-36)%2==0 for p in pairs)
        ledger.phase='feasibility_diagnosis_complete';progress()
    finally:ledger.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--background',action='store_true');parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--workers',type=int,default=4);a=parser.parse_args();a.output=a.output.resolve()
    if not a.run:print(json.dumps(status(a.output),indent=2));return
    if not a.background:run(a.output,a.workers);return
    a.output.mkdir(parents=True,exist_ok=True);launch=a.output/'launch.json'
    if launch.exists() and process_alive(json.loads(launch.read_text())['pid']):raise RuntimeError('Already running')
    with (a.output/'run.log').open('ab') as log:
        p=subprocess.Popen([str(ROOT/'.venv/Scripts/pythonw.exe'),'-m','readout_workflow.fine_readout_feasibility','--run','--output',str(a.output),'--workers',str(a.workers)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS|subprocess.CREATE_BREAKAWAY_FROM_JOB)
    record=dict(pid=p.pid,started_utc=utc(),windowless=True,optimize_readout=False);atomic_json(launch,record);print(json.dumps(record))
if __name__=='__main__':main()
