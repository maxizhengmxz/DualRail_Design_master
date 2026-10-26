"""Bounded, reproducible validation/performance benchmark; no large sweep.

Compare identical saved inputs under legacy complex/full, real/full, and
adaptive validation, with four workers and one BLAS thread per worker.
The legacy implementation is reconstructed in worker memory only. Existing
calculation results are read-only; benchmark results have their own directory.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,inspect,textwrap
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'readout_workflow/data/validation_speed_test'

def worker(task):
    from readout_workflow import model as module, calculation as s
    from readout_workflow.validation import audit_point
    s.CONDITIONS=replace(s.CONDITIONS,logical_frequency_ghz=(.14,.16));s.BASELINE=dict(s.BASELINE,logical_frequency=(.14,.16))
    p=task['point'];mode=task['mode'];original=module.ReadoutModel.evaluate
    if mode=='legacy_complex_full':
        source=textwrap.dedent(inspect.getsource(original))
        source=source.replace('qr = self.qr_zpf * (a.T + a)','qr = 1j * self.qr_zpf * (a.T - a)')
        source=source.replace('ham = np.kron(pair, ir)','ham = np.kron(pair, ir).astype(complex)')
        namespace={};exec(source,module.__dict__,namespace);module.ReadoutModel.evaluate=namespace['evaluate']
    try:
        with threadpool_limits(limits=1):
            started=time.perf_counter();q=audit_point(p,force_full=mode!='adaptive');elapsed=time.perf_counter()-started
        oldprobes={(x['ej1_fraction'],x['ej2_fraction']):x for x in p['chi_sign_stability']['probes']}
        errors=[abs(x[k]-oldprobes[(x['ej1_fraction'],x['ej2_fraction'])][k])*1e6 for x in q.get('chi_sign_stability',{}).get('probes',[]) for k in ('chi_m_ghz','chi_p_ghz')]
        return dict(case=task['case'],mode=mode,seconds=elapsed,failures=s.failures(q,require_sign_check=True),nominal_chi_error_khz=max(abs(q[k]-p[k])*1e6 for k in ('chi_m_ghz','chi_p_ghz')),max_probe_chi_error_khz=max(errors,default=0),representative=q['validation']['representative'],nominal_basis=q['basis'],probe_basis=q.get('chi_sign_probe_basis'),probe_count=q.get('chi_sign_stability',{}).get('probe_count'))
    finally:module.ReadoutModel.evaluate=original

def search_worker(task):
    from readout_workflow.direct_ec_feasibility import solve
    return solve(task)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    saved=json.loads((ROOT/'readout_workflow/data/direct_ec_step5/results.json').read_text())
    subset=sorted(set(np.linspace(0,len(saved)-1,12,dtype=int)))
    summary={};allresults=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for mode,indices in [('legacy_complex_full',subset),('real_full',subset),('adaptive',range(len(saved)))]:
            start=time.perf_counter();futures={pool.submit(worker,dict(mode=mode,case=int(i),point=saved[i]['point'])):i for i in indices};results=[]
            for f in as_completed(futures):
                r=f.result();results.append(r);print(json.dumps(dict(mode=mode,finished=len(results),total=len(futures))),flush=True)
            wall=time.perf_counter()-start;allresults.extend(results)
            summary[mode]=dict(cases=len(results),wall_seconds=wall,wall_seconds_per_case=wall/len(results),mean_worker_seconds=float(np.mean([r['seconds'] for r in results])),passed=sum(not r['failures'] for r in results),representative_full_checks=sum(r['representative'] for r in results),max_nominal_chi_error_khz=max(r['nominal_chi_error_khz'] for r in results),max_probe_chi_error_khz=max(r['max_probe_chi_error_khz'] for r in results))
            (OUT/f'{mode}.json').write_text(json.dumps(results,indent=2));(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
        # Re-run the previous 48-point feasibility test with identical warm
        # starts; this exercises search plus adaptive validation end to end.
        prior=json.loads((ROOT/'readout_workflow/data/feasibility_150mhz_step1_diag2/results.json').read_text());good=[r['point'] for r in prior if r['status'] in ('validated_pass','recovered_viable')];xy=np.array([[p['loaded_ec1_mhz'],p['loaded_ec2_mhz']] for p in good])
        start=time.perf_counter();futures=[]
        for r in saved:
            e=[r['ec1_target_mhz'],r['ec2_target_mhz']];near=np.argsort(np.sum((xy-e)**2,axis=1))[:3];futures.append(pool.submit(search_worker,dict(ec=e,warm=[good[i] for i in near])))
        results=[]
        for f in as_completed(futures):
            results.append(f.result());print(json.dumps(dict(mode='new_end_to_end_search',finished=len(results),total=len(futures))),flush=True)
        summary['new_end_to_end_search']=dict(cases=len(results),wall_seconds=time.perf_counter()-start,mean_worker_seconds=float(np.mean([r['seconds'] for r in results])),passed=sum(r['status']=='validated_pass' for r in results))
        oldprogress=json.loads((ROOT/'readout_workflow/data/direct_ec_step5/progress.json').read_text())
        from datetime import datetime
        oldfirst=datetime.fromisoformat(min(r['completed_utc'] for r in saved));oldlast=datetime.fromisoformat(max(r['completed_utc'] for r in saved))
        summary['historical_search_reference']=dict(mean_worker_seconds=float(np.mean([r['seconds'] for r in saved])),completion_span_seconds=(oldlast-oldfirst).total_seconds(),note='Historical run: same 48 targets and four workers, but different time/system load; direct fixed-input benchmark above is the controlled comparison.')
        (OUT/'new_search_results.json').write_text(json.dumps(results,indent=2));(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':run()
