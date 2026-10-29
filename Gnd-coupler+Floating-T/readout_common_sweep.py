"""Fixed-difference common-frequency sweeps; no differential recalibration."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from standalone_pad_readout import evaluate,policy,study
from core.transmon import Basis
from integrated_readout_range import frozen_model
from seven_readout import SevenReadout
from floating_coupler.design import ej_for_frequency


# Empirically checked against 6/12/5 on both calibrated rails across +/-100 MHz.
# Four local levels / eight pair states failed chi matching and are not defaults.
START_BASIS=dict(local_levels=6,pair_states=12,coupler_levels=3,ncut=35)
AUDIT_BASIS=dict(local_levels=6,pair_states=12,coupler_levels=5,ncut=35)


def audit_reasons(m,window,shift,shifts):
    """Escalation guards, not additional physical acceptance constraints."""
    reasons=[]
    if shift in (min(shifts),max(shifts)) or abs(shift)==min(map(abs,shifts)):
        reasons.append('center_or_endpoint')
    for key,limit,guard in [('chi_ratio_percent',1,.05),
            ('mean_chi_min_abs_mhz',1,.002),('H_min',.95,.0003),
            ('overlap_min',.95,.0001),('lambda_max',.1,.0001)]:
        if abs(m[key]-limit)<=guard:reasons.append(key)
    if min(abs(m[k]-v*1000) for k in ('logical_min_mhz','logical_max_mhz') for v in window)<.01:
        reasons.append('logical_gap')
    return reasons


def _sweep_pair(task):
    """One independent rail per spawned BLAS=1 process; atomic point progress."""
    import os,time
    from datetime import datetime,timezone
    from dataclasses import asdict
    from floating_coupler.batch import digest,_write
    from threadpoolctl import threadpool_info
    began=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    pair=task['pair'];cell=task['cell'];ro=task['readout'];case=ro['case']
    policy(case);base=frozen_model(cell['physical']);idx=[0,1] if pair=='A' else [3,4]
    anchor=np.array([cell['physical']['bare_frequencies_ghz'][base.names[i]] for i in idx])
    standalone_anchor=np.array(ro['coordinates'][:2]);device=study.Device(**ro['device'])
    standalone_basis=Basis(**task['standalone_basis']);path=Path(task['output'])/f'readout-{pair}.json'
    identity=digest(task)
    data=json.loads(path.read_text()) if path.exists() else dict(rows=[],task_id=identity)
    if data.get('task_id')!=identity:raise ValueError('Readout sweep inputs changed; use a fresh output directory')
    data.update(pair=pair,pid=os.getpid(),started_utc=started,off_ghz=cell['off_ghz'],
        integrated_anchor_bare_ghz=anchor.tolist(),standalone_anchor_bare_ghz=standalone_anchor.tolist(),
        integrated_basis=task['basis'],standalone_basis=asdict(standalone_basis),
        scope='Common shifts around each calibrated center; differential settings, hardware, opposite rail and C OFF EJ fixed. No per-point recalibration or sign-grid audit.')
    for shift in task['shifts_mhz']:
        if any(r['shift_mhz']==shift for r in data['rows']):continue
        point_began=time.perf_counter()
        x=np.array(ro['coordinates']);x[:2]=standalone_anchor+shift/1000
        p,_=evaluate(case,x,standalone_basis,device)
        ej=base.ej.copy();frequencies=anchor+shift/1000
        for i,f in zip(idx,frequencies):ej[i]=ej_for_frequency(float(f),base.ec[i],task['basis']['ncut'])
        s=SevenReadout(replace(base,ej=ej),cell['physical'],cell['off_ghz'],basis=task['basis']).solve()
        rr=s.readout(pair)
        initial=metrics(rr,ej[idx],base.ec[idx],case['dressed_split_window_ghz'])
        reasons=audit_reasons(initial,case['dressed_split_window_ghz'],shift,task['shifts_mhz']) if task['adaptive'] else []
        validation=dict(starting_basis=task['basis'],selected_basis=task['basis'],reasons=reasons,
            photons=[0,1,2],scope='Readout screening; not a Hz-level residual certificate')
        if reasons and task['basis']!=AUDIT_BASIS:
            s=SevenReadout(replace(base,ej=ej),cell['physical'],cell['off_ghz'],basis=AUDIT_BASIS).solve()
            rr=s.readout(pair)
            final=metrics(rr,ej[idx],base.ec[idx],case['dressed_split_window_ghz'])
            validation.update(selected_basis=AUDIT_BASIS,
                metric_changes={k:final[k]-initial[k] for k in initial if k!='failures'},
                starting_failures=initial['failures'])
        row=dict(shift_mhz=shift,bare_frequencies_ghz=frequencies.tolist(),
            standalone=metrics([p],[p['ej1_ghz'],p['ej2_ghz']],np.array(case['loaded_ec_ghz']),case['dressed_split_window_ghz']),
            integrated=metrics(rr,ej[idx],base.ec[idx],case['dressed_split_window_ghz']),
            raw_standalone=p,raw_integrated=rr,validation=validation,seconds=time.perf_counter()-point_began)
        data['rows'].append(row);data['rows'].sort(key=lambda r:r['shift_mhz']);_write(path,data)
        print('SWEEP',pair,shift,row['integrated']['failures'],flush=True)
    data.update(seconds=time.perf_counter()-began,finished_utc=datetime.now(timezone.utc).isoformat(),
                blas_threads=[p['num_threads'] for p in threadpool_info() if p.get('user_api')=='blas'])
    _write(path,data)
    import hashlib
    return dict(pair=pair,path=str(path),seconds=data['seconds'],pid=data['pid'],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def plot_comparison(data,path):
    """Both traces retain their own calibrated center and dressed x coordinate."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[('logical_min_mhz','Logical splitting (MHz)',None),('lambda_max','Worst dispersive ratio',.1),
        ('chi_ratio_percent','Chi mismatch (%)',1),('H_min','H score',.95),
        ('mean_chi_min_abs_mhz','Minimum |mean chi| (MHz)',1),('overlap_min','Reference overlap',.95),
        ('ej_ec_min','Minimum EJ/EC',45),('ej_max_ghz','Maximum EJ/h (GHz)',25)]
    fig,axes=plt.subplots(4,2,figsize=(10,12))
    for ax,(key,label,limit) in zip(axes.flat,panels):
        for version in ['standalone','integrated']:
            ax.plot([r[version]['operation_ghz'] for r in data['rows']],
                    [r[version][key] for r in data['rows']],label=version)
        if key=='logical_min_mhz':
            for v in ([140,160] if data['pair']=='A' else [80,100]):ax.axhline(v,color='black',ls='--',lw=.7)
        elif limit is not None:ax.axhline(limit,color='black',ls='--',lw=.7)
        if key=='chi_ratio_percent':ax.set_yscale('symlog',linthresh=.1)
        ax.set_ylabel(label);ax.set_xlabel('Mean dressed operating frequency (GHz)');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle(f"Rail {data['pair']}: frozen calibrated idle sweep")
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def run_sweeps(cell,readouts,output,shifts_mhz=None,workers=2,basis=None,make_plots=True):
    """Concurrent A/B comparison sweeps, with guarded three-level coupler screening.

    Numerical outputs are source/input-hashed and resumable. Worker times are
    separate from parent elapsed time; two workers do not imply twice the speed.
    """
    import time,hashlib
    from floating_coupler.batch import run_tasks,source_signature,digest,_write
    began=time.perf_counter();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if not cell.get('idle_calibration'):
        raise ValueError('Supply the calibrated idle cell before integrated readout sweeps')
    if workers not in (1,2):raise ValueError('Use one or two rail workers')
    adaptive=basis is None
    basis=basis or START_BASIS.copy()
    shifts=list(map(float,np.arange(-100,101,10) if shifts_mhz is None else shifts_mhz))
    if not shifts or len(set(shifts))!=len(shifts):raise ValueError('Supply distinct sweep shifts')
    tasks=[]
    for pair in ['A','B']:
        ro=readouts[pair]
        tasks.append(dict(pair=pair,cell=cell,readout=dict(case=ro['case'],coordinates=ro['coordinates'],device=ro['nominal']['device']),
            shifts_mhz=shifts,basis=basis,adaptive=adaptive,standalone_basis=dict(ncut=25,transmon_levels=8,resonator_levels=6,eigenlevels=40),output=str(out)))
    # Coupler signature already covers shared core and channel policy. Include
    # the independent readout workflow, whose equations are also dependencies.
    root=Path(__file__).resolve().parents[1]
    signature=digest(dict(coupler=source_signature(),readout=[
        (str(p.relative_to(root)),hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted((root/'readout_workflow').glob('*.py'))]))
    results=run_tasks(_sweep_pair,tasks,out/'checkpoints',workers=workers,signature=signature)
    for r in results:
        path=Path(r['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Saved sweep output changed')
    compute_elapsed=time.perf_counter()-began
    if make_plots:
        for r in results:plot_comparison(json.loads(Path(r['path']).read_text()),out/f"readout-{r['pair']}.png")
    summary=dict(workers=workers,blas_threads_per_worker=1,pairs=results,
        compute_elapsed_seconds=compute_elapsed,total_elapsed_seconds=time.perf_counter()-began,
        summed_worker_seconds=sum(r['seconds'] for r in results),
        timing_note='Worker times may overlap; resume may reuse completed jobs. Do not sum as wall time.',
        source_signature=signature)
    _write(out/'timing.json',summary)
    return summary

def metrics(rows,ej,ec,window):
    result={
      'operation_ghz':float(np.mean([p['operating_frequency_ghz'] for p in rows])),
      'logical_min_mhz':min(p['splitting_ghz']*1000 for p in rows),
      'logical_max_mhz':max(p['splitting_ghz']*1000 for p in rows),
      'lambda_max':max(p['lambda_max'] for p in rows),
      'chi_ratio_percent':100*max(p['ratio'] for p in rows),
      'mean_chi_min_abs_mhz':min(abs((p['chi_m_ghz']+p['chi_p_ghz'])/2)*1000 for p in rows),
      'chi_minus_min_mhz':min(p['chi_m_ghz']*1000 for p in rows),
      'chi_plus_min_mhz':min(p['chi_p_ghz']*1000 for p in rows),
      'H_min':min(p['hybridization'] for p in rows),
      'overlap_min':min(p['min_assignment_overlap'] for p in rows),
      'ej_max_ghz':float(max(ej)), 'ej_ec_min':float(min(np.array(ej)/ec))}
    failures=[]
    for bad,name in [(result['logical_min_mhz']<window[0]*1000 or result['logical_max_mhz']>window[1]*1000,'logical gap'),(result['lambda_max']>.1,'dispersivity'),(result['chi_ratio_percent']>1,'chi mismatch'),(result['mean_chi_min_abs_mhz']<1,'chi magnitude'),(result['H_min']<.95,'hybridization'),(result['overlap_min']<.95,'overlap'),(result['ej_max_ghz']>=25,'EJ ceiling'),(result['ej_ec_min']<=45,'EJ/EC floor')]:
        if bad:failures.append(name)
    result['failures']=failures
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--readout-A', required=True)
    parser.add_argument('--readout-B', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--shifts-mhz', type=float, nargs='+')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    run_sweeps(
        json.loads(Path(args.cell).read_text()),
        {pair: json.loads(Path(getattr(args, 'readout_'+pair)).read_text())
         for pair in ['A', 'B']},
        args.output, args.shifts_mhz, args.workers,
        make_plots=not args.no_plots)
