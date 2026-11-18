"""B differential-detuning sweep, common frequency recalibrated to phase pi.

Continue the same nearby selected-channel static gate. Hold duration, coupler
ON, A controls, all ECs, all charge interactions and resonators fixed. Only B
differential bare frequency (prescribed) and its mean (phase correction) vary.
No gate-time/coupler optimization or resonance reselection. Preserve standard
pointwise dressed logical references and local-Z-corrected gate metrics.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'Gnd-coupler+Floating-T')]
from floating_coupler.paired_design import selected_channel
from floating_coupler.precision import correct_gate_phase
from studies.gate_sensitivity import PRODUCTION,AUDIT,write,evaluator,gate_point

CONTEXT=None


def initialize(context):
    global CONTEXT
    CONTEXT=context


def gate(delta,basis):
    return evaluator(CONTEXT['cell'],basis,b_detuning_ghz=float(delta)/1000)


def entry(delta,cal,row,calls):
    point=CONTEXT['point'];mean=row['fb_ghz'];detail=cal.spectrum(point['fc_ghz'],mean)[3]
    if abs(row['time_ns']-point['time_ns'])>1e-12 or row['fc_ghz']!=point['fc_ghz']:
        raise RuntimeError('A frozen gate control changed')
    return dict(delta_B_mhz=float(delta),mean_B_frequency_ghz=mean,
        mean_shift_mhz=(mean-point['fb_ghz'])*1000,
        B1_bare_frequency_ghz=mean+delta/2000,B2_bare_frequency_ghz=mean-delta/2000,
        phase_calibration_calls=calls,phase_pi=bool(abs(row['phase_error_rad'])<1e-7),
        raw_infidelity=1-row['raw_cz_fidelity'],selected_infidelity=1-row['postselected_cz_fidelity'],
        selected_channel=selected_channel(detail),metrics=row)


def branch_worker(sign,grid,output):
    """Warm Newton/secant continuation, always verifying the actual phase."""
    began=time.perf_counter();point=CONTEXT['point'];fc=point['fc_ghz'];duration=point['time_ns']
    seed=CONTEXT['production_seed'];center=seed['fb_ghz'];zero=gate(0.,PRODUCTION)
    baseline=zero.at(fc,center,duration)
    if abs(baseline['phase_error_rad'])>=1e-7:
        baseline=correct_gate_phase(lambda f:zero.at(fc,float(f),duration),baseline,
                                   bounds=CONTEXT['mean_bounds_ghz'],max_evaluations=8)
        center=baseline['fb_ghz']
    adjacent=zero.at(fc,center+1e-6,duration)
    slope=(adjacent['phase_error_rad']-baseline['phase_error_rad'])/1e-6
    if not np.isfinite(slope) or abs(slope)<1e-8:raise RuntimeError('No reliable common-frequency phase slope')
    history=[entry(0.,zero,baseline,1)];steps=[]
    for delta in sorted([v for v in grid if v*sign>0],key=abs):
        cal=gate(delta,PRODUCTION)
        if len(history)>=2:
            a,b=history[-2:]
            prediction=b['mean_B_frequency_ghz']+(delta-b['delta_B_mhz'])*(b['mean_B_frequency_ghz']-a['mean_B_frequency_ghz'])/(b['delta_B_mhz']-a['delta_B_mhz'])
        else:prediction=center
        mean=float(prediction);last=None;best=None;attempts=[]
        for number in range(7):
            row=cal.at(fc,mean,duration);error=row['phase_error_rad']
            attempts.append(dict(mean_ghz=mean,phase_error_rad=error))
            if best is None or abs(error)<abs(best['phase_error_rad']):best=row
            if abs(error)<1e-7:break
            if last is not None:
                previous_mean,previous_error=last
                if abs(mean-previous_mean)>1e-12 and abs(error-previous_error)<np.pi:
                    measured=(error-previous_error)/(mean-previous_mean)
                    if np.isfinite(measured) and abs(measured)>1e-8:slope=measured
            last=(mean,error)
            move=float(np.clip(-error/slope,-.002,.002))
            # Explicit mean-frequency search bounds, not assumed SQUID hardware.
            mean=float(np.clip(mean+move,*CONTEXT['mean_bounds_ghz']))
        result=entry(delta,cal,best,len(attempts));result['attempts']=attempts
        if not result['phase_pi']:
            result['status']='Closest phase after finite local calibration; not a phase-pi gate'
        else:
            result['status']='Phase calibrated'
            history.append(result)
        steps.append(result)
        write(Path(output)/('positive.json' if sign>0 else 'negative.json'),
              dict(rows=steps,seconds=time.perf_counter()-began))
        if len(steps)%5==0:print('BRANCH',sign,len(steps),delta,best['postselected_cz_fidelity'],flush=True)
    return dict(rows=steps,baseline=history[0],seconds=time.perf_counter()-began)


def audit_worker(row):
    delta=row['delta_B_mhz'];cal=gate(delta,AUDIT);point=CONTEXT['point']
    fc=point['fc_ghz'];duration=point['time_ns'];before=cal.at(fc,row['mean_B_frequency_ghz'],duration)
    corrected=correct_gate_phase(lambda f:cal.at(fc,float(f),duration),row['metrics'],
        bounds=(row['mean_B_frequency_ghz']-.002,row['mean_B_frequency_ghz']+.002),max_evaluations=8)
    result=entry(delta,cal,corrected,corrected['phase_correction_evaluations'])
    result['fixed_controls']=before
    result['fixed_probability_changes']={k:before[k]-row['metrics'][k] for k in
        ('raw_cz_fidelity','postselected_cz_fidelity','logical_survival')}
    result['phase_corrected_probability_changes']={k:corrected[k]-row['metrics'][k] for k in
        ('raw_cz_fidelity','postselected_cz_fidelity','logical_survival')}
    return result


def run(fixed_sweep,calibration_file,output,mean_bounds=None):
    began=time.perf_counter();fixed=json.loads(Path(fixed_sweep).read_text())
    if fixed.get('schema')!='nominal-b-detuning-v1':
        raise ValueError('Use a current nominal B-detuning sweep; historical intervention records are unsupported')
    calibration=json.loads(Path(calibration_file).read_text()) if calibration_file else None
    cell=json.loads(Path(fixed['input']).read_text())
    point=gate_point(cell,calibration) if calibration else fixed['frozen_gate_point'];output=Path(output)
    if any(abs(point[k]-fixed['frozen_gate_point'][k])>1e-12 for k in ('fc_ghz','fb_ghz','time_ns')):
        raise ValueError('Fixed-mean comparison and selected calibration do not match')
    bounds=tuple(mean_bounds or (max(.1,point['fb_ghz']-.2),point['fb_ghz']+.2))
    if not 0<bounds[0]<point['fb_ghz']<bounds[1]:raise ValueError('Mean-frequency bounds must contain the reference bias')
    seed=calibration['best_production'] if calibration and 'best_production' in calibration else next(
        row['metrics'] for row in fixed['rows'] if row['delta_B_mhz']==0.)
    context=dict(cell=cell,point=point,production_seed=seed,mean_bounds_ghz=bounds)
    grid=sorted(r['delta_B_mhz'] for r in fixed['rows'])
    result=dict(scope=__doc__,fixed_mean_comparison=str(fixed_sweep),calibration=str(calibration_file),
        frozen_gate_point=point,mean_frequency_bounds_ghz=bounds,rows=[],audit=[])
    path=output/'result.json';write(path,result)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize,initargs=(context,)) as pool:
        futures=[pool.submit(branch_worker,sign,grid,str(output)) for sign in (-1,1)]
        baselines=[]
        for future in as_completed(futures):
            branch=future.result();result['rows'].extend(branch['rows']);baselines.append(branch['baseline'])
            write(path,result)
        result['rows'].append(baselines[0]);result['rows'].sort(key=lambda r:r['delta_B_mhz'])
        # Modest basis checks at center and the worst selected-fidelity point
        # on each side. Do not restart a high-basis gate optimizer.
        checks=[baselines[0]]
        for sign in (-1,1):
            candidates=[r for r in result['rows'] if r['delta_B_mhz']*sign>0 and r['phase_pi']]
            if candidates:checks.append(min(candidates,key=lambda r:r['metrics']['postselected_cz_fidelity']))
        write(path,result)
        futures=[pool.submit(audit_worker,row) for row in checks]
        for future in as_completed(futures):result['audit'].append(future.result());write(path,result)
    fields=['delta_B_mhz','mean_shift_mhz','mean_B_frequency_ghz','B1_bare_frequency_ghz','B2_bare_frequency_ghz',
            'raw_fidelity','selected_fidelity','survival','phase_error_rad','intended_g_mhz','intended_detuning_mhz','phase_pi']
    with (output/'sweep.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in result['rows']:
            m=r['metrics'];writer.writerow({**{key:r[key] for key in fields if key in r},
                'raw_fidelity':m['raw_cz_fidelity'],'selected_fidelity':m['postselected_cz_fidelity'],
                'survival':m['logical_survival'],'phase_error_rad':m['phase_error_rad'],
                'intended_g_mhz':r['selected_channel']['g_mhz'],'intended_detuning_mhz':r['selected_channel']['detuning_mhz']})
    result.update(status='complete',seconds=time.perf_counter()-began);write(path,result)
    plot(result,fixed,output)
    print(json.dumps(dict(seconds=result['seconds'],points=len(result['rows']),
        phase_pi_count=sum(r['phase_pi'] for r in result['rows']),
        samples=[{key:r[key] for key in ('delta_B_mhz','mean_shift_mhz','selected_infidelity','raw_infidelity')}
                 for r in result['rows'] if r['delta_B_mhz'] in (-20,-10,-2,-1,0,1,2,10,20)]),indent=2))


def plot(result,fixed,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,limit in zip(axes[0],(20,2)):
        for data,style,prefix in ((fixed['rows'],'--','Fixed mean'),(result['rows'],'-','Mean recalibrated')):
            for key,label,color in [('selected_infidelity','selected','tab:orange'),('raw_infidelity','raw','tab:blue')]:
                ax.semilogy([r['delta_B_mhz'] for r in data],[max(r[key],1e-9) for r in data],
                    style,color=color,label=prefix+' | '+label)
        ax.axhline(1e-3,color='grey',ls=':',label='99.9% fidelity')
        ax.axhline(1e-4,color='red',ls=':',label='Probability accuracy target')
        ax.set(xlim=(-limit,limit),xlabel=r'$\delta_B=f_{B1}-f_{B2}$ (MHz)',ylabel='1 - fidelity')
        ax.legend(fontsize=7)
    rows=result['rows'];x=[r['delta_B_mhz'] for r in rows]
    axes[1,0].plot(x,[r['mean_shift_mhz'] for r in rows]);axes[1,0].set(
        xlabel=r'$\delta_B$ (MHz)',ylabel='Recalibrated mean-frequency shift (MHz)')
    for key,label in [('raw_cz_fidelity','Raw'),('postselected_cz_fidelity','Postselected'),('logical_survival','Survival')]:
        axes[1,1].plot(x,[r['metrics'][key]*100 for r in rows],label=label)
    axes[1,1].set(xlabel=r'$\delta_B$ (MHz)',ylabel='Percent after phase calibration');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    point=result['frozen_gate_point'];fig.suptitle(f"B detuning with mean-frequency phase correction: C {point['fc_ghz']:.2f} GHz, "
        f"{point['time_ns']:.3f} ns fixed; same Hamiltonian as fixed-mean comparison")
    fig.tight_layout();fig.savefig(output/'B-detuning-mean-recalibrated.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fixed-sweep',required=True)
    parser.add_argument('--calibration',help='Optional matching calibration provenance')
    parser.add_argument('--mean-bounds-ghz',nargs=2,type=float)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.fixed_sweep,args.calibration,args.output,args.mean_bounds_ghz)
