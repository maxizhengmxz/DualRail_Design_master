"""Static B differential-frequency sensitivity at a frozen calibrated CZ point.

delta_B=f01(B1)-f01(B2); f01(B1)=mean+delta/2 and f01(B2)=mean-delta/2.
Mean bare frequency, coupler ON, duration, A EJs, loaded ECs, all charge
coefficients and resonators remain fixed. No conditional-phase recalibration.
Use the same pointwise coupler-dressed static logical convention as the saved
gate, including standard local-Z phase removal; this is not an idle/ramp test.
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
from final_seven_validation import Gate
from integrated_readout_range import frozen_model
from floating_coupler.paired_design import selected_channel
from floating_coupler.design import ej_for_frequency
from studies.gate_sensitivity import PRODUCTION,AUDIT,write,gate_point

CONTEXT=None


def selected_point_record(cell,point,source):
    """Record saved physical hardware and selected controls without changing J."""
    model=frozen_model(cell['physical'])
    gate_ej=model.ej.copy()
    for i in (3,4):gate_ej[i]=ej_for_frequency(point['fb_ghz'],model.ec[i],40)
    gate_ej[2]=ej_for_frequency(point['fc_ghz'],model.ec[2],40)
    return dict(schema='nominal-static-cz-point-v1',
        design_id='static_gate_sensitivity_point',
        model_kind='Saved physical circuit at specified static gate controls',
        nominal_design_source=str(source),nominal_circuit_provenance=cell,
        transmon_model=dict(mode_order=list(model.names),loaded_ec_ghz=model.ec.tolist(),
            charge_J_ghz=model.j.tolist(),idle_EJ_ghz=model.ej.tolist(),gate_EJ_ghz=gate_ej.tolist()),
        resonator_parameters=cell['physical']['resonator_parameters'],
        resonator_K12_per_ff=cell['physical']['K_per_ff'][5][6],selected_static_gate=point)


def initialize(context):
    global CONTEXT
    CONTEXT=context


def evaluate(job):
    delta,basis_name=job;began=time.perf_counter();cell=CONTEXT['cell'];point=CONTEXT['point']
    basis=PRODUCTION if basis_name=='production' else AUDIT
    model=frozen_model(cell['physical'])
    cal=Gate(cell,basis,b_detuning_ghz=float(delta)/1000)
    metrics=cal.at(point['fc_ghz'],point['fb_ghz'],point['time_ns'])
    detail=cal.spectrum(point['fc_ghz'],point['fb_ghz'])[3]
    channel=selected_channel(detail)
    b1=point['fb_ghz']+float(delta)/2000;b2=point['fb_ghz']-float(delta)/2000
    if abs((b1+b2)/2-point['fb_ghz'])>1e-12:raise RuntimeError('Mean bare frequency changed')
    return dict(delta_B_mhz=float(delta),basis_name=basis_name,basis=basis,
        B1_bare_frequency_ghz=b1,B2_bare_frequency_ghz=b2,
        B1_EJ_ghz=ej_for_frequency(b1,model.ec[3],basis['ncut']),
        B2_EJ_ghz=ej_for_frequency(b2,model.ec[4],basis['ncut']),
        raw_infidelity=1-metrics['raw_cz_fidelity'],selected_infidelity=1-metrics['postselected_cz_fidelity'],
        selected_channel=channel,metrics=metrics,seconds=time.perf_counter()-began)


def tolerance_interval(rows,threshold=.999):
    """Interpolate the connected central interval; do not merge distant lobes."""
    rows=sorted(rows,key=lambda row:row['delta_B_mhz']);center=min(range(len(rows)),key=lambda i:abs(rows[i]['delta_B_mhz']))
    if rows[center]['metrics']['postselected_cz_fidelity']<threshold:return None
    interval={}
    for sign,label in [(-1,'lower_mhz'),(1,'upper_mhz')]:
        index=center
        while 0<=index+sign<len(rows):
            adjacent=index+sign;a=rows[index];b=rows[adjacent]
            fa=a['metrics']['postselected_cz_fidelity'];fb=b['metrics']['postselected_cz_fidelity']
            if fb<threshold:
                interval[label]=a['delta_B_mhz']+(threshold-fa)/(fb-fa)*(b['delta_B_mhz']-a['delta_B_mhz'])
                break
            index=adjacent
        else:interval[label]=None
    return interval


def run(source,calibration_file,output,deltas=None):
    began=time.perf_counter();cell=json.loads(Path(source).read_text(encoding='utf-8'))
    calibration=json.loads(Path(calibration_file).read_text(encoding='utf-8')) if calibration_file else None
    point=gate_point(cell,calibration);output=Path(output)
    context=dict(cell=cell,point=point)
    result=dict(schema='nominal-b-detuning-v1',input=str(source),
        calibration=str(calibration_file),scope=__doc__,frozen_gate_point=point,rows=[],audit=[])
    snapshot=selected_point_record(cell,point,source)
    write(output/'selected_static_point.json',snapshot)
    grid=sorted(set(round(float(delta),8) for delta in (deltas if deltas is not None else
        np.r_[np.arange(-20,20.01,2),np.arange(-2,2.001,.1)])))
    if 0. not in grid or not np.isfinite(grid).all():raise ValueError('Detuning grid must be finite and include zero')
    path=output/'result.json';write(path,result)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,initargs=(context,)) as pool:
        futures=[pool.submit(evaluate,(delta,'production')) for delta in grid]
        for future in as_completed(futures):
            row=future.result();result['rows'].append(row)
            result['rows'].sort(key=lambda r:r['delta_B_mhz']);write(path,result)
            if len(result['rows'])%10==0:print('SWEEP',len(result['rows']),len(grid),flush=True)
        interval=tolerance_interval(result['rows']);result['selected_fidelity_99p9_interval_interpolated']=interval
        result['selected_fidelity_99_interval_interpolated']=tolerance_interval(result['rows'],.99)
        checks=[0.]
        if interval:
            checks.extend(round(value,6) for value in interval.values() if value is not None)
        # Same-control basis checks: never retune mean frequency or duration.
        futures=[pool.submit(evaluate,(delta,'audit')) for delta in checks]
        for future in as_completed(futures):
            row=future.result();result['audit'].append(row);write(path,result)
    baseline=next(r for r in result['rows'] if r['delta_B_mhz']==0.)
    zero_audit=next(r for r in result['audit'] if r['delta_B_mhz']==0.)
    result['zero_detuning_reproduction_changes']={key:zero_audit['metrics'][key]-point[key]
        for key in ('raw_cz_fidelity','postselected_cz_fidelity','logical_survival','phase_error_rad') if key in point}
    result['reference_reproduction_within_probability_precision']=all(abs(value)<=1e-4 for key,value
        in result['zero_detuning_reproduction_changes'].items() if key!='phase_error_rad')
    result['zero_detuning_basis_changes']={key:zero_audit['metrics'][key]-baseline['metrics'][key]
        for key in ('raw_cz_fidelity','postselected_cz_fidelity','logical_survival')}
    result.update(status='complete',seconds=time.perf_counter()-began);write(path,result)
    fields=['delta_B_mhz','B1_bare_frequency_ghz','B2_bare_frequency_ghz','raw_fidelity','selected_fidelity',
            'survival','phase_error_rad','intended_g_mhz','intended_detuning_mhz']
    with (output/'sweep.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in result['rows']:
            metrics=r['metrics'];writer.writerow(dict(delta_B_mhz=r['delta_B_mhz'],
                B1_bare_frequency_ghz=r['B1_bare_frequency_ghz'],B2_bare_frequency_ghz=r['B2_bare_frequency_ghz'],
                raw_fidelity=metrics['raw_cz_fidelity'],selected_fidelity=metrics['postselected_cz_fidelity'],
                survival=metrics['logical_survival'],phase_error_rad=metrics['phase_error_rad'],
                intended_g_mhz=r['selected_channel']['g_mhz'],intended_detuning_mhz=r['selected_channel']['detuning_mhz']))
    plot(result,output)
    print(json.dumps(dict(seconds=result['seconds'],points=len(grid),selected_99p9_interval=interval,
                         basis_changes=result['zero_detuning_basis_changes']),indent=2))


def plot(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=result['rows'];x=np.array([r['delta_B_mhz'] for r in rows])
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,limit in zip(axes[0],(20,2)):
        for key,label in [('raw_infidelity','Raw error'),('selected_infidelity','Postselected error')]:
            ax.semilogy(x,[max(r[key],1e-9) for r in rows],'o-',ms=2,label=label)
        ax.axhline(.001,color='grey',ls=':',label='99.9% fidelity')
        ax.axhline(.0001,color='red',ls='--',lw=.8,label='Probability accuracy target')
        ax.set(xlim=(-limit,limit),ylabel='1 - fidelity',xlabel=r'$\delta_B=f_{B1}-f_{B2}$ (MHz)')
        ax.legend(fontsize=8)
    axes[1,0].plot(x,[r['metrics']['phase_error_rad'] for r in rows]);axes[1,0].axhline(0,color='grey',ls=':')
    axes[1,0].set(xlabel=r'$\delta_B$ (MHz)',ylabel='Conditional phase error (rad)')
    axes[1,1].plot(x,[r['selected_channel']['g_mhz'] for r in rows],label='Intended g')
    axes[1,1].plot(x,[r['selected_channel']['detuning_mhz'] for r in rows],label='Intended detuning')
    axes[1,1].set(xlabel=r'$\delta_B$ (MHz)',ylabel='MHz');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    point=result['frozen_gate_point']
    fig.suptitle(f"B differential-frequency sensitivity: mean {point['fb_ghz']:.8f} GHz, "
                 f"C {point['fc_ghz']:.2f} GHz, {point['time_ns']:.3f} ns; no recalibration")
    fig.tight_layout();fig.savefig(output/'B-detuning-sensitivity.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True)
    parser.add_argument('--calibration',help='Saved calibrated gate; absent uses the input static point')
    parser.add_argument('--detunings-mhz',nargs='+',type=float)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    run(args.input,args.calibration,args.output,args.detunings_mhz)
