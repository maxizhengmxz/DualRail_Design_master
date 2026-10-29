"""Temporary physical-exchange OFF capacitor refinement from a saved three-body geometry.

Cancellation minimizes combined rail loading with physical 0100-0010 exchange zero.
The optional historical ZZ diagnostic minimizes |idle secular logical ZZ| and permits
residual exchange. Both retain fixed loaded rail ECs and an actual resonant
5 MHz ON interaction, ON>7 GHz, ON<OFF<configured ceiling (10 GHz by default) and the agreed loading limits.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import hashlib
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grounded_five import construct
from floating_coupler.refine_capacitors import CompensatedCircuit,DesignEvaluator
from floating_coupler.batch import run_tasks,source_signature
from floating_coupler.precision import RESIDUAL_HZ
from floating_coupler.channel_policy import require_selection,off_target_definition,require_off_definition

COARSE=dict(local_levels=5,pair_states=10,coupler_levels=4,ncut=25)
FINE=dict(local_levels=6,pair_states=15,coupler_levels=6,ncut=35)
LARGE=dict(local_levels=7,pair_states=18,coupler_levels=7,ncut=40)


def prepare(request_path,selection_path):
    request=json.loads(Path(request_path).read_text())
    raw=json.loads(Path(selection_path).read_text())
    selection=(raw['variants']['frozen_physical_rail_J']['selection'] if 'variants' in raw
               else raw.get('channel_selection',raw))
    selected=require_selection(selection)
    if (selected.source_logical,selected.auxiliary)!=('10','B20') or abs(selected.target_g_mhz-5)>1e-8:
        raise ValueError('This runner implements ON CZ source10-B20 at common g=5 MHz; OFF independently nulls physical0100-0010. Use matching ON equations for another gate target')
    request['channel_selection']=selection
    request['coupler_link_caps_frozen']=False
    model,record=construct(request['seed_grounded_three_body'],request)
    return request,selection,CompensatedCircuit(request,record,model)


def constraints(row,mode):
    m=row['metrics'];p=row['physical'];x=row['x']
    eq=[(m['g_on_mhz']+5)/5,m['detuning_on_mhz']/50]
    if mode=='cancel':eq.append(m['g_off_mhz']/.001)
    elif mode=='zz_zero':eq.append(m['logical_zz_hz']/1000)
    ine=[(m['own_coupler_fraction']-.55001)*10,
         (x[5]-x[4]-1e-5),
         (.19999-m['on_maximum_unintended_ratio'])*5,
         m['on_minimum_overlap']-.8,m['off_minimum_overlap']-.8]
    # Report both the historical rail-to-C-only metric and complete branch
    # loading. The complete branch (including the direct capacitor) stays <10%.
    ine.extend((.09999-p['full_branch_budget'][n]['specified_links_loading_fraction'])*10 for n in ('A2','B1'))
    return np.array(eq),np.array(ine)


def optimize(evaluator,start,mode,output,maxiter=70,off_upper_ghz=10.,ec_lower_ghz=.15):
    origin=np.array(start,dtype=float);scale=np.array([10.,10.,1.,.2,1.,1.,.1])
    lower=np.array([.001,.001,0.,ec_lower_ghz,7.0001,7.0002,.1])
    upper=np.array([120.,160.,30.,.6,off_upper_ghz-.001,off_upper_ghz-.0001,np.inf])
    bounds=list(zip((lower-origin)/scale,(upper-origin)/scale))
    count=0;last=[];began=time.time();stable=[]
    class PrecisionStop(Exception):
        def __init__(self,y):self.y=np.array(y)
    def at(y):return evaluator(origin+scale*np.array(y))
    def objective(y):
        try:
            m=at(y)['metrics']
            if mode=='exchange_min':return (m['g_off_mhz']/.001)**2
            return (m['logical_zz_hz']/1000)**2 if mode=='zz_min' else m['rail_loading_sum']*100
        except (ValueError,np.linalg.LinAlgError):return 1e4
    def eq(y):
        try:return constraints(at(y),mode)[0]
        except (ValueError,np.linalg.LinAlgError):return np.ones(2 if mode in ('zz_min','exchange_min') else 3)*100
    def ine(y):
        try:return constraints(at(y),mode)[1]
        except (ValueError,np.linalg.LinAlgError):return np.ones(7)*-100
    def feasible(eq_values,ine_values):
        tolerances=np.array([2e-5,2e-5]+([RESIDUAL_HZ/1000] if len(eq_values)==3 else []))
        return bool(np.all(abs(eq_values)<=tolerances) and min(ine_values)>=-2e-6)
    def callback(y):
        nonlocal count
        count+=1
        try:
            row=at(y);last[:]=[row];er,ir=constraints(row,mode)
            if feasible(er,ir):
                stable.append((objective(y),row['metrics']['g_off_mhz'],row['metrics']['g_on_mhz'],row['metrics'].get('logical_zz_hz',0.)))
                stable[:]=stable[-4:]
                if len(stable)==4:
                    span=np.ptp(np.array(stable),axis=0)
                    if (span[0]<=1e-6 or mode in ('exchange_min','zz_min')) and max(span[1:3])*1e6<=RESIDUAL_HZ and (mode!='zz_min' or span[3]<=RESIDUAL_HZ):
                        raise PrecisionStop(y)
            else:stable.clear()
            if count%5==0:
                print('ITER',mode,count,json.dumps(dict(x=row['x'],metrics=row['metrics'],eq=er.tolist(),minimum_inequality=float(min(ir)))) ,flush=True)
                Path(output).write_text(json.dumps(dict(mode=mode,iteration=count,last=row),indent=2))
        except (ValueError,np.linalg.LinAlgError):pass
    try:
        fit=minimize(objective,np.zeros(7),method='SLSQP',bounds=bounds,
                     constraints=[dict(type='eq',fun=eq),dict(type='ineq',fun=ine)],
                     callback=callback,options=dict(maxiter=maxiter,ftol=1e-8,eps=2e-5))
    except PrecisionStop as stop:
        fit=SimpleNamespace(x=stop.y,success=True,message='Feasible plateau: changes below 1 Hz and numerical loading tolerance')
    row=at(fit.x);er,ir=constraints(row,mode)
    return dict(off_definition=off_target_definition('five'),mode=mode,start=list(start),success=bool(fit.success),message=str(fit.message),
                seconds=time.time()-began,evaluations=evaluator.calls,
                feasible=feasible(er,ir),residual_accuracy_hz=RESIDUAL_HZ,
                equality_residuals=er.tolist(),minimum_inequality=float(min(ir)),result=row)


def _run_start(task):
    """Picklable independent start; verify immutable request contents in worker."""
    for key in ('request','selection'):
        if hashlib.sha256(Path(task[key]).read_bytes()).hexdigest()!=task[key+'_sha256']:
            raise ValueError('Input changed during batch execution')
    _,selection,circuit=prepare(task['request'],task['selection'])
    try:
        result=optimize(DesignEvaluator(circuit,selection,task['basis']),task['start'],
                        task['mode'],task['progress'],off_upper_ghz=task['off_upper'],
                        ec_lower_ghz=task['ec_lower'])
        json.dumps(result,allow_nan=False)
        return result
    except (ValueError,np.linalg.LinAlgError) as error:
        return dict(success=False,feasible=False,mode=task['mode'],start=task['start'],
                    message=str(error),off_definition=off_target_definition('five'))



def run(request,selection,output,stage='coarse',mode='cancel',off_upper_ghz=10.,ec_lower_ghz=.15,workers=4):
    if mode=='zz-diagnostic':mode='zz'
    request_data,selected,circuit=prepare(request,selection)
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'inputs.json').exists():
        require_off_definition(json.loads((folder/'inputs.json').read_text()).get('off_definition'))
    base=request_data['seed_grounded_three_body']['parameters']
    starts=[[*base[:4],7.65,9.9,4.862],
            [base[0]*.8,base[1]*.8,base[2]*.64,base[3],7.08,9.95,4.862],
            [base[0]*.8*.8,base[1]*.8/ .8,base[2]*.64,.35,7.08,9.95,4.862]]
    provenance=dict(source_signature=source_signature(),off_definition=off_target_definition('five'),seed=request_data['seed_grounded_three_body'],selection=selected,
                    off_upper_ghz=off_upper_ghz,ec_lower_ghz=ec_lower_ghz,
                    request=request_data,basis=COARSE if stage=='coarse' else FINE,
                    objectives=dict(cancel='Minimum combined complete-branch rail loading with selected exchange zero',
                                    zz='Minimum magnitude of idle secular logical ZZ; residual exchange allowed'),
                    numerical_cap_bounds_ff=dict(A2_C=[.001,120],C_B1=[.001,160],direct=[0,30]),
                    numerical_note='Wide capacitor bounds are numerical, not additional fabrication constraints; report any active numerical boundary.')
    input_path=folder/'inputs.json'
    if input_path.exists():
        existing=json.loads(input_path.read_text())
        if any(existing.get(k)!=value for k,value in provenance.items() if k!='basis'):
            raise ValueError('Source or physics inputs changed; use a fresh output directory')
    else:input_path.write_text(json.dumps(provenance,indent=2))
    (folder/f'{stage}-inputs.json').write_text(json.dumps(provenance,indent=2))
    if stage=='coarse':
        for key in (['cancel','zz'] if mode=='both' else [mode]):
            evaluator=DesignEvaluator(circuit,selected,COARSE)
            tasks=[dict(request=str(Path(request).resolve()),selection=str(Path(selection).resolve()),
                        request_sha256=hashlib.sha256(Path(request).read_bytes()).hexdigest(),
                        selection_sha256=hashlib.sha256(Path(selection).read_bytes()).hexdigest(),
                        basis=COARSE,start=start,mode='cancel' if key=='cancel' else 'zz_zero',
                        off_upper=off_upper_ghz,ec_lower=ec_lower_ghz,
                        progress=str((folder/f'{key}-start-{i}-progress.json').resolve()))
                   for i,start in enumerate(starts)]
            attempts=run_tasks(_run_start,tasks,folder/f'{key}-checkpoints',workers=workers)
            (folder/f'{key}-attempts.json').write_text(json.dumps(attempts,indent=2))
            for i,row in enumerate(attempts):
                print('ATTEMPT',key,i,row['feasible'],(row.get('result') or {}).get('metrics',row.get('message')),flush=True)
            feasible=[r for r in attempts if r['feasible']]
            if not feasible and key=='zz':
                for start in starts:
                    row=optimize(evaluator,start,'zz_min',folder/'zz-progress.json',off_upper_ghz=off_upper_ghz,ec_lower_ghz=ec_lower_ghz)
                    attempts.append(row);(folder/'zz-attempts.json').write_text(json.dumps(attempts,indent=2))
                feasible=[r for r in attempts if r['feasible']]
            if feasible:
                rank=(lambda r:r['result']['metrics']['rail_loading_sum']) if key=='cancel' else (lambda r:(abs(r['result']['metrics']['logical_zz_hz']),r['result']['metrics']['rail_loading_sum']))
                best=min(feasible,key=rank)
                (folder/f'{key}-coarse.json').write_text(json.dumps(best,indent=2))
                print('COARSE_ACCEPTED',key,json.dumps(best['result']['metrics']),best['result']['x'],flush=True)
    else:
        for key in (['cancel','zz'] if mode=='both' else [mode]):
            path=folder/f'{key}-coarse.json'
            if not path.exists():continue
            previous=json.loads(path.read_text());require_off_definition(previous.get('off_definition'));start=previous['result']['x']
            evaluator=DesignEvaluator(circuit,selected,FINE)
            fit=optimize(evaluator,start,'cancel' if key=='cancel' else 'zz_zero',folder/f'{key}-fine-progress.json',maxiter=40,off_upper_ghz=off_upper_ghz,ec_lower_ghz=ec_lower_ghz)
            (folder/f'{key}-fine.json').write_text(json.dumps(fit,indent=2))
            print('FINE_RESULT',key,fit['feasible'],fit['result']['x'],fit['result']['metrics'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True);p.add_argument('--selection',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['coarse','fine'],default='coarse');p.add_argument('--mode',choices=['cancel','zz-diagnostic'],default='cancel')
    p.add_argument('--off-upper',type=float,default=10.);p.add_argument('--ec-lower',type=float,default=.15)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.request,a.selection,a.output,a.stage,a.mode,a.off_upper,a.ec_lower,a.workers)
