"""Five-transmon OFF validation of the matched common-allocation circuits.

Freeze all four nominal rail EJs and the synthesized EC/J coefficients. Compare
the original OFF bias with coupler-only recalibration in 7--10 GHz. Resonators
are excluded dynamically; their synthesized charging-matrix coefficients remain
in the five-mode model. Physical four-reference OFF is the optimization target;
logical exchange and secular ZZ are diagnostics, never substitute targets.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import brentq,minimize_scalar
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'Gnd-coupler+Floating-T')]
from integrated_readout_range import frozen_model
from floating_coupler.pairs import PairModel
from floating_coupler.on_gate import secular_logical_shifts

def metrics(pair,fc):
    d=pair.evaluate(fc)
    h=np.asarray(d['logical_exchange']['local_energy_hamiltonian_ghz'])
    return dict(fc_ghz=float(fc),physical_g_hz=d['physical_exchange']['g_mhz']*1e6,
        logical_01_10_g_hz=float(h[1,2]*1e9),logical_00_11_g_hz=float(h[0,3]*1e9),
        logical_01_10_detuning_mhz=float((h[1,1]-h[2,2])*1000),
        logical_00_11_detuning_mhz=float((h[0,0]-h[3,3])*1000),
        secular=secular_logical_shifts(d),physical=d['physical_exchange'])

def run(source,nominal,output,values=None):
    began=time.perf_counter();data=json.loads(Path(source).read_text());cell=json.loads(Path(nominal).read_text())
    idle=cell['operating_points']['idle_off'];fc0=idle['bare_f01_ghz']['C']
    request=cell.get('request',{})
    low=float(request.get('on_lower_exclusive_ghz',7.))+1e-5
    high=float(request.get('off_upper_exclusive_ghz',10.))-1e-5
    controls=np.array([idle['ej_ghz'][n] for n in ('A1','A2','C','B1','B2')])
    available=sorted(float(p) for p in data['solutions'])
    if not available:raise ValueError('No matched physical circuit in the allocation result')
    if values is None:
        values=[available[i] for i in np.unique(np.linspace(0,len(available)-1,min(10,len(available)),dtype=int))]
    if any(str(float(p)) not in data['solutions'] for p in values):
        raise ValueError('Select only accepted physical fractions from the allocation result')
    selected=[('nominal',cell['physical'])]+[(str(float(p)),data['solutions'][str(float(p))]['physical']) for p in values]
    result=dict(scope=__doc__,original_fc_ghz=fc0,search_bounds_ghz=[low,high],rail_ej_ghz=controls.tolist(),records=[])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for label,physical in selected:
        model=replace(frozen_model(physical),ej=controls.copy())
        row=dict(label=label)
        for tag,basis in [('production',dict(local_levels=6,pair_states=12,coupler_levels=5,ncut=35)),
                          ('audit',dict(local_levels=7,pair_states=15,coupler_levels=6,ncut=40))]:
            pair=PairModel(model,**basis)
            def g(fc):
                d=pair.physical_off(float(fc))['physical_exchange']
                if not d['valid']:raise ValueError('Invalid physical OFF projection')
                return d['g_mhz']*1e6
            if tag=='production':
                grid=np.linspace(low,high,25);gs=[g(f) for f in grid];roots=[]
                for a,b,ga,gb in zip(grid[:-1],grid[1:],gs[:-1],gs[1:]):
                    if ga*gb<0:roots.append(float(brentq(g,a,b,xtol=1e-11)))
                if roots:best=min(roots,key=lambda f:abs(f-fc0));status='root'
                else:
                    k=int(np.argmin(np.abs(gs)));lo=grid[max(0,k-1)];hi=grid[min(len(grid)-1,k+1)]
                    opt=minimize_scalar(lambda f:abs(g(f)),bounds=(lo,hi),method='bounded',options={'xatol':1e-10})
                    best=min([low,high,float(opt.x)],key=lambda f:abs(g(f)));status='minimum_without_root'
                row.update(search_status=status,roots_ghz=roots,scan_fc_ghz=grid.tolist(),scan_physical_g_hz=gs)
            # Audit fixed production controls: do not hide basis drift by re-rooting.
            row[tag]=dict(original=metrics(pair,fc0),retuned=metrics(pair,best))
            if tag=='audit':
                a,b=max(low,best-.002),min(high,best+.002)
                if g(a)*g(b)<0:
                    refined=brentq(g,a,b,xtol=1e-11);audit_status='root'
                else:
                    grid=np.linspace(low,high,25);gs=[g(f) for f in grid]
                    roots=[brentq(g,a,b,xtol=1e-11) for a,b,ga,gb in zip(grid[:-1],grid[1:],gs[:-1],gs[1:]) if ga*gb<0]
                    if roots:refined=min(roots,key=lambda f:abs(f-best));audit_status='root'
                    else:
                        opt=minimize_scalar(lambda f:abs(g(f)),bounds=(low,high),method='bounded')
                        refined=min([low,high,float(opt.x)],key=lambda f:abs(g(f)));audit_status='minimum_without_root'
                row[tag]['recalibrated']=dict(metrics(pair,refined),search_status=audit_status)
                if label=='nominal':nominal_five_fc=refined
                row[tag]['common_nominal_five_mode_off']=metrics(pair,nominal_five_fc)
                if label in ('nominal',selected[1][0],selected[-1][0]):
                    larger=PairModel(model,local_levels=8,pair_states=18,coupler_levels=7,ncut=45)
                    row['larger_fixed_audit_root']=metrics(larger,refined)
        result['records'].append(row)
        result['elapsed_seconds']=time.perf_counter()-began
        (output/'result.json').write_text(json.dumps(result,indent=2))
        print(label,status,'fc',best,'physical',row['production']['retuned']['physical_g_hz'],
              'logical',row['production']['retuned']['logical_01_10_g_hz'],
              row['production']['retuned']['logical_00_11_g_hz'],flush=True)
    result['status']='complete';(output/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    p.add_argument('--nominal',required=True);p.add_argument('--output',required=True)
    p.add_argument('--self-fractions',nargs='+',type=float)
    args=p.parse_args();run(args.source,args.nominal,args.output,args.self_fractions)
