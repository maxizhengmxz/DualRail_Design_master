"""Matched physical self/ground allocation on chosen floating rail transmons.

p=Cs/(Cs+Cg/2), with equal pad grounds. At each p prescribe the SAME p on
A1,A2,B1,B2, a selected pair, or A2 only. Nonselected rails retain their nominal
physical fractions. Match seven charging diagonals and nine intended off-diagonals;
retain saved loading limits and positivity. Minimize absolute remaining local
exchange magnitudes among feasible solutions, with component movement as a
tie-break. A finite multi-start local search does not certify infeasibility or
global optimality. No gate, OFF or readout optimization runs in this study.
"""
from copy import deepcopy
from functools import lru_cache
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares,minimize

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'Gnd-coupler+Floating-T')]
from core.transmon import Basis,transmon_spectrum
from floating_coupler.model import SCALE
from studies.capacitance_model import circuit,coefficients,synchronize_physical
from studies.capacitance_model import INTENDED,self_fraction

RAILS=('A1','A2','B1','B2')


def run(source,output,values=None,scope='all'):
    began=time.perf_counter();nominal=json.loads(Path(source).read_text());ph=nominal['physical']
    base=circuit(ph);k0=base['K'];embedding=base['embedding'];nodes=base['node_order'];names=base['modal_order']
    targets=ph['loaded_ec_ghz'];ceff={n:SCALE/e for n,e in targets.items()}
    selected={'all':RAILS,'A':('A1','A2'),'B':('B1','B2'),'A2':('A2',)}[scope]
    nominal_fractions={n:self_fraction(ph,n) for n in RAILS}
    n01={n:abs(transmon_spectrum(nominal['operating_points']['idle_off']['ej_ghz'][n],targets[n],
                               Basis(ncut=35,transmon_levels=5))[1][0,1]) for n in ph['pad_pairs']}
    reference=coefficients(ph,k0,n01)
    def direction(a,b=None):
        d=np.zeros(len(nodes))
        if a in nodes:d[nodes.index(a)]=1
        if b in nodes:d[nodes.index(b)]-=1
        return np.outer(d,d)
    keys=[('self',n) for n in names[:5]]+[('link',i) for i in range(len(ph['physical_links']))]+[('resonator',r) for r in ph['resonator_self_capacitances_ff']]
    nominal_caps=np.array([ph['pad_mutual_capacitances_ff'][key] if kind=='self' else
        ph['physical_links'][key]['capacitance_ff'] if kind=='link' else ph['resonator_self_capacitances_ff'][key]
        for kind,key in keys])
    intended=[(names.index(a),names.index(b)) for a,b in INTENDED]
    pairs=[(i,i) for i in range(7)]+intended;ii,jj=np.array(pairs).T;scales=abs(k0[ii,jj])
    remaining=[(i,j) for i in range(7) for j in range(i+1,7) if (i,j) not in intended and (j,i) not in intended]
    ri,rj=np.array(remaining).T
    weights=np.array([reference[f'g01_{names[i]}_{names[j]}_mhz']/k0[i,j] for i,j in remaining])
    own_min=1-nominal['request']['coupler_loading_max'];rail_limit=nominal['request']['rail_loading_per_coupler_max']
    c_index=keys.index(('self','C'));removed=[i for i,(kind,key) in enumerate(keys) if kind=='link' and key<3]
    grid=list(np.round(np.linspace(.1,.9,81),8)) if values is None else sorted(set(values))
    if any(not .1<=p<=.9 for p in grid):raise ValueError('Self fractions must lie in [0.1,0.9]')
    # Start near equal self/ground allocation, then continue both ways; retain
    # fresh starts independently so one failed continuation cannot poison a scan.
    center=min(grid,key=lambda p:abs(p-.5));order=[center]+sorted([p for p in grid if p<center],reverse=True)+sorted(p for p in grid if p>center)
    records=[];solutions={};warm=None;warm_p=None;output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for p in order:
        fractions={n:(p if n in selected else nominal_fractions[n]) for n in RAILS}
        t0=time.perf_counter();directions=[]
        for kind,key in keys:
            if kind=='self':
                pair=ph['pad_pairs'][key];matrix=direction(*pair)
                if key!='C':matrix=matrix+2*(1-fractions[key])/fractions[key]*sum(direction(pad) for pad in pair)
            elif kind=='link':
                edge=ph['physical_links'][key];matrix=direction(edge['from'],edge['to'])
            else:matrix=direction(key)
            directions.append(matrix)
        matrices=np.array(directions)*nominal_caps[:,None,None]
        @lru_cache(maxsize=2)
        def evaluate(key):
            x=np.array(key);factors=np.exp(x);nodal=np.einsum('i,ijk->jk',factors,matrices)
            response=np.linalg.solve(nodal,embedding);k=embedding.T@response
            derivatives=-np.einsum('ai,nab,bj->nij',response,matrices,response)*factors[:,None,None]
            return k,(k[ii,jj]-k0[ii,jj])/scales,derivatives[:,ii,jj].T/scales[:,None],derivatives,nodal
        def residual(x):return evaluate(tuple(x))[1]
        def jacobian(x):return evaluate(tuple(x))[2]
        def budget(x):
            k,_,_,_,nodal=evaluate(tuple(x));factors=np.exp(x)
            removed_matrix=np.einsum('i,ijk->jk',factors[removed],matrices[removed])
            without=embedding.T@np.linalg.solve(nodal-removed_matrix,embedding)
            return dict(coupler_own_fraction=float(nominal_caps[c_index]*factors[c_index]*k[2,2]),
                A2_full_branch_loading=float(1-k[1,1]/without[1,1]),B1_full_branch_loading=float(1-k[3,3]/without[3,3]))
        def guards(x):
            b=budget(x);return np.array([b['coupler_own_fraction']-own_min,
                rail_limit-b['A2_full_branch_loading'],rail_limit-b['B1_full_branch_loading']])
        lower=np.full(len(keys),1e-6);upper=np.full(len(keys),1e5)
        guess=nominal_caps.copy()
        for i,(kind,key) in enumerate(keys):
            if kind=='self':
                upper[i]=ceff[key]*(fractions[key] if key!='C' else 1.)
                if key!='C':guess[i]=.93*fractions[key]*ceff[key]
                else:lower[i]=own_min*ceff[key]
            elif kind=='resonator':upper[i]=1/k0[names.index(key),names.index(key)]
        lo=np.log(lower/nominal_caps);hi=np.log(upper/nominal_caps)
        fresh=np.log(guess/nominal_caps);starts=[fresh]
        if warm is not None:
            continued=warm.copy()
            for i,(kind,key) in enumerate(keys):
                if kind=='self' and key in selected:continued[i]+=np.log(p/warm_p)
            starts.insert(0,np.clip(continued,lo+1e-9,hi-1e-9))
        rng=np.random.default_rng(int(round(p*1000))+1811)
        starts.extend(np.clip(fresh+rng.normal(0,.25,len(keys)),lo+1e-8,hi-1e-8) for _ in range(4))
        candidates=[];attempts=[];closest=None;closest_error=np.inf
        for start in starts:
            fit=least_squares(residual,start,jac=jacobian,bounds=(lo,hi),max_nfev=180,
                ftol=1e-12,xtol=1e-12,gtol=1e-12)
            k,error,jac,_,_=evaluate(tuple(fit.x));mismatch=float(max(abs(error)))
            if mismatch<closest_error:closest_error=mismatch;closest=fit.x.copy()
            loaded_error_hz=float(max(abs(SCALE*np.diag(k)[:5]-np.array([targets[n] for n in names[:5]])))*1e9)
            valid=bool(mismatch<1e-8 and loaded_error_hz<=1 and min(guards(fit.x))>=-1e-8)
            attempts.append(dict(max_relative_mismatch=mismatch,loaded_EC_error_hz=loaded_error_hz,
                                 loading=budget(fit.x),valid=valid,evaluations=fit.nfev))
            if not valid:continue
            _,r,pivots=qr(jac.T,pivoting=True,mode='economic');rank=int(sum(abs(np.diag(r))>1e-9))
            x=fit.x.copy()
            if rank<len(keys):
                selected=pivots[:rank]
                def objective(v):
                    matrix,_,_,derivative,_=evaluate(tuple(v));g=matrix[ri,rj]*weights
                    dj=derivative[:,ri,rj].T*weights[:,None]
                    return float(g@g),2*g@dj
                opt=minimize(objective,x,jac=True,method='SLSQP',bounds=list(zip(lo,hi)),
                    constraints=[dict(type='eq',fun=lambda v:residual(v)[selected],jac=lambda v:jacobian(v)[selected]),
                                 dict(type='ineq',fun=guards)],options=dict(maxiter=160,ftol=1e-10))
                if max(abs(residual(opt.x)))<1e-8 and min(guards(opt.x))>=-1e-8:x=opt.x
            if not any(np.linalg.norm(x-other['x'])<1e-5 for other in candidates):
                matrix=evaluate(tuple(x))[0];g=matrix[ri,rj]*weights
                candidates.append(dict(x=x.copy(),rank=rank,rms=float(np.sqrt(np.mean(g*g)))))
            # A second independent root check suffices for regular full-rank
            # cells; unresolved cells keep all starts and closest diagnostics.
            if len(attempts)>=2 and len(candidates)==1 and rank==len(keys):break
        if not candidates:
            row=dict(self_fraction=p,ground_fraction=1-p,status='No accepted match found',attempts=attempts,
                closest_max_relative_mismatch=closest_error,seconds=time.perf_counter()-t0)
            if closest is not None:
                matrix=evaluate(tuple(closest))[0];row['closest_coefficients']=coefficients(ph,matrix,n01)
                row['closest_capacitors_ff']=(nominal_caps*np.exp(closest)).tolist()
        else:
            def movement(candidate):
                trial_caps=nominal_caps*np.exp(candidate['x'])
                changes=list(trial_caps/nominal_caps-1)
                for n in RAILS:
                    cs=trial_caps[keys.index(('self',n))];ratio=fractions[n]
                    for pad in ph['pad_pairs'][n]:
                        changes.append((2*cs*(1-ratio)/ratio)/ph['ground_capacitances_ff'][pad]-1)
                return max(abs(np.array(changes))),float(np.sum(np.square(changes)))
            best_rms=min(candidate['rms'] for candidate in candidates)
            comparable=[candidate for candidate in candidates if candidate['rms']<=best_rms+1e-6]
            chosen=min(comparable,key=movement)
            x=chosen['x'];warm=x.copy();warm_p=p;k=evaluate(tuple(x))[0];caps=nominal_caps*np.exp(x)
            physical=deepcopy(ph)
            for (kind,key),cap in zip(keys,caps):
                if kind=='self':
                    physical['pad_mutual_capacitances_ff'][key]=float(cap)
                    if key!='C':
                        ratio=fractions[key]
                        for pad in physical['pad_pairs'][key]:physical['ground_capacitances_ff'][pad]=float(2*cap*(1-ratio)/ratio)
                elif kind=='link':physical['physical_links'][key]['capacitance_ff']=float(cap)
                else:physical['resonator_self_capacitances_ff'][key]=float(cap)
            physical=synchronize_physical(physical);verified=circuit(physical)['K']
            if max(abs((verified[ii,jj]-k0[ii,jj])/scales))>1e-8:raise RuntimeError('Independent matching check failed')
            for n in RAILS:
                cs=physical['pad_mutual_capacitances_ff'][n];cg=physical['ground_capacitances_ff'][physical['pad_pairs'][n][0]]
                if abs(cs/(cs+cg/2)-fractions[n])>1e-12:raise RuntimeError('Physical allocation ratio changed')
            current=coefficients(physical,k,n01);g=k[ri,rj]*weights;dg=(k[ri,rj]-k0[ri,rj])*weights
            row=dict(self_fraction=p,rail_self_fractions=fractions,ground_fraction=1-p,status='matched',rank=chosen['rank'],
                remaining_continuous_degrees_of_freedom=len(keys)-chosen['rank'],independent_roots=len(candidates),
                max_relative_constraint_error=float(max(abs(residual(x)))),
                rms_unintended_exchange_mhz=float(np.sqrt(np.mean(g*g))),
                maximum_unintended_exchange_mhz=float(max(abs(g))),
                rms_unintended_exchange_change_mhz=float(np.sqrt(np.mean(dg*dg))),
                K_relative_frobenius_pct=float(np.linalg.norm(k-k0)/np.linalg.norm(k0)*100),
                Cmodal_relative_frobenius_pct=float(np.linalg.norm(np.linalg.inv(k)-np.linalg.inv(k0))/np.linalg.norm(np.linalg.inv(k0))*100),
                max_component_change_pct=float(movement(chosen)[0]*100),
                attempts=attempts,seconds=time.perf_counter()-t0,**current,**budget(x))
            for key,value in reference.items():row['delta_'+key]=row[key]-value
            row['capacitors_ff']={f'{kind}:{key}':float(cap) for (kind,key),cap in zip(keys,caps)}
            row['ground_each_ff']={n:physical['ground_capacitances_ff'][ph['pad_pairs'][n][0]] for n in RAILS}
            row['loaded_budget_ground_fraction']={n:(physical['ground_capacitances_ff'][ph['pad_pairs'][n][0]]/2)/ceff[n] for n in RAILS}
            solutions[str(p)]=dict(physical=physical,K_per_ff=k.tolist(),Cmodal_ff=np.linalg.inv(k).tolist())
        records.append(row);records.sort(key=lambda r:r['self_fraction'])
        write(output/'result.json',dict(input=str(source),scope=__doc__,allocation_scope=scope,
            nominal_rail_self_fractions=nominal_fractions,p_definition='Cs/(Cs+Cg_each/2)',
            fixed_pairs=INTENDED,unintended_pairs=[(names[i],names[j]) for i,j in remaining],reference=reference,
            parameter_keys=keys,rail_loading_limit=rail_limit,coupler_self_minimum=own_min,
            residual_tie_tolerance_mhz=1e-6,
            nominal_rms_unintended_exchange_mhz=float(np.sqrt(np.mean((k0[ri,rj]*weights)**2))),
            records=records,solutions=solutions))
        if len(records)%10==0:print('POINTS',len(records),len(grid),p,row['status'],flush=True)
    result=json.loads((output/'result.json').read_text());result.update(status='complete',seconds=time.perf_counter()-began)
    write(output/'result.json',result)
    keys_out=list(dict.fromkeys(key for row in records for key in row if key not in ('attempts','capacitors_ff','closest_coefficients','closest_capacitors_ff','ground_each_ff','loaded_budget_ground_fraction','rail_self_fractions')))
    with (output/'sweep.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys_out,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    plot(result,output)
    print(json.dumps(dict(seconds=result['seconds'],matched=sum(r['status']=='matched' for r in records),
        points=len(records),samples=[{key:r.get(key) for key in ('self_fraction','status','rank',
        'rms_unintended_exchange_mhz','max_component_change_pct')} for r in records
        if any(abs(r['self_fraction']-p)<1e-8 for p in (.1,.3,.5,.7,.9))]),indent=2))


def write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def plot(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in result['records'] if r['status']=='matched'];x=[r['self_fraction']*100 for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for edge in ('A1_C','A1_B1','A1_B2','A2_B2','C_B2'):
        key='g01_'+edge+'_mhz';axes[0,0].plot(x,[abs(r[key]) for r in rows],label=edge.replace('_','-'))
    axes[0,0].set_ylabel('Unintended transmon exchange magnitude (MHz)')
    for edge in ('C_r1','C_r2','A1_r2','A2_r2','B1_r1','B2_r1','r1_r2'):
        key='g01_'+edge+'_mhz';axes[0,1].plot(x,[abs(r[key]) for r in rows],label=edge.replace('_','-'))
    axes[0,1].set_ylabel('Other unintended exchange magnitudes (MHz)')
    for key,label in [('rms_unintended_exchange_mhz','RMS magnitude'),('rms_unintended_exchange_change_mhz','RMS change from nominal')]:
        axes[1,0].plot(x,[r[key] for r in rows],label=label)
    axes[1,0].set_ylabel('MHz')
    axes[1,0].axhline(result['nominal_rms_unintended_exchange_mhz'],color='0.4',ls=':',label='Nominal RMS magnitude')
    axes[1,1].plot(x,[r['coupler_own_fraction']*100 for r in rows],label='Coupler self contribution')
    axes[1,1].plot(x,[r['A2_full_branch_loading']*100 for r in rows],label='A2 branch loading')
    axes[1,1].plot(x,[r['B1_full_branch_loading']*100 for r in rows],label='B1 branch loading')
    axes[1,1].set_ylabel('Percent')
    for ax in axes.flat:
        ax.set_xlabel('Prescribed self fraction p = Cs / (Cs + Cg/2) (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    if not rows:
        for ax in axes.flat:ax.text(.5,.5,'No accepted match found',transform=ax.transAxes,ha='center')
    fig.suptitle(f"Matched ECs and intended couplings | allocation scope: {result.get('allocation_scope','all')}")
    fig.tight_layout();fig.savefig(output/'matched-allocation-residuals.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--self-fractions',nargs='+',type=float)
    parser.add_argument('--scope',choices=['all','A','B','A2'],default='all')
    args=parser.parse_args();run(args.input,args.output,args.self_fractions,args.scope)
