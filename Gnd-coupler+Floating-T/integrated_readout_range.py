"""Frozen seven-mode coupler-OFF frequency corridors, rail EJ tuning only."""
import argparse,json,sys
from pathlib import Path
from functools import lru_cache
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from seven_readout import SevenReadout
from floating_coupler.model import Model,FULL,SCALE,local_spectrum
from floating_coupler.design import ej_for_frequency
from core.transmon import Basis


def frozen_model(physical):
    ec=np.array([physical['loaded_ec_ghz'][n] for n in FULL]);ej=np.array([ej_for_frequency(physical['bare_frequencies_ghz'][n],e,25) for n,e in zip(FULL,ec)]);j=8*SCALE*np.array(physical['K_per_ff'])[:5,:5];np.fill_diagonal(j,0.)
    metadata={}
    if physical.get('channel_selection') is not None:
        from floating_coupler.channel_policy import require_selection
        metadata['channel_selection']=require_selection(physical['channel_selection']).to_dict()
    return Model(FULL,ec,ej,j,Basis(ncut=25,transmon_levels=6),metadata)


def constraint_values(rows,model,pair,case):
    idx=[0,1] if pair=='A' else [3,4];ec=model.ec[idx];ej=model.ej[idx];low,high=case['dressed_split_window_ghz'];values=[]
    for p in rows:
        values.extend([(.01-p['ratio'])/.01,abs((p['chi_m_ghz']+p['chi_p_ghz'])/2)*1000-1,(.1-p['lambda_max'])/.1,(p['hybridization']-.95)/.05,(p['min_assignment_overlap']-.95)/.05,(p['splitting_ghz']-low)/.01,(high-p['splitting_ghz'])/.01,p['operating_frequency_ghz']-4,8-p['operating_frequency_ghz'],p['dressed_r_ghz']-p['operating_frequency_ghz']])
    values.extend((25-1e-8-ej)/10);values.extend((ej/ec-45-1e-8)/50)
    return np.array(values)


def run(cell,case,pair,output,trim_mhz=None):
    physical=cell['physical'];base=frozen_model(physical);fc=cell['off_ghz'];idx=[0,1] if pair=='A' else [3,4];guide=case['bare_frequency_guide_ghz'];count=0;path=Path(output);previous=json.loads(path.read_text()) if path.exists() else {}
    @lru_cache(maxsize=128)
    def at(key):
        nonlocal count
        ej=base.ej.copy()
        for i,f in zip(idx,key):ej[i]=ej_for_frequency(float(f),base.ec[i],25)
        model=replace(base,ej=ej);s=SevenReadout(model,physical,fc).solve();rows=s.readout(pair);constraints=constraint_values(rows,model,pair,case);mean=float(np.mean([p['operating_frequency_ghz'] for p in rows]));count+=1
        if count%10==0:print('RANGE_EVAL',cell['request']['on_lower_exclusive_ghz'],pair,count,'f',list(key),'operation',mean,'slack',float(min(constraints)),flush=True)
        return rows,constraints,mean
    bounds=[]
    for i in idx:bounds.append((float(local_spectrum(45.00001*base.ec[i],base.ec[i],base.basis)[0][1]),float(local_spectrum(24.99999,base.ec[i],base.basis)[0][1])))
    if trim_mhz is not None:bounds=[(max(a,guide-trim_mhz/1000),min(b,guide+trim_mhz/1000)) for a,b in bounds]
    start=np.array([physical['bare_frequencies_ghz'][FULL[i]] for i in idx]);start[1 if pair=='A' else 0]+=.023
    start=np.clip(start,np.array(bounds)[:,0]+1e-6,np.array(bounds)[:,1]-1e-6)
    def ineq(x):return at(tuple(x))[1]
    center=minimize(lambda x:((at(tuple(x))[2]-guide)/.01)**2+np.mean([p['ratio']**2 for p in at(tuple(x))[0]]),start,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':ineq}],options={'maxiter':50,'ftol':1e-10,'eps':2e-6})
    center_rows,slack,centerF=at(tuple(center.x));out={'stage':'Full seven-mode fixed-coupler-OFF range','pair':pair,'case':case,'off_frequency_ghz':fc,'off_ej_ghz':cell['off_ej_ghz'],'frozen_hardware':physical,'other_rail_reference_bare_frequencies_ghz':{n:physical['bare_frequencies_ghz'][n] for n in FULL if n not in [FULL[i] for i in idx] and n!='C'},'center':{'bare_frequencies_ghz':center.x.tolist(),'operating_frequency_ghz':centerF,'rows':center_rows,'minimum_slack':float(min(slack))},'endpoints':[],'bare_trim_limit_mhz':trim_mhz,'scope':'Characterization ranges vary only the selected pair EJs over the strict EJ/EJEC hardware bounds. All capacitors, resonator L, other-pair EJs and coupler EJ fixed. This is a connected best-found corridor, not a global all-branches proof.'}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2))
    if min(slack)<-1e-5:out['status']='no feasible center found';path.write_text(json.dumps(out,indent=2));print('NO_CENTER',pair,center.message,flush=True);return out
    if previous.get('frozen_hardware',{}).get('K_per_ff')==physical['K_per_ff']:out['endpoints']=previous.get('endpoints',[])
    for direction in [1,-1]:
        if any(r['side']==('lower' if direction==1 else 'upper') for r in out['endpoints']):continue
        fit=minimize(lambda x:direction*at(tuple(x))[2],center.x,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':ineq}],options={'maxiter':60,'ftol':1e-10,'eps':2e-6})
        rows,c,F=at(tuple(fit.x));fraction=min(.25,.001/max(abs(F-centerF),.001));inside=fit.x+fraction*(center.x-fit.x);rows,c,F=at(tuple(inside));out['endpoints'].append({'side':'lower' if direction==1 else 'upper','optimizer_success':bool(fit.success),'message':str(fit.message),'bare_frequencies_ghz':inside.tolist(),'operation_frequency_ghz':F,'rows':rows,'minimum_slack':float(min(c)),'boundary_bare_frequencies_ghz':fit.x.tolist(),'boundary_operation_ghz':at(tuple(fit.x))[2]});path.write_text(json.dumps(out,indent=2));print('ENDPOINT',cell['request']['on_lower_exclusive_ghz'],pair,out['endpoints'][-1],flush=True)
    if len(out['endpoints'])==2:
        mid=(np.array(out['endpoints'][0]['boundary_bare_frequencies_ghz'])+np.array(out['endpoints'][1]['boundary_bare_frequencies_ghz']))/2
        midF=at(tuple(mid))[2]
        for row in out['endpoints']:
            edge=np.array(row['boundary_bare_frequencies_ghz']);edgeF=at(tuple(edge))[2];fraction=min(.25,.001/max(abs(edgeF-midF),.001))
            for attempt in range(8):
                inside=edge+fraction*(mid-edge);rr,cc,ff=at(tuple(inside))
                if min(cc)>1e-6:break
                fraction=min(.95,max(fraction*2,fraction+.01))
            row.update(bare_frequencies_ghz=inside.tolist(),operation_frequency_ghz=ff,rows=rr,minimum_slack=float(min(cc)),guard_fraction_toward_midpoint=fraction)
        out['base_basis_midpoint']={'bare_frequencies_ghz':mid.tolist(),'rows':at(tuple(mid))[0],'minimum_slack':float(min(at(tuple(mid))[1]))}
    out['status']='base-basis corridor found';path.write_text(json.dumps(out,indent=2));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--readout-config',required=True);p.add_argument('--pair',choices=['A','B'],required=True);p.add_argument('--output',required=True);p.add_argument('--trim-mhz',type=float);a=p.parse_args();cell=json.loads(Path(a.cell).read_text());case=json.loads(Path(a.readout_config).read_text())['pairs'][a.pair];run(cell,case,a.pair,a.output,a.trim_mhz)
