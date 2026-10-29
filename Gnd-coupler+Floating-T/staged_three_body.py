"""Staged reduced CZ-channel capacitance search. H/h GHz, physical caps fF."""
import argparse,json,sys
from pathlib import Path
from functools import lru_cache
from dataclasses import replace
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq,linear_sum_assignment,minimize
sys.path[:0]=[str(Path(__file__).resolve().parents[1])]
from core.transmon import Basis
from floating_coupler.model import Model,PORT,SCALE,build_model,local_spectrum,port_exchange
from floating_coupler.design import ej_for_frequency
from floating_coupler.pairs import pair_spectrum
from floating_coupler.channel_policy import selection_from_request,off_target_definition
from floating_coupler.off_search import exchange_roots
from floating_coupler.pad_electrostatics import fit_pad_shunts_with_own_ground_fraction,pad_capacitance_budget

frequency_ej=lru_cache(maxsize=20000)(ej_for_frequency)


def single_excitation(model,fc):
    """OFF seed: physical A2/B1 01-10 exchange, C in its ground state."""
    row=port_exchange(model,frequency_ej(float(fc),float(model.ec[1]),model.basis.ncut))
    return dict(row,g_mhz=row['exchange_mhz'],detuning_mhz=-row['diagonal_detuning_mhz'],
                source='01',destination='10',minimum_overlap=row['minimum_subspace_overlap'])

def two_excitation(model,fc):
    """Canonical three-reference effective Hamiltonian, including 20,11,02."""
    ham,_=model.hamiltonian(frequency_ej(float(fc),float(model.ec[1]),model.basis.ncut))
    energy,vectors=eigh(ham)
    n=model.basis.transmon_levels
    refs=[np.ravel_multi_index(s,(n,n,n)) for s in [(2,0,0),(1,0,1),(0,0,2)]]
    projection=vectors[refs,:];_,cols=linear_sum_assignment(-abs(projection)**2)
    u,s,vh=np.linalg.svd(projection[:,cols]);r=u@vh
    effective=(r*energy[cols])@r.T
    return dict(g_mhz=float(effective[1,2]*1000),minimum_overlap=float(s.min()**2))


def circuit_at(p,request):
    ca,cb,direct,ec,eta,ratio=p[:6]
    ecs={'A2':request['ec_a2_ghz'],'C':ec,'B1':request['ec_b1_ghz']}
    pads={'A2':['A21','A22'],'C':['C1','C2'],'B1':['B11','B12']}
    grounds={pad:2*request['rail_ground_fraction']*SCALE/ecs[n] for n in PORT for pad in pads[n]}
    links=[dict(zip(['from','to','capacitance_ff'],edge)) for edge in [('A22','C1',ca),('C1','B11',cb),('A22','B11',direct)]]
    c=fit_pad_shunts_with_own_ground_fraction(pads,ecs,grounds,links,{'C':eta},{'C':ratio})
    config={'nodes':list(PORT),'transmons':{n:{'ec_ghz':ecs[n],'ej_ghz':frequency_ej(float(f),float(ecs[n]),request['ncut'])} for n,f in zip(PORT,[request['idle_fa_ghz'],8.,request['idle_fb_ghz']])},'electrostatics':{'mode':'explicit_pads','pad_pairs':pads,'ground_capacitances_ff':c.ground_capacitances_ff,'pad_mutual_capacitances_ff':c.pad_mutual_capacitances_ff,'physical_links':links},'basis':{'levels':request['levels'],'ncut':request['ncut']}}
    return build_model(config),c,config


def speed_seed(request):
    selected=selection_from_request(request)
    if selected.moved_pair!='B':raise ValueError('This speed-seed kernel moves B only; do not silently move the wrong rail')
    if selected.auxiliary not in ('B20','A02'):raise ValueError('This reduced port seed supports B20 or A02; provide a different mapping for another auxiliary')
    product=1.;details={}
    for label,ecs,f,split,index in [('A',request['ec_pair_a_ghz'],request['idle_fa_ghz'],request['splittings_mhz'][0],1),('B',request['ec_pair_b_ghz'],request['idle_fb_ghz'],request['splittings_mhz'][1],0)]:
        def pair(j,frequency):
            m=Model(('x','y'),np.array(ecs),np.array([frequency_ej(float(frequency),float(e),request['ncut']) for e in ecs]),np.array([[0,j],[j,0]]),Basis(ncut=request['ncut'],transmon_levels=6),{})
            return pair_spectrum(m,(0,1),6,request['ncut'])
        j=brentq(lambda j:pair(j,f)['splitting_mhz']-split,1e-8,.2)
        gatef=f if label=='A' else request['gate_fb_seed_ghz'];p=pair(j,gatef)
        source=int(selected.source_logical[0 if label=='A' else 1])
        excited_aux=selected.auxiliary.startswith(label)
        target=p['auxiliary'][selected.auxiliary[1:]] if excited_aux else 0
        value=abs(p['charge'][index][target,p['one'][source]])
        localindices=(1,2) if excited_aux else (0,1)
        _,charge=local_spectrum(frequency_ej(float(gatef),float(ecs[index]),request['ncut']),ecs[index],Basis(ncut=request['ncut'],transmon_levels=6))
        factor=float(value/abs(charge[localindices]));product*=factor
        details[label]={'isolated_pair_J_ghz':j,'pair_to_local_charge_ratio':factor}
    return {'channel_selection':selected.to_dict(),'pair_details':details,'logical_seed_mhz':500/request['time_ns'],'three_body_11_to_B1_two_seed_mhz':500/request['time_ns']/product,'scope':'direct charge-matrix estimate for the S1-selected channel only; not five-body gate certification'}


def off_roots(model,request):
    fs=np.linspace(*request['off_scan_ghz'],request['off_scan_points'])
    def objective(fc):
        row=single_excitation(model,fc)
        if not row['valid']:raise ValueError('Unreliable physical 01-10 subspace')
        return row['g_mhz']
    return exchange_roots(objective,fs,tolerance=1e-6)


def run(request,output):
    selected=selection_from_request(request)
    if selected.auxiliary!='B20':raise ValueError('This staged search implements reduced 11-to-02 (B20) only; use a matching kernel for another auxiliary')
    seed=speed_seed(request);target=seed['three_body_11_to_B1_two_seed_mhz']
    data={'off_definition':off_target_definition('three'),'request':request,'speed_seed':seed,'off_maps':[],'optimization_attempts':[],'selected':{}}
    previous=json.loads(Path(output).read_text()) if Path(output).exists() else {}
    if previous and previous.get('off_definition')!=off_target_definition('three'):
        raise ValueError('Choose a new output path; historical CZ-auxiliary OFF data must not be overwritten or resumed')
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    def save():path.write_text(json.dumps(data,indent=2))
    for asymmetric in [False,True]:
        if asymmetric and len(data['selected'])==2:break
        families=[]
        cached=next((m['feasible_families'] for m in previous.get('off_maps',[]) if m['unequal_coupler_grounds']==asymmetric),None)
        for ec in ([] if cached is not None else request['ec_map_ghz']):
            for eta in request['eta_map']:
                for ratio in (request['asymmetric_ground_ratio_map'] if asymmetric else [1.]):
                    for q in np.geomspace(*request['link_ratio_map_bounds'],request['link_ratio_map_points']):
                        for dscale in request['direct_map_scales']:
                            ca=request['map_link_scale_ff']*np.sqrt(q);cb=request['map_link_scale_ff']/np.sqrt(q)
                            direct=dscale*ca*cb/(SCALE/ec)
                            p=[float(ca),float(cb),float(direct),ec,eta,ratio]
                            model,_,_=circuit_at(p,request);roots=off_roots(model,request)
                            if roots:families.append({'parameters':p,'roots_ghz':roots})
        if cached is not None:families=cached
        data['off_maps'].append({'unequal_coupler_grounds':asymmetric,'feasible_families':families,'not_a_global_feasibility_proof':True});save()
        print('OFF MAP',asymmetric,'families',len(families),flush=True)
        for lower in request['on_lower_versions_ghz']:
            if str(lower) in data['selected']:continue
            valid=[(f,r) for f in families for r in f['roots_ghz'] if r>lower+.03]
            # Spread starts over root position and ratio, rather than restricting link ratios during optimization.
            valid=sorted(valid,key=lambda fr:fr[1]);chosen=[valid[i] for i in np.linspace(0,len(valid)-1,min(len(valid),request['optimization_starts']),dtype=int)] if valid else []
            for family,root in chosen:
                base=family['parameters'];scale=request['initial_strength_scale'];p0=np.array([base[0]*scale,base[1]*scale,max(base[2]*scale**2,1e-5),*base[3:],root,lower+.001])
                bounds=[(.05,120),(.05,120),(1e-9,5),request['ec_search_bounds_ghz'],request['eta_search_bounds'],([1,20] if asymmetric else [1,1]),(lower+.002,9.999),(lower+.001,9.998)]
                # Independent logarithmic link coordinates span both magnitude and ratio continuously.
                logindices=[0,1,2];z0=p0.copy()
                for i in logindices:z0[i]=np.log(p0[i]);bounds[i]=tuple(np.log(bounds[i]))
                cache={}
                def evaluate(z):
                    key=tuple(z)
                    if key in cache:return cache[key]
                    p=np.array(z);p[logindices]=np.exp(p[logindices]);model,c,cfg=circuit_at(p,request)
                    idle=single_excitation(model,p[6]);ej=model.ej.copy();ej[2]=frequency_ej(request['gate_fb_seed_ghz'],request['ec_b1_ghz'],request['ncut'])
                    on=two_excitation(replace(model,ej=ej),p[7]);links=c.physical_links[:2];budget=pad_capacitance_budget(c,links)
                    loads=np.array([budget[n]['specified_links_net_loading_fraction'] for n in PORT])
                    result=(p,idle,on,budget,loads,cfg);cache[key]=result;return result
                def objective(z):
                    try:return float(evaluate(z)[4][[0,2]].sum())
                    except ValueError:return 10.
                def equalities(z):
                    try:
                        _,idle,on,_,_,_=evaluate(z);return np.array([idle['g_mhz']/target,(abs(on['g_mhz'])-target)/target])
                    except ValueError:return np.array([100.,100.])
                def inequalities(z):
                    try:
                        p,idle,on,_,loads,_=evaluate(z)
                        return np.r_[np.array([.1,.45,.1])-loads,(p[6]-p[7]-.0001),idle['minimum_overlap']-.8,on['minimum_overlap']-.8]
                    except ValueError:return np.ones(6)*-100
                fit=minimize(objective,z0,method='SLSQP',bounds=bounds,constraints=[{'type':'eq','fun':equalities},{'type':'ineq','fun':inequalities}],options={'maxiter':request['maxiter'],'ftol':1e-10,'eps':2e-6})
                try:
                    p,idle,on,budget,loads,cfg=evaluate(fit.x)
                    feasible=bool(max(abs(equalities(fit.x)))<1e-4 and min(inequalities(fit.x))>=-1e-7)
                    cfg['transmons']['C']['ej_ghz']=frequency_ej(float(p[6]),float(p[3]),request['ncut'])
                    row={'lower_ghz':lower,'unequal_coupler_grounds':asymmetric,'success':bool(fit.success),'message':str(fit.message),'feasible':feasible,'parameters':p.tolist(),'off':idle,'on':on,'loading':budget,'objective_sum_rail_fractions':objective(fit.x),'config':cfg}
                    data['optimization_attempts'].append(row)
                    if feasible and (str(lower) not in data['selected'] or row['objective_sum_rail_fractions']<data['selected'][str(lower)]['objective_sum_rail_fractions']):data['selected'][str(lower)]=row
                    print('OPT',lower,asymmetric,'feasible',feasible,'loads',loads,'on/off',p[7],p[6],flush=True)
                except ValueError as e:data['optimization_attempts'].append({'error':str(e)})
                save()
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)

