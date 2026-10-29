"""Integrate independently designed physical readout links into grounded coupler cells."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.constants import elementary_charge as e,h,hbar
from scipy.optimize import brentq,least_squares
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.transmon import Basis
from floating_coupler.model import Model,FULL,SCALE,local_spectrum
from floating_coupler.design import ej_for_frequency
from floating_coupler.pairs import PairModel,pair_spectrum
from floating_coupler.paired_design import off_channel
from floating_coupler.channel_policy import selection_from_request
from floating_coupler.mixed_electrostatics import fit_mixed_shunts,mixed_budget


def construct(request,readouts,frequencies=None,refit_intra=True):
    targets=request['loaded_ec_ghz'];pads={'A1':['A11','A12'],'A2':['A21','A22'],'C':['C1','C2'],'B1':['B11','B12'],'B2':['B21','B22']};grounds={'C1':0.,'C2':0.};links=deepcopy(request['seed_grounded_three_body']['physical']['physical_links']);rcaps={};lr={};freq={}
    for name,r in readouts.items():
        p=r['physical'];grounds.update(p['ground_capacitances_ff']);links.extend(deepcopy(p['physical_links']));rn=r['case']['resonator_node'];rcaps[rn]=p['resonator_self_capacitance_ff'];lr[rn]=p['resonator_inductance_nh']
        freq.update(dict(zip(r['case']['nodes'],r['coordinates'][:2])))
    freq['C']=request['idle_bare_frequencies_ghz']['C']
    if frequencies is not None:freq.update(frequencies)
    if refit_intra:
        nominal_ec=np.array(list(targets.values()));ejfixed=np.array([ej_for_frequency(freq[n],charging,35) for n,charging in zip(FULL,nominal_ec)])
        def residual(caps):
            links[3]['capacitance_ff']=float(caps[0]);links[6]['capacitance_ff']=float(caps[1]);sh,cc=fit_mixed_shunts(pads,grounds,links,targets,resonator_caps_ff=rcaps);jj=8*SCALE*cc['K'][:5,:5];np.fill_diagonal(jj,0.)
            mm=Model(FULL,nominal_ec,ejfixed,jj,Basis(ncut=35,transmon_levels=6),{})
            return [(pair_spectrum(mm,(0,1),6,35)['splitting_mhz']-150)/150,(pair_spectrum(mm,(3,4),6,35)['splitting_mhz']-90)/90]
        fit=least_squares(residual,[links[3]['capacitance_ff'],links[6]['capacitance_ff']],bounds=([.005,.005],[40,40]),xtol=1e-11,ftol=1e-11,gtol=1e-11)
        if max(abs(np.asarray(residual(fit.x))))>1e-7:raise ValueError('Integrated isolated-pair target fit failed')
    shunts,c=fit_mixed_shunts(pads,grounds,links,targets,resonator_caps_ff=rcaps)
    ec=c['loaded_ec_ghz'];j=8*SCALE*c['K'][:5,:5];np.fill_diagonal(j,0.);ej=np.array([ej_for_frequency(freq[n],charging,35) for n,charging in zip(FULL,ec)]);model=Model(FULL,ec,ej,j,Basis(ncut=35,transmon_levels=6),{})
    budget=mixed_budget(pads,grounds,shunts,links,request['seed_grounded_three_body']['physical']['physical_links'][:2],resonator_caps_ff=rcaps)
    physical={'pad_pairs':pads,'ground_capacitances_ff':grounds,'pad_mutual_capacitances_ff':shunts,'physical_links':links,'resonator_self_capacitances_ff':rcaps,'resonator_inductances_nh':lr,'loaded_ec_ghz':dict(zip(FULL,ec.tolist())),'bare_frequencies_ghz':freq,'K_per_ff':c['K'].tolist(),'budget':budget,'compensation_policy':'One-time self-shunt and intra-pair compensation preserves loaded EC and isolated A150/B90. The readout links and all three coupler capacitors remain fixed. Freeze all resulting components for frequency sweeps.'}
    resonators={}
    for k,r in enumerate(rcaps):
        idx=5+k;ceff=1e-15/c['K'][idx,idx];omega=1/np.sqrt(lr[r]*1e-9*ceff);q=np.sqrt(hbar*ceff*omega/2);gammas=2*e*q/(h*1e9)*c['K'][:5,idx]*1e15
        resonators[r]={'fr_ghz':omega/(2*np.pi*1e9),'effective_capacitance_ff':ceff*1e15,'q_zpf_coulomb':q,'gamma_ghz_per_n':gammas.tolist()}
    physical['resonator_parameters']=resonators
    if request.get('channel_selection') or request.get('gate_target',{}).get('channel_selection'):
        selection=selection_from_request(request).to_dict();model.electrostatics['channel_selection']=selection;physical['channel_selection']=selection
    return model,physical


def local_coupler_diagnostics(model,physical,fc):
    en,n=local_spectrum(ej_for_frequency(fc,model.ec[2],35),model.ec[2],Basis(ncut=35,transmon_levels=7));out={}
    for r,p in physical['resonator_parameters'].items():
        gamma=p['gamma_ghz_per_n'][2];rows=[]
        for i in [0,1,2]:
            f=float(en[i+1]-en[i]);g=float(abs(gamma*n[i,i+1])*1000);delta=(f-p['fr_ghz'])*1000
            rows.append({'transition':f'{i}-{i+1}','frequency_ghz':f,'g_mhz':g,'detuning_mhz':delta,'abs_g_over_delta':abs(g/delta)})
        out[r]=rows
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--readout-A',required=True);p.add_argument('--readout-B',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=json.loads(Path(a.config).read_text());readouts={n:json.loads(Path(getattr(a,'readout_'+n)).read_text()) for n in ['A','B']};model,physical=construct(r,readouts)
    system=PairModel(model,local_levels=5,pair_states=12,coupler_levels=5,ncut=35);samples=[];roots=[]
    for fc in np.linspace(r['on_lower_exclusive_ghz']+.001,12,35):samples.append({'fc':float(fc),'channel':off_channel(system.evaluate(float(fc)))})
    for x,y in zip(samples[:-1],samples[1:]):
        if x['channel']['g_mhz']*y['channel']['g_mhz']<0:roots.append(brentq(lambda f:off_channel(system.evaluate(f))['g_mhz'],x['fc'],y['fc']))
    fc=r.get('calibrated_gate_bare_frequencies_ghz',{}).get('C',r['gate_search']['fc_bounds_ghz'][0])
    from floating_coupler.channel_policy import off_target_definition
    result={'off_definition':off_target_definition('five'),'physical':physical,'matter_only_off_samples':samples,'matter_only_off_roots':roots,'local_coupler_readout_at_provisional_ON':local_coupler_diagnostics(model,physical,fc),'provisional_ON_ghz':fc,'scope':'Exact seven-mode electrostatics. Matter-only preliminary OFF ignores photon dynamics and is not the final integrated OFF certification.'};Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print('INTEGRATED',r['on_lower_exclusive_ghz'],'roots',roots,'budget',physical['budget'],'g',result['local_coupler_readout_at_provisional_ON'],flush=True)

