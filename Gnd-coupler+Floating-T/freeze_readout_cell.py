"""Freeze an integrated readout geometry and locate its seven-mode physical 0100-0010 exchange OFF."""
import argparse,json,sys
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrated_readout import construct
from seven_readout import SevenReadout
from floating_coupler.paired_design import off_channel
from floating_coupler.channel_policy import off_target_definition,require_off_definition
from floating_coupler.off_search import exchange_roots

def with_refined_five_body(request,refinement):
    """Replace the reduced three-body seed by one certified refined geometry."""
    request=deepcopy(request);result=refinement.get('result',refinement);physical=result['physical']
    intended=[{'A22','C1'},{'C1','B11'},{'A22','B11'}]
    links=[link for endpoints in intended for link in physical['links']
           if {link['from'],link['to']}==endpoints]
    if len(links)!=3:raise ValueError('Refined five-body record must contain each intended coupler link exactly once')
    targets=physical['targets_ec_ghz']
    if any(abs(float(targets[n])-float(request['loaded_ec_ghz'][n]))>1e-10 for n in ('A1','A2','B1','B2')):
        raise ValueError('Refined five-body rail ECs do not match the integration request')
    request['seed_grounded_three_body']['physical']['physical_links']=deepcopy(links)
    request['loaded_ec_ghz']['C']=float(targets['C'])
    request['idle_bare_frequencies_ghz']['C']=float(physical['idle_frequencies_ghz']['C'])
    request['refined_five_body_provenance']={
        'physical_links':deepcopy(links),'loaded_coupler_ec_ghz':float(targets['C']),
        'pair_splittings_mhz':result.get('pair_splittings_mhz'),
        'off_definition':result.get('off_definition'),
    }
    return request

def freeze(request,readouts,output,allow_out_of_range_off_seed=False):
    from floating_coupler.channel_policy import selection_from_request
    if request.get('channel_selection') or request.get('gate_target',{}).get('channel_selection'):selection_from_request(request)
    if Path(output).exists():require_off_definition(json.loads(Path(output).read_text()).get('off_definition'),'seven')
    model,physical=construct(request,readouts);lo=max(request['off_bracket_ghz'][0],request['on_lower_exclusive_ghz']+.001);hi=request['off_upper_exclusive_ghz']-.0001;samples=[]
    def at(f):
        s=SevenReadout(model,physical,f).solve();detail=s.matter_detail()
        if detail['minimum_subspace_overlap']<.8:raise ValueError('Unreliable matter reference space')
        return off_channel(detail)['g_mhz']
    for f in np.linspace(lo,hi,17):samples.append((float(f),at(float(f))))
    roots=exchange_roots(at,np.linspace(lo,hi,17))
    out_of_range_seed=False
    if not roots and allow_out_of_range_off_seed:
        diagnostic_upper=max(12.,hi)
        diagnostic_roots=exchange_roots(at,np.linspace(hi,diagnostic_upper,17))
        if diagnostic_roots:
            roots=[min(diagnostic_roots)];out_of_range_seed=True
    if len(roots)!=1:raise ValueError(f'Need one selected OFF root before freezing; found {roots}')
    root=roots[0];physical['bare_frequencies_ghz']['C']=root;s=SevenReadout(model,physical,root).solve()
    targets=request['idle_isolated_splittings_mhz']
    out={'off_definition':off_target_definition('seven'),'request':request,'physical':physical,'off_ghz':root,'off_ej_ghz':s.pair.hamiltonian(root)[1],'channel':off_channel(s.matter_detail()),'initial_off_seed_out_of_range':out_of_range_seed,'initial_off_seed_acceptance':'diagnostic_only_requires_idle_calibration_rescue' if out_of_range_seed else 'inside_requested_bound','A_at_reference_EJs':s.readout('A'),'B_at_reference_EJs':s.readout('B'),'scope':f'Hardware fitted once at standalone nominal rail EJs for loaded EC and supplied isolated A={targets[0]:.12g}/B={targets[1]:.12g} MHz targets. All physical readout links and three coupler capacitors fixed. Freeze the resulting complete geometry and coupler EJ during frequency characterization; only the selected rail EJs vary.'}
    if physical['budget']['C']['specified_links_loading_fraction']>request['coupler_loading_max']+1e-9:raise ValueError('Coupler capacitance-loading limit violated')
    if max(physical['budget'][n]['specified_links_loading_fraction'] for n in ['A2','B1'])>request['rail_loading_per_coupler_max']+1e-9:raise ValueError('Rail coupler-loading limit violated')
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(out,indent=2));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--refined-five');p.add_argument('--allow-out-of-range-off-seed',action='store_true');p.add_argument('--readout-A',required=True);p.add_argument('--readout-B',required=True);p.add_argument('--output',required=True);a=p.parse_args();request=json.loads(Path(a.config).read_text());request=with_refined_five_body(request,json.loads(Path(a.refined_five).read_text())) if a.refined_five else request;freeze(request,{n:json.loads(Path(getattr(a,'readout_'+n)).read_text()) for n in ['A','B']},a.output,a.allow_out_of_range_off_seed)
