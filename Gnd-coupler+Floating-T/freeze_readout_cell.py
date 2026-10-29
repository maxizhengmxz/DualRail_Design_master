"""Freeze an integrated readout geometry and locate its seven-mode physical 0100-0010 exchange OFF."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrated_readout import construct
from seven_readout import SevenReadout
from floating_coupler.paired_design import off_channel
from floating_coupler.channel_policy import off_target_definition,require_off_definition
from floating_coupler.off_search import exchange_roots

def freeze(request,readouts,output):
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
    if len(roots)!=1:raise ValueError(f'Need one selected OFF root before freezing; found {roots}')
    root=roots[0];physical['bare_frequencies_ghz']['C']=root;s=SevenReadout(model,physical,root).solve()
    out={'off_definition':off_target_definition('seven'),'request':request,'physical':physical,'off_ghz':root,'off_ej_ghz':s.pair.hamiltonian(root)[1],'channel':off_channel(s.matter_detail()),'A_at_reference_EJs':s.readout('A'),'B_at_reference_EJs':s.readout('B'),'scope':'Hardware fitted once at standalone nominal rail EJs for loaded EC and isolated A150/B90. All physical readout links and three coupler capacitors fixed. Freeze the resulting complete geometry and coupler EJ during frequency characterization; only the selected rail EJs vary.'}
    if physical['budget']['C']['specified_links_loading_fraction']>request['coupler_loading_max']+1e-9:raise ValueError('Coupler capacitance-loading limit violated')
    if max(physical['budget'][n]['specified_links_loading_fraction'] for n in ['A2','B1'])>request['rail_loading_per_coupler_max']+1e-9:raise ValueError('Rail coupler-loading limit violated')
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(out,indent=2));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--readout-A',required=True);p.add_argument('--readout-B',required=True);p.add_argument('--output',required=True);a=p.parse_args();freeze(json.loads(Path(a.config).read_text()),{n:json.loads(Path(getattr(a,'readout_'+n)).read_text()) for n in ['A','B']},a.output)
