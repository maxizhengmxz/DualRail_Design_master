import sys,json,argparse
from pathlib import Path
from scipy.optimize import least_squares,brentq
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1])]
from floating_coupler.model import FULL,SCALE
from floating_coupler.mixed_electrostatics import fit_mixed_shunts,mixed_model,mixed_budget,mixed_fanout
from floating_coupler.pairs import pair_spectrum,PairModel
from floating_coupler.paired_design import selected_channel,off_channel
from floating_coupler.channel_policy import selection_from_request,off_target_definition


def construct(seed,request=None):
    if request is None:raise ValueError('Supply explicit five-body request inputs')
    targets=request['loaded_ec_ghz']
    pads={'A1':['A11','A12'],'A2':['A21','A22'],'C':['C1','C2'],'B1':['B11','B12'],'B2':['B21','B22']}
    grounds={p:(0. if n=='C' else 2*request['rail_ground_series_fraction']*SCALE/targets[n]) for n in FULL for p in pads[n]}
    fixed=seed['physical']['physical_links'];freqs=request['idle_bare_frequencies_ghz']
    def at(caps):
        links=fixed+[{'from':'A12','to':'A21','capacitance_ff':float(caps[0])},{'from':'B12','to':'B21','capacitance_ff':float(caps[1])}]
        shunts,c=fit_mixed_shunts(pads,grounds,links,targets);model,_=mixed_model(pads,grounds,shunts,links,freqs,request['local_levels'],request['ncut'])
        paira=pair_spectrum(model,(0,1),request['local_levels'],request['ncut']);pairb=pair_spectrum(model,(3,4),request['local_levels'],request['ncut'])
        return model,shunts,links,paira,pairb
    def residual(caps):
        *_,a,b=at(caps);return [(a['splitting_mhz']-request['idle_isolated_splittings_mhz'][0])/request['idle_isolated_splittings_mhz'][0],(b['splitting_mhz']-request['idle_isolated_splittings_mhz'][1])/request['idle_isolated_splittings_mhz'][1]]
    fit=least_squares(residual,[12.4,6.35],bounds=([request['pair_cap_bounds_ff'][0]]*2,[request['pair_cap_bounds_ff'][1]]*2),xtol=1e-12,ftol=1e-12,gtol=1e-12)
    model,shunts,links,a,b=at(fit.x)
    if max(abs(np.array(residual(fit.x))))>1e-8:raise ValueError('Isolated rail splitting calibration failed')
    budget=mixed_budget(pads,grounds,shunts,links,fixed[:2])
    if budget['C']['specified_links_loading_fraction']>request['coupler_loading_max']+1e-9:raise ValueError('Coupler loading constraint violated after five-body construction')
    if max(budget[n]['specified_links_loading_fraction'] for n in ('A2','B1'))>request['rail_loading_per_coupler_max']+1e-9:raise ValueError('Per-coupler rail loading constraint violated')
    selection=selection_from_request(request).to_dict() if request.get('channel_selection') or request.get('gate_target',{}).get('channel_selection') else None
    if selection is not None:model.electrostatics['channel_selection']=selection
    return model,dict(pad_pairs=pads,grounds_ff=grounds,shunts_ff=shunts,links=links,targets_ec_ghz=targets,idle_frequencies_ghz=freqs,intra_pair_caps_ff=fit.x.tolist(),pair_splittings_mhz=[a['splitting_mhz'],b['splitting_mhz']],budget=budget,channel_selection=selection)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--gate',action='store_true');parser.add_argument('--fanout',action='store_true');args=parser.parse_args()
    request=json.loads(Path(args.config).read_text());seed=request['seed_grounded_three_body'];model,record=construct(seed,request)
    out={'off_definition':off_target_definition('five'),'circuit':record,'off_scans':[]};path=Path(args.output);path.mkdir(parents=True,exist_ok=True)
    if (path/'initial-five.json').exists():
        from floating_coupler.channel_policy import require_off_definition
        require_off_definition(json.loads((path/'initial-five.json').read_text()).get('off_definition'))
    print('CIRCUIT',json.dumps(record),flush=True)
    for basis in [dict(local_levels=5,pair_states=12,coupler_levels=5,ncut=35),dict(local_levels=6,pair_states=15,coupler_levels=6,ncut=35)]:
        system=PairModel(model,**basis);samples=[];roots=[]
        for f in np.linspace(*request['off_bracket_ghz'],17):
            detail=system.evaluate(float(f));row={'fc':float(f),'channel':off_channel(detail),'overlap':detail['minimum_subspace_overlap']};samples.append(row)
        from floating_coupler.off_search import exchange_roots
        for root in exchange_roots(lambda f:off_channel(system.evaluate(f))['g_mhz'],np.linspace(*request['off_bracket_ghz'],17)):
            detail=system.evaluate(root);roots.append({'fc':root,'detail':detail,'channel':off_channel(detail),'within_off_bound':root<request['off_upper_exclusive_ghz']})
        out['off_scans'].append({'basis':basis,'samples':samples,'roots':roots});(path/'initial-five.json').write_text(json.dumps(out,indent=2));print('OFF',basis,[(r['fc'],r['channel']) for r in roots],flush=True)

    if args.fanout:
        (path/'four-coupler-static.json').write_text(json.dumps({n:mixed_fanout(record,n) for n in ('A2','B1')},indent=2))
    if args.gate:
        selection_from_request(request)
        from floating_coupler.calibrate_on import optimize_coupler_fixed_time
        g=request['gate_search']
        if not request['on_lower_exclusive_ghz']<g['fc_bounds_ghz'][0]<g['fc_bounds_ghz'][1]<request['off_upper_exclusive_ghz']:raise ValueError('Gate search bounds violate frequency constraints')
        roots=out['off_scans'][-1]['roots']
        if not roots or not all(r['within_off_bound'] for r in roots):raise ValueError('No accepted OFF root before gate calibration')
        if g['fc_bounds_ghz'][1]>=max(r['fc'] for r in roots):raise ValueError('ON search must remain below selected OFF root')
        gate=optimize_coupler_fixed_time(model,request['target_static_hold_ns'],g['basis'],g['fb_bounds_ghz'],g['fc_bounds_ghz'],points=g['points'],local_starts=g['local_starts'],phase_samples=g['phase_samples'],minimum_midpoint_b20=g['minimum_midpoint_b20'])
        (path/'phase-calibration.json').write_text(json.dumps(gate,indent=2))
        print('GATE',json.dumps(gate['best']),flush=True)
