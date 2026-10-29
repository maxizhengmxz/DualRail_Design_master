"""Five-mode synthesis with symmetric FIXED grounds and fitted physical pad shunts.

Every optimizer trial reconstructs the full physical ten-pad capacitance
matrix. Neither physical links nor their signs are inherited from a reduced
grounded circuit. Ground fractions are explicit design-family assumptions.
"""
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares
from .model import FULL,SCALE,build_model
from .design import ej_for_frequency
from .pairs import PairModel
from .paired_design import selected_channel,off_channel
from .channel_policy import selection_from_request,off_target_definition
from .pad_electrostatics import fit_pad_mutual_capacitances,fit_pad_shunts_with_own_ground_fraction,pad_capacitance_budget


def validate_pad_design_constraints(spec):
    """Reject a wrong operating branch BEFORE any numerical synthesis.

    Bounds are explicit request inputs, never inferred from an optimizer seed.
    Equality does not satisfy the user's strict frequency inequalities.
    """
    required=('on_frequency_lower_exclusive_ghz','off_frequency_upper_exclusive_ghz')
    if any(k not in spec for k in required):
        raise ValueError('Supply explicit ON lower and OFF upper frequency constraints')
    lower=float(spec['on_frequency_lower_exclusive_ghz'])
    on=float(spec['on_fc_ghz']);off=float(spec['off_fc_ghz'])
    upper=float(spec['off_frequency_upper_exclusive_ghz'])
    if not np.all(np.isfinite([lower,on,off,upper])):
        raise ValueError('Frequency bounds and points must be finite')
    if not lower<on<off<upper:
        raise ValueError('Required branch is ON_lower < f_ON < f_OFF < OFF_upper; lower-OFF candidates are invalid')


def synthesize_pad_coupler(spec,initial=None):
    validate_pad_design_constraints(spec)
    selection=selection_from_request(spec).to_dict()
    ec={n:spec['coupler_ec_ghz'] if n=='C' else spec['rail_ec_ghz'][n] for n in FULL}
    pads=spec['pad_pairs'];basis=spec['basis']
    fractions={n:spec['ground_fraction_coupler'] if n=='C' else spec['ground_fraction_rails'] for n in FULL}
    if any(not 0<f<1 for f in fractions.values()):raise ValueError('Ground fractions must be between zero and one')
    grounds={p:2*fractions[n]*SCALE/ec[n] for n in FULL for p in pads[n]}
    freq=[spec['idle_fa_ghz'],spec['idle_fa_ghz'],spec['off_fc_ghz'],spec['idle_fb_ghz'],spec['idle_fb_ghz']]
    ejs={n:ej_for_frequency(f,ec[n],basis['ncut']) for n,f in zip(FULL,freq)}
    ratio=spec['link_scale_ratio'];factor=np.array([np.sqrt(ratio),1/np.sqrt(ratio)])
    budgets=np.array([SCALE/ec['A2'],SCALE/ec['B1']])
    evaluations=0;cache={}
    def at(x):
        nonlocal evaluations
        key=tuple(x)
        if key in cache:return cache[key]
        evaluations+=1
        link_a,link_b=4*x[0]*factor*budgets
        caps=dict(A2_C=link_a,C_B1=link_b,A2_B1=x[1],A1_A2=x[2],B1_B2=x[3])
        links=[dict(name=k,**{'from':v[0],'to':v[1]},capacitance_ff=float(caps[k]))
               for k,v in spec['link_endpoints'].items()]
        if 'own_ground_fraction_coupler' in spec:
            circuit=fit_pad_shunts_with_own_ground_fraction(pads,ec,grounds,links,{'C':spec['own_ground_fraction_coupler']})
        else:
            circuit=fit_pad_mutual_capacitances(pads,ec,grounds,{},links)
        config=dict(nodes=list(FULL),channel_selection=selection,transmons={n:dict(ec_ghz=ec[n],ej_ghz=ejs[n]) for n in FULL},
          electrostatics=dict(mode='explicit_pads',pad_pairs=pads,ground_capacitances_ff=circuit.ground_capacitances_ff,
             pad_mutual_capacitances_ff=circuit.junction_capacitances_ff,physical_links=links),
          basis=dict(levels=basis['local_levels'],ncut=basis['ncut']))
        model=build_model(config);system=PairModel(model,**basis);off=system.evaluate(spec['off_fc_ghz'])
        ej=model.ej.copy()
        for i in (3,4):ej[i]=ej_for_frequency(x[4],model.ec[i],basis['ncut'])
        on=PairModel(replace(model,ej=ej),**basis).evaluate(spec['on_fc_ghz'])
        removed=[e for e in links if e['name'] in ('A2_C','C_B1')]
        loading=pad_capacitance_budget(circuit,removed)
        offch,onch=off_channel(off),selected_channel(on)
        residual=np.array([(system.a['splitting_mhz']-spec['pair_splittings_mhz'][0])/150,
            (system.b['splitting_mhz']-spec['pair_splittings_mhz'][1])/90,
            offch['g_mhz']/spec.get('off_exchange_scale_mhz',.001),(onch['g_mhz']-spec['on_g_target_mhz'])/5,onch['detuning_mhz']/50,
            max(0,loading['A2']['specified_links_net_loading_fraction']-spec['maximum_rail_net_loading'])*100,
            max(0,loading['B1']['specified_links_net_loading_fraction']-spec['maximum_rail_net_loading'])*100,
            max(0,loading['C']['specified_links_net_loading_fraction']-spec['maximum_coupler_net_loading'])*100])
        value=(residual,config,circuit,loading,off,on)
        cache[key]=value
        return value
    x0=initial if initial is not None else spec['initial_solution_vector']
    bounds=([.003,1e-8,.05,.05,spec['gate_fb_bounds_ghz'][0]],
            [.18,5.,40.,35.,spec['gate_fb_bounds_ghz'][1]])
    def residual(x):
        try:return at(x)[0]
        except ValueError:return np.ones(8)*100
    fit=least_squares(residual,np.clip(x0,np.array(bounds[0])+1e-7,np.array(bounds[1])-1e-7),
          bounds=bounds,diff_step=2e-5,x_scale=[.05,.5,10,6,.1],max_nfev=85,
          xtol=2e-9,ftol=2e-9,gtol=2e-9)
    res,config,circuit,loading,off,on=at(fit.x)
    return dict(off_definition=off_target_definition('five'),spec=spec,feasible=bool(max(abs(res))<2e-5 and min(off['minimum_subspace_overlap'],on['minimum_subspace_overlap'])>.8),
                solution_vector=fit.x.tolist(),maximum_scaled_residual=float(max(abs(res))),
                optimizer_success=bool(fit.success),evaluations=evaluations,config=config,
                electrostatics=circuit.summary(),loading=loading,off=off,on=on,
                residual=res.tolist(),gate_fb_ghz=float(fit.x[4]))
