"""Static four-coupler fanout with physical copies, not multiplied fractions.

One central rail pair is shared. Each branch is a copy of the OTHER complete
rail pair, coupler, both coupler links and direct rail link. Branch ground and
pad-mutual capacitors remain physically identical to the single-cell design.
An optional central-pair shunt compensation restores ONLY its target loaded
ECs; any resulting branch EC drift is explicitly reported.
"""
import numpy as np
from scipy.optimize import least_squares
from .model import SCALE
from .pad_electrostatics import build_pad_circuit,pad_capacitance_budget


def replicate_couplers(config,central_node,copies=4):
    if copies<1 or central_node not in ('A2','B1'):raise ValueError('Choose A2 or B1 and a positive copy count')
    e=config['electrostatics'];original=e['pad_pairs'];shared=tuple(n for n in (('A1','A2') if central_node=='A2' else ('B1','B2')) if n in original)
    others=[n for n in original if n not in shared];pads={n:original[n] for n in shared}
    grounds={p:e['ground_capacitances_ff'][p] for n in shared for p in original[n]}
    mutual={n:e['pad_mutual_capacitances_ff'][n] for n in shared};links=[]
    shared_pads={p for n in shared for p in original[n]}
    for edge in e['physical_links']:
        if edge['from'] in shared_pads and edge['to'] in shared_pads:links.append(dict(edge))
    coupler_pads=[]
    for k in range(copies):
        remap={p:p for p in shared_pads}
        for n in others:
            name=f'{n}@{k}';pair=[f'{p}@{k}' for p in original[n]];pads[name]=pair
            for old,new in zip(original[n],pair):remap[old]=new;grounds[new]=e['ground_capacitances_ff'][old]
            mutual[name]=e['pad_mutual_capacitances_ff'][n]
            if n=='C':coupler_pads.extend(pair)
        for edge in e['physical_links']:
            if edge['from'] in shared_pads and edge['to'] in shared_pads:continue
            links.append(dict(edge,**{'from':remap[edge['from']],'to':remap[edge['to']]},name=f"{edge.get('name','link')}@{k}"))
    circuit=build_pad_circuit(pads,grounds,{},links,mutual)
    remove=[edge for edge in links if edge['from'] in coupler_pads or edge['to'] in coupler_pads]
    return circuit,remove,shared


def compensate_central_pair(circuit,targets):
    names=list(circuit.transmons);which=[names.index(n) for n in targets];target=np.array(list(targets.values()))
    base=circuit.effective_modal_capacitance_ff.copy()
    for n,i in zip(targets,which):base[i,i]-=circuit.junction_capacitances_ff[n]
    def evaluate(x):
        matrix=base.copy();caps=np.exp(x);matrix[which,which]+=caps;k=np.linalg.inv(matrix)
        return (SCALE*np.diag(k)[which]-target)/target,-SCALE*k[np.ix_(which,which)]**2*caps[None,:]/target[:,None]
    start=np.log([circuit.junction_capacitances_ff[n] for n in targets])
    fit=least_squares(lambda x:evaluate(x)[0],start,jac=lambda x:evaluate(x)[1],bounds=(-20.,25.),
                      xtol=1e-12,ftol=1e-12,gtol=1e-12)
    if not fit.success or max(abs(evaluate(fit.x)[0]))>1e-8:raise ValueError('Fanout target EC cannot be restored with positive central pad mutuals')
    mutual=dict(circuit.junction_capacitances_ff);mutual.update(zip(targets,np.exp(fit.x)))
    return build_pad_circuit(circuit.pad_pairs,circuit.ground_capacitances_ff,{},circuit.physical_links,mutual)


def fanout_report(config,central_node,copies=4):
    circuit,remove,shared=replicate_couplers(config,central_node,copies)
    def record(c):
        budget=pad_capacitance_budget(c,remove)
        per_branch=[]
        for k in range(copies):
            cpads=set(c.pad_pairs[f'C@{k}'])
            one=[e for e in remove if e['from'] in cpads or e['to'] in cpads]
            per_branch.append(pad_capacitance_budget(c,one)[central_node]['specified_links_net_loading_fraction'])
        return dict(central_pair={n:dict(budget[n],loaded_ec_ghz=c.loaded_ec_ghz[n]) for n in shared},
                    branch_loaded_ec_ghz={n:v for n,v in c.loaded_ec_ghz.items() if n not in shared},
                    central_node_per_coupler_net_loading_fractions=per_branch,
                    identical_coupler_ground_and_mutual_caps_retained=True)
    output=dict(copies=copies,central_node=central_node,
                construction='Share the central rail pair; copy coupler plus opposite rail pair and all branch capacitors.',
                fixed_physical_hardware=record(circuit))
    try:
        targets={n:config['transmons'][n]['ec_ghz'] for n in shared}
        compensated=compensate_central_pair(circuit,targets)
        output['central_pair_mutuals_compensated']=record(compensated)
        output['compensation_scope']='Only central pair pad-to-pad shunts change. Grounds and every branch component stay fixed; branch loaded EC shifts are reported, not silently retuned.'
    except ValueError as error:output['compensation_error']=str(error)
    return output

