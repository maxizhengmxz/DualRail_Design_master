"""Mixed floating-rail/grounded-coupler capacitance matrices and EC fitting."""
import numpy as np
from scipy.optimize import least_squares
from core.transmon import Basis
from .model import Model,SCALE
from .design import ej_for_frequency


def mixed_circuit(pad_pairs,grounds,shunts,links,grounded_pads=('C2',),resonator_caps_ff=None):
    names=tuple(pad_pairs);grounded=set(grounded_pads)
    resonators=tuple(resonator_caps_ff or {})
    nodes=tuple(p for n in names for p in pad_pairs[n] if p not in grounded)+resonators;idx={p:i for i,p in enumerate(nodes)}
    c=np.diag([float(grounds.get(p,0.)) for p in nodes]);embedding=np.zeros((len(nodes),len(names)+len(resonators)))
    for i,r in enumerate(resonators):c[idx[r],idx[r]]+=(resonator_caps_ff or {})[r];embedding[idx[r],len(names)+i]=1
    def edge(a,b,cap):
        if cap<0:raise ValueError('Capacitances must be nonnegative')
        for p in (a,b):
            if p not in grounded:c[idx[p],idx[p]]+=cap
        if a not in grounded and b not in grounded:c[idx[a],idx[b]]-=cap;c[idx[b],idx[a]]-=cap
    for i,n in enumerate(names):
        a,b=pad_pairs[n];edge(a,b,shunts[n])
        if a not in grounded:embedding[idx[a],i]=1
        if b not in grounded:embedding[idx[b],i]=-1
    for e in links:edge(e['from'],e['to'],e['capacitance_ff'])
    np.linalg.cholesky(c);k=embedding.T@np.linalg.solve(c,embedding)
    return dict(node_order=list(nodes),modal_order=list(names+resonators),nodal_capacitance_ff=c,embedding=embedding,K=k,loaded_ec_ghz=SCALE*np.diag(k)[:len(names)])


def fit_mixed_shunts(pads,grounds,links,targets,grounded_pads=('C2',),resonator_caps_ff=None):
    names=tuple(pads);target=np.array([targets[n] for n in names]);zero={n:0. for n in names}
    # A positive seed matrix is needed only for the matrix topology, not an inverse.
    seed={n:SCALE/targets[n]*.5 for n in names}
    initial=mixed_circuit(pads,grounds,seed,links,grounded_pads,resonator_caps_ff)
    base=initial['nodal_capacitance_ff'].copy();nodes=initial['node_order'];p=initial['embedding'];dirs=[]
    for n in names:
        d=np.zeros(len(nodes))
        for pad,sign in zip(pads[n],[1,-1]):
            if pad in nodes:d[nodes.index(pad)]=sign
        dirs.append(np.outer(d,d));base-=seed[n]*dirs[-1]
    dirs=np.asarray(dirs)
    def ev(x):
        vals=np.exp(x);c=base+np.einsum('i,ijk->jk',vals,dirs);response=np.linalg.solve(c,p);k=p.T@response
        residual=(SCALE*np.diag(k)[:len(names)]-target)/target
        deriv=-np.einsum('ai,nab,bi->in',response,dirs,response)
        return residual,SCALE*deriv[:len(names),:]*vals[None,:]/target[:,None]
    fit=least_squares(lambda x:ev(x)[0],np.log(list(seed.values())),jac=lambda x:ev(x)[1],bounds=(-20,25),xtol=1e-12,ftol=1e-12,gtol=1e-12)
    if max(abs(ev(fit.x)[0]))>1e-8:raise ValueError('Positive mixed shunts cannot reproduce loaded ECs')
    shunts=dict(zip(names,np.exp(fit.x)));return shunts,mixed_circuit(pads,grounds,shunts,links,grounded_pads,resonator_caps_ff)


def mixed_budget(pads,grounds,shunts,links,removed,grounded_pads=('C2',),resonator_caps_ff=None):
    c=mixed_circuit(pads,grounds,shunts,links,grounded_pads,resonator_caps_ff);without=mixed_circuit(pads,grounds,shunts,[e for e in links if e not in removed],grounded_pads,resonator_caps_ff)
    result={}
    for i,n in enumerate(pads):
        a,b=pads[n];g1=grounds.get(a,0.);g2=grounds.get(b,0.)
        own=shunts[n]+(g1 if b in grounded_pads else g2 if a in grounded_pads else g1*g2/(g1+g2))
        ceff=SCALE/c['loaded_ec_ghz'][i]
        result[n]={'loaded_ec_ghz':float(c['loaded_ec_ghz'][i]),'loaded_Ceff_ff':float(ceff),'own_capacitance_ff':float(own),'own_fraction':float(own/ceff),'all_external_fraction':float(1-own/ceff),'specified_links_loading_fraction':float(1-c['loaded_ec_ghz'][i]/without['loaded_ec_ghz'][i])}
    return result


def mixed_model(pads,grounds,shunts,links,frequencies,levels=5,ncut=35):
    c=mixed_circuit(pads,grounds,shunts,links);ec=c['loaded_ec_ghz'];names=tuple(pads)
    ej=np.array([ej_for_frequency(frequencies[n],e,ncut) for n,e in zip(names,ec)]);j=8*SCALE*c['K'];np.fill_diagonal(j,0.)
    return Model(names,ec,ej,j,Basis(ncut=ncut,transmon_levels=levels),{}),c

def mixed_fanout(record,central_node,copies=4):
    """Static physical branch replication; compensate only shared rail-pair shunts."""
    shared=('A1','A2') if central_node=='A2' else ('B1','B2')
    original=record['pad_pairs'];pads={n:original[n] for n in shared};grounds={p:record['grounds_ff'][p] for n in shared for p in original[n]};shunts={n:record['shunts_ff'][n] for n in shared}
    sharedpads={p for n in shared for p in original[n]};links=[dict(e) for e in record['links'] if e['from'] in sharedpads and e['to'] in sharedpads];grounded=[]
    for branch in range(copies):
        remap={p:p for p in sharedpads}
        for n in original:
            if n in shared:continue
            name=f'{n}@{branch}';pads[name]=[f'{p}@{branch}' for p in original[n]];shunts[name]=record['shunts_ff'][n]
            for old,new in zip(original[n],pads[name]):remap[old]=new;grounds[new]=record['grounds_ff'][old]
            if n=='C':grounded.append(pads[name][1])
        for edge in record['links']:
            if edge['from'] in sharedpads and edge['to'] in sharedpads:continue
            links.append(dict(edge,**{'from':remap[edge['from']],'to':remap[edge['to']]}))
    cpads={p for n in pads if n.startswith('C@') for p in pads[n]};remove=[e for e in links if e['from'] in cpads or e['to'] in cpads]
    def report(s):
        budget=mixed_budget(pads,grounds,s,links,remove,grounded)
        return {'shared_rail_pair':{n:budget[n] for n in shared},'branch_loaded_ec_ghz':{n:v['loaded_ec_ghz'] for n,v in budget.items() if n not in shared},'central_shunts_ff':{n:float(s[n]) for n in shared}}
    out={'construction':'Four physical coupler/opposite-pair branches; central rail pair shared. Static budget only.','fixed_hardware':report(shunts)}
    target=np.array([record['targets_ec_ghz'][n] for n in shared]);positions=[list(pads).index(n) for n in shared]
    def residual(x):
        s=dict(shunts);s.update(zip(shared,np.exp(x)));c=mixed_circuit(pads,grounds,s,links,grounded);return (c['loaded_ec_ghz'][positions]-target)/target
    fit=least_squares(residual,np.log([shunts[n] for n in shared]),bounds=(-20,25),xtol=1e-12,ftol=1e-12,gtol=1e-12)
    if max(abs(residual(fit.x)))<1e-8:
        s=dict(shunts);s.update(zip(shared,np.exp(fit.x)));out['shared_pair_EC_compensated']=report(s)
    else:out['compensation_failed']=True
    return out

