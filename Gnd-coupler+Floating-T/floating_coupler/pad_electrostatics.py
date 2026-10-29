"""Explicit transmon pads, ground capacitors and ground-referenced resonators.

Physical mutual capacitors are matrix edges, never effective/mediated values.
All capacitances are fF; energies are E/h in GHz. In the sector with zero
conserved total charge on each transmon, pad charges are (+q,-q). Retain every
resonator signal charge explicitly. K=P.T C_node^-1 P is the complete inverse
capacitance for those charges. The differential effective capacitance is
K^-1, not a sub-block of C_node with common voltages artificially grounded.

The current model fixes equal bare pad grounds and fits a geometric pad-to-pad
shunt. The older fit_balanced_pad_grounds function is retained for the previous
equal-PHYSICAL-diagonal convention; it is not the current synthesis default.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
from scipy.optimize import least_squares
from .model import SCALE


@dataclass
class PadCircuit:
    nodes: tuple
    transmons: tuple
    resonators: tuple
    pad_pairs: dict
    ground_capacitances_ff: dict
    junction_capacitances_ff: dict
    physical_links: list
    nodal_capacitance_ff: np.ndarray
    charge_embedding: np.ndarray
    inverse_modal_capacitance_per_ff: np.ndarray

    @property
    def pad_mutual_capacitances_ff(self):
        """Across-pad shunts; legacy storage name is not intrinsic JJ C alone."""
        return self.junction_capacitances_ff

    @property
    def loaded_ec_ghz(self):
        k=self.inverse_modal_capacitance_per_ff
        return {name:float(SCALE*k[i,i]) for i,name in enumerate(self.transmons)}

    @property
    def effective_modal_capacitance_ff(self):
        return np.linalg.inv(self.inverse_modal_capacitance_per_ff)

    @property
    def charge_coefficients_ghz(self):
        """8*SCALE*K; use transmon off-diagonals as J in J*n_i*n_j.

        Resonator charges are dimensional and must use their own zero-point
        charges in H, not this Cooper-pair-normalized coefficient blindly.
        """
        return 8*SCALE*self.inverse_modal_capacitance_per_ff

    def summary(self):
        diag=np.diag(self.nodal_capacitance_ff)
        return dict(node_order=list(self.nodes),modal_order=list(self.transmons+self.resonators),
                    ground_capacitances_ff=self.ground_capacitances_ff,
                    across_pad_capacitances_ff=self.pad_mutual_capacitances_ff,
                    physical_links=self.physical_links,loaded_ec_ghz=self.loaded_ec_ghz,
                    pad_total_capacitances_ff={name:[float(diag[self.nodes.index(p)]) for p in self.pad_pairs[name]]
                                              for name in self.transmons},
                    nodal_capacitance_ff=self.nodal_capacitance_ff.tolist(),
                    inverse_modal_capacitance_per_ff=self.inverse_modal_capacitance_per_ff.tolist(),
                    effective_modal_capacitance_ff=self.effective_modal_capacitance_ff.tolist())


def nonnegative(value,label):
    if value is None or not np.isfinite(value) or value<0:
        raise ValueError(f'{label} requires explicit finite nonnegative fF')
    return float(value)


def build_pad_circuit(pad_pairs:Mapping,ground_capacitances_ff:Mapping,
                      resonator_capacitances_ff:Mapping,physical_links,
                      junction_capacitances_ff:Mapping|None=None):
    """Construct the complete physical matrix without fitting or retuning.

    Pad order fixes the differential voltage sign. Intrinsic/across-pad
    capacitances are separate explicit inputs; zero is the agreed ideal limit.
    There are no automatically added links or parasitic ground capacitors.
    """
    names=tuple(pad_pairs);pairs={n:tuple(pad_pairs[n]) for n in names}
    if any(len(p)!=2 or p[0]==p[1] for p in pairs.values()):
        raise ValueError('Each transmon requires two distinct pad labels')
    resonators=tuple(resonator_capacitances_ff)
    nodes=tuple(p for n in names for p in pairs[n])+resonators
    if len(nodes)!=len(set(nodes)):raise ValueError('Pad/resonator labels must be unique')
    pad_nodes=nodes[:2*len(names)]
    if set(ground_capacitances_ff)!=set(pad_nodes):
        raise ValueError('Supply a ground capacitor for every transmon pad, and no extra pads')
    grounds={p:nonnegative(ground_capacitances_ff[p],p+' ground capacitance') for p in pad_nodes}
    jj={n:nonnegative((junction_capacitances_ff or {}).get(n,0.),n+' junction capacitance') for n in names}
    if junction_capacitances_ff is not None and set(junction_capacitances_ff)!=set(names):
        raise ValueError('Supply the junction-capacitance mapping for all transmons')
    index={n:i for i,n in enumerate(nodes)};c=np.zeros((len(nodes),len(nodes)))
    for pad,value in grounds.items():c[index[pad],index[pad]]+=value
    for node,value in resonator_capacitances_ff.items():
        value=nonnegative(value,node+' resonator capacitance')
        if value<=0:raise ValueError('Each grounded resonator needs positive self capacitance')
        c[index[node],index[node]]+=value
    def add(a,b,value):
        if a not in index or b not in index or a==b:raise ValueError('Invalid physical capacitor endpoints')
        i,j=index[a],index[b];c[i,i]+=value;c[j,j]+=value;c[i,j]-=value;c[j,i]-=value
    for n in names:add(*pairs[n],jj[n])
    links=[];seen=set();jj_pairs={frozenset(p) for p in pairs.values()}
    for item in physical_links:
        a,b=item['from'],item['to'];edge=frozenset((a,b))
        if edge in seen or edge in jj_pairs:raise ValueError('Duplicate or junction edge in physical-link list')
        seen.add(edge);value=nonnegative(item['capacitance_ff'],f'{a}-{b}')
        add(a,b,value);links.append(dict(item,capacitance_ff=value))
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as error:
        raise ValueError('Grounded physical capacitance matrix is not positive definite') from error
    p=np.zeros((len(nodes),len(names)+len(resonators)))
    for i,n in enumerate(names):p[index[pairs[n][0]],i]=1;p[index[pairs[n][1]],i]=-1
    for i,n in enumerate(resonators,start=len(names)):p[index[n],i]=1
    k=p.T@np.linalg.solve(c,p)
    return PadCircuit(nodes,names,resonators,pairs,grounds,jj,links,c,p,k)


def fit_balanced_pad_grounds(pad_pairs,loaded_ec_ghz,resonator_capacitances_ff,
                             physical_links,junction_capacitances_ff=None):
    """Fit positive pad grounds to exact loaded ECs AND equal physical pad loads.

    One free mean ground capacitance per transmon remains after analytically
    balancing attached physical loads. All means are fitted jointly using the
    full inverse matrix, retaining resonator and non-neighbor couplings.
    """
    names=tuple(pad_pairs)
    if set(loaded_ec_ghz)!=set(names):raise ValueError('Provide loaded EC for every transmon')
    target=np.array([loaded_ec_ghz[n] for n in names],float)
    if not np.all(np.isfinite(target)) or np.any(target<=0):raise ValueError('Loaded ECs must be finite positive GHz')
    attached={p:0. for n in names for p in pad_pairs[n]}
    for edge in physical_links:
        value=nonnegative(edge['capacitance_ff'],'physical link')
        for endpoint in (edge['from'],edge['to']):
            if endpoint in attached:attached[endpoint]+=value
    difference=np.array([(attached[pad_pairs[n][1]]-attached[pad_pairs[n][0]])/2 for n in names])
    lower=np.abs(difference)+1e-9
    jj=np.array([(junction_capacitances_ff or {}).get(n,0.) for n in names])
    def circuit(log_excess):
        means=lower+np.exp(log_excess)
        grounds={p:float(means[i]+sign*difference[i])
                 for i,n in enumerate(names) for p,sign in zip(pad_pairs[n],(1,-1))}
        return build_pad_circuit(pad_pairs,grounds,resonator_capacitances_ff,physical_links,junction_capacitances_ff)
    def residual(x):
        result=circuit(x)
        return (np.array(list(result.loaded_ec_ghz.values()))-target)/target
    initial=np.maximum(2*(SCALE/target-jj)-lower,1.)
    fit=least_squares(residual,np.log(initial),bounds=(-20.,25.),
                      xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=150)
    result=circuit(fit.x)
    if not fit.success or max(abs(residual(fit.x)))>1e-8:
        raise ValueError('Cannot match loaded EC and pad balance with positive ground capacitances')
    return result


def fit_pad_mutual_capacitances(pad_pairs,loaded_ec_ghz,ground_capacitances_ff,
                                resonator_capacitances_ff,physical_links):
    """Fix actual ground capacitors; fit positive across-pad shunts to loaded EC.

    Across-pad shunts only add to the differential diagonal AFTER the common
    coordinates are eliminated. Reusing this identity avoids nested full-pad
    inversions in geometry searches. The final physical matrix is rebuilt and
    checked independently. The shunt includes geometric pad mutual C; intrinsic
    junction C may be neglected without removing this shunt.
    """
    names=tuple(pad_pairs);count=len(names)
    target=np.array([loaded_ec_ghz[n] for n in names],float)
    if np.any(~np.isfinite(target)) or np.any(target<=0):raise ValueError('Loaded ECs must be finite positive GHz')
    baseline=build_pad_circuit(pad_pairs,ground_capacitances_ff,resonator_capacitances_ff,
                               physical_links,{n:0. for n in names})
    base=baseline.effective_modal_capacitance_ff
    def evaluate(x):
        caps=np.exp(x);matrix=base.copy();matrix[np.arange(count),np.arange(count)]+=caps
        inverse=np.linalg.inv(matrix)
        residual=(SCALE*np.diag(inverse)[:count]-target)/target
        jac=-SCALE*inverse[:count,:count]**2*caps[None,:]/target[:,None]
        return residual,jac
    initial=np.maximum(SCALE/target-np.diag(base)[:count],.1)
    fit=least_squares(lambda x:evaluate(x)[0],np.log(initial),jac=lambda x:evaluate(x)[1],
                      bounds=(-20.,25.),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=80)
    if not fit.success or max(abs(evaluate(fit.x)[0]))>1e-8:
        raise ValueError('Cannot match loaded EC with positive pad-to-pad capacitances at the fixed grounds')
    result=build_pad_circuit(pad_pairs,ground_capacitances_ff,resonator_capacitances_ff,
                              physical_links,dict(zip(names,np.exp(fit.x))))
    if max(abs(np.array(list(result.loaded_ec_ghz.values()))/target-1))>1e-8:
        raise ValueError('Physical-pad reconstruction disagrees with reduced capacitance fit')
    return result


def pad_capacitance_budget(circuit,remove_links=None):
    """Own ground, own pad-mutual and external fractions of loaded Ceff.

    Own contributions are defined by removing ALL inter-device capacitors at
    fixed physical ground/shunt capacitors. An optional specified set supplies
    a separate net-loading counterfactual; fractions are not linear EC shares.
    """
    removed=[] if remove_links is None else list(remove_links)
    selected={frozenset((e['from'],e['to'])) for e in removed}
    without=build_pad_circuit(circuit.pad_pairs,circuit.ground_capacitances_ff,
             {n:float(circuit.nodal_capacitance_ff[circuit.nodes.index(n),circuit.nodes.index(n)]-
                 sum(e['capacitance_ff'] for e in circuit.physical_links if n in (e['from'],e['to'])))
              for n in circuit.resonators},
             [e for e in circuit.physical_links if frozenset((e['from'],e['to'])) not in selected],
             circuit.junction_capacitances_ff) if removed else circuit
    result={}
    for n in circuit.transmons:
        g1,g2=[circuit.ground_capacitances_ff[p] for p in circuit.pad_pairs[n]]
        ground=g1*g2/(g1+g2);mutual=circuit.junction_capacitances_ff[n]
        loaded=SCALE/circuit.loaded_ec_ghz[n];counterfactual=SCALE/without.loaded_ec_ghz[n]
        result[n]=dict(ground_capacitances_ff=[g1,g2],pad_mutual_capacitance_ff=mutual,
                       ground_series_equivalent_ff=ground,loaded_effective_capacitance_ff=loaded,
                       ground_fraction=ground/loaded,pad_mutual_fraction=mutual/loaded,
                       own_total_fraction=(ground+mutual)/loaded,
                       all_external_fraction=1-(ground+mutual)/loaded,
                       specified_links_net_loading_fraction=1-counterfactual/loaded)
    return result


def fit_pad_shunts_with_own_ground_fraction(pad_pairs,loaded_ec_ghz,ground_capacitances_ff,
                                           physical_links,own_ground_fractions,ground_ratios=None):
    """Fit loaded EC with specified own-ground shares and optional ground ratios.

    By default selected transmons have equal grounds; ground_ratios[n] sets Cg2/Cg1. For equal grounds, fit a common capacitor G on both pads and
    set Cm=G*(1-eta)/(2*eta). For others, retain their fixed grounds and fit Cm.
    Thus eta=(G/2)/(Cm+G/2) is an own-capacitance ratio, not a loaded fraction.
    All optimized components are physical positive capacitors.
    """
    names=tuple(pad_pairs);target=np.array([loaded_ec_ghz[n] for n in names])
    selected=set(own_ground_fractions)
    ratios={n:float((ground_ratios or {}).get(n,1.)) for n in selected}
    if any(not np.isfinite(r) or r<=0 for r in ratios.values()):raise ValueError("Ground ratios must be positive")
    if not selected.issubset(names):raise ValueError('Unknown transmon in own-ground fraction map')
    if any(not 0<e<1 for e in own_ground_fractions.values()):raise ValueError('Own-ground ratios must be strictly between zero and one')
    initial_grounds=dict(ground_capacitances_ff)
    for n in selected:
        g=own_ground_fractions[n]*SCALE/loaded_ec_ghz[n]*(1+ratios[n])/ratios[n]
        initial_grounds[pad_pairs[n][0]]=g
        initial_grounds[pad_pairs[n][1]]=ratios[n]*g
    initial=build_pad_circuit(pad_pairs,initial_grounds,{},physical_links,{n:0. for n in names})
    base=initial.nodal_capacitance_ff.copy();embedding=initial.charge_embedding
    directions=[];starts=[]
    for n in names:
        d=np.zeros(len(initial.nodes));pp,pm=[initial.nodes.index(p) for p in pad_pairs[n]];d[pp]=1;d[pm]=-1
        if n in selected:
            eta=own_ground_fractions[n];r=ratios[n];series=r/(1+r);t=series*(1-eta)/eta
            diagonal=np.zeros(len(initial.nodes));diagonal[pp]=1;diagonal[pm]=r
            base[pp,pp]-=initial_grounds[pad_pairs[n][0]];base[pm,pm]-=initial_grounds[pad_pairs[n][1]]
            directions.append(np.diag(diagonal)+t*np.outer(d,d));starts.append(eta*SCALE/loaded_ec_ghz[n]/series)
        else:
            g1,g2=[initial_grounds[p] for p in pad_pairs[n]]
            directions.append(np.outer(d,d));starts.append(max(SCALE/loaded_ec_ghz[n]-g1*g2/(g1+g2),.1))
    directions=np.asarray(directions)
    def evaluate(x):
        values=np.exp(x);matrix=base+np.einsum('n,nij->ij',values,directions)
        response=np.linalg.solve(matrix,embedding);k=embedding.T@response
        residual=(SCALE*np.diag(k)-target)/target
        derivative=-np.einsum('ai,nab,bi->in',response,directions,response)
        jac=SCALE*derivative*values[None,:]/target[:,None]
        return residual,jac
    fit=least_squares(lambda x:evaluate(x)[0],np.log(starts),jac=lambda x:evaluate(x)[1],bounds=(-20.,25.),
                      xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=100)
    if not fit.success or max(abs(evaluate(fit.x)[0]))>1e-8:raise ValueError('Positive equal-ground/own-ratio fit failed')
    grounds=dict(initial_grounds);mutual={}
    for n,value in zip(names,np.exp(fit.x)):
        if n in selected:
            grounds[pad_pairs[n][0]]=float(value)
            grounds[pad_pairs[n][1]]=float(value*ratios[n])
            mutual[n]=float(value*ratios[n]/(1+ratios[n])*(1-own_ground_fractions[n])/own_ground_fractions[n])
        else:mutual[n]=float(value)
    return build_pad_circuit(pad_pairs,grounds,{},physical_links,mutual)

