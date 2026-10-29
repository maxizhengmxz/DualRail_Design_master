"""Grounded-C2 five-mode capacitor refinement with immutable loaded rail ECs.

The three coupler capacitors seed the search. Positive physical shunts and
intra-rail capacitors are solved simultaneously for loaded ECs and the required
idle pair charge coefficients; no EC drift is absorbed into an EJ adjustment.
"""
from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares

from core.transmon import Basis
from .model import FULL, SCALE, Model
from .mixed_electrostatics import mixed_circuit, mixed_budget
from .design import ej_for_frequency
from .pairs import PairModel
from .on_gate import energy_axis_matter_hamiltonian, secular_logical_shifts
from .paired_design import channel_scores,off_channel
from .channel_policy import require_selection,off_target_definition


class CompensatedCircuit:
    def __init__(self, request, initial_record, initial_model):
        self.request=request
        self.pads=initial_record['pad_pairs']
        self.grounds=initial_record['grounds_ff']
        self.frequencies=request['idle_bare_frequencies_ghz']
        self.target_j=np.array([initial_model.j[0,1],initial_model.j[3,4]])
        self.initial=np.array([initial_record['shunts_ff'][n] for n in FULL]
                              +initial_record['intra_pair_caps_ff'])
        empty=[dict(**{'from':'A12','to':'A21'},capacitance_ff=0.),
               dict(**{'from':'B12','to':'B21'},capacitance_ff=0.)]
        c=mixed_circuit(self.pads,self.grounds,{n:1. for n in FULL},empty)
        self.nodes=c['node_order'];self.embedding=c['embedding']
        self.base=c['nodal_capacitance_ff'];index={p:i for i,p in enumerate(self.nodes)}
        edges=[self.pads[n] for n in FULL]+[['A12','A21'],['B12','B21']]
        self.directions=[]
        for a,b in edges:
            v=np.zeros(len(self.nodes))
            if a in index:v[index[a]]+=1
            if b in index:v[index[b]]-=1
            self.directions.append(np.outer(v,v))
        self.directions=np.array(self.directions)
        self.base=self.base-self.directions[:5].sum(axis=0)
        self.link_directions=[]
        for a,b in [('A22','C1'),('C1','B11'),('A22','B11')]:
            v=np.zeros(len(self.nodes));v[index[a]]=1;v[index[b]]=-1
            self.link_directions.append(np.outer(v,v))
        self.link_directions=np.array(self.link_directions)

    def fit(self, caps, ec_c):
        target=np.array([self.request['loaded_ec_ghz'][n] if n!='C' else ec_c for n in FULL])
        base=self.base+np.einsum('i,ijk->jk',caps,self.link_directions)
        def evaluate(logcaps):
            values=np.exp(logcaps)
            c=base+np.einsum('i,ijk->jk',values,self.directions)
            response=np.linalg.solve(c,self.embedding);k=self.embedding.T@response
            dk=-np.einsum('ai,nab,bj->ijn',response,self.directions,response)
            residual=np.r_[(SCALE*np.diag(k)-target)/target,
                            (8*SCALE*k[0,1]-self.target_j[0])/abs(self.target_j[0]),
                            (8*SCALE*k[3,4]-self.target_j[1])/abs(self.target_j[1])]
            jac=np.vstack([SCALE*dk[np.arange(5),np.arange(5),:]/target[:,None],
                           8*SCALE*dk[0,1,:]/abs(self.target_j[0]),
                           8*SCALE*dk[3,4,:]/abs(self.target_j[1])])*values
            return residual,jac,k
        fit=least_squares(lambda x:evaluate(x)[0],np.log(self.initial),
                          jac=lambda x:evaluate(x)[1],bounds=(-20,25),
                          xtol=2e-12,ftol=2e-12,gtol=2e-12,max_nfev=40)
        residual,_,k=evaluate(fit.x)
        if max(abs(residual))>2e-9:raise ValueError('No positive EC/pair-splitting compensation')
        values=np.exp(fit.x)
        links=[dict(**{'from':a,'to':b},capacitance_ff=float(cap))
               for (a,b),cap in zip([('A22','C1'),('C1','B11'),('A22','B11'),
                                    ('A12','A21'),('B12','B21')],list(caps)+list(values[5:]),strict=True)]
        shunts={n:float(v) for n,v in zip(FULL,values[:5])}
        rail_links=mixed_budget(self.pads,self.grounds,shunts,links,links[:2])
        full_branch=mixed_budget(self.pads,self.grounds,shunts,links,links[:3])
        j=8*SCALE*k;np.fill_diagonal(j,0.)
        # ECs are compensated to these targets; do not perturb EJ to mask EC drift.
        ec=SCALE*np.diag(k)
        return ec,j,dict(pad_pairs=self.pads,grounds_ff=self.grounds,shunts_ff=shunts,
                    links=links,targets_ec_ghz=dict(zip(FULL,map(float,target))),
                    idle_frequencies_ghz=self.frequencies,intra_pair_caps_ff=values[5:].tolist(),
                    rail_link_budget=rail_links,full_branch_budget=full_branch,
                    maximum_relative_EC_fit_error=float(max(abs((ec-target)/target))),
                    pair_charge_coefficients_ghz=[float(j[0,1]),float(j[3,4])])


class DesignEvaluator:
    """x = Ca,Cb,Cdirect (fF), ECc (GHz), fON,fOFF,fB_gate (GHz)."""
    def __init__(self, circuit, selection, basis):
        self.circuit=circuit;self.selection=require_selection(selection)
        self.basis=basis;self.calls=0
        self.rail_ej={n:ej_for_frequency(circuit.frequencies[n],circuit.request['loaded_ec_ghz'][n],basis['ncut'])
                      for n in FULL if n!='C'}

    @lru_cache(maxsize=64)
    def evaluate(self, x):
        self.calls+=1
        ca,cb,cd,ec_c,fc_on,fc_off,fb=x
        if not np.isfinite(x).all() or not 0<fc_on<fc_off or min(ca,cb,ec_c,fb)<=0 or cd<0:
            raise ValueError('Invalid capacitance/frequency ordering before quantum solve')
        ec,j,physical=self.circuit.fit([ca,cb,cd],ec_c)
        rail_limit=self.circuit.request.get('rail_loading_per_coupler_max',.1)
        own_min=1-self.circuit.request.get('coupler_loading_max',.45)
        if any(physical['full_branch_budget'][n]['specified_links_loading_fraction']>rail_limit+1e-10
               for n in ('A2','B1')):
            raise ValueError('Rail loading limit exceeded before quantum solve')
        if physical['rail_link_budget']['C']['own_fraction']<own_min-1e-10:
            raise ValueError('Coupler loading limit exceeded before quantum solve')
        ej=np.array([self.rail_ej[n] if n!='C' else ej_for_frequency(fc_off,ec_c,self.basis['ncut']) for n in FULL])
        model=Model(FULL,ec,ej,j,Basis(ncut=self.basis['ncut'],transmon_levels=self.basis['local_levels']),
                    {'channel_selection':self.selection.to_dict()})
        idle_system=PairModel(model,**self.basis);off=idle_system.evaluate(fc_off)
        ej_on=ej.copy()
        for i in (3,4):ej_on[i]=ej_for_frequency(fb,ec[i],self.basis['ncut'])
        on=PairModel(replace(model,ej=ej_on),**self.basis).evaluate(fc_on)
        labels=on['reference_labels'];i,k=labels.index(self.selection.source_logical),labels.index(self.selection.auxiliary)
        h_on=energy_axis_matter_hamiltonian(on);h_off=energy_axis_matter_hamiltonian(off)
        sec=secular_logical_shifts(off)
        scores=on['selected_channel_score_diagnostics']
        metrics=dict(g_on_mhz=float(h_on[i,k]*1000),detuning_on_mhz=float((h_on[i,i]-h_on[k,k])*1000),
                     g_off_mhz=off_channel(off)['g_mhz'],detuning_off_mhz=off_channel(off)['detuning_mhz'],
                     logical_exchange_off_mhz=off['logical_exchange']['g_mhz'],
                     logical_double_flip_off_mhz=off['logical_exchange']['local_energy_frame']['double_flip_mhz'],
                     selected_cz_g_off_mhz=float(h_off[i,k]*1000),
                     selected_cz_detuning_off_mhz=float((h_off[i,i]-h_off[k,k])*1000),
                     logical_zz_hz=sec['secular_zz_hz'],secular_well_identified=sec['well_identified'],
                     on_maximum_unintended_ratio=scores['maximum_logical_ratio'],on_S1=scores['S1'],
                     on_minimum_overlap=on['minimum_subspace_overlap'],off_minimum_overlap=off['minimum_subspace_overlap'],
                     rail_loading_sum=sum(physical['full_branch_budget'][n]['specified_links_loading_fraction'] for n in ('A2','B1')),
                     own_coupler_fraction=physical['rail_link_budget']['C']['own_fraction'])
        return dict(off_definition=off_target_definition('five'),x=list(x),metrics=metrics,physical=physical,on=on,off=off,secular=sec,
                    pair_splittings_mhz=[idle_system.a['splitting_mhz'],idle_system.b['splitting_mhz']])

    def __call__(self, x):return self.evaluate(tuple(map(float,x)))
