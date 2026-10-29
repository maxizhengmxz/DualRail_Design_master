"""Dressed-matter coupler/readout matrix elements and multilevel dispersive screen."""
import sys,json,argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrated_readout_range import frozen_model
from seven_readout import SevenReadout
from floating_coupler.design import ej_for_frequency


def probe(cell,fc,fb,fa=None):
    model=frozen_model(cell['physical']);ej=model.ej.copy()
    fixed=cell.get('request',{}).get('idle_bare_frequencies_ghz',cell['physical']['bare_frequencies_ghz'])
    a_frequencies=[fixed[n] for n in ['A1','A2']] if fa is None else [fa,fa]
    for i,f in [(0,a_frequencies[0]),(1,a_frequencies[1]),(3,fb),(4,fb)]:ej[i]=ej_for_frequency(f,model.ec[i],25)
    model=replace(model,ej=ej);audits=[]
    for basis in [dict(local_levels=6,pair_states=10,coupler_levels=4,ncut=25),dict(local_levels=6,pair_states=12,coupler_levels=5,ncut=35)]:
        s=SevenReadout(model,cell['physical'],fc,basis=basis);energy=s.matter_energy;v=s.matter_vectors;p=basis['pair_states'];c=basis['coupler_levels'];refs=[np.ravel_multi_index((0,k,0),(p,c,p)) for k in range(4)];_,cols=linear_sum_assignment(-abs(v[refs,:])**2);res={}
        for name,pars in cell['physical']['resonator_parameters'].items():
            q=sum(float(g)*op for g,op in zip(pars['gamma_ghz_per_n'],s.matter_charge));couplings=abs(v.T@(q@v));transitions=[];worst=[]
            for i in range(3):
                a,b=cols[i],cols[i+1];f=float(energy[b]-energy[a]);g=float(couplings[a,b]*1000);delta=(f-pars['fr_ghz'])*1000;transitions.append({'transition':f'{i}-{i+1}','matter_transition_frequency_ghz':f,'g_mhz':g,'detuning_mhz':delta,'abs_g_over_delta':abs(g/delta),'initial_bare_C_weight':float(v[refs[i],a]**2),'final_bare_C_weight':float(v[refs[i+1],b]**2)})
                ratios=[(abs(couplings[a,b]/(energy[b]-energy[a]-pars['fr_ghz'])),b,float(energy[b]-energy[a]-pars['fr_ghz'])) for b in range(len(energy)) if energy[b]>energy[a] and couplings[a,b]>1e-9];maximum=max(ratios);worst.append({'initial_C_branch':i,'maximum_upward_ratio':float(maximum[0]),'final_matter_index':int(maximum[1]),'detuning_ghz':maximum[2]})
            res[name]={'coupler_transitions':transitions,'all_upward_transition_worst':worst}
        audits.append({'basis':basis,'resonators':res})
    return {'fc_on_ghz':fc,'fb_gate_ghz':fb,'fa_gate_ghz':fa,'fa_gate_bare_frequencies_ghz':a_frequencies,'audits':audits,'definition':'g=<Ci|sum_j gamma_jr*n_j|Ci+1> using exact five-transmon matter eigenstates at gate controls, with rail ground references. Delta uses the loaded oscillator frequency; both readout oscillators are retained in the complete circuit electrostatics. Not a driven-readout or CZ fidelity certification.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--fc',type=float,required=True);p.add_argument('--fb',type=float,required=True);p.add_argument('--fa',type=float);p.add_argument('--output',required=True);a=p.parse_args();result=probe(json.loads(Path(a.cell).read_text()),a.fc,a.fb,a.fa);Path(a.output).write_text(json.dumps(result,indent=2))
