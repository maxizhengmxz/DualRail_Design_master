"""Seven-mode sparse Hamiltonian: pair A x C x pair B x r1 x r2."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh
from floating_coupler.pairs import PairModel
from floating_coupler.model import local_spectrum
from floating_coupler.design import ej_for_frequency
from core.transmon import Basis
from scipy.constants import h


def kron(*ops):
    result=sparse.csr_matrix([[1.]])
    for op in ops:result=sparse.kron(result,sparse.csr_matrix(op),format='csr')
    return result


class SevenReadout:
    def __init__(self,model,physical,fc,basis=None,photons=3,eigenlevels=100,coupler_probes=False,off_only=False):
        self.model=model;self.physical=physical;self.fc=fc;self.photons=photons;self.eigenlevels=eigenlevels;self.off_only=off_only
        self.basis=basis or dict(local_levels=6,pair_states=10,coupler_levels=4,ncut=25)
        self.pair=PairModel(model,**self.basis);p=self.basis['pair_states'];c=self.basis['coupler_levels'];r=photons;self.dims=(p,c,p,r,r)
        matter,_=self.pair.hamiltonian(fc);self.matter_ham=matter
        self.matter_assignment={};self.matter_energy=None;self.matter_vectors=None
        if not off_only:
            self.matter_energy,self.matter_vectors=eigh(matter)
            matter_projection=self.matter_vectors[self.pair.ref_indices,:]
            _,assigned=linear_sum_assignment(-abs(matter_projection)**2)
            self.matter_assignment=dict(zip(self.pair.labels,assigned))
        identityp=np.eye(p);identityc=np.eye(c);identityr=np.eye(r);identitym=sparse.eye(p*c*p)
        en,nc=local_spectrum(ej_for_frequency(fc,model.ec[2],self.basis['ncut']),model.ec[2],Basis(ncut=self.basis['ncut'],transmon_levels=c))
        self.matter_charge=[kron(self.pair.a['charge'][i][:p,:p],identityc,identityp) for i in [0,1]]+[kron(identityp,nc,identityp)]+[kron(identityp,identityc,self.pair.b['charge'][i][:p,:p]) for i in [0,1]]
        self.charge=[kron(q,identityr,identityr) for q in self.matter_charge]
        a=np.diag(np.sqrt(np.arange(1,r)),1);nr=np.diag(np.arange(r));q=a+a.T
        ham=kron(matter,identityr,identityr)
        res=physical['resonator_parameters'];self.resnames=list(res)
        for i,name in enumerate(self.resnames):
            pars=res[name];resnr=kron(nr,identityr) if i==0 else kron(identityr,nr);ham+=sparse.kron(identitym,resnr,format='csr')*pars['fr_ghz']
            qm=sum(float(g)*op for g,op in zip(pars['gamma_ghz_per_n'],self.matter_charge));ham+=kron(qm,q,identityr) if i==0 else kron(qm,identityr,q)
        k=np.array(physical['K_per_ff']);g12=k[5,6]*1e15*res[self.resnames[0]]['q_zpf_coulomb']*res[self.resnames[1]]['q_zpf_coulomb']/(h*1e9)
        ham+=kron(identitym,q,q)*g12;self.ham=ham.tocsr()
        self.parity=(self.pair.parity[:,None,None]*((-1.)**np.arange(r))[None,:,None]*((-1.)**np.arange(r))[None,None,:]).ravel()
        self.refs={}
        for label,ai,bi in self.pair.references:
            for ra,rb in [(0,0),(1,0),(0,1)]:self.refs[(label,ra,rb)]=np.ravel_multi_index((ai,0,bi,ra,rb),self.dims)
        if coupler_probes:
            for ci in [1,2]:
                for ra,rb in [(0,0),(1,0),(0,1)]:self.refs[(f'C{ci}',ra,rb)]=np.ravel_multi_index((0,ci,0,ra,rb),self.dims)
            if c>=4:self.refs[('C3',0,0)]=np.ravel_multi_index((0,3,0,0,0),self.dims)
        self.eigen={};self.weights={};self.bare_reference_weights={};self.vectors={};self.canonical={};self._disp_cache={}

    def solve(self):
        keys=list(self.refs)
        if self.off_only:keys=[key for key in keys if key[1:]==(0,0) and key[0] in ('0g','1g','g0','g1')]
        for parity in ([-1] if self.off_only else [1,-1]):
            sector=np.flatnonzero(self.parity==parity);wanted=[key for key in keys if self.parity[self.refs[key]]==parity];positions=np.searchsorted(sector,[self.refs[key] for key in wanted]);block=self.ham[sector,:][:,sector]
            count=min(self.eigenlevels,len(sector)-2)
            # Dense selected spectrum is faster than Lanczos for these small low-energy blocks.
            energy,v=eigh(block.toarray(),subset_by_index=(0,count-1),check_finite=False)
            overlap=v[positions,:];assignment_projection=overlap.copy()
            for i,key in enumerate(wanted):
                if key[0] in self.matter_assignment:
                    reference=np.kron(np.kron(self.matter_vectors[:,self.matter_assignment[key[0]]],np.eye(self.photons)[:,key[1]]),np.eye(self.photons)[:,key[2]])
                    assignment_projection[i]=reference[sector]@v
            _,columns=linear_sum_assignment(-abs(assignment_projection)**2)
            if max(columns)>count-3 and count<len(sector)-2:raise ValueError('Seven-mode eigenlevel window too small')
            for i,key in enumerate(wanted):
                full=np.zeros(self.ham.shape[0]);full[sector]=v[:,columns[i]];self.vectors[key]=full;self.eigen[key]=float(energy[columns[i]]);self.bare_reference_weights[key]=float(abs(overlap[i,columns[i]])**2)
                if key[0] in self.matter_assignment:
                    reference=np.kron(np.kron(self.matter_vectors[:,self.matter_assignment[key[0]]],np.eye(self.photons)[:,key[1]]),np.eye(self.photons)[:,key[2]])
                    self.weights[key]=float(abs(reference@full)**2)
                else:self.weights[key]=self.bare_reference_weights[key]
            self.canonical[parity]=(wanted,energy,v,positions)
        return self

    def effective(self,keys):
        parity=int(self.parity[self.refs[keys[0]]]);sector=np.flatnonzero(self.parity==parity);positions=np.searchsorted(sector,[self.refs[k] for k in keys]);wanted,energy,v,_=self.canonical[parity];projection=v[positions,:];_,cols=linear_sum_assignment(-abs(projection)**2);u,s,vh=np.linalg.svd(projection[:,cols]);rotation=u@vh;ham=(rotation*energy[cols])@rotation.T
        return {'g_mhz':float(ham[0,1]*1000),'detuning_mhz':float((ham[0,0]-ham[1,1])*1000),'minimum_subspace_overlap':float(s.min()**2)}

    def physical_off_detail(self):
        """Physical OFF observable using only the odd sector; readout is deferred."""
        from floating_coupler.off_observables import physical_single_excitation_references
        from floating_coupler.channel_policy import physical_exchange_from_eigenpairs,off_target_definition
        sector=np.flatnonzero(self.parity==-1)
        _,energy,v,_=self.canonical[-1]
        refs=physical_single_excitation_references(self.pair)
        vacuum=np.zeros(self.photons**2);vacuum[0]=1.
        refs=np.kron(refs,vacuum[None,:])
        result=physical_exchange_from_eigenpairs(energy,refs[:,sector]@v,model='seven')
        return dict(physical_exchange=result,off_definition=off_target_definition('seven'))

    def matter_detail(self):
        if self.off_only:raise ValueError("Use a full solve for logical/readout diagnostics")
        from floating_coupler.pairs import local_energy_frame
        from floating_coupler.channel_policy import logical_exchange_from_eigenpairs,off_target_definition,logical_exchange_definition,physical_exchange_from_eigenpairs
        labels=self.pair.labels;matrix=np.zeros((len(labels),len(labels)));overlaps=[];logical_exchange=None
        for parity in [1,-1]:
            keys=[(l,0,0) for l in labels if int(self.parity[self.refs[(l,0,0)]])==parity];sector=np.flatnonzero(self.parity==parity);positions=np.searchsorted(sector,[self.refs[k] for k in keys]);wanted,energy,v,_=self.canonical[parity];projection=v[positions,:];_,cols=linear_sum_assignment(-abs(projection)**2);u,s,vh=np.linalg.svd(projection[:,cols]);rot=u@vh;hm=(rot*energy[cols])@rot.T;which=[labels.index(k[0]) for k in keys];matrix[np.ix_(which,which)]=hm;overlaps.append(float(s.min()**2))
        wanted,energy,v,_=self.canonical[1]
        sector=np.flatnonzero(self.parity==1)
        positions=np.searchsorted(sector,[self.refs[(label,0,0)] for label in ['00','01','10','11']])
        logical_exchange=logical_exchange_from_eigenpairs(energy,v[positions,:])
        logical_exchange['observable_definition']=logical_exchange_definition('seven')
        physical_exchange=self.physical_off_detail()['physical_exchange']
        logical=[labels.index(l) for l in ['00','01','10','11']]
        return {'physical_exchange':physical_exchange,'logical_exchange':logical_exchange,'off_definition':off_target_definition('seven'),'reference_labels':labels,'effective_hamiltonian_ghz':matrix.tolist(),'logical_local_energy_frame':local_energy_frame(matrix[np.ix_(logical,logical)]),'minimum_subspace_overlap':min(overlaps),'channel_selection':self.model.electrostatics.get('channel_selection')}

    def dispersive(self,pair):
        if pair in self._disp_cache:return self._disp_cache[pair]
        resindex=0 if pair=='A' else 1;pars=self.physical['resonator_parameters'][self.resnames[resindex]];op=sum(float(g)*q for g,q in zip(pars['gamma_ghz_per_n'],self.matter_charge));v=self.matter_vectors;energy=self.matter_energy;coupling=abs(v.T@(op@v));rows=[]
        for spectator in [0,1]:
            labels=([f'g{spectator}']+[f'{i}{spectator}' for i in [0,1]]) if pair=='A' else ([f'{spectator}g']+[f'{spectator}{i}' for i in [0,1]])
            worst=(0.,None,None,None)
            for label in labels:
                i=self.matter_assignment[label]
                for j in range(len(energy)):
                    if energy[j]<=energy[i]+1e-9 or coupling[i,j]<1e-9:continue
                    delta=energy[j]-energy[i]-pars['fr_ghz'];ratio=abs(coupling[i,j]/delta)
                    if ratio>worst[0]:worst=(float(ratio),label,j,float(delta))
            rows.append({'spectator_logical':spectator,'lambda_max':worst[0],'worst_initial':worst[1],'worst_final_index':worst[2],'worst_detuning_ghz':worst[3]})
        self._disp_cache[pair]=rows
        return rows

    def readout(self,pair):
        idx=[0,1] if pair=='A' else [3,4];phot=(1,0) if pair=='A' else (0,1);rows=[]
        for spectator in [0,1]:
            ground=f'g{spectator}' if pair=='A' else f'{spectator}g';logical=[f'{i}{spectator}' if pair=='A' else f'{spectator}{i}' for i in [0,1]];base=(ground,0,0);groundfreq=self.eigen[(ground,*phot)]-self.eigen[base];chis=[];energies=[];weights=[self.weights[base],self.weights[(ground,*phot)]]
            for label in logical:
                chis.append(self.eigen[(label,*phot)]-self.eigen[(label,0,0)]-groundfreq);energies.append(self.eigen[(label,0,0)]-self.eigen[base]);weights.extend([self.weights[(label,0,0)],self.weights[(label,*phot)]])
            strengths=np.array([[abs(self.vectors[base]@(self.charge[i]@self.vectors[(label,0,0)]))**2 for label in logical] for i in idx]);participation=strengths/strengths.sum(axis=1,keepdims=True);hybrid=1-2*np.mean(abs(participation-.5));mean=sum(chis)/2
            rows.append({'spectator_logical':spectator,'operating_frequency_ghz':float(sum(energies)/2),'splitting_ghz':float(abs(energies[1]-energies[0])),'chi_m_ghz':float(chis[0]),'chi_p_ghz':float(chis[1]),'ratio':float(abs((chis[1]-chis[0])/mean)) if abs(mean)>1e-15 else float('inf'),'hybridization':float(hybrid),'min_assignment_overlap':float(min(weights)),'dressed_r_ghz':float(groundfreq),'logical_mean_readout_ghz':float(groundfreq+mean),'lambda_max':self.dispersive(pair)[spectator]['lambda_max']})
        return rows
