"""Physical three-body ZZ diagnostics, distinct from five-body logical ZZ.

Reusable numerical checks only: no saved-case defaults, scans, root campaigns,
or compulsory large-basis audits. ZZ remains a residual, never the OFF goal.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from .design import ej_for_frequency as frequency_ej

def physical_zz(model,fc):
    ham,_=model.hamiltonian(frequency_ej(float(fc),float(model.ec[1]),model.basis.ncut));energy,v=eigh(ham)
    n=model.basis.transmon_levels;states=[(0,0,0),(1,0,0),(0,0,1),(1,0,1)];refs=[np.ravel_multi_index(s,(n,n,n)) for s in states]
    projection=abs(v[refs,:])**2;_,columns=linear_sum_assignment(-projection);weights=projection[np.arange(4),columns];assigned=energy[columns]
    if min(weights)<.8:raise ValueError('Physical state assignment unreliable')
    return {'fc_ghz':float(fc),'ej_c_ghz':frequency_ej(float(fc),float(model.ec[1]),model.basis.ncut),'physical_zz_hz':float((assigned[3]-assigned[1]-assigned[2]+assigned[0])*1e9),'minimum_assignment_weight':float(min(weights))}


def zz_derivatives(model,fc,step=.001):
    """Hellmann-Feynman eigenvalue derivatives of the smooth truncated H(fc).

    Five-point matrix differences differentiate the local-basis H, including
    charge-matrix and local-basis changes. The eigenvalue second derivative
    includes mixing with every other retained eigenstate.
    """
    def ham(f):return model.hamiltonian(frequency_ej(float(f),float(model.ec[1]),model.basis.ncut))[0]
    h0=ham(fc);energy,v=eigh(h0);hm2,hm1,hp1,hp2=[ham(fc+k*step) for k in [-2,-1,1,2]]
    dh=(hm2-8*hm1+8*hp1-hp2)/(12*step)
    ddh=(-hp2+16*hp1-30*h0+16*hm1-hm2)/(12*step**2)
    n=model.basis.transmon_levels;states=[(0,0,0),(1,0,0),(0,0,1),(1,0,1)];refs=[np.ravel_multi_index(s,(n,n,n)) for s in states]
    weights=abs(v[refs,:])**2;_,cols=linear_sum_assignment(-weights);assigned=energy[cols];signs=np.array([1,-1,-1,1]);first=[];second=[]
    for k in cols:
        w=v[:,k];couplings=v.T@dh@w;denom=energy[k]-energy;mask=np.arange(len(energy))!=k
        if np.any(abs(denom[mask])<1e-10):raise ValueError('Degenerate tracked eigenstate: derivative requires a degenerate treatment')
        first.append(w@dh@w);second.append(w@ddh@w+2*np.sum(couplings[mask]**2/denom[mask]))
    dejdf=(frequency_ej(float(fc+step),float(model.ec[1]),model.basis.ncut)-frequency_ej(float(fc-step),float(model.ec[1]),model.basis.ncut))/(2*step)
    return {'fc_ghz':float(fc),'ej_c_ghz':frequency_ej(float(fc),float(model.ec[1]),model.basis.ncut),'zz_hz':float(signs@assigned*1e9),'dzz_df_hz_per_ghz':float(signs@np.array(first)*1e9),'d2zz_df2_hz_per_ghz2':float(signs@np.array(second)*1e9),'dzz_dEJ_hz_per_ghz':float(signs@np.array(first)*1e9/dejdf),'minimum_assignment_weight':float(weights[np.arange(4),cols].min())}

