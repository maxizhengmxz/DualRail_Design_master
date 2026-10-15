"""Independent checks of reported AWS parameters; no notebook inputs changed.

2023: reproduce the specified three-mode rotating-wave Duffing Hamiltonian,
not a fitted physical capacitance network. 2024: infer single-transmon cosine
parameters from f01 and anharmonicity, neglecting network dressing corrections.
2026: keep unpublished EJ, EC and transverse couplings explicitly unknown.
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares, linear_sum_assignment
from threadpoolctl import threadpool_limits

from model import Basis, _transmon


def infer_cosine(f01_ghz, anharmonicity_magnitude_ghz):
    basis=Basis(ncut=25,transmon_levels=4,resonator_levels=4,eigenlevels=16)
    def residual(x):
        en,_=_transmon(x[0],x[1],basis)
        return [en[1]-f01_ghz,2*en[1]-en[2]-anharmonicity_magnitude_ghz]
    ec=anharmonicity_magnitude_ghz
    guess=[(f01_ghz+ec)**2/(8*ec),ec]
    sol=least_squares(residual,guess,bounds=([1,.03],[60,.8]),xtol=1e-13,ftol=1e-13,gtol=1e-13)
    return {"ej_ghz":float(sol.x[0]),"ec_ghz":float(sol.x[1]),
            "ej_over_ec":float(sol.x[0]/sol.x[1]),"residual_ghz":float(max(abs(sol.fun)))}


def duffing_theory(levels=6):
    wq,wr,alpha,j,g=6.5,7.3,-.25,.058,.085
    a=np.diag(np.sqrt(np.arange(1,levels)),1)
    identity=np.eye(levels)
    n=a.T@a
    local=np.diag(wq*np.arange(levels)+alpha/2*np.arange(levels)*(np.arange(levels)-1))
    a1,a2=np.kron(a,identity),np.kron(identity,a)
    pair=np.kron(local,identity)+np.kron(identity,local)+j*(a1.T@a2+a2.T@a1)
    pe,ps=eigh(pair)
    hp=np.kron(pair,identity)+np.kron(np.eye(levels**2),wr*n)
    hp+=g*(np.kron(a1+a2,a.T)+np.kron(a1.T+a2.T,a))
    energy,states=eigh(hp,subset_by_index=(0,min(70,levels**3-1)))
    refs=np.column_stack([np.kron(ps[:,rail],identity[:,photon]) for photon in range(4) for rail in range(3)])
    overlap=abs(refs.T@states)**2
    r,c=linear_sum_assignment(-overlap)
    assigned=energy[c].reshape(4,3)
    n_results=[]
    for photon in range(3):
        shifts=assigned[photon+1,:]-assigned[photon,:]
        chi_m,chi_p=shifts[1]-shifts[0],shifts[2]-shifts[0]
        n_results.append({"photon":photon,"physical_chi_m_mhz":float(chi_m*1000),
                          "physical_chi_p_mhz":float(chi_p*1000),
                          "paper_chi_mhz":float((chi_m+chi_p)*250),
                          "paper_chi_dr_khz":float((chi_p-chi_m)*500000),
                          "paper_ratio":float(abs(2*(chi_p-chi_m)/(chi_p+chi_m))),
                          "notebook_ratio_p":float(abs((chi_p-chi_m)/chi_p))})
    coupling=abs(ps.T@(g*(a1+a2))@ps)
    ratios=[]
    for i in range(3):
        for k in range(len(pe)):
            if pe[k] <= pe[i] or coupling[i,k]<1e-10:
                continue
            ratios.append(coupling[i,k]/abs(pe[k]-pe[i]-wr))
    return {"model":"RWA Duffing, paper Eq E1-E2", "local_g_over_detuning":g/abs(wr-wq),
            "max_pair_transition_ratio":float(max(ratios)),
            "logical_splitting_mhz":float((assigned[0,2]-assigned[0,1])*1000),
            "min_assignment_overlap":float(min(overlap[r,c])),"photon_results":n_results}


def main():
    with threadpool_limits(limits=1):
        result={"sources":{
            "2023_theory":"https://journals.aps.org/prx/pdf/10.1103/PhysRevX.13.041022",
            "2024_experiment":"https://arxiv.org/html/2307.08737v3",
            "2026_experiment":"https://arxiv.org/html/2604.16292v1"},
            "theory_2023":duffing_theory(),
            "theory_2023_cosine_inference":infer_cosine(6.5,.25),
            "experiment_2024_cosine_inference":{"Q1":infer_cosine(5.1,.193),"Q2":infer_cosine(5.1,.204)},
            "experiment_2024_ancilla_matching_ratio":.096/((1.514+1.610)/2),
            "experiment_2024_ancilla_local_g_over_detuning":.0817/(5.1-3.74),
            "experiment_2026":{"operating_frequency_ghz":4.5,"resonator_ground_transition_ghz":6.7916,
                               "logical_splitting_mhz":94.150,"paper_chi_mhz":-4.25,"paper_chi_dr_khz":-.7,
                               "paper_matching_ratio":.0007/4.25,
                               "ec_ej_transverse_couplings":"not reported in the manuscript parameter table"}}
    out=Path(__file__).resolve().parent/"data"/"aws_papers"
    out.mkdir(parents=True,exist_ok=True)
    (out/"checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
