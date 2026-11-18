"""Electrostatic reconstruction shared by optional matched-allocation studies.

No sweep or gate calculation runs on import. ECs and intended couplings are
matched in K; physical capacitors and all derived metadata are rebuilt together.
"""
from copy import deepcopy
import numpy as np
from scipy.constants import elementary_charge, h, hbar
from floating_coupler.model import SCALE
from floating_coupler.mixed_electrostatics import mixed_circuit, mixed_budget

INTENDED = (("A1", "A2"), ("A2", "C"), ("C", "B1"), ("A2", "B1"), ("B1", "B2"),
            ("A1", "r1"), ("A2", "r1"), ("B1", "r2"), ("B2", "r2"))


def self_fraction(physical, name):
    """Physical own-capacitance fraction; loaded EC is not its denominator."""
    a, b = physical["pad_pairs"][name]
    g1, g2 = (physical["ground_capacitances_ff"][pad] for pad in (a,b))
    series = g1*g2/(g1+g2)
    cs = physical["pad_mutual_capacitances_ff"][name]
    return cs/(cs+series)


def circuit(ph):
    return mixed_circuit(ph['pad_pairs'],ph['ground_capacitances_ff'],
        ph['pad_mutual_capacitances_ff'],ph['physical_links'],
        resonator_caps_ff=ph['resonator_self_capacitances_ff'])


def synchronize_physical(physical):
    """Rebuild derived metadata after component edits; never retain nominal K.

    Resonator couplings must use the changed full K, including residual paths.
    The budget counterfactual removes the complete first three coupler links.
    This metadata rebuild does not certify a new OFF bias or gate calibration.
    """
    ph=deepcopy(physical);value=circuit(ph);k=value['K']
    ph['K_per_ff']=k.tolist()
    ph['loaded_ec_ghz']=dict(zip(ph['pad_pairs'],value['loaded_ec_ghz'].tolist()))
    ph['resonator_parameters']={}
    for i,r in enumerate(ph['resonator_self_capacitances_ff'],start=5):
        cap=1e-15/k[i,i];omega=1/np.sqrt(ph['resonator_inductances_nh'][r]*1e-9*cap)
        charge=np.sqrt(hbar*cap*omega/2)
        ph['resonator_parameters'][r]=dict(fr_ghz=float(omega/(2*np.pi*1e9)),
            effective_capacitance_ff=float(cap*1e15),q_zpf_coulomb=float(charge),
            gamma_ghz_per_n=(2*elementary_charge*charge/(h*1e9)*k[:5,i]*1e15).tolist())
    ph['budget']=mixed_budget(ph['pad_pairs'],ph['ground_capacitances_ff'],ph['pad_mutual_capacitances_ff'],
        ph['physical_links'],ph['physical_links'][:3],resonator_caps_ff=ph['resonator_self_capacitances_ff'])
    ph['budget_definition']='Complete three-link branch removal; derived fields rebuilt from the listed physical components'
    return ph


def coefficients(ph,k,n01):
    names=list(ph['pad_pairs']);result={};resonator_charges={}
    for i,a in enumerate(names):
        result['EC_'+a+'_mhz']=SCALE*k[i,i]*1000
        for j in range(i+1,len(names)):
            b=names[j]
            result['J_'+a+'_'+b+'_mhz']=8*SCALE*k[i,j]*1000
            result['g01_'+a+'_'+b+'_mhz']=8*SCALE*k[i,j]*1000*n01[a]*n01[b]
    for j,r in enumerate(ph['resonator_self_capacitances_ff'],start=5):
        cap=1e-15/k[j,j]
        omega=1/np.sqrt(ph['resonator_inductances_nh'][r]*1e-9*cap)
        charge=np.sqrt(hbar*cap*omega/2)
        resonator_charges[r]=(j,charge)
        result['f_'+r+'_ghz']=omega/(2*np.pi*1e9)
        for i,a in enumerate(names):
            result['g01_'+a+'_'+r+'_mhz']=2*elementary_charge*charge/(h*1e9)*k[i,j]*1e15*n01[a]*1000
    for index,a in enumerate(resonator_charges):
        for b in list(resonator_charges)[index+1:]:
            i,qa=resonator_charges[a];j,qb=resonator_charges[b]
            result['g01_'+a+'_'+b+'_mhz']=qa*qb/(h*1e9)*k[i,j]*1e15*1000
    return result
