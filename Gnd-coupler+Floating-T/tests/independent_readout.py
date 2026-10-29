"""Independent readout reference calculations used only by regression tests.

No saved case defaults, production optimizer, CLI or large default audit basis.
This intentionally reconstructs energy loops independently of ReadoutModel's
chi helpers, so a labeling/sign bug is not checked against the same code.
"""
from dataclasses import replace
import numpy as np
from scipy.constants import e,h,hbar
from scipy.linalg import eigh
from core.transmon import Basis,transmon_spectrum
from readout_workflow import calculation as s

def policy(case):
    # Local process only: the shared workflow defaults/source are unchanged.
    s.CONDITIONS = replace(s.Conditions(), ej_over_ec_min=45.,
                           logical_frequency_ghz=case['window'])
    s.BASELINE = dict(s.BASELINE, ej_ec=45., logical_frequency=case['window'])


def device_for_ec(ec, c12, c1r, c2r, lc_frequency):
    """Solve positive self caps for exact loaded ECs using Schur complement."""
    omega = 2*np.pi*lc_frequency*1e9
    cr = 1e15/(omega*50.)
    lr = 50./omega*1e9
    r = cr+c1r+c2r
    d1, d2 = np.asarray(ec)/1000/s.CHARGE_CONSTANT
    b = c12+c1r*c2r/r
    determinant = (1+np.sqrt(1+4*d1*d2*b*b))/(2*d1*d2)
    c1 = d2*determinant-c12-c1r+c1r*c1r/r
    c2 = d1*determinant-c12-c2r+c2r*c2r/r
    return s.Device(c1, c2, c12, c1r, c2r, cr, lr)


def evaluate(case, x, basis, device=None):
    # coordinates: local mean GHz, local detuning MHz, effective links fF,
    # and unloaded LC frequency GHz. At fixed device only first two vary.
    if device is None:
        device = device_for_ec(case['ec'], *x[2:])
    m = s.ReadoutModel(device, basis)
    ej = [s.inverse_frequency(x[0]+sign*x[1]/2000, m.ec_ghz[i], basis)
          for i, sign in enumerate((1, -1))]
    return s.annotate(m, *ej)


def equal_coordinates(z):
    """One shared physical readout capacitor parameter, never two free links."""
    return np.array([z[0],z[1],z[2],z[3],z[3],z[4]])


def float_physical(device):
    return dict(self_capacitances_ff=[device.C1_shunt_fF, device.C2_shunt_fF],
                pair_link_each_ff=2*device.C12_fF,
                transmon_resonator_link_each_ff=[2*device.C1r_fF, 2*device.C2r_fF],
                resonator_capacitance_ff=device.Cr_fF, resonator_inductance_nh=device.Lr_nH,
                link_endpoints='pair: +/+ and -/-; readout: +/signal and -/return',
                interpretation='Each reported link value is one of TWO equal physical capacitors.')


def audit(point,basis):
    d=point['device'];b=basis
    c1,c2,c12,u,v,cr=[d[k]*1e-15 for k in ('C1_shunt_fF','C2_shunt_fF','C12_fF','C1r_fF','C2r_fF','Cr_fF')]
    ci=np.linalg.inv([[c1+c12+u,-c12,-u],[-c12,c2+c12+v,-v],[-u,-v,cr+u+v]])
    ec=e*e*np.diag(ci)/(2*h*1e9)
    levels=[transmon_spectrum(point[f'ej{i+1}_ghz'],ec[i],b) for i in range(2)]
    ident=np.eye(b.transmon_levels);ir=np.eye(b.resonator_levels)
    n1=np.kron(levels[0][1],ident);n2=np.kron(ident,levels[1][1])
    hp=np.diag((levels[0][0][:,None]+levels[1][0][None,:]).ravel())+4*e*e*ci[0,1]/(h*1e9)*(n1@n2)
    _,pv=eigh(hp)
    omega=1/np.sqrt(d['Lr_nH']*1e-9/ci[2,2]);fr=omega/(2*np.pi*1e9)
    qz=np.sqrt(hbar*omega/(2*ci[2,2]))
    a=np.diag(np.sqrt(np.arange(1,b.resonator_levels)),1)
    photon=np.diag(np.arange(b.resonator_levels))
    ham=np.kron(hp,ir)+np.kron(np.eye(hp.shape[0]),fr*photon)
    ham+=2*e*qz/(h*1e9)*np.kron(ci[0,2]*n1+ci[1,2]*n2,a+a.T)
    energies,states=eigh(ham,subset_by_index=[0,b.eigenlevels-1]);energies-=energies[0]
    labels=['g','m','p','r','mr','pr'];refs=[np.kron(pv[:,i],ir[:,j]) for i,j in [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]]
    indices=[int(np.argmax(np.abs(ref@states)**2)) for ref in refs]
    if len(set(indices))!=6:raise ValueError('Independent labels are ambiguous/duplicate')
    data={label:dict(energy_ghz=float(energies[index]),index=index,
                    reference_overlap=float(abs(ref@states[:,index])**2),
                    mean_photons=float(states[:,index]@np.kron(np.eye(hp.shape[0]),photon)@states[:,index]))
          for label,index,ref in zip(labels,indices,refs)}
    en={k:v['energy_ghz'] for k,v in data.items()}
    chi=np.array([en['mr']-en['m']-en['r']+en['g'],en['pr']-en['p']-en['r']+en['g']])
    return dict(states=data,chi_minus_mhz=1000*chi[0],chi_plus_mhz=1000*chi[1],
                delta_chi_khz=1e6*(chi[1]-chi[0]),ratio=float(abs((chi[1]-chi[0])/np.mean(chi))),
                agreement_ghz=(chi-np.array([point['chi_m_ghz'],point['chi_p_ghz']])).tolist(),
                equal_readout_capacitors=bool(u==v))

