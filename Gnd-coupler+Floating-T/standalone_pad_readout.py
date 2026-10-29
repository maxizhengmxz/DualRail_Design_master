"""Independent explicit-pad pair/readout synthesis; no coupler or other pair."""
import sys,json,argparse
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from scipy.optimize import least_squares,minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.transmon import Basis
from floating_coupler.model import Model,SCALE
from floating_coupler.design import ej_for_frequency
from floating_coupler.pairs import pair_spectrum
from floating_coupler.pad_electrostatics import fit_pad_mutual_capacitances
from readout_workflow import calculation as study
from readout_workflow.validation import audit_point

FAST=Basis(20,6,5,30)
MEDIUM=Basis(25,8,7,45)


def policy(case):
    study.CONDITIONS=replace(study.Conditions(),ej_over_ec_min=45.,logical_frequency_ghz=tuple(case['dressed_split_window_ghz']))
    study.BASELINE=dict(study.BASELINE,ej_ec=45.,logical_frequency=tuple(case['dressed_split_window_ghz']))


def build_device(case,x):
    f1,f2,cpair,cread,flc,eta1,eta2=x
    names=case['nodes'];pads=case['pad_pairs'];ec=dict(zip(names,case['loaded_ec_ghz']))
    grounds={p:2*eta*SCALE/ec[n] for n,eta in zip(names,[eta1,eta2]) for p in pads[n]}
    r=case['resonator_node'];cr=1/(2*np.pi*flc*1e9*50)*1e15;lr=50/(2*np.pi*flc*1e9)*1e9
    edges=[(pads[names[0]][1],pads[names[1]][0],cpair),(pads[names[0]][1],r,cread),(pads[names[1]][0],r,cread)]
    links=[{'from':a,'to':b,'capacitance_ff':float(c)} for a,b,c in edges]
    circuit=fit_pad_mutual_capacitances(pads,ec,grounds,{r:cr},links)
    # Flip the first transmon charge coordinate: the exact modal C then has
    # the standard readout-workflow grounded-node sign convention.
    gauge=np.diag([-1.,1.,1.]);cm=gauge@circuit.effective_modal_capacitance_ff@gauge
    c12,c1r,c2r=-cm[0,1],-cm[0,2],-cm[1,2]
    device=study.Device(cm[0,0]-c12-c1r,cm[1,1]-c12-c2r,c12,c1r,c2r,cm[2,2]-c1r-c2r,lr)
    physical={'pad_pairs':pads,'ground_capacitances_ff':grounds,'pad_mutual_capacitances_ff':circuit.pad_mutual_capacitances_ff,'physical_links':links,'resonator_self_capacitance_ff':cr,'resonator_inductance_nh':lr,'nominal_unloaded_impedance_ohm':50,'effective_device_for_exact_equivalent_H':asdict(device),'note':'Effective modal capacitors are not fabricated capacitor values. Each readout physical link appears exactly once.'}
    return device,physical


def evaluate(case,x,basis=FAST,device=None):
    if device is None:device,physical=build_device(case,x)
    else:physical=None
    model=study.ReadoutModel(device,basis);ej=[ej_for_frequency(float(f),float(e),basis.ncut) for f,e in zip(x[:2],model.ec_ghz)]
    p=study.annotate(model,*ej)
    pairmodel=Model(tuple(case['nodes']),model.ec_ghz[:2],np.array(ej),np.array([[0,model.J_ghz[0,1]],[model.J_ghz[0,1],0]]),basis,{})
    pair=pair_spectrum(pairmodel,(0,1),basis.transmon_levels,basis.ncut);p['isolated_pair_splitting_mhz']=pair['splitting_mhz'];p['logical_mean_readout_ghz']=p['dressed_r_ghz']+(p['chi_m_ghz']+p['chi_p_ghz'])/2
    return p,physical


def design(case,output):
    policy(case);guide=case['bare_frequency_guide_ghz'];fR=case['readout_target_ghz'];target=case['isolated_split_mhz'];ec=case['loaded_ec_ghz']
    bounds=([guide-.025,guide-.025,.005,.1,fR-.05,.1,.1],[guide+.025,guide+.025,40.,25.,fR+.15,.85,.85])
    cache={}
    def ev(x,basis=FAST):
        key=(tuple(x),basis)
        if key not in cache:cache[key]=evaluate(case,x,basis)
        return cache[key]
    def residual(x,basis=FAST):
        try:
            p,_=ev(x,basis);mean=(p['chi_m_ghz']+p['chi_p_ghz'])/2
            signed=(p['chi_p_ghz']-p['chi_m_ghz'])/max(abs(mean),1e-9)
            return np.r_[(p['isolated_pair_splitting_mhz']-target)/10,(p['logical_mean_readout_ghz']-fR)*100,(p['operating_frequency_ghz']-guide)*50,signed*50,(abs(mean)*1000-1.03),max(0,.9505-p['hybridization'])*100,max(0,p['lambda_max']-.0995)*100,max(0,.9505-p['min_assignment_overlap'])*100,max(0,45.001-p['ej1_over_ec1']),max(0,45.001-p['ej2_over_ec2']),max(0,p['ej1_ghz']-24.999),max(0,p['ej2_ghz']-24.999)]
        except ValueError:return np.ones(12)*100
    starts=[]
    cguess=13 if target==150 else 6.5
    for detune in [-.005,.005,-.015,.015]:
        x0=[guide+detune/2,guide-detune/2,cguess,10 if guide<5 else 7,fR+.04,.5,.5]
        fit=least_squares(residual,x0,bounds=bounds,x_scale=[.02,.02,10,10,.05,.5,.5],diff_step=2e-5,max_nfev=110,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        p,physical=ev(fit.x);row={'x':fit.x.tolist(),'cost':float(fit.cost),'residual':residual(fit.x).tolist(),'nominal_failures':study.failures(p),'isolated_split_mhz':p['isolated_pair_splitting_mhz'],'dressed_split_mhz':p['splitting_ghz']*1000,'metrics':{k:p[k] for k in ['operating_frequency_ghz','hybridization','ratio','lambda_max','chi_m_ghz','chi_p_ghz']}};starts.append(row);print(case['name'],'FIT',json.dumps(row),flush=True)
        if max(abs(residual(fit.x)))<1e-4:break
    best=min(starts,key=lambda r:r['cost']);x=best['x']
    fit=least_squares(lambda x:residual(x,MEDIUM),x,bounds=bounds,x_scale=[.02,.02,10,10,.05,.5,.5],diff_step=2e-5,max_nfev=70,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    def equalities(x):
        p,_=ev(x,MEDIUM)
        return [(p['isolated_pair_splitting_mhz']-target)/100,(p['logical_mean_readout_ghz']-fR),(p['operating_frequency_ghz']-guide)]
    def inequalities(x):
        p,_=ev(x,MEDIUM);mean=abs((p['chi_p_ghz']+p['chi_m_ghz'])/2)*1000
        return np.array([.0099-p['ratio'],mean-1.005,p['hybridization']-.9501,.0999-p['lambda_max'],p['min_assignment_overlap']-.9501,p['ej1_over_ec1']-45.0001,p['ej2_over_ec2']-45.0001,24.9999-p['ej1_ghz'],24.9999-p['ej2_ghz']])
    precise=minimize(lambda x: x[3],fit.x,method='SLSQP',bounds=list(zip(*bounds)),constraints=[{'type':'eq','fun':equalities},{'type':'ineq','fun':inequalities}],options={'maxiter':150,'ftol':1e-11,'eps':1e-6})
    fit=precise
    print('PRECISE',case['name'],fit.success,fit.message,'eq',equalities(fit.x),'ineq',inequalities(fit.x),flush=True)
    p,physical=ev(fit.x,MEDIUM);audit=audit_point(p,force_full=True);pairpoint,_=evaluate(case,fit.x,Basis(25,12,10,60));audit['isolated_pair_splitting_mhz']=pairpoint['isolated_pair_splitting_mhz'];audit['logical_mean_readout_ghz']=pairpoint['logical_mean_readout_ghz']
    result={'stage':'Independent pair + resonator only, no coupler and no other pair','case':case,'coordinates':fit.x.tolist(),'physical':physical,'nominal':audit,'workflow_failures':study.failures(audit,require_sign_check=True),'isolated_split_error_mhz':audit['isolated_pair_splitting_mhz']-target,'nominal_readout_error_ghz':audit['logical_mean_readout_ghz']-fR,'attempts':starts,'residual':residual(fit.x,MEDIUM).tolist()}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2));print('FINAL',case['name'],json.dumps({k:v for k,v in result.items() if k in ['workflow_failures','isolated_split_error_mhz','nominal_readout_error_ghz','residual']}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--pair',choices=['A','B'],required=True);p.add_argument('--output',required=True);a=p.parse_args();request=json.loads(Path(a.config).read_text());design(request['pairs'][a.pair],a.output)
