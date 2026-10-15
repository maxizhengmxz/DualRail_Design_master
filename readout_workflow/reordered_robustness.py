"""Reordered rescue ladder; operation frequency retains the dressed mean."""
from dataclasses import asdict
import time
import numpy as np
from scipy.optimize import least_squares, minimize
from threadpoolctl import threadpool_limits
from model import Basis, Device, ReadoutModel, _transmon
from refined_pair_search import inverse_frequency
from robustness_rescue import CAPS, PARAMETERS, LARGE, perturb, annotate

TIERS=(
    dict(name='baseline',ec_min=150.,H=.95,gap=[.09,.11],ej_ec=50.,fixed=True),
    dict(name='EC120',ec_min=120.,H=.95,gap=[.09,.11],ej_ec=50.,fixed=True),
    dict(name='free_operation_frequency',ec_min=120.,H=.95,gap=[.09,.11],ej_ec=50.,fixed=False),
    dict(name='gap80_120_H95',ec_min=120.,H=.95,gap=[.08,.12],ej_ec=50.,fixed=False),
    dict(name='H90',ec_min=120.,H=.90,gap=[.08,.12],ej_ec=50.,fixed=False),
    dict(name='EJ_EC40',ec_min=120.,H=.90,gap=[.08,.12],ej_ec=40.,fixed=False),
)

def reasons(p,tier,target):
    tests={
        'chi_matching':p['ratio']<=.01+1e-8,
        'dispersive':p['lambda_max']<=.1+1e-9,
        'hybridization':p['hybridization']>=tier['H']-1e-8,
        'logical_gap':tier['gap'][0]-1e-8<=p['splitting_ghz']<=tier['gap'][1]+1e-8,
        'EC_range':min(p['loaded_ec1_mhz'],p['loaded_ec2_mhz'])>=tier['ec_min']-1e-7 and max(p['loaded_ec1_mhz'],p['loaded_ec2_mhz'])<400,
        'EJ_limit':max(p['ej1_ghz'],p['ej2_ghz'])<25,
        'EJ_over_EC':min(p['ej1_over_ec1'],p['ej2_over_ec2'])>tier['ej_ec'],
        'assignment_overlap':p['min_assignment_overlap']>=.95-1e-8,
        'operation_frequency':4-1e-8<=p['operating_frequency_ghz']<=8+1e-8,
    }
    if tier['fixed']: tests['fixed_operation_frequency']=abs(p['operating_frequency_ghz']-target)<=1e-7
    return [key for key,ok in tests.items() if not ok]


def slacks(p,tier,include_H=False,margin=0.):
    # EC is immutable for a fabricated circuit and is screened separately.
    values=[(.01-margin*1e-6-p['ratio'])/.01,
        (.1-margin*1e-7-p['lambda_max'])/.1,
        (p['min_assignment_overlap']-.95-margin*1e-7)/.05,
        (p['splitting_ghz']-tier['gap'][0]-margin*1e-7)/.01,
        (tier['gap'][1]-margin*1e-7-p['splitting_ghz'])/.01,
        (p['operating_frequency_ghz']-4-margin*1e-7),
        (8-margin*1e-7-p['operating_frequency_ghz']),
        (25-margin*1e-6-p['ej1_ghz'])/10,
        (25-margin*1e-6-p['ej2_ghz'])/10,
        (p['ej1_over_ec1']-tier['ej_ec']-margin*1e-5)/50,
        (p['ej2_over_ec2']-tier['ej_ec']-margin*1e-5)/50]
    if include_H: values.append((p['hybridization']-tier['H'])/.05)
    return np.nan_to_num(values,nan=-1e4,posinf=1e4,neginf=-1e4)


def x_from(p):
    return np.array([(p['bare_f1_ghz']+p['bare_f2_ghz'])/2,
                     (p['bare_f1_ghz']-p['bare_f2_ghz'])/.01])


class Calibration:
    def __init__(self,device,reference,basis=Basis()):
        self.m=ReadoutModel(device,basis);self.reference=reference
        self.target=reference['operating_frequency_ghz'];self.cache={};self.evaluations=0
    def point(self,x):
        key=tuple(map(float,x))
        if key not in self.cache:
            f,d=key
            ej=[inverse_frequency(f+sign*.005*d,ec,self.m.basis) for sign,ec in zip((1,-1),self.m.ec_ghz[:2])]
            self.cache[key]=annotate(self.m,*ej);self.evaluations+=1
        return self.cache[key]
    def solve(self,tier,warm=None,deep=False):
        start=time.monotonic();original=x_from(self.reference)
        seeds=[x_from(warm) if warm else original,original,original+[0,.6],original-[0,.6]]
        if not tier['fixed']:
            seeds += [original+[-.25,0],original+[.25,0],np.array([4.08,original[1]])]
            lo=max(4.02,*(_transmon((tier['ej_ec']+.01)*ec,ec,self.m.basis)[0][1] for ec in self.m.ec_ghz[:2]))
            hi=min(7.98,*(_transmon(24.999,ec,self.m.basis)[0][1] for ec in self.m.ec_ghz[:2]))
            if lo<hi:
                seeds += [np.array([lo+fraction*(hi-lo),original[1]]) for fraction in (.5,.1,.9)]
        seen=[];attempts=[];best=None
        def merit(p):
            rest=[r for r in reasons(p,tier,self.target) if r!='hybridization']
            if not rest: return (1.,p['hybridization'],0.)
            violation=np.minimum(slacks(p,tier),0.)
            eq=(p['operating_frequency_ghz']-self.target)/.02 if tier['fixed'] else 0.
            return (0.,-float(np.linalg.norm(np.r_[violation,eq])),p['hybridization'])
        def keep(p):
            nonlocal best
            if best is None or merit(p)>merit(best): best=p
            return p
        def point(x): return keep(self.point(x))
        for index,seed in enumerate(seeds):
            seed=np.clip(seed,[3.7,-20],[8.3,20])
            if any(np.linalg.norm(seed-s)<1e-6 for s in seen): continue
            seen.append(seed)
            if index and time.monotonic()-start>(100 if deep else 35): break
            def residual(x):
                p=point(x);v=np.minimum(slacks(p,tier,margin=1),0.)
                return np.r_[v,(p['operating_frequency_ghz']-self.target)/.02] if tier['fixed'] else v
            fit=least_squares(residual,seed,bounds=([3.7,-20],[8.3,20]),
                diff_step=2e-5,xtol=1e-9,ftol=1e-9,gtol=1e-8,max_nfev=80 if deep else 40)
            p=point(fit.x)
            constraints=[dict(type='ineq',fun=lambda x:slacks(point(x),tier,margin=1))]
            if tier['fixed']:
                constraints.append(dict(type='eq',fun=lambda x:(point(x)['operating_frequency_ghz']-self.target)/.02))
            opt=minimize(lambda x:-point(x)['hybridization'],fit.x,method='SLSQP',
                bounds=[(3.7,8.3),(-20,20)],constraints=constraints,
                options={'ftol':1e-9,'maxiter':85 if deep else 45,'eps':1e-5})
            point(opt.x)
            attempts.append(dict(seed=seed.tolist(),fit_nfev=int(fit.nfev),optimizer_success=bool(opt.success),
                optimizer_message=str(opt.message),best_failures=reasons(best,tier,self.target)))
            # One polished solution plus an independent detuning start for
            # fixed-operation-frequency optimization. Retry more seeds for unsuccessful cases.
            if best is not None and not reasons(best,tier,self.target) and len(attempts)>=2: break
        return dict(point=best,failures=reasons(best,tier,self.target),attempts=attempts,
                    evaluations=self.evaluations,seconds=time.monotonic()-start)



def evaluate(task):
    label,reference,scenario,previous=task
    started=time.monotonic();device=perturb(reference,scenario['errors'])
    with threadpool_limits(limits=1):
        small=Calibration(device,reference);large=Calibration(device,reference,LARGE)
        ec=small.m.ec_ghz[:2]*1000;stages=[];first_pass=None;warm=None
        pool=[s['point'] for s in previous.get('stages',[]) if s.get('point')]
        assert all(p['device']==asdict(device) for p in pool)
        structural=bool(min(ec)<120-1e-7 or max(ec)>=400)
        if structural:
            return dict(design=label,**scenario,device=asdict(device),stages=[],first_pass=None,
                structural_failure='EC_range',loaded_ec_mhz=ec.tolist(),seconds=time.monotonic()-started)
        for index,tier in enumerate(TIERS):
            if min(ec)<tier['ec_min']-1e-7:
                p=small.point(x_from(reference))
                result=dict(point=p,failures=['EC_range'],attempts=[],seconds=0.,source='immutable_EC_screen')
            elif index==1 and min(ec)>=150-1e-7:
                result=dict(stages[-1]);result['source']='EC relaxation inactive: identical feasible set'
            elif index==0 and previous.get('stages'):
                result=dict(previous['stages'][0]);p=result['point']
                result['failures']=reasons(p,tier,small.target)
                result['source']='unchanged previous baseline'
                if not result['failures']:
                    q=annotate(large.m,p['ej1_ghz'],p['ej2_ghz'])
                    result['point']=q;result['failures']=reasons(q,tier,large.target)
            else:
                valid=[p for p in pool if not reasons(p,tier,small.target)]
                result=None
                for p in sorted(valid,key=lambda p:p['hybridization'],reverse=True):
                    q=annotate(large.m,p['ej1_ghz'],p['ej2_ghz'])
                    if not reasons(q,tier,large.target):
                        result=dict(point=q,failures=[],attempts=[],seconds=0.,source='revalidated_previous_witness')
                        break
                if result is None:
                    if pool:
                        warm=max(pool,key=lambda p:(-np.linalg.norm(np.minimum(slacks(p,tier),0.)),p['hybridization']))
                    result=small.solve(tier,warm)
                    warm=result['point']
                    if not result['failures']:
                        q=annotate(large.m,warm['ej1_ghz'],warm['ej2_ghz'])
                        result['screening_point']=warm;result['point']=q;result['failures']=reasons(q,tier,large.target)
                        if result['failures']:
                            repair=large.solve(tier,q,deep=True);result['repair']=repair
                            result['point']=repair['point'];result['failures']=repair['failures']
                    elif max(abs(np.minimum(slacks(warm,tier,include_H=True),0.)))<.03:
                        retry=large.solve(tier,warm,deep=True);result['large_basis_retry']=retry
                        if not retry['failures']: result['point']=retry['point'];result['failures']=[]
            result['tier']=tier['name'];result['passes']=not result['failures'];stages.append(result)
            pool.append(result['point'])
            if result['passes']:
                first_pass=index;break
        return dict(design=label,**scenario,device=asdict(device),stages=stages,first_pass=first_pass,
                    structural_failure=None,loaded_ec_mhz=ec.tolist(),seconds=time.monotonic()-started)
