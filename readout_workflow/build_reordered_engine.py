"""Create a separate, auditable engine revision; preserve previous sources."""
from pathlib import Path

BASE=Path(__file__).resolve().parent
source=(BASE/'robustness_rescue.py').read_text()
shared=source[source.index('def reasons('):source.index('\ndef evaluate(')]
shared=shared.replace('>=150-1e-7',">=tier['ec_min']-1e-7")
shared=shared.replace("'fixed_carrier'","'fixed_operation_frequency'").replace("'carrier'","'operation_frequency'")
shared=shared.replace('fixed-carrier','fixed-operation-frequency')
header='''"""Reordered rescue ladder; operation frequency retains the dressed mean."""
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

'''
footer='''

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
'''
(BASE/'reordered_robustness.py').write_text(header+shared+footer,encoding='utf-8')

runner=(BASE/'run_selected_robustness.py').read_text()
runner=runner.replace('from robustness_rescue import','from reordered_robustness import')
runner=runner.replace("OUT=BASE/'data'/'selected_robustness_2026_10_15'","OLD=BASE/'data'/'selected_robustness_2026_10_15'\nOUT=BASE/'data'/'reordered_robustness_2026_10_15'")
start=runner.index('def initialize():');end=runner.index('\n\ndef main():',start)
runner=runner[:start]+'''def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('designs.json','sampling.json'):
        if not (OUT/name).exists(): (OUT/name).write_bytes((OLD/name).read_bytes())
    designs=json.loads((OUT/'designs.json').read_text())
    vectors=json.loads((OUT/'sampling.json').read_text())['scenarios']
    return designs,vectors
'''+runner[end:]
runner=runner.replace("designs,vectors=initialize()", "designs,vectors=initialize()\n    previous={(r['design'],r['id']):r for r in json.loads((OLD/'samples.json').read_text())}")
runner=runner.replace("('robustness_rescue.py','model.py','study_dispersive.py')", "('reordered_robustness.py','robustness_rescue.py','model.py','study_dispersive.py')")
runner=runner.replace("evaluate((label,p,selected[0]))", "evaluate((label,p,selected[0],previous[(label,selected[0]['id'])]))")
runner=runner.replace('range(5)','range(len(TIERS))')
runner=runner.replace("tasks=[(label,p,v) for", "tasks=[(label,p,v,previous[(label,v['id'])]) for")
runner=runner.replace('strict EC and EJ bounds retained at every tier.',
    'EC floor follows rescue tiers; EJ ceiling and chi/dispersive screens retained. Operation frequency preserves the original dressed mean. Same 900 perturbed circuits as previous study.')
(BASE/'run_reordered_robustness.py').write_text(runner,encoding='utf-8')

