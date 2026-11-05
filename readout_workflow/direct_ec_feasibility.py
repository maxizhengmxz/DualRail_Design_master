"""Direct loaded-EC feasibility search at fixed EC1 and EC2.

Capacitances are solved analytically from the Schur complement of the full
three-node C matrix at every trial. This preserves the requested loaded ECs
as C12 and equal readout coupling capacitors change. No readout optimization,
failed-target retries, contour refinement, or failure diagnosis is performed.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,subprocess,sys,time
from pathlib import Path
from dataclasses import replace,asdict
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import ConvexHull,Delaunay
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
# Also support an explicit direct-file invocation from an IDE.
if not __package__:
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from readout_workflow import calculation as s
from readout_workflow.validation import audit_point
from readout_workflow.storage import atomic_json,utc
from Readout_ParamSpace_Robustness import process_alive
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'readout_workflow/data/direct_ec_step5'
SOURCE=ROOT/'readout_workflow/data/feasibility_150mhz_step1_diag2/results.json'

def target_grid(xy,domain,step_mhz=5,hull_scale=1.0,ec_bounds_mhz=(150.0,400.0)):
    """Return an EC lattice inside the observed rectangle or a scaled hull."""
    xy=np.asarray(xy,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.all(np.isfinite(xy)):
        raise ValueError('xy must contain at least three finite (EC1, EC2) points')
    if domain not in ('hull','rectangle'):raise ValueError("domain must be 'hull' or 'rectangle'")
    if not np.isfinite(step_mhz) or step_mhz<=0:raise ValueError('step_mhz must be positive')
    if not np.isfinite(hull_scale) or hull_scale<1:raise ValueError('hull_scale must be at least 1')
    if len(ec_bounds_mhz)!=2 or not 0<ec_bounds_mhz[0]<ec_bounds_mhz[1]:raise ValueError('invalid EC bounds')
    hull=ConvexHull(xy);vertices=xy[hull.vertices]
    if domain=='hull':
        center=vertices.mean(axis=0);selection_xy=center+hull_scale*(vertices-center)
    else:selection_xy=xy
    xs=np.arange(np.ceil(selection_xy[:,0].min()/step_mhz)*step_mhz,selection_xy[:,0].max()+1e-8,step_mhz)
    ys=np.arange(np.ceil(selection_xy[:,1].min()/step_mhz)*step_mhz,selection_xy[:,1].max()+1e-8,step_mhz)
    rectangle=[(int(x),int(y)) for x in xs for y in ys]
    rectangle=[point for point in rectangle if all(ec_bounds_mhz[0]<=value<ec_bounds_mhz[1] for value in point)]
    if domain=='rectangle':return rectangle
    tri=Delaunay(selection_xy)
    return [point for point in rectangle if tri.find_simplex([point])[0]>=0]

def mirrored_target_grid(xy,step_mhz=5,hull_scale=1.5,ec_bounds_mhz=(150.0,400.0)):
    """Union of separately scaled original and label-swapped readout hulls."""
    original=set(target_grid(xy,'hull',step_mhz,hull_scale,ec_bounds_mhz))
    mirrored=set(target_grid(np.asarray(xy,float)[:,::-1],'hull',step_mhz,hull_scale,ec_bounds_mhz))
    return sorted(original|mirrored),original,mirrored

def mirrored_warm_point(point):
    """Return an exact label-swapped readout warm start without forging observables."""
    x=np.asarray(s.x_from_point(point),float)
    x[1]*=-1
    return {'x':x.tolist(),'readout_symmetry_source':'swapped_1_2'}

def logical_range_ghz(bounds_mhz):
    """Validate a CLI logical-frequency interval and convert MHz to GHz."""
    if len(bounds_mhz)!=2 or not all(np.isfinite(bounds_mhz)) or not 0<bounds_mhz[0]<bounds_mhz[1]:
        raise ValueError('logical-frequency-mhz requires two positive increasing values')
    return tuple(float(value)/1000 for value in bounds_mhz)

def initialize(logical_frequency_ghz=(.14,.16)):
    s.CONDITIONS=replace(s.CONDITIONS,logical_frequency_ghz=tuple(logical_frequency_ghz))
    s.BASELINE=dict(s.BASELINE,logical_frequency=tuple(logical_frequency_ghz))

def capacitances(ec1,ec2,c12,cg):
    """Physical C1,C2 (fF) with EXACT prescribed inverse-matrix diagonals.

    Eliminate resonator: A=[[X,-b],[-b,Y]], b=C12+Cg^2/R,
    R=Cr+2Cg. If diag(A^-1)=(d1,d2), determinant D satisfies
    d1*d2*D^2-D-b^2=0. X=d2*D, Y=d1*D; subtract attached C.
    """
    cr=s.Device().Cr_fF;r=cr+2*cg
    d1=(ec1/1000)/s.CHARGE_CONSTANT;d2=(ec2/1000)/s.CHARGE_CONSTANT
    b=c12+cg*cg/r
    determinant=(1+np.sqrt(1+4*d1*d2*b*b))/(2*d1*d2)
    correction=c12+cg-cg*cg/r
    return d2*determinant-correction,d1*determinant-correction

def point(ec1,ec2,x,basis):
    c1,c2=capacitances(ec1,ec2,x[2],x[3])
    return s.point_from_x(c1,c2,x,basis)

def solve(task):
    initialize(task.get('logical_frequency_ghz',(.14,.16)));ec1,ec2=task['ec'];started=time.monotonic();attempts=[];chosen=None;closest=None
    lower,upper=map(np.array,s.nominal_search_bounds());cache={}
    def evaluate(x):
        k=tuple(map(float,x))
        if k not in cache:cache[k]=point(ec1,ec2,x,s.SCREENING_BASIS)
        return cache[k]
    def residual(x):
        p=evaluate(x);d=p['device']
        return np.r_[s.violations(p,margin=.3),max(0,36-d['C1_shunt_fF'])/10,max(0,d['C1_shunt_fF']-128)/10,max(0,36-d['C2_shunt_fF'])/10,max(0,d['C2_shunt_fF']-128)/10]
    seeds=[]
    for p in task['warm']:
        x=np.clip(p['x'] if 'x' in p else s.x_from_point(p),lower+1e-6,upper-1e-6)
        if not any(np.linalg.norm(x-y)<1e-5 for y in seeds):seeds.append(x)
    with threadpool_limits(limits=1):
        for seed in seeds[:3]:
            if attempts and time.monotonic()-started>25:break
            fit=least_squares(residual,seed,bounds=(lower,upper),x_scale=[1,1,2,5],diff_step=2e-5,ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=35)
            p=evaluate(fit.x);attempts.append(dict(seed=seed.tolist(),x=fit.x.tolist(),nfev=int(fit.nfev),cost=float(fit.cost),failures=s.failures(p),endpoint=p))
            if closest is None or np.linalg.norm(s.violations(p))<np.linalg.norm(s.violations(closest)):closest=p
            candidates=[q for q in cache.values() if not s.failures(q)]
            if candidates:
                # First acceptable numerical witness; do not rank by readout.
                q=audit_point(candidates[0]);d=q['device']
                errs=[abs(q['loaded_ec1_mhz']-ec1),abs(q['loaded_ec2_mhz']-ec2)]
                failed=s.failures(q,require_sign_check=True)
                if any(v>1e-6 for v in errs):failed.append('loaded_EC_target')
                if not all(36<=d[k]<=128 for k in ('C1_shunt_fF','C2_shunt_fF')):failed.append('capacitance_search_range')
                attempts[-1]['audit_failures']=failed
                if not failed:chosen=q;break
    return dict(ec1_target_mhz=ec1,ec2_target_mhz=ec2,status='validated_pass' if chosen else 'no_validated_solution_found',point=chosen,closest_screening_point=closest,attempts=attempts,seconds=time.monotonic()-started,completed_utc=utc(),readout_optimized=False)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--background',action='store_true');parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--source',type=Path,default=SOURCE,help='Fine C1-C2 results.json for the same logical-frequency interval.');parser.add_argument('--domain',choices=('hull','rectangle'),default='hull',help='Select targets inside the scaled witness hull (default) or the unscaled full EC min/max rectangle.');parser.add_argument('--orientation-domain',choices=('full','reduced'),default='full',help='Use the union of original and label-swapped readout hulls (default), or only the calculated triangular orientation.');parser.add_argument('--hull-scale',type=float,default=1.5,help='Expansion about the hull-vertex center (fixed campaign default: 1.5).');parser.add_argument('--ec-step-mhz',type=float,default=5,help='Loaded-EC lattice spacing in MHz (default: 5).');parser.add_argument('--workers',type=int,default=4);parser.add_argument('--logical-frequency-mhz',type=float,nargs=2,metavar=('LOW','HIGH'),default=(140,160),help='Accepted logical-frequency interval in MHz (default: 140 160).');a=parser.parse_args();OUT=a.output.resolve();source=a.source.resolve()
    if not a.run:print('Use --run to calculate; no automatic calculation on import.');return
    if a.workers<1:raise ValueError('workers must be positive')
    if a.background:
        OUT.mkdir(parents=True,exist_ok=True);launch=OUT/'launch.json'
        if launch.exists() and process_alive(json.loads(launch.read_text())['pid']):raise RuntimeError('Already running')
        command=[sys.executable,'-m','readout_workflow.direct_ec_feasibility','--run','--output',str(OUT),'--source',str(source),'--domain',a.domain,'--orientation-domain',a.orientation_domain,'--hull-scale',str(a.hull_scale),'--ec-step-mhz',str(a.ec_step_mhz),'--workers',str(a.workers),'--logical-frequency-mhz',*map(str,a.logical_frequency_mhz)]
        with (OUT/'run.log').open('ab') as log:
            kwargs=dict(cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            if os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS|subprocess.CREATE_BREAKAWAY_FROM_JOB
            else:kwargs['start_new_session']=True
            process=subprocess.Popen(command,**kwargs)
        record=dict(pid=process.pid,started_utc=utc(),background=True,command=command);atomic_json(launch,record);print(json.dumps(record));return
    logical_frequency_ghz=logical_range_ghz(a.logical_frequency_mhz);initialize(logical_frequency_ghz);OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads(source.read_text());good=[r['point'] for r in old if r['status'] in ('validated_pass','recovered_viable')];xy=np.array([[p['loaded_ec1_mhz'],p['loaded_ec2_mhz']] for p in good])
    if a.orientation_domain=='full' and a.domain=='hull':targets,original_targets,mirrored_targets=mirrored_target_grid(xy,step_mhz=a.ec_step_mhz,hull_scale=a.hull_scale,ec_bounds_mhz=s.CONDITIONS.loaded_ec_mhz)
    else:
        targets=target_grid(xy,a.domain,step_mhz=a.ec_step_mhz,hull_scale=a.hull_scale,ec_bounds_mhz=s.CONDITIONS.loaded_ec_mhz);original_targets=set(targets);mirrored_targets=set()
    source_xy=[tuple(v) for v in xy];warm_points=list(good)
    if a.orientation_domain=='full':source_xy.extend(tuple(v) for v in xy[:,::-1]);warm_points.extend(mirrored_warm_point(p) for p in good)
    source_xy=np.asarray(source_xy,float)
    selection=(f'union of separately constructed original and label-swapped readout hulls, each scaled by {a.hull_scale:g} about its vertex center; membership is not assumed feasible' if a.orientation_domain=='full' and a.domain=='hull' else f'convex hull of prior validated witnesses, scaled by {a.hull_scale:g} about its vertex center; membership is not assumed feasible' if a.domain=='hull' else 'full 5 MHz Cartesian grid spanning the coordinate-wise loaded-EC minima and maxima of prior validated witnesses; membership is not assumed feasible')
    policy=dict(conditions=asdict(s.CONDITIONS),logical_frequency_mhz=list(map(float,a.logical_frequency_mhz)),ec_step_mhz=float(a.ec_step_mhz),target_domain=a.domain,orientation_domain=a.orientation_domain,readout_label_swap_symmetry=a.orientation_domain=='full',hull_scale=float(a.hull_scale) if a.domain=='hull' else None,targets=targets,original_orientation_targets=len(original_targets),mirrored_orientation_targets=len(mirrored_targets),selection=selection,source_results=str(source),source_witnesses=len(good),effective_warm_witnesses=len(warm_points),source_sha256=sha256(source.read_bytes()).hexdigest(),capacitance_range_ff=[36,128],budget=dict(starts=3,max_nfev=35,screening_soft_seconds=25),optimize_readout=False,retry_failed=False)
    from readout_workflow.validation import POLICY
    files=[Path(__file__),*[ROOT/'readout_workflow'/name for name in ('calculation.py','model.py','validation.py')],*sorted((ROOT/'core').glob('*.py'))]
    policy['validation_policy']=POLICY
    policy['source_hashes']={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}
    settings_path=OUT/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=json.loads(json.dumps(policy)):
        raise RuntimeError('Saved settings/source/validation policy differ. Use --output with a new directory; existing results are preserved.')
    atomic_json(settings_path,policy)
    # Verify exact capacitor reconstruction on every previous physical witness.
    worst=0
    for p in good:
        d=p['device'];c1,c2=capacitances(p['loaded_ec1_mhz'],p['loaded_ec2_mhz'],d['C12_fF'],d['C1r_fF']);worst=max(worst,abs(c1-d['C1_shunt_fF']),abs(c2-d['C2_shunt_fF']))
    assert worst<1e-8
    atomic_json(OUT/'capacitance_reconstruction_check.json',dict(witnesses_checked=len(good),max_capacitance_error_ff=worst))
    rows=[]
    for e in targets:
        f=OUT/f'ec_{e[0]}_{e[1]}.json'
        if f.exists():rows.append(json.loads(f.read_text()))
    done={(r['ec1_target_mhz'],r['ec2_target_mhz']) for r in rows}
    def progress(phase='running'):atomic_json(OUT/'progress.json',dict(phase=phase,target_domain=a.domain,total=len(targets),completed=len(rows),passed=sum(r['status']=='validated_pass' for r in rows),updated_utc=utc(),pid=os.getpid()))
    progress()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={}
        for e in targets:
            if e in done:continue
            near=np.argsort(np.sum((source_xy-np.array(e))**2,axis=1))[:3]
            membership=dict(in_original_scaled_hull=e in original_targets,in_mirrored_scaled_hull=e in mirrored_targets)
            futures[pool.submit(solve,dict(ec=e,warm=[warm_points[i] for i in near],logical_frequency_ghz=logical_frequency_ghz))]=(e,membership)
        for future in as_completed(futures):
            e,membership=futures[future];r=future.result()|membership;rows.append(r);atomic_json(OUT/f"ec_{r['ec1_target_mhz']}_{r['ec2_target_mhz']}.json",r);progress();print(json.dumps(dict(ec=e,status=r['status'],seconds=r['seconds'],completed=len(rows))),flush=True)
    rows.sort(key=lambda r:(r['ec1_target_mhz'],r['ec2_target_mhz']));atomic_json(OUT/'results.json',rows);progress('complete');print('COMPLETE',flush=True)
if __name__=='__main__':main()
