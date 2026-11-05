"""Opt-in quantum validation of frozen, individually compensated parasitics.

Idle rail EJs are frozen. OFF tunes only coupler EJ using all four physical
references. ON uses the saved static gate duration/coupler bias and corrects
the moved B-rail frequency to phase pi. This is not a pulsed/noisy gate model.
Source-hashed historical runs must use a new output after source changes.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

WF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WF),str(WF.parent)]


def secular(logical):
    from floating_coupler.on_gate import secular_logical_shifts
    projection = logical['logical_exchange']
    return secular_logical_shifts(dict(reference_labels=projection['logical_reference_order'],
                                       effective_hamiltonian_ghz=projection['effective_hamiltonian_ghz'],
                                       logical_local_energy_frame=projection['local_energy_frame']))


def find_off(search, seed, slope, bounds):
    """Evaluate local secant proposals, then fall back to verified bracketing."""
    import numpy as np
    from scipy.optimize import brentq
    first = search.at(float(seed))
    points = [first]
    if abs(first['A2_B1_hz'])<1:
        return dict(found=True,point=first,method='verified seed')
    proposal = float(np.clip(seed-first['A2_B1_hz']/slope,*bounds))
    for _ in range(4):
        if any(abs(proposal-p['frequency_ghz'])<1e-12 for p in points):
            break
        current = search.at(proposal)
        points.append(current)
        if abs(current['A2_B1_hz'])<1:
            return dict(found=True,point=current,method='evaluated secant; below 1 Hz')
        left,right = sorted(points[-2:],key=lambda p:p['frequency_ghz'])
        if left['A2_B1_hz']*right['A2_B1_hz']<0:
            root = brentq(lambda f:search.at(float(f))['A2_B1_hz'],
                          left['frequency_ghz'],right['frequency_ghz'],xtol=1e-8)
            point = search.at(float(root))
            if abs(point['A2_B1_hz'])<1:
                return dict(found=True,point=point,method='verified bracketed root')
        previous = points[-2]
        step = current['frequency_ghz']-previous['frequency_ghz']
        derivative = (current['A2_B1_hz']-previous['A2_B1_hz'])/step
        if abs(derivative)<1:
            break
        proposal = float(np.clip(proposal-current['A2_B1_hz']/derivative,*bounds))
    grid = [search.at(float(f)) for f in np.linspace(*bounds,9)]
    roots = []
    for left,right in zip(grid[:-1],grid[1:]):
        if left['A2_B1_hz']*right['A2_B1_hz']<0:
            roots.append(search.root(left['frequency_ghz'],right['frequency_ghz']))
    if roots:
        point = min(roots,key=lambda p:abs(p['frequency_ghz']-seed))
        return dict(found=True,point=point,method='verified scan bracket')
    point = min([*points,*grid],key=lambda p:abs(p['A2_B1_hz']))
    return dict(found=False,point=point,method='best evaluated allowed bias; no null found')


def validate_case(job):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from studies.parasitic_absorption import quantum_compare,write,PRODUCTION,AUDIT
    from studies.parasitic_absorption_audit import logical_audit
    from studies.group_two_parasitic import OffSearch
    from final_seven_validation import Gate

    candidate,context,output,identity = job
    name = f"g{candidate['group']}-{candidate['ports']}-{str(candidate['amount_ff']).replace('.','p')}ff"
    path = Path(output)/'cases'/(name+'.json')
    if path.exists():
        result = json.loads(path.read_text())
        if result['identity']!=identity:
            raise ValueError('Validation inputs/source changed; choose a fresh output')
    else:
        result = dict(identity=identity,group=candidate['group'],ports=candidate['ports'],
                      amount_ff=candidate['amount_ff'],fit=candidate['fit'],timings={},status='running')
    if result['status']=='complete':
        return dict(name=name,status='cached',seconds=sum(result['timings'].values()))
    physical = candidate['fit']['physical']
    cell = context['cell']
    point = context['gate_point']
    start = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            if 'quantum' not in result:
                stage = time.perf_counter()
                result['quantum'] = quantum_compare(cell,physical,point)
                result['timings']['gate_and_fixed_idle_seconds'] = time.perf_counter()-stage
                write(path,result)
            if 'off' not in result:
                stage = time.perf_counter()
                search = OffSearch(physical,AUDIT,capture_participation=True)
                result['off'] = find_off(search,context['nominal_off_ghz'],context['off_slope'],
                                         (point['fc_ghz']+.00001,9.99999))
                result['off_at_nominal'] = search.at(context['nominal_off_ghz'])
                result['off']['point']['basis'] = AUDIT
                result['timings']['off_search_seconds'] = time.perf_counter()-stage
                write(path,result)
            frequency = result['off']['point']['frequency_ghz']
            if 'logical_at_off' not in result:
                stage = time.perf_counter()
                result['logical_at_off'] = logical_audit(physical,frequency)
                result['secular_at_off'] = secular(result['logical_at_off'])
                result['timings']['logical_residual_seconds'] = time.perf_counter()-stage
                write(path,result)
            if 'off_production_check' not in result:
                stage = time.perf_counter()
                result['off_production_check'] = OffSearch(physical,PRODUCTION).at(frequency)
                result['timings']['off_basis_check_seconds'] = time.perf_counter()-stage
                write(path,result)
            if 'calibrated_gate_basis_check' not in result:
                stage = time.perf_counter()
                calibrated = result['quantum']['phase_recalibrated_gate']
                audit = Gate(dict(cell,physical=physical),AUDIT).at(
                    point['fc_ghz'],calibrated['fb_ghz'],point['time_ns'])
                fields = ('raw_cz_fidelity','postselected_cz_fidelity','logical_survival')
                differences = {field:float(audit[field]-calibrated[field])
                               for field in fields if field in audit and field in calibrated}
                result['calibrated_gate_basis_check'] = dict(metrics=audit,
                    changes=differences,agrees_to_1e4=all(abs(v)<=1e-4 for v in differences.values()))
                result['timings']['calibrated_gate_basis_seconds'] = time.perf_counter()-stage
                write(path,result)
        result['status'] = 'complete'
        result['off_pass'] = bool(result['off']['found'] and abs(result['off']['point']['A2_B1_hz'])<1)
        calibrated = result['quantum']['phase_recalibrated_gate']
        result['phase_pi_pass'] = abs(calibrated['phase_error_rad'])<1e-5
        result['total_seconds'] = sum(result['timings'].values())
        write(path,result)
        return dict(name=name,status='complete',seconds=result['total_seconds'],off_ghz=frequency,
                    off_pass=result['off_pass'],raw=calibrated['raw_cz_fidelity'],
                    selected=calibrated['postselected_cz_fidelity'])
    except Exception as error:
        result['status'] = 'error'
        result['error'] = str(error)
        result['failed_attempt_seconds'] = time.perf_counter()-start
        write(path,result)
        return dict(name=name,status='error',error=str(error))
    finally:
        gc.collect()


def run(candidates,output,workers,gate_file,reference_file):
    from studies.parasitic_absorption import write
    study = json.loads(candidates.read_text())
    if not study['cases'] or any(not row.get('success',False) for row in study['cases']):
        raise ValueError('Resolve failed electrostatic fits before quantum validation')
    cell = json.loads(Path(study['source']).read_text())
    # Explicit saved nominal controls replace dependencies on exploratory folders.
    from studies.parasitic_study import gate_controls
    gate = gate_controls(json.loads(gate_file.read_text()))
    reference = json.loads(reference_file.read_text())
    nominal = reference['nominal_off']
    seed = nominal['root']['frequency_ghz']
    slope = reference['off_slope']
    if workers < 1 or workers > 4:
        raise ValueError('Use one to four bounded workers')
    if abs(slope) < 1:
        raise ValueError('Supply a nonzero nominal OFF slope in Hz/GHz')
    from studies.parasitic_absorption import rebuild
    import numpy as np
    from floating_coupler.model import SCALE
    _, circuit = rebuild(cell['physical'])
    if not np.allclose(8*SCALE*circuit['K'][:5,:5]*1e9,
                       reference['baseline']['charge_coefficients_hz'],rtol=1e-12,atol=1e-6):
        raise ValueError('Nominal reference belongs to a different circuit')
    context = dict(cell=cell,gate_point=gate,nominal_off_ghz=seed,off_slope=slope)
    files = [Path(__file__),WF/'studies/parasitic_absorption.py',WF/'studies/parasitic_absorption_audit.py',
             WF/'studies/group_two_parasitic.py',WF/'studies/group_two_idle_participation.py',
             WF/'seven_readout.py',WF.parent/'core/cz_channel_policy.py']
    identity = hashlib.sha256(candidates.read_bytes()+json.dumps(context,sort_keys=True).encode()+
                              b''.join(path.read_bytes() for path in files)).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    manifest = output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:
        raise ValueError('Validation campaign inputs changed')
    nominal_participation = reference['nominal_participation']
    record = dict(identity=identity,candidate_source=str(candidates),source=study['source'],
                  scope='Frozen minimum-change geometries; saved idle rail EJs; coupler-only OFF retuning; static ON B-phase correction at fixed coupler/time.',
                  gate_point=gate,workers=workers,nominal_off_ghz=seed,nominal_off=nominal,
                  baseline=reference['baseline'],nominal_participation=nominal_participation,
                  results=[],case_count=len(study['cases']))
    start = time.perf_counter()
    write(manifest,record)
    jobs = [(row,context,str(output),identity) for row in study['cases']]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(validate_case,job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            record['results'].append(row)
            record['wall_seconds'] = time.perf_counter()-start
            write(manifest,record)
            print(json.dumps(row),flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    run(args.candidates,args.output,args.workers,args.gate,args.reference)
