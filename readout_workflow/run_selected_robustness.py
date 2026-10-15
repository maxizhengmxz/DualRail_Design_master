"""Checkpointed 3 x 300 paired six-parameter robustness/rescue study."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from scipy.stats import qmc
from robustness_rescue import PARAMETERS,TIERS,evaluate

BASE=Path(__file__).resolve().parent
OUT=BASE/'data'/'selected_robustness_2026_10_15'
SEED=20261015


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'designs.json').exists(): designs=json.loads((OUT/'designs.json').read_text())
    else:
        picks=json.loads((BASE/'data'/'refined_shunt_map_2026_10_14'/'stratified_robustness_proposed_picks.json').read_text())
        rows=json.loads((BASE/'data'/'refined_shunt_map_2026_10_14'/'uniform_points.json').read_text())
        lookup={(r['c1'],r['c2']):r['point'] for r in rows if r['status']=='validated_pass'}
        designs={str(i+1):lookup[(b['selected']['C1_fF'],b['selected']['C2_fF'])] for i,b in enumerate(picks['bands'])}
        assert [(p['device']['C1_shunt_fF'],p['device']['C2_shunt_fF']) for p in designs.values()]==[(87.,73.),(109.,90.),(119.,81.)]
        (OUT/'designs.json').write_text(json.dumps(designs,indent=2))
    if (OUT/'sampling.json').exists(): vectors=json.loads((OUT/'sampling.json').read_text())['scenarios']
    else:
        vectors=[dict(kind='interior',errors=(.02*v).tolist()) for v in 2*qmc.LatinHypercube(d=6,seed=SEED).random(224)-1]
        vectors += [dict(kind='corner',errors=(.02*np.array(v)).tolist()) for v in product((-1,1),repeat=6)]
        for i,key in enumerate(PARAMETERS):
            for sign in (-1,1):
                v=np.zeros(6);v[i]=sign*.02
                vectors.append(dict(kind='one_at_a_time',parameter=key,errors=v.tolist()))
        vectors=[dict(v,id=i) for i,v in enumerate(vectors)]
        (OUT/'sampling.json').write_text(json.dumps(dict(seed=SEED,parameters=PARAMETERS,scenarios=vectors),indent=2))
    return designs,vectors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    designs,vectors=initialize()
    fingerprint=sha256(b''.join((BASE/name).read_bytes() for name in ('robustness_rescue.py','model.py','study_dispersive.py'))).hexdigest()
    if args.pilot:
        selected=[vectors[i] for i in (0,224,287)]
        for label,p in designs.items():
            r=evaluate((label,p,selected[0]));(OUT/f'pilot_{label}.json').write_text(json.dumps(r,indent=2))
            print(json.dumps(dict(design=label,id=r['id'],first_pass=r['first_pass'],structural=r['structural_failure'],seconds=r['seconds'],stages=[dict(tier=s['tier'],failures=s['failures'],H=s['point']['hybridization']) for s in r['stages']])),flush=True)
        return
    db=sqlite3.connect(OUT/'study.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS results(design TEXT,id INTEGER,result TEXT,PRIMARY KEY(design,id))')
    db.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)')
    old=db.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
    if old and json.loads(old[0])!=fingerprint: raise RuntimeError('Numerical source changed; do not mix study revisions')
    db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('fingerprint',json.dumps(fingerprint)))
    db.commit();done={(a,b) for a,b in db.execute('SELECT design,id FROM results')}
    settings=dict(tiers=TIERS,seed=SEED,parameters=PARAMETERS,tolerance=.02,samples_per_design=300,
        interior=224,corners=64,one_at_a_time=12,source_fingerprint=fingerprint,
        note='Both EJs recalibrated; physical caps fixed after perturbation; constant resonator mode impedance; strict EC and EJ bounds retained at every tier.')
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2));start=time.time()
    def progress(stage):
        statuses={}
        for label in designs:
            rows=[json.loads(r[0]) for r in db.execute('SELECT result FROM results WHERE design=?',(label,))]
            statuses[label]={str(i):sum(r.get('first_pass')==i for r in rows) for i in range(5)}
            statuses[label].update(completed=len(rows),unrescued=sum(r.get('first_pass') is None for r in rows))
        p=dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),total_completed=len(done),total_requested=900,designs=statuses,elapsed_seconds=time.time()-start)
        (OUT/'progress.json').write_text(json.dumps(p,indent=2));print(json.dumps(p),flush=True)
    progress('running')
    tasks=[(label,p,v) for label,p in designs.items() for v in vectors if (label,v['id']) not in done]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(evaluate,t):t for t in tasks}
        for f in as_completed(futures):
            t=futures[f]
            try: r=f.result()
            except Exception as exc: r=dict(design=t[0],**t[2],first_pass=None,structural_failure=None,stages=[],error=repr(exc))
            db.execute('INSERT OR REPLACE INTO results VALUES(?,?,?)',(r['design'],r['id'],json.dumps(r)));db.commit();done.add((r['design'],r['id']))
            if len(done)%15==0: progress('running')
    rows=[json.loads(r[0]) for r in db.execute('SELECT result FROM results ORDER BY design,id')]
    (OUT/'samples.json').write_text(json.dumps(rows,indent=2));progress('numerical_complete')
    db.execute('PRAGMA wal_checkpoint(FULL)');db.close()


if __name__=='__main__': main()
