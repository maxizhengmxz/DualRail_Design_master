"""Independent ledger coverage checks and an extra basis-convergence sample."""
import json
from pathlib import Path
import sqlite3
import numpy as np
from refined_pair_search import failures, validate
from model import Basis
from run_refined_study import UNIFORM

OUT=Path(__file__).resolve().parent/'data'/'refined_shunt_map_2026_10_14'


def main():
    db=sqlite3.connect(OUT/'study.sqlite')
    rows=[json.loads(r[0]) for r in db.execute('SELECT result FROM points')]
    meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM meta')}
    db.close()
    bypair={(r['c1'],r['c2']):r for r in rows}
    requested={tuple(p) for p in meta['uniform_requested']}
    assert requested=={(float(a),float(b)) for a in UNIFORM for b in UNIFORM if a>b}
    missing=sorted(requested-set(bypair))
    good=[r for pair,r in bypair.items() if pair in requested and r['status']=='validated_pass']
    assert all(not failures(r['point']) for r in good)
    assert all(r['point']['basis']=={'ncut':20,'transmon_levels':9,'resonator_levels':8,'eigenlevels':40} for r in good)
    assert all(abs(r['point']['device']['C1r_fF']-r['point']['device']['C2r_fF'])<1e-12 for r in good)
    keys=('ratio','hybridization','lambda_max','splitting_ghz','loaded_ec1_mhz','loaded_ec2_mhz','operating_frequency_ghz','min_assignment_overlap','ej1_over_ec1','ej2_over_ec2')
    selected={}
    for key in keys:
        for reverse in (False,True):
            r=sorted(good,key=lambda r:r['point'][key],reverse=reverse)[0]
            selected[(r['c1'],r['c2'])]=r
    convergence=[]
    for pair,row in selected.items():
        higher=validate(row,Basis(25,12,10,60))
        delta={key:higher['point'][key]-row['point'][key] for key in keys}
        convergence.append({'pair':pair,'basis':higher['point']['basis'],'status':higher['status'],
                            'failures':higher['failures'],'changes':delta})
    counts={}
    for pair in requested:
        status=bypair.get(pair,{}).get('status','not_sampled')
        counts[status]=counts.get(status,0)+1
    report={'requested_uniform':len(requested),'missing_uniform':missing,'status_counts':counts,
            'accepted_constraint_violations':0,'extra_basis_samples':len(convergence),
            'extra_basis_failures':[r for r in convergence if r['failures']],
            'extra_basis_convergence':convergence,
            'max_absolute_changes':{key:max(abs(r['changes'][key]) for r in convergence) for key in keys},
            'carrier_above_6GHz_count':sum(r['point']['operating_frequency_ghz']>6 for r in good)}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({key:value for key,value in report.items() if key!='extra_basis_convergence'},indent=2),flush=True)


if __name__=='__main__': main()
