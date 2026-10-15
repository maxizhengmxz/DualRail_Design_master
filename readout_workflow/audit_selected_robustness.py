"""Audit all saved witnesses and selected harder convergence cases."""
from dataclasses import asdict
import json
import numpy as np
from threadpoolctl import threadpool_limits
from model import Basis,Device,ReadoutModel
from robustness_rescue import TIERS,perturb,annotate,reasons
from run_selected_robustness import OUT


def main():
    rows=json.loads((OUT/'samples.json').read_text());designs=json.loads((OUT/'designs.json').read_text())
    assert len(rows)==900 and len({(r['design'],r['id']) for r in rows})==900
    if not (OUT/'raw_search_samples.json').exists():
        (OUT/'raw_search_samples.json').write_text(json.dumps(rows,indent=2))
    reassignments=[]
    for r in rows:
        if r.get('first_pass') is None: continue
        found=r.setdefault('discovered_at_tier',r['first_pass'])
        point=r['stages'][-1]['point'];target=designs[r['design']]['operating_frequency_ghz']
        earliest=next(i for i,t in enumerate(TIERS) if not reasons(point,t,target))
        r['earliest_feasible_tier']=earliest;r['first_pass']=earliest
        r['required_relaxations_for_witness']=[name for name,needed in (
            ('carrier',abs(point['operating_frequency_ghz']-target)>1e-7),
            ('hybridization',point['hybridization']<.95-1e-8),
            ('logical_gap',not .09-1e-8<=point['splitting_ghz']<=.11+1e-8),
            ('EJ_over_EC',min(point['ej1_over_ec1'],point['ej2_over_ec2'])<=50)) if needed]
        if earliest<found: reassignments.append(dict(design=r['design'],id=r['id'],discovered_at_tier=found,credited_tier=earliest))
    (OUT/'samples.json').write_text(json.dumps(rows,indent=2))
    accepted=[];errors=[];structural=0
    for r in rows:
        if r.get('error'): errors.append(dict(design=r['design'],id=r['id'],error=r['error']));continue
        reference=designs[r['design']];expected=asdict(perturb(reference,r['errors']))
        assert all(abs(r['device'][k]-v)<1e-10 for k,v in expected.items())
        d=r['device'];f=1/(2*np.pi*np.sqrt(d['Cr_fF']*1e-15*d['Lr_nH']*1e-9))/1e9
        z=np.sqrt(d['Lr_nH']*1e-9/(d['Cr_fF']*1e-15))
        assert abs(f-6*(1+r['errors'][5]))<1e-12 and abs(z-50)<1e-10
        if r.get('structural_failure'):
            ec=ReadoutModel(Device(**d)).ec_ghz[:2]*1000
            assert min(ec)<150-1e-7 or max(ec)>=400
            assert r['first_pass'] is None;structural+=1
        if r['first_pass'] is not None:
            stage=r['stages'][-1];p=stage['point'];i=r['first_pass']
            assert stage['tier']==TIERS[r['discovered_at_tier']]['name'] and not reasons(p,TIERS[i],reference['operating_frequency_ghz'])
            assert p['basis']==asdict(Basis(20,9,8,40))
            assert p['device']==r['device']
            accepted.append(r)
    selected={}
    for label in designs:
        group=[r for r in accepted if r['design']==label]
        for key in ('hybridization','ratio','lambda_max','splitting_ghz','ej1_over_ec1','ej2_over_ec2'):
            for reverse in (False,True):
                if not group: continue
                r=sorted(group,key=lambda r:r['stages'][-1]['point'][key],reverse=reverse)[0]
                selected[(label,r['id'])]=r
    checks=[];failed_checks=[]
    with threadpool_limits(limits=1):
        for (label,identifier),r in selected.items():
            p=r['stages'][-1]['point'];tier=TIERS[r['first_pass']]
            m=ReadoutModel(Device(**r['device']),Basis(25,12,10,60));q=annotate(m,p['ej1_ghz'],p['ej2_ghz'])
            checks.append(dict(design=label,id=identifier,tier=tier['name'],failures=reasons(q,tier,designs[label]['operating_frequency_ghz']),
                changes={key:q[key]-p[key] for key in ('hybridization','ratio','lambda_max','splitting_ghz','min_assignment_overlap')}))
        for label in designs:
            for i,tier in enumerate(TIERS):
                options=[]
                for r in rows:
                    if r['design']!=label: continue
                    for s in r.get('stages',[]):
                        if s['tier']==tier['name'] and not s['passes']: options.append((r,s))
                if not options: continue
                r,s=min(options,key=lambda item:abs(item[1]['point']['hybridization']-tier['H']))
                p=s['point'];m=ReadoutModel(Device(**r['device']),Basis(20,9,8,40));q=annotate(m,p['ej1_ghz'],p['ej2_ghz'])
                failures=reasons(q,tier,designs[label]['operating_frequency_ghz'])
                failed_checks.append(dict(design=label,id=r['id'],tier=tier['name'],screening_failures=s['failures'],large_basis_failures=failures,
                    classification_changed=not failures,H_difference=q['hybridization']-p['hybridization']))
    report=dict(samples=900,accepted_checks=len(accepted),structural_EC_failures=structural,numerical_errors=errors,
        tier_reassignments=reassignments,
        extra_basis_samples=len(checks),extra_basis_failures=[r for r in checks if r['failures']],extra_basis_checks=checks,
        failed_witness_checks=failed_checks,failed_witness_classification_changes=[r for r in failed_checks if r['classification_changed']])
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('extra_basis_checks','failed_witness_checks')},indent=2),flush=True)


if __name__=='__main__': main()
