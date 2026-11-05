"""Fabrication study: independent readout triples followed by five transmons.

No seven-mode quantum calculation and no fabrication-time EC compensation.
The two nominal circuits are separately synthesized approximations. Shared
physical capacitor *fractional* errors and calibrated bare-frequency *shifts*
are transferred between them; their different static loading is not equated.
"""
from copy import deepcopy
from functools import lru_cache
import time
import json
import numpy as np
from scipy.optimize import brentq
from core.transmon import Basis
from .model import Model, FULL, SCALE
from .mixed_electrostatics import mixed_circuit, mixed_budget
from .pad_electrostatics import build_pad_circuit
from .design import ej_for_frequency
from .pairs import PairModel
from .calibrate_on import PhaseCalibration
from .precision import residual_metrics, compare, correct_gate_phase, refine_gate_controls

READOUT_FAST=Basis(20,6,5,30)
READOUT_MEDIUM=Basis(25,8,6,40)
READOUT_AUDIT=Basis(25,12,10,60)
SEARCH=dict(local_levels=5,pair_states=12,coupler_levels=5,ncut=30)
GATE_BASES=[SEARCH,dict(local_levels=6,pair_states=15,coupler_levels=6,ncut=35),dict(local_levels=7,pair_states=18,coupler_levels=7,ncut=40)]
OFF_BASES=[dict(local_levels=6,pair_states=15,coupler_levels=6,ncut=35),dict(local_levels=7,pair_states=18,coupler_levels=7,ncut=40),dict(local_levels=8,pair_states=20,coupler_levels=8,ncut=45)]


def edge_key(a,b):
    return 'link:'+':'.join(sorted((a,b)))


def parameter_names(base):
    ph=base['five_physical'];names=['self:'+n for n in FULL]
    names+=['ground:'+p for p,c in ph['grounds_ff'].items() if c>0]
    names += [edge_key(e['from'],e['to']) for e in ph['links']]
    for ro in base['readouts'].values():
        names += [edge_key(e['from'],e['to']) for e in ro['physical']['physical_links']]
        names += ['resonator:'+ro['case']['resonator_node']]
    return sorted(set(names))


def varied_parts(physical,errors,five=False):
    """Perturb actual positive capacitors once, without any refit or rebalance."""
    p=deepcopy(physical)
    g='grounds_ff' if five else 'ground_capacitances_ff'
    s='shunts_ff' if five else 'pad_mutual_capacitances_ff'
    l='links' if five else 'physical_links'
    for key,value in p[g].items():p[g][key]=value*(1+errors.get('ground:'+key,0.))
    for key,value in p[s].items():p[s][key]=value*(1+errors.get('self:'+key,0.))
    from .parasitics import with_parasitics
    nodes={pad for pair in p['pad_pairs'].values() for pad in pair}
    if not five:
        nodes.update(e[endpoint] for e in p[l] for endpoint in ('from','to'))
    p[l]=with_parasitics(p[l],p.get('parasitic_capacitances_ff'),nodes)
    for e in p[l]:e['capacitance_ff']*=1+errors.get(edge_key(e['from'],e['to']),0.)
    return p


def readout_device(ro,errors):
    from readout_workflow.calculation import Device
    p=varied_parts(ro['physical'],errors)
    rn=ro['case']['resonator_node']
    cr=p['resonator_self_capacitance_ff']*(1+errors.get('resonator:'+rn,0.))
    circuit=build_pad_circuit(p['pad_pairs'],p['ground_capacitances_ff'],{rn:cr},p['physical_links'],p['pad_mutual_capacitances_ff'])
    gauge=np.diag([-1.,1.,1.]);cm=gauge@circuit.effective_modal_capacitance_ff@gauge
    c12,c1r,c2r=-cm[0,1],-cm[0,2],-cm[1,2]
    return Device(cm[0,0]-c12-c1r,cm[1,1]-c12-c2r,c12,c1r,c2r,cm[2,2]-c1r-c2r,p['resonator_inductance_nh'])


def readout_fit(ro,errors,representative=False):
    """Shared readout rescue ladder; never hand off a rejected endpoint."""
    from standalone_pad_readout import policy
    from readout_workflow.recovery import recover
    policy(ro['case'])
    case=dict(ro['case'])
    case['recovery_target_dressed_ghz']=float(ro.get('nominal',{}).get('operating_frequency_ghz',case['bare_frequency_guide_ghz']))
    return recover(readout_device(ro,errors),ro['coordinates'][:2],case,representative)


@lru_cache(maxsize=64)
def _cached_readout(ro_json,errors_json,representative):
    return readout_fit(json.loads(ro_json),json.loads(errors_json),representative)


def readout_for_sample(ro,errors,representative=False):
    """Exact cache excludes components absent from this isolated subsystem."""
    p=ro['physical']
    relevant={'self:'+n for n in p['pad_pairs']} | {'ground:'+n for n in p['ground_capacitances_ff']}
    relevant |= {edge_key(e['from'],e['to']) for e in p['physical_links']}
    relevant.add('resonator:'+ro['case']['resonator_node'])
    subset={k:v for k,v in errors.items() if k in relevant and v!=0}
    return _cached_readout(json.dumps(ro,sort_keys=True),json.dumps(subset,sort_keys=True),representative)

def five_model(base,errors,idle_shifts):
    p=varied_parts(base['five_physical'],errors,True)
    circuit=mixed_circuit(p['pad_pairs'],p['grounds_ff'],p['shunts_ff'],p['links'])
    ec=circuit['loaded_ec_ghz'];j=8*SCALE*circuit['K'];np.fill_diagonal(j,0.)
    fs={n:base['five_physical']['idle_frequencies_ghz'][n]+idle_shifts.get(n,0.) for n in FULL}
    ej=np.array([ej_for_frequency(fs[n],ec[i],35) for i,n in enumerate(FULL)])
    model=Model(FULL,ec,ej,j,Basis(ncut=35,transmon_levels=6),{'channel_selection':base['selection']})
    budget=mixed_budget(p['pad_pairs'],p['grounds_ff'],p['shunts_ff'],p['links'],p['links'][:3])
    return model,dict(loaded_ec_ghz=dict(zip(FULL,ec.tolist())),bare_idle_frequencies_ghz=fs,budget=budget)


def off_rescue_plan():
    """Broaden to <11 GHz in the same basis BEFORE any basis rescue."""
    return [(OFF_BASES[0],9.9999,'original interval'),
            (OFF_BASES[0],10.9999,'broadened interval'),
            *[(basis,10.9999,'larger-basis rescue') for basis in OFF_BASES[1:]]]


def _off_diagnostics(detail):
    """Keep physical/logical exchanges, ZZ and channel detunings together."""
    physical=detail['physical_exchange'];actual=float(physical['g_mhz']*1e6)
    try:
        metrics=residual_metrics(detail)
        from final_seven_validation import residuals
        channels=residuals(detail);status='evaluated'
    except (ValueError,KeyError,TypeError):
        metrics=dict(physical_exchange_hz=actual,logical_exchange_hz=None,logical_double_flip_hz=None,secular_logical_zz_hz=None)
        channels=[];status='logical/auxiliary projection unreliable; physical value reported separately'
    return dict(metrics=metrics,residual_channels=channels,diagnostic_status=status,
                physical_projection_valid=physical.get('valid',True))


def find_off(model,nominal_fc):
    """Physical 0100-0010 passes at |g|<1 Hz; basis accuracy is separate."""
    from .off_search import threshold_off_search
    from .channel_policy import off_target_definition
    systems={};history=[]
    for basis,upper,label in off_rescue_plan():
        key=tuple(basis.values())
        if key not in systems:systems[key]=PairModel(model,**basis)
        system=systems[key];cache={}
        def g(f):
            f=float(f)
            if f not in cache:
                row=system.physical_off(f)['physical_exchange']
                cache[f]=row['g_mhz'] if row.get('valid',True) else float('nan')
            return cache[f]
        grid=np.unique(np.r_[np.linspace(7.0001,upper,25),np.clip(nominal_fc,7.0001,upper)])
        search=threshold_off_search(g,grid,nominal_fc,threshold_hz=1.)
        entry=dict(basis=basis,range_ghz=[7.0001,upper],stage=label,search=search)
        history.append(entry)
        if not search['passed']:continue
        fc=search['fc_ghz'];detail=system.evaluate(fc);physical=detail['physical_exchange']
        actual=float(physical['g_mhz']*1e6)
        if not physical.get('valid',True) or not abs(actual)<1.:
            entry['full_projection_residual_hz']=actual;entry['passed']=False;continue
        diagnostics=_off_diagnostics(detail)
        reference=None
        if fc>10.:
            reference=dict(fc_ghz=10.,ej_ghz=ej_for_frequency(10.,model.ec[2],basis['ncut']),basis=basis,
                           scope='Fixed 10 GHz reference; same fabricated capacitors, loaded EC/J and idle rail EJs as recovered OFF. Only coupler EJ changes.')
            try:reference.update(_off_diagnostics(system.evaluate(10.)))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                reference.update(diagnostic_status='unavailable: projection/eigensolver failure',error=repr(error))
        return dict(found=True,passed=True,residual_pass=True,status='Physical exchange magnitude below 1 Hz',
                    fc_ghz=fc,ej_ghz=ej_for_frequency(fc,model.ec[2],basis['ncut']),
                    acceptance_threshold_hz=1.,target_residual_hz=actual,basis=basis,
                    allowed_off_upper_exclusive_ghz=11.,off_definition=off_target_definition('five'),
                    converged=False,convergence_checked=False,
                    convergence_status='Not assessed: magnitude acceptance is separate from basis agreement',
                    history=history,reference_10ghz=reference,minimum_overlap=detail['minimum_subspace_overlap'],**diagnostics)
    return dict(found=False,passed=False,residual_pass=False,status='No verified physical residual below 1 Hz in finite rescue search below 11 GHz',
                acceptance_threshold_hz=1.,allowed_off_upper_exclusive_ghz=11.,history=history,
                converged=False,convergence_checked=False,off_definition=off_target_definition('five'))


def gate_precision(history):
    """Fidelity/survival certification; input leakage stays a detailed diagnostic."""
    for item in reversed(history):
        comparison=item.get('final_phase_corrected_same_control_comparison',item.get('comparison'))
        if comparison:
            changes=comparison['changes']
            passed=all(changes.get(k,float('inf'))<=1e-4 for k in
                       ['raw_cz_fidelity','postselected_cz_fidelity','logical_survival'])
            return dict(converged=passed,fidelity_precision_passed=passed,
                        leakage_precision_passed=changes.get('worst_input_leakage_change',float('inf'))<=1e-4,
                        precision_warning=not passed,changes=changes)
    return dict(converged=False,fidelity_precision_passed=False,precision_warning=True)


def readout_handoff(result):
    """Use an accepted point or the last rescue tier's best diagnostic, flagged."""
    if result.get('passed'):
        return dict(bare_frequencies_ghz=result['bare_frequencies_ghz'],accepted=True,
                    source='accepted readout')
    point=result.get('diagnostic_point')
    if point is None:raise ValueError('No finite readout candidate available for downstream calculation')
    return dict(bare_frequencies_ghz=[point['bare_f1_ghz'],point['bare_f2_ghz']],accepted=False,
                source='last rescue tier best diagnostic; readout constraints unmet',
                failed_constraints=result.get('nominal_failures',[]),point=point)


def gate_fit(model,seed,off,representative=False):
    """Maximize postselected fidelity; raw/survival are reported, never cut."""
    bounds=(7.0001,min(10.999,off.get('fc_ghz',11.)-.001));calibrators={};approximate=[]
    def cal(fc,basis=SEARCH,wide=False):
        key=(float(fc),tuple(basis.values()),wide)
        if key not in calibrators:
            width=.2 if wide else .05
            calibrators[key]=PhaseCalibration(model,float(fc),basis,[seed['fb_ghz']-width,seed['fb_ghz']+width],phase_samples=81 if wide else 25)
            original_at=calibrators[key].at
            def tracked_at(fb,t):
                row=original_at(fb,t)
                tracked_valid(row)
                return row
            calibrators[key].at=tracked_at
        return calibrators[key]
    def rail_controls_valid(row):
        ej=model.ej.copy()
        for i in (3,4):ej[i]=ej_for_frequency(row['fb_ghz'],model.ec[i],35)
        return all(45*model.ec[i]<ej[i]<25 for i in (0,1,3,4))
    def gate_valid(row):
        return (rail_controls_valid(row) and row['actual_bias_score_diagnostics']['maximum_logical_ratio']<.2 and row['minimum_dressed_reference_overlap']>=.8)
    def tracked_valid(row):
        valid=gate_valid(row)
        if valid and np.isfinite(row.get('phase_error_rad',float('nan'))):
            approximate.append(row)
        return valid
    def approximate_result(status,rows):
        valid=[r for r in rows if gate_valid(r) and np.isfinite(r.get('phase_error_rad',float('nan')))]
        if not valid:return dict(found=False,approximate=False,status=status+'; no valid tracked candidate')
        error=min(abs(np.angle(np.exp(1j*r['phase_error_rad']))) for r in valid)
        close=[r for r in valid if abs(np.angle(np.exp(1j*r['phase_error_rad'])))<=error+1e-7]
        best=max(close,key=lambda r:r['postselected_cz_fidelity'])
        return dict(found=False,approximate=True,phase_pi_met=False,best=best,status=status,
                    optimization_objective='wrapped phase error first; postselected fidelity within 1e-7 rad tie',
                    conditional_phase_rad=float(np.angle(np.exp(1j*(np.pi+best['phase_error_rad'])))),
                    converged=False,precision_warning=False,precision_status='Approximate candidate; basis precision not assessed')
    candidates=[]
    for fc in np.unique(np.clip([seed['fc_ghz'],seed['fc_ghz']-.1,seed['fc_ghz']+.1],*bounds)):
        c=cal(float(fc))
        for t in np.unique(np.r_[90.,100.,110.,120.,seed['time_ns']]):
            try:
                r=c.phase_root(float(t),objective='postselected_cz_fidelity',minimum_return=None,acceptance=tracked_valid)
                candidates.append(r)
            except ValueError:pass
    if not candidates:
        for fc in np.linspace(*bounds,7):
            c=cal(float(fc),wide=True)
            for t in [70.,85.,100.,115.,130.]:
                try:candidates.append(c.phase_root(t,objective='postselected_cz_fidelity',minimum_return=None,acceptance=tracked_valid))
                except ValueError:pass
    if not candidates:
        # Refine phase locally from the best grid row; cached spectra serve all times.
        from scipy.optimize import minimize
        if approximate:
            start=min(approximate,key=lambda r:abs(r['phase_error_rad']))
            def objective(x):
                row=cal(float(x[0]),wide=True).at(float(x[1]),float(x[2]))
                return abs(np.angle(np.exp(1j*row['phase_error_rad']))) if gate_valid(row) else 10.
            minimize(objective,[start['fc_ghz'],start['fb_ghz'],start['time_ns']],method='Powell',
                     bounds=[bounds,(seed['fb_ghz']-.2,seed['fb_ghz']+.2),(70.,130.)],
                     options=dict(maxfev=36,xtol=1e-5,ftol=1e-6))
        recovered=[r for r in approximate if abs(r['phase_error_rad'])<1e-7]
        if recovered:candidates=recovered
        else:return approximate_result('Approximate gate; phase-pi condition unmet in finite search',approximate)
    best=max(candidates,key=lambda r:r['postselected_cz_fidelity']);origin=np.array([best['fc_ghz'],best['fb_ghz'],best['time_ns']]);scale=np.array([.05,.005,10.])
    def at(x):
        fc,fb,t=origin+scale*x
        if fb<=0:return dict(raw_cz_fidelity=0.,phase_error_rad=float('nan'))
        row=cal(float(fc)).at(float(fb),float(t))
        if not rail_controls_valid(row) or row['actual_bias_score_diagnostics']['maximum_logical_ratio']>=.2 or row['minimum_dressed_reference_overlap']<.8:row=dict(row,phase_error_rad=float('nan'))
        return row
    limits=[((bounds[0]-origin[0])/.05,(bounds[1]-origin[0])/.05),(None,None),((70-origin[2])/10,(130-origin[2])/10)]
    try:
        refinement=refine_gate_controls(at,[0.,0.,0.],limits,maxiter=8,max_evaluations=32,objective="postselected_cz_fidelity")
        if refinement['best']['postselected_cz_fidelity']>best['postselected_cz_fidelity']:best=refinement['best']
    except (ValueError,RuntimeError):refinement=dict(stop_reason='finite local optimizer failed; best feasible seed retained')
    history=[];last=None;converged=False
    for b in GATE_BASES:
        c=cal(best['fc_ghz'],b);row=c.at(best['fb_ghz'],best['time_ns']);h=dict(basis=b,metrics=row)
        if last is not None:
            h['comparison']=compare(last,row,'gate');converged=gate_precision([h])['converged']
        history.append(h);last=row
        if converged:break
    try:final=correct_gate_phase(lambda f:c.at(f,best['time_ns']),last,max_evaluations=6)
    except ValueError:return approximate_result('Approximate gate; high-basis phase correction exhausted finite budget',[last])
    if not rail_controls_valid(final) or final['actual_bias_score_diagnostics']['maximum_logical_ratio']>=.2:
        return approximate_result('Approximate gate; final phase-corrected point invalid',approximate)
    if len(history)>1:
        prior_at_final=cal(final['fc_ghz'],history[-2]['basis']).at(final['fb_ghz'],final['time_ns'])
        final_comparison=compare(prior_at_final,final,'gate');converged=final_comparison['converged']
        history.append(dict(final_phase_corrected_same_control_comparison=final_comparison))
    return dict(found=True,phase_pi_met=True,approximate=False,best=final,search_best=best,optimization_objective='postselected_cz_fidelity',fidelity_failure_threshold=None,minimum_return_cutoff=None,history=history,**gate_precision(history),refinement={k:v for k,v in refinement.items() if k not in ['best','x']},scope='Best found within finite local starts and 70-130 ns; not a global optimum. Static dressed ON, no decoherence or ramps.')


def sample(task):
    began=time.perf_counter();base=task['base'];errors=task['errors'];result=dict(id=task['id'],kind=task['kind'],errors=errors,stages={})
    try:
        t=time.perf_counter();readouts={};shifts={}
        for pair in ['A','B']:
            retained=task.get('retained_readouts',{}).get(pair)
            if retained is not None:
                r=deepcopy(retained);r['recovery_provenance']='retained original accepted readout';r.setdefault('tier','retained_original_pass')
            else:r=readout_for_sample(base['readouts'][pair],errors,task.get('representative',False))
            readouts[pair]=r
            handoff=readout_handoff(r);r['downstream_handoff']=handoff
            for i,n in enumerate(base['readouts'][pair]['case']['nodes']):shifts[n]=handoff['bare_frequencies_ghz'][i]-base['nominal_readout_positions'][pair][i]
        result['readout']=readouts;result['idle_shifts_ghz']=shifts;result['stages']['readout_seconds']=time.perf_counter()-t
        accepted=all(r['passed'] for r in readouts.values())
        result['recovery_status']='readout_recovered' if accepted else 'readout_unrecovered_continued'
        result['downstream_readout_accepted']=accepted
        model,physical=five_model(base,errors,shifts);result['five_physical_summary']=physical
        t=time.perf_counter();result['off']=find_off(model,base['off_ghz']);result['stages']['off_seconds']=time.perf_counter()-t
        seed=dict(base['gate_seed']);seed['fb_ghz']+=sum(shifts[n] for n in ['B1','B2'])/2
        t=time.perf_counter();result['gate']=gate_fit(model,seed,result['off'],task.get('representative',False));result['stages']['gate_seconds']=time.perf_counter()-t
        result['status']='evaluated'
    except Exception as error:
        import traceback
        result.update(status='calculation_error',error=repr(error),traceback=traceback.format_exc())
    result['seconds']=time.perf_counter()-began
    return result
