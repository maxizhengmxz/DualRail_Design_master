"""Frozen-hardware seven-mode OFF, readout maps, and dressed static CZ validation."""
import argparse,json,sys,time
from pathlib import Path
from dataclasses import replace
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq,linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from seven_readout import SevenReadout
from integrated_readout_range import frozen_model
from floating_coupler.design import ej_for_frequency
from floating_coupler.pairs import local_energy_frame
from floating_coupler.paired_design import selected_channel,channel_scores,off_channel
from floating_coupler.on_gate import secular_logical_shifts,energy_axis_matter_hamiltonian
from floating_coupler.gate import gate_metrics
from readout_common_sweep import metrics
from floating_coupler.channel_policy import selection_from_request,require_selection,require_same_channel,LOGICAL_ORDER,off_target_definition,require_off_definition
from floating_coupler.precision import refine_gate_controls,correct_gate_phase
from floating_coupler.calibrate_on import cached_duration_search,warm_start_gate
from floating_coupler.on_gate import dressed_on_metrics_many

def off_exchange(detail):
    return off_channel(detail)

SEARCH=dict(local_levels=6,pair_states=10,coupler_levels=4,ncut=25)
AUDIT=dict(local_levels=6,pair_states=12,coupler_levels=5,ncut=35)
LARGE=dict(local_levels=7,pair_states=15,coupler_levels=6,ncut=40)

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2))

def residuals(detail):
    h=energy_axis_matter_hamiltonian(detail);labels=detail['reference_labels'];logical=['00','01','10','11'];aux=[x for x in labels if x.startswith(('A','B'))];rows=[]
    for i,a in enumerate(labels):
        for j in range(i+1,len(labels)):
            b=labels[j]
            relevant=(a in logical and b in logical) or (a in logical and b in aux) or (b in logical and a in aux) or (a.endswith('g') and b.startswith('g'))
            if not relevant:continue
            if a in logical and b in logical:
                lh=np.asarray(detail['logical_exchange']['local_energy_hamiltonian_ghz'])
                li,lj=logical.index(a),logical.index(b)
                g=float(lh[li,lj]*1000);delta=float((lh[li,li]-lh[lj,lj])*1000)
            else:
                g=float(h[i,j]*1000);delta=float((h[i,i]-h[j,j])*1000)
            rows.append(dict(left=a,right=b,g_mhz=g,detuning_mhz=delta,abs_g_over_delta=abs(g/delta) if abs(delta)>1e-12 else None))
    return sorted(rows,key=lambda r:abs(r['g_mhz']),reverse=True)

def off(cell,folder):
    previous=folder/'off.json'
    if previous.exists():require_off_definition(json.loads(previous.read_text()).get('off_definition'),'seven')
    base=frozen_model(cell['physical']);seed=cell['off_ghz'];cache={}
    bounds=cell.get('off_bracket_ghz',[7.00001,9.99999])
    def at(fc):
        if fc not in cache:cache[fc]=SevenReadout(base,cell['physical'],float(fc),basis=AUDIT).solve()
        return cache[fc]
    def g(fc):
        system=SevenReadout(base,cell['physical'],float(fc),basis=AUDIT,off_only=True).solve()
        return off_exchange(system.physical_off_detail())['g_mhz']
    coarse=[]
    for fc in np.linspace(*bounds,17):
        s0=SevenReadout(base,cell['physical'],float(fc),basis=SEARCH,off_only=True).solve();coarse.append((float(fc),off_exchange(s0.physical_off_detail())['g_mhz']))
    brackets=[(a,b) for (a,ga),(b,gb) in zip(coarse[:-1],coarse[1:]) if ga*gb<0]
    from floating_coupler.off_search import exchange_roots
    roots=exchange_roots(g,np.linspace(*bounds,17),tolerance=1e-6)
    root=min(roots,key=lambda f:abs(f-seed)) if roots else min(coarse,key=lambda row:abs(row[1]))[0]
    s=at(root);detail=s.matter_detail();result=dict(off_definition=off_target_definition('seven'),source_off_ghz=seed,off_ghz=root,off_ej_ghz=ej_for_frequency(root,base.ec[2],35),basis=AUDIT,physical=cell['physical'],channel_selection=base.electrostatics.get('channel_selection'),channel_source='0100',channel_target='0010',coarse_scan=coarse,all_roots_ghz=roots,channel_cancelled=bool(roots),status='Verified cancellation' if roots else 'No cancellation in requested OFF interval; reporting best sampled bias, not a successful OFF zero',channel=off_exchange(detail),logical_residuals=detail['logical_exchange'],off_bracket_ghz=bounds,secular=secular_logical_shifts(detail),readout_A=s.readout('A'),readout_B=s.readout('B'),residual_channels=residuals(detail),minimum_subspace_overlap=detail['minimum_subspace_overlap'],scope='All physical components and original saved idle rail EJs fixed; only coupler EJ varied. OFF temporarily nulls physical 0100-0010; ON CZ selection is separate provenance. No geometry redesign or extension beyond requested OFF bounds. Not a readout reoptimization.')
    write(folder/'off.json',result);print('FINAL_OFF',json.dumps({k:v for k,v in result.items() if k not in ['physical','residual_channels']}),flush=True)
    return result

def scan(cell,folder,pair):
    offdata=json.loads((folder/'off.json').read_text());require_off_definition(offdata.get('off_definition'),'seven');base=frozen_model(cell['physical']);idx=[0,1] if pair=='A' else [3,4];names=['A1','A2'] if pair=='A' else ['B1','B2'];anchor=np.array([cell['physical']['bare_frequencies_ghz'][n] for n in names]);path=folder/f'readout-{pair}.json';result=json.loads(path.read_text()) if path.exists() else {'rows':[]}
    if result.get('off_ghz') is not None and abs(result['off_ghz']-offdata['off_ghz'])>1e-9:result['rows']=[]
    result.update(pair=pair,off_ghz=offdata['off_ghz'],basis=AUDIT,anchor_bare_ghz=anchor.tolist(),scope='Common shift of the saved standalone idle rail f01 values. Geometry, resonator L, opposite rail EJs and finalized OFF coupler EJ fixed. No differential reoptimization. Worst over spectator logical states; sign-grid certification is not repeated at every frequency.')
    for delta in np.arange(-100,101,10):
        if any(r['shift_mhz']==delta for r in result['rows']):continue
        f=anchor+delta/1000;ej=base.ej.copy()
        for i,ff in zip(idx,f):ej[i]=ej_for_frequency(float(ff),base.ec[i],35)
        model=replace(base,ej=ej);s=SevenReadout(model,cell['physical'],offdata['off_ghz'],basis=AUDIT).solve();rr=s.readout(pair);window=(.14,.16) if pair=='A' else (.08,.10)
        row=dict(shift_mhz=float(delta),bare_frequencies_ghz=f.tolist(),mean_transmon_frequency_ghz=float(np.mean(f)),**metrics(rr,ej[idx],base.ec[idx],window),raw_rows=rr);result['rows'].append(row);result['rows'].sort(key=lambda r:r['shift_mhz']);write(path,result);print('READOUT',pair,delta,row['failures'],flush=True)
    return result

def gate_detail(system):
    """Canonical vacuum matter references, retaining logical and CZ auxiliary states.

    Exact selected seven-mode eigenpairs embed this retained manifold. Only the
    even parity sector is needed for the four logical gate inputs.
    """
    labels=[label for label in system.pair.labels if system.parity[system.refs[(label,0,0)]]==1]
    sector=np.flatnonzero(system.parity==1);refs=np.array([system.refs[(label,0,0)] for label in labels]);positions=np.searchsorted(sector,refs)
    en,v=eigh(system.ham[sector,:][:,sector].toarray(),subset_by_index=(0,system.eigenlevels-1),check_finite=False)
    projection=v[positions,:];_,cols=linear_sum_assignment(-projection**2)
    if max(cols)>system.eigenlevels-3:raise ValueError('Insufficient gate eigenlevel window')
    u,ss,vh=np.linalg.svd(projection[:,cols]);rotation=u@vh;ham=(rotation*en[cols])@rotation.T
    logical=[labels.index(x) for x in ['00','01','10','11']]
    return dict(reference_labels=labels,effective_hamiltonian_ghz=ham.tolist(),logical_local_energy_frame=local_energy_frame(ham[np.ix_(logical,logical)]),minimum_subspace_overlap=float(ss.min()**2),channel_selection=system.model.electrostatics.get('channel_selection'))

class Gate:
    def __init__(self,cell,basis,photons=3):
        self.cell=cell;self.base=frozen_model(cell['physical']);self.basis=basis;self.photons=photons;self.calls=0
        self.selection=require_selection(self.base.electrostatics.get('channel_selection'))
        if self.selection.moved_pair!='B':raise ValueError('This case-specific seven-mode runner supports moving B only; do not silently move the wrong rail')
    @lru_cache(maxsize=256)
    def spectrum(self,fc,fb):
        ej=self.base.ej.copy()
        # Preserve the recorded stationary idle controls, including mandatory
        # integrated dressed-idle calibration when present in this cell.
        fixed=self.cell['request']['idle_bare_frequencies_ghz']
        for i,n in [(0,'A1'),(1,'A2')]:ej[i]=ej_for_frequency(fixed[n],self.base.ec[i],self.basis['ncut'])
        for i in [3,4]:ej[i]=ej_for_frequency(float(fb),self.base.ec[i],self.basis['ncut'])
        system=SevenReadout(replace(self.base,ej=ej),self.cell['physical'],float(fc),basis=self.basis,photons=self.photons)
        detail=gate_detail(system);ham=energy_axis_matter_hamiltonian(detail);en,v=eigh(ham);labels=detail['reference_labels'];ii=[labels.index(x) for x in LOGICAL_ORDER];aux=labels.index(self.selection.auxiliary);self.calls+=1
        if self.calls%10==0:print('GATE_EVAL',self.basis,self.calls,fc,fb,flush=True)
        detail['_gate_vectors']=v
        detail['_gate_scores']=channel_scores(detail,self.selection.source_logical,self.selection.auxiliary)
        return en-en.min(),v[ii,:].T,v[aux,:],detail
    def at(self,fc,fb,t):
        return self.at_times(fc,fb,[t])[0]

    def duration_evaluator(self,fc,fb):
        """Capture one eigenspectrum and score once; time scans do no diagonalization."""
        en,coeff,aux,detail=self.spectrum(float(fc),float(fb))
        scores=detail['_gate_scores'];selection=self.selection.to_dict()
        source=LOGICAL_ORDER.index(self.selection.source_logical)
        def evaluate(times):
            rows=dressed_on_metrics_many(en,coeff,times)
            for row in rows:
                t=row['time_ns']
                midpoint=aux@(np.exp(-1j*np.pi*en*t)*coeff[:,source])
                row.update(fc_ghz=float(fc),fb_ghz=float(fb),
                    minimum_dressed_reference_overlap=detail['minimum_subspace_overlap'],
                    midpoint_selected_auxiliary_from_selected_source=float(abs(midpoint)**2),
                    channel_selection=selection,actual_bias_score_diagnostics=scores,
                    worst_input_nonlogical_population=float(max(row['leakage_by_input'])))
            return rows
        return evaluate

    def at_times(self,fc,fb,times):
        return self.duration_evaluator(fc,fb)(times)

def gate_precision_audit(cell,point,evaluators=()):
    """Modest fixed-control checks; no mandatory huge matter/photon audit."""
    from floating_coupler.precision import verify_fixed_controls
    plans=cell.get('gate_audit_bases',[dict(basis=SEARCH,photons=3),dict(basis=AUDIT,photons=3)])
    def evaluate(plan):
        cal=next((cal for cal in evaluators if cal.cell is cell and
                  cal.basis==plan['basis'] and cal.photons==plan.get('photons',3)),None)
        if cal is None:cal=Gate(cell,plan['basis'],plan.get('photons',3))
        return cal.at(point['fc_ghz'],point['fb_ghz'],point['time_ns'])
    return verify_fixed_controls(evaluate,plans,'gate')


def diagnose_gate(cell,folder):
    path=folder/'gate.json'
    if path.exists():point=json.loads(path.read_text())['best']
    else:
        d=json.loads((folder/'gate-search.json').read_text());point=max(d['results'],key=lambda r:r['raw_cz_fidelity'])
    cal=Gate(cell,AUDIT);require_same_channel(cal.selection,point.get('channel_selection'));en,c,aux,detail=cal.spectrum(point['fc_ghz'],point['fb_ghz']);v=detail['_gate_vectors'];evolved=v@(np.exp(-1j*np.pi*en*point['time_ns'])[:,None]*c)
    populations={label:(abs(evolved[i,:])**2).tolist() for i,label in enumerate(detail['reference_labels']) if label.startswith(('A','B'))}
    peak=np.zeros_like(evolved.real)
    for t in np.linspace(0,point['time_ns'],501):
        psi=v@(np.exp(-2j*np.pi*en*t)[:,None]*c);peak=np.maximum(peak,abs(psi)**2)
    maxima={label:peak[i,:].tolist() for i,label in enumerate(detail['reference_labels']) if label.startswith(('A','B'))}
    out=dict(point=point,logical_input_order=['00','01','10','11'],midpoint_auxiliary_populations=populations,maximum_auxiliary_populations_over_gate=maxima)
    write(folder/'gate-mechanism.json',out);print(json.dumps(out,indent=2),flush=True)
    return out

def gate(cell,folder,previous):
    """Five-body warm start; outer frequency search, cached inner duration search."""
    began=time.perf_counter()
    seed=previous.get('best',previous.get('gate_final',previous.get('phase_recalibrated_endpoint',previous)))
    if not all(k in seed for k in ('fc_ghz','fb_ghz','time_ns','channel_selection')):
        raise ValueError('Supply a calibrated five-body gate point with frequencies, duration and selection')
    require_same_channel(cell['physical'].get('channel_selection'),seed['channel_selection'])
    cal=Gate(cell,SEARCH)
    lo=cell['request'].get('on_lower_exclusive_ghz',7.)+1e-6
    fcbounds=(lo,cell['off_ghz']-.001)
    tbounds=tuple(cell.get('gate_time_bounds_ns',previous.get('time_bounds_ns',[70.,130.])))
    warm=warm_start_gate(cal,seed,fcbounds,tbounds)
    start=warm['best']
    origin=np.array([start['fc_ghz'],start['fb_ghz']]);scale=np.array([.05,.005])
    bounds=[((fcbounds[0]-origin[0])/scale[0],(fcbounds[1]-origin[0])/scale[0]),(None,None)]

    @lru_cache(maxsize=64)
    def at_bias(fc,fb):
        if fb<=0:return None
        if fc==start['fc_ghz'] and fb==start['fb_ghz']:return start
        scan=cached_duration_search(cal.duration_evaluator(fc,fb),tbounds,start['time_ns'],points=201)
        candidate=scan['best_return']
        if candidate is None:return None
        # Time alone need not cross phase pi at an offset frequency. Choose the
        # best cached return, THEN perform the necessary local frequency/phase
        # correction. All reported candidates use the corrected Hamiltonian.
        try:
            row=correct_gate_phase(lambda f:cal.at(fc,f,candidate['time_ns']),candidate)
        except ValueError:return None
        if min(row['return_by_input'])<=.5 or row['actual_bias_score_diagnostics']['maximum_logical_ratio']>=.2:
            return None
        return row

    # Warm bias reuses its spectrum and already solved duration exactly.
    def at(x):
        fc,fb=map(float,origin+scale*x)
        row=at_bias(fc,fb)
        if row is None:
            return dict(raw_cz_fidelity=0.,postselected_cz_fidelity=0.,logical_survival=0.,
                        phase_error_rad=float('nan'),leakage_by_input=[1.]*4)
        return row

    refinement=refine_gate_controls(at,[0.,0.],bounds,constrain_phase=False,maxiter=6,max_evaluations=12)
    best=refinement['best'];audit=Gate(cell,AUDIT)
    final=correct_gate_phase(lambda fb:audit.at(best['fc_ghz'],fb,best['time_ns']),best)
    final.update(optimizer_message='High-basis phase correction only; coupler and time frozen')
    validation=gate_precision_audit(cell,final,evaluators=[cal,audit]);audits=validation['history']
    trace=audit.at_times(final['fc_ghz'],final['fb_ghz'],np.linspace(0,1.4*final['time_ns'],240))
    result=dict(best=final,search_best=best,search_seeds=[start],search_results=[refinement['best']],
        warm_start=warm,previous_five_body_gate=seed,
        search_optimizer=dict(success=refinement['optimizer_success'],message=refinement['stop_reason'],evaluations=refinement['evaluations']),
        basis=AUDIT,photons=3,audits=audits,numerical_precision_validation=validation,time_trace=trace,
        bounds=dict(fc=fcbounds,fb=None,time=tbounds),
        gate_A_bare_frequencies_ghz=[cell['request']['idle_bare_frequencies_ghz'][n] for n in ['A1','A2']],
        performance=dict(elapsed_seconds=time.perf_counter()-began,search_basis_diagonalizations=cal.calls,
                         audit_basis_diagonalizations=audit.calls,bias_cache=at_bias.cache_info()._asdict(),
                         duration_policy='One spectrum per duration grid; only subsequent frequency/phase corrections need new spectra'),
        scope='Static dressed ON benchmark; five-body warm start, cached duration search and finite local frequency refinement. No global optimum, ramp, quench, decoherence or independent coupling rescaling claim.')
    write(folder/'gate-search.json',dict(seeds=[start],results=[best],search_basis=SEARCH,
                                       warm_start=warm,bounds=result['bounds']))
    write(folder/'gate.json',result);print('FINAL_GATE',json.dumps(final),flush=True)
    return result

def final_gate(cell,folder,previous):
    """Compatibility entry point: use the same bounded warm-start procedure."""
    return gate(cell,folder,previous)

def gate_off(cell,folder):
    offdata=json.loads((folder/'off.json').read_text());require_off_definition(offdata.get('off_definition'),'seven');gate=json.loads((folder/'gate.json').read_text())['best'];base=frozen_model(cell['physical']);ej=base.ej.copy()
    for i,n in [(0,'A1'),(1,'A2')]:ej[i]=ej_for_frequency(cell['request']['idle_bare_frequencies_ghz'][n],base.ec[i],35)
    for i in [3,4]:ej[i]=ej_for_frequency(gate['fb_ghz'],base.ec[i],35)
    require_same_channel(base.electrostatics.get('channel_selection'),offdata.get('channel_selection'))
    require_same_channel(base.electrostatics.get('channel_selection'),gate.get('channel_selection'))
    bounds=offdata.get('off_bracket_ghz',cell.get('off_bracket_ghz',[7.00001,9.99999]))
    system=SevenReadout(replace(base,ej=ej),cell['physical'],offdata['off_ghz'],basis=AUDIT).solve();detail=system.matter_detail()
    out=dict(fc_ghz=offdata['off_ghz'],fb_gate_ghz=gate['fb_ghz'],channel_selection=base.electrostatics.get('channel_selection'),channel_source='0100',channel_target='0010',channel=off_exchange(detail),logical_residuals=detail['logical_exchange'],off_bracket_ghz=bounds,secular=secular_logical_shifts(detail),residual_channels=residuals(detail),basis=AUDIT)
    write(folder/'gate-rails-coupler-off.json',out)
    from coupler_readout_probe import probe
    write(folder/'on-coupler-readout.json',probe(cell,gate['fc_ghz'],gate['fb_ghz']))
    return out

def off_audit(cell,folder):
    from floating_coupler.precision import residual_metrics,verify_fixed_controls
    data=json.loads((folder/'off.json').read_text())
    require_off_definition(data.get('off_definition'),'seven')
    base=frozen_model(cell['physical'])
    # Keep seven-mode default verification modest. Exhaustion is uncertified,
    # not permission to allocate successively enormous dense matrices.
    bases=cell.get('off_audit_bases',[SEARCH,AUDIT])
    def evaluate(basis):
        system=SevenReadout(base,cell['physical'],data['off_ghz'],basis=basis).solve()
        detail=system.matter_detail()
        if not off_exchange(detail)['valid'] or not detail['logical_exchange']['valid']:
            raise ValueError('Unreliable subspace in fixed-control OFF audit')
        return residual_metrics(detail)
    result=verify_fixed_controls(evaluate,bases,'residual')
    result.update(off_ghz=data['off_ghz'],off_definition=off_target_definition('seven'))
    write(folder/'off-convergence.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--output',required=True);p.add_argument('--stage',choices=['off','A','B','readout','gate','final-gate','gate-off','diagnose-gate','off-audit'],required=True);p.add_argument('--previous-gate');p.add_argument('--readout-A');p.add_argument('--readout-B');p.add_argument('--workers',type=int,default=2);a=p.parse_args();cell=json.loads(Path(a.cell).read_text());folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True)
    request=cell if cell.get('channel_selection') else cell.get('request',{})
    if request.get('channel_selection') or request.get('gate_target',{}).get('channel_selection'):
        cell['physical']['channel_selection']=selection_from_request(request).to_dict()
    elif a.stage in ['gate','final-gate','diagnose-gate']:selection_from_request(request)
    if a.stage=='off':off(cell,folder)
    elif a.stage=='readout':
        from readout_common_sweep import run_sweeps
        if not a.readout_A or not a.readout_B:p.error('The readout stage requires --readout-A and --readout-B')
        run_sweeps(cell,{pair:json.loads(Path(getattr(a,'readout_'+pair)).read_text()) for pair in ['A','B']},folder,workers=a.workers)
    elif a.stage in ['A','B']:scan(cell,folder,a.stage)
    elif a.stage=='gate':gate(cell,folder,json.loads(Path(a.previous_gate).read_text()))
    elif a.stage=='final-gate':final_gate(cell,folder,json.loads(Path(a.previous_gate).read_text()))
    elif a.stage=='gate-off':gate_off(cell,folder)
    elif a.stage=='diagnose-gate':diagnose_gate(cell,folder)
    else:off_audit(cell,folder)
