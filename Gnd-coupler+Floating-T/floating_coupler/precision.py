"""User-approved engineering accuracy; numerical tolerances, not physics limits.

Residual couplings: 1 Hz absolute. Gate error: 0.01 percentage point = 1e-4
absolute infidelity, not 0.01% relative error. Compare the SAME controls before
any recalibration. Stop escalation on agreement; do not certify nonconvergence.
"""
import numpy as np

RESIDUAL_HZ = 1.
GATE_ERROR_ABS = 1e-4
GATE_PHASE_TOL = 1e-7
SEARCH_BASIS = dict(local_levels=4,pair_states=10,coupler_levels=4,ncut=25)
VERIFY_BASES = [dict(local_levels=l,pair_states=p,coupler_levels=l,ncut=n)
                for l,p,n in [(5,12,30),(6,15,35),(7,18,40),(8,20,45)]]


class _RefinementStop(Exception):
    pass


class GateRefinement:
    """Finite local refinement, stopping on physically meaningful metric changes.

    This is an optimizer stop, NOT a basis-convergence or fidelity certificate.
    No fidelity target is imposed here; raw, postselected and survival metrics
    must all plateau. Phase-pi feasibility is retained independently.
    """
    def __init__(self, evaluate, max_evaluations=48):
        self.evaluate=evaluate
        self.max_evaluations=max_evaluations
        self.cache={}
        self.best=None
        self.best_x=None
        self.window=[]
        self.reason='optimizer_complete'

    def at(self,x):
        key=tuple(map(float,np.atleast_1d(x)))
        if key not in self.cache:
            if len(self.cache)>=self.max_evaluations:
                self.reason='evaluation_budget'
                raise _RefinementStop()
            row=self.evaluate(np.array(key))
            self.cache[key]=row
            if (np.isfinite(row.get('phase_error_rad',np.nan)) and
                    abs(row['phase_error_rad'])<=GATE_PHASE_TOL and
                    all(np.isfinite(row.get(k,np.nan)) for k in
                        ('raw_cz_fidelity','postselected_cz_fidelity','logical_survival'))):
                if self.best is None or row['raw_cz_fidelity']>self.best['raw_cz_fidelity']:
                    self.best,self.best_x=row,np.array(key)
        return self.cache[key]

    def accepted(self,x):
        row=self.at(x)
        if not np.isfinite(row.get('phase_error_rad',np.nan)) or abs(row['phase_error_rad'])>GATE_PHASE_TOL:
            self.window.clear()
            return
        # Duplicate callbacks do not establish a plateau.
        key=tuple(map(float,np.atleast_1d(x)))
        if self.window and key==self.window[-1][0]:return
        self.window.append((key,row))
        self.window=self.window[-3:]
        if len(self.window)==3 and all(compare(a[1],b[1],'gate')['converged']
                                      for a in self.window for b in self.window):
            self.reason='metric_plateau_1e-4'
            raise _RefinementStop()

    def result(self,success=False):
        if self.best is None:
            raise ValueError('Local refinement found no phase-pi feasible endpoint')
        return dict(best=self.best,x=self.best_x.tolist(),
                    optimizer_success=bool(success),stop_reason=self.reason,
                    evaluations=len(self.cache),gate_error_accuracy_absolute=GATE_ERROR_ABS)


def refine_gate_controls(evaluate,start,bounds,maxiter=12,max_evaluations=48,constrain_phase=True):
    """SLSQP in caller-scaled controls; retain best phase-pi feasible point."""
    from scipy.optimize import minimize
    track=GateRefinement(evaluate,max_evaluations)
    success=False
    try:
        track.at(start)
        fit=minimize(lambda x:-100*track.at(x)['raw_cz_fidelity'],start,
                     method='SLSQP',bounds=bounds,
                     constraints=([dict(type='eq',fun=lambda x:track.at(x)['phase_error_rad'])]
                                  if constrain_phase else []),
                     callback=track.accepted,
                     options=dict(maxiter=maxiter,ftol=1e-6,eps=2e-4))
        track.at(fit.x)
        success=fit.success
        if not fit.success:track.reason='optimizer_budget_or_failure: '+str(fit.message)
    except _RefinementStop:
        pass
    return track.result(success)


def refine_gate_scalar(evaluate,bounds,initial,xatol,maxiter=12):
    """Refine an already sampled local peak, without sub-tolerance polishing."""
    from scipy.optimize import minimize_scalar
    track=GateRefinement(lambda x:evaluate(float(x[0])),max_evaluations=maxiter+4)
    success=False
    def objective(x):
        row=track.at([x])
        if 'raw_cz_fidelity' not in row:return 1.
        track.accepted([x])
        return -row['raw_cz_fidelity']
    try:
        track.at([initial])
        fit=minimize_scalar(objective,bounds=bounds,method='bounded',
                            options=dict(xatol=xatol,maxiter=maxiter))
        track.at([fit.x]);success=fit.success
    except _RefinementStop:
        pass
    return track.result(success)


def correct_gate_phase(evaluate,point,bounds=(None,None),max_evaluations=6):
    """One-dimensional phase correction at fixed coupler bias and duration.

    Deliberately does not restart a high-basis frequency/time optimization.
    A safeguarded local Newton correction preserves the selected nearby branch.
    """
    fb=float(point['fb_ghz'])
    row=evaluate(fb)
    calls=1
    while abs(row['phase_error_rad'])>GATE_PHASE_TOL and calls+2<=max_evaluations:
        step=1e-6
        if bounds[1] is not None and fb+step>bounds[1]:step=-step
        adjacent=evaluate(fb+step)
        slope=(adjacent['phase_error_rad']-row['phase_error_rad'])/step
        if not np.isfinite(slope) or abs(slope)<1e-8:
            raise ValueError('No reliable local phase slope; do not launch blind fine optimization')
        move=float(np.clip(-row['phase_error_rad']/slope,-.002,.002))
        fb+=move
        fb=float(np.clip(fb,-np.inf if bounds[0] is None else bounds[0],
                         np.inf if bounds[1] is None else bounds[1]))
        row=evaluate(fb);calls+=2
    if not np.isfinite(row['phase_error_rad']) or abs(row['phase_error_rad'])>GATE_PHASE_TOL:
        raise ValueError('Phase correction budget exhausted; endpoint is not phase-pi calibrated')
    return dict(row,phase_correction_evaluations=calls,
                phase_correction_scope='Only moved-rail frequency; coupler and duration fixed')


def residual_metrics(detail):
    if not detail['physical_exchange']['valid'] or not detail['logical_exchange']['valid']:
        raise ValueError('Invalid dressed assignments cannot certify numerical precision')
    h=np.asarray(detail['logical_exchange']['local_energy_hamiltonian_ghz'])
    from .on_gate import secular_logical_shifts
    return dict(physical_exchange_hz=detail['physical_exchange']['g_mhz']*1e6,
                logical_exchange_hz=float(h[1,2]*1e9),
                logical_double_flip_hz=float(h[0,3]*1e9),
                secular_logical_zz_hz=secular_logical_shifts(detail)['secular_zz_hz'])


def compare(previous,current,kind):
    """All required metrics must be present and finite; never accept NaN."""
    if kind=='residual':
        keys=('physical_exchange_hz','logical_exchange_hz','logical_double_flip_hz')
        tolerance=RESIDUAL_HZ
    elif kind=='gate':
        keys=('raw_cz_fidelity','postselected_cz_fidelity','logical_survival')
        tolerance=GATE_ERROR_ABS
    else:raise ValueError('Precision kind must be residual or gate')
    changes={k:abs(float(current[k])-float(previous[k])) for k in keys}
    if kind=='residual' and ('secular_logical_zz_hz' in previous or 'secular_logical_zz_hz' in current):
        changes['secular_logical_zz_hz']=abs(current['secular_logical_zz_hz']-previous['secular_logical_zz_hz'])
    if kind=='gate' and ('leakage_by_input' in previous or 'leakage_by_input' in current):
        changes['worst_input_leakage_change']=float(np.max(abs(
            np.asarray(current['leakage_by_input'])-np.asarray(previous['leakage_by_input']))))
    return dict(converged=bool(all(np.isfinite(v) and v<=tolerance for v in changes.values())),
                tolerance=tolerance,changes=changes,kind=kind,
                scope='Same hardware and controls; no retuning between bases')


def verify_fixed_controls(evaluate,bases,kind):
    """Evaluate successively larger modest bases, stopping on first agreement.

    `evaluate(basis)` must hold ALL controls fixed, return the metrics in compare,
    and raise for bad assignments or failed physical constraints. Basis names are
    recorded with the actual check; exhaustion is a failure, not permission to
    silently escalate without bound or claim convergence.
    """
    history=[]
    for basis in bases:
        if history and basis==history[-1]['basis']:continue
        metrics=evaluate(basis)
        row=dict(basis=basis,metrics=metrics)
        if history:
            row['comparison']=compare(history[-1]['metrics'],metrics,kind)
        history.append(row)
        if len(history)>1 and row['comparison']['converged']:
            return dict(converged=True,history=history,final_basis=basis,metrics=metrics)
    return dict(converged=False,history=history,reason='Accuracy not reached within requested basis ladder')
