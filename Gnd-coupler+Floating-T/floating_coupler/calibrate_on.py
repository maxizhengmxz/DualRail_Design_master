"""Phase-pi recalibration at fixed geometry and fixed on coupler frequency."""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eigh

from .on_gate import dressed_on_spectrum, dressed_on_metrics, dressed_on_metrics_many
from .on_gate import energy_axis_matter_hamiltonian
from .channel_policy import require_selection,LOGICAL_ORDER
from .precision import refine_gate_scalar,GATE_PHASE_TOL,correct_gate_phase


def cached_duration_search(evaluate_times,time_bounds,seed_time,points=33):
    """Find phase-pi returns using a FIXED spectrum; never refit frequency here.

    evaluate_times must close over one already diagonalized bias. Root finding
    in time is then spectral arithmetic only. Phase-wrap jumps are not roots.
    best_return is only a proposal and may require a frequency/phase correction;
    best is already phase-pi feasible at this fixed bias.
    This is a bounded static-duration search, not a pulse/ramp calculation.
    """
    if not 0<time_bounds[0]<time_bounds[1]:raise ValueError('Invalid duration range')
    times=np.unique(np.r_[np.linspace(*time_bounds,points),
                           np.clip(seed_time,*time_bounds)])
    rows=evaluate_times(times)
    def reliable(row):
        score=row.get('actual_bias_score_diagnostics')
        return (np.isfinite(row['phase_error_rad']) and
                min(row['return_by_input'])>.5 and
                row.get('minimum_dressed_reference_overlap',1.)>=.8 and
                (score is None or score['maximum_logical_ratio']<.2))
    def usable(row):
        return reliable(row) and abs(row['phase_error_rad'])<=GATE_PHASE_TOL
    roots=[row for row in rows if usable(row)]
    for left,right in zip(rows[:-1],rows[1:]):
        a,b=left['phase_error_rad'],right['phase_error_rad']
        if not np.isfinite(a+b) or a*b>=0 or abs(a-b)>np.pi:continue
        root=brentq(lambda t:evaluate_times([t])[0]['phase_error_rad'],
                    left['time_ns'],right['time_ns'],xtol=1e-7)
        row=evaluate_times([root])[0]
        if usable(row) and not any(abs(root-r['time_ns'])<1e-6 for r in roots):
            roots.append(row)
    returns=[row for row in rows if reliable(row)]
    return dict(best=max(roots,key=lambda r:r['raw_cz_fidelity']) if roots else None,
                best_return=max(returns,key=lambda r:r['raw_cz_fidelity']) if returns else None,
                phase_pi_candidates=roots,grid=rows,
                scope='Fixed coupler and rail frequencies; cached-spectrum duration search')


def warm_start_gate(cal,seed,fc_bounds,time_bounds,max_biases=9):
    """Start at the five-body solution; expand locally only if no phase-pi return.

    The +/-200 MHz bound is on initial starts, not a new physical tuning limit.
    Coupler bias and duration both start from the preceding solution rather than
    resetting to a generic 100 ns scan at three arbitrary coupler frequencies.
    """
    require_selection(seed['channel_selection'])
    from .channel_policy import require_same_channel
    require_same_channel(cal.selection,seed['channel_selection'])
    fc=float(np.clip(seed['fc_ghz'],*fc_bounds))
    offsets=[0.,.002,-.002,.005,-.005,.012,-.012,.03,-.03,.08,-.08,.2,-.2]
    attempts=[]
    for offset in offsets[:max_biases]:
        fb=float(seed['fb_ghz']+offset)
        if fb<=0:continue
        search=cached_duration_search(cal.duration_evaluator(fc,fb),time_bounds,seed['time_ns'])
        attempts.append(dict(fc_ghz=fc,fb_ghz=fb,phase_pi_found=search['best'] is not None))
        if search['best'] is not None:
            return dict(best=search['best'],attempts=attempts,
                        frequency_policy='Five-body seed first; local fallback starts within 200 MHz',
                        previous_five_body_gate=seed)
        if offset==0.:
            # Integration can offset the phase without giving any pi root in
            # the duration range at the *exact* old bias. Transfer the five-body
            # solution locally in frequency before considering any wider starts.
            try:
                corrected=correct_gate_phase(
                    lambda f:cal.at(fc,f,seed['time_ns']),seed,
                    (max(1e-6,seed['fb_ghz']-.2),seed['fb_ghz']+.2),max_evaluations=10)
                transferred=cached_duration_search(
                    cal.duration_evaluator(fc,corrected['fb_ghz']),time_bounds,seed['time_ns'])
                attempts.append(dict(fc_ghz=fc,fb_ghz=corrected['fb_ghz'],
                    phase_pi_found=transferred['best'] is not None,
                    method='local five-body phase transfer',
                    phase_evaluations=corrected['phase_correction_evaluations']))
                if transferred['best'] is not None:
                    return dict(best=transferred['best'],attempts=attempts,
                        frequency_policy='Five-body seed with local phase transfer; starts within 200 MHz',
                        previous_five_body_gate=seed)
            except ValueError:
                attempts.append(dict(fc_ghz=fc,fb_ghz=fb,phase_pi_found=False,
                                     method='local phase transfer exhausted finite budget'))
    raise ValueError('No reliable phase-pi return in finite five-body warm-start neighborhood')


class PhaseCalibration:
    def __init__(self, model, fc, basis, fb_bounds, phase_samples=9, minimum_midpoint_b20=None,selection=None):
        self.model, self.fc, self.basis = model, fc, basis
        self.fb_bounds = fb_bounds
        self.phase_samples = phase_samples
        self.minimum_midpoint_b20 = minimum_midpoint_b20
        self.selection=require_selection(selection or (model.electrostatics.get('channel_selection') if model is not None else None))
        if self.selection.moved_pair!='B':
            raise ValueError('This calibration kernel moves B only; an A-moving request needs the corresponding tuning kernel')

    @lru_cache(maxsize=128)
    def spectrum(self, fb):
        energy, coefficients, detail = dressed_on_spectrum(self.model, fb, self.fc, self.basis)
        if self.minimum_midpoint_b20 is None:
            return energy, coefficients, {"overlap":detail["minimum_subspace_overlap"],
                "score_diagnostics":detail.get('selected_channel_score_diagnostics')}
        return energy, coefficients, {
            "overlap":detail['minimum_subspace_overlap'],
            "auxiliary_coefficients":np.asarray(detail['on_auxiliary_coefficients'][self.selection.auxiliary]),
            "score_diagnostics":detail.get('selected_channel_score_diagnostics')}

    def at(self, fb, time_ns):
        energy, coeff, metadata = self.spectrum(float(fb))
        return self._decorate(dressed_on_metrics(energy,coeff,time_ns),fb,time_ns,energy,coeff,metadata)

    def at_times(self,fb,times_ns):
        """Batch durations at fixed hardware/bias without another eigensolve."""
        energy,coeff,metadata=self.spectrum(float(fb))
        rows=dressed_on_metrics_many(energy,coeff,times_ns)
        return [self._decorate(row,fb,row['time_ns'],energy,coeff,metadata) for row in rows]

    def duration_evaluator(self,fb):
        """Closure reusing one spectrum across grids and phase-root iterations."""
        energy,coeff,metadata=self.spectrum(float(fb))
        def evaluate(times):
            rows=dressed_on_metrics_many(energy,coeff,times)
            return [self._decorate(row,fb,row['time_ns'],energy,coeff,metadata) for row in rows]
        return evaluate

    def _decorate(self,metrics,fb,time_ns,energy,coeff,metadata):
        overlap = metadata["overlap"] if isinstance(metadata, dict) else metadata
        row = dict(metrics, fb_ghz=float(fb),
                   fc_ghz=float(self.fc), minimum_dressed_reference_overlap=overlap)
        if isinstance(metadata, dict) and metadata.get('score_diagnostics') is not None:
            row['actual_bias_score_diagnostics']=metadata['score_diagnostics']
        if isinstance(metadata, dict) and 'auxiliary_coefficients' in metadata:
            source=LOGICAL_ORDER.index(self.selection.source_logical)
            midpoint = metadata["auxiliary_coefficients"] @ (np.exp(-1j * np.pi * energy * time_ns) * coeff[:, source])
            row["midpoint_selected_auxiliary_from_selected_source"] = float(abs(midpoint)**2)
            if self.selection.source_logical=='01' and self.selection.auxiliary=='B20':
                row["midpoint_b20_from_logical01"]=float(abs(midpoint)**2)
        row['channel_selection']=self.selection.to_dict()
        return row

    def phase_root(self, time_ns):
        """Solve a signed near-pi error; never accept a +/-pi branch jump."""
        bounds = np.linspace(*self.fb_bounds, self.phase_samples)
        rows = [self.at(fb, time_ns) for fb in bounds]
        solutions = []
        for left, right in zip(rows[:-1], rows[1:], strict=True):
            a, b = left["phase_error_rad"], right["phase_error_rad"]
            if a * b > 0 or abs(a - b) > np.pi:
                continue
            fb = brentq(lambda frequency: self.at(frequency, time_ns)["phase_error_rad"],
                         left["fb_ghz"], right["fb_ghz"], xtol=2e-12)
            row = self.at(fb, time_ns)
            if abs(row["phase_error_rad"]) < 1e-7 and min(row["return_by_input"]) > 0.5:
                if self.minimum_midpoint_b20 is None or row["midpoint_selected_auxiliary_from_selected_source"] >= self.minimum_midpoint_b20:
                    solutions.append(row)
        if not solutions:
            raise ValueError("No reliable phase-pi root in the B-frequency bracket")
        return max(solutions, key=lambda row: row["raw_cz_fidelity"])

    def optimize_time(self, time_bounds, points=21):
        """Maximize raw fidelity along the phase-pi constraint, locally in time."""
        grid = [self.phase_root(float(t)) for t in np.linspace(*time_bounds, points)]
        best_index = int(np.argmax([row["raw_cz_fidelity"] for row in grid]))
        left = grid[max(0, best_index - 1)]["time_ns"]
        right = grid[min(points - 1, best_index + 1)]["time_ns"]
        refinement = refine_gate_scalar(self.phase_root,(left,right),
                                        grid[best_index]['time_ns'],xatol=.02)
        endpoint = refinement['best']
        return {"grid": grid, "best": max(grid + [endpoint], key=lambda row: row["raw_cz_fidelity"]),
                "optimizer_success": refinement['optimizer_success'],
                "refinement":refinement}


def optimize_coupler_fixed_time(model, time_ns, basis, fb_bounds, fc_bounds,
                                points=21, local_starts=3, phase_samples=9, minimum_midpoint_b20=None):
    """Search C at fixed duration, with B root-solved to phase pi throughout."""
    @lru_cache(maxsize=256)
    def at(fc):
        cal = PhaseCalibration(model, float(fc), basis, fb_bounds, phase_samples, minimum_midpoint_b20)
        try:
            return cal.phase_root(time_ns)
        except ValueError as error:
            return {"fc_ghz": float(fc), "valid": False, "error": str(error)}

    frequencies = np.linspace(*fc_bounds, points)
    grid = [at(float(fc)) for fc in frequencies]
    valid = [i for i, row in enumerate(grid) if "raw_cz_fidelity" in row]
    if not valid:
        raise ValueError("No reliable phase-pi points in the coupler search interval")
    starts = sorted(valid, key=lambda i: grid[i]["raw_cz_fidelity"], reverse=True)[:local_starts]
    candidates = [grid[i] for i in valid]
    for i in starts:
        left, right = frequencies[max(0, i - 1)], frequencies[min(points - 1, i + 1)]
        refinement=refine_gate_scalar(at,(left,right),frequencies[i],xatol=1e-5)
        row = refinement['best']
        if "raw_cz_fidelity" in row:
            candidates.append(dict(row, optimizer_success=refinement['optimizer_success'],
                                   refinement_stop_reason=refinement['stop_reason']))
    return {"basis": basis, "fc_bounds_ghz": fc_bounds, "fb_bounds_ghz": fb_bounds,
            "time_ns": time_ns, "grid": grid,
            "best": max(candidates, key=lambda row: row["raw_cz_fidelity"])}
