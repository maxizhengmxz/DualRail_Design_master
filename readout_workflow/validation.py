"""Adaptive readout validation; nominal constraints are unchanged.

Routine candidates/probes use 6x6x5 states. Near thresholds, increase the
nominal basis; a deterministic 5% sample checks the full basis and full probe
grid. Guard widths are engineering tolerances, not rigorous error bounds.
Use force_full=True for final selected designs and explicit convergence work.
"""
from dataclasses import asdict
from hashlib import sha256
import json
from . import calculation as study
from core.transmon import Basis, transmon_spectrum
from core.electrostatics import exchange_j01_ghz

FAST = Basis(15, 6, 5, 20)
INTERMEDIATE = Basis(20, 8, 6, 40)
AUDIT = Basis(25, 12, 10, 60)  # Retain explicit final-design/full-audit option.
POLICY = 'adaptive-readout-v1'
SAMPLE_MODULUS = 20


def _sample(point):
    identity=dict(device=point['device'],ej1=point['ej1_ghz'],ej2=point['ej2_ghz'])
    return int(sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16],16)%SAMPLE_MODULUS == 0


def _near_boundary(p, chi_error_ghz):
    """Conservative escalation triggers, not extra acceptance constraints."""
    c=study.CONDITIONS
    denominator=max(abs((p['chi_p_ghz']+p['chi_m_ghz'])/2),1e-12)
    ratio_guard=(2+c.mismatch_max)*chi_error_ghz/denominator
    reasons=[]
    if abs(p['ratio']-c.mismatch_max)<=ratio_guard:reasons.append('chi_matching')
    if abs(study.absolute_mean_chi_mhz(p)-c.abs_mean_chi_min_mhz)<=1000*chi_error_ghz:reasons.append('readout_strength')
    for key,threshold in [('hybridization',c.hybridization_min),('lambda_max',c.dispersive_max),('min_assignment_overlap',c.reference_overlap_min)]:
        if abs(p[key]-threshold)<=1e-5:reasons.append(key)
    for key,bounds in [('splitting_ghz',c.logical_frequency_ghz),('operating_frequency_ghz',c.operation_frequency_ghz)]:
        if min(abs(p[key]-v) for v in bounds)<=chi_error_ghz:reasons.append(key)
    if c.operation_below_readout and abs(p['dressed_r_ghz']-p['operating_frequency_ghz'])<=chi_error_ghz:reasons.append('operation_below_readout')
    return reasons


def _nominal(point,basis):
    model=study.ReadoutModel(study.Device(**point['device']),basis)
    return study.annotate(model,point['ej1_ghz'],point['ej2_ghz']),model


def audit_point(point,sign=True,*,force_full=False,representative=None):
    """Fixed-input validation; no retuning and no readout optimization.

    Main reported quantities use the selected nominal basis. A sign certificate
    separately records the probe basis, so basis identity is never misreported.
    Passing nominal constraints is required before expensive sign probes.
    """
    sampled=_sample(point) if representative is None else bool(representative)
    sampled=sampled and not force_full
    assessments=[]
    if force_full:
        result,model=_nominal(point,AUDIT)
    else:
        result,model=_nominal(point,FAST)
        near=_near_boundary(result,1e-6)  # 1 kHz per conditional shift.
        assessments.append(dict(basis=asdict(FAST),near=near))
        if near or sampled:
            fast=result
            result,model=_nominal(point,INTERMEDIATE)
            error=max(abs(result[k]-fast[k]) for k in ('chi_m_ghz','chi_p_ghz'))
            near=_near_boundary(result,1e-7)  # 0.1 kHz medium-stage guard.
            assessments.append(dict(basis=asdict(INTERMEDIATE),near=near,chi_change_ghz=error))
            if near or sampled or error>1e-6:
                result,model=_nominal(point,AUDIT)
                assessments.append(dict(basis=asdict(AUDIT),reason='representative' if sampled else 'threshold_or_convergence'))
    result['x']=study.x_from_point(result)
    result['g12_over_2pi_mhz']=result['J01_12_mhz']
    result['validation']=dict(policy=POLICY,force_full=force_full,representative=sampled,nominal_basis=asdict(model.basis),assessments=assessments)
    # Same acceptance rules; skip probes only when nominal acceptance failed.
    if sign and not study.failures(result):
        probe_basis=AUDIT if (force_full or sampled) else FAST
        result['chi_sign_probe_basis']=asdict(probe_basis)
        result=study.check_chi_sign_stability(result,early_exit=True)
        cert=result['chi_sign_stability']
        ambiguous=any(min(abs(p['chi_m_ghz']),abs(p['chi_p_ghz']))<=1e-6 or abs(p['min_assignment_overlap']-study.CONDITIONS.reference_overlap_min)<=1e-5 for p in cert['probes'])
        if probe_basis!=AUDIT and ambiguous:
            # Probe signs/assignments too close to numerical resolution: retain
            # the original full-basis neighborhood check, not a coarse verdict.
            result['chi_sign_probe_basis']=asdict(AUDIT)
            result.pop('chi_sign_stability',None)
            result=study.check_chi_sign_stability(result,early_exit=True)
            result['validation']['probe_escalation']='near_zero_chi_or_overlap_threshold'
        result['validation']['probe_basis']=result['chi_sign_probe_basis']
    return result


def metrics(point):
    """Keep signed sum/difference and distinguish their magnitudes."""
    minus,plus=point['chi_m_ghz'],point['chi_p_ghz']
    return dict(chi_minus_mhz=1000*minus,chi_plus_mhz=1000*plus,delta_chi_khz=1e6*(plus-minus),chi_sum_mhz=1000*(plus+minus),absolute_chi_sum_mhz=1000*abs(plus+minus),absolute_mean_chi_mhz=study.absolute_mean_chi_mhz(point),ratio=point['ratio'])
