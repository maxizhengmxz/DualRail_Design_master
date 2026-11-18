"""Shared controls for optional static-CZ diagnostics on a saved physical circuit."""
import json
from pathlib import Path

from final_seven_validation import Gate

PRODUCTION=dict(local_levels=6,pair_states=12,coupler_levels=5,ncut=35)
AUDIT=dict(local_levels=7,pair_states=15,coupler_levels=6,ncut=40)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def gate_point(cell,record=None):
    """Read calibrated controls explicitly, without inventing a 100 ns gate."""
    if record is not None:
        for key in ('best_audit_phase_corrected','selected_static_gate','best'):
            if key in record:return record[key]
        if all(k in record for k in ('fc_ghz','fb_ghz','time_ns')):return record
        raise ValueError('Supply calibrated gate controls, not a nominal capacitor record')
    point=cell['operating_points']['static_gate_on']
    b1,b2=point['bare_f01_ghz']['B1'],point['bare_f01_ghz']['B2']
    if abs(b1-b2)>1e-10:raise ValueError('Reference gate has differential B bias; supply an explicit calibrated point')
    return dict(fc_ghz=point['bare_f01_ghz']['C'],fb_ghz=(b1+b2)/2,time_ns=point['duration_ns'])


def evaluator(cell,basis,b_detuning_ghz=0.):
    return Gate(cell,basis,b_detuning_ghz=b_detuning_ghz)
