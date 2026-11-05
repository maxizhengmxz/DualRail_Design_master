"""Opt-in one-at-a-time electrostatic compensation and static CZ comparison.

Each case starts from the same saved seven-mode circuit. Fit physical capacitors
with equal rail pad grounds and immutable loaded transmon ECs. Retain the saved
OFF/idle EJs and ON controls for comparisons, with a separate B phase correction.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.constants import elementary_charge, h, hbar
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

WF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WF), str(WF.parent)]
from floating_coupler.mixed_electrostatics import mixed_circuit
from floating_coupler.model import SCALE
from floating_coupler.precision import correct_gate_phase
from integrated_readout_range import frozen_model
from seven_readout import SevenReadout
from final_seven_validation import Gate

PRODUCTION = dict(local_levels=6, pair_states=12, coupler_levels=5, ncut=35)
AUDIT = dict(local_levels=7, pair_states=15, coupler_levels=6, ncut=40)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def rebuild(physical):
    """Recompute resonator frequencies and charges from the changed geometry."""
    ph = deepcopy(physical)
    circuit = mixed_circuit(ph['pad_pairs'], ph['ground_capacitances_ff'],
                            ph['pad_mutual_capacitances_ff'], ph['physical_links'],
                            resonator_caps_ff=ph['resonator_self_capacitances_ff'])
    k = circuit['K']
    ph['K_per_ff'] = k.tolist()
    ph['loaded_ec_ghz'] = dict(zip(ph['pad_pairs'], circuit['loaded_ec_ghz'].tolist()))
    parameters = {}
    for i, name in enumerate(ph['resonator_self_capacitances_ff'], start=5):
        capacitance = 1e-15 / k[i, i]
        omega = 1 / np.sqrt(ph['resonator_inductances_nh'][name] * 1e-9 * capacitance)
        charge = np.sqrt(hbar * capacitance * omega / 2)
        parameters[name] = dict(fr_ghz=omega / (2 * np.pi * 1e9),
                                effective_capacitance_ff=capacitance * 1e15,
                                q_zpf_coulomb=charge,
                                gamma_ghz_per_n=(2 * elementary_charge * charge /
                                                (h * 1e9) * k[:5, i] * 1e15).tolist())
    ph['resonator_parameters'] = parameters
    return ph, circuit


def idle_diagnostics(system):
    detail = system.matter_detail()
    effective = np.array(detail['physical_exchange']['effective_hamiltonian_ghz'])
    _, ua = eigh(effective[:2, :2])
    _, ub = eigh(effective[2:, 2:])
    rotation = np.zeros((4, 4))
    rotation[:2, :2] = ua
    rotation[2:, 2:] = ub
    rail = rotation.T @ effective @ rotation
    _, vectors = eigh(rail)
    _, assigned = linear_sum_assignment(-abs(vectors)**2)
    populations = [float(sum(abs(vectors[2:, assigned[i]])**2)) if i < 2
                   else float(sum(abs(vectors[:2, assigned[i]])**2)) for i in range(4)]
    logical = detail['logical_exchange']
    matrix = np.asarray(logical['local_energy_hamiltonian_ghz'])
    channels = []
    for left, right, i, j in [('01L', '10L', 1, 2), ('00L', '11L', 0, 3)]:
        coupling = float(matrix[i, j] * 1e9)
        detuning = float((matrix[i, i] - matrix[j, j]) * 1e9)
        channels.append(dict(left=left, right=right, g_hz=coupling,
                             detuning_mhz=detuning / 1e6,
                             amplitude_ratio=abs(coupling / detuning),
                             eigenstate_admixture=(coupling / detuning)**2))
    return dict(physical_exchange=detail['physical_exchange'], logical_exchange=logical,
                logical_channels=channels, maximum_wrong_rail_population=max(populations),
                minimum_subspace_overlap=detail['minimum_subspace_overlap'])


def quantum_compare(cell, physical, point):
    """Compare static gate controls, then locally correct B's conditional phase.

    The returned idle block uses the saved nominal bias before OFF retuning.
    Final OFF residuals/participation are computed separately by the validator.
    Perturbative admixture fields in idle_diagnostics are diagnostic estimates.
    Actual physical-state probabilities come from participation_detail.
    """
    start = time.perf_counter()
    updated = deepcopy(cell)
    updated['physical'], circuit = rebuild(physical)
    model = frozen_model(updated['physical'])
    system = SevenReadout(model, updated['physical'], cell['off_ghz'],
                          basis=PRODUCTION).solve()
    idle = idle_diagnostics(system)
    calibration = Gate(updated, PRODUCTION)
    fixed = calibration.at(point['fc_ghz'], point['fb_ghz'], point['time_ns'])
    corrected = correct_gate_phase(
        lambda frequency: calibration.at(point['fc_ghz'], frequency, point['time_ns']),
        fixed, (point['fb_ghz'] - .012, point['fb_ghz'] + .012), max_evaluations=10)
    audit = Gate(updated, AUDIT).at(point['fc_ghz'], point['fb_ghz'], point['time_ns'])
    return dict(production_basis=PRODUCTION, audit_basis=AUDIT, idle=idle,
                fixed_gate=fixed, phase_recalibrated_gate=corrected,
                large_basis_fixed_gate=audit,
                charge_coefficients_hz=(8 * SCALE * circuit['K'][:5, :5] * 1e9).tolist(),
                seconds=time.perf_counter() - start)


