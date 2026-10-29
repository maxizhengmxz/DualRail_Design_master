"""Compensate integrated idle dressing with individual rail EJs, then re-null C.

All geometry, loaded ECs and charge coefficients are frozen. Frequencies in GHz
are bare EJ coordinates; targets are mean dressed logical transitions. Existing
readout H/chi/overlap/dispersivity thresholds certify near-resonant operation.
This does not redefine OFF or impose equality of effective diagonal terms.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares, brentq, minimize

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from integrated_readout_range import frozen_model, constraint_values
from seven_readout import SevenReadout
from floating_coupler.design import ej_for_frequency
from floating_coupler.model import FULL, local_spectrum
from floating_coupler.channel_policy import off_target_definition

SEARCH = dict(local_levels=6, pair_states=10, coupler_levels=4, ncut=25)
AUDIT = dict(local_levels=6, pair_states=12, coupler_levels=5, ncut=35)


def calibrate(cell, cases, output, cycles=3):
    began = time.perf_counter()
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    original = deepcopy(cell)
    cell = deepcopy(cell)
    base = frozen_model(cell['physical'])
    fc = cell['off_ghz']
    frequencies = np.array([cell['physical']['bare_frequencies_ghz'][n] for n in FULL])
    history = []

    def model_at(fs):
        return replace(base, ej=np.array([ej_for_frequency(float(f), float(e), 35)
                                         for f, e in zip(fs, base.ec)]))

    for cycle in range(cycles):
        basis = SEARCH if cycle == 0 else AUDIT
        pair_results = {}
        for pair, idx in [('A', [0, 1]), ('B', [3, 4])]:
            target = cases[pair]['bare_frequency_guide_ghz']
            count = 0

            @lru_cache(maxsize=64)
            def at(key):
                nonlocal count
                fs = frequencies.copy()
                fs[idx] = key
                model = model_at(fs)
                system = SevenReadout(model, cell['physical'], fc, basis=basis).solve()
                rows = system.readout(pair)
                count += 1
                return rows, constraint_values(rows, model, pair, cases[pair])

            def residual(fs):
                rows, _ = at(tuple(map(float, fs)))
                mean = np.mean([r['operating_frequency_ghz'] for r in rows])
                chi = np.mean([(r['chi_p_ghz']-r['chi_m_ghz']) /
                               max(abs((r['chi_p_ghz']+r['chi_m_ghz'])/2), 1e-9)
                               for r in rows])
                return [(mean-target)*100, chi]

            lower, upper = [], []
            for i in idx:
                lower.append(float(local_spectrum(45.00001*base.ec[i], base.ec[i], base.basis)[0][1]))
                upper.append(float(local_spectrum(24.99999, base.ec[i], base.basis)[0][1]))
            start = np.clip(frequencies[idx], np.array(lower)+1e-6, np.array(upper)-1e-6)
            initial_rows, initial_constraints = at(tuple(map(float, start)))
            if (max(r['ratio'] for r in initial_rows) < .01 and
                    abs(np.mean([r['operating_frequency_ghz'] for r in initial_rows])-target) < .001):
                fit = SimpleNamespace(x=start, success=True)
            else:
                fit = least_squares(residual, start, bounds=(lower, upper),
                                    diff_step=2e-6, max_nfev=12, xtol=1e-8,
                                    ftol=1e-6, gtol=1e-6, x_scale=.02)
            rows, constraints = at(tuple(map(float, fit.x)))
            if min(constraints) < 0:
                # Exact mean/chi equality is only a calibration seed. Existing
                # readout feasibility takes precedence; choose the closest
                # feasible dressed operating mean and report its displacement.
                class FeasibleIdle(Exception):
                    def __init__(self, fs): self.fs = np.array(fs)
                def stop_when_feasible(fs):
                    if min(at(tuple(map(float, fs)))[1]) >= .001:
                        raise FeasibleIdle(fs)
                try:
                    fit = minimize(lambda fs: residual(fs)[0]**2+.01*residual(fs)[1]**2,
                                   fit.x, method='SLSQP', bounds=list(zip(lower, upper)),
                                   constraints=[dict(type='ineq', fun=lambda fs:
                                       at(tuple(map(float, fs)))[1]-.002)],
                                   callback=stop_when_feasible,
                                   options=dict(maxiter=15, ftol=1e-6, eps=2e-6))
                except FeasibleIdle as stop:
                    fit = SimpleNamespace(x=stop.fs, success=True)
                rows, constraints = at(tuple(map(float, fit.x)))
            frequencies[idx] = fit.x
            pair_results[pair] = dict(bare_frequencies_ghz=fit.x.tolist(),
                                     target_dressed_mean_ghz=target, rows=rows,
                                     minimum_constraint_slack=float(min(constraints)),
                                     optimizer_success=bool(fit.success), evaluations=count)
            print('IDLE', cycle, pair, json.dumps(pair_results[pair]), flush=True)

        model = model_at(frequencies)
        def exchange(f):
            return SevenReadout(model, cell['physical'], float(f), basis=basis,
                                off_only=True).solve().physical_off_detail()['physical_exchange']['g_mhz']
        request = cell.get('request', {})
        lower = max(request.get('on_lower_exclusive_ghz', 7.)+1e-5, fc-.15)
        upper = min(request.get('off_upper_exclusive_ghz', 10.)-1e-5, fc+.15)
        fc = brentq(exchange, lower, upper, xtol=1e-9)
        system = SevenReadout(model, cell['physical'], fc, basis=basis).solve()
        checked = {pair: dict(rows=system.readout(pair),
                             minimum_constraint_slack=float(min(constraint_values(
                                 system.readout(pair), model, pair, cases[pair]))))
                   for pair in ['A', 'B']}
        history.append(dict(cycle=cycle, basis=basis, before_off_retune=pair_results,
                            off_ghz=fc, joint_readout=checked,
                            physical_exchange=system.physical_off_detail()['physical_exchange']))
        (folder/'progress.json').write_text(json.dumps(history, indent=2))
        print('JOINT', cycle, fc, {p: checked[p]['minimum_constraint_slack'] for p in checked}, flush=True)
        if cycle > 0 and all(v['minimum_constraint_slack'] >= 0 for v in checked.values()):
            break

    for n, f in zip(FULL, frequencies):
        cell['physical']['bare_frequencies_ghz'][n] = float(f)
    cell['physical']['bare_frequencies_ghz']['C'] = fc
    cell['off_ghz'] = fc
    cell['off_ej_ghz'] = ej_for_frequency(fc, base.ec[2], 35)
    cell['off_definition'] = off_target_definition('seven')
    # Downstream static gates must keep the newly calibrated stationary rail,
    # rather than silently restoring the original pre-dressing bare settings.
    cell.setdefault('request', {})['idle_bare_frequencies_ghz'] = deepcopy(
        cell['physical']['bare_frequencies_ghz'])
    cell['idle_calibration'] = dict(policy='individual_EJ_dressed_readout_idle_v1',
                                    geometry_and_loaded_EC_frozen=True,
                                    original_bare_frequencies_ghz=original['physical']['bare_frequencies_ghz'])
    result = dict(original_idle_frequencies_ghz=original['physical']['bare_frequencies_ghz'],
                  calibrated_cell=cell, history=history,
                  nominal_readout_passed=all(v['minimum_constraint_slack'] >= 0 for v in checked.values()),
                  basis=basis, seconds=time.perf_counter()-began,
                  scope='Fixed capacitors, loaded ECs and resonator L; individual rail EJs and C OFF EJ only. Sign-neighborhood certification and residual convergence are separate checks.')
    (folder/'result.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--readout-A', required=True)
    parser.add_argument('--readout-B', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cases = {p: json.loads(Path(getattr(args, 'readout_'+p)).read_text())['case'] for p in ['A', 'B']}
    calibrate(json.loads(Path(args.cell).read_text()), cases, args.output)
