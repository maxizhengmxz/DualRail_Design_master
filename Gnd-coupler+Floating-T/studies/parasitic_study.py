"""Explicit preparation stages for the optional single-parasitic study.

No calculation runs on import. Fit, validate, fixed-control audit and report
are separate stages so electrostatic candidates cannot inherit quantum claims.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

WF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WF), str(WF.parent)]

GROUP_ONE = ('A21-B11', 'A11-r1', 'A22-r1', 'B22-r2', 'B11-r2',
             'B12-A22', 'A11-A22', 'B22-B11')
GROUP_TWO = ('A11-C1', 'A12-C1', 'B21-C1', 'B22-C1')
INTENTIONAL_PAIRS = (('A1','A2'), ('A2','C'), ('C','B1'), ('A2','B1'),
                     ('B1','B2'), ('A1','r1'), ('A2','r1'),
                     ('B1','r2'), ('B2','r2'))


def gate_controls(record):
    """Read either a saved gate result or the parameters-only named design."""
    if 'operating_points' in record:
        point = record['operating_points']['static_gate_on']
        frequencies = point['bare_f01_ghz']
        return dict(fc_ghz=frequencies['C'], fb_ghz=frequencies['B1'],
                    time_ns=point['duration_ns'])
    return record['best']


def fit_candidates(source, output, amounts, tolerance):
    """Fit independent perturbations, retaining failed cases without promotion."""
    from studies.parasitic_absorption import rebuild, write
    from studies.bounded_parasitic_compensation import compensate_bounded
    cell = json.loads(source.read_text())
    physical, circuit = rebuild(cell['physical'])
    intended = {frozenset(pair) for pair in INTENTIONAL_PAIRS}
    excluded = [pair for pair in combinations(circuit['modal_order'], 2)
                if frozenset(pair) not in intended]
    result = dict(source=str(source.resolve()), match_rtol=tolerance, cases=[])
    for amount in amounts:
        for group, ports in ((1, GROUP_ONE), (2, GROUP_TWO)):
            for port in ports:
                row = dict(group=group, ports=port, amount_ff=amount)
                try:
                    row['fit'] = compensate_bounded(
                        physical, port, amount, max_fraction=.95,
                        match_rtol=tolerance,
                        excluded_mode_pairs=excluded if group == 2 else ())
                    row['success'] = True
                except ValueError as error:
                    row.update(success=False, error=str(error))
                result['cases'].append(row)
                write(output/'study.json', result)
                print(json.dumps({k:v for k,v in row.items() if k != 'fit'}), flush=True)
    return result


def prepare_reference(source, gate_file, output):
    """Calculate this nominal circuit's own audit-basis baseline once."""
    from studies.parasitic_absorption import AUDIT, quantum_compare, write
    from studies.group_two_parasitic import OffSearch
    from studies.parasitic_absorption_audit import logical_audit
    cell = json.loads(source.read_text())
    point = gate_controls(json.loads(gate_file.read_text()))
    search = OffSearch(cell['physical'], AUDIT, capture_participation=True)
    root = search.local_root(cell['off_ghz'], (point['fc_ghz']+.00001, 9.99999))
    if root is None:
        raise ValueError('Nominal OFF reference has no verified local root below 10 GHz')
    frequency = root['frequency_ghz']
    low, high = search.at(frequency-.00001), search.at(frequency+.00001)
    reference = dict(
        nominal_off=dict(basis=AUDIT, found=True, root=root,
                         logical=logical_audit(cell['physical'], frequency)),
        off_slope=(high['A2_B1_hz']-low['A2_B1_hz'])/.00002,
        baseline=quantum_compare(cell, cell['physical'], point),
        nominal_participation=root['state_participation'])
    write(output, reference)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--input', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--amounts', type=float, nargs='+', default=[.1, .2])
    fit.add_argument('--match-rtol', type=float, default=.001)
    reference = commands.add_parser('reference')
    reference.add_argument('--input', type=Path, required=True)
    reference.add_argument('--gate', type=Path, required=True)
    reference.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'fit':
        fit_candidates(args.input, args.output, args.amounts, args.match_rtol)
    else:
        prepare_reference(args.input, args.gate, args.output)
