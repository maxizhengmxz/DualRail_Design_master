"""Explicit physical parasitic capacitors (fF); never inferred modal couplings.

Mirror convention: A11/A12/A21/A22 <-> B22/B21/B12/B11, r1 <-> r2.
Only edges whose two endpoints exist in a chosen subsystem are retained.
This projection omits removed devices; it does not replace them by grounds.
"""
from math import isfinite

PARASITIC_PORTS = (
    ('A11', 'r1'), ('A22', 'r1'), ('A11', 'A22'),
    ('A21', 'C1'), ('A21', 'B11'),
    ('A12', 'C1'), ('A11', 'C1'),
    ('B22', 'r2'), ('B11', 'r2'), ('B22', 'B11'),
    ('B12', 'C1'), ('B12', 'A22'),
    ('B21', 'C1'), ('B22', 'C1'),
)
DEFAULT_PARASITIC_CAPACITANCES_FF = {f'{a}-{b}': 0.0 for a, b in PARASITIC_PORTS}


def with_parasitics(links, values=None, available_nodes=None):
    """Merge explicit overrides once, preserving intended-link order/metadata.

    Missing entries mean zero. Existing tagged parasitics are preserved unless
    explicitly overridden; an intended link at the same endpoints is an error.
    No self-capacitance compensation or EJ retuning is performed here.
    """
    values = {} if values is None else values
    unknown = set(values) - set(DEFAULT_PARASITIC_CAPACITANCES_FF)
    if unknown:
        raise ValueError(f'Unknown parasitic capacitor names: {sorted(unknown)}')
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f'{name} requires finite nonnegative parasitic capacitance in fF')
    result = [dict(e) for e in links]
    nodes = None if available_nodes is None else set(available_nodes)
    for a, b in PARASITIC_PORTS:
        name = f'{a}-{b}'
        if nodes is not None and not {a, b} <= nodes:
            continue
        matches = [e for e in result if {e['from'], e['to']} == {a, b}]
        if name not in values and not any(e.get('parasitic', False) for e in matches):
            continue
        if len(matches) > 1:
            raise ValueError(f'Duplicate parasitic edge: {name}')
        if matches:
            edge = matches[0]
            if not edge.get('parasitic', False):
                raise ValueError(f'Parasitic edge conflicts with intended capacitor: {name}')
            if name not in values:
                continue
            edge.update(capacitance_ff=float(values[name]), name=f'Cpar_{name}', parasitic=True)
        elif name in values:
            result.append(dict(name=f'Cpar_{name}', **{'from': a, 'to': b},
                               capacitance_ff=float(values[name]), parasitic=True))
    return result
