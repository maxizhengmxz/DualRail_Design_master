"""Report geometric link budgets and net EC loading for balanced floating links."""

from collections import defaultdict

import numpy as np

from .model import SCALE, build_model, floating_inverse


def coupler_loading(config, model=None):
    """Remove all C links at fixed shunts for the net-loading counterfactual.

    The geometric metric is sum(c/2)/Ceff for equal +/+ and -/- capacitors.
    It is deliberately distinct from 1-EC_loaded/EC_without_links. Direct and
    intra-rail links stay present in the counterfactual; no shunts are refitted.
    """
    if config['electrostatics']['mode']=='explicit_pads':
        from .pad_electrostatics import build_pad_circuit,pad_capacitance_budget
        physical=config['electrostatics']
        c=build_pad_circuit(physical['pad_pairs'],physical['ground_capacitances_ff'],{},
                           physical['physical_links'],physical['pad_mutual_capacitances_ff'])
        cpads=set(c.pad_pairs['C'])
        removed=[e for e in c.physical_links if e['from'] in cpads or e['to'] in cpads]
        budget=pad_capacitance_budget(c,removed)
        return dict(counterfactual='Remove all coupler links at fixed pad grounds, pad mutuals and other physical links.',
                    nodes={n:dict(v,net_effective_loading_fraction=v['specified_links_net_loading_fraction'],
                                  self_capacitance_fraction=v['own_total_fraction']) for n,v in budget.items()})
    if config["electrostatics"]["mode"] != "island_capacitors":
        raise ValueError("Physical capacitor configuration is required")
    model = build_model(config) if model is None else model
    caps = config["electrostatics"]["capacitors"]
    groups = defaultdict(lambda: defaultdict(float))
    retained = []
    for cap in caps:
        a, b = cap["from"], cap["to"]
        if "C" not in (a[:-1], b[:-1]):
            retained.append(cap)
            continue
        if a[-1] != b[-1]:
            raise ValueError("Geometric loading requires same-polarity balanced links")
        groups[tuple(sorted((a[:-1], b[:-1])))][a[-1]] += cap["capacitance_ff"]
    additions = dict.fromkeys(model.names, 0.)
    for (a, b), values in groups.items():
        if not np.isclose(values["+"], values["-"], rtol=1e-10, atol=1e-12):
            raise ValueError("Geometric loading requires equal +/+ and -/- capacitors")
        for node in (a, b):
            additions[node] += values["+"] / 2
    shunts = [model.electrostatics["junction_capacitances_ff"][n] for n in model.names]
    inverse, _ = floating_inverse(model.names, shunts, retained)
    without = 1 / np.diag(inverse)
    loaded = SCALE / model.ec
    return {
        "counterfactual": "remove all C links at fixed physical shunts and all other links",
        "nodes": {name: {
            "loaded_effective_capacitance_ff": float(loaded[i]),
            "physical_across_terminal_capacitance_ff": float(shunts[i]),
            "self_capacitance_fraction": float(shunts[i] / loaded[i]),
            "without_C_links_effective_capacitance_ff": float(without[i]),
            "differential_C_link_capacitance_ff": float(additions[name]),
            "link_budget_fraction": float(additions[name] / loaded[i]),
            "net_effective_loading_fraction": float(1 - without[i] / loaded[i]),
        } for i, name in enumerate(model.names)},
        "combined_rail_link_budget_fraction": float(sum(additions[n] / loaded[i]
            for i, n in enumerate(model.names) if n != "C")),
    }
