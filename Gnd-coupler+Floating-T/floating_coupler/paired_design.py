"""Constructive floating-capacitor synthesis against the dressed CZ channel.

Fixed ratios of normalized differential loading define search families, not a
global minimum over all island layouts. Rail ECs/idle splittings are preserved.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from .design import ej_for_frequency
from .model import FULL, SCALE, build_model
from .pairs import PairModel
from .on_gate import energy_axis_matter_hamiltonian
from .channel_policy import channel_indices,require_selection,selection_from_request,interaction_scores,off_target_definition,require_off_definition


def off_channel(detail):
    """Temporary physical 0100-0010 OFF target; logical exchange is residual."""
    if 'physical_exchange' not in detail:
        raise ValueError('Recompute physical single-excitation dressing; legacy logical/CZ records do not certify OFF')
    result=detail['physical_exchange']
    require_off_definition(result.get('off_definition'),detail.get('off_definition',{}).get('model','five'))
    if not result['valid']:raise ValueError('Physical OFF subspace is not well identified')
    return result


def channel_scores(detail, source, auxiliary):
    """Shared toy S1/S2 formula in this explicitly retained dressed basis.

    Actual fixed-hardware score at this bias, not a common-g design certificate.
    Never compare it directly with scores using a different retained basis.
    """
    ham=energy_axis_matter_hamiltonian(detail)
    labels=detail['reference_labels']
    logical=[labels.index(x) for x in ('00','01','10','11')]
    scores=interaction_scores(np.diag(ham),ham,logical,labels.index(source),
                              labels.index(auxiliary),active_mask=abs(ham)>1e-12)
    return dict(scores,source_logical=source,auxiliary=auxiliary,
                intended_g_mhz=float(abs(ham[labels.index(source),labels.index(auxiliary)])*1000),
                score_basis_id=f"retained-dressed-matter-{len(labels)}",
                scope='Actual fixed-hardware bias diagnostic; common-g selection must be evaluated separately')


def selected_channel(detail, source=None, target=None, *, selection=None):
    """No silent source default. Explicit source/target are diagnostic requests."""
    if source is None and target is None:
        chosen=require_selection(selection or detail.get('channel_selection'))
        source,target=chosen.source_logical,chosen.auxiliary
    elif source is None or target is None:
        raise ValueError('Supply both diagnostic source and target, or a toy-model channel selection')
    else:
        chosen=None
        if selection is not None:
            chosen=require_selection(selection)
            if (source,target)!=(chosen.source_logical,chosen.auxiliary):raise ValueError('Diagnostic arguments disagree with selected toy channel')
    ham = energy_axis_matter_hamiltonian(detail)
    labels = detail["reference_labels"]
    i, j = labels.index(source), labels.index(target)
    return {"source_logical":source,"auxiliary":target,
            "selection_origin":'toy_model_S1' if chosen is not None else 'explicit_diagnostic',
            "g_mhz": float(ham[i, j] * 1000),
            "detuning_mhz": float((ham[i, i] - ham[j, j]) * 1000)}


def synthesize_paired(spec, initial=None):
    """Solve loading/direct/rail caps/gate B for off zero, on resonance and g."""
    selected=selection_from_request(spec).to_dict()
    ec = np.array([spec["rail_ec_ghz"][n] if n != "C" else spec["coupler_ec_ghz"] for n in FULL])
    basis = spec["basis"]
    frequencies = [spec["idle_fa_ghz"], spec["idle_fa_ghz"], spec["off_fc_ghz"],
                   spec["idle_fb_ghz"], spec["idle_fb_ghz"]]
    ejs = [ej_for_frequency(f, charging, basis["ncut"]) for f, charging in zip(frequencies, ec, strict=True)]
    rail_budgets = SCALE / ec[[1, 3]]
    ratio = float(spec.get("loading_ratio_a_over_b", 1.))
    if ratio <= 0:
        raise ValueError("Loading ratio must be positive")
    loading_factors = np.array([np.sqrt(ratio), 1 / np.sqrt(ratio)])

    def config_at(x):
        loading, direct, pair_a, pair_b, _ = x
        each = 2 * loading * loading_factors * rail_budgets
        caps = []
        for (a, b), cap in zip((("A2", "C"), ("C", "B1"), ("A2", "B1"), ("A1", "A2"), ("B1", "B2")),
                               (*each, direct, pair_a, pair_b), strict=True):
            caps.extend({"from": a + sign, "to": b + sign, "capacitance_ff": float(cap)} for sign in ("+", "-"))
        return {"nodes": list(FULL), "channel_selection":selected,"transmons": {n: {"ec_ghz": float(charging), "ej_ghz": float(ej)}
                  for n, charging, ej in zip(FULL, ec, ejs, strict=True)},
                "electrostatics": {"mode": "island_capacitors", "capacitors": caps},
                "basis": {"levels": basis["local_levels"], "ncut": basis["ncut"]}}

    evaluations = 0

    def evaluate(x, detailed=False):
        nonlocal evaluations
        evaluations += 1
        model = build_model(config_at(x))
        off_system = PairModel(model, **basis)
        off = off_system.evaluate(spec["off_fc_ghz"])
        ej = model.ej.copy()
        for i in (3, 4):
            ej[i] = ej_for_frequency(x[4], model.ec[i], basis["ncut"])
        on = PairModel(replace(model, ej=ej), **basis).evaluate(spec["on_fc_ghz"])
        off_ch, on_ch = off_channel(off), selected_channel(on)
        residual = np.array([
            (off_system.a["splitting_mhz"] - spec["pair_splittings_mhz"][0]) / spec["pair_splittings_mhz"][0],
            (off_system.b["splitting_mhz"] - spec["pair_splittings_mhz"][1]) / spec["pair_splittings_mhz"][1],
            off_ch["g_mhz"] / spec.get("off_exchange_scale_mhz", .001),
            (on_ch["g_mhz"] - spec["on_g_target_mhz"]) / 5,
            on_ch["detuning_mhz"] / 50])
        return (residual, model, off, on) if detailed else residual

    x0 = initial if initial is not None else [0.07, 0.3, 3.7, 5.3, 4.9]
    lower = [0.005, 1e-8, 0.05, 0.05, spec["gate_fb_bounds_ghz"][0]]
    upper = [spec["maximum_loading_fraction"] / max(loading_factors), 5., 20., 25., spec["gate_fb_bounds_ghz"][1]]
    coupler_limit = spec.get("maximum_coupler_loading_fraction")
    if coupler_limit is not None:
        if not 0 < coupler_limit < 1:
            raise ValueError("Coupler loading limit must be strictly between zero and one")
        # Both balanced links contribute c/2 to the coupler differential diagonal.
        # Bound the geometric loading variable before solving, not after fitting.
        upper[0] = min(upper[0], coupler_limit * (SCALE / ec[2])
                       / np.dot(loading_factors, rail_budgets))
    if upper[0] <= lower[0]:
        raise ValueError("Loading bounds leave no feasible numerical interval")
    fit = least_squares(evaluate, np.clip(x0, np.array(lower) + 1e-7, np.array(upper) - 1e-7),
                        bounds=(lower, upper), diff_step=2e-5, x_scale="jac",
                        max_nfev=70, xtol=2e-9, ftol=2e-9, gtol=2e-9)
    residual, model, off, on = evaluate(fit.x, detailed=True)
    loading, direct, pair_a, pair_b, fb = fit.x
    laplacian = np.asarray(model.electrostatics["island_laplacian_ff"])
    fractions = loading * loading_factors
    caps_each = 2 * fractions * rail_budgets
    return {"off_definition":off_target_definition("five"),"spec": spec, "optimizer_success": bool(fit.success), "message": str(fit.message),
            "maximum_scaled_residual": float(max(abs(residual))), "evaluations": evaluations,
            "feasible": bool(max(abs(residual)) < 2e-5 and off["minimum_subspace_overlap"] > 0.8
                             and on["minimum_subspace_overlap"] > 0.8),
            "solution_vector": fit.x.tolist(), "config": config_at(fit.x),
            "electrostatics": model.electrostatics,
            "differential_loading_fraction": float(max(fractions)),
            "differential_loading_fractions": fractions.tolist(),
            "coupler_link_each_ff": caps_each.tolist(), "direct_each_ff": float(direct),
            "combined_rail_loading_fraction": float(sum(fractions)),
            "coupler_link_loading_fraction": float(sum(caps_each) / 2 / (SCALE / ec[2])),
            "pair_link_each_ff": [float(pair_a), float(pair_b)], "gate_fb_ghz": float(fb),
            "physical_island_loading_fractions": [float(caps_each[0] / laplacian[2, 2]),
                                                   float(caps_each[1] / laplacian[6, 6])],
            "off": off, "on": on,
            "coupler_on_ej_over_ec": on["coupler_ej_ghz"] / spec["coupler_ec_ghz"]}
