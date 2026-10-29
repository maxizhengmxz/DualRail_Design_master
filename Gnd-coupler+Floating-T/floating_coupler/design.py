"""Capacitance-space three-transmon off/on synthesis.

Balanced floating wiring: each link has equal +/+ and -/- capacitors.
The requested fraction sets EACH physical link capacitor divided by
e^2/(2h EC_rail). For equal +/+ and -/- capacitors c, the differential loading
is c/2, hence half this fraction. Physical island loading is checked separately.
"""

from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from core.transmon import Basis, transmon_spectrum
from .model import PORT, SCALE, build_model, port_exchange, positive, local_spectrum


@lru_cache(maxsize=2048)
def ej_for_frequency(frequency_ghz, ec_ghz, ncut=25):
    """Invert the numerical full-cosine local f01, not the harmonic formula."""
    frequency = positive(frequency_ghz, "frequency")
    ec = positive(ec_ghz, "EC")
    basis = Basis(ncut=ncut, transmon_levels=3)

    def residual(ej):
        return local_spectrum(ej, ec, basis)[0][1] - frequency

    low, high = ec * 1e-5, max(ec, (frequency + ec)**2 / (8 * ec) * 2)
    if residual(low) >= 0:
        raise ValueError("Requested frequency is below the local zero-EJ gap")
    for _ in range(20):
        if residual(high) > 0:
            return brentq(residual, low, high, xtol=1e-11)
        high *= 2
    raise ValueError("Could not bracket EJ for supplied frequency")


def balanced_config(spec, ec_c, direct_each_ff, levels=4, ncut=25):
    """Construct one physical capacitor layout at the off reference frequencies."""
    rail_ec = np.asarray(spec["rail_ec_ghz"], float)
    fractions = np.asarray(spec["link_loading_fractions"], float)
    if rail_ec.shape != (2,) or fractions.shape != (2,):
        raise ValueError("Supply two rail ECs and two loading fractions")
    if np.any(rail_ec <= 0) or np.any(fractions <= 0) or np.any(fractions >= 0.1):
        raise ValueError("Rail ECs must be positive and loading fractions strictly between 0 and 0.1")
    each = SCALE / rail_ec * fractions
    capacitors = []
    for (a, b), cap in zip((("A2", "C"), ("C", "B1"), ("A2", "B1")),
                           (*each, direct_each_ff), strict=True):
        for sign in ("+", "-"):
            capacitors.append({"from": a + sign, "to": b + sign, "capacitance_ff": float(cap)})
    ecs = [rail_ec[0], ec_c, rail_ec[1]]
    frequencies = [spec["fa_ghz"], spec["off_fc_ghz"], spec["off_fb_ghz"]]
    return {"nodes": list(PORT), "transmons": {
        name: {"ec_ghz": float(ec), "ej_ghz": ej_for_frequency(f, ec, ncut)}
        for name, ec, f in zip(PORT, ecs, frequencies, strict=True)},
        "electrostatics": {"mode": "island_capacitors", "capacitors": capacitors},
        "basis": {"ncut": ncut, "levels": levels}}


def operating_point(model, fb, fc):
    """Retune EJs only; preserve the constructed network and all EC/J."""
    ejs = model.ej.copy()
    ejs[2] = ej_for_frequency(fb, model.ec[2], model.basis.ncut)
    ejs[1] = ej_for_frequency(fc, model.ec[1], model.basis.ncut)
    point = replace(model, ej=ejs)
    row = port_exchange(point, ejs[1])
    row.update(b1_bare_f01_ghz=float(fb), ej_ghz=ejs.tolist(),
               ej_over_ec=(ejs / model.ec).tolist())
    return row


def synthesize(spec):
    """Find direct capacitor for off, then coupler EC for on, at chosen frequencies.

    This is a constructive feasibility solve, not a global optimality claim.
    Geometry remains identical between the two operating points.
    """
    for key in ("fa_ghz", "off_fb_ghz", "on_fb_ghz", "off_fc_ghz", "on_fc_ghz"):
        positive(spec[key], key)
    for value in spec["rail_ec_ghz"]:
        positive(value, "rail EC")
    for value in spec["coupler_ec_bounds_ghz"]:
        positive(value, "coupler EC bound")
    if len(spec["coupler_ec_bounds_ghz"]) != 2 or spec["coupler_ec_bounds_ghz"][1] <= spec["coupler_ec_bounds_ghz"][0]:
        raise ValueError("Supply increasing coupler EC bounds")
    levels = spec.get("search_levels", 4)
    ncut = spec.get("search_ncut", 25)
    ec_bounds = spec["coupler_ec_bounds_ghz"]
    direct_max = positive(spec["direct_each_max_ff"], "direct capacitor upper bound")
    target = positive(spec["target_exchange_mhz"], "target exchange")
    trace = []

    def design_at_ec(ec):
        def off_residual(direct):
            config = balanced_config(spec, ec, direct, levels, ncut)
            model = build_model(config)
            row = port_exchange(model, model.ej[1])
            if not row["valid"]:
                raise ValueError("Off-point retained subspace unreliable")
            return row["exchange_mhz"]

        direct = brentq(off_residual, 0, direct_max, xtol=1e-12)
        config = balanced_config(spec, ec, direct, levels, ncut)
        model = build_model(config)
        on = operating_point(model, spec["on_fb_ghz"], spec["on_fc_ghz"])
        if not on["valid"]:
            raise ValueError("On-point retained subspace unreliable")
        trace.append({"ec_c_ghz": float(ec), "direct_each_ff": direct,
                      "on_exchange_mhz": on["exchange_mhz"]})
        return abs(on["exchange_mhz"]) - target, config

    endpoint_residuals = [design_at_ec(value)[0] for value in ec_bounds]
    if np.prod(endpoint_residuals) > 0:
        raise ValueError(f"EC bracket does not cross target: {trace}")
    ec = brentq(lambda value: design_at_ec(value)[0], *ec_bounds, xtol=1e-10)
    _, config = design_at_ec(ec)
    model = build_model(config)
    budgets = SCALE / np.asarray(spec["rail_ec_ghz"])
    each = budgets * np.asarray(spec["link_loading_fractions"])
    laplacian = np.asarray(model.electrostatics["island_laplacian_ff"])
    physical_fractions = [float(each[0] / laplacian[0, 0]), float(each[1] / laplacian[4, 4])]
    if max(physical_fractions) >= 0.1:
        raise ValueError("Physical per-island link loading exceeds 10 percent")
    return {"design_specification": spec, "config": config,
            "loading_audit": {"rail_effective_capacitance_ff": budgets.tolist(),
                              "link_each_island_capacitance_ff": each.tolist(),
                              "link_differential_loading_ff": (each / 2).tolist(),
                              "link_differential_loading_fractions": (each / (2 * budgets)).tolist(),
                              "link_physical_island_loading_fractions": physical_fractions},
            "electrostatics": model.electrostatics,
            "off": operating_point(model, spec["off_fb_ghz"], spec["off_fc_ghz"]),
            "on": operating_point(model, spec["on_fb_ghz"], spec["on_fc_ghz"]),
            "solve_trace": trace}


def audit_design(result, bases):
    """Re-evaluate fixed hardware/EJs in larger bases, without recalibration."""
    original = build_model(result["config"])
    checks = []
    for ncut, levels in bases:
        model = replace(original, basis=Basis(ncut=ncut, transmon_levels=levels))
        entry = {"ncut": ncut, "levels": levels}
        for name in ("off", "on"):
            ej = np.array(result[name]["ej_ghz"])
            operating_model = replace(model, ej=ej)
            row = port_exchange(operating_model, ej[1])
            row["computational_residuals"] = computational_residuals(operating_model)
            entry[name] = row
        checks.append(entry)
    return checks


def computational_residuals(model):
    """Physical-qubit four-state diagnostics, not dual-rail logical quantities.

    Reference order is |000>, |001>, |100>, |101> in A2,C,B1 order.
    Coupler occupation is its local LEVEL index expectation, not photon number.
    """
    if model.names != PORT:
        raise ValueError("Computational residuals require A2,C,B1")
    ham, _ = model.hamiltonian()
    energy, vectors = eigh(ham)
    levels = model.basis.transmon_levels
    refs = np.array([np.ravel_multi_index(state, (levels,) * 3)
                     for state in ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1))])
    projection = vectors[refs, :]
    _, columns = linear_sum_assignment(-abs(projection)**2)
    selected = projection[:, columns]
    u, singular, vh = np.linalg.svd(selected)
    rotation = u @ vh
    effective = (rotation * energy[columns]) @ rotation.T
    coupler_levels = np.indices((levels,) * 3)[1].ravel()
    coupler_mean = coupler_levels @ (abs(vectors[:, columns])**2)
    diagonal = effective.diagonal()
    return {"reference_labels": ["000", "001", "100", "101"],
            "physical_zz_mhz": float((diagonal[3] - diagonal[2] - diagonal[1] + diagonal[0]) * 1000),
            "double_flip_mhz": float(effective[0, 3] * 1000),
            "minimum_subspace_overlap": float(singular.min()**2),
            "assigned_eigenstate_coupler_mean_local_level": coupler_mean.tolist(),
            "effective_hamiltonian_ghz": effective.tolist()}


def refine_frequencies(result, ncut=40, levels=7, half_width_ghz=0.1):
    """Freeze the synthesized geometry and refine only coupler frequencies."""
    model = build_model(result["config"])
    model = replace(model, basis=Basis(ncut=ncut, transmon_levels=levels))
    refined = {"ncut": ncut, "levels": levels,
               "frequency_bracket_half_width_ghz": half_width_ghz}
    for name in ("off", "on"):
        fb = result["design_specification"][name + "_fb_ghz"]
        center = result["design_specification"][name + "_fc_ghz"]
        target = 0 if name == "off" else result["on"]["exchange_mhz"]

        def residual(fc):
            row = operating_point(model, fb, fc)
            if not row["valid"]:
                raise ValueError("Invalid retained subspace during frequency refinement")
            return row["exchange_mhz"] - target

        frequency = brentq(residual, center - half_width_ghz, center + half_width_ghz, xtol=1e-11)
        row = operating_point(model, fb, frequency)
        refined[name] = row
    audit_input = dict(result, off=refined["off"], on=refined["on"])
    refined["fixed_hardware_convergence"] = audit_design(audit_input, [(ncut, levels), (45, 8), (50, 9)])
    return refined
