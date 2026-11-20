"""Allocate a frozen seven-mode pad circuit at a common rail self fraction p.

This is an opt-in post-design electrostatic study.  It preserves the seven
diagonal entries and nine intended off-diagonal entries of K while enforcing
equal grounds and Cs/(Cs+Cg_each/2)=p on A1, A2, B1 and B2.  It does not inherit
OFF, readout or gate certification from the nominal geometry.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares, minimize

from studies.parasitic_absorption import rebuild, write
from floating_coupler.mixed_electrostatics import mixed_budget


RAILS = ("A1", "A2", "B1", "B2")
INTENDED = (
    ("A1", "A2"), ("A2", "C"), ("C", "B1"), ("A2", "B1"),
    ("B1", "B2"), ("A1", "rA"), ("A2", "rA"),
    ("B1", "rB"), ("B2", "rB"),
)


def self_fraction(physical, name):
    a, b = physical["pad_pairs"][name]
    g1 = physical["ground_capacitances_ff"][a]
    g2 = physical["ground_capacitances_ff"][b]
    series = g1 * g2 / (g1 + g2)
    cs = physical["pad_mutual_capacitances_ff"][name]
    return float(cs / (cs + series))


def allocate(nominal, target_p, max_nfev=4000, minimum_coupler_own_fraction=0.55):
    if not 0 < target_p < 1:
        raise ValueError("target p must lie strictly between zero and one")
    began = time.perf_counter()
    nominal, reference = rebuild(nominal)
    order = reference["modal_order"]
    index = {name: i for i, name in enumerate(order)}
    target_k = reference["K"]
    intended_indices = np.asarray([(index[a], index[b]) for a, b in INTENDED])
    intended_scale = np.abs(target_k[intended_indices[:, 0], intended_indices[:, 1]])
    if np.any(intended_scale == 0):
        raise ValueError("An intended nominal K entry is zero")

    parameters = []
    for name in RAILS:
        a, b = nominal["pad_pairs"][name]
        g1 = nominal["ground_capacitances_ff"][a]
        g2 = nominal["ground_capacitances_ff"][b]
        if not np.isclose(g1, g2, rtol=0, atol=1e-10):
            raise ValueError(f"{name} does not have equal nominal pad grounds")
        parameters.append(("ground", name, float((g1 + g2) / 2)))
    for i, edge in enumerate(nominal["physical_links"]):
        parameters.append(("link", i, float(edge["capacitance_ff"])))
    for name, value in nominal["resonator_self_capacitances_ff"].items():
        parameters.append(("resonator", name, float(value)))
    for name, value in nominal["pad_mutual_capacitances_ff"].items():
        parameters.append(("self", name, float(value)))
    base_values = np.asarray([row[2] for row in parameters])

    def physical_at(log_ratio):
        values = base_values * np.exp(log_ratio)
        physical = deepcopy(nominal)
        for (kind, key, _), value in zip(parameters, values, strict=True):
            if kind == "ground":
                for pad in physical["pad_pairs"][key]:
                    physical["ground_capacitances_ff"][pad] = float(value)
            elif kind == "link":
                physical["physical_links"][key]["capacitance_ff"] = float(value)
            elif kind == "resonator":
                physical["resonator_self_capacitances_ff"][key] = float(value)
            else:
                physical["pad_mutual_capacitances_ff"][key] = float(value)
        return physical

    def residual(log_ratio):
        physical, circuit = rebuild(physical_at(log_ratio))
        k = circuit["K"]
        diagonal = np.diag(k) / np.diag(target_k) - 1
        intended = (
            k[intended_indices[:, 0], intended_indices[:, 1]]
            - target_k[intended_indices[:, 0], intended_indices[:, 1]]
        ) / intended_scale
        p_error = np.asarray([self_fraction(physical, name) - target_p for name in RAILS])
        return np.r_[diagonal, intended, p_error]

    initial = np.zeros(len(parameters))
    for name in RAILS:
        gi = next(i for i, row in enumerate(parameters) if row[:2] == ("ground", name))
        si = next(i for i, row in enumerate(parameters) if row[:2] == ("self", name))
        a, b = nominal["pad_pairs"][name]
        ground = nominal["ground_capacitances_ff"][a]
        own = nominal["pad_mutual_capacitances_ff"][name] + ground / 2
        initial[gi] = np.log((2 * (1 - target_p) * own) / ground)
        initial[si] = np.log((target_p * own) / nominal["pad_mutual_capacitances_ff"][name])

    fit = least_squares(
        residual, initial, bounds=(np.log(1e-3), np.log(100.0)),
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=max_nfev,
    )
    exact_fit = fit

    def evaluated(log_ratio):
        physical, circuit = rebuild(physical_at(log_ratio))
        budget = mixed_budget(
            physical["pad_pairs"], physical["ground_capacitances_ff"],
            physical["pad_mutual_capacitances_ff"], physical["physical_links"],
            physical["physical_links"][:3],
            resonator_caps_ff=physical["resonator_self_capacitances_ff"],
        )
        return physical, circuit, budget

    exact_physical, exact_circuit, exact_budget = evaluated(fit.x)
    loading_repair = None
    if exact_budget["C"]["own_fraction"] < minimum_coupler_own_fraction - 1e-10:
        def objective(log_ratio):
            _, circuit, _ = evaluated(log_ratio)
            k = circuit["K"]
            intended = (
                k[intended_indices[:, 0], intended_indices[:, 1]]
                - target_k[intended_indices[:, 0], intended_indices[:, 1]]
            ) / intended_scale
            return float(intended @ intended + 1e-8 * (log_ratio @ log_ratio))

        def equality(log_ratio):
            physical, circuit, _ = evaluated(log_ratio)
            diagonal = np.diag(circuit["K"]) / np.diag(target_k) - 1
            p_error = np.asarray(
                [self_fraction(physical, name) - target_p for name in RAILS]
            )
            return np.r_[diagonal, p_error]

        def loading(log_ratio):
            return evaluated(log_ratio)[2]["C"]["own_fraction"] - minimum_coupler_own_fraction

        repaired = minimize(
            objective, fit.x, method="SLSQP",
            bounds=[(np.log(1e-3), np.log(100.0))] * len(parameters),
            constraints=[dict(type="eq", fun=equality), dict(type="ineq", fun=loading)],
            options=dict(ftol=1e-13, maxiter=max_nfev, disp=False),
        )
        loading_repair = dict(success=bool(repaired.success), message=repaired.message,
                              evaluations=int(repaired.nfev))
        if repaired.success:
            fit = repaired
    physical, circuit = rebuild(physical_at(fit.x))
    final_residual = residual(fit.x)
    k = circuit["K"]
    intended_error = (
        k[intended_indices[:, 0], intended_indices[:, 1]]
        / target_k[intended_indices[:, 0], intended_indices[:, 1]] - 1
    )
    matched = np.zeros_like(k, dtype=bool)
    np.fill_diagonal(matched, True)
    for i, j in intended_indices:
        matched[i, j] = matched[j, i] = True
    residual_mask = np.triu(~matched, 1)
    normalized = (k - target_k) / np.sqrt(np.outer(np.diag(target_k), np.diag(target_k)))
    removed = physical["physical_links"][:3]
    budget = mixed_budget(
        physical["pad_pairs"], physical["ground_capacitances_ff"],
        physical["pad_mutual_capacitances_ff"], physical["physical_links"], removed,
        resonator_caps_ff=physical["resonator_self_capacitances_ff"],
    )
    changes = []
    for (kind, key, nominal_value), ratio, value in zip(
        parameters, np.exp(fit.x), base_values * np.exp(fit.x), strict=True
    ):
        label = key if kind != "link" else (
            nominal["physical_links"][key]["from"] + "-" + nominal["physical_links"][key]["to"]
        )
        changes.append(dict(kind=kind, component=label, nominal_ff=nominal_value,
                            allocated_ff=float(value), ratio=float(ratio),
                            percent_change=float(100 * (ratio - 1))))
    return dict(
        schema="tied-p-allocation-v1", target_p=float(target_p),
        converged=bool(
            fit.success
            and np.max(np.abs(np.diag(k) / np.diag(target_k) - 1)) < 1e-8
            and max(abs(self_fraction(physical, name) - target_p) for name in RAILS) < 1e-8
            and budget["C"]["own_fraction"] >= minimum_coupler_own_fraction - 1e-9
            and np.max(np.abs(intended_error)) <= 0.001
        ),
        optimizer_success=bool(fit.success), optimizer_message=fit.message,
        evaluations=int(fit.nfev), physical=physical, budget=budget,
        achieved_p={name: self_fraction(physical, name) for name in RAILS},
        maximum_diagonal_relative_error=float(np.max(np.abs(np.diag(k) / np.diag(target_k) - 1))),
        maximum_intended_relative_error=float(np.max(np.abs(intended_error))),
        residual_normalized_rms=float(np.sqrt(np.mean(normalized[residual_mask] ** 2))),
        maximum_fractional_capacitor_change=float(np.max(np.abs(np.exp(fit.x) - 1))),
        minimum_coupler_own_fraction=float(minimum_coupler_own_fraction),
        exact_unconstrained_coupler_own_fraction=float(exact_budget["C"]["own_fraction"]),
        loading_repair=loading_repair,
        changes=changes, intended_pairs=[list(pair) for pair in INTENDED],
        seconds=float(time.perf_counter() - began),
        scope=("Electrostatic allocation only. OFF, readout and gate certificates from the "
               "nominal geometry are not inherited."),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--p", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=4000)
    args = parser.parse_args()
    source = json.loads(args.cell.read_text())
    cell = source.get("calibrated_cell", source)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"source": str(args.cell.resolve()), "cases": []}
    for target in args.p:
        row = allocate(cell["physical"], target, args.max_nfev)
        path = args.output / ("p" + str(target).replace(".", "p") + ".json")
        write(path, row)
        summary["cases"].append(dict(target_p=target, file=path.name, **{
            key: row[key] for key in (
                "converged", "maximum_diagonal_relative_error",
                "maximum_intended_relative_error", "residual_normalized_rms",
                "maximum_fractional_capacitor_change", "seconds")
        }))
        print(json.dumps(summary["cases"][-1]), flush=True)
    write(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
