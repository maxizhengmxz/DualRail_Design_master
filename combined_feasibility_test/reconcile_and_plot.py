"""Overlay independent maps and reconcile fixed reduced-model charge coefficients."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from readout_workflow import calculation as study
from readout_workflow.direct_ec_feasibility import capacitances
from readout_workflow.storage import atomic_json
from readout_workflow.validation import audit_point

DATA = HERE / "data"


def configure(logical_range):
    study.CONDITIONS = replace(study.CONDITIONS, logical_frequency_ghz=logical_range)
    study.BASELINE = dict(study.BASELINE, logical_frequency=logical_range)


def reconcile_rail(ec, target_j, initial, logical_range):
    configure(logical_range)
    lower, upper = map(np.asarray, study.nominal_search_bounds())
    seed = np.clip(study.x_from_point(initial), lower + 1e-6, upper - 1e-6)
    cache = {}

    def point(x, basis=study.SCREENING_BASIS):
        key = (basis, *map(float, x))
        if key not in cache:
            c1, c2 = capacitances(ec[0], ec[1], x[2], x[3])
            cache[key] = study.point_from_x(c1, c2, x, basis)
        return cache[key]

    def residual(x):
        candidate = point(x)
        device = candidate["device"]
        return np.r_[
            study.violations(candidate, margin=0.3),
            (candidate["J12_ghz"] - target_j) / 1e-4,
            max(0, 36 - device["C1_shunt_fF"]) / 10,
            max(0, device["C1_shunt_fF"] - 128) / 10,
            max(0, 36 - device["C2_shunt_fF"]) / 10,
            max(0, device["C2_shunt_fF"] - 128) / 10,
        ]

    with threadpool_limits(limits=1):
        fit = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            x_scale=[1, 1, 2, 5],
            diff_step=1e-5,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=120,
        )
    screened = point(fit.x)
    failures = study.failures(screened)
    audited = None
    if not failures:
        audited = audit_point(screened)
        failures = study.failures(audited, require_sign_check=True)
        if abs(audited["J12_ghz"] - target_j) > 1e-5:
            failures.append("fixed_J12_target")
        device = audited["device"]
        if not all(36 <= device[k] <= 128 for k in ("C1_shunt_fF", "C2_shunt_fF")):
            failures.append("capacitance_search_range")
    return {
        "status": "validated_pass" if not failures else "no_compatible_witness_found",
        "target_J12_ghz": target_j,
        "achieved_J12_ghz": None if audited is None else audited["J12_ghz"],
        "failures": failures,
        "point": audited,
        "fit_x": fit.x.tolist(),
        "nfev": int(fit.nfev),
    }


def main():
    a_selected = json.loads((DATA / "selected_rail_a_readout_witness.json").read_text())
    b_rows = json.loads((DATA / "rail_b_readout_map_official/results.json").read_text())
    b_by_ec = {
        (r["ec1_target_mhz"], r["ec2_target_mhz"]): r
        for r in b_rows
        if r["status"] == "validated_pass"
    }
    gate_rows = []
    for path in (DATA / "gate_central_output/points").glob("*.json"):
        record = json.loads(path.read_text())
        final = record["result"].get("final") or {}
        ec = tuple(round(1000 * x) for x in record["payload"]["ec_GHz"][2:])
        gate_rows.append((ec, record, final))
    overlay = [
        (ec, record, final)
        for ec, record, final in gate_rows
        if ec in b_by_ec
        and final.get("raw_fidelity", 0) > 0.95
        and final.get("postselected_fidelity", 0) > 0.999
    ]
    output = DATA / "reconciliation"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (ec, record, final) in enumerate(sorted(overlay), 1):
        couplings = final["couplings_GHz"]
        a_result = reconcile_rail(
            (165, 240), couplings["A1_A2"], a_selected["point"], (0.14, 0.16)
        )
        b_result = reconcile_rail(
            ec, couplings["B1_B2"], b_by_ec[ec]["point"], (0.08, 0.10)
        )
        passed = a_result["status"] == b_result["status"] == "validated_pass"
        row = {
            "ec_b_mhz": ec,
            "raw_fidelity": final["raw_fidelity"],
            "postselected_fidelity": final["postselected_fidelity"],
            "fixed_interrail_J_target_ghz": couplings["A2_B1"],
            "rail_a": a_result,
            "rail_b": b_result,
            "reduced_fixed_hardware_pass": passed,
        }
        rows.append(row)
        atomic_json(output / f"ec_{ec[0]}_{ec[1]}.json", row)
        print(json.dumps({"completed": index, "ec": ec, "pass": passed}), flush=True)
    passed = [r for r in rows if r["reduced_fixed_hardware_pass"]]
    summary = {
        "independent_readout_a_pass": 48,
        "independent_readout_b_pass": len(b_by_ec),
        "gate_and_readout_overlay": len(overlay),
        "reduced_fixed_hardware_pass": len(passed),
        "scope": "Matches loaded EC and intra-rail charge coefficients. Full combined capacitance loading and resonators are not yet validated.",
    }
    atomic_json(output / "results.json", rows)
    atomic_json(output / "summary.json", summary)

    targets = sorted((r["ec1_target_mhz"], r["ec2_target_mhz"]) for r in b_rows)
    xs, ys = sorted({x for x, _ in targets}), sorted({y for _, y in targets})
    grid = np.full((len(ys), len(xs)), np.nan)
    xi, yi = {x: i for i, x in enumerate(xs)}, {y: i for i, y in enumerate(ys)}
    gate_pass = {ec for ec, _, f in gate_rows if f.get("raw_fidelity", 0) > .95 and f.get("postselected_fidelity", 0) > .999}
    overlay_set = {r["ec_b_mhz"] for r in rows}
    final_set = {r["ec_b_mhz"] for r in passed}
    for ec in targets:
        grid[yi[ec[1]], xi[ec[0]]] = 0
        if ec in b_by_ec: grid[yi[ec[1]], xi[ec[0]]] = 1
        if ec in gate_pass and ec in b_by_ec: grid[yi[ec[1]], xi[ec[0]]] = 2
        if ec in final_set: grid[yi[ec[1]], xi[ec[0]]] = 3
    cmap = ListedColormap(["#d9d9d9", "#8ecae6", "#f4a261", "#2a9d8f"])
    norm = BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    ax.imshow(grid, origin="lower", interpolation="nearest", cmap=cmap, norm=norm,
              extent=[min(xs)-2.5,max(xs)+2.5,min(ys)-2.5,max(ys)+2.5], aspect="auto")
    ax.set(xlabel=r"$E_{C,B1}$ (MHz)", ylabel=r"$E_{C,B2}$ (MHz)",
           title="Independent readout/gate overlay and hardware reconciliation")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cmap(i),label=l) for i,l in enumerate([
        "B readout not validated","B readout validated","Gate + readout overlay","Reduced fixed-hardware pass"])],fontsize=8)
    plots=DATA/"plots";plots.mkdir(exist_ok=True)
    fig.savefig(plots/"agreed_test_overlay.png",dpi=220);fig.savefig(plots/"agreed_test_overlay.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
