"""Readout matching versus coupling, detuning, C12 and coupling asymmetry.

This investigation does not change the notebook's manual inputs. Frequencies
are GHz, capacitances fF, impedances equivalent-mode ohms. A low-power
dispersive screen checks all upward pair transitions from g,m,p against the
resonator, including the two-excitation manifold relevant to chi.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge as e, h
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar, brentq
from threadpoolctl import threadpool_limits

from model import Basis, Device, ReadoutModel, _transmon
from guided_search import predict_resonances


def diagnostics(model, point):
    b = model.basis
    en1, n1 = _transmon(point["ej1_ghz"], model.ec_ghz[0], b)
    en2, n2 = _transmon(point["ej2_ghz"], model.ec_ghz[1], b)
    n1p, n2p = np.kron(n1, np.eye(b.transmon_levels)), np.kron(np.eye(b.transmon_levels), n2)
    hp = np.diag((en1[:, None]+en2[None, :]).ravel())
    hp += 4*e**2*model.cinv[0, 1]/(h*1e9)*(n1p@n2p)
    energy, vectors = eigh(hp)
    coupling = 2*e*model.qr_zpf/(h*1e9)*(model.cinv[0, 2]*n1p+model.cinv[1, 2]*n2p)
    gij = np.abs(vectors.T @ coupling @ vectors)
    ratios = []
    for i in range(3):
        for j in range(i+1, len(energy)):
            if gij[i, j] < 1e-9:
                continue
            detuning = abs(energy[j]-energy[i]-model.fr_ghz)
            ratios.append((gij[i, j]/detuning if detuning > 0 else np.inf, i, j))
    worst = max(ratios) if ratios else (0., 0, 0)
    return {"lambda_max": float(worst[0]), "worst_pair_initial": int(worst[1]),
            "worst_pair_final": int(worst[2]),
            "lambda_ground": float(max(gij[0,j]/abs(energy[j]-energy[0]-model.fr_ghz) for j in (1,2))),
            "local_lambda1": point["g1r_over_2pi_mhz"]*1e-3/abs(point["bare_f1_ghz"]-model.fr_ghz),
            "local_lambda2": point["g2r_over_2pi_mhz"]*1e-3/abs(point["bare_f2_ghz"]-model.fr_ghz),
            "ratio_m": abs(point["delta_chi_ghz"]/point["chi_m_ghz"]) if point["chi_m_ghz"] else np.inf}


def resonance(model, ej1):
    prediction, _ = predict_resonances(model, ej1, "EJ2_GHz", (1, 45), samples=17)
    if len(prediction) != 1:
        return None
    center = prediction[0]
    # In this far-detuned investigation there is one local rail crossing.
    result = minimize_scalar(lambda ej2: model.evaluate(ej1, ej2)["splitting_ghz"],
                             bounds=(.95*center, 1.05*center), method="bounded",
                             options={"xatol": 2e-6})
    if not result.success:
        raise RuntimeError("Resonance refinement failed")
    point = model.evaluate(ej1, result.x)
    point.update(diagnostics(model, point))
    return point


def intervals(points, key):
    groups, group = [], []
    for p in points:
        if p[key]:
            group.append(p)
        else:
            if group:
                groups.append(group)
            group = []
    if group:
        groups.append(group)
    return [{"lower_ghz": g[0]["operating_frequency_ghz"], "upper_ghz": g[-1]["operating_frequency_ghz"],
             "width_mhz": (g[-1]["operating_frequency_ghz"]-g[0]["operating_frequency_ghz"])*1e3,
             "count": len(g), "min_abs_chi_mhz": min(min(abs(p["chi_m_ghz"]),abs(p["chi_p_ghz"])) for p in g)*1e3}
            for g in groups]


def run_case(cap, fr, c12, asymmetry=1., grid=None, impedance=50., shunts=(64.,54.)):
    # Preserve the user's current shunts and impedance; vary only named factors.
    omega, z = 2*np.pi*fr*1e9, impedance
    d = Device(C1_shunt_fF=shunts[0], C2_shunt_fF=shunts[1], C12_fF=c12,
               C1r_fF=cap, C2r_fF=cap*asymmetry,
               Cr_fF=1/(omega*z)*1e15, Lr_nH=z/omega*1e9)
    m = ReadoutModel(d)
    grid = np.linspace(6, 16, 41) if grid is None else grid
    points = []
    for ej1 in grid:
        p = resonance(m, ej1)
        if p is None:
            continue
        # Both normalization choices must pass; numerical reference-state
        # assignment must also remain reliable for conditional shifts.
        p["matched"] = bool(max(p["ratio_m"],p["abs_delta_chi_over_chi_p"]) < 1e-2 and p["hybridization"] >= .95)
        p["dispersive"] = bool(p["lambda_max"] <= .1 and p["min_assignment_overlap"] >= .95)
        p["good"] = p["matched"] and p["dispersive"]
        points.append(p)
    eligible = [p for p in points if p["hybridization"] >= .95]
    best = min(eligible, key=lambda p:p["abs_delta_chi_over_chi_p"]) if eligible else None
    return {"cap_fF": cap, "fr_ghz": fr, "c12_fF": c12, "asymmetry": asymmetry,
            "device": asdict(d), "points": points, "best": best,
            "matched_intervals": intervals(points,"matched"),
            "good_intervals": intervals(points,"good")}


def main():
    out = Path(__file__).resolve().parent/"data"/"dispersive_study"
    out.mkdir(parents=True,exist_ok=True)
    cases = []
    with threadpool_limits(limits=1):
        for cap in (6.,10.,15.):
            for fr in (5.5,5.99,6.5,7.):
                for c12 in (1.,2.3,4.):
                    case = run_case(cap,fr,c12)
                    cases.append(case)
                    best = case["best"]
                    print(f"cap={cap:g} fr={fr:g} C12={c12:g}: minR={best['abs_delta_chi_over_chi_p']:.5g} at {best['operating_frequency_ghz']:.4f} GHz; good={case['good_intervals']}",flush=True)
                    (out/"scan.json").write_text(json.dumps(cases,indent=2),encoding="utf-8")
        for cap in (6.,10.):
            for asym in (.8,.9,1.1,1.2):
                case = run_case(cap,5.99,2.3,asym)
                cases.append(case)
                best=case["best"]
                print(f"cap={cap:g} asym={asym:g}: minR={best['abs_delta_chi_over_chi_p']:.5g}; good={case['good_intervals']}",flush=True)
                (out/"scan.json").write_text(json.dumps(cases,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
