"""Resonator-inclusive fixed-hardware validation of reconciled XM-CZ candidates.

Node order is A1,A2,rA,B1,B2,rB.  One physical capacitance matrix is solved
per candidate and reused at idle and gate; only EJs change between endpoints.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.constants import elementary_charge as e, h, hbar
from scipy.linalg import eigh
from scipy.optimize import least_squares, minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from core.electrostatics import charge_coupling_ghz, loaded_charging_energies_ghz
from core.transmon import Basis, transmon_spectrum
from readout_workflow.storage import atomic_json


DATA = HERE / "data"
OUTPUT = DATA / "full_network_validation"
NODES_A = (0, 1, 2)
NODES_B = (3, 4, 5)
TRANSMONS = (0, 1, 3, 4)


def add_branch(matrix, left, right, value):
    matrix[left, left] += value
    matrix[right, right] += value
    matrix[left, right] -= value
    matrix[right, left] -= value


def capacitance_matrix(variables, witness_a, witness_b):
    """Return the physical six-node capacitance matrix in fF."""
    shunts = variables[:4]
    direct, c12_a, c12_b = variables[4:]
    da, db = witness_a["device"], witness_b["device"]
    matrix = np.zeros((6, 6), dtype=float)
    matrix[np.diag_indices(6)] = [
        shunts[0], shunts[1], da["Cr_fF"], shunts[2], shunts[3], db["Cr_fF"]
    ]
    for branch in (
        (0, 1, c12_a),
        (0, 2, da["C1r_fF"]),
        (1, 2, da["C2r_fF"]),
        (3, 4, c12_b),
        (3, 5, db["C1r_fF"]),
        (4, 5, db["C2r_fF"]),
        (1, 3, direct),
    ):
        add_branch(matrix, *branch)
    return matrix


def solve_hardware(candidate):
    wa = candidate["rail_a"]["point"]
    wb = candidate["rail_b"]["point"]
    target_ec = np.array([165, 240, *candidate["ec_b_mhz"]], dtype=float) / 1000
    target_j = np.array(
        [
            candidate["rail_a"]["target_J12_ghz"],
            candidate["rail_b"]["target_J12_ghz"],
            candidate["fixed_interrail_J_target_ghz"],
        ]
    )
    start = np.array(
        [
            wa["device"]["C1_shunt_fF"],
            wa["device"]["C2_shunt_fF"],
            wb["device"]["C1_shunt_fF"],
            wb["device"]["C2_shunt_fF"],
            0.5,
            wa["device"]["C12_fF"],
            wb["device"]["C12_fF"],
        ]
    )
    lower = np.array([1, 1, 1, 1, 1e-5, 0.005, 0.005])
    upper = np.array([200, 200, 200, 200, 10, 15, 15])

    def quantities(x):
        cap = capacitance_matrix(x, wa, wb) * 1e-15
        inverse = np.linalg.inv(cap)
        ec = loaded_charging_energies_ghz(inverse)[list(TRANSMONS)]
        j = charge_coupling_ghz(inverse)
        return cap, inverse, ec, np.array([j[0, 1], j[3, 4], j[1, 3]])

    def residual(x):
        try:
            cap, _, ec, coupling = quantities(x)
            if np.linalg.eigvalsh(cap).min() <= 0:
                return np.ones(7) * 1e3
            return np.r_[(ec - target_ec) / target_ec, (coupling - target_j) / target_j]
        except np.linalg.LinAlgError:
            return np.ones(7) * 1e3

    fit = least_squares(
        residual,
        np.clip(start, lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper),
        x_scale=[50, 50, 50, 50, 1, 2, 2],
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=1000,
    )
    cap, inverse, ec, coupling = quantities(fit.x)
    error = residual(fit.x)
    return {
        "success": bool(np.max(np.abs(error)) < 1e-7),
        "variables": {
            "shunt_A1_fF": fit.x[0], "shunt_A2_fF": fit.x[1],
            "shunt_B1_fF": fit.x[2], "shunt_B2_fF": fit.x[3],
            "direct_A2_B1_fF": fit.x[4], "C12_A_fF": fit.x[5], "C12_B_fF": fit.x[6],
        },
        "target_ec_ghz": target_ec.tolist(),
        "achieved_ec_ghz": ec.tolist(),
        "target_j_ghz": target_j.tolist(),
        "achieved_j_ghz": coupling.tolist(),
        "maximum_relative_target_error": float(np.max(np.abs(error))),
        "capacitance_matrix_fF": (cap * 1e15).tolist(),
        "inverse_capacitance_per_F": inverse.tolist(),
        "nfev": int(fit.nfev),
    }


def embed(operator, position, dims):
    result = np.ones((1, 1))
    for index, dimension in enumerate(dims):
        result = np.kron(result, operator if index == position else np.eye(dimension))
    return result


def local_block(cap_inverse, j, nodes, ejs, inductance_nh, levels, photons, keep):
    dims = (levels, levels, photons)
    ec_all = loaded_charging_energies_ghz(cap_inverse)
    energies, operators = [], []
    for position, (node, ej) in enumerate(zip(nodes[:2], ejs, strict=True)):
        energy, charge = transmon_spectrum(
            ej, ec_all[node], Basis(25, levels, max(photons, 3), max(6, levels))
        )
        energies.append(energy)
        operators.append(embed(charge, position, dims))
    resonator = nodes[2]
    cr_eff = 1 / cap_inverse[resonator, resonator]
    omega = 1 / np.sqrt(inductance_nh * 1e-9 * cr_eff)
    fr = omega / (2 * np.pi * 1e9)
    a = np.diag(np.sqrt(np.arange(1, photons)), 1)
    qr = np.sqrt(hbar * cr_eff * omega / 2) * (a + a.T) / (2 * e)
    energies.append(np.arange(photons) * fr)
    operators.append(embed(qr, 2, dims))
    ham = sum(embed(np.diag(value), index, dims) for index, value in enumerate(energies))
    for left in range(3):
        for right in range(left + 1, 3):
            ham += j[nodes[left], nodes[right]] * operators[left] @ operators[right]
    values, vectors = eigh(ham)
    values -= values[0]
    values, vectors = values[:keep], vectors[:, :keep]

    # Isolated-pair references define g,m,p; the oscillator adds r,mr,pr.
    pair_dims = (levels, levels)
    pair_ham = np.diag((energies[0][:, None] + energies[1][None, :]).ravel())
    n_left = embed(
        transmon_spectrum(ejs[0], ec_all[nodes[0]], Basis(25, levels, 3, max(6, levels)))[1],
        0,
        pair_dims,
    )
    n_right = embed(
        transmon_spectrum(ejs[1], ec_all[nodes[1]], Basis(25, levels, 3, max(6, levels)))[1],
        1,
        pair_dims,
    )
    pair_ham += j[nodes[0], nodes[1]] * n_left @ n_right
    _, pair_vectors = eigh(pair_ham)
    fock = np.eye(photons)
    refs = [
        np.kron(pair_vectors[:, rail], fock[:, photon])
        for rail, photon in ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
    ]
    overlap = np.abs(np.asarray(refs) @ vectors) ** 2
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-overlap)
    labels = dict(zip(("g", "m", "p", "r", "mr", "pr"), cols, strict=True))
    bare20 = np.zeros(np.prod(dims)); bare20[np.ravel_multi_index((2, 0, 0), dims)] = 1
    auxiliary = int(np.argmax(np.abs(bare20 @ vectors) ** 2))
    transformed = [vectors.T @ operator @ vectors for operator in operators]
    strengths = []
    ground, logical = vectors[:, labels["g"]], vectors[:, [labels["m"], labels["p"]]]
    for operator in operators[:2]:
        strengths.append(np.abs(ground @ operator @ logical) ** 2)
    strengths = np.asarray(strengths)
    participation = strengths / np.maximum(strengths.sum(axis=1, keepdims=True), 1e-30)
    hybridization = float(np.clip(1 - 2 * np.mean(np.abs(participation - .5)), 0, 1))
    return {
        "energies": values,
        "operators": transformed,
        "labels": labels,
        "auxiliary": auxiliary,
        "assignment_overlap": float(overlap[rows, cols].min()),
        "hybridization": hybridization,
        "resonator_frequency_ghz": float(fr),
    }


def full_hamiltonian(block_a, block_b, j):
    ka, kb = len(block_a["energies"]), len(block_b["energies"])
    ham = np.diag((block_a["energies"][:, None] + block_b["energies"][None, :]).ravel())
    interaction = np.zeros_like(ham)
    for ia, node_a in enumerate(NODES_A):
        for ib, node_b in enumerate(NODES_B):
            term = j[node_a, node_b] * np.kron(block_a["operators"][ia], block_b["operators"][ib])
            ham += term
            interaction += term
    return ham, interaction, (ka, kb)


def idle_readout(block_a, block_b, ham, ranges):
    labels_a, labels_b = block_a["labels"], block_b["labels"]
    requests = set()
    for side in (0, 1):
        own = labels_a if side == 0 else labels_b
        other = labels_b if side == 0 else labels_a
        for spectator in ("g", "m", "p"):
            for state in ("g", "m", "p", "r", "mr", "pr"):
                requests.add((own[state], other[spectator]) if side == 0 else (other[spectator], own[state]))
    values, vectors = eigh(ham)
    indices = [np.ravel_multi_index(ref, (len(block_a["energies"]), len(block_b["energies"]))) for ref in requests]
    overlap = np.abs(vectors[indices, :]) ** 2
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-overlap)
    lookup = dict(zip(requests, cols, strict=True))
    reports = []
    for side, (own, other, bounds, block) in enumerate(
        ((labels_a, labels_b, ranges[0], block_a), (labels_b, labels_a, ranges[1], block_b))
    ):
        for spectator in ("g", "m", "p"):
            def energy(state):
                ref = (own[state], other[spectator]) if side == 0 else (other[spectator], own[state])
                return values[lookup[ref]]
            ground = energy("g")
            m, p, r = energy("m"), energy("p"), energy("r")
            chi_m = energy("mr") - m - r + ground
            chi_p = energy("pr") - p - r + ground
            mean = (chi_m + chi_p) / 2
            mismatch = abs((chi_p - chi_m) / mean) if mean else np.inf
            splitting = abs(p - m)
            operation = (p + m) / 2 - ground
            resonator = r - ground
            passed = (
                abs(mean) >= .001 and mismatch <= .01 and bounds[0] <= splitting <= bounds[1]
                and operation < resonator and block["hybridization"] >= .95
                and block["assignment_overlap"] >= .95 and overlap[rows, cols].min() >= .95
            )
            reports.append({
                "rail": "A" if side == 0 else "B", "spectator": spectator,
                "chi_m_ghz": float(chi_m), "chi_p_ghz": float(chi_p),
                "abs_mean_chi_mhz": float(1000 * abs(mean)), "chi_mismatch": float(mismatch),
                "logical_frequency_mhz": float(1000 * splitting),
                "operation_frequency_ghz": float(operation), "resonator_frequency_ghz": float(resonator),
                "hybridization": block["hybridization"],
                "local_assignment_overlap": block["assignment_overlap"],
                "full_assignment_overlap": float(overlap[rows, cols].min()), "passed": bool(passed),
            })
    return reports


def updated_s1(block_a, block_b, interaction):
    la, lb = block_a["labels"], block_b["labels"]
    logical = [(la[a], lb[b]) for a in ("m", "p") for b in ("m", "p")]
    source = (la["m"], lb["p"])
    target = (la["g"], block_b["auxiliary"])
    dims = (len(block_a["energies"]), len(block_b["energies"]))
    energies = (block_a["energies"][:, None] + block_b["energies"][None, :]).ravel()
    scores, maxima = [], []
    for ref in logical:
        index = np.ravel_multi_index(ref, dims)
        mask = np.arange(len(energies)) != index
        if ref == source:
            mask[np.ravel_multi_index(target, dims)] = False
        detuning = np.abs(energies[mask] - energies[index])
        element = np.abs(interaction[index, mask])
        valid = (element > 1e-12) & (detuning > 1e-12)
        ratios = element[valid] / detuning[valid]
        scores.append(float(np.sum(ratios**2)))
        maxima.append(float(max(ratios, default=0)))
    intended = abs(interaction[np.ravel_multi_index(source, dims), np.ravel_multi_index(target, dims)])
    return {"S1": float(np.mean(scores)), "S1_by_input": scores,
            "maximum_ratio": max(maxima), "intended_matrix_element_mhz": float(1000 * intended)}


def gate_fidelity(block_a, block_b, ham, duration_ns):
    la, lb = block_a["labels"], block_b["labels"]
    refs = [(la[a], lb[b]) for a in ("m", "p") for b in ("m", "p")]
    dims = (len(block_a["energies"]), len(block_b["energies"]))
    indices = [np.ravel_multi_index(ref, dims) for ref in refs]
    values, vectors = eigh(ham)
    coefficients = vectors[indices, :].T
    matrix = coefficients.conj().T @ (np.exp(-2j * np.pi * values * duration_ns)[:, None] * coefficients)
    target = np.array([1, 1, 1, -1], complex)

    def trace_amplitude(phases):
        local = np.array([1, np.exp(1j * phases[1]), np.exp(1j * phases[0]), np.exp(1j * sum(phases))])
        return abs(np.sum(np.conj(target) * local * np.diag(matrix)))

    fit = minimize(lambda phases: -trace_amplitude(phases), np.zeros(2), method="Nelder-Mead")
    trace = trace_amplitude(fit.x)
    acceptance = float(np.trace(matrix.conj().T @ matrix).real / 4)
    raw = float((4 * acceptance + trace**2) / 20)
    return {
        "raw_fidelity": raw, "postselected_fidelity": raw / acceptance,
        "acceptance": acceptance, "local_z_phases_rad": fit.x.tolist(),
        "logical_projected_matrix_real": matrix.real.tolist(),
        "logical_projected_matrix_imag": matrix.imag.tolist(),
        "leakage_by_input": (1 - np.sum(abs(matrix)**2, axis=0)).real.tolist(),
    }


def matter_dispersive(inverse, j, ejs, inductances, levels=5, keep=8):
    ec = loaded_charging_energies_ghz(inverse)
    blocks = []
    for nodes, pair_ej in (((0, 1), ejs[:2]), ((3, 4), ejs[2:])):
        energies, charges = [], []
        for node, ej in zip(nodes, pair_ej, strict=True):
            en, charge = transmon_spectrum(ej, ec[node], Basis(25, levels, 3, max(6, levels)))
            energies.append(en); charges.append(charge)
        n1=np.kron(charges[0],np.eye(levels));n2=np.kron(np.eye(levels),charges[1])
        ham=np.diag((energies[0][:,None]+energies[1][None,:]).ravel())+j[nodes[0],nodes[1]]*n1@n2
        vals,vecs=eigh(ham);vals-=vals[0];vals,vecs=vals[:keep],vecs[:,:keep]
        blocks.append((vals,[vecs.T@n1@vecs,vecs.T@n2@vecs]))
    ea,oa=blocks[0];eb,ob=blocks[1];dims=(keep,keep)
    ham=np.diag((ea[:,None]+eb[None,:]).ravel())
    for ia,na in enumerate((0,1)):
        for ib,nb in enumerate((3,4)):ham+=j[na,nb]*np.kron(oa[ia],ob[ib])
    vals,vecs=eigh(ham)
    reports=[]
    for side,(res_node,inductance) in enumerate(zip((2,5),inductances,strict=True)):
        cr=1/inverse[res_node,res_node];omega=1/np.sqrt(inductance*1e-9*cr);fr=omega/(2*np.pi*1e9)
        qr=np.sqrt(hbar*cr*omega/2)/(2*e)
        operator=np.zeros_like(ham)
        for pos,node in enumerate((0,1)):operator+=j[node,res_node]*qr*np.kron(oa[pos],np.eye(keep))
        for pos,node in enumerate((3,4)):operator+=j[node,res_node]*qr*np.kron(np.eye(keep),ob[pos])
        dressed=vecs.T@operator@vecs
        # The lowest nine pair-product references cover g,m,p on both rails.
        refs=[np.ravel_multi_index((a,b),dims) for a in range(3) for b in range(3)]
        overlap=np.abs(vecs[refs,:])**2
        from scipy.optimize import linear_sum_assignment
        rows,cols=linear_sum_assignment(-overlap)
        initial=[cols[k] for k,(a,b) in enumerate(( (a,b) for a in range(3) for b in range(3) )) if (a>=1 if side==0 else b>=1)]
        ratios=[]
        for state in initial:
            transition=vals-vals[state];g=np.abs(dressed[state]);det=np.abs(transition-fr)
            mask=(transition>1e-9)&(g>1e-12);ratios.extend((g[mask]/np.maximum(det[mask],1e-15)).tolist())
        reports.append({"rail":"A" if side==0 else "B","dispersive_ratio":float(max(ratios,default=0)),
                        "resonator_frequency_ghz":float(fr),"assignment_overlap":float(overlap[rows,cols].min())})
    return reports


def validate_candidate(candidate, levels=5, photons=3, keep=12):
    hardware = solve_hardware(candidate)
    if not hardware["success"]:
        return {"status": "hardware_solve_failed", "hardware": hardware}
    cap = np.array(hardware["capacitance_matrix_fF"]) * 1e-15
    inverse = np.linalg.inv(cap);j=charge_coupling_ghz(inverse)
    wa,wb=candidate["rail_a"]["point"],candidate["rail_b"]["point"]
    inductances=(wa["device"]["Lr_nH"],wb["device"]["Lr_nH"])
    idle_ejs=(wa["ej1_ghz"],wa["ej2_ghz"],wb["ej1_ghz"],wb["ej2_ghz"])
    gate_ejs=None;gate_record=None
    for path in (DATA/"gate_central_output/points").glob("*.json"):
        record=json.loads(path.read_text());ec=tuple(round(1000*x) for x in record["payload"]["ec_GHz"][2:])
        if ec==tuple(candidate["ec_b_mhz"]):gate_record=record;break
    final=gate_record["result"]["final"];gate_ejs=tuple(final["EJ_GHz"])

    def blocks(ejs):
        a=local_block(inverse,j,NODES_A,ejs[:2],inductances[0],levels,photons,keep)
        b=local_block(inverse,j,NODES_B,ejs[2:],inductances[1],levels,photons,keep)
        ham,interaction,_=full_hamiltonian(a,b,j);return a,b,ham,interaction
    idle_a,idle_b,idle_ham,_=blocks(idle_ejs)
    readout=idle_readout(idle_a,idle_b,idle_ham,((.14,.16),(.08,.10)))
    dispersive=matter_dispersive(inverse,j,idle_ejs,inductances,levels=max(5,levels),keep=min(10,keep))
    gate_a,gate_b,gate_ham,interaction=blocks(gate_ejs)
    s1=updated_s1(gate_a,gate_b,interaction)
    fidelity=gate_fidelity(gate_a,gate_b,gate_ham,final["time_ns"])
    passed=(all(r["passed"] for r in readout) and all(r["dispersive_ratio"]<=.1 for r in dispersive)
            and fidelity["raw_fidelity"]>.95 and fidelity["postselected_fidelity"]>.999
            and s1["maximum_ratio"]<.2)
    return {"status":"validated_pass" if passed else "validation_failed","basis":{"transmon_levels":levels,"resonator_levels":photons,"retained_local_states":keep},
            "hardware":hardware,"idle_ej_ghz":idle_ejs,"gate_ej_ghz":gate_ejs,"readout":readout,
            "dispersive":dispersive,"updated_s1":s1,"gate":fidelity,"original_gate":{
                "raw_fidelity":final["raw_fidelity"],"postselected_fidelity":final["postselected_fidelity"],"time_ns":final["time_ns"]}}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rows=json.loads((DATA/"reconciliation/results.json").read_text())
    candidates=[r for r in rows if r["reduced_fixed_hardware_pass"]]
    results=[]
    for index,candidate in enumerate(candidates,1):
        ec=tuple(candidate["ec_b_mhz"]);path=OUTPUT/f"ec_{ec[0]}_{ec[1]}.json"
        if path.exists():result=json.loads(path.read_text())
        else:
            result={"ec_b_mhz":ec,"base":validate_candidate(candidate,5,3,12)}
            # Always perform the larger retained-basis validation for this final stage.
            result["larger_basis"]=validate_candidate(candidate,6,4,16)
            b=result["base"];l=result["larger_basis"]
            if b.get("gate") and l.get("gate"):
                result["convergence"]={
                    "raw_fidelity_abs_change":abs(b["gate"]["raw_fidelity"]-l["gate"]["raw_fidelity"]),
                    "postselected_fidelity_abs_change":abs(b["gate"]["postselected_fidelity"]-l["gate"]["postselected_fidelity"]),
                    "S1_abs_change":abs(b["updated_s1"]["S1"]-l["updated_s1"]["S1"]),
                }
            atomic_json(path,result)
        results.append(result);print(json.dumps({"completed":index,"ec":ec,"base":result["base"]["status"],"larger":result["larger_basis"]["status"]}),flush=True)
    summary={
        "candidates":len(results),
        "base_pass":sum(r["base"]["status"]=="validated_pass" for r in results),
        "larger_basis_pass":sum(r["larger_basis"]["status"]=="validated_pass" for r in results),
        "both_pass":sum(r["base"]["status"]==r["larger_basis"]["status"]=="validated_pass" for r in results),
        "physical_realization":"six-node A1,A2,rA,B1,B2,rB with direct A2-B1 capacitor",
    }
    atomic_json(OUTPUT/"results.json",results);atomic_json(OUTPUT/"summary.json",summary);print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
