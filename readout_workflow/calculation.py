"""Batch device defaults and nominal readout acceptance/search equations.

Units: physical capacitances fF; inductance nH; energies E/h and frequencies
GHz. Matrices represent H/h. Both capacitive loading and interactions come
from the SAME inverse capacitance matrix; no junction parasitic C is added.

Circuit construction, diagonalization and basic observables live in model.py.
Reading order here: Conditions -> batch Device defaults -> additional diagnostics
-> nominal search -> acceptance inequalities. Existing defaults are preserved.

Operation frequency is the mean DRESSED logical transition frequency. Search
coordinates use local uncoupled frequencies only as numerical EJ coordinates.
H is normalized charge-transition balance, not full-model state fidelity.
The sign grid rejects resonance-sensitive positions; it does not model MIST.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.constants import elementary_charge as e
from scipy.constants import h
from scipy.linalg import eigh
from scipy.optimize import brentq, least_squares
from threadpoolctl import threadpool_limits

from core.transmon import Basis, transmon_spectrum
from .model import Device as PairDevice, ReadoutModel as PairReadoutModel

# Existing callers may still import the historical private helper.
_transmon = transmon_spectrum

BARE_RESONATOR_GHZ = 6.0

MODE_IMPEDANCE_OHM = 50.0

CAPS = ("C1_shunt_fF", "C2_shunt_fF", "C12_fF", "C1r_fF", "C2r_fF")

PARAMETERS = (*CAPS, "bare_resonator_frequency")

CHARGE_CONSTANT = e**2 / (2 * h) * 1e6  # EC/h [GHz] = constant / C_eff [fF].


@dataclass(frozen=True)
class Conditions:
    """All acceptance thresholds; experimental tolerances are not solver margins."""

    mismatch_max: float = 0.01
    chi_ratio_definition: str = "absolute_mean"
    abs_mean_chi_min_mhz: float = 1.0  # Inclusive; signed mean before absolute value.
    dispersive_max: float = 0.1
    hybridization_min: float = 0.95
    reference_overlap_min: float = 0.95
    operation_frequency_ghz: tuple[float, float] = (4.0, 8.0)
    # Keep the dressed logical mean below the dressed resonator transition,
    # intended to suppress measurement-induced state transitions (MIST).
    # This is a design restriction; this static model does not simulate MIST.
    operation_below_readout: bool = True
    # Purpose: avoid operating positions too close to qubit-resonator resonances.
    # Reject resonance-sensitive candidates if either conditional shift changes
    # sign on an independent +/-1% EJ1/EJ2 neighborhood. This finite grid is an
    # extra resonance diagnostic alongside the multilevel dispersive screen.
    chi_sign_stability_required: bool = True
    chi_sign_ej_fraction: float = 0.01
    chi_sign_samples_per_axis: int = 5
    chi_sign_zero_tolerance_ghz: float = 1e-9  # 1 Hz: unresolved sign near zero.
    logical_frequency_ghz: tuple[float, float] = (0.09, 0.11)
    loaded_ec_mhz: tuple[float, float] = (150.0, 400.0)  # Upper bound strict.
    ej_upper_ghz: float = 25.0  # Strict.
    ej_over_ec_min: float = 50.0  # Strict.
    c12_bounds_ff: tuple[float, float] = (0.005, 15.0)
    readout_cap_bounds_ff: tuple[float, float] = (0.1, 25.0)
    fixed_frequency_tolerance_ghz: float = 1e-7  # 100 Hz; numerical tolerance.


CONDITIONS = Conditions()

BASELINE = dict(
    ec_min=CONDITIONS.loaded_ec_mhz[0],
    H=CONDITIONS.hybridization_min,
    logical_frequency=CONDITIONS.logical_frequency_ghz,
    ej_ec=CONDITIONS.ej_over_ec_min,
    fixed=False,
)

SEARCH_EFFORT = {
    # Number of starts, maximum least-squares evaluations, wall seconds/pixel.
    "fast": (3, 35, 25.0),
    "normal": (9, 65, 90.0),
    "deep": (24, 110, 240.0),
}


@dataclass(frozen=True)
class Device(PairDevice):
    """Batch-search defaults for the shared pair circuit (not notebook defaults)."""

    C1_shunt_fF: float = 87.0
    C2_shunt_fF: float = 73.0
    C12_fF: float = 1.7691402536230083
    C1r_fF: float = 5.110135439087974
    C2r_fF: float = 5.110135439087974
    Cr_fF: float = 530.5164769729845
    Lr_nH: float = 1.3262911924324612


class ReadoutModel(PairReadoutModel):
    """Shared circuit evaluator with the existing batch default Device."""

    def __init__(self, device=None, basis=None):
        super().__init__(Device() if device is None else device, basis)


SCREENING_BASIS = Basis()

VALIDATION_BASIS = Basis(20, 9, 8, 40)

CONVERGENCE_BASIS = Basis(25, 12, 10, 60)


def diagnostics(model, point):
    """Screen ALL upward isolated-pair transitions out of g,m,p, not just f01.

    The worst |g_ij/Delta_ij| can involve the two-excitation manifold. A small
    local g/Delta alone is insufficient for the chi calculation used here.
    Both g_ij and Delta_ij use GHz, so their ratio is dimensionless.
    """
    b = model.basis
    en1, n1 = transmon_spectrum(point["ej1_ghz"], model.ec_ghz[0], b)
    en2, n2 = transmon_spectrum(point["ej2_ghz"], model.ec_ghz[1], b)
    n1p, n2p = (
        np.kron(n1, np.eye(b.transmon_levels)),
        np.kron(np.eye(b.transmon_levels), n2),
    )
    hp = np.diag((en1[:, None] + en2[None, :]).ravel())
    hp += model.J_ghz[0, 1] * (n1p @ n2p)
    energy, vectors = eigh(hp)
    coupling = (
        2
        * e
        * model.qr_zpf
        / (h * 1e9)
        * (model.cinv[0, 2] * n1p + model.cinv[1, 2] * n2p)
    )
    gij = np.abs(vectors.T @ coupling @ vectors)
    ratios = []
    for i in range(3):
        for j in range(i + 1, len(energy)):
            if gij[i, j] < 1e-9:
                continue
            detuning = abs(energy[j] - energy[i] - model.fr_ghz)
            ratios.append((gij[i, j] / detuning if detuning > 0 else np.inf, i, j))
    worst = max(ratios) if ratios else (0.0, 0, 0)
    return {
        "lambda_max": float(worst[0]),
        "worst_pair_initial": int(worst[1]),
        "worst_pair_final": int(worst[2]),
        "lambda_ground": float(
            max(gij[0, j] / abs(energy[j] - energy[0] - model.fr_ghz) for j in (1, 2))
        ),
        "local_lambda1": point["g1r_over_2pi_mhz"]
        * 1e-3
        / abs(point["bare_f1_ghz"] - model.fr_ghz),
        "local_lambda2": point["g2r_over_2pi_mhz"]
        * 1e-3
        / abs(point["bare_f2_ghz"] - model.fr_ghz),
        "ratio_m": abs(point["delta_chi_ghz"] / point["chi_m_ghz"])
        if point["chi_m_ghz"]
        else np.inf,
    }


def make_model(c1, c2, c12, cg, basis=SCREENING_BASIS):
    """Construct a nominal pixel with equal readout capacitances and bare LC."""
    omega = 2 * np.pi * BARE_RESONATOR_GHZ * 1e9
    return ReadoutModel(
        Device(
            C1_shunt_fF=c1,
            C2_shunt_fF=c2,
            C12_fF=c12,
            C1r_fF=cg,
            C2r_fF=cg,
            Cr_fF=1 / (omega * MODE_IMPEDANCE_OHM) * 1e15,
            Lr_nH=MODE_IMPEDANCE_OHM / omega * 1e9,
        ),
        basis,
    )


def annotate(m, ej1, ej2):
    """Attach loaded ECs, EJ/EC, the physical inputs and worst chi ratio."""
    p = m.evaluate(float(ej1), float(ej2))
    p.update(diagnostics(m, p))
    p.update(
        device=asdict(m.device),
        basis=asdict(m.basis),
        loaded_ec1_mhz=float(1000 * m.ec_ghz[0]),
        loaded_ec2_mhz=float(1000 * m.ec_ghz[1]),
        ej1_over_ec1=float(ej1 / m.ec_ghz[0]),
        ej2_over_ec2=float(ej2 / m.ec_ghz[1]),
    )
    p["ratio"] = chi_ratio(p)
    return p


def check_chi_sign_stability(point, model=None, *, early_exit=False):
    """Attach a sign-stability certificate at fixed fabricated components.

    Purpose: avoid operating positions too close to qubit-resonator resonances,
    using chi sign sensitivity to small EJ changes as an additional diagnostic.
    Probe both EJs independently on a 5x5 grid: -1%, -0.5%, 0, +0.5%, +1%.
    Each logical branch must retain its own nominal chi sign. Probe reference
    overlaps must remain >=95%; near-zero/nonfinite shifts are rejected rather
    than assigned a trustworthy sign. No EJ retuning or full feasibility test
    is performed at the probes. Endpoints alone cannot rule out sign recrossing,
    and a finite grid cannot certify the absence of every narrow resonance.
    """
    c = CONDITIONS
    if not c.chi_sign_stability_required:
        return dict(point)
    if chi_sign_certificate_valid(point):
        return dict(point)
    if c.chi_sign_samples_per_axis < 3 or c.chi_sign_samples_per_axis % 2 != 1:
        raise ValueError("Use an odd chi-sign grid size of at least three.")
    if not 0 < c.chi_sign_ej_fraction < 1 or c.chi_sign_zero_tolerance_ghz < 0:
        raise ValueError("Invalid chi-sign perturbation or zero tolerance.")
    probe_basis = point.get("chi_sign_probe_basis", point["basis"])
    if model is None:
        model = ReadoutModel(Device(**point["device"]), Basis(**probe_basis))
    if asdict(model.device) != point["device"] or asdict(model.basis) != probe_basis:
        raise ValueError("Sign probes must use the fixed device and recorded probe basis.")
    nominal = np.array([point["chi_m_ghz"], point["chi_p_ghz"]], dtype=float)
    nominal_ok = bool(
        np.all(np.isfinite(nominal))
        and np.all(np.abs(nominal) > c.chi_sign_zero_tolerance_ghz)
    )
    offsets = np.linspace(
        -c.chi_sign_ej_fraction, c.chi_sign_ej_fraction, c.chi_sign_samples_per_axis
    )
    probes = []
    with threadpool_limits(limits=1):
        stopped = False
        for delta1 in offsets:
            if stopped:
                break
            for delta2 in offsets:
                if delta1 == 0 and delta2 == 0:
                    continue
                ej1 = point["ej1_ghz"] * (1 + float(delta1))
                ej2 = point["ej2_ghz"] * (1 + float(delta2))
                q = model.evaluate(ej1, ej2)
                chi = np.array([q["chi_m_ghz"], q["chi_p_ghz"]], dtype=float)
                resolved = bool(
                    np.all(np.isfinite(chi))
                    and np.all(np.abs(chi) > c.chi_sign_zero_tolerance_ghz)
                )
                reliable = bool(
                    q["min_assignment_overlap"] >= c.reference_overlap_min - 1e-8
                )
                same = bool(
                    nominal_ok and resolved and np.all(np.sign(chi) == np.sign(nominal))
                )
                probes.append(
                    dict(
                        ej1_fraction=float(delta1),
                        ej2_fraction=float(delta2),
                        ej1_ghz=ej1,
                        ej2_ghz=ej2,
                        chi_m_ghz=float(chi[0]),
                        chi_p_ghz=float(chi[1]),
                        min_assignment_overlap=float(q["min_assignment_overlap"]),
                        resolved_sign=resolved,
                        reliable_assignment=reliable,
                        same_sign=same,
                    )
                )
                if early_exit and not (same and reliable):
                    stopped = True
                    break
    certificate = dict(
        schema="chi-sign-grid-v2" if probe_basis != point["basis"] else "chi-sign-grid-v1",
        probe_basis=dict(probe_basis),
        grid_complete=len(probes) == c.chi_sign_samples_per_axis**2 - 1,
        passed=bool(
            nominal_ok
            and all(p["same_sign"] and p["reliable_assignment"] for p in probes)
        ),
        ej_fraction=c.chi_sign_ej_fraction,
        samples_per_axis=c.chi_sign_samples_per_axis,
        zero_tolerance_ghz=c.chi_sign_zero_tolerance_ghz,
        reference_overlap_min=c.reference_overlap_min,
        device=dict(point["device"]),
        basis=dict(point["basis"]),
        ej1_ghz=point["ej1_ghz"],
        ej2_ghz=point["ej2_ghz"],
        nominal_chi_m_ghz=float(nominal[0]),
        nominal_chi_p_ghz=float(nominal[1]),
        nominal_sign_resolved=nominal_ok,
        probe_count=len(probes),
        probes=probes,
    )
    return dict(point, chi_sign_stability=certificate)


def chi_sign_certificate_valid(point):
    """Invalidate a saved certificate when parameters, basis or policy change."""
    d = point.get("chi_sign_stability")
    c = CONDITIONS
    return bool(
        d
        and d.get("schema") in ("chi-sign-grid-v1", "chi-sign-grid-v2")
        and d.get("probe_basis", d["basis"]) == point.get("chi_sign_probe_basis", point["basis"])
        and d["device"] == point["device"]
        and d["basis"] == point["basis"]
        and d["ej1_ghz"] == point["ej1_ghz"]
        and d["ej2_ghz"] == point["ej2_ghz"]
        and d["ej_fraction"] == c.chi_sign_ej_fraction
        and d["samples_per_axis"] == c.chi_sign_samples_per_axis
        and d["zero_tolerance_ghz"] == c.chi_sign_zero_tolerance_ghz
        and d["reference_overlap_min"] == c.reference_overlap_min
        and d["nominal_chi_m_ghz"] == point["chi_m_ghz"]
        and d["nominal_chi_p_ghz"] == point["chi_p_ghz"]
    )


def inverse_frequency(frequency, ec, basis):
    """Invert the numerically diagonalized local f01, not sqrt(8 EJ EC)-EC."""
    return brentq(
        lambda ej: transmon_spectrum(ej, ec, basis)[0][1] - frequency, 0.02, 100.0, xtol=2e-10
    )


def point_from_x(c1, c2, x, basis=SCREENING_BASIS):
    """Map local-frequency optimization coordinates to a full-circuit witness."""
    fmean, detune_units, c12, cg = map(float, x)
    m = make_model(c1, c2, c12, cg, basis)
    detune = 0.01 * detune_units
    ej1 = inverse_frequency(fmean + detune / 2, m.ec_ghz[0], basis)
    ej2 = inverse_frequency(fmean - detune / 2, m.ec_ghz[1], basis)
    p = m.evaluate(ej1, ej2)
    p.update(diagnostics(m, p))
    ec = m.ec_ghz[:2]
    p.update(
        device=asdict(m.device),
        basis=asdict(basis),
        x=list(map(float, x)),
        loaded_ec1_mhz=float(ec[0] * 1000),
        loaded_ec2_mhz=float(ec[1] * 1000),
        ej1_over_ec1=float(ej1 / ec[0]),
        ej2_over_ec2=float(ej2 / ec[1]),
        participation1=float(c1 * 1e-15 * m.cinv[0, 0]),
        participation2=float(c2 * 1e-15 * m.cinv[1, 1]),
    )
    p["ratio"] = chi_ratio(p)
    return p


def x_from_point(p):
    """Recover the four nominal-search coordinates from an archived witness."""
    return [
        (p["bare_f1_ghz"] + p["bare_f2_ghz"]) / 2,
        (p["bare_f1_ghz"] - p["bare_f2_ghz"]) / 0.01,
        p["device"]["C12_fF"],
        p["device"]["C1r_fF"],
    ]


def nominal_search_bounds():
    """Optimization box is wider than the dressed-frequency acceptance window."""
    c = CONDITIONS
    return (
        [
            c.operation_frequency_ghz[0] - 0.2,
            -15,
            c.c12_bounds_ff[0],
            c.readout_cap_bounds_ff[0],
        ],
        [
            c.operation_frequency_ghz[1] + 0.2,
            15,
            c.c12_bounds_ff[1],
            c.readout_cap_bounds_ff[1],
        ],
    )


def seed_candidates(c1, c2, warm=()):
    """Try both sides of the resonator and both signs of differential detuning.

    Analytic transmon estimates only choose initial guesses; every accepted
    point is diagonalized and checked with its actual inverse C matrix.
    """
    seeds = []
    for p in warm:
        if p is not None:
            seeds.append(np.array(p.get("x", x_from_point(p)), float))
    # Prioritize both sides of the resonator; actual energy windows provide
    # additional seeds rather than eliminating points on approximate bounds.
    windows = []
    for cg in (1.0, 4.0, 8.0):
        m = make_model(c1, c2, 2.0, cg)
        ec = m.ec_ghz[:2]
        # Actual local spectra give plausible EJ/EC and EJ energy windows.
        # These add seeds, never remove a pixel from the nominal map.
        lower = max(
            CONDITIONS.operation_frequency_ghz[0] + 0.02,
            *(
                transmon_spectrum(
                    (CONDITIONS.ej_over_ec_min + 0.05) * value, value, SCREENING_BASIS
                )[0][1]
                for value in ec
            ),
        )
        upper = min(
            CONDITIONS.operation_frequency_ghz[1] - 0.02,
            *(
                transmon_spectrum(CONDITIONS.ej_upper_ghz - 0.01, value, SCREENING_BASIS)[0][1]
                for value in ec
            ),
        )
        if lower < upper:
            windows.extend(
                (
                    lower + 0.1 * (upper - lower),
                    (lower + upper) / 2,
                    upper - 0.1 * (upper - lower),
                )
            )
    freqs = list(
        dict.fromkeys(round(f, 5) for f in ([4.4, 7.5, 4.05, 7.95, 5.0, 6.8] + windows))
    )
    for f in freqs:
        m = make_model(c1, c2, 2.0, 4.0)
        ec = m.ec_ghz[:2]
        if any(
            (f + v) ** 2 / (8 * v) > 29 or (f + v) ** 2 / (8 * v) / v < 40 for v in ec
        ):
            continue
        ej = [(f + v) ** 2 / (8 * v) for v in ec]
        p = m.evaluate(*ej)
        p.update(diagnostics(m, p))
        cg = float(np.clip(4 * 0.075 / max(p["lambda_max"], 0.005), 0.15, 18.0))
        ceff = np.sqrt((CHARGE_CONSTANT / ec[0]) * (CHARGE_CONSTANT / ec[1]))
        c12 = float(
            np.clip(
                0.1 * ceff / f - cg * cg / m.device.Cr_fF, *CONDITIONS.c12_bounds_ff
            )
        )
        # Include differential detuning in both directions; it is not fixed to
        # the resonance minimum or to exact chi matching.
        for detune in (0.0, 0.5, -0.5):
            seeds.append(np.array([f, detune, c12, cg]))
    unique = []
    for seed in seeds:
        seed = np.clip(seed, *nominal_search_bounds())
        if not any(np.linalg.norm(seed - prior) < 1e-5 for prior in unique):
            unique.append(seed)
    return unique


def search(task):
    """Bounded multistart inequality search; absence of a witness is not a proof."""
    c1, c2 = map(float, task["pair"])
    if not c1 > c2:
        raise ValueError("Search is restricted to strict C1>C2")
    level = task.get("level", "normal")
    basis = Basis(**task["basis"]) if "basis" in task else Basis()
    limits = SEARCH_EFFORT
    nstarts, max_nfev, max_seconds = limits[level]
    started = time.monotonic()
    attempts = []
    best = None
    best_pass = None
    evals = 0
    with threadpool_limits(limits=1):
        seeds = seed_candidates(c1, c2, task.get("warm", ()))
        # Cold and nearby-solution seeds are interleaved so upper-band starts
        # are tested before spending all trials in one basin.
        if len(task.get("warm", ())) and len(seeds) > len(task["warm"]) + 3:
            nw = min(len(task["warm"]), len(seeds))
            ordered = []
            for i in range(max(nw, len(seeds) - nw)):
                if i < nw:
                    ordered.append(seeds[i])
                if nw + i < len(seeds):
                    ordered.append(seeds[nw + i])
            seeds = ordered
        cache = {}

        def evaluate(x):
            nonlocal best, best_pass, evals
            key = tuple(map(float, x))
            if key not in cache:
                p = point_from_x(c1, c2, x, basis)
                evals += 1
                cache[key] = p
                if best is None or quality(p) > quality(best):
                    best = p
                if not failures(p) and (
                    best_pass is None or quality(p) > quality(best_pass)
                ):
                    best_pass = p
            return cache[key]

        for index, seed in enumerate(seeds[:nstarts]):
            if index and time.monotonic() - started > max_seconds:
                break
            margin = 0.5 if level != "deep" else (0.2 if index % 2 == 0 else 0.0)
            try:
                sol = least_squares(
                    lambda x, margin=margin: np.nan_to_num(
                        violations(evaluate(x), margin), nan=1e3, posinf=1e3
                    ),
                    seed,
                    bounds=nominal_search_bounds(),
                    x_scale=[1.0, 1.0, 2.0, 5.0],
                    diff_step=2e-5,
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                    max_nfev=max_nfev,
                )
                p = evaluate(sol.x)
                attempts.append(
                    {
                        "seed": seed.tolist(),
                        "x": sol.x.tolist(),
                        "cost": float(sol.cost),
                        "nfev": int(sol.nfev),
                        "failures": failures(p),
                        "ratio": p["ratio"],
                        "H": p["hybridization"],
                        "lambda": p["lambda_max"],
                        "logical_frequency_ghz": p["splitting_ghz"],
                    }
                )
            except (ValueError, FloatingPointError) as exc:
                attempts.append({"seed": seed.tolist(), "error": str(exc)})
            if best_pass is not None:
                best_pass = check_chi_sign_stability(best_pass, early_exit=True)
                if not failures(best_pass, require_sign_check=True):
                    break
                attempts[-1]["chi_sign_rejection"] = best_pass["chi_sign_stability"]
                best_pass = None
    selected = best_pass or best
    return {
        "c1": c1,
        "c2": c2,
        "level": level,
        "status": "screened_pass" if best_pass else "no_witness_found",
        "point": selected,
        "failures": failures(selected, require_sign_check=True)
        if selected
        else ["no_evaluation"],
        "attempts": attempts,
        "evaluations": evals,
        "elapsed_seconds": time.monotonic() - started,
    }


def validate(result, basis=None):
    """Shared adaptive validation; an explicit basis requests a fixed audit.

    Lazy import avoids the calculation/validation module dependency cycle.
    Physical device inputs and EJs remain fixed in both paths.
    """
    p = result["point"]
    with threadpool_limits(limits=1):
        if basis is None:
            from .validation import audit_point
            q = audit_point(p)
        else:
            m = ReadoutModel(Device(**p["device"]), basis)
            q = annotate(m, p["ej1_ghz"], p["ej2_ghz"])
            q["x"] = x_from_point(q)
            if not failures(q):
                q = check_chi_sign_stability(q, m, early_exit=True)
        for name in ("participation1", "participation2"):
            if name in p:
                q[name] = p[name]
    out = dict(result)
    out["screening_point"] = p
    out["point"] = q
    out["failures"] = failures(q, require_sign_check=True)
    out["status"] = "validated_pass" if not out["failures"] else "validation_failed"
    return out


def perturb(reference, errors):
    """Independent relative errors in five capacitances and bare mode frequency.

    Each error lies in [-0.02,+0.02]. C1r and C2r can differ after fabrication.
    Divide Cr and Lr by the frequency factor: fr changes, sqrt(Lr/Cr) does not.
    """
    errors = np.asarray(errors, dtype=float)
    if (
        errors.shape != (len(PARAMETERS),)
        or not np.all(np.isfinite(errors))
        or np.any(errors <= -1)
    ):
        raise ValueError(
            "Supply six finite fractional errors with positive component/frequency factors."
        )
    d = Device(**reference["device"])
    edits = {
        key: getattr(d, key) * (1 + float(error))
        for key, error in zip(CAPS, errors[: len(CAPS)], strict=True)
    }
    scale = 1 + float(errors[5])
    # Both LC components scale inversely with frequency, preserving sqrt(L/C).
    return replace(d, **edits, Cr_fF=d.Cr_fF / scale, Lr_nH=d.Lr_nH / scale)


def chi_ratio(point):
    """Current signed-mean denominator; legacy policy is explicit for archives."""
    delta = abs(point["chi_p_ghz"] - point["chi_m_ghz"])
    if CONDITIONS.chi_ratio_definition == "absolute_mean":
        denominator = abs((point["chi_p_ghz"] + point["chi_m_ghz"]) / 2)
    elif CONDITIONS.chi_ratio_definition == "worst_branch":
        denominator = min(abs(point["chi_p_ghz"]), abs(point["chi_m_ghz"]))
    else:
        raise ValueError("Unknown chi-ratio definition")
    return float(delta / denominator) if denominator else float("inf")


def absolute_mean_chi_mhz(point):
    """Readout strength: magnitude of the signed mean shift, in MHz."""
    return float(1000 * abs((point["chi_p_ghz"] + point["chi_m_ghz"]) / 2))


def violations(point, margin=0.0):
    """Nonnegative scaled violations; zero means feasible before strict checks."""
    c = CONDITIONS
    logical = point["splitting_ghz"]
    operation = point["operating_frequency_ghz"]
    ec = np.array([point["loaded_ec1_mhz"], point["loaded_ec2_mhz"]]) / 1000
    return np.array(
        [
            max(0, point["ratio"] - (c.mismatch_max - margin * 0.001)) / c.mismatch_max,
            max(0, c.abs_mean_chi_min_mhz - absolute_mean_chi_mhz(point)),
            max(0, c.hybridization_min + margin * 0.002 - point["hybridization"])
            / 0.05,
            max(0, point["lambda_max"] - (c.dispersive_max - margin * 0.002))
            / c.dispersive_max,
            max(
                0,
                c.reference_overlap_min
                + margin * 0.002
                - point["min_assignment_overlap"],
            )
            / 0.05,
            max(0, c.logical_frequency_ghz[0] + margin * 0.0002 - logical) / 0.01,
            max(0, logical - c.logical_frequency_ghz[1] + margin * 0.0002) / 0.01,
            max(0, c.operation_frequency_ghz[0] + margin * 0.001 - operation),
            max(0, operation - c.operation_frequency_ghz[1] + margin * 0.001),
            max(0, operation - point["dressed_r_ghz"] + margin * 0.001)
            if c.operation_below_readout
            else 0.0,
            *(
                max(0, c.loaded_ec_mhz[0] / 1000 + margin * 0.0001 - value) / 0.1
                for value in ec
            ),
            *(
                max(0, value - c.loaded_ec_mhz[1] / 1000 + margin * 0.0001) / 0.1
                for value in ec
            ),
            *(
                max(0, point[f"ej{i}_ghz"] - c.ej_upper_ghz + margin * 0.01) / 10
                for i in (1, 2)
            ),
            *(
                max(0, c.ej_over_ec_min + margin * 0.02 - point[f"ej{i}_over_ec{i}"])
                / 50
                for i in (1, 2)
            ),
        ]
    )


def failures(point, *, require_sign_check=False):
    """Nominal map acceptance, including STRICT EC upper/EJ upper/EJ/EC bounds."""
    return reasons(
        point,
        BASELINE,
        point["operating_frequency_ghz"],
        require_sign_check=require_sign_check,
    )


def quality(point):
    """Rank feasible witnesses by safety margin; no exact chi equality target."""
    if failures(point):
        return (-1.0, -float(np.linalg.norm(violations(point))), point["hybridization"])
    c = CONDITIONS
    margins = [
        (c.mismatch_max - point["ratio"]) / c.mismatch_max,
        (point["hybridization"] - c.hybridization_min) / 0.05,
        (c.dispersive_max - point["lambda_max"]) / c.dispersive_max,
        (point["splitting_ghz"] - c.logical_frequency_ghz[0]) / 0.02,
        (c.logical_frequency_ghz[1] - point["splitting_ghz"]) / 0.02,
    ]
    return (1.0, float(min(margins)), point["hybridization"])


def reasons(point, tier, target, *, require_sign_check=False):
    """List failed constraints; include_H is never silently disabled here."""
    c = CONDITIONS
    low, high = tier["logical_frequency"]
    ec1, ec2 = point["loaded_ec1_mhz"], point["loaded_ec2_mhz"]
    checks = {
        "chi_matching": point["ratio"] <= c.mismatch_max + 1e-8,
        "absolute_mean_chi": absolute_mean_chi_mhz(point) >= c.abs_mean_chi_min_mhz,
        "dispersive": point["lambda_max"] <= c.dispersive_max + 1e-9,
        "hybridization": point["hybridization"] >= tier["H"] - 1e-8,
        "logical_frequency": low - 1e-8 <= point["splitting_ghz"] <= high + 1e-8,
        "EC_range": min(ec1, ec2) >= tier["ec_min"] - 1e-7
        and max(ec1, ec2) < c.loaded_ec_mhz[1],
        "EJ_limit": max(point["ej1_ghz"], point["ej2_ghz"]) < c.ej_upper_ghz,
        "EJ_over_EC": min(point["ej1_over_ec1"], point["ej2_over_ec2"]) > tier["ej_ec"],
        "assignment_overlap": point["min_assignment_overlap"]
        >= c.reference_overlap_min - 1e-8,
        "operation_frequency": c.operation_frequency_ghz[0] - 1e-8
        <= point["operating_frequency_ghz"]
        <= c.operation_frequency_ghz[1] + 1e-8,
    }
    if tier["fixed"]:
        checks["fixed_operation_frequency"] = (
            abs(point["operating_frequency_ghz"] - target)
            <= c.fixed_frequency_tolerance_ghz
        )
    if c.operation_below_readout:
        checks["operation_below_readout"] = (
            point["operating_frequency_ghz"] < point["dressed_r_ghz"]
        )
    if c.chi_sign_stability_required:
        if point.get("chi_sign_stability") is not None:
            checks["chi_sign_stability"] = (
                chi_sign_certificate_valid(point)
                and point["chi_sign_stability"]["passed"]
            )
        elif require_sign_check:
            checks["chi_sign_stability_unchecked"] = False
    return [name for name, passed in checks.items() if not passed]


def slacks(point, tier, include_H=False, margin=0.0):
    """Positive values satisfy recalibration inequalities; EC is screened first.

    H is the objective during constrained polishing. It is checked against the
    tier floor on acceptance, rather than constrained to its maximum or H=1.
    """
    c = CONDITIONS
    values = [
        (c.mismatch_max - margin * 1e-6 - point["ratio"]) / c.mismatch_max,
        absolute_mean_chi_mhz(point) - c.abs_mean_chi_min_mhz,
        (c.dispersive_max - margin * 1e-7 - point["lambda_max"]) / c.dispersive_max,
        (point["min_assignment_overlap"] - c.reference_overlap_min - margin * 1e-7)
        / 0.05,
        (point["splitting_ghz"] - tier["logical_frequency"][0] - margin * 1e-7) / 0.01,
        (tier["logical_frequency"][1] - margin * 1e-7 - point["splitting_ghz"]) / 0.01,
        point["operating_frequency_ghz"] - c.operation_frequency_ghz[0] - margin * 1e-7,
        c.operation_frequency_ghz[1] - margin * 1e-7 - point["operating_frequency_ghz"],
        *((c.ej_upper_ghz - margin * 1e-6 - point[f"ej{i}_ghz"]) / 10 for i in (1, 2)),
        *(
            (point[f"ej{i}_over_ec{i}"] - tier["ej_ec"] - margin * 1e-5) / 50
            for i in (1, 2)
        ),
    ]
    if include_H:
        values.append((point["hybridization"] - tier["H"]) / 0.05)
    if c.operation_below_readout:
        values.append(
            point["dressed_r_ghz"] - point["operating_frequency_ghz"] - margin * 1e-7
        )
    return np.nan_to_num(values, nan=-1e4, posinf=1e4, neginf=-1e4)


def search_and_validate(task):
    """Worker: multistart screening, unchanged-input validation, then basis repair."""
    result = search(task)
    if result["status"] != "screened_pass":
        return result
    checked = validate(result)
    if checked["status"] == "validated_pass":
        return checked
    # Repair numerical edge cases at the validation basis rather than coloring
    # an unconverged screening point as viable.
    repair = search(
        task
        | dict(basis=asdict(VALIDATION_BASIS), warm=[checked["point"]], level="normal")
    )
    return validate(repair) if repair["status"] == "screened_pass" else checked
