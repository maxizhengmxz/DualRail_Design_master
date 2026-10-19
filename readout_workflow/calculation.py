"""Single-junction circuit model and nominal acceptance/search equations.

Units: physical capacitances fF; inductance nH; energies E/h and frequencies
GHz. Matrices represent H/h. Both capacitive loading and interactions come
from the SAME inverse capacitance matrix; no junction parasitic C is added.

Reading order: Conditions -> Device/Basis -> Hamiltonian and state diagnostics
-> nominal search -> acceptance inequalities. The source order preserves
function defaults and the equations used by the saved checkpoint.

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
from scipy.constants import h, hbar
from scipy.linalg import eigh
from scipy.optimize import brentq, least_squares, linear_sum_assignment
from threadpoolctl import threadpool_limits

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
class Device:
    """Physical circuit inputs; internal field names retain the archive schema."""

    # Physical capacitances, all in fF. No EC values disguised as capacitances.
    C1_shunt_fF: float = 87.0
    C2_shunt_fF: float = 73.0
    C12_fF: float = 1.7691402536230083
    C1r_fF: float = 5.110135439087974
    C2r_fF: float = 5.110135439087974
    Cr_fF: float = 530.5164769729845
    Lr_nH: float = 1.3262911924324612

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name in ("C1_shunt_fF", "C2_shunt_fF", "Cr_fF", "Lr_nH") and value == 0:
                raise ValueError(f"{name} must be positive")

    def capacitance_matrix(self):
        # Node order: transmon 1, transmon 2, resonator.
        # Diagonals sum attached capacitors; off-diagonals are negative mutual C.
        # No junction parasitic capacitances are added.
        c1, c2, c12, c1r, c2r, cr = (
            self.C1_shunt_fF,
            self.C2_shunt_fF,
            self.C12_fF,
            self.C1r_fF,
            self.C2r_fF,
            self.Cr_fF,
        )
        return 1e-15 * np.array(
            [
                [c1 + c12 + c1r, -c12, -c1r],
                [-c12, c2 + c12 + c2r, -c2r],
                [-c1r, -c2r, cr + c1r + c2r],
            ]
        )


@dataclass(frozen=True)
class Basis:
    """Charge cutoff, retained local levels, photons and tracked eigenlevels."""

    ncut: int = 15  # Charge states -ncut ... ncut, for each transmon.
    transmon_levels: int = 6  # Local eigenstates kept per transmon.
    resonator_levels: int = 5
    eigenlevels: int = 20  # Full-circuit eigenstates used for branch assignment.

    def __post_init__(self):
        if any(not isinstance(v, int) for v in asdict(self).values()):
            raise ValueError("Basis sizes must be integers")
        if self.ncut < 2 or not 3 <= self.transmon_levels <= 2 * self.ncut + 1:
            raise ValueError("Invalid transmon basis")
        if self.resonator_levels < 3:
            raise ValueError("Keep at least three resonator levels")
        if not 6 <= self.eigenlevels <= self.transmon_levels**2 * self.resonator_levels:
            raise ValueError("Invalid eigenlevel count")


def _transmon(ej_ghz, ec_ghz, basis):
    """Diagonalize 4 EC n^2 - EJ cos(phi), returning local energies and n."""
    if not np.isfinite(ej_ghz) or ej_ghz <= 0:
        raise ValueError("Josephson energies must be positive, in GHz")
    n = np.arange(-basis.ncut, basis.ncut + 1)
    # One ideal junction: -EJ*cos(phi); cos(phi) shifts charge by one Cooper pair.
    ham = np.diag(4 * ec_ghz * n * n) - ej_ghz / 2 * (
        np.diag(np.ones(len(n) - 1), 1) + np.diag(np.ones(len(n) - 1), -1)
    )
    energy, states = eigh(ham, subset_by_index=(0, basis.transmon_levels - 1))
    charge = states.T @ (n[:, None] * states)
    return energy - energy[0], charge


class ReadoutModel:
    """Exact diagonalization within a truncated transmon x transmon x Fock basis."""

    def __init__(self, device=None, basis=None):
        device = Device() if device is None else device
        basis = Basis() if basis is None else basis
        self.device, self.basis = device, basis
        self.cinv = np.linalg.inv(device.capacitance_matrix())
        # H_C = Q.T C^-1 Q / 2, Q_transmon = 2e*n.
        self.ec_ghz = e**2 * np.diag(self.cinv) / (2 * h * 1e9)
        self.cr_eff = 1 / self.cinv[2, 2]
        self.omega_r = 1 / np.sqrt(device.Lr_nH * 1e-9 * self.cr_eff)
        self.fr_ghz = self.omega_r / (2 * np.pi * 1e9)
        self.qr_zpf = np.sqrt(hbar * self.cr_eff * self.omega_r / 2)

    def evaluate(self, ej1_ghz, ej2_ghz):
        """Assign six distinct branches; extract conditional chi and charge balance.

        g,m,p are ground/lower/upper pair branches; r,mr,pr add one photon.
        chi_s = (E_sr-E_s) - (E_r-E_g), in GHz. This convention is the full
        conditional resonator-frequency difference, without an extra factor 2.
        The independent reference-overlap screen tests branch-label reliability.
        """
        b = self.basis
        en1, n1 = _transmon(ej1_ghz, self.ec_ghz[0], b)
        en2, n2 = _transmon(ej2_ghz, self.ec_ghz[1], b)
        it = np.eye(b.transmon_levels)
        ir = np.eye(b.resonator_levels)
        n1_pair, n2_pair = np.kron(n1, it), np.kron(it, n2)
        pair = np.diag((en1[:, None] + en2[None, :]).ravel())
        pair += 4 * e**2 * self.cinv[0, 1] / (h * 1e9) * (n1_pair @ n2_pair)
        _, pair_states = eigh(pair)
        a = np.diag(np.sqrt(np.arange(1, b.resonator_levels)), 1)
        qr = 1j * self.qr_zpf * (a.T - a)
        ham = np.kron(pair, ir).astype(complex)
        ham += np.kron(
            np.eye(pair.shape[0]), np.diag(np.arange(b.resonator_levels) * self.fr_ghz)
        )
        ham += (
            2
            * e
            / (h * 1e9)
            * np.kron(self.cinv[0, 2] * n1_pair + self.cinv[1, 2] * n2_pair, qr)
        )
        energy, states = eigh(ham, subset_by_index=(0, b.eigenlevels - 1))
        energy -= energy[0]

        # Match g,m,p,r,mr,pr to isolated-pair eigenstates x resonator Fock states.
        # Hungarian assignment keeps the six dressed-state labels distinct.
        labels = ("g", "m", "p", "r", "mr", "pr")
        references = np.column_stack(
            [
                np.kron(pair_states[:, rail], ir[:, photon])
                for rail, photon in ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
            ]
        )
        overlaps = np.abs(references.conj().T @ states) ** 2
        rows, cols = linear_sum_assignment(-overlaps)
        levels = dict(zip((labels[i] for i in rows), cols, strict=True))
        tracked = {name: energy[levels[name]] for name in labels}
        ground = states[:, levels["g"]]
        rail_states = states[:, [levels["m"], levels["p"]]]
        strengths = np.array(
            [
                np.abs(ground.conj() @ np.kron(op, ir) @ rail_states) ** 2
                for op in (n1_pair, n2_pair)
            ]
        )
        totals = strengths.sum(axis=1, keepdims=True)
        participation = np.divide(
            strengths, totals, out=np.zeros_like(strengths), where=totals > 0
        )
        hybridization = np.clip(1 - 2 * np.mean(np.abs(participation - 0.5)), 0, 1)
        chi_p = tracked["pr"] - tracked["p"] - tracked["r"] + tracked["g"]
        chi_m = tracked["mr"] - tracked["m"] - tracked["r"] + tracked["g"]
        return {
            "ej1_ghz": float(ej1_ghz),
            "ej2_ghz": float(ej2_ghz),
            "operating_frequency_ghz": (tracked["p"] + tracked["m"]) / 2 - tracked["g"],
            "dressed_m_ghz": tracked["m"] - tracked["g"],
            "dressed_p_ghz": tracked["p"] - tracked["g"],
            "dressed_r_ghz": tracked["r"] - tracked["g"],
            "splitting_ghz": abs(tracked["p"] - tracked["m"]),
            "hybridization": float(hybridization),
            "chi_p_ghz": chi_p,
            "chi_m_ghz": chi_m,
            "delta_chi_ghz": chi_p - chi_m,
            "abs_delta_chi_over_chi_p": abs((chi_p - chi_m) / chi_p)
            if chi_p != 0
            else np.inf,
            "min_assignment_overlap": float(overlaps[rows, cols].min()),
            "bare_f1_ghz": en1[1],
            "bare_f2_ghz": en2[1],
            # Magnitudes of local |0> <-> |1> exchange matrix elements divided
            # by h, in MHz. Use charge elements from the numerical transmons,
            # not their harmonic approximation or half a dressed splitting.
            "g1r_over_2pi_mhz": abs(2 * e * self.cinv[0, 2] * self.qr_zpf * n1[0, 1])
            / (h * 1e6),
            "g2r_over_2pi_mhz": abs(2 * e * self.cinv[1, 2] * self.qr_zpf * n2[0, 1])
            / (h * 1e6),
        }


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
    en1, n1 = _transmon(point["ej1_ghz"], model.ec_ghz[0], b)
    en2, n2 = _transmon(point["ej2_ghz"], model.ec_ghz[1], b)
    n1p, n2p = (
        np.kron(n1, np.eye(b.transmon_levels)),
        np.kron(np.eye(b.transmon_levels), n2),
    )
    hp = np.diag((en1[:, None] + en2[None, :]).ravel())
    hp += 4 * e**2 * model.cinv[0, 1] / (h * 1e9) * (n1p @ n2p)
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


def check_chi_sign_stability(point, model=None):
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
    if model is None:
        model = ReadoutModel(Device(**point["device"]), Basis(**point["basis"]))
    if asdict(model.device) != point["device"] or asdict(model.basis) != point["basis"]:
        raise ValueError("Sign probes must use the nominal point's device and basis.")
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
        for delta1 in offsets:
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
    certificate = dict(
        schema="chi-sign-grid-v1",
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
        and d.get("schema") == "chi-sign-grid-v1"
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
        lambda ej: _transmon(ej, ec, basis)[0][1] - frequency, 0.02, 100.0, xtol=2e-10
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
                _transmon(
                    (CONDITIONS.ej_over_ec_min + 0.05) * value, value, SCREENING_BASIS
                )[0][1]
                for value in ec
            ),
        )
        upper = min(
            CONDITIONS.operation_frequency_ghz[1] - 0.02,
            *(
                _transmon(CONDITIONS.ej_upper_ghz - 0.01, value, SCREENING_BASIS)[0][1]
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
                best_pass = check_chi_sign_stability(best_pass)
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


def validate(result, basis=VALIDATION_BASIS):
    """Recompute unchanged physical inputs/EJs at the larger validation basis."""
    p = result["point"]
    with threadpool_limits(limits=1):
        m = ReadoutModel(Device(**p["device"]), basis)
        q = m.evaluate(p["ej1_ghz"], p["ej2_ghz"])
        q.update(diagnostics(m, q))
        ec = m.ec_ghz[:2]
        q.update(
            device=p["device"],
            basis=asdict(basis),
            x=p["x"],
            loaded_ec1_mhz=float(ec[0] * 1000),
            loaded_ec2_mhz=float(ec[1] * 1000),
            ej1_over_ec1=float(p["ej1_ghz"] / ec[0]),
            ej2_over_ec2=float(p["ej2_ghz"] / ec[1]),
            participation1=p["participation1"],
            participation2=p["participation2"],
        )
        q["ratio"] = chi_ratio(q)
        if not failures(q):
            q = check_chi_sign_stability(q, m)
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
