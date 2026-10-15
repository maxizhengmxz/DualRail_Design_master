r"""Reproduce the single-junction dual-rail readout study in one Python file.

START HERE (commands run from the repository root):
    .\.venv\Scripts\python.exe readout_workflow/reproduce_readout.py
        Rebuild maps, CDFs, mixing and robustness plots from the archived study.
        This is quick; it does NOT pretend to rerun the numerical optimization.

    .\.venv\Scripts\python.exe readout_workflow/reproduce_readout.py --mode verify
        Also diagonalize the three selected circuits, test label-exchange
        symmetry, validate archived witnesses and reproduce the perturbations.

    .\.venv\Scripts\python.exe readout_workflow/reproduce_readout.py --mode map --workers 4
        Search the full uniform 1 fF triangle (4,278 centers), with checkpoints.

    .\.venv\Scripts\python.exe readout_workflow/reproduce_readout.py --mode robustness --workers 4
        Recalibrate both EJs for 300 perturbations of each selected circuit.
        This reruns the latest EC-first cumulative rescue ladder.

    .\.venv\Scripts\python.exe readout_workflow/reproduce_readout.py --mode all --workers 4
        Rerun the map and robustness, then plot the NEW numerical results.

Use --limit 3 for a small numerical pilot; --pair 87 73 selects a map pixel.
Reruns can take hours. Stop/restart with the same arguments to resume the ledger.
--warm-start-archive reuses candidate witnesses, with numerical revalidation;
without it, archive solutions are NOT supplied to the optimizer.
--help lists locations and controls. Output is separate from the archived study.

Dependencies: numpy, scipy, matplotlib, threadpoolctl. Nothing is imported from
other project Python files. The default cached path needs the two archived JSON
data directories; numerical searches and the three selected design inputs are
self-contained. PNG figures, CSV tables and a Markdown summary are written.

UNITS AND DEFINITIONS
    Capacitances: fF. Inductance: nH. Energies: E/h in GHz. Frequencies: GHz.
    Hamiltonian matrices therefore represent H/h, NOT H/hbar.
    Operation frequency = mean of the two DRESSED logical transitions.
    Logical frequency = separation of those transitions (splitting_ghz in JSON).
    R = max(|Delta chi / chi_minus|, |Delta chi / chi_plus|).
    H = 1 - 2 mean(|p_is - 1/2|), with normalized charge-transition strengths.
    In the ideal two-state limit, bare populations are H/2 and 1-H/2.
    H>=95% therefore means mixing no worse than 47.5%/52.5% in that limit.
    Full-model H is transition balance, not a state overlap or Bell fidelity.

METHOD AND LIMITS
    C_i, E_Ji specify each ideal single-junction transmon. C12 couples the two
    transmons; C1r and C2r couple them to a shared resonator. No junction C_J.
    All local ECs and charge interactions use the SAME full inverse C matrix.
    Equal C1r=C2r is enforced for NOMINAL map pixels, not after fabrication error.
    90% participation motivated the original bounds; it is not an equality here.
    The 6 GHz, 50 ohm resonator uses equivalent-mode impedance. A CPW's line
    impedance needs an appropriate mode conversion before using this LC model.
    A gray pixel means bounded search found no witness, not proven infeasibility.
    Optimizer budgets, warm starts and platform can change rediscovered boundaries
    and rescue counts. Use cached mode to reproduce the recorded figures exactly.
    No absolute-chi, readout-speed, finite-drive, Purcell or noise screen is applied.

READING ORDER
    1. Settings and selected designs; 2. Circuit Hamiltonian and diagnostics;
    3. Nominal feasibility search; 4. EJ-only calibration and rescue ladder;
    5. Sampling/checkpoints; 6. Analysis/plots; 7. Verification and CLI.
"""

from __future__ import annotations

# BLAS environment and the headless plotting backend must precede scientific imports.
# ruff: noqa: E402
import argparse
import csv
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
from pathlib import Path

# One BLAS thread per worker avoids multiplying worker count by BLAS threads.
# Explicit user environment settings are respected; threadpool_limits below
# additionally bounds linear-algebra calls in each numerical job.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.constants import elementary_charge as e
from scipy.constants import h, hbar
from scipy.linalg import eigh
from scipy.optimize import brentq, least_squares, linear_sum_assignment, minimize
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

# %% 1. Study settings -- edit here before starting a NEW output directory.
# The numeric bounds below define the latest agreed study, not the historical
# equal-g searches or the original flux-tuned two-device notebook.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "readout_reproduction"
MAP_ARCHIVE = "refined_shunt_map_2026_10_14"
ROBUSTNESS_ARCHIVE = "reordered_robustness_2026_10_15"
SCHEMA = "standalone-readout-v1"
SEED = 20261015
BARE_RESONATOR_GHZ = 6.0
MODE_IMPEDANCE_OHM = 50.0
CAPACITANCE_GRID_FF = np.arange(36.0, 129.0, 1.0)
CAPS = ("C1_shunt_fF", "C2_shunt_fF", "C12_fF", "C1r_fF", "C2r_fF")
PARAMETERS = (*CAPS, "bare_resonator_frequency")
VARIATION = 0.02
INTERIOR_SAMPLES = 224  # Plus 64 corners and 12 single-parameter endpoints.
CHARGE_CONSTANT = e**2 / (2 * h) * 1e6  # EC/h [GHz] = constant / C_eff [fF].


@dataclass(frozen=True)
class Conditions:
    """All acceptance thresholds; experimental tolerances are not solver margins."""

    mismatch_max: float = 0.01
    dispersive_max: float = 0.1
    hybridization_min: float = 0.95
    reference_overlap_min: float = 0.95
    operation_frequency_ghz: tuple[float, float] = (4.0, 8.0)
    logical_frequency_ghz: tuple[float, float] = (0.09, 0.11)
    loaded_ec_mhz: tuple[float, float] = (150.0, 400.0)  # Upper bound strict.
    ej_upper_ghz: float = 25.0  # Strict.
    ej_over_ec_min: float = 50.0  # Strict.
    c12_bounds_ff: tuple[float, float] = (0.005, 15.0)
    readout_cap_bounds_ff: tuple[float, float] = (0.1, 25.0)
    fixed_frequency_tolerance_ghz: float = 1e-7  # 100 Hz; numerical tolerance.


CONDITIONS = Conditions()
SEARCH_EFFORT = {
    # Number of starts, maximum least-squares evaluations, wall seconds/pixel.
    "fast": (3, 35, 25.0),
    "normal": (9, 65, 90.0),
    "deep": (24, 110, 240.0),
}

# Full-precision COMPONENT inputs, not rounded values copied from the report.
# These three picks were user-directed redraws, not an unbiased yield sample.
SELECTED_INPUTS = {
    "1": dict(
        C1=87.0,
        C2=73.0,
        C12=1.7691402536230083,
        Cg=5.110135439087974,
        EJ1=14.13207797323162,
        EJ2=12.204850182595784,
    ),
    "2": dict(
        C1=109.0,
        C2=90.0,
        C12=2.57420270857928,
        Cg=6.733651810326046,
        EJ1=15.961139641589087,
        EJ2=13.574113420955165,
    ),
    "3": dict(
        C1=119.0,
        C2=81.0,
        C12=2.507648284518531,
        Cg=6.601150835819695,
        EJ1=17.182167508792233,
        EJ2=12.45033306247459,
    ),
}

# %% 2. Physical circuit, local bases, dressed spectrum and branch diagnostics.
# C's node order is T1,T2,resonator. EC_i=e^2 (C^-1)_ii/2, while off-diagonal
# elements supply the charge-charge terms. Diagonal loading and interaction
# are both retained, so coupling-capacitor loading must not be added a second time.


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
LARGE = VALIDATION_BASIS


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
    p["ratio"] = float(max(p["ratio_m"], p["abs_delta_chi_over_chi_p"]))
    return p


# %% 3. Nominal map search: four continuous variables per C1>C2 pixel.
# x = (mean loaded local f01 [GHz], local detuning / 0.01 GHz, C12, Cg).
# These local search coordinates are not the dressed operation frequency.
# Each trial inverts f01(EJ,EC) numerically; EJs can tune independently.


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
    p["ratio"] = float(max(p["ratio_m"], p["abs_delta_chi_over_chi_p"]))
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
                break
    selected = best_pass or best
    return {
        "c1": c1,
        "c2": c2,
        "level": level,
        "status": "screened_pass" if best_pass else "no_witness_found",
        "point": selected,
        "failures": failures(selected) if selected else ["no_evaluation"],
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
        q["ratio"] = float(max(q["ratio_m"], q["abs_delta_chi_over_chi_p"]))
    out = dict(result)
    out["screening_point"] = p
    out["point"] = q
    out["failures"] = failures(q)
    out["status"] = "validated_pass" if not out["failures"] else "validation_failed"
    return out


# %% 4. Fabrication perturbation and EJ-only calibration.
# All components freeze after a perturbation. Only the two EJs optimize.
# Maximize H subject to matching and the other inequalities; neither
# maximum mixing nor exact chi matching is imposed as an equality.


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


def x_from(p):
    return np.array(
        [
            (p["bare_f1_ghz"] + p["bare_f2_ghz"]) / 2,
            (p["bare_f1_ghz"] - p["bare_f2_ghz"]) / 0.01,
        ]
    )


class Calibration:
    """Recalibrate EJs at immutable fabricated capacitances and resonator inputs."""

    def __init__(self, device, reference, basis=SCREENING_BASIS):
        self.m = ReadoutModel(device, basis)
        self.reference = reference
        self.target = reference["operating_frequency_ghz"]
        self.cache = {}
        self.evaluations = 0

    def point(self, x):
        key = tuple(map(float, x))
        if key not in self.cache:
            f, d = key
            ej = [
                inverse_frequency(f + sign * 0.005 * d, ec, self.m.basis)
                for sign, ec in zip((1, -1), self.m.ec_ghz[:2], strict=True)
            ]
            self.cache[key] = annotate(self.m, *ej)
            self.evaluations += 1
        return self.cache[key]

    def solve(self, tier, warm=None, deep=False):
        start = time.monotonic()
        original = x_from(self.reference)
        seeds = [
            x_from(warm) if warm else original,
            original,
            original + [0, 0.6],
            original - [0, 0.6],
        ]
        if not tier["fixed"]:
            seeds += [
                original + [-0.25, 0],
                original + [0.25, 0],
                np.array([CONDITIONS.operation_frequency_ghz[0] + 0.08, original[1]]),
            ]
            lo = max(
                CONDITIONS.operation_frequency_ghz[0] + 0.02,
                *(
                    _transmon((tier["ej_ec"] + 0.01) * ec, ec, self.m.basis)[0][1]
                    for ec in self.m.ec_ghz[:2]
                ),
            )
            hi = min(
                CONDITIONS.operation_frequency_ghz[1] - 0.02,
                *(
                    _transmon(CONDITIONS.ej_upper_ghz - 0.001, ec, self.m.basis)[0][1]
                    for ec in self.m.ec_ghz[:2]
                ),
            )
            if lo < hi:
                seeds += [
                    np.array([lo + fraction * (hi - lo), original[1]])
                    for fraction in (0.5, 0.1, 0.9)
                ]
        seen = []
        attempts = []
        best = None

        def merit(p):
            rest = [r for r in reasons(p, tier, self.target) if r != "hybridization"]
            if not rest:
                return (1.0, p["hybridization"], 0.0)
            violation = np.minimum(slacks(p, tier), 0.0)
            eq = (
                (p["operating_frequency_ghz"] - self.target) / 0.02
                if tier["fixed"]
                else 0.0
            )
            return (
                0.0,
                -float(np.linalg.norm(np.r_[violation, eq])),
                p["hybridization"],
            )

        def keep(p):
            nonlocal best
            if best is None or merit(p) > merit(best):
                best = p
            return p

        def point(x):
            return keep(self.point(x))

        for index, seed in enumerate(seeds):
            seed = np.clip(seed, [3.7, -20], [8.3, 20])
            if any(np.linalg.norm(seed - s) < 1e-6 for s in seen):
                continue
            seen.append(seed)
            if index and time.monotonic() - start > (100 if deep else 35):
                break

            def residual(x):
                p = point(x)
                v = np.minimum(slacks(p, tier, margin=1), 0.0)
                return (
                    np.r_[v, (p["operating_frequency_ghz"] - self.target) / 0.02]
                    if tier["fixed"]
                    else v
                )

            fit = least_squares(
                residual,
                seed,
                bounds=([3.7, -20], [8.3, 20]),
                diff_step=2e-5,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-8,
                max_nfev=80 if deep else 40,
            )
            point(fit.x)
            constraints = [
                dict(type="ineq", fun=lambda x: slacks(point(x), tier, margin=1))
            ]
            if tier["fixed"]:
                constraints.append(
                    dict(
                        type="eq",
                        fun=lambda x: (
                            (point(x)["operating_frequency_ghz"] - self.target) / 0.02
                        ),
                    )
                )
            opt = minimize(
                lambda x: -point(x)["hybridization"],
                fit.x,
                method="SLSQP",
                bounds=[(3.7, 8.3), (-20, 20)],
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 85 if deep else 45, "eps": 1e-5},
            )
            point(opt.x)
            attempts.append(
                dict(
                    seed=seed.tolist(),
                    fit_nfev=int(fit.nfev),
                    optimizer_success=bool(opt.success),
                    optimizer_message=str(opt.message),
                    best_failures=reasons(best, tier, self.target),
                )
            )
            # One polished solution plus an independent detuning start for
            # fixed-operation-frequency optimization. Retry more seeds for unsuccessful cases.
            if (
                best is not None
                and not reasons(best, tier, self.target)
                and len(attempts) >= 2
            ):
                break
        return dict(
            point=best,
            failures=reasons(best, tier, self.target),
            attempts=attempts,
            evaluations=self.evaluations,
            seconds=time.monotonic() - start,
        )


def evaluate(task):
    """Apply the cumulative rescue ladder, validating every accepted witness.

    Optional previous points are warm starts, never permission to relax the
    fabricated circuit. Their component dictionaries must match exactly.
    """
    label, reference, scenario, previous = task
    started = time.monotonic()
    device = perturb(reference, scenario["errors"])
    with threadpool_limits(limits=1):
        small = Calibration(device, reference)
        large = Calibration(device, reference, LARGE)
        ec = small.m.ec_ghz[:2] * 1000
        stages = []
        first_pass = None
        warm = None
        pool = [s["point"] for s in previous.get("stages", []) if s.get("point")]
        assert all(p["device"] == asdict(device) for p in pool)
        structural = bool(
            min(ec) < min(tier["ec_min"] for tier in TIERS) - 1e-7
            or max(ec) >= CONDITIONS.loaded_ec_mhz[1]
        )
        if structural:
            return dict(
                design=label,
                **scenario,
                device=asdict(device),
                stages=[],
                first_pass=None,
                structural_failure="EC_range",
                loaded_ec_mhz=ec.tolist(),
                seconds=time.monotonic() - started,
            )
        for index, tier in enumerate(TIERS):
            if min(ec) < tier["ec_min"] - 1e-7:
                p = small.point(x_from(reference))
                result = dict(
                    point=p,
                    failures=["EC_range"],
                    attempts=[],
                    seconds=0.0,
                    source="immutable_EC_screen",
                )
            elif index == 1 and min(ec) >= CONDITIONS.loaded_ec_mhz[0] - 1e-7:
                result = dict(stages[-1])
                result["source"] = "EC relaxation inactive: identical feasible set"
            elif index == 0 and previous.get("stages"):
                result = dict(previous["stages"][0])
                p = result["point"]
                result["failures"] = reasons(p, tier, small.target)
                result["source"] = "unchanged previous baseline"
                if not result["failures"]:
                    q = annotate(large.m, p["ej1_ghz"], p["ej2_ghz"])
                    result["point"] = q
                    result["failures"] = reasons(q, tier, large.target)
            else:
                valid = [p for p in pool if not reasons(p, tier, small.target)]
                result = None
                for p in sorted(valid, key=lambda p: p["hybridization"], reverse=True):
                    q = annotate(large.m, p["ej1_ghz"], p["ej2_ghz"])
                    if not reasons(q, tier, large.target):
                        result = dict(
                            point=q,
                            failures=[],
                            attempts=[],
                            seconds=0.0,
                            source="revalidated_previous_witness",
                        )
                        break
                if result is None:
                    if pool:
                        warm = max(
                            pool,
                            key=lambda p: (
                                -np.linalg.norm(np.minimum(slacks(p, tier), 0.0)),
                                p["hybridization"],
                            ),
                        )
                    result = small.solve(tier, warm)
                    warm = result["point"]
                    if not result["failures"]:
                        q = annotate(large.m, warm["ej1_ghz"], warm["ej2_ghz"])
                        result["screening_point"] = warm
                        result["point"] = q
                        result["failures"] = reasons(q, tier, large.target)
                        if result["failures"]:
                            repair = large.solve(tier, q, deep=True)
                            result["repair"] = repair
                            result["point"] = repair["point"]
                            result["failures"] = repair["failures"]
                    elif (
                        max(abs(np.minimum(slacks(warm, tier, include_H=True), 0.0)))
                        < 0.03
                    ):
                        retry = large.solve(tier, warm, deep=True)
                        result["large_basis_retry"] = retry
                        if not retry["failures"]:
                            result["point"] = retry["point"]
                            result["failures"] = []
            result["tier"] = tier["name"]
            result["passes"] = not result["failures"]
            stages.append(result)
            pool.append(result["point"])
            if result["passes"]:
                first_pass = index
                break
        return dict(
            design=label,
            **scenario,
            device=asdict(device),
            stages=stages,
            first_pass=first_pass,
            structural_failure=None,
            loaded_ec_mhz=ec.tolist(),
            seconds=time.monotonic() - started,
        )


# Named constraints are used by BOTH acceptance and optimization. Small margins
# below guide the optimizer away from a boundary; they do not change acceptance.
# Existing JSON keys (e.g. splitting_ghz) are retained so archives stay readable.
def violations(point, margin=0.0):
    """Nonnegative scaled violations; zero means feasible before strict checks."""
    c = CONDITIONS
    logical = point["splitting_ghz"]
    operation = point["operating_frequency_ghz"]
    ec = np.array([point["loaded_ec1_mhz"], point["loaded_ec2_mhz"]]) / 1000
    return np.array(
        [
            max(0, point["ratio"] - (c.mismatch_max - margin * 0.001)) / c.mismatch_max,
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


def failures(point):
    """Nominal map acceptance, including STRICT EC upper/EJ upper/EJ/EC bounds."""
    return reasons(point, TIERS[0] | {"fixed": False}, point["operating_frequency_ghz"])


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


def rescue_tiers():
    """Cumulative relaxation: EC, operation frequency, logical frequency, H, EJ/EC."""
    c = CONDITIONS
    baseline = dict(
        name="baseline",
        ec_min=c.loaded_ec_mhz[0],
        H=c.hybridization_min,
        logical_frequency=c.logical_frequency_ghz,
        ej_ec=c.ej_over_ec_min,
        fixed=True,
    )
    ec = baseline | dict(name="EC120", ec_min=120.0)
    flexible = ec | dict(name="flexible_operation_frequency", fixed=False)
    logical = flexible | dict(
        name="logical_frequency80_120_H95", logical_frequency=(0.08, 0.12)
    )
    mixing = logical | dict(name="H90", H=0.90)
    energy = mixing | dict(name="EJ_EC40", ej_ec=40.0)
    return (baseline, ec, flexible, logical, mixing, energy)


TIERS = rescue_tiers()


def reasons(point, tier, target):
    """List failed constraints; include_H is never silently disabled here."""
    c = CONDITIONS
    low, high = tier["logical_frequency"]
    ec1, ec2 = point["loaded_ec1_mhz"], point["loaded_ec2_mhz"]
    checks = {
        "chi_matching": point["ratio"] <= c.mismatch_max + 1e-8,
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
    return [name for name, passed in checks.items() if not passed]


def slacks(point, tier, include_H=False, margin=0.0):
    """Positive values satisfy recalibration inequalities; EC is screened first.

    H is the objective during constrained polishing. It is checked against the
    tier floor on acceptance, rather than constrained to its maximum or H=1.
    """
    c = CONDITIONS
    values = [
        (c.mismatch_max - margin * 1e-6 - point["ratio"]) / c.mismatch_max,
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
    return np.nan_to_num(values, nan=-1e4, posinf=1e4, neginf=-1e4)


# %% 5. Sampling, provenance and resumable numerical jobs.
def selected_designs():
    """Recompute the three full-precision nominal input circuits, in the large basis."""
    designs = {}
    with threadpool_limits(limits=1):
        for label, inputs in SELECTED_INPUTS.items():
            model = make_model(
                inputs["C1"],
                inputs["C2"],
                inputs["C12"],
                inputs["Cg"],
                VALIDATION_BASIS,
            )
            designs[label] = annotate(model, inputs["EJ1"], inputs["EJ2"])
            designs[label]["x"] = x_from_point(designs[label])
            if failures(designs[label]):
                raise RuntimeError(
                    f"Design {label} violates the configured nominal conditions: {failures(designs[label])}"
                )
    return designs


def sampling_scenarios():
    """Same 300 deterministic vectors/design; stress samples are not yield estimates."""
    unit = (
        2 * qmc.LatinHypercube(d=len(PARAMETERS), seed=SEED).random(INTERIOR_SAMPLES)
        - 1
    )
    scenarios = [dict(kind="interior", errors=(VARIATION * v).tolist()) for v in unit]
    scenarios += [
        dict(kind="corner", errors=(VARIATION * np.array(v)).tolist())
        for v in product((-1, 1), repeat=len(PARAMETERS))
    ]
    for index, parameter in enumerate(PARAMETERS):
        for sign in (-1, 1):
            vector = np.zeros(len(PARAMETERS))
            vector[index] = sign * VARIATION
            scenarios.append(
                dict(kind="one_at_a_time", parameter=parameter, errors=vector.tolist())
            )
    return [scenario | dict(id=index) for index, scenario in enumerate(scenarios)]


def load_json(path):
    """Read UTF-8 study data with a useful missing-file message."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Use --data-dir for the archived data, or a numerical --mode to rerun."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    """Atomic checkpoint export; the SQLite ledger is the numerical source of truth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class ResultStore:
    """One main-process SQLite writer; separate ledgers for map and robustness.

    A source/configuration fingerprint prevents accidental mixing of different
    code revisions. Use a new --output-dir after changing numerical settings.
    """

    def __init__(self, path, configuration):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY,payload TEXT)"
        )
        fingerprint = sha256(
            Path(__file__).read_bytes()
            + json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest()
        previous = self.db.execute(
            "SELECT value FROM metadata WHERE key='fingerprint'"
        ).fetchone()
        if previous and previous[0] != fingerprint:
            self.db.close()
            raise RuntimeError(
                "Source or settings changed. Choose a new --output-dir; the previous ledger is preserved."
            )
        self.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)", ("fingerprint", fingerprint)
        )
        self.db.commit()
        self.rows = {
            key: json.loads(value)
            for key, value in self.db.execute("SELECT key,payload FROM results")
        }

    def save(self, key, result):
        self.rows[key] = result
        self.db.execute(
            "INSERT OR REPLACE INTO results VALUES(?,?)", (key, json.dumps(result))
        )
        self.db.commit()

    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(FULL)")
        self.db.close()


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


def run_map(args, designs):
    """Uniform strict triangle; coarse traversal supplies seeds without interpolating.

    This cleaned runner starts on the 2 fF lattice, fills every 1 fF pixel, then
    challenges failed pixels with deeper starts. The original 12-hour run used
    additional hand-scheduled continuation rounds. Exact rediscovered counts
    need not be identical; cached mode preserves that recorded result.
    """
    pairs = (
        [tuple(args.pair)]
        if args.pair
        else [
            (float(c1), float(c2))
            for c1 in CAPACITANCE_GRID_FF
            for c2 in CAPACITANCE_GRID_FF
            if c1 > c2
        ]
    )
    if any(c1 <= c2 for c1, c2 in pairs):
        raise ValueError(
            "Only C1>C2 is computed; swap transmon labels for the mirror design."
        )
    # Stable ordering guarantees the same coverage in a resumed or limited run.
    pairs.sort(key=lambda p: (int(p[0]) % 2 or int(p[1]) % 2, p[0], p[1]))
    if args.limit:
        pairs = pairs[: args.limit]
    archive = {}
    if args.warm_start_archive:
        archive = {
            (r["c1"], r["c2"]): r
            for r in load_json(args.data_dir / MAP_ARCHIVE / "uniform_points.json")
        }
    configuration = dict(
        schema=SCHEMA,
        mode="map",
        conditions=asdict(CONDITIONS),
        grid=CAPACITANCE_GRID_FF.tolist(),
        effort=SEARCH_EFFORT,
        bare_frequency=BARE_RESONATOR_GHZ,
        impedance=MODE_IMPEDANCE_OHM,
        warm_start_archive=args.warm_start_archive,
        seed=SEED,
        input_designs=SELECTED_INPUTS,
    )
    store = ResultStore(args.output_dir / "map.sqlite", configuration)
    deadline = time.monotonic() + args.hours * 3600 if args.hours else float("inf")
    try:
        # Chunked submission lets new neighbors become warm starts on the next
        # chunk. Independent jobs parallelize; checkpoint writes stay sequential.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for round_index, level in enumerate(("normal", "deep")):
                pending = [
                    pair
                    for pair in pairs
                    if store.rows.get(f"{pair[0]}:{pair[1]}", {}).get("status")
                    != "validated_pass"
                    and store.rows.get(f"{pair[0]}:{pair[1]}", {}).get(
                        "search_round", -1
                    )
                    < round_index
                ]
                for offset in range(0, len(pending), args.workers * 2):
                    if time.monotonic() >= deadline:
                        break
                    passing = [
                        r["point"]
                        for r in store.rows.values()
                        if r["status"] == "validated_pass"
                    ]
                    futures = {}
                    for pair in pending[offset : offset + args.workers * 2]:
                        candidates = list(designs.values()) + passing
                        candidates.sort(
                            key=lambda p: (
                                (p["device"]["C1_shunt_fF"] - pair[0]) ** 2
                                + (p["device"]["C2_shunt_fF"] - pair[1]) ** 2
                            )
                        )
                        warm = candidates[:4]
                        prior = store.rows.get(f"{pair[0]}:{pair[1]}")
                        if prior and prior.get("point"):
                            warm = [prior["point"]] + warm
                        if archive.get(pair, {}).get("point"):
                            warm = [archive[pair]["point"]] + warm
                        futures[
                            pool.submit(
                                search_and_validate,
                                dict(pair=pair, level=level, warm=warm),
                            )
                        ] = pair
                    for future in as_completed(futures):
                        pair = futures[future]
                        try:
                            result = future.result()
                        except Exception as error:
                            result = dict(
                                c1=pair[0],
                                c2=pair[1],
                                status="unresolved_error",
                                point=None,
                                error=repr(error),
                            )
                        result["search_round"] = round_index
                        store.save(f"{pair[0]}:{pair[1]}", result)
                    print(
                        f"Map round {round_index + 1}: {len(store.rows)} checkpointed pixels",
                        flush=True,
                    )
                if time.monotonic() >= deadline:
                    break
        results = sorted(store.rows.values(), key=lambda r: (r["c1"], r["c2"]))
        save_json(args.output_dir / "map_results.json", results)
        return results
    finally:
        store.close()


def credit_earliest_tier(result, reference):
    """A later-discovered witness gets credit for the earliest tier it satisfies."""
    if result.get("first_pass") is not None:
        result["discovered_at_tier"] = result["first_pass"]
        point = result["stages"][-1]["point"]
        result["first_pass"] = next(
            index
            for index, tier in enumerate(TIERS)
            if not reasons(point, tier, reference["operating_frequency_ghz"])
        )
    return result


def run_robustness(args, designs):
    """Fresh EJ-only optimization, with optional explicit archive warm starts."""
    vectors = sampling_scenarios()
    if args.limit:
        vectors = vectors[: args.limit]
    previous = {}
    if args.warm_start_archive:
        previous = {
            (r["design"], r["id"]): r
            for r in load_json(args.data_dir / ROBUSTNESS_ARCHIVE / "samples.json")
        }
    configuration = dict(
        schema=SCHEMA,
        mode="robustness",
        conditions=asdict(CONDITIONS),
        tiers=TIERS,
        seed=SEED,
        variation=VARIATION,
        interior=INTERIOR_SAMPLES,
        input_designs=SELECTED_INPUTS,
        warm_start_archive=args.warm_start_archive,
    )
    store = ResultStore(args.output_dir / "robustness.sqlite", configuration)
    deadline = time.monotonic() + args.hours * 3600 if args.hours else float("inf")
    tasks = [
        (label, point, scenario, previous.get((label, scenario["id"]), {}))
        for label, point in designs.items()
        for scenario in vectors
        if f"{label}:{scenario['id']}" not in store.rows
    ]
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for offset in range(0, len(tasks), args.workers * 2):
                if time.monotonic() >= deadline:
                    break
                futures = {
                    pool.submit(evaluate, task): task
                    for task in tasks[offset : offset + args.workers * 2]
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = credit_earliest_tier(future.result(), task[1])
                    except Exception as error:
                        result = dict(
                            design=task[0],
                            **task[2],
                            stages=[],
                            first_pass=None,
                            error=repr(error),
                        )
                    store.save(f"{task[0]}:{task[2]['id']}", result)
                print(
                    f"Robustness: {len(store.rows)} checkpointed fabricated circuits",
                    flush=True,
                )
        results = sorted(store.rows.values(), key=lambda r: (r["design"], r["id"]))
        save_json(args.output_dir / "robustness_results.json", results)
        save_json(
            args.output_dir / "sampling.json",
            dict(seed=SEED, parameters=PARAMETERS, scenarios=sampling_scenarios()),
        )
        return results
    finally:
        store.close()


# %% 6. Analysis and publication-style figures -- no optimization in this section.
def analyze_robustness(rows, designs):
    """Separate interior/error-corner statistics and rebuild the conditional H CDF.

    The CDF uses fixed nominal operation frequency, EC>=120 MHz, R<=1%,
    and all other baseline screens EXCEPT H. Only eligible interior samples
    contribute. We do not assign a fabricated score to excluded samples.
    A finite search gives a best-found H, not a guaranteed global maximum.
    """
    analysis, cdfs = {}, {}
    for label, reference in designs.items():
        group = [row for row in rows if row["design"] == label]

        def counts(subset):
            return dict(
                total=len(subset),
                cumulative=[
                    sum(
                        row.get("first_pass") is not None and row["first_pass"] <= index
                        for row in subset
                    )
                    for index in range(len(TIERS))
                ],
            )

        analysis[label] = counts(group) | dict(
            by_kind={
                kind: counts([row for row in group if row["kind"] == kind])
                for kind in ("interior", "corner", "one_at_a_time")
            }
        )
        values = []
        for row in group:
            if row["kind"] != "interior":
                continue
            candidates = [
                stage["point"]
                for stage in row.get("stages", [])
                if stage["tier"] in ("baseline", "EC120") and stage.get("point")
            ]
            eligible = [
                point
                for point in candidates
                if not [
                    failure
                    for failure in reasons(
                        point, TIERS[1], reference["operating_frequency_ghz"]
                    )
                    if failure != "hybridization"
                ]
            ]
            if eligible:
                values.append(max(point["hybridization"] for point in eligible))
        cdfs[label] = dict(
            eligible=len(values),
            total_interior=analysis[label]["by_kind"]["interior"]["total"],
            hybridization_values=sorted(values),
        )
    return analysis, cdfs


def map_arrays(rows):
    """Uniform 1 fF cells; retain unresolved and uncomputed states explicitly."""
    grid = CAPACITANCE_GRID_FF
    shape = (len(grid), len(grid))
    index = {float(value): i for i, value in enumerate(grid)}
    status = np.full(
        shape, -1
    )  # -1 excluded/uncomputed, 0 no witness, 1 pass, 2 unresolved.
    keys = (
        "H",
        "operation_frequency",
        "Cg",
        "logical_frequency",
        "EC1",
        "EC2",
        "mismatch",
        "dispersive",
        "EJ_EC",
        "overlap",
    )
    arrays = {key: np.full(shape, np.nan) for key in keys}
    for row in rows:
        c1, c2 = row["c1"], row["c2"]
        if c1 not in index or c2 not in index or c1 <= c2:
            continue
        i, j = index[c2], index[c1]
        if row["status"] == "validated_pass":
            status[i, j] = 1
            point = row["point"]
            values = dict(
                H=point["hybridization"],
                operation_frequency=point["operating_frequency_ghz"],
                Cg=point["device"]["C1r_fF"],
                logical_frequency=1000 * point["splitting_ghz"],
                EC1=point["loaded_ec1_mhz"],
                EC2=point["loaded_ec2_mhz"],
                mismatch=100 * point["ratio"],
                dispersive=point["lambda_max"],
                EJ_EC=min(point["ej1_over_ec1"], point["ej2_over_ec2"]),
                overlap=point["min_assignment_overlap"],
            )
            for key, value in values.items():
                arrays[key][i, j] = value
        elif row["status"] == "no_witness_found":
            status[i, j] = 0
        else:
            status[i, j] = 2
    return status, arrays


def draw_map(ax, status, values, title, limits=None, designs=None):
    """Plot real grid cells, never interpolation across passing or missing regions."""
    grid = CAPACITANCE_GRID_FF
    half_step = (grid[1] - grid[0]) / 2
    edges = np.r_[grid - half_step, grid[-1] + half_step]
    for code, color in ((0, "#e6e6e6"), (2, "#f7cf99")):
        ax.pcolormesh(
            edges,
            edges,
            np.ma.masked_where(status != code, np.zeros_like(status)),
            cmap=ListedColormap([color]),
            vmin=0,
            vmax=1,
            shading="flat",
        )
    kwargs = dict(vmin=limits[0], vmax=limits[1]) if limits else {}
    image = ax.pcolormesh(
        edges,
        edges,
        np.ma.masked_invalid(values),
        cmap="viridis",
        shading="flat",
        **kwargs,
    )
    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], ":", color="black", lw=0.8)
    if designs:
        offsets = {"1": (-20, 10), "2": (10, 10), "3": (10, -18)}
        for label, point in designs.items():
            position = (point["device"]["C1_shunt_fF"], point["device"]["C2_shunt_fF"])
            ax.scatter(
                *position,
                marker="*",
                s=150,
                color="#d21b3c",
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )
            ax.annotate(
                label,
                position,
                xytext=offsets[label],
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=.15", fc="white", ec="black", lw=0.5),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
                zorder=6,
            )
    ax.set(
        xlabel=r"$C_1$ (fF)",
        ylabel=r"$C_2$ (fF)",
        title=title,
        xlim=(edges[0], edges[-1]),
        ylim=(edges[0], edges[-1]),
    )
    ax.set_aspect("equal")
    return image


def plot_results(rows, designs, analysis, cdfs, output_dir, origin):
    """Recreate the agreed map/mixing/robustness figures plus EC and constraint maps."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    produced = []

    def save(fig, name):
        # PNG keeps the reproduction path lightweight; the existing narrative
        # PDF stays untouched. A single Python file remains the only code needed.
        path = figure_dir / f"{name}.png"
        fig.savefig(path, dpi=200)
        produced.append(str(path))
        plt.close(fig)

    status, arrays = map_arrays(rows)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), layout="constrained")
    image = draw_map(
        ax,
        status,
        arrays["operation_frequency"],
        "Viable pairs and robustness designs",
        (4, 8),
        designs,
    )
    fig.colorbar(image, ax=ax, shrink=0.88, label="Operation frequency (GHz)")
    fig.suptitle(f"{origin} | uniform 1 fF pixels; only " + r"$C_1>C_2$", fontsize=11)
    save(fig, "01_viable_pairs_marked")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    specs = [
        ("H", "Hybridization score", (0.95, 1)),
        ("operation_frequency", "Operation frequency (GHz)", (4, 8)),
        ("Cg", r"Equal nominal $C_{1r}=C_{2r}=C_g$ (fF)", None),
        ("logical_frequency", "Logical frequency (MHz)", (90, 110)),
    ]
    for ax, (key, title, limits) in zip(axes.flat, specs, strict=True):
        fig.colorbar(
            draw_map(ax, status, arrays[key], title, limits), ax=ax, shrink=0.82
        )
    fig.suptitle(f"{origin} | chosen witness at each viable pair", fontsize=12)
    save(fig, "02_design_maps")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for ax, key in zip(axes[:2], ("EC1", "EC2"), strict=True):
        title = rf"Loaded $E_{{C{key[-1]}}}/h$ (MHz)"
        fig.colorbar(
            draw_map(ax, status, arrays[key], title, (150, 400)), ax=ax, shrink=0.85
        )
    ax = axes[2]
    for key, label in (("EC1", "Larger-C rail (T1)"), ("EC2", "Smaller-C rail (T2)")):
        values = np.sort(arrays[key][np.isfinite(arrays[key])])
        if len(values):
            ax.step(
                np.r_[150, values, 400],
                np.r_[0, np.arange(1, len(values) + 1) / len(values), 1],
                where="post",
                label=label,
            )
    pooled = np.sort(
        np.r_[
            arrays["EC1"][np.isfinite(arrays["EC1"])],
            arrays["EC2"][np.isfinite(arrays["EC2"])],
        ]
    )
    if len(pooled):
        ax.step(
            np.r_[150, pooled, 400],
            np.r_[0, np.arange(1, len(pooled) + 1) / len(pooled), 1],
            where="post",
            color="black",
            ls="--",
            label="Pooled mirror-label distribution",
        )
    ax.set(
        xlim=(150, 400),
        ylim=(0, 1.02),
        xlabel=r"Loaded $E_C/h$ (MHz)",
        ylabel="Empirical CDF",
        title="Equal weight per passing pixel",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle(f"{origin} | ordered rails can have different EC CDFs", fontsize=12)
    save(fig, "03_loaded_charging_energies")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    specs = [
        ("mismatch", r"Worst $|\Delta\chi/\chi_i|$ (%)", (0, 1)),
        ("dispersive", r"Maximum multilevel $|g_{ij}/\Delta_{ij}|$", (0, 0.1)),
        ("EJ_EC", r"Minimum $E_J/E_C$", None),
        ("overlap", "Minimum reference-state overlap", (0.95, 1)),
    ]
    for ax, (key, title, limits) in zip(axes.flat, specs, strict=True):
        fig.colorbar(
            draw_map(ax, status, arrays[key], title, limits), ax=ax, shrink=0.82
        )
    fig.suptitle(f"{origin} | constraint diagnostics", fontsize=12)
    save(fig, "04_constraint_maps")

    fig, ax = plt.subplots(figsize=(7.4, 2.6), layout="constrained")
    score = np.linspace(0.8, 1, 301)
    ax.plot(score * 100, (1 - score / 2) * 100, label=r"Larger population $1-H/2$")
    ax.plot(score * 100, score / 2 * 100, label=r"Smaller population $H/2$")
    ax.axvline(95, color="gray", ls="--", lw=0.9)
    ax.scatter([95, 95], [52.5, 47.5], s=20, zorder=5)
    ax.set(
        xlim=(80, 100),
        ylim=(38, 62),
        xticks=[80, 85, 90, 95, 100],
        xlabel=r"Hybridization score $H$ (%)",
        ylabel="Bare-state population (%)",
        title="Ideal single-excitation mixing",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="center left")
    save(fig, "05_state_mixing")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.4, 1]},
        layout="constrained",
    )
    names = [
        "Baseline",
        r"$E_C/h$ floor" + "\n120 MHz",
        "Flexible\noperation\nfrequency",
        "Logical\nfrequency\n80-120 MHz",
        r"$H$ floor" + "\n90%",
        r"$E_J/E_C$ floor" + "\n40",
    ]
    for label in designs:
        interior = analysis[label]["by_kind"]["interior"]
        if interior["total"]:
            axes[0].plot(
                range(6),
                np.array(interior["cumulative"]) / interior["total"] * 100,
                "o-",
                label=f"Design {label}",
            )
        values = cdfs[label]["hybridization_values"]
        if values:
            axes[1].step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                label=f"Design {label}: {len(values)}/{interior['total']} eligible",
            )
    axes[0].set(
        xticks=range(6),
        xticklabels=names,
        ylabel="Cumulative interior pass fraction (%)",
        ylim=(0, 105),
        xlim=(-0.2, 5.2),
        title="Cumulative rescue ladder",
        xlabel="Successive cumulative relaxations",
    )
    axes[0].tick_params(axis="x", labelsize=11, pad=8)
    axes[1].axvline(0.95, color="black", ls="--", lw=1)
    axes[1].set(
        xlabel=r"Hybridization score $H$",
        ylabel="Conditional empirical CDF",
        ylim=(0, 1.02),
        title="Fixed operation frequency\n" + r"$E_C/h$ floor 120 MHz",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9, loc="upper left")
    fig.suptitle(
        f"{origin} | interiors only; conditional CDF excludes ineligible samples",
        fontsize=11,
    )
    save(fig, "06_robustness")
    return produced


def write_tables_and_summary(
    rows, designs, samples, analysis, cdfs, output_dir, origin
):
    """Save machine-readable witnesses alongside a short explanation of the run."""
    fields = [
        "C1_fF",
        "C2_fF",
        "status",
        "EJ1_GHz",
        "EJ2_GHz",
        "C12_fF",
        "Cg_fF",
        "EC1_MHz",
        "EC2_MHz",
        "operation_frequency_GHz",
        "logical_frequency_MHz",
        "H",
        "R",
        "dispersive_ratio",
    ]
    with (output_dir / "capacitance_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = dict(C1_fF=row["c1"], C2_fF=row["c2"], status=row["status"])
            if row.get("point"):
                p = row["point"]
                record.update(
                    EJ1_GHz=p["ej1_ghz"],
                    EJ2_GHz=p["ej2_ghz"],
                    C12_fF=p["device"]["C12_fF"],
                    Cg_fF=p["device"]["C1r_fF"],
                    EC1_MHz=p["loaded_ec1_mhz"],
                    EC2_MHz=p["loaded_ec2_mhz"],
                    operation_frequency_GHz=p["operating_frequency_ghz"],
                    logical_frequency_MHz=1000 * p["splitting_ghz"],
                    H=p["hybridization"],
                    R=p["ratio"],
                    dispersive_ratio=p["lambda_max"],
                )
            writer.writerow(record)
    with (output_dir / "robustness_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["design", "sample_kind", "total", *[tier["name"] for tier in TIERS]]
        )
        for label, stats in analysis.items():
            for kind, values in {"all": stats, **stats["by_kind"]}.items():
                writer.writerow([label, kind, values["total"], *values["cumulative"]])
    save_json(output_dir / "selected_designs.json", designs)
    save_json(output_dir / "robustness_analysis.json", analysis)
    save_json(output_dir / "hybridization_cdf.json", cdfs)
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in sorted({r["status"] for r in rows})
    }
    lines = [
        "# Dual-rail readout reproduction",
        "",
        f"Source: **{origin}**.",
        "",
        "## Conditions",
        "",
        f"- Bare resonator: {BARE_RESONATOR_GHZ:g} GHz; equivalent-mode impedance {MODE_IMPEDANCE_OHM:g} ohm.",
        f"- Nominal conditions: `{json.dumps(asdict(CONDITIONS))}`.",
        "- Only C1>C2; uniform 1 fF pixel centers from 36 to 128 fF.",
        "- Each nominal pixel can choose both EJs, C12 and common C1r=C2r.",
        "- 90% participation is an initial estimate, not an enforced equality.",
        "- Operation frequency means the mean dressed logical transitions; logical frequency means their separation.",
        "- H measures charge-transition balance. Ideal bare-state populations are H/2 and 1-H/2.",
        "- H=95% means 47.5%/52.5% populations only in the ideal two-state limit; H is not Bell fidelity.",
        "- Cr and C12 enter the inverse C matrix: changing them can spoil a chosen matched witness; EJ recalibration can sometimes restore it.",
        "",
        "## Map",
        "",
        f"{len(rows)} computed/saved centers; status counts: `{json.dumps(counts)}`.",
        "Gray means no witness found by bounded search; orange means unresolved; white is excluded/uncomputed.",
        "A passing pixel is an existence result for its full component set, not a guarantee for arbitrary C12 or Cr.",
        "",
        "## Robustness",
        "",
        f"{len(samples)} saved fabricated circuits across three designs.",
        f"Each of {len(PARAMETERS)} parameters can vary independently by +/-{100 * VARIATION:g}%: {', '.join(PARAMETERS)}.",
        "Cr and Lr scale inversely with bare resonator frequency, preserving mode impedance.",
        "Only EJs recalibrate. Fixed operation frequency uses the nominal dressed mean within 100 Hz numerical tolerance.",
        "Flexible operation frequency permits 4-8 GHz. Logical frequency and H follow the cumulative ladder.",
        "224 interior samples, 64 corners and 12 isolated endpoints/design; combined fractions are not fabrication yields.",
        "",
        "| Design | Total | Baseline | EC120 | Flexible frequency | Logical frequency 80-120 | H90 | EJ/EC40 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in analysis.items():
        lines.append(
            "| "
            + " | ".join(map(str, [label, stats["total"], *stats["cumulative"]]))
            + " |"
        )
    lines += [
        "",
        "The H CDF is conditional on all other baseline screens, with EC>=120 MHz and fixed operation frequency.",
        "No score is invented for ineligible samples. Counts and CDFs are best-found bounded-search results.",
        "",
        "## Files",
        "",
        "- `figures/`: six PNG figures, including the marked map, loaded-EC CDFs and latest rescue comparison.",
        "- `capacitance_pairs.csv`: all map statuses and witness metrics.",
        "- `robustness_counts.csv`: separate interior, corner, endpoint and combined counts.",
        "- `selected_designs.json`, `hybridization_cdf.json`, `robustness_analysis.json`: numerical inputs/results.",
        "- `provenance.json`: source hashes, configuration, dependencies and run mode.",
        "",
        "Cached mode redraws archived results. Numerical reruns may rediscover different witnesses/counts because the finite search is not exhaustive.",
        "The original study data and narrative PDF are preserved.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# %% 7. Verification and command-line entry point.
def verify_results(map_rows, designs, samples, data_dir):
    """Audit saved inequalities AND independently diagonalize physical witnesses.

    This is a short verification, not a fresh optimization of every pixel/error
    vector. Full numerical recalibration is provided by the other CLI modes.
    """
    checks = []
    expected = selected_designs()
    metric_keys = (
        "operating_frequency_ghz",
        "splitting_ghz",
        "hybridization",
        "ratio",
        "lambda_max",
        "min_assignment_overlap",
        "loaded_ec1_mhz",
        "loaded_ec2_mhz",
    )
    with threadpool_limits(limits=1):
        for label, fresh in expected.items():
            for key in metric_keys:
                np.testing.assert_allclose(
                    fresh[key],
                    designs[label][key],
                    rtol=0,
                    atol=2e-7,
                    err_msg=f"Design {label}, {key}",
                )
            original = Device(**fresh["device"])
            mirror = replace(
                original,
                C1_shunt_fF=original.C2_shunt_fF,
                C2_shunt_fF=original.C1_shunt_fF,
                C1r_fF=original.C2r_fF,
                C2r_fF=original.C1r_fF,
            )
            reflected = annotate(
                ReadoutModel(mirror, VALIDATION_BASIS),
                fresh["ej2_ghz"],
                fresh["ej1_ghz"],
            )
            for key in metric_keys[:6]:
                np.testing.assert_allclose(
                    fresh[key],
                    reflected[key],
                    rtol=0,
                    atol=2e-8,
                    err_msg=f"Transmon label exchange, design {label}, {key}",
                )
            large = annotate(
                ReadoutModel(original, CONVERGENCE_BASIS),
                fresh["ej1_ghz"],
                fresh["ej2_ghz"],
            )
            if failures(large):
                raise AssertionError(
                    f"Convergence changed design {label}: {failures(large)}"
                )
            checks.append(
                dict(
                    design=label,
                    nominal=not failures(fresh),
                    mirror_symmetry=True,
                    larger_basis_failures=failures(large),
                    changes={key: large[key] - fresh[key] for key in metric_keys[:6]},
                )
            )
    seen = set()
    for row in map_rows:
        pair = (row["c1"], row["c2"])
        if pair in seen or pair[0] <= pair[1]:
            raise AssertionError(f"Duplicate/non-triangular map pixel: {pair}")
        seen.add(pair)
        if row["status"] == "validated_pass":
            if failures(row["point"]):
                raise AssertionError(
                    f"Saved map witness {pair}: {failures(row['point'])}"
                )
            if row["point"]["basis"] != asdict(VALIDATION_BASIS):
                raise AssertionError(
                    f"Saved map witness {pair} lacks large-basis validation"
                )
    scenarios = sampling_scenarios()
    recorded = load_json(data_dir / ROBUSTNESS_ARCHIVE / "sampling.json")["scenarios"]
    if recorded != scenarios:
        raise AssertionError(
            "Sampling vectors differ from the archived 224+64+12 design."
        )
    for row in samples:
        if row.get("error"):
            raise AssertionError(
                f"Saved numerical error: design {row['design']}, sample {row['id']}"
            )
        if row["first_pass"] is None:
            continue
        point = row["stages"][-1]["point"]
        reference = designs[row["design"]]
        fabricated = perturb(reference, row["errors"])
        if point["device"] != asdict(fabricated):
            raise AssertionError(
                "An accepted recalibration changed fabricated component inputs."
            )
        if point["basis"] != asdict(VALIDATION_BASIS):
            raise AssertionError("An accepted sample lacks large-basis validation.")
        earliest = next(
            (
                index
                for index, tier in enumerate(TIERS)
                if not reasons(point, tier, reference["operating_frequency_ghz"])
            ),
            None,
        )
        if earliest != row["first_pass"]:
            raise AssertionError(
                f"Tier credit mismatch in design {row['design']}, sample {row['id']}"
            )
        frequency = (
            1
            / (2 * np.pi * np.sqrt(fabricated.Cr_fF * 1e-15 * fabricated.Lr_nH * 1e-9))
            / 1e9
        )
        impedance = np.sqrt(fabricated.Lr_nH * 1e-9 / (fabricated.Cr_fF * 1e-15))
        np.testing.assert_allclose(
            frequency, BARE_RESONATOR_GHZ * (1 + row["errors"][5]), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(impedance, MODE_IMPEDANCE_OHM, rtol=0, atol=1e-10)
    analysis, cdfs = analyze_robustness(samples, designs)
    expected_counts = {
        "1": [30, 30, 151, 178, 300, 300],
        "2": [235, 235, 266, 266, 300, 300],
        "3": [211, 300, 300, 300, 300, 300],
    }
    if {
        label: stats["cumulative"] for label, stats in analysis.items()
    } != expected_counts:
        raise AssertionError("Recorded latest rescue counts were not reproduced.")
    if [cdfs[label]["eligible"] for label in designs] != [74, 224, 224]:
        raise AssertionError("Conditional hybridization CDF eligibility differs.")
    return dict(
        nominal_design_checks=checks,
        saved_map_witnesses=sum(r["status"] == "validated_pass" for r in map_rows),
        saved_robustness_witnesses=sum(r["first_pass"] is not None for r in samples),
        perturbation_vectors_exact=True,
        rescue_counts_exact=True,
        conditional_CDF_exact=True,
        note="Saved constraints audited for every witness; independent numerical recomputation covers the three nominal circuits and their mirrors/larger bases.",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=("cached", "verify", "map", "robustness", "all"),
        default="cached",
        help="cached is quick; verify checks physics; the other modes launch resumable numerical searches",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="read-only archived data root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="separate figures, tables and checkpoint directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="numerical worker processes; one BLAS thread each",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="optional per-stage wall budget; stops after current worker batch",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="pilot: number of map pixels or error vectors PER DESIGN",
    )
    parser.add_argument(
        "--pair",
        type=float,
        nargs=2,
        metavar=("C1_FF", "C2_FF"),
        help="map mode: search only this capacitance pair",
    )
    parser.add_argument(
        "--warm-start-archive",
        action="store_true",
        help="explicitly reuse archived candidate points and unchanged baseline stages; accepted witnesses are revalidated",
    )
    args = parser.parse_args()
    if (
        args.workers < 1
        or (args.limit is not None and args.limit < 1)
        or (args.hours is not None and args.hours <= 0)
    ):
        parser.error("workers, limit and hours must be positive")
    if args.pair and args.mode not in ("map", "all"):
        parser.error("--pair applies only to map/all mode")
    if args.pair and (not all(np.isfinite(args.pair)) or args.pair[0] <= args.pair[1]):
        parser.error("--pair requires finite C1>C2")
    if args.pair and any(value not in CAPACITANCE_GRID_FF for value in args.pair):
        parser.error(
            "--pair must use integer centers inside the configured 36-128 fF grid"
        )
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == args.data_dir or args.output_dir.is_relative_to(
        args.data_dir
    ):
        parser.error("Keep output outside the archived data directory.")
    return args


def main():
    """One entry point; never launch a long optimizer run by default."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_path = args.data_dir / MAP_ARCHIVE / "uniform_points.json"
    samples_path = args.data_dir / ROBUSTNESS_ARCHIVE / "samples.json"
    numerical = args.mode in ("map", "robustness", "all")
    if numerical:
        designs = selected_designs()
        map_rows = (
            run_map(args, designs)
            if args.mode in ("map", "all")
            else (load_json(map_path) if map_path.exists() else [])
        )
        samples = (
            run_robustness(args, designs)
            if args.mode in ("robustness", "all")
            else (load_json(samples_path) if samples_path.exists() else [])
        )
        origin = {
            "map": "Rerun map; archived robustness",
            "robustness": "Archived map; rerun robustness",
            "all": "Numerical rerun",
        }[args.mode]
        if args.limit or args.hours or args.pair:
            origin += " (potentially partial coverage)"
    else:
        map_rows = load_json(map_path)
        designs = load_json(args.data_dir / ROBUSTNESS_ARCHIVE / "designs.json")
        samples = load_json(samples_path)
        origin = "Archived study, 14-15 October 2026"
        if args.mode == "verify":
            audit = verify_results(map_rows, designs, samples, args.data_dir)
            save_json(args.output_dir / "verification.json", audit)
            print(
                "Verification passed: all saved witnesses, three independent nominal/mirror/convergence checks, exact sampling and rescue counts.",
                flush=True,
            )
    analysis, cdfs = analyze_robustness(samples, designs)
    produced = plot_results(map_rows, designs, analysis, cdfs, args.output_dir, origin)
    write_tables_and_summary(
        map_rows, designs, samples, analysis, cdfs, args.output_dir, origin
    )
    import scipy

    provenance = dict(
        schema=SCHEMA,
        run_mode=args.mode,
        origin=origin,
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        conditions=asdict(CONDITIONS),
        tiers=TIERS,
        seed=SEED,
        numerical_search=numerical,
        warm_start_archive=args.warm_start_archive,
        software=dict(
            numpy=np.__version__,
            scipy=scipy.__version__,
            matplotlib=matplotlib.__version__,
        ),
        archives={
            str(path): sha256(path.read_bytes()).hexdigest()
            for path in (map_path, samples_path)
            if path.exists()
        },
        figures=produced,
    )
    save_json(args.output_dir / "provenance.json", provenance)
    errors = [
        dict(design=row.get("design"), id=row.get("id"), error=row["error"])
        for row in samples
        if row.get("error")
    ]
    save_json(args.output_dir / "numerical_errors.json", errors)
    print(
        json.dumps(
            dict(
                mode=args.mode,
                map_pixels=len(map_rows),
                validated_map_pixels=sum(
                    r["status"] == "validated_pass" for r in map_rows
                ),
                robustness_samples=len(samples),
                numerical_errors=len(errors),
                cumulative_counts={
                    label: stats["cumulative"] for label, stats in analysis.items()
                },
                conditional_CDF_eligible={
                    label: stats["eligible"] for label, stats in cdfs.items()
                },
                output_dir=str(args.output_dir),
            ),
            indent=2,
        )
    )
    if errors:
        raise SystemExit(
            "Some numerical jobs failed; see numerical_errors.json. Failed jobs are not marked as passing."
        )


if __name__ == "__main__":
    main()
