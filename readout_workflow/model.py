"""Single-junction dual-rail circuit, in a truncated local eigenstate basis.

All Hamiltonians and eigenvalues are in GHz (H/h). There are no flux variables
or junction capacitances. The complete capacitance matrix sets both the local
charging energies and charge-charge interactions.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge as e, h, hbar
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment, minimize_scalar
from threadpoolctl import threadpool_limits

# Preserve standalone notebook use from this directory.
if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.electrostatics import charge_coupling_ghz, exchange_j01_ghz, loaded_charging_energies_ghz
from core.transmon import Basis, transmon_spectrum

# Existing callers may still import the historical private helper.
_transmon = transmon_spectrum


@dataclass(frozen=True)
class Device:
    # Physical capacitances, all in fF. No EC values disguised as capacitances.
    C1_shunt_fF: float = 54.71839021084497
    C2_shunt_fF: float = 44.63297219962931
    C12_fF: float = 2.3
    C1r_fF: float = 15.0
    C2r_fF: float = 15.0
    Cr_fF: float = 531.0
    Lr_nH: float = 1.33

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
            self.C1_shunt_fF, self.C2_shunt_fF, self.C12_fF,
            self.C1r_fF, self.C2r_fF, self.Cr_fF,
        )
        return 1e-15 * np.array([
            [c1 + c12 + c1r, -c12, -c1r],
            [-c12, c2 + c12 + c2r, -c2r],
            [-c1r, -c2r, cr + c1r + c2r],
        ])


class ReadoutModel:
    """Exact diagonalization within a truncated transmon x transmon x Fock basis."""

    def __init__(self, device=None, basis=None):
        device = Device() if device is None else device
        basis = Basis() if basis is None else basis
        self.device, self.basis = device, basis
        self.cinv = np.linalg.inv(device.capacitance_matrix())
        self.J_ghz = charge_coupling_ghz(self.cinv)
        # H_C = Q.T C^-1 Q / 2, Q_transmon = 2e*n.
        self.ec_ghz = loaded_charging_energies_ghz(self.cinv)
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
        en1, n1 = transmon_spectrum(ej1_ghz, self.ec_ghz[0], b)
        en2, n2 = transmon_spectrum(ej2_ghz, self.ec_ghz[1], b)
        it = np.eye(b.transmon_levels)
        ir = np.eye(b.resonator_levels)
        n1_pair, n2_pair = np.kron(n1, it), np.kron(it, n2)
        pair = np.diag((en1[:, None] + en2[None, :]).ravel())
        pair += self.J_ghz[0, 1] * (n1_pair @ n2_pair)
        _, pair_states = eigh(pair)
        a = np.diag(np.sqrt(np.arange(1, b.resonator_levels)), 1)
        # U = I_pair tensor diag(i**n) makes the resonator charge real.
        # H_real = U^dagger H_complex U. Fock-state references acquire only
        # global phases, and transmon charge operators commute with U; energies,
        # overlaps, hybridization and conditional shifts are unchanged.
        qr = self.qr_zpf * (a.T + a)
        ham = np.kron(pair, ir)
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
            "J12_ghz": float(self.J_ghz[0, 1]),
            "J01_12_mhz": 1000 * exchange_j01_ghz(self.J_ghz[0, 1], n1[0, 1], n2[0, 1]),
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



def find_readout_points(model, ej1_values, ej2_values, min_hybridization=0.95, xtol=1e-4):
    """On each EJ2 slice, refine interior minima of rail splitting versus EJ1.

    Same candidate-search criterion as the original workflow. Readout mismatch
    is evaluated at the candidates; it is not the search objective.
    """
    ej1_values, ej2_values = np.asarray(ej1_values), np.asarray(ej2_values)
    for values, minimum in ((ej1_values, 3), (ej2_values, 1)):
        if values.ndim != 1 or len(values) < minimum or not np.all(np.isfinite(values)):
            raise ValueError("Supply finite one-dimensional EJ grids")
        if np.any(values <= 0) or np.any(np.diff(values) <= 0):
            raise ValueError("EJ grids must be positive and strictly increasing")
    if not 0 <= min_hybridization <= 1 or not np.isfinite(xtol) or xtol <= 0:
        raise ValueError("Invalid hybridization threshold or minimizer tolerance")
    found, scan = [], []
    with threadpool_limits(limits=1):
        for ej2 in ej2_values:
            cache = {}
            def evaluate(ej1):
                if float(ej1) not in cache:
                    cache[float(ej1)] = model.evaluate(float(ej1), float(ej2))
                return cache[float(ej1)]
            coarse = [evaluate(ej1) for ej1 in ej1_values]
            scan.extend(coarse)
            splitting = np.array([point["splitting_ghz"] for point in coarse])
            accepted = []
            for i in range(1, len(ej1_values)-1):
                if not (splitting[i] <= splitting[i-1] and splitting[i] <= splitting[i+1]
                        and (splitting[i] < splitting[i-1] or splitting[i] < splitting[i+1])):
                    continue
                optimum = minimize_scalar(
                    lambda ej1: evaluate(ej1)["splitting_ghz"],
                    bounds=(ej1_values[i-1], ej1_values[i+1]),
                    method="bounded", options={"xatol": xtol},
                )
                if not optimum.success:
                    raise RuntimeError(f"EJ1 refinement failed at EJ2={ej2}")
                point = evaluate(optimum.x)
                if point["hybridization"] >= min_hybridization and all(
                    abs(optimum.x-old) > 10*xtol for old in accepted
                ):
                    found.append(point)
                    accepted.append(optimum.x)
    return found, scan


def save_results(path, model, points, scan, settings, diagnostics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": "single-junction-readout-v2", "device": asdict(model.device),
                "basis": asdict(model.basis), "settings": settings,
                "hamiltonian_units": "GHz", "junction_capacitance": "absent",
                "diagnostics": diagnostics or []}
    keys = list(scan[0]) if scan else list(points[0]) if points else []
    np.savez_compressed(path, metadata=json.dumps(metadata), columns=np.array(keys),
                        points=np.array([[p[k] for k in keys] for p in points]).reshape(len(points), len(keys)),
                        scan=np.array([[p[k] for k in keys] for p in scan]).reshape(len(scan), len(keys)))
    return path
