"""Full-cosine SQUID inversion and explicitly specified pure-dephasing models."""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import sici
from floating_coupler.design import ej_for_frequency

ECHO_FACTOR = np.pi * np.sqrt(2 * np.log(2))


def squid_design(ec, lower, upper, ncut=40):
    if not 0 < lower < upper:
        raise ValueError("Require positive, increasing SQUID frequency endpoints")
    minimum = ej_for_frequency(lower, ec, ncut)
    maximum = ej_for_frequency(upper, ec, ncut)
    return dict(
        ec_ghz=float(ec),
        lower_ghz=lower,
        upper_ghz=upper,
        ej_max_ghz=maximum,
        ej_min_ghz=minimum,
        ej_large_ghz=(maximum + minimum) / 2,
        ej_small_ghz=(maximum - minimum) / 2,
        asymmetry=minimum / maximum,
    )


def ej_at_flux(phi, design):
    return design["ej_max_ghz"] * np.sqrt(
        np.cos(np.pi * phi) ** 2 + design["asymmetry"] ** 2 * np.sin(np.pi * phi) ** 2
    )


def flux_for_frequency(frequency, design, ncut=40):
    ej = ej_for_frequency(frequency, design["ec_ghz"], ncut)
    value = ((ej / design["ej_max_ghz"]) ** 2 - design["asymmetry"] ** 2) / (
        1 - design["asymmetry"] ** 2
    )
    if not -1e-9 <= value <= 1 + 1e-9:
        raise ValueError("Frequency outside the specified SQUID tuning range")
    return float(np.arccos(np.sqrt(np.clip(value, 0, 1))) / np.pi)


def dephasing_time(asd_hz, kind, low_cutoff_hz=1.0):
    """First-order 1/f 1/e envelope time; infinity if no cutoff-limited crossing."""
    if asd_hz <= 0:
        return float("inf")
    if kind == "echo":
        return 1 / (ECHO_FACTOR * asd_hz)
    if kind != "ramsey" or low_cutoff_hz <= 0:
        raise ValueError("Require echo/ramsey and positive low-frequency cutoff")

    def exponent(t):
        x = low_cutoff_hz * t
        if x < 1e-4:
            return (
                2
                * np.pi**2
                * asd_hz**2
                * t
                * t
                * np.log(np.exp(1.5 - np.euler_gamma) / (2 * np.pi * x))
            )
        integral = (
            np.sin(np.pi * x) ** 2 / (2 * x * x)
            + np.pi * np.sin(2 * np.pi * x) / (2 * x)
            - np.pi**2 * sici(2 * np.pi * x)[1]
        )
        return 2 * asd_hz**2 * t * t * integral

    grid = np.geomspace(1e-13, 10 / low_cutoff_hz, 180)
    hit = next((i for i, t in enumerate(grid) if exponent(t) >= 1), None)
    return (
        float("inf")
        if hit is None
        else brentq(
            lambda t: exponent(t) - 1, grid[max(0, hit - 1)], grid[hit], xtol=1e-16
        )
    )


def exponential_cz(
    derivatives, amplitude_microphi0, kind, duration_ns=100.0, low_cutoff_hz=1.0
):
    """Ideal source10/B20 cycle with correlated B-state phase derivatives.

    Derivatives: B0,B1,B20 rows; physical independent B1/B2 noise columns.
    Reference 1/f times are converted to exponential rates per coherence.
    This substitution is phenomenological, not a Gaussian short-gate prediction.
    """
    derivatives = np.asarray(derivatives)
    refs = derivatives[[0, 1, 0, 1, 2]]
    distances = (
        amplitude_microphi0
        * 1e-6
        * np.linalg.norm(refs[:, None] - refs[None, :], axis=2)
    )
    gamma = np.zeros((5, 5))
    for i in range(5):
        for j in range(i):
            gamma[i, j] = gamma[j, i] = 1 / dephasing_time(
                distances[i, j], kind, low_cutoff_hz
            )
    center = np.eye(5) - np.ones((5, 5)) / 5
    spectrum = np.linalg.eigvalsh(-0.5 * center @ gamma @ center)
    if spectrum.min() < -max(1e-7, np.linalg.norm(gamma) * 1e-10):
        raise ValueError("Reference-rate matrix is not a valid dephasing generator")
    time = duration_ns * 1e-9
    ham = np.zeros((5, 5))
    ham[2, 4] = ham[4, 2] = 1 / (2 * time)
    identity, target = np.eye(5), np.diag([1, 1, -1, 1])
    generator = -2j * np.pi * (
        np.kron(identity, ham) - np.kron(ham.T, identity)
    ) - np.diag(gamma.ravel(order="F"))
    propagator = expm(generator * time)
    trace = overlap = 0.0
    for i in range(4):
        for j in range(4):
            op = np.zeros((5, 5))
            op[i, j] = 1
            evolved = (propagator @ op.ravel(order="F")).reshape((5, 5), order="F")[
                :4, :4
            ]
            overlap += (target.conj().T @ evolved @ target)[i, j].real
            if i == j:
                trace += np.trace(evolved).real
    raw = (trace + overlap) / 20
    survival = trace / 4
    return dict(
        raw_fidelity=float(raw),
        postselected_fidelity=float(raw / survival),
        survival=float(survival),
    )
