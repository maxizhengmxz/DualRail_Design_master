"""Whole-pulse ideal-bandwidth propagation, including sinc pre/post tails.

The filter acts on EJ, not SQUID flux. A real even brick-wall response retains
time symmetry: U_full = U_half.T @ U_half. Rebuild the filtered waveform for EACH
hold duration; a constant ON spectrum cannot replace the ringing plateau.
"""

from functools import lru_cache
from time import perf_counter

import numpy as np

from pulse_bandwidth import brickwall, edge_duration, waveform
from .gate import gate_metrics
from .pulsed_gate import propagate_schedule


@lru_cache(maxsize=48)
def filtered_samples(
    shape, rise_ns, hold_ns, cutoff_ghz=1.0, padding_ns=16.0, span_ns=4096.0, power=20
):
    duration = edge_duration(shape, rise_ns)
    half = duration + hold_ns / 2 + padding_ns
    if half >= span_ns / 4:
        raise ValueError("FFT padding span too small")
    dt = span_ns / 2**power
    t = (np.arange(2**power) - 2 ** (power - 1)) * dt
    command = waveform(t, shape, rise_ns, hold_ns)
    delivered = brickwall(command, dt, cutoff_ghz)
    # Store only the propagation window; interpolation needs one extra sample.
    mask = (t >= -half - dt) & (t <= dt)
    return (
        t[mask].copy(),
        delivered[mask].copy(),
        dict(
            nominal_pulse_ns=2 * duration + hold_ns,
            simulated_total_ns=2 * half,
            padding_each_side_ns=padding_ns,
            maximum_full_excursion_distortion=float(np.max(abs(delivered - command))),
            boundary_excursion=float(abs(np.interp(-half, t, delivered))),
            FFT_dt_ns=dt,
            FFT_span_ns=span_ns,
        ),
    )


def filtered_gate(
    pulse,
    fb,
    fc,
    shape,
    rise_ns,
    hold_ns,
    *,
    cutoff_ghz=1.0,
    padding_ns=16.0,
    step_ns=0.075,
    span_ns=4096.0,
    power=20,
):
    started = perf_counter()
    t, s, diagnostic = filtered_samples(
        shape, float(rise_ns), float(hold_ns), cutoff_ghz, padding_ns, span_ns, power
    )
    half = diagnostic["simulated_total_ns"] / 2
    gate_ej = pulse.controls(fb, fc)
    ham = pulse.hamiltonian(gate_ej)
    vectors, integration = propagate_schedule(
        pulse.h_idle,
        ham - pulse.h_idle,
        pulse.inputs,
        lambda u: np.interp(u - half, t, s),
        half,
        step_ns,
    )
    # Transpose, not adjoint: the second half reverses a real symmetric H(t).
    matrix = vectors.T @ vectors
    result = gate_metrics(matrix)
    result.update(
        diagnostic,
        shape=shape,
        rise_ns=rise_ns,
        hold_ns=float(hold_ns),
        total_ns=diagnostic["nominal_pulse_ns"],
        fb_ghz=float(fb),
        fc_ghz=float(fc),
        worst_nonlogical_population=max(result["leakage_by_input"]),
        integration=integration,
        seconds=perf_counter() - started,
        norm_error=float(np.max(abs(vectors.conj().T @ vectors - np.eye(4)))),
    )
    return result
