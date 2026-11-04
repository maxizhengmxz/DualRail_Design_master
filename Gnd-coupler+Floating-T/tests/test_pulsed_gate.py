"""Pulse units, noncommuting edge ordering and reverse-edge transport."""

import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm
from floating_coupler.pulsed_gate import (
    PulseModel,
    edge_envelope,
    edge_parameters,
    propagate_edge,
    symmetric_pulse_matrix,
)


class PulseTests(unittest.TestCase):
    def test_filtered_whole_pulse_against_direct_propagation(self):
        from types import SimpleNamespace
        from floating_coupler.filtered_pulse import filtered_gate, filtered_samples
        from floating_coupler.gate import gate_metrics

        h0 = np.diag([0.01, 0.03, 0.05, 0.07])
        delta = np.array(
            [
                [0.02, 0.015, 0, 0.005],
                [0.015, -0.01, 0.012, 0],
                [0, 0.012, 0.03, -0.007],
                [0.005, 0, -0.007, -0.02],
            ]
        )
        pulse = SimpleNamespace(
            h_idle=h0,
            inputs=np.eye(4),
            controls=lambda fb, fc: np.ones(5),
            hamiltonian=lambda ej: h0 + delta,
        )
        kwargs = dict(padding_ns=2.0, span_ns=512.0, power=16)
        result = filtered_gate(
            pulse, 4.87, 7.05, "cosine", 0.5, 2.3, step_ns=0.015, **kwargs
        )
        t, s, diagnostic = filtered_samples("cosine", 0.5, 2.3, 1.0, **kwargs)
        total = diagnostic["simulated_total_ns"]

        def rhs(u, v):
            value = np.interp(-abs(u - total / 2), t, s)
            return (-2j * np.pi * (h0 + value * delta) @ v.reshape(4, 4)).ravel()

        direct = (
            solve_ivp(
                rhs,
                (0, total),
                np.eye(4, dtype=complex).ravel(),
                method="DOP853",
                rtol=1e-10,
                atol=1e-12,
            )
            .y[:, -1]
            .reshape(4, 4)
        )
        expected = gate_metrics(direct)
        for key in ("raw_cz_fidelity", "postselected_cz_fidelity", "logical_survival"):
            self.assertAlmostEqual(result[key], expected[key], delta=2e-8)
        self.assertLess(result["norm_error"], 1e-11)

    def test_cosine_edge_uses_same_ten_ninety_convention(self):
        from pulse_bandwidth import edge, edge_duration

        duration = edge_duration("cosine", 10.0)
        np.testing.assert_allclose(
            edge(np.array([duration / 2 - 5, duration / 2 + 5]), "cosine", 10.0),
            [0.1, 0.9],
            atol=1e-14,
        )
        self.assertAlmostEqual(float(edge(0.0, "cosine", 10.0)), 0.0)
        self.assertAlmostEqual(float(edge(duration, "cosine", 10.0)), 1.0)

    def test_brickwall_keeps_passband_and_removes_stopband(self):
        from pulse_bandwidth import brickwall

        t = np.arange(4096) / 128
        wanted = 0.3 + np.sin(2 * np.pi * 0.25 * t)
        signal = wanted + 0.2 * np.sin(2 * np.pi * 3 * t)
        np.testing.assert_allclose(brickwall(signal, 1 / 128, 1.0), wanted, atol=1e-13)

    def test_exact_endpoints_and_ten_ninety_width(self):
        tau, duration = edge_parameters(10.0)
        self.assertAlmostEqual(float(edge_envelope(0.0)), 0.0)
        self.assertAlmostEqual(float(edge_envelope(duration)), 1.0)
        self.assertAlmostEqual(float(edge_envelope(duration / 2 - 5)), 0.1)
        self.assertAlmostEqual(float(edge_envelope(duration / 2 + 5)), 0.9)

    def test_noncommuting_edge_against_direct_schrodinger_equation(self):
        h0 = np.array([[0.03, 0.015], [0.015, -0.03]])
        delta = np.array([[0.01, -0.02], [-0.02, -0.01]])
        _, duration = edge_parameters(1.0)
        numerical, _ = propagate_edge(h0, delta, np.eye(2), 1.0, 0.02)
        direct = (
            solve_ivp(
                lambda t, y: (
                    -2j * np.pi * (h0 + edge_envelope(t, 1.0) * delta) @ y.reshape(2, 2)
                ).ravel(),
                (0, duration),
                np.eye(2, dtype=complex).ravel(),
                method="DOP853",
                rtol=1e-12,
                atol=1e-13,
            )
            .y[:, -1]
            .reshape(2, 2)
        )
        np.testing.assert_allclose(numerical, direct, atol=2e-9)

    def test_symmetric_fall_is_transpose_not_inverse(self):
        h0 = np.array([[0.03, 0.015], [0.015, -0.03]])
        delta = np.array([[0.01, -0.02], [-0.02, -0.01]])
        up, _ = propagate_edge(h0, delta, np.eye(2), 1.0, 0.02)
        e, v = eigh(h0 + delta)
        hold = 2.3
        combined = symmetric_pulse_matrix(e, v, up, hold)
        expected = up.T @ expm(-2j * np.pi * (h0 + delta) * hold) @ up
        np.testing.assert_allclose(combined, expected, atol=1e-12)
        self.assertGreater(
            np.linalg.norm(
                combined - up.conj().T @ expm(-2j * np.pi * (h0 + delta) * hold) @ up
            ),
            0.1,
        )

    def test_changing_EJ_keeps_off_diagonal_cos_terms_in_one_fixed_frame(self):
        from core.transmon import Basis
        from floating_coupler.model import Model, FULL
        from floating_coupler.design import ej_for_frequency

        ec = np.full(5, 0.2)
        frequencies = [4.61, 4.61, 9.8, 5.39, 5.39]
        ej = np.array([ej_for_frequency(f, e, 20) for f, e in zip(frequencies, ec)])
        j = np.zeros((5, 5))
        j[0, 1] = j[1, 0] = 0.02
        j[3, 4] = j[4, 3] = 0.015
        model = Model(FULL, ec, ej, j, Basis(ncut=20, transmon_levels=3), {})
        pulse = PulseModel(
            model,
            9.8,
            4.87,
            7.05,
            local_levels=3,
            coupler_levels=3,
            excitation_cutoff=10,
            ncut=20,
        )
        gate = pulse.controls(4.87, 7.05)
        midpoint = 0.5 * (pulse.idle_ej + gate)
        np.testing.assert_allclose(
            pulse.hamiltonian(midpoint),
            0.5 * (pulse.h_idle + pulse.hamiltonian(gate)),
            atol=1e-12,
        )
        delta = pulse.hamiltonian(gate) - pulse.h_idle
        self.assertGreater(np.linalg.norm(delta - np.diag(np.diag(delta))), 0.01)
