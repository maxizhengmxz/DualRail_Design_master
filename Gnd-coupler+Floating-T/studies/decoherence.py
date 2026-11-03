"""Optional isolated-rail and coupler-induced longitudinal flux-noise study."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment, minimize_scalar
from core.transmon import Basis
from floating_coupler.fabrication import five_model
from floating_coupler.pairs import pair_spectrum
from floating_coupler.model import local_spectrum
from floating_coupler.design import ej_for_frequency
from .noise import (
    squid_design,
    flux_for_frequency,
    ej_at_flux,
    dephasing_time,
    exponential_cz,
)


class Subsystem:
    """Fixed loaded EC/J; omit other-rail dynamics without redesigning capacitors."""

    def __init__(self, model, rail="B", levels=7):
        self.model = model
        self.indices = (0, 1) if rail == "A" else (3, 4)
        self.levels = levels

    def energies(self, ej, coupler):
        levels = self.levels
        pair = pair_spectrum(replace(self.model, ej=ej), self.indices, levels, 40)
        wanted = [*pair["one"], pair["auxiliary"]["20"]]
        if not coupler:
            return pair["energy"][wanted] * 1e9
        ce, nc = local_spectrum(
            ej[2], self.model.ec[2], Basis(ncut=40, transmon_levels=levels)
        )
        n = levels**2
        ham = np.diag((pair["energy"][:, None] + ce[None, :]).ravel())
        for k, i in enumerate(self.indices):
            ham += self.model.j[i, 2] * np.kron(pair["charge"][k], nc)
        parity = (
            pair["parity"][:, None] * (-1.0) ** np.arange(levels)[None, :]
        ).ravel()
        assigned = {}
        groups = [
            (
                1,
                ["gg", "aux", "aux11", "aux02"],
                [0, *[pair["auxiliary"][k] for k in ["20", "11", "02"]]],
            ),
            (-1, ["low", "high"], list(pair["one"])),
        ]
        for sign, labels, states in groups:
            sector = np.flatnonzero(parity == sign)
            energy, v = eigh(
                ham[np.ix_(sector, sector)],
                subset_by_index=(0, min(47, len(sector) - 1)),
            )
            refs = [
                np.searchsorted(sector, np.ravel_multi_index((i, 0), (n, levels)))
                for i in states
            ]
            _, columns = linear_sum_assignment(-(abs(v[refs, :]) ** 2))
            for name, col in zip(labels, columns):
                assigned[name] = energy[col]
        return (
            np.array([assigned[k] - assigned["gg"] for k in ["low", "high", "aux"]])
            * 1e9
        )

    def at_frequencies(self, mean, delta, fc=None):
        ej = self.model.ej.copy()
        for i, f in zip(self.indices, [mean + delta / 2, mean - delta / 2]):
            ej[i] = ej_for_frequency(f, self.model.ec[i], 40)
        if fc is not None:
            ej[2] = ej_for_frequency(fc, self.model.ec[2], 40)
        return self.energies(ej, fc is not None)

    def derivatives(self, mean, delta, designs, fc=None, step=1e-5):
        indices = list(self.indices) + ([2] if fc is not None else [])
        frequencies = [mean + delta / 2, mean - delta / 2] + (
            [fc] if fc is not None else []
        )
        flux = [flux_for_frequency(f, d) for f, d in zip(frequencies, designs)]
        ej = self.model.ej.copy()
        for i, p, d in zip(indices, flux, designs):
            ej[i] = ej_at_flux(p, d)
        columns = []
        for i, p, d in zip(indices, flux, designs):
            upper, lower = ej.copy(), ej.copy()
            upper[i], lower[i] = ej_at_flux(p + step, d), ej_at_flux(p - step, d)
            columns.append(
                (
                    self.energies(upper, fc is not None)
                    - self.energies(lower, fc is not None)
                )
                / (2 * step)
            )
        return self.energies(ej, fc is not None), np.column_stack(columns), flux


def coupler_participation(base):
    """Five-mode OFF participation; separate from independent-rail dephasing.

    Physical states use the same four-reference canonical embedding as OFF.
    Decay weights use local a_C and require a bath model to become rates.
    """
    from floating_coupler.pairs import PairModel
    from floating_coupler.off_observables import physical_single_excitation_references

    model, _ = five_model(base, {}, {})
    system = PairModel(model, local_levels=7, pair_states=18, coupler_levels=7, ncut=40)
    ham, _ = system.hamiltonian(base["off_ghz"])
    odd = np.flatnonzero(system.parity == -1)
    even = np.flatnonzero(system.parity == 1)
    energies, v = eigh(ham[np.ix_(odd, odd)], subset_by_index=(0, 47))
    refs = physical_single_excitation_references(system)[:, odd]
    projection = refs @ v
    _, cols = linear_sum_assignment(-(abs(projection) ** 2))
    u, sv, vh = np.linalg.svd(projection[:, cols])
    embedding = v[:, cols] @ (u @ vh).T
    ground_energy, g = eigh(ham[np.ix_(even, even)], subset_by_index=(0, 0))
    ground = np.zeros(len(ham))
    ground[even] = g[:, 0]
    physical = np.zeros((len(ham), 4))
    physical[odd] = embedding
    levels = np.indices(system.dims)[1].ravel()
    lowering = np.kron(
        np.kron(
            np.eye(system.dims[0]), np.diag(np.sqrt(np.arange(1, system.dims[1])), 1)
        ),
        np.eye(system.dims[2]),
    )
    jump = ground @ lowering @ physical
    result = {}
    for i, label in enumerate(["1000", "0100", "0010", "0001"]):
        result[label] = dict(
            coupler_excited_probability=float(
                np.sum(abs(physical[levels > 0, i]) ** 2)
            ),
            coupler_mean_level=float(levels @ abs(physical[:, i]) ** 2),
            ground_decay_weight=float(jump[i] ** 2),
        )
    return dict(
        fc_ghz=base["off_ghz"],
        states=result,
        minimum_subspace_overlap=float(min(sv) ** 2),
        scope="Five-mode canonical dressed physical single-excitation basis; no resonators. Local coupler participation includes virtual vacuum dressing. Ground-decay weights are not total participation or a T1 prediction.",
    )


def run(args):
    base = json.loads(Path(args.input).read_text())
    model, _ = five_model(base, {}, {})
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    amplitudes = np.unique(
        np.r_[
            np.geomspace(args.amplitudes[0], args.amplitudes[1], 81),
            [
                a
                for a in [1.0, 10.0, 100.0]
                if args.amplitudes[0] <= a <= args.amplitudes[1]
            ],
        ]
    )
    bdesign = [squid_design(model.ec[i], *args.rail_tuning) for i in [3, 4]]
    cdesign = squid_design(model.ec[2], *args.coupler_tuning)
    off, on = base["off_ghz"], base["gate_seed"]["fc_ghz"]
    idle = {
        rail: base["readouts"][rail]["case"]["bare_frequency_guide_ghz"]
        for rail in ["A", "B"]
    }
    gate = base["gate_seed"]["fb_ghz"]
    isolated = {}
    system = Subsystem(model, levels=args.levels)
    for position, mean in [("idle", idle["B"]), ("gate", gate)]:
        energy, D, flux = system.derivatives(mean, 0, bdesign)
        d = bdesign[0]
        p = flux[0]
        h = 1e-5

        def b1_frequency(phi):
            en, _ = local_spectrum(
                ej_at_flux(phi, d),
                model.ec[3],
                Basis(ncut=40, transmon_levels=args.levels),
            )
            return en[1] * 1e9

        ref = (b1_frequency(p + h) - b1_frequency(p - h)) / (2 * h)
        rows = []
        for amplitude in amplitudes:
            trans = {}
            for label, g in [
                ("logical_B", D[1] - D[0]),
                ("B20_B0", D[2] - D[0]),
                ("B1_01", np.array([ref])),
            ]:
                trans[label] = {
                    kind: 1e6
                    * dephasing_time(
                        amplitude * 1e-6 * np.linalg.norm(g), kind, args.low_cutoff
                    )
                    for kind in ["echo", "ramsey"]
                }
            row = dict(amplitude_microphi0=float(amplitude), times_us=trans)
            if position == "gate":
                row["fidelity"] = {
                    kind: exponential_cz(
                        D, amplitude, kind, args.gate_time, args.low_cutoff
                    )
                    for kind in ["echo", "ramsey"]
                }
            rows.append(row)
        isolated[position] = dict(
            bare_frequency_ghz=mean,
            logical_frequency_mhz=float((energy[1] - energy[0]) / 1e6),
            derivatives_hz_per_phi0=D.tolist(),
            flux_phi0=flux,
            rows=rows,
        )
    coupled = {}
    for rail in ["A", "B"]:
        sub = Subsystem(model, rail, args.levels)
        fit = minimize_scalar(
            lambda delta: np.diff(sub.at_frequencies(idle[rail], delta, off)[:2])[0],
            bounds=(-0.15, 0.15),
            method="bounded",
            options=dict(xatol=1e-8, maxiter=24),
        )
        delta = float(fit.x)
        for position, mean in [("idle", idle[rail])] + (
            [("gate", gate)] if rail == "B" else []
        ):
            points = []
            for bias, fc in [("OFF", off), ("ON", on)]:
                ej = model.ej.copy()
                for i, f in zip(sub.indices, [mean + delta / 2, mean - delta / 2]):
                    ej[i] = ej_for_frequency(f, model.ec[i], 40)
                phi = flux_for_frequency(fc, cdesign)
                ej[2] = ej_at_flux(phi, cdesign)
                energy = sub.energies(ej, True)
                up, down = ej.copy(), ej.copy()
                up[2] = ej_at_flux(phi + 1e-5, cdesign)
                down[2] = ej_at_flux(phi - 1e-5, cdesign)
                D = ((sub.energies(up, True) - sub.energies(down, True)) / (2e-5))[
                    :, None
                ]
                flux = [phi]
                noise = np.array([D[1, 0] - D[0, 0], D[2, 0] - D[0, 0]]) * 100e-6
                trans = {
                    label: {
                        kind: dephasing_time(abs(noise[k]), kind, args.low_cutoff)
                        for kind in ["echo", "ramsey"]
                    }
                    for k, label in enumerate(["logical", "aux_B0"])
                }
                points.append(
                    dict(
                        bias=bias,
                        fc_ghz=fc,
                        flux_phi0=flux,
                        energy_hz=energy.tolist(),
                        reference_times_seconds=trans,
                        derivatives_hz_per_phi0=D.tolist(),
                    )
                )
            response = []
            for fc in np.linspace(*args.coupler_tuning, args.response_points):
                en = sub.at_frequencies(mean, delta, float(fc))
                response.append(
                    dict(fc_ghz=float(fc), logical_mhz=float((en[1] - en[0]) / 1e6))
                )
            coupled[rail + "_" + position] = dict(
                mean_bare_ghz=mean,
                differential_bare_ghz=delta,
                points=points,
                frequency_response=response,
            )
    result = dict(
        schema="coupler-decoherence-v1",
        input=base,
        scope="Isolated B pair; separate independent A1-A2-C/C-B1-B2 coupler-noise estimates. Fixed loaded EC/J; no resonators or other-rail dynamics.",
        assumptions=dict(
            rail_tuning=list(args.rail_tuning),
            coupler_tuning=list(args.coupler_tuning),
            low_cutoff_hz=args.low_cutoff,
            gate_time_ns=args.gate_time,
            levels=args.levels,
            coupler_reference_amplitude_microphi0=100.0,
            gate_model="First-order echo/Ramsey references substituted into exponential pure dephasing; full B20 correlations; A noiseless. No T1, quadratic/transverse noise or ramp errors.",
        ),
        b_squid_designs=bdesign,
        c_squid_design=cdesign,
        isolated=isolated,
        coupled=coupled,
    )
    if getattr(args, "participation", False):
        (out / "participation.json").write_text(
            json.dumps(coupler_participation(base), indent=2)
        )
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    from .decoherence_report import generate

    return generate(out)
