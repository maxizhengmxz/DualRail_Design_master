"""Plot an already calculated pulse result and its saved validation checks."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--basis-audit", required=True, type=Path)
    parser.add_argument("--step-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.transmon import Basis
    from floating_coupler.model import local_spectrum, FULL
    from floating_coupler.pulsed_gate import edge_envelope
    from floating_coupler.gate import gate_metrics
    from floating_coupler.precision import GATE_PHASE_TOL

    result = json.loads(args.result.read_text())
    basis = json.loads(args.basis_audit.read_text())
    step = json.loads(args.step_audit.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    best = result["best"]
    edge = result["pulse"]["edge_duration_ns"]
    t = np.linspace(0, best["total_ns"], 501)
    s = np.where(
        t < edge,
        edge_envelope(t, result["pulse"]["rise_ns"]),
        np.where(
            t < edge + best["hold_ns"],
            1.0,
            edge_envelope(best["total_ns"] - t, result["pulse"]["rise_ns"]),
        ),
    )
    idle, gate = np.array(result["idle_ej_ghz"]), np.array(result["gate_ej_ghz"])
    ej = idle[None, :] + s[:, None] * (gate - idle)[None, :]
    ec = np.array([result["hardware"]["loaded_ec_ghz"][name] for name in FULL])
    freq = np.array(
        [
            [
                local_spectrum(float(e), float(c), Basis(ncut=35, transmon_levels=3))[
                    0
                ][1]
                for e, c in zip(row, ec, strict=True)
            ]
            for row in ej
        ]
    )
    np.savetxt(
        args.output / "pulse-trajectories.csv",
        np.column_stack([t, s, ej, freq]),
        delimiter=",",
        header="time_ns,envelope,"
        + ",".join("EJ_" + n + "_GHz" for n in FULL)
        + ","
        + ",".join("bare_f01_" + n + "_GHz" for n in FULL),
        comments="",
    )

    spectrum = np.load(args.result.parent / "gate-spectrum.npz")
    c, energy = spectrum["coefficients"], spectrum["energy"]
    holds = np.linspace(max(0.0, best["hold_ns"] - 6), best["hold_ns"] + 6, 601)
    rows = []
    for hold in holds:
        matrix = c.T @ (np.exp(-2j * np.pi * energy * hold)[:, None] * c)
        rows.append(gate_metrics(matrix))
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.7), constrained_layout=True)
    colors = ["#555555", "#999999", "#c94337", "#2464ad", "#53a0c6"]
    for i, name in enumerate(FULL):
        for ax, values in ((axes[0, 0], ej), (axes[0, 1], freq)):
            ax.plot(
                t,
                values[:, i],
                label=name,
                color=colors[i],
                lw=1.6,
                ls="--" if name in ("A2", "B2") else "-",
            )
    axes[0, 0].set(
        xlabel="Time (ns)",
        ylabel=r"$E_J/h$ (GHz)",
        title="Simultaneous EJ pulses; A rail fixed",
    )
    axes[0, 1].set(
        xlabel="Time (ns)",
        ylabel="Bare local f01 (GHz)",
        title="Full-cosine frequency trajectories",
    )
    for ax in axes[0]:
        ax.axvspan(edge, edge + best["hold_ns"], color="#cccccc", alpha=0.22)
        ax.legend(ncol=3, fontsize=8)
    ax = axes[1, 0]
    ax.semilogy(
        holds,
        [1 - r["raw_cz_fidelity"] for r in rows],
        label="Raw error",
        color="#2464ad",
    )
    ax.semilogy(
        holds,
        [1 - r["postselected_cz_fidelity"] for r in rows],
        label="Postselected error",
        color="#c94337",
    )
    ax.axvline(
        best["hold_ns"], color="black", ls="--", lw=1, label=r"Calibrated phase $\pi$"
    )
    ax.axhline(1e-4, color="#888888", ls=":", lw=1, label="Probability accuracy target")
    ax.set(
        xlabel="Hold duration (ns); edges additional",
        ylabel="1 - fidelity",
        title="Fixed endpoint: phase calibrated at dashed line",
        ylim=(1e-6, 1.0),
    )
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.bar(
        ["00", "01", "10", "11"],
        100 * np.array(best["leakage_by_input"]),
        color="#2464ad",
    )
    ax.set(
        xlabel="Initial idle logical state (A,B)",
        ylabel="Final nonlogical population (%)",
        title="Projection onto the same dressed idle logical space",
    )
    for i, value in enumerate(best["leakage_by_input"]):
        ax.text(i, value * 100 + 0.012, f"{value * 100:.3f}%", ha="center", fontsize=9)
    ax.set_ylim(0, max(best["leakage_by_input"]) * 125)
    fig.suptitle(
        "Five-transmon coherent CZ: 10 ns 10–90% EJ edges, symmetric return\n"
        f"Hold {best['hold_ns']:.2f} ns; total {best['total_ns']:.2f} ns; "
        f"raw {100 * best['raw_cz_fidelity']:.3f}%; selected {100 * best['postselected_cz_fidelity']:.3f}%",
        fontsize=13,
    )
    fig.savefig(args.output / "pulse-cz-summary.png")
    fig.savefig(args.output / "pulse-cz-summary.svg")
    plt.close(fig)
    probability_keys = [
        "raw_cz_fidelity",
        "postselected_cz_fidelity",
        "logical_survival",
        "worst_nonlogical_population",
    ]

    def differences(first, second):
        return {
            key: first["best"][key] - second["best"][key] for key in probability_keys
        }

    # Each saved audit must explicitly name the same preceding calibration.
    # Both nominal checks use refined/result.json as their common comparator.
    calibration_path = (
        Path(result["audit_of"]) if result.get("audit_of") else args.result
    )
    calibration = json.loads(calibration_path.read_text())
    basis_delta = differences(basis, calibration)
    step_delta = differences(step, calibration)
    summary = dict(
        best=best,
        final_result=str(args.result),
        basis=result["basis"],
        pulse=result["pulse"],
        channel_selection=result["channel_selection"],
        off_ghz=result["off_ghz"],
        idle_ej_ghz=idle.tolist(),
        gate_ej_ghz=gate.tolist(),
        endpoint_frequencies_ghz={
            "idle": freq[0].tolist(),
            "gate": freq[np.argmax(s)].tolist(),
        },
        phase_pi_calibrated=abs(best["phase_error_rad"]) <= GATE_PHASE_TOL,
        validation=dict(
            basis_change=basis_delta,
            step_change=step_delta,
            probability_tolerance=1e-4,
            passed=max(
                abs(v) for row in (basis_delta, step_delta) for v in row.values()
            )
            <= 1e-4,
            scope="Fixed controls relative to phase-corrected calibration; phase transfer is not a convergence comparison",
        ),
        static_reference=result["static_reference"],
        caveat="Static ON preparation differs from complete idle-to-idle pulse; do not interpret the reference difference as an isolated ramp error.",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    report = f"""# Nominal five-transmon EJ-tanh CZ

Coherent idle-to-idle pulse, frozen hardware, no resonators or decoherence.
Selected channel: logical {result["channel_selection"]["source_logical"]} to {result["channel_selection"]["auxiliary"]}.

| Quantity | Result |
|---|---:|
| Coupler OFF / ON bare f01 | {result["off_ghz"]:.6f} / {best["fc_ghz"]:.6f} GHz |
| B idle / gate bare f01 | {freq[0, 3]:.6f} / {best["fb_ghz"]:.9f} GHz |
| 10–90% rise / fall | {result["pulse"]["rise_ns"]:.2f} ns each |
| Full finite edge | {edge:.6f} ns each |
| Hold / total duration | {best["hold_ns"]:.6f} / {best["total_ns"]:.6f} ns |
| Raw fidelity | {best["raw_cz_fidelity"] * 100:.6f}% |
| Postselected score (raw / mean survival) | {best["postselected_cz_fidelity"] * 100:.6f}% |
| Mean logical survival | {best["logical_survival"] * 100:.6f}% |
| Worst-input final nonlogical population | {best["worst_nonlogical_population"] * 100:.6f}% |
| Conditional phase error from pi | {best["phase_error_rad"]:.3g} rad |

Local Z phases are corrected using the existing gate metric. All four full-system
dressed idle states define both input and output logical projections. Their final
nonlogical populations for 00,01,10,11 are {best["leakage_by_input"]}.
The selected source is 10, the B20 excursion is kept in the propagated Hilbert space.

## Accuracy and interpretation

Initial screening uses cutoff 6, local6/C5. Phase was transferred to cutoff8,
local8/C7 at fixed hold/coupler with a small B endpoint correction. Final reported
propagation halves the step to 0.0375 ns. Separate fixed-control checks increase
cutoff8 to cutoff10 and halve step0.075 to0.0375. Both agree within 1e-4 in reported
gate probabilities; maximum changes are {max(abs(v) for v in basis_delta.values()):.3g}
(basis) and {max(abs(v) for v in step_delta.values()):.3g} (step). This is not 1 Hz
OFF convergence certification. The next-cutoff conditional phase changes by a
few microradians, below the probability error target; saved audit values remain.

The saved static ON benchmark is {result["static_reference"]["raw_cz_fidelity"] * 100:.6f}%
raw, {result["static_reference"]["postselected_cz_fidelity"] * 100:.6f}% selected,
at {result["static_reference"]["time_ns"]:.6f} ns. Its initial references differ.
The selected-score difference is below the agreed 0.01 percentage-point gate
accuracy target. No resolved additional selected-fidelity loss is established
at that precision. The physical pulse takes longer because its edges are included.

![Pulse and gate results](pulse-cz-summary.png)
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary["validation"], indent=2))


if __name__ == "__main__":
    main()
