"""Plot saved calibrated pulse-shape results; no quantum calibration on import/run."""

import argparse
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]


def tuning_schematic(folder):
    """Local bare f01 trajectories from saved EJ endpoints; no coupled propagation."""
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.fft import rfft, irfft, rfftfreq
    from pulse_bandwidth import waveform, edge_duration
    from floating_coupler.design import ej_for_frequency
    from core.transmon import Basis, transmon_spectrum

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained", sharex=True)
    trajectories = {}
    metadata = {}
    for col, shape in enumerate(("tanh", "cosine")):
        result = json.loads((folder / f"{shape}-10ns/result.json").read_text())
        best = result["best"]
        ec = result["hardware"]["loaded_ec_ghz"]
        idle = dict(result["hardware"]["bare_idle_frequencies_ghz"])
        idle["C"] = result["input_snapshot"]["off_ghz"]
        target = dict(idle, C=best["fc_ghz"], B1=best["fb_ghz"], B2=best["fb_ghz"])
        names = ["A1", "A2", "C", "B1", "B2"]
        ei = np.array([ej_for_frequency(idle[n], ec[n], 35) for n in names])
        eg = np.array([ej_for_frequency(target[n], ec[n], 35) for n in names])
        duration = edge_duration(shape, 10.0)
        total = best["nominal_pulse_ns"]
        hold = best["hold_ns"]
        count = 2**20
        dt = 4096 / count
        grid = (np.arange(count) - count // 2) * dt
        command = waveform(grid, shape, 10.0, hold)
        freq = rfftfreq(count, dt)
        transfer = (freq < 1.0).astype(float)
        transfer[np.isclose(freq, 1.0, atol=1e-12, rtol=0)] = 0.5
        filtered = irfft(rfft(command) * transfer, n=count)
        t = np.linspace(-16.0, 176.0, 641)
        s = np.interp(t - total / 2, grid, filtered)
        ej = ei[None, :] + s[:, None] * (eg - ei)[None, :]
        f = np.empty_like(ej)
        for j, n in enumerate(names):
            f[:, j] = [
                transmon_spectrum(
                    float(v), float(ec[n]), Basis(ncut=35, transmon_levels=3)
                )[0][1]
                for v in ej[:, j]
            ]
        trajectories[shape] = dict(
            time_ns=t.tolist(), bare_f01_ghz=f.tolist(), EJ_ghz=ej.tolist(), nodes=names
        )
        metadata[shape] = dict(
            rise_10_90_ns=10.0,
            full_edge_ns=duration,
            hold_ns=hold,
            programmed_ns=total,
            idle_bare_ghz=idle,
            gate_bare_ghz=target,
        )
        for ax in axs[:, col]:
            for lo, hi, color in [
                (0, duration, "#dceaf7"),
                (duration, duration + hold, "#eeeeee"),
                (duration + hold, total, "#dceaf7"),
            ]:
                ax.axvspan(lo, hi, color=color, alpha=0.55, zorder=0)
            for v in [0, duration, duration + hold, total]:
                ax.axvline(v, color="#aaaaaa", lw=0.6, ls=":")
            ax.grid(alpha=0.15)
            ax.set_xlim(-16, 176)
        axs[0, col].plot(t, f[:, 2], color="#7253a1", lw=1.7, label="C")
        axs[0, col].set(
            title=f"10 ns {shape}: {total:.1f} ns gate",
            ylim=(6.8, 10.35),
            ylabel="Coupler bare $f_{01}$ (GHz)",
        )
        # The markers denote EJ percentages, rather than frequency percentages.
        t10, t90 = duration / 2 - 5, duration / 2 + 5
        axs[0, col].annotate(
            "",
            (t90, 10.03),
            (t10, 10.03),
            arrowprops=dict(
                arrowstyle="<->", lw=0.9, mutation_scale=6, shrinkA=0, shrinkB=0
            ),
        )
        axs[0, col].text(
            (t10 + t90) / 2, 10.15, "10 ns in $E_J$", ha="center", fontsize=7.5
        )
        axs[0, col].text(
            duration + hold / 2, 7.23, "ON: 7.05 GHz", ha="center", fontsize=8
        )
        axs[0, col].text(110, 9.94, "OFF: 9.80 GHz", ha="center", fontsize=8)
        axs[1, col].plot(t, f[:, 0], color="#555555", lw=1.5, label="A1, A2 fixed")
        axs[1, col].plot(t, f[:, 3], color="#c76a2e", lw=1.6, label="B1")
        axs[1, col].plot(t, f[:, 4], color="#287cb0", lw=1.0, ls="--", label="B2")
        axs[1, col].set(
            ylim=(4.5, 5.6),
            ylabel="Rail bare $f_{01}$ (GHz)",
            xlabel="Time from command start (ns)",
        )
        for lo, hi, label in [
            (0, duration, "Ramp"),
            (duration, duration + hold, "Hold"),
            (duration + hold, total, "Return"),
        ]:
            axs[1, col].text((lo + hi) / 2, 5.52, label, ha="center", fontsize=7)
        axs[1, col].text(
            duration + hold / 2, 4.94, "B gate: 4.8713 GHz", ha="center", fontsize=7.5
        )
        axs[1, col].legend(loc="lower left", fontsize=7, ncol=3)
    fig.suptitle("EJ-controlled frequency trajectories | ideal 1 GHz line", fontsize=12)
    fig.savefig(folder / "frequency-tuning-schematic.png", dpi=220)
    fig.savefig(folder / "frequency-tuning-schematic.svg")
    plt.close(fig)
    (folder / "frequency-tuning-schematic.json").write_text(
        json.dumps(
            dict(
                definition="Bare local transmon f01 obtained from filtered EJ and fixed loaded EC; not dressed logical energies or propagated populations",
                controls=metadata,
                trajectories=trajectories,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--tuning-only", action="store_true")
    args = parser.parse_args()
    if args.tuning_only:
        tuning_schematic(args.output)
        return
    if not args.baseline:
        parser.error("--baseline is required for the fidelity comparison plot")
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pulse_bandwidth import evaluate

    rows = []
    for path in sorted(args.output.glob("*/result.json")):
        result = json.loads(path.read_text())
        if not result.get("best"):
            continue
        best = result["best"]
        bandwidth, _ = evaluate(
            result["shape"], result["rise_ns"], hold_ns=best["hold_ns"]
        )
        row = dict(
            shape=result["shape"],
            input_rise_ns=result["rise_ns"],
            output_rise_ns=bandwidth["output_rise_10_90_ns"],
            hold_ns=best["hold_ns"],
            programmed_total_ns=best["nominal_pulse_ns"],
            simulated_total_ns=best["simulated_total_ns"],
            raw_fidelity=best["raw_cz_fidelity"],
            postselected_fidelity=best["postselected_cz_fidelity"],
            survival=best["logical_survival"],
            worst_nonlogical_population=best["worst_nonlogical_population"],
            gate_B_bare_f01_ghz=best["fb_ghz"],
            gate_C_bare_f01_ghz=best["fc_ghz"],
            phase_error_rad=best["phase_error_rad"],
            max_waveform_distortion_percent=bandwidth["max_error_percent"],
            phase_pi_calibrated=result.get("phase_pi_calibrated", False),
            numerical_validation=result.get("validation", {}),
        )
        rows.append(row)
    rows.sort(key=lambda r: (r["shape"], -r["input_rise_ns"]))
    baseline = json.loads(args.baseline.read_text())
    summary = dict(
        filter="1 GHz ideal sharp cutoff, zero phase, EJ variable",
        rows=rows,
        baseline_10ns_unfiltered=baseline,
        calibration="Bounded local raw-fidelity refinement at conditional phase pi; same idle logical references",
        timing="Programmed duration is rise+hold+fall. Simulated window includes explicit pre/post ringing; ideal brick wall has infinite tails.",
        precision="Probability comparison target 1e-4; see per-case validation, not a quality threshold",
    )
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2))
    if not rows:
        raise ValueError("No completed calibrated pulses")
    columns = [key for key in rows[0] if key != "numerical_validation"]
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
    colors = {"tanh": "#2464ad", "cosine": "#c94337"}
    for shape in ("tanh", "cosine"):
        selected = sorted(
            (r for r in rows if r["shape"] == shape), key=lambda r: r["input_rise_ns"]
        )
        if not selected:
            continue
        x = [r["input_rise_ns"] for r in selected]
        for key, style, label in [
            ("raw_fidelity", "-", "raw"),
            ("postselected_fidelity", "--", "postselected"),
        ]:
            axs[0, 0].plot(
                x,
                [max(1e-12, 1 - r[key]) for r in selected],
                style + "o",
                color=colors[shape],
                label=f"{shape}: {label}",
            )
        axs[0, 1].plot(
            x,
            [r["programmed_total_ns"] for r in selected],
            "-o",
            color=colors[shape],
            label=shape,
        )
        axs[1, 0].plot(
            x,
            [100 * r["worst_nonlogical_population"] for r in selected],
            "-o",
            color=colors[shape],
            label=shape,
        )
        axs[1, 1].plot(
            x,
            [r["max_waveform_distortion_percent"] for r in selected],
            "-o",
            color=colors[shape],
            label=shape,
        )
    axs[0, 0].axhline(1e-4, color="gray", ls=":", label="Probability accuracy target")
    for key, marker in [("raw_cz_fidelity", "s"), ("postselected_cz_fidelity", "D")]:
        axs[0, 0].scatter(
            [10],
            [1 - baseline["best"][key]],
            marker=marker,
            facecolors="none",
            edgecolors="black",
            s=60,
            zorder=5,
            label="Previous 10 ns, unfiltered" if marker == "s" else None,
        )
    axs[0, 0].set(
        yscale="log",
        ylabel="1 - fidelity",
        title="Calibrated coherent CZ error; 1 GHz line",
    )
    axs[0, 1].set(
        ylabel="Programmed gate duration (ns)",
        title="Complete rise + calibrated hold + fall",
    )
    axs[1, 0].set(
        ylabel="Worst-input final nonlogical population (%)",
        title="Same dressed idle logical input/output space",
    )
    axs[1, 1].set(
        yscale="log",
        ylabel="Maximum EJ-envelope distortion (%)",
        title="Full-excursion error, including pre/post ringing",
    )
    for ax in axs.flat:
        ax.set_xscale("log")
        ax.set_xlabel("Commanded 10–90% rise (ns)")
        ax.set_xticks([0.5, 1, 2, 5, 10], labels=["0.5", "1", "2", "5", "10"])
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Nominal five-transmon design | simultaneous EJ tuning | A fixed | C ON endpoint 7.05 GHz",
        fontsize=13,
    )
    fig.savefig(args.output / "pulse-shape-comparison.png", dpi=190)
    fig.savefig(args.output / "pulse-shape-comparison.svg")
    fig.canvas.draw()
    fig.set_layout_engine(None)
    renderer = fig.canvas.get_renderer()
    for name, ax in zip(("error", "duration", "leakage", "distortion"), axs.flat):
        box = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(
            args.output / f"pulse-comparison-{name}.png",
            bbox_inches=box.expanded(1.02, 1.025),
            dpi=190,
        )
    plt.close(fig)
    if all(
        (args.output / f"{shape}-10ns/result.json").exists()
        for shape in ("tanh", "cosine")
    ):
        tuning_schematic(args.output)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
