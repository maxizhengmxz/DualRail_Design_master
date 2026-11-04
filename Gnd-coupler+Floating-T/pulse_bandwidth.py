"""Waveform-only ideal brick-wall comparison; no gate optimization.

H(f)=1 for |f|<fc and 0 outside; zero phase (pure delay removed). Symmetric
ringing is therefore expected. Results apply to a normalized line variable,
not automatically to EJ when the actual line controls nonlinear SQUID flux.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.optimize import brentq
from scipy.special import sici


def edge_duration(shape, rise_ns):
    if shape == "tanh":
        tau = rise_ns / (2 * np.arctanh(0.8 * np.tanh(4.0)))
        return 8 * tau
    if shape == "cosine":
        return rise_ns / (1 - 2 * np.arccos(0.8) / np.pi)
    raise ValueError("Unknown edge shape")


def edge(t, shape, rise_ns):
    duration = edge_duration(shape, rise_ns)
    x = np.clip(np.asarray(t), 0.0, duration)
    if shape == "cosine":
        return 0.5 * (1 - np.cos(np.pi * x / duration))
    return (np.tanh((x - duration / 2) / (duration / 8)) + np.tanh(4.0)) / (
        2 * np.tanh(4.0)
    )


def waveform(t, shape, rise_ns, hold_ns):
    duration = edge_duration(shape, rise_ns)
    total = 2 * duration + hold_ns
    return edge(t + total / 2, shape, rise_ns) - edge(t - hold_ns / 2, shape, rise_ns)


def brickwall(signal, dt_ns, cutoff_ghz=1.0):
    f = rfftfreq(len(signal), dt_ns)
    transfer = (f < cutoff_ghz).astype(float)
    # Half weight at the discontinuity for the Fourier-integration convention.
    transfer[np.isclose(f, cutoff_ghz, atol=1e-12, rtol=0)] = 0.5
    return irfft(rfft(signal) * transfer, n=len(signal))


def crossing(t, y, threshold, center):
    indices = np.flatnonzero((y[:-1] < threshold) & (y[1:] >= threshold))
    if not len(indices):
        raise ValueError("No rising crossing")
    k = min(indices, key=lambda i: abs(t[i] - center))
    return float(t[k] + (threshold - y[k]) * (t[k + 1] - t[k]) / (y[k + 1] - y[k]))


def evaluate(
    shape,
    rise_ns,
    *,
    hold_ns=85.28436900082397,
    cutoff_ghz=1.0,
    power=20,
    span_ns=4096.0,
):
    count = 2**power
    dt = span_ns / count
    t = (np.arange(count) - count // 2) * dt
    signal = waveform(t, shape, rise_ns, hold_ns)
    filtered = brickwall(signal, dt, cutoff_ghz)
    duration = edge_duration(shape, rise_ns)
    center = -(hold_ns + duration) / 2
    t10 = crossing(t, filtered, 0.1, center)
    t90 = crossing(t, filtered, 0.9, center)
    delta = filtered - signal
    row = dict(
        shape=shape,
        input_rise_10_90_ns=float(rise_ns),
        edge_support_ns=duration,
        pulse_support_ns=2 * duration + hold_ns,
        output_rise_10_90_ns=t90 - t10,
        max_error_percent=float(np.max(abs(delta)) * 100),
        overshoot_percent=float(max(0.0, np.max(filtered) - 1.0) * 100),
        undershoot_percent=float(max(0.0, -np.min(filtered)) * 100),
        rms_error_percent=float(
            np.sqrt(np.mean(delta[abs(t) < duration + hold_ns / 2 + 10] ** 2)) * 100
        ),
        peak_time_error_ns=float(t[np.argmax(abs(delta))]),
        dt_ns=dt,
        span_ns=span_ns,
    )
    mask = abs(t - center) < max(12.0, rise_ns * 2.5)
    return row, (t[mask] - center, signal[mask], filtered[mask])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff-ghz", type=float, default=1.0)
    parser.add_argument("--hold-ns", type=float, default=85.28436900082397)
    args = parser.parse_args()
    if args.cutoff_ghz <= 0 or args.hold_ns < 0:
        raise ValueError("Invalid cutoff or hold")
    args.output.mkdir(parents=True, exist_ok=True)
    rises = [0.1, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    rows = []
    traces = {}
    for shape in ("tanh", "cosine"):
        for rise in rises:
            row, trace = evaluate(
                shape, rise, cutoff_ghz=args.cutoff_ghz, hold_ns=args.hold_ns
            )
            rows.append(row)
            traces[shape, rise] = trace
    # Dense speed sweep; keep a supported region rather than assuming that
    # spectral cancellations make cosine errors globally monotonic.
    scan = []
    for shape in ("tanh", "cosine"):
        for rise in np.geomspace(0.2, 10.0, 101):
            row, _ = evaluate(
                shape, rise, cutoff_ghz=args.cutoff_ghz, hold_ns=args.hold_ns
            )
            scan.append(row)
    limits = {}
    for shape in ("tanh", "cosine"):
        selected = [r for r in scan if r["shape"] == shape]
        limits[shape] = {}
        for bound in (1.0, 0.1, 0.01):
            # Threshold at which every subsequently sampled slower edge passes.
            candidate = next(
                (
                    r
                    for i, r in enumerate(selected)
                    if all(s["max_error_percent"] <= bound for s in selected[i:])
                ),
                None,
            )
            limits[shape][str(bound) + "_percent_error"] = candidate
    audit = []
    for shape, rise in (
        ("tanh", 10.0),
        ("cosine", 10.0),
        ("tanh", 1.0),
        ("cosine", 1.0),
        ("tanh", 0.25),
    ):
        nominal = next(
            r for r in rows if r["shape"] == shape and r["input_rise_10_90_ns"] == rise
        )
        finer, _ = evaluate(
            shape, rise, cutoff_ghz=args.cutoff_ghz, hold_ns=args.hold_ns, power=21
        )
        wider, _ = evaluate(
            shape,
            rise,
            cutoff_ghz=args.cutoff_ghz,
            hold_ns=args.hold_ns,
            power=21,
            span_ns=8192.0,
        )
        audit.append(
            dict(
                shape=shape,
                rise_ns=rise,
                max_error_percent_change_step=finer["max_error_percent"]
                - nominal["max_error_percent"],
                max_error_percent_change_padding=wider["max_error_percent"]
                - nominal["max_error_percent"],
                rise_change_ns_step=finer["output_rise_10_90_ns"]
                - nominal["output_rise_10_90_ns"],
            )
        )
    x = brentq(lambda x: sici(x)[0] - 0.4 * np.pi, 0.0, np.pi)
    ideal_step = dict(
        rise_10_90_ns=x / (np.pi * args.cutoff_ghz),
        overshoot_percent=(sici(np.pi)[0] / np.pi - 0.5) * 100,
    )
    data = dict(
        filter="zero-phase ideal brick wall; linear phase only adds a delay",
        cutoff_ghz=args.cutoff_ghz,
        hold_ns=args.hold_ns,
        metric="max absolute deviation divided by full excursion, INCLUDING ringing outside pulse",
        tanh="K=4 endpoint-normalized, identical shape to five-mode pulse study",
        cosine="half-cosine 0.5*(1-cos(pi*t/T)), compact support, same 10-90% rise",
        rows=rows,
        speed_scan=scan,
        thresholds=limits,
        ideal_step=ideal_step,
        audit=audit,
        scope="Linear waveform filtering only; does not predict calibrated CZ fidelity or nonlinear flux-to-EJ distortion",
    )
    (args.output / "results.json").write_text(json.dumps(data, indent=2))
    header = list(rows[0])
    (args.output / "comparison.csv").write_text(
        ",".join(header)
        + "\n"
        + "\n".join(",".join(str(r[k]) for k in header) for r in rows)
        + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    colors = {"tanh": "#2464ad", "cosine": "#c94337"}
    for shape in ("tanh", "cosine"):
        selected = [r for r in scan if r["shape"] == shape]
        axes[0, 0].loglog(
            [r["input_rise_10_90_ns"] for r in selected],
            [r["max_error_percent"] for r in selected],
            label=shape,
            color=colors[shape],
        )
        axes[0, 1].plot(
            [r["input_rise_10_90_ns"] for r in selected],
            [r["output_rise_10_90_ns"] for r in selected],
            label=shape,
            color=colors[shape],
        )
        for rise, style in ((0.5, "--"), (1.0, "-")):
            t, signal, filtered = traces[shape, rise]
            axes[1, 0].plot(
                t,
                filtered,
                color=colors[shape],
                ls=style,
                label=f"{shape}, {rise:g} ns",
            )
        t, signal, filtered = traces[shape, 10.0]
        axes[1, 1].plot(t, (filtered - signal) * 100, label=shape, color=colors[shape])
    for bound in (0.01, 0.1, 1.0):
        axes[0, 0].axhline(bound, color="#888888", ls=":", lw=0.8)
    axes[0, 0].set(
        xlabel="Input 10–90% rise (ns)",
        ylabel="Maximum waveform error (% of full swing)",
        title="Full pulse, including all pre/post-ringing",
    )
    axes[0, 1].plot([0, 3], [0, 3], color="gray", ls=":", label="Undistorted")
    axes[0, 1].axhline(
        ideal_step["rise_10_90_ns"],
        color="black",
        ls="--",
        lw=1,
        label="Filtered ideal step",
    )
    axes[0, 1].set(
        xlabel="Input rise (ns)",
        ylabel="Output rise (ns)",
        xlim=(0, 3),
        ylim=(0, 3),
        title="Fast edge broadening",
    )
    axes[1, 0].set(
        xlabel="Time from rising-edge midpoint (ns)",
        ylabel="Normalized filtered control",
        xlim=(-2, 2),
        title="Faster pulses: ringing versus smooth transition",
    )
    axes[1, 1].set(
        xlabel="Time from rising-edge midpoint (ns)",
        ylabel="Filtered − intended (% of full swing)",
        title="Original 10 ns edges: residual distortion",
    )
    for ax in axes.ravel():
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Ideal {args.cutoff_ghz:g} GHz brick-wall flux-line response; equal 10–90% rise convention",
        fontsize=14,
    )
    fig.savefig(args.output / "bandwidth-comparison.png", dpi=180)
    fig.savefig(args.output / "bandwidth-comparison.svg")
    plt.close(fig)
    print(
        json.dumps(
            dict(rows=rows, thresholds=limits, ideal_step=ideal_step, audit=audit),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
