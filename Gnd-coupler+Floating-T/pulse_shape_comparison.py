"""Explicit fixed-hardware tanh/cosine CZ comparison with an ideal EJ brick wall.

One shape per process, bounded local calibration, saved per-case checkpoints.
No calculation on import; source-hashed progress protects reproducibility.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]


def source_identity():
    files = [
        Path(__file__),
        HERE / "floating_coupler/filtered_pulse.py",
        HERE / "floating_coupler/pulsed_gate.py",
        HERE / "pulse_bandwidth.py",
        HERE / "floating_coupler/gate.py",
    ]
    return {
        str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def compare_case(base, shape, rise, folder, *, step=0.075, padding=16.0, audits=True):
    import numpy as np
    from scipy.optimize import minimize, minimize_scalar
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    from floating_coupler.fabrication import five_model
    from floating_coupler.pulsed_gate import PulseModel, phase_pi_candidates
    from floating_coupler.filtered_pulse import filtered_gate
    from floating_coupler.precision import GATE_PHASE_TOL

    started = perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    model, hardware = five_model(base, {}, {})
    seed = base["gate_seed"]
    fb, fc = seed["fb_ghz"], seed["fc_ghz"]
    pulse = PulseModel(
        model,
        base["off_ghz"],
        fb,
        fc,
        local_levels=6,
        coupler_levels=5,
        excitation_cutoff=6,
    )
    spectrum_cache = {}
    history = []

    def unfiltered(frequency):
        key = float(frequency)
        if key not in spectrum_cache:
            spectrum = pulse.spectrum(key, fc, rise, step, shape=shape)
            roots = phase_pi_candidates(spectrum, (40.0, 170.0))
            spectrum_cache[key] = (
                max(roots, key=lambda r: r["raw_cz_fidelity"]) if roots else None
            )
        return spectrum_cache[key]

    for frequency in np.linspace(fb - 0.012, fb + 0.012, 7):
        unfiltered(frequency)
    feasible = [(f, r) for f, r in spectrum_cache.items() if r]
    if not feasible:
        raise ValueError(
            "No phase-pi unfiltered seed in initial search; do not switch channel"
        )
    center, seed_row = max(feasible, key=lambda item: item[1]["raw_cz_fidelity"])
    minimize_scalar(
        lambda f: 1 - unfiltered(f)["raw_cz_fidelity"] if unfiltered(f) else 1.0,
        bounds=(center - 0.004, center + 0.004),
        method="bounded",
        options=dict(xatol=2e-5, maxiter=9),
    )
    seed_row = max(
        (r for r in spectrum_cache.values() if r), key=lambda r: r["raw_cz_fidelity"]
    )
    cache = {}
    center = seed_row["fb_ghz"]
    optimizing = True

    class SearchStop(Exception):
        """Stop line-search trials before they exceed engineering precision."""

    def plateau():
        if len(history) < 8:
            return False
        tail = history[-4:]
        keys = ("raw_cz_fidelity", "postselected_cz_fidelity", "logical_survival")
        return (
            max(abs(r["phase_error_rad"]) for r in tail) < 0.01
            and all(
                max(r[k] for r in tail) - min(r[k] for r in tail) <= 1e-4 for k in keys
            )
            and np.max(np.ptp([r["leakage_by_input"] for r in tail], axis=0)) <= 1e-4
        )

    def evaluate(x):
        key = tuple(map(float, x))
        if key not in cache:
            if optimizing and (len(cache) >= 12 or plateau()):
                raise SearchStop(
                    "metric plateau at 1e-4" if plateau() else "12-trial budget"
                )
            row = filtered_gate(
                pulse,
                center + key[0] * 0.001,
                fc,
                shape,
                rise,
                key[1],
                step_ns=step,
                padding_ns=padding,
            )
            cache[key] = row
            history.append(row)
            (folder / "progress.json").write_text(json.dumps(history, indent=2))
            print(
                json.dumps(
                    dict(
                        shape=shape,
                        rise=rise,
                        evaluation=len(cache),
                        raw=row["raw_cz_fidelity"],
                        selected=row["postselected_cz_fidelity"],
                        phase=row["phase_error_rad"],
                        seconds=row["seconds"],
                    )
                ),
                flush=True,
            )
        return cache[key]

    x0 = np.array([0.0, seed_row["hold_ns"]])
    # In scaled units, 1 == 1 MHz or 1 ns. The broad hold bounds match the
    # earlier tanh workflow; this is a local improvement, not a global proof.
    try:
        opt = minimize(
            lambda x: 1 - evaluate(x)["raw_cz_fidelity"],
            x0,
            method="SLSQP",
            constraints=[dict(type="eq", fun=lambda x: evaluate(x)["phase_error_rad"])],
            bounds=[(-12.0, 12.0), (40.0, 170.0)],
            options=dict(maxiter=5, ftol=2e-6, eps=0.005),
        )
    except SearchStop as stopped:
        from scipy.optimize import OptimizeResult

        # Phase still must be calibrated below; a plateau alone is not acceptance.
        closest = min(cache, key=lambda key: abs(cache[key]["phase_error_rad"]))
        opt = OptimizeResult(x=np.array(closest), success=False, message=str(stopped))
    finally:
        optimizing = False
    row = evaluate(opt.x)
    x = np.array(opt.x, dtype=float)
    # Enforce the shared strict phase tolerance without chasing fidelity digits.
    for _ in range(3):
        if abs(row["phase_error_rad"]) <= GATE_PHASE_TOL:
            break
        probe = x + np.array([0.01, 0.0])
        trial = evaluate(probe)
        derivative = (
            np.angle(np.exp(1j * (trial["phase_error_rad"] - row["phase_error_rad"])))
            / 0.01
        )
        if abs(derivative) < 1e-5:
            raise ValueError("Flat phase response in local B correction")
        correction = -row["phase_error_rad"] / derivative
        if abs(correction) > 2:
            raise ValueError("Phase correction >2 MHz; inspect before broadening")
        x[0] += correction
        row = evaluate(x)
    phase_rows = [r for r in history if abs(r["phase_error_rad"]) <= GATE_PHASE_TOL]
    if not phase_rows:
        raise ValueError("No phase-pi filtered pulse; diagnostics retained")
    best = max(phase_rows, key=lambda r: r["raw_cz_fidelity"])
    result = dict(
        shape=shape,
        rise_ns=rise,
        input_snapshot=base,
        hardware=hardware,
        sources=source_identity(),
        screen_basis=pulse.basis,
        unfiltered_seed=seed_row,
        unfiltered_evaluations=len(spectrum_cache),
        filtered_evaluations=len(cache),
        optimizer=dict(success=bool(opt.success), message=str(opt.message)),
        screening=best,
        filter=dict(
            cutoff_ghz=1.0,
            variable="EJ",
            phase="zero; pure delay removed",
            whole_pulse=True,
            padding_each_side_ns=padding,
        ),
        scope="Full five-transmon coherent pulse; fixed idle logical input/output; no resonators/noise",
    )
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    if audits:
        high = PulseModel(
            model,
            base["off_ghz"],
            fb,
            fc,
            local_levels=8,
            coupler_levels=7,
            excitation_cutoff=8,
        )
        high_row = filtered_gate(
            high,
            best["fb_ghz"],
            fc,
            shape,
            rise,
            best["hold_ns"],
            step_ns=step,
            padding_ns=padding,
        )
        names = [
            "raw_cz_fidelity",
            "postselected_cz_fidelity",
            "logical_survival",
            "worst_nonlogical_population",
        ]
        result["fixed_control_basis_change"] = {k: high_row[k] - best[k] for k in names}
        transfers = [high_row]
        new_fb = best["fb_ghz"]
        # Reuse a phase slope already measured in screening, then update it by
        # secants of actual high-basis results. Avoid a fresh probe every time.
        derivative = None
        for a, b in zip(history[:-1], history[1:]):
            if (
                abs(a["hold_ns"] - b["hold_ns"]) < 1e-10
                and abs(a["fb_ghz"] - b["fb_ghz"]) > 1e-9
            ):
                derivative = np.angle(
                    np.exp(1j * (b["phase_error_rad"] - a["phase_error_rad"]))
                ) / (b["fb_ghz"] - a["fb_ghz"])
                break
        for _ in range(3):
            if abs(high_row["phase_error_rad"]) <= GATE_PHASE_TOL:
                break
            if derivative is None:
                trial = filtered_gate(
                    high,
                    new_fb + 1e-5,
                    fc,
                    shape,
                    rise,
                    best["hold_ns"],
                    step_ns=step,
                    padding_ns=padding,
                )
                derivative = (
                    np.angle(
                        np.exp(
                            1j
                            * (trial["phase_error_rad"] - high_row["phase_error_rad"])
                        )
                    )
                    / 1e-5
                )
            if not np.isfinite(derivative) or abs(derivative) < 1:
                raise ValueError("Flat high-basis phase response")
            previous = high_row
            correction = -high_row["phase_error_rad"] / derivative
            if abs(correction) > 0.001:
                raise ValueError("High-basis phase transfer >1 MHz")
            new_fb += correction
            high_row = filtered_gate(
                high,
                new_fb,
                fc,
                shape,
                rise,
                best["hold_ns"],
                step_ns=step,
                padding_ns=padding,
            )
            if abs(new_fb - previous["fb_ghz"]) > 1e-12:
                derivative = np.angle(
                    np.exp(
                        1j * (high_row["phase_error_rad"] - previous["phase_error_rad"])
                    )
                ) / (new_fb - previous["fb_ghz"])
            transfers.append(high_row)
        result.update(
            basis=high.basis,
            best=high_row,
            phase_transfer=transfers,
            phase_pi_calibrated=abs(high_row["phase_error_rad"]) <= GATE_PHASE_TOL,
        )
    else:
        result.update(basis=pulse.basis, best=best, phase_pi_calibrated=True)
    result["elapsed_seconds"] = perf_counter() - started
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                finished=shape,
                rise=rise,
                best=result["best"],
                seconds=result["elapsed_seconds"],
            )
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shape", required=True, choices=["tanh", "cosine"])
    parser.add_argument(
        "--rises", type=float, nargs="+", default=[10.0, 5.0, 2.0, 1.0, 0.5]
    )
    parser.add_argument(
        "--step-ns",
        type=float,
        default=None,
        help="Default: 0.0375 ns for rises <=1 ns; 0.075 ns otherwise",
    )
    parser.add_argument("--padding-ns", type=float, default=16.0)
    parser.add_argument("--no-audits", action="store_true")
    a = parser.parse_args()
    base = json.loads(a.input.read_text(encoding="utf-8-sig"))
    for rise in a.rises:
        step = a.step_ns if a.step_ns is not None else (0.0375 if rise <= 1 else 0.075)
        if rise <= 0 or step <= 0 or a.padding_ns <= 0:
            raise ValueError("Positive rise, step and tail window required")
        folder = a.output / f"{a.shape}-{rise:g}ns"
        saved = folder / "result.json"
        if saved.exists():
            d = json.loads(saved.read_text())
            if d.get("sources") != source_identity() or d.get("input_snapshot") != base:
                raise ValueError("Checkpoint source/input mismatch")
            if d.get("best") and d.get("phase_pi_calibrated"):
                continue
        compare_case(
            base,
            a.shape,
            rise,
            folder,
            step=step,
            padding=a.padding_ns,
            audits=not a.no_audits,
        )
