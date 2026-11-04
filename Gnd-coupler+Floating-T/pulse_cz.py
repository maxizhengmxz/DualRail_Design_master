"""Explicit five-transmon pulse CZ study; no calculation on import."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rise-ns", type=float, default=10.0)
    parser.add_argument("--step-ns", type=float, default=0.1)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--coupler-levels", type=int, default=5)
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--fb-points", type=int, default=7)
    parser.add_argument(
        "--fb-halfwidth-ghz",
        type=float,
        default=0.012,
        help="Initial local scan width, not a hardware acceptance bound",
    )
    parser.add_argument(
        "--hold-bounds-ns",
        type=float,
        nargs=2,
        default=[40.0, 170.0],
        help="Initial plateau search interval, excluding tanh tails",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Fixed-control audit of a preceding result; no new optimization",
    )
    parser.add_argument(
        "--phase-correct",
        action="store_true",
        help="In an audit, correct only B frequency at fixed duration/coupler (at most six ramp builds)",
    )
    args = parser.parse_args()
    if (
        args.rise_ns <= 0
        or args.step_ns <= 0
        or args.levels < 3
        or args.coupler_levels < 3
        or args.cutoff < 2
    ):
        raise ValueError(
            "Require positive edge/step and sufficient local levels/cutoff"
        )
    if (
        args.fb_points < 3
        or args.fb_halfwidth_ghz <= 0
        or not 0 <= args.hold_bounds_ns[0] < args.hold_bounds_ns[1]
    ):
        raise ValueError("Invalid initial pulse search interval")
    if args.phase_correct and not args.calibration:
        raise ValueError("--phase-correct requires a preceding calibration")
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    import numpy as np
    from scipy.optimize import minimize_scalar
    from floating_coupler.fabrication import five_model
    from floating_coupler.pulsed_gate import (
        PulseModel,
        metrics_at,
        phase_pi_candidates,
        edge_parameters,
    )
    from floating_coupler.precision import GATE_PHASE_TOL

    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.input.read_text(encoding="utf-8-sig"))
    selection = base["selection"]
    if selection["moved_pair"] != "B":
        raise ValueError("This first pulse workflow moves B only")
    model, hardware = five_model(base, {}, {})
    seed = base["gate_seed"]
    fb, fc = seed["fb_ghz"], seed["fc_ghz"]
    audited = json.loads(args.calibration.read_text()) if args.calibration else None
    if audited:
        fb, fc = audited["best"]["fb_ghz"], audited["best"]["fc_ghz"]
    pulse = PulseModel(
        model,
        base["off_ghz"],
        fb,
        fc,
        local_levels=args.levels,
        coupler_levels=args.coupler_levels,
        excitation_cutoff=args.cutoff,
    )
    rows, cache = [], {}

    def at(frequency):
        key = float(frequency)
        if key not in cache:
            spectrum = pulse.spectrum(key, fc, args.rise_ns, args.step_ns)
            roots = phase_pi_candidates(spectrum, args.hold_bounds_ns)
            best = max(roots, key=lambda r: r["raw_cz_fidelity"]) if roots else None
            cache[key] = (spectrum, best)
            rows.append(
                dict(
                    fb_ghz=key,
                    seconds=spectrum["seconds"],
                    roots=roots,
                    edge_norm_error=spectrum["edge_norm_error"],
                )
            )
            (args.output / "progress.json").write_text(json.dumps(rows, indent=2))
            print(
                json.dumps(dict(fb_ghz=key, seconds=spectrum["seconds"], best=best)),
                flush=True,
            )
        return cache[key]

    if audited:
        spectrum = pulse.spectrum(fb, fc, args.rise_ns, args.step_ns)
        best = metrics_at(spectrum, audited["best"]["hold_ns"])
        if args.phase_correct:
            # Bounded phase transfer only, not a large-basis gate optimization.
            transfer = [dict(fb_ghz=fb, phase_error_rad=best["phase_error_rad"])]
            for iteration in range(2):
                if abs(best["phase_error_rad"]) <= GATE_PHASE_TOL:
                    break
                delta = 1e-5
                probe = pulse.spectrum(fb + delta, fc, args.rise_ns, args.step_ns)
                probe_row = metrics_at(probe, audited["best"]["hold_ns"])
                derivative = (
                    np.angle(
                        np.exp(
                            1j
                            * (probe_row["phase_error_rad"] - best["phase_error_rad"])
                        )
                    )
                    / delta
                )
                if not np.isfinite(derivative) or abs(derivative) < 1:
                    raise ValueError("Cannot transfer pulse phase locally")
                correction = -best["phase_error_rad"] / derivative
                if abs(correction) > 0.001:
                    raise ValueError(
                        "Phase transfer exceeds local 1 MHz guard; inspect before extending"
                    )
                fb += correction
                spectrum = pulse.spectrum(fb, fc, args.rise_ns, args.step_ns)
                best = metrics_at(spectrum, audited["best"]["hold_ns"])
                transfer.append(
                    dict(fb_ghz=fb, phase_error_rad=best["phase_error_rad"])
                )
            rows = [dict(phase_transfer=transfer)]
    else:
        for frequency in np.linspace(
            fb - args.fb_halfwidth_ghz, fb + args.fb_halfwidth_ghz, args.fb_points
        ):
            at(frequency)
        feasible = [
            (frequency, item[1]) for frequency, item in cache.items() if item[1]
        ]
        if not feasible:
            raise ValueError(
                "No phase-pi pulse in this initial local search; progress retained"
            )
        center, _ = max(feasible, key=lambda item: item[1]["raw_cz_fidelity"])
        spacing = 2 * args.fb_halfwidth_ghz / (args.fb_points - 1)

        def objective(frequency):
            row = at(frequency)[1]
            return 1 - row["raw_cz_fidelity"] if row else 1.0

        minimize_scalar(
            objective,
            bounds=(center - spacing, center + spacing),
            method="bounded",
            options=dict(xatol=2e-5, maxiter=10),
        )
        feasible = [
            (frequency, item[1]) for frequency, item in cache.items() if item[1]
        ]
        frequency, best = max(feasible, key=lambda item: item[1]["raw_cz_fidelity"])
        spectrum = cache[frequency][0]
        fb = frequency
    result = dict(
        schema="five-mode-tanh-cz-v1",
        input=str(args.input.resolve()),
        input_snapshot=base,
        hardware=hardware,
        basis=pulse.basis,
        channel_selection=selection,
        off_ghz=base["off_ghz"],
        idle_ej_ghz=pulse.idle_ej.tolist(),
        gate_ej_ghz=spectrum["gate_ej"].tolist(),
        reference_ej_ghz=pulse.reference_ej.tolist(),
        pulse=dict(
            control="EJ",
            rise_definition="10-90%",
            rise_ns=args.rise_ns,
            truncation_tau=4.0,
            symmetric_return=True,
            simultaneous=True,
            edge_duration_ns=edge_parameters(args.rise_ns)[1],
            integration=spectrum["edge"],
        ),
        reference_definition="Four full-system dressed idle logical eigenstates; same input/output frame",
        idle_assignment_weights=pulse.input_weights.tolist(),
        idle_product_projection_norms=pulse.product_norms.tolist(),
        idle_logical_energies_ghz=pulse.idle_logical_energies.tolist(),
        best=best,
        phase_pi_calibrated=abs(best["phase_error_rad"]) <= GATE_PHASE_TOL,
        edge_norm_error=spectrum["edge_norm_error"],
        static_reference=seed,
        optimization_history=rows,
        audit_of=str(args.calibration) if args.calibration else None,
        comparison_scope="phase-corrected controls"
        if args.phase_correct
        else "identical controls"
        if audited
        else "screening calibration",
        elapsed_seconds=perf_counter() - started,
        physics="Coherent full-cosine ng=0; all capacitive charge terms; fixed hardware; no T1/noise/resonators",
    )
    if audited:
        keys = [
            "raw_cz_fidelity",
            "postselected_cz_fidelity",
            "logical_survival",
            "worst_nonlogical_population",
        ]
        changes = {key: best[key] - audited["best"][key] for key in keys}
        result[
            "phase_transfer_changes" if args.phase_correct else "fixed_control_changes"
        ] = changes
        result["probabilities_converged_1e4"] = (
            None
            if args.phase_correct
            else max(abs(x) for x in changes.values()) <= 1e-4
        )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        args.output / "gate-spectrum.npz",
        energy=spectrum["energy"],
        coefficients=spectrum["coefficients"],
    )
    print(
        json.dumps(dict(final=best, elapsed_seconds=result["elapsed_seconds"])),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
