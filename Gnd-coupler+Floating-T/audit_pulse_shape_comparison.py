"""Fixed-control checks for selected completed bandwidth-pulse cases.

No fidelity re-optimization: compare half step, doubled tail window and higher
excitation cutoff. A changed conditional phase in an audit is reported, not
silently recalibrated. Audit only explicitly named cases.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]


def audit(folder):
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    import numpy as np
    from floating_coupler.fabrication import five_model
    from floating_coupler.pulsed_gate import PulseModel
    from floating_coupler.filtered_pulse import filtered_gate, filtered_samples

    path = folder / "result.json"
    result = json.loads(path.read_text())
    if not result.get("best"):
        raise ValueError("Case has no completed high-basis result")
    started = perf_counter()
    base = result["input_snapshot"]
    best = result["best"]
    model, _ = five_model(base, {}, {})
    seed = base["gate_seed"]
    basis = result["basis"]
    pulse = PulseModel(
        model,
        base["off_ghz"],
        seed["fb_ghz"],
        seed["fc_ghz"],
        local_levels=basis["local_levels"],
        coupler_levels=basis["coupler_levels"],
        excitation_cutoff=basis["excitation_cutoff"],
    )
    common = dict(
        pulse=pulse,
        fb=best["fb_ghz"],
        fc=best["fc_ghz"],
        shape=result["shape"],
        rise_ns=result["rise_ns"],
        hold_ns=best["hold_ns"],
    )
    step = best["integration"]["step_ns"]
    padding = best["padding_each_side_ns"]
    checks = {}

    def save():
        result["validation"] = dict(
            checks=checks,
            probability_tolerance=1e-4,
            all_checked_probabilities_converged=all(
                q["probabilities_converged_1e4"] for q in checks.values()
            ),
            seconds=perf_counter() - started,
        )
        path.write_text(json.dumps(result, indent=2))

    for name, change in [
        ("halved_time_step", dict(step_ns=step / 2, padding_ns=padding)),
        ("doubled_tail_window", dict(step_ns=step, padding_ns=2 * padding)),
        ("excitation_cutoff_plus_two", dict(step_ns=step, padding_ns=padding)),
    ]:
        if name == "excitation_cutoff_plus_two":
            high = PulseModel(
                model,
                base["off_ghz"],
                seed["fb_ghz"],
                seed["fc_ghz"],
                local_levels=basis["local_levels"],
                coupler_levels=basis["coupler_levels"],
                excitation_cutoff=basis["excitation_cutoff"] + 2,
            )
            active = dict(common, pulse=high)
        else:
            active = common
        row = filtered_gate(**active, **change)
        keys = [
            "raw_cz_fidelity",
            "postselected_cz_fidelity",
            "logical_survival",
            "worst_nonlogical_population",
        ]
        delta = {key: row[key] - best[key] for key in keys}
        delta["max_per_input_leakage_change"] = max(
            abs(a - b)
            for a, b in zip(row["leakage_by_input"], best["leakage_by_input"])
        )
        checks[name] = dict(
            result=row,
            changes=delta,
            probabilities_converged_1e4=max(abs(v) for v in delta.values()) <= 1e-4,
        )
        save()
        print(json.dumps(dict(case=folder.name, check=name, changes=delta)), flush=True)
    # Filter sampling/padding are independent of the propagation time step.
    args = (result["shape"], result["rise_ns"], best["hold_ns"], 1.0, padding)
    t, s, _ = filtered_samples(*args)
    fine_t, fine_s, _ = filtered_samples(*args, power=21)
    wide_t, wide_s, _ = filtered_samples(*args, power=21, span_ns=8192.0)
    result["validation"]["FFT_checks"] = dict(
        half_sample_max_envelope_change=float(
            np.max(abs(s - np.interp(t, fine_t, fine_s)))
        ),
        doubled_span_max_envelope_change=float(
            np.max(abs(s - np.interp(t, wide_t, wide_s)))
        ),
    )
    path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", required=True, nargs="+")
    args = parser.parse_args()
    for case in args.cases:
        audit(args.output / case)
