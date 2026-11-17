# ruff: noqa: E402
# BLAS thread limits must be set before importing numerical libraries.
"""Continuous operation-frequency bounds for a completed resonator/readout map.

Uses the same fixed hardware and constraints as resonator_operation_map. Seeds
come from accepted map cells and the earlier one-dimensional validated sweep.
Extrema are numerical witnesses; they do not certify global optimality or the
absence of disconnected branches. Source data are never overwritten.
"""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from . import calculation as study
from .resonator_frequency_tolerance import device_at_frequency, policy
from .resonator_operation_map import constraints
from .storage import atomic_json
from .validation import INTERMEDIATE, audit_point


def extrema(ro, frequency, seeds):
    policy(ro["case"])
    device, _ = device_at_frequency(ro, frequency)
    model = study.ReadoutModel(device, INTERMEDIATE)
    cache = {}
    results = {}
    validated_seeds = list(seeds)

    def xfrom(p):
        return np.array(
            [
                0.5 * (p["bare_f1_ghz"] + p["bare_f2_ghz"]),
                1000 * (p["bare_f1_ghz"] - p["bare_f2_ghz"]),
            ]
        )

    def evaluate(x):
        key = tuple(map(float, x))
        if key not in cache:
            fs = [x[0] + x[1] * 0.0005, x[0] - x[1] * 0.0005]
            ej = [
                study.inverse_frequency(float(f), float(ec), INTERMEDIATE)
                for f, ec in zip(fs, model.ec_ghz[:2], strict=True)
            ]
            cache[key] = study.annotate(model, *ej)
        return cache[key]

    def slacks(x):
        p = evaluate(x)
        return np.r_[constraints(p) - 1e-7, study.absolute_mean_chi_mhz(p) - 1.00002]

    objectives = {
        "minimum_operation": lambda p: p["operating_frequency_ghz"],
        "maximum_operation": lambda p: -p["operating_frequency_ghz"],
        "maximum_mean_chi": lambda p: -study.absolute_mean_chi_mhz(p),
    }
    with threadpool_limits(limits=1):
        for name, fn in objectives.items():
            seed = min(validated_seeds, key=fn)
            x0 = xfrom(seed)
            fit = minimize(
                lambda x, fn=fn: fn(evaluate(x)),
                x0,
                method="SLSQP",
                bounds=[(3.8, 5.8), (-40, 40)],
                constraints=[{"type": "ineq", "fun": slacks}],
                options={"maxiter": 35, "ftol": 2e-8, "eps": 1e-5},
            )
            candidates = [p for p in cache.values() if not study.failures(p)]
            candidates.sort(key=fn)
            chosen = None
            checks = []
            proposals = (
                [xfrom(p) for p in candidates[:3]]
                + [fit.x * (1 - t) + x0 * t for t in [0.001, 0.01, 0.1]]
                + [x0]
            )
            for x in proposals:
                p = evaluate(x)
                if study.failures(p):
                    continue
                q = audit_point(p, representative=False)
                failed = study.failures(q, require_sign_check=True)
                checks.append(dict(failures=failed, point=q))
                if not failed:
                    chosen = q
                    break
            # Preserve already validated extrema if a new threshold-edge
            # proposal fails a higher-basis audit. Never fall back to a weaker
            # nominal seed when a stronger certified witness already exists.
            prior = [
                p
                for p in validated_seeds
                if not study.failures(p, require_sign_check=True)
            ]
            if prior:
                previous = min(prior, key=fn)
                if chosen is None or fn(previous) < fn(chosen):
                    chosen = previous
            if chosen is not None:
                validated_seeds.append(chosen)
            results[name] = dict(
                point=chosen,
                optimizer_success=bool(fit.success),
                optimizer_message=str(fit.message),
                audits=checks,
            )
    return dict(
        bare_resonator_frequency_ghz=frequency,
        extrema=results,
        unique_evaluations=len(cache),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--map", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frequencies", nargs="*", type=float)
    a = p.parse_args()
    ro = json.loads(a.input.read_text())
    policy(ro["case"])
    rows = json.loads(a.map.read_text())
    old = json.loads(a.prior.read_text())
    a.output.mkdir(exist_ok=True, parents=True)
    freqs = a.frequencies or sorted(
        set(r["bare_resonator_frequency_ghz"] for r in rows)
    )
    results = []
    for f in freqs:
        seeds = [
            r["point"]
            for r in rows + old
            if r.get("point") and abs(r["bare_resonator_frequency_ghz"] - f) < 1e-8
        ]
        if not seeds:
            continue
        path = a.output / f"extrema_{f:.9f}.json"
        if path.exists():
            r = json.loads(path.read_text())
        else:
            r = extrema(ro, f, seeds)
            atomic_json(path, r)
        results.append(r)
        print(
            json.dumps(
                dict(
                    frequency=f,
                    extrema={
                        k: v["point"]["operating_frequency_ghz"] if v["point"] else None
                        for k, v in r["extrema"].items()
                    },
                )
            ),
            flush=True,
        )
    atomic_json(a.output / "results.json", results)


if __name__ == "__main__":
    main()
