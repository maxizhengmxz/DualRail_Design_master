# ruff: noqa: E402
# BLAS thread limits must be set before importing numerical libraries.
"""Improve nominal chi and H at one fixed loaded-EC tuple.

C12 and equal C1r=C2r vary; C1/C2 are solved analytically to preserve nominal
loaded ECs at bare resonator 6 GHz / 50 Ohm. Both EJs recalibrate. Maximize
absolute signed-mean chi at explicit H floors; certify candidates separately.
Finite local multi-start search, not a global optimum certificate.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
import json
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from . import calculation as s
from .direct_ec_feasibility import point
from .ec_resonator_tolerance import set_policy
from .storage import atomic_json
from .validation import INTERMEDIATE, audit_point


def slacks(p, hfloor):
    """Dimensionless signed constraints, including original capacitor bounds."""
    c = s.CONDITIONS
    d = p["device"]
    return np.array(
        [
            (c.mismatch_max - p["ratio"]) / 0.01,
            (s.absolute_mean_chi_mhz(p) - c.abs_mean_chi_min_mhz),
            (p["hybridization"] - hfloor) / 0.05,
            (c.dispersive_max - p["lambda_max"]) / 0.1,
            (p["min_assignment_overlap"] - c.reference_overlap_min) / 0.05,
            (p["splitting_ghz"] - c.logical_frequency_ghz[0]) / 0.02,
            (c.logical_frequency_ghz[1] - p["splitting_ghz"]) / 0.02,
            *[(c.ej_upper_ghz - p[f"ej{i}_ghz"]) / 10 for i in (1, 2)],
            *[(p[f"ej{i}_over_ec{i}"] - c.ej_over_ec_min) / 50 for i in (1, 2)],
            p["operating_frequency_ghz"] - c.operation_frequency_ghz[0],
            c.operation_frequency_ghz[1] - p["operating_frequency_ghz"],
            p["dressed_r_ghz"] - p["operating_frequency_ghz"],
            *[(d[k] - 36) / 50 for k in ("C1_shunt_fF", "C2_shunt_fF")],
            *[(128 - d[k]) / 50 for k in ("C1_shunt_fF", "C2_shunt_fF")],
        ]
    )


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--run", action="store_true")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument(
        "--source",
        type=Path,
        default=Path("readout_workflow/data/B_ec_resonator_12h_20261109"),
        help="Directory containing the original settings.json and results.json",
    )
    a.add_argument("--ec", type=float, nargs=2, default=(200, 238))
    a.add_argument(
        "--h-floors",
        type=float,
        nargs="+",
        default=[0.95, 0.97, 0.98, 0.99, 0.995, 0.999],
    )
    a.add_argument("--logical-min-mhz", type=float, default=80.0)
    a.add_argument("--dispersive-max", type=float, default=0.1)
    args = a.parse_args()
    if not args.run:
        print("Use --run explicitly.")
        return
    source = args.source.resolve()
    config = json.loads((source / "settings.json").read_text())
    set_policy(config["conditions"])
    rows = json.loads((source / "results.json").read_text())
    old = next(
        r for r in rows if (r["ec1_target_mhz"], r["ec2_target_mhz"]) == tuple(args.ec)
    )
    nom = old["nominal"]
    args.output.mkdir(exist_ok=True, parents=True)
    atomic_json(args.output / "original.json", old)
    settings = dict(
        ec_mhz=args.ec,
        conditions=asdict(s.CONDITIONS),
        h_floors=args.h_floors,
        logical_min_target_mhz=args.logical_min_mhz,
        dispersive_target=args.dispersive_max,
        basis=asdict(INTERMEDIATE),
        source_sha256=sha256((source / "results.json").read_bytes()).hexdigest(),
        condition_input_sha256=sha256(
            (source / "settings.json").read_bytes()
        ).hexdigest(),
        driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        objective="maximum absolute signed-mean chi at each H floor",
        scope="Fixed loaded EC tuple at 6 GHz. Variable C12, equal C1r=C2r, EJs; C1/C2 analytically synthesized. Original 36-128 fF C1/C2 bounds retained.",
    )
    atomic_json(args.output / "settings.json", settings)
    cache = {}
    results = []
    validated = []
    started = time.monotonic()

    def evaluate(x):
        key = tuple(map(float, x))
        if key not in cache:
            cache[key] = point(*args.ec, x, INTERMEDIATE)
        return cache[key]

    def design_slacks(p, hfloor):
        return np.r_[
            slacks(p, hfloor),
            (p["splitting_ghz"] - args.logical_min_mhz / 1000) / 0.02,
            (args.dispersive_max - p["lambda_max"]) / 0.1,
        ]

    x0 = np.array(s.x_from_point(nom))
    lo, hi = map(np.array, s.nominal_search_bounds())
    bounds = list(zip(lo, hi, strict=True))
    seeds = [x0]
    # Different operation frequencies and logical-coupling margins help avoid
    # following only the original threshold-adjacent local branch.
    for shift, scale in [(-0.3, 1.1), (0.3, 1.1), (-0.6, 1.2), (0.1, 1.2)]:
        x = x0.copy()
        x[0] += shift
        x[2] *= scale
        seeds.append(np.clip(x, lo + 1e-5, hi - 1e-5))
    with threadpool_limits(limits=1):
        for hfloor in settings["h_floors"]:
            trials = []
            starts = (
                [np.array(s.x_from_point(validated[-1]))] if validated else []
            ) + seeds
            for index, seed in enumerate(starts):
                fit = minimize(
                    lambda x: -s.absolute_mean_chi_mhz(evaluate(x)),
                    seed,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=[
                        {
                            "type": "ineq",
                            "fun": lambda x, hfloor=hfloor: (
                                design_slacks(evaluate(x), hfloor) - 2e-5
                            ),
                        }
                    ],
                    options={"maxiter": 90, "ftol": 1e-8, "eps": 1e-5},
                )
                p = evaluate(fit.x)
                trials.append(
                    dict(
                        seed=seed.tolist(),
                        x=fit.x.tolist(),
                        point=p,
                        success=bool(fit.success),
                        message=str(fit.message),
                        iterations=int(fit.nit),
                    )
                )
                atomic_json(args.output / f"H_{hfloor:.4f}_trials.json", trials)
                print(
                    json.dumps(
                        dict(
                            H_floor=hfloor,
                            start=index + 1,
                            chi_mhz=s.absolute_mean_chi_mhz(p),
                            H=p["hybridization"],
                            slack=float(min(design_slacks(p, hfloor))),
                            elapsed_seconds=time.monotonic() - started,
                        )
                    ),
                    flush=True,
                )
            candidates = sorted(
                [p for p in cache.values() if min(design_slacks(p, hfloor)) >= 0],
                key=s.absolute_mean_chi_mhz,
                reverse=True,
            )
            chosen = None
            audits = []
            used = []
            for p in candidates:
                if len(used) >= 8:
                    break
                x = np.array(s.x_from_point(p))
                if any(np.linalg.norm(x - y) < 1e-7 for y in used):
                    continue
                used.append(x)
                q = audit_point(p, representative=False)
                failed = s.failures(q, require_sign_check=True)
                if q["hybridization"] < hfloor:
                    failed.append("requested_H_floor")
                if min(design_slacks(q, hfloor)) < -1e-7:
                    failed.append("design_margin_or_bounds")
                audits.append(dict(point=q, failures=failed))
                if not failed:
                    chosen = q
                    break
            if chosen:
                validated.append(chosen)
            result = dict(
                H_floor=hfloor,
                point=chosen,
                audits=audits,
                trials=len(trials),
                unique_evaluations=len(cache),
            )
            results.append(result)
            atomic_json(args.output / "results.json", results)
    atomic_json(
        args.output / "completion.json",
        dict(
            phase="complete",
            elapsed_seconds=time.monotonic() - started,
            validated_H_floors=[r["H_floor"] for r in results if r["point"]],
            unique_evaluations=len(cache),
        ),
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
