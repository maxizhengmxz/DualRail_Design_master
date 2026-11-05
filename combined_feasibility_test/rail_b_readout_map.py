"""Independent 80--100 MHz readout map on the direct-EC 5 MHz hull."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from readout_workflow import calculation as study
from readout_workflow.direct_ec_feasibility import capacitances
from readout_workflow.storage import atomic_json
from readout_workflow.validation import audit_point


DATA = HERE / "data"
A_RESULTS = DATA / "rail_a_readout_map" / "results.json"
OUTPUT = DATA / "rail_b_readout_map"


def initialize() -> None:
    study.CONDITIONS = replace(study.CONDITIONS, logical_frequency_ghz=(0.08, 0.10))
    study.BASELINE = dict(study.BASELINE, logical_frequency=(0.08, 0.10))


def evaluate_target(ec1: float, ec2: float, x, basis):
    c1, c2 = capacitances(ec1, ec2, x[2], x[3])
    return study.point_from_x(c1, c2, x, basis)


def solve(ec: tuple[int, int], warm: list[dict]) -> dict:
    ec1, ec2 = ec
    started = time.monotonic()
    lower, upper = map(np.asarray, study.nominal_search_bounds())
    cache = {}

    def point(x):
        key = tuple(map(float, x))
        if key not in cache:
            cache[key] = evaluate_target(ec1, ec2, x, study.SCREENING_BASIS)
        return cache[key]

    def residual(x):
        candidate = point(x)
        device = candidate["device"]
        return np.r_[
            study.violations(candidate, margin=0.3),
            max(0, 36 - device["C1_shunt_fF"]) / 10,
            max(0, device["C1_shunt_fF"] - 128) / 10,
            max(0, 36 - device["C2_shunt_fF"]) / 10,
            max(0, device["C2_shunt_fF"] - 128) / 10,
        ]

    seeds = [np.clip(study.x_from_point(p), lower + 1e-6, upper - 1e-6) for p in warm]
    attempts = []
    chosen = None
    with threadpool_limits(limits=1):
        for seed in seeds[:3]:
            fit = least_squares(
                residual,
                seed,
                bounds=(lower, upper),
                x_scale=[1, 1, 2, 5],
                diff_step=2e-5,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=50,
            )
            candidate = point(fit.x)
            failures = study.failures(candidate)
            attempt = {"x": fit.x.tolist(), "screening_failures": failures}
            attempts.append(attempt)
            if failures:
                continue
            audited = audit_point(candidate)
            failures = study.failures(audited, require_sign_check=True)
            device = audited["device"]
            if not all(36 <= device[k] <= 128 for k in ("C1_shunt_fF", "C2_shunt_fF")):
                failures.append("capacitance_search_range")
            attempt["audit_failures"] = failures
            if not failures:
                chosen = audited
                break
    return {
        "ec1_target_mhz": ec1,
        "ec2_target_mhz": ec2,
        "status": "validated_pass" if chosen else "no_validated_solution_found",
        "point": chosen,
        "attempts": attempts,
        "seconds": time.monotonic() - started,
    }


def main() -> int:
    initialize()
    source = json.loads(A_RESULTS.read_text())
    targets = [(row["ec1_target_mhz"], row["ec2_target_mhz"]) for row in source]
    # The independent A witnesses seed the optimizer but are not reused as B results.
    seeds = [row["point"] for row in source if row["status"] == "validated_pass"]
    seed_xy = np.array([[p["loaded_ec1_mhz"], p["loaded_ec2_mhz"]] for p in seeds])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, ec in enumerate(targets, 1):
        distances = np.sum((seed_xy - np.asarray(ec)) ** 2, axis=1)
        warm = [seeds[i] for i in np.argsort(distances)[:3]]
        result = solve(ec, warm)
        rows.append(result)
        atomic_json(OUTPUT / f"ec_{ec[0]}_{ec[1]}.json", result)
        print(json.dumps({"completed": index, "ec": ec, "status": result["status"]}), flush=True)
    atomic_json(OUTPUT / "results.json", rows)
    atomic_json(
        OUTPUT / "summary.json",
        {
            "targets": len(rows),
            "validated_pass": sum(r["status"] == "validated_pass" for r in rows),
            "logical_frequency_mhz": [80, 100],
            "independent_of_gate_map": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
