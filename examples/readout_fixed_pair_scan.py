"""Map viable readout designs at the fixed shunt pair C1=100 fF, C2=75 fF.

The physical design variables are the two tunable Josephson energies, C12 and
the common readout capacitance C1r=C2r.  Internally the Josephson energies are
represented by the existing readout workflow's mean/differential local bare
frequency coordinates.

This program does two complementary calculations:

* ``boundaries`` directly minimizes and maximizes the logical frequency under
  all nominal readout inequalities.  This is the inexpensive way to estimate
  the viable interval.
* ``scan`` follows a configurable logical-frequency grid, using validated
  solutions from neighboring frequencies as continuation seeds.  Every
  distinct solution found is audited at the convergence basis and retained.

Because the design variables are continuous, no finite numerical run can list
the mathematical "complete set" of viable configurations.  The scan instead
produces a resolution-defined, deduplicated catalog and records enough policy
metadata to state exactly what was searched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from readout_workflow import calculation as study


C1_FF = 100.0
C2_FF = 75.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "readout_workflow/data/fixed_pair_100_75"
X_SCALE = np.array([1.0, 5.0, 2.0, 5.0])
DEFAULT_TARGET_STEP_MHZ = 0.5
DEFAULT_TARGET_TOLERANCE_MHZ = 0.02


def search_bounds(fixed_c12_ff=None, fixed_readout_cap_ff=None):
    """Return the optimizer box, optionally fixing either capacitance coordinate."""

    lower, upper = (
        array.copy() for array in map(np.asarray, study.nominal_search_bounds())
    )
    fixed = {2: fixed_c12_ff, 3: fixed_readout_cap_ff}
    labels = {2: "C12", 3: "C1r=C2r"}
    for index, value in fixed.items():
        if value is None:
            continue
        value = float(value)
        if not lower[index] <= value <= upper[index]:
            raise ValueError(
                f"fixed {labels[index]}={value:g} fF is outside the nominal "
                f"range [{lower[index]:g}, {upper[index]:g}] fF"
            )
        lower[index] = value
        upper[index] = value
    return lower, upper


def apply_fixed_capacitances(x, fixed_c12_ff=None, fixed_readout_cap_ff=None):
    """Return a copy of x with requested fixed capacitances imposed exactly."""

    x = np.asarray(x, dtype=float).copy()
    if fixed_c12_ff is not None:
        x[2] = fixed_c12_ff
    if fixed_readout_cap_ff is not None:
        x[3] = fixed_readout_cap_ff
    return x


class Evaluator:
    """Memoize the expensive circuit calculation separately for each basis."""

    def __init__(self):
        self.cache = {}

    def __call__(self, x, basis):
        key = (basis, tuple(np.round(np.asarray(x, dtype=float), 12)))
        if key not in self.cache:
            self.cache[key] = study.point_from_x(C1_FF, C2_FF, x, basis)
        return self.cache[key]


def inequality_slacks(point, target_ghz=None, tolerance_ghz=0.0):
    """Positive values pass all smooth nominal constraints.

    Strict EC/EJ/EJ-over-EC checks receive small numerical guard bands.  The
    discontinuous chi-sign certificate is deliberately reserved for the final
    audit and is never approximated inside SLSQP.
    """

    c = study.CONDITIONS
    slacks = list(study.slacks(point, study.BASELINE, include_H=True, margin=1.0))
    slacks.extend(
        [
            (point["loaded_ec1_mhz"] - (c.loaded_ec_mhz[0] + 1e-4)) / 50,
            (point["loaded_ec2_mhz"] - (c.loaded_ec_mhz[0] + 1e-4)) / 50,
            ((c.loaded_ec_mhz[1] - 1e-4) - point["loaded_ec1_mhz"]) / 50,
            ((c.loaded_ec_mhz[1] - 1e-4) - point["loaded_ec2_mhz"]) / 50,
        ]
    )
    if target_ghz is not None:
        delta = point["splitting_ghz"] - target_ghz
        slacks.extend((tolerance_ghz - delta, tolerance_ghz + delta))
    return np.asarray(slacks, dtype=float)


def sobol_seeds(count, seed, fixed_c12_ff=None, fixed_readout_cap_ff=None):
    """Add deterministic space-filling starts to the workflow's physics seeds."""

    if count <= 0:
        return []
    # Generate in the original non-degenerate box, then project fixed coordinates.
    # scipy.stats.qmc.scale requires every upper bound to exceed its lower bound.
    lower, upper = map(np.asarray, study.nominal_search_bounds())
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    values = qmc.scale(sampler.random(count), lower, upper)
    return [
        apply_fixed_capacitances(row, fixed_c12_ff, fixed_readout_cap_ff)
        for row in values
    ]


def unique_seeds(
    warm, starts, limit, fixed_c12_ff=None, fixed_readout_cap_ff=None
):
    lower, upper = map(np.asarray, study.nominal_search_bounds())
    raw = [np.asarray(x, dtype=float) for x in warm]
    raw.extend(np.asarray(x, dtype=float) for x in starts)
    raw.extend(study.seed_candidates(C1_FF, C2_FF, ()))
    unique = []
    for x in raw:
        x = np.clip(x, lower + 1e-7, upper - 1e-7)
        x = apply_fixed_capacitances(x, fixed_c12_ff, fixed_readout_cap_ff)
        if not any(np.linalg.norm((x - old) / X_SCALE) < 1e-5 for old in unique):
            unique.append(x)
        if len(unique) >= limit:
            break
    return unique


def audit(x, target_ghz=None, tolerance_ghz=0.0):
    """Recompute unchanged inputs at the largest standard basis and sign-check."""

    model = study.make_model(
        C1_FF, C2_FF, float(x[2]), float(x[3]), study.CONVERGENCE_BASIS
    )
    point = study.point_from_x(C1_FF, C2_FF, x, study.CONVERGENCE_BASIS)
    failures = study.failures(point)
    if (
        target_ghz is not None
        and abs(point["splitting_ghz"] - target_ghz) > tolerance_ghz
    ):
        failures = [*failures, "target_logical_frequency"]
    if not failures:
        point = study.check_chi_sign_stability(point, model)
        failures = study.failures(point, require_sign_check=True)
    return point, failures


def optimize_from_seed(
    evaluator,
    seed,
    objective,
    target_ghz,
    tolerance_ghz,
    fixed_c12_ff=None,
    fixed_readout_cap_ff=None,
):
    lower, upper = search_bounds(fixed_c12_ff, fixed_readout_cap_ff)
    seed = apply_fixed_capacitances(seed, fixed_c12_ff, fixed_readout_cap_ff)

    def point(x):
        return evaluator(x, study.SCREENING_BASIS)

    fit = minimize(
        lambda x: objective(point(x), x),
        seed,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: inequality_slacks(
                    point(x), target_ghz, tolerance_ghz
                ),
            }
        ],
        options={"maxiter": 100, "ftol": 2e-10, "eps": 1e-6},
    )
    # Re-polish at the validation basis before the expensive convergence audit.
    polish = minimize(
        lambda x: objective(evaluator(x, study.VALIDATION_BASIS), x),
        fit.x,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: inequality_slacks(
                    evaluator(x, study.VALIDATION_BASIS),
                    target_ghz,
                    tolerance_ghz,
                ),
            }
        ],
        options={"maxiter": 80, "ftol": 1e-11, "eps": 5e-7},
    )
    return polish


def fingerprint(point):
    """Round only for numerical de-duplication; saved points remain full precision."""

    x = np.asarray(study.x_from_point(point))
    return tuple(np.round(x / np.array([1e-5, 1e-3, 1e-4, 1e-4])).astype(int))


def summarize_point(point, target_mhz=None):
    return {
        "target_logical_frequency_mhz": target_mhz,
        "logical_frequency_mhz": 1000 * point["splitting_ghz"],
        "operation_frequency_ghz": point["operating_frequency_ghz"],
        "ej1_ghz": point["ej1_ghz"],
        "ej2_ghz": point["ej2_ghz"],
        "c12_fF": point["device"]["C12_fF"],
        "c1r_fF": point["device"]["C1r_fF"],
        "c2r_fF": point["device"]["C2r_fF"],
        "loaded_ec1_mhz": point["loaded_ec1_mhz"],
        "loaded_ec2_mhz": point["loaded_ec2_mhz"],
        "ej1_over_ec1": point["ej1_over_ec1"],
        "ej2_over_ec2": point["ej2_over_ec2"],
        "mean_chi_mhz": 500 * (point["chi_p_ghz"] + point["chi_m_ghz"]),
        "abs_mean_chi_mhz": study.absolute_mean_chi_mhz(point),
        "chi_mismatch": point["ratio"],
        "hybridization": point["hybridization"],
        "dispersive_ratio": point["lambda_max"],
        "min_assignment_overlap": point["min_assignment_overlap"],
        "x": study.x_from_point(point),
    }


def find_boundary(
    direction,
    starts,
    max_starts,
    fixed_c12_ff=None,
    fixed_readout_cap_ff=None,
):
    evaluator = Evaluator()
    sign = 1.0 if direction == "min" else -1.0
    objective = lambda point, _x: sign * point["splitting_ghz"]
    candidates, attempts = [], []
    for seed in unique_seeds(
        (), starts, max_starts, fixed_c12_ff, fixed_readout_cap_ff
    ):
        started = time.monotonic()
        try:
            fit = optimize_from_seed(
                evaluator,
                seed,
                objective,
                None,
                0.0,
                fixed_c12_ff,
                fixed_readout_cap_ff,
            )
            point, failures = audit(fit.x)
            attempts.append(
                {
                    "success": bool(fit.success),
                    "message": str(fit.message),
                    "seconds": time.monotonic() - started,
                    "x": fit.x.tolist(),
                    "logical_frequency_mhz": 1000 * point["splitting_ghz"],
                    "failures": failures,
                }
            )
            if not failures:
                candidates.append(point)
        except (ValueError, FloatingPointError) as error:
            attempts.append(
                {"error": repr(error), "seconds": time.monotonic() - started}
            )
    if not candidates:
        return None, attempts
    key = lambda p: p["splitting_ghz"]
    endpoint = min(candidates, key=key) if direction == "min" else max(candidates, key=key)
    return endpoint, attempts


def scan_targets(
    targets_mhz,
    starts,
    max_starts,
    tolerance_mhz,
    fixed_c12_ff=None,
    fixed_readout_cap_ff=None,
):
    evaluator = Evaluator()
    accepted, attempts = [], []
    seen = set()
    continuation = []
    tolerance_ghz = tolerance_mhz / 1000
    lower, upper = search_bounds(fixed_c12_ff, fixed_readout_cap_ff)

    for target_mhz in targets_mhz:
        target_ghz = target_mhz / 1000

        def objective(point, x):
            target_error = (point["splitting_ghz"] - target_ghz) / tolerance_ghz
            # Once the target band is reached, prefer safe interior points and
            # stay weakly near the seed through the bounded optimizer geometry.
            margin = float(np.min(inequality_slacks(point)))
            box_center = (lower + upper) / 2
            regularizer = 1e-7 * np.sum(((x - box_center) / X_SCALE) ** 2)
            return target_error**2 - 1e-3 * min(margin, 1.0) + regularizer

        target_attempts = []
        seeds = unique_seeds(
            continuation,
            starts,
            max_starts,
            fixed_c12_ff,
            fixed_readout_cap_ff,
        )
        new_points = []
        for seed in seeds:
            started = time.monotonic()
            try:
                fit = optimize_from_seed(
                    evaluator,
                    seed,
                    objective,
                    target_ghz,
                    tolerance_ghz,
                    fixed_c12_ff,
                    fixed_readout_cap_ff,
                )
                point, failures = audit(fit.x, target_ghz, tolerance_ghz)
                target_attempts.append(
                    {
                        "success": bool(fit.success),
                        "message": str(fit.message),
                        "seconds": time.monotonic() - started,
                        "x": fit.x.tolist(),
                        "logical_frequency_mhz": 1000 * point["splitting_ghz"],
                        "failures": failures,
                    }
                )
                mark = fingerprint(point)
                if not failures and mark not in seen:
                    seen.add(mark)
                    accepted.append(point)
                    new_points.append(point)
            except (ValueError, FloatingPointError) as error:
                target_attempts.append(
                    {"error": repr(error), "seconds": time.monotonic() - started}
                )
        if new_points:
            continuation = [np.asarray(study.x_from_point(p)) for p in new_points]
        attempts.append(
            {
                "target_logical_frequency_mhz": target_mhz,
                "accepted": len(new_points),
                "attempts": target_attempts,
            }
        )
        print(
            json.dumps(
                {
                    "target_logical_frequency_mhz": target_mhz,
                    "new_valid_configurations": len(new_points),
                    "total_valid_configurations": len(accepted),
                }
            ),
            flush=True,
        )
    return accepted, attempts


def save_results(output, payload, points):
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "fixed_pair_scan.json"
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = [summarize_point(point) for point in points]
    csv_path = output / "viable_configurations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != "x"])
            writer.writeheader()
            writer.writerows({k: v for k, v in row.items() if k != "x"} for row in rows)
    return json_path, csv_path


def target_grid(low_mhz, high_mhz, step_mhz):
    count = int(math.floor((high_mhz - low_mhz) / step_mhz + 1e-10))
    values = low_mhz + step_mhz * np.arange(count + 1)
    if values[-1] < high_mhz - 1e-9:
        values = np.r_[values, high_mhz]
    return [float(v) for v in values]


def run_study(
    *,
    mode="all",
    output=None,
    low_mhz=90.0,
    high_mhz=110.0,
    step_mhz=DEFAULT_TARGET_STEP_MHZ,
    target_tolerance_mhz=DEFAULT_TARGET_TOLERANCE_MHZ,
    starts=24,
    sobol_starts_count=16,
    seed=10075,
    fixed_c12_ff=None,
    fixed_readout_cap_ff=None,
):
    """Run one complete policy and return its serializable result payload.

    A fixed capacitance value removes that coordinate from the effective search
    box.  Passing ``None`` leaves the coordinate continuously variable.
    """

    if mode not in ("boundaries", "scan", "all"):
        raise ValueError("mode must be 'boundaries', 'scan', or 'all'")
    if not 0 < step_mhz or not 0 < target_tolerance_mhz:
        raise ValueError("step and target tolerance must be positive")
    if not low_mhz < high_mhz:
        raise ValueError("low_mhz must be smaller than high_mhz")
    if starts < 1 or sobol_starts_count < 0:
        raise ValueError("starts must be positive and sobol_starts_count nonnegative")
    # Validate fixed values even for a zero-Sobol or boundary-only run.
    search_bounds(fixed_c12_ff, fixed_readout_cap_ff)

    started = time.monotonic()
    initial_starts = sobol_seeds(
        sobol_starts_count, seed, fixed_c12_ff, fixed_readout_cap_ff
    )
    boundaries, boundary_attempts = {}, {}
    points = []
    with threadpool_limits(limits=1):
        if mode in ("boundaries", "all"):
            for direction in ("min", "max"):
                point, attempts = find_boundary(
                    direction,
                    initial_starts,
                    starts,
                    fixed_c12_ff,
                    fixed_readout_cap_ff,
                )
                boundaries[direction] = summarize_point(point) if point else None
                boundary_attempts[direction] = attempts
                if point:
                    points.append(point)
                    print(
                        json.dumps(
                            {
                                "boundary": direction,
                                "logical_frequency_mhz": 1000
                                * point["splitting_ghz"],
                            }
                        ),
                        flush=True,
                    )

        scan_attempts = []
        if mode in ("scan", "all"):
            scanned, scan_attempts = scan_targets(
                target_grid(low_mhz, high_mhz, step_mhz),
                initial_starts,
                starts,
                target_tolerance_mhz,
                fixed_c12_ff,
                fixed_readout_cap_ff,
            )
            points.extend(scanned)

    unique = {}
    for point in points:
        unique.setdefault(fingerprint(point), point)
    points = sorted(unique.values(), key=lambda p: p["splitting_ghz"])
    found_range = (
        [1000 * points[0]["splitting_ghz"], 1000 * points[-1]["splitting_ghz"]]
        if points
        else None
    )
    payload = {
        "schema": "fixed-pair-logical-range-v2",
        "pair_fF": [C1_FF, C2_FF],
        "mode": mode,
        "conditions": asdict(study.CONDITIONS),
        "basis": {
            "screening": asdict(study.SCREENING_BASIS),
            "validation": asdict(study.VALIDATION_BASIS),
            "audit": asdict(study.CONVERGENCE_BASIS),
        },
        "scan_policy": {
            "low_mhz": low_mhz,
            "high_mhz": high_mhz,
            "step_mhz": step_mhz,
            "target_tolerance_mhz": target_tolerance_mhz,
            "max_starts_per_problem": starts,
            "sobol_starts": sobol_starts_count,
            "seed": seed,
            "fixed_c12_fF": fixed_c12_ff,
            "fixed_readout_cap_fF": fixed_readout_cap_ff,
            "continuation": True,
            "claim": (
                "resolution-defined best-found catalog; "
                "not an exhaustive continuum proof"
            ),
        },
        "found_logical_frequency_range_mhz": found_range,
        "boundaries": boundaries,
        "viable_configuration_count": len(points),
        "viable_configurations": [summarize_point(point) for point in points],
        "boundary_attempts": boundary_attempts,
        "scan_attempts": scan_attempts,
        "elapsed_seconds": time.monotonic() - started,
    }
    if output is not None:
        json_path, csv_path = save_results(Path(output).resolve(), payload, points)
        payload["output"] = {"json": str(json_path), "csv": str(csv_path)}
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("boundaries", "scan", "all"), default="all"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--low-mhz", type=float, default=90.0)
    parser.add_argument("--high-mhz", type=float, default=110.0)
    parser.add_argument("--step-mhz", type=float, default=DEFAULT_TARGET_STEP_MHZ)
    parser.add_argument(
        "--target-tolerance-mhz", type=float, default=DEFAULT_TARGET_TOLERANCE_MHZ
    )
    parser.add_argument("--starts", type=int, default=24)
    parser.add_argument("--sobol-starts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=10075)
    parser.add_argument(
        "--fixed-c12-ff",
        type=float,
        help="fix C12 at this value in fF; omit to optimize it",
    )
    parser.add_argument(
        "--fixed-readout-cap-ff",
        type=float,
        help="fix the common C1r=C2r at this value in fF; omit to optimize it",
    )
    args = parser.parse_args()
    try:
        payload = run_study(
            mode=args.mode,
            output=args.output,
            low_mhz=args.low_mhz,
            high_mhz=args.high_mhz,
            step_mhz=args.step_mhz,
            target_tolerance_mhz=args.target_tolerance_mhz,
            starts=args.starts,
            sobol_starts_count=args.sobol_starts,
            seed=args.seed,
            fixed_c12_ff=args.fixed_c12_ff,
            fixed_readout_cap_ff=args.fixed_readout_cap_ff,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                **payload["output"],
                "viable_configuration_count": payload["viable_configuration_count"],
                "found_logical_frequency_range_mhz": payload[
                    "found_logical_frequency_range_mhz"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
