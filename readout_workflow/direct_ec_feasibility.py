"""Direct loaded-EC feasibility search at fixed EC1 and EC2.

Capacitances are solved analytically from the Schur complement of the full
three-node C matrix at every trial. This preserves the requested loaded ECs
as C12 and equal readout coupling capacitors change. No readout optimization,
failed-target retries, contour refinement, or failure diagnosis is performed.
"""

# ruff: noqa: E402 -- BLAS limits and direct-file path setup precede numerical imports.
from __future__ import annotations

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, Delaunay
from threadpoolctl import threadpool_limits

# Also support an explicit direct-file invocation from an IDE.
if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Readout_ParamSpace_Robustness import process_alive
from readout_workflow import calculation as s
from readout_workflow.storage import atomic_json, utc
from readout_workflow.validation import audit_point

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "readout_workflow/data/direct_ec_step5"
SOURCE = ROOT / "readout_workflow/data/feasibility_150mhz_step1_diag2/results.json"


def target_grid(xy, domain, step_mhz=5, hull_scale=1.0, ec_bounds_mhz=(150.0, 400.0)):
    """Return an EC lattice inside the observed rectangle or a scaled hull."""
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.all(np.isfinite(xy)):
        raise ValueError("xy must contain at least three finite (EC1, EC2) points")
    if domain not in ("hull", "rectangle"):
        raise ValueError("domain must be 'hull' or 'rectangle'")
    if not np.isfinite(step_mhz) or step_mhz <= 0:
        raise ValueError("step_mhz must be positive")
    if not np.isfinite(hull_scale) or hull_scale < 1:
        raise ValueError("hull_scale must be at least 1")
    if len(ec_bounds_mhz) != 2 or not 0 < ec_bounds_mhz[0] < ec_bounds_mhz[1]:
        raise ValueError("invalid EC bounds")
    hull = ConvexHull(xy)
    vertices = xy[hull.vertices]
    if domain == "hull":
        center = vertices.mean(axis=0)
        selection_xy = center + hull_scale * (vertices - center)
    else:
        selection_xy = xy
    xs = np.arange(
        np.ceil(selection_xy[:, 0].min() / step_mhz) * step_mhz,
        selection_xy[:, 0].max() + 1e-8,
        step_mhz,
    )
    ys = np.arange(
        np.ceil(selection_xy[:, 1].min() / step_mhz) * step_mhz,
        selection_xy[:, 1].max() + 1e-8,
        step_mhz,
    )
    rectangle = [(float(x), float(y)) for x in xs for y in ys]
    rectangle = [
        point
        for point in rectangle
        if all(ec_bounds_mhz[0] <= value < ec_bounds_mhz[1] for value in point)
    ]
    if domain == "rectangle":
        return rectangle
    tri = Delaunay(selection_xy)
    return [point for point in rectangle if tri.find_simplex([point])[0] >= 0]


def mirrored_target_grid(xy, step_mhz=5, hull_scale=1.5, ec_bounds_mhz=(150.0, 400.0)):
    """Union of separately scaled original and label-swapped readout hulls."""
    original = set(target_grid(xy, "hull", step_mhz, hull_scale, ec_bounds_mhz))
    mirrored = set(
        target_grid(
            np.asarray(xy, float)[:, ::-1], "hull", step_mhz, hull_scale, ec_bounds_mhz
        )
    )
    return sorted(original | mirrored), original, mirrored


def mirrored_warm_point(point):
    """Return an exact label-swapped readout warm start without forging observables."""
    x = np.asarray(s.x_from_point(point), float)
    x[1] *= -1
    return {"x": x.tolist(), "readout_symmetry_source": "swapped_1_2"}


def logical_range_ghz(bounds_mhz):
    """Validate a CLI logical-frequency interval and convert MHz to GHz."""
    if (
        len(bounds_mhz) != 2
        or not all(np.isfinite(bounds_mhz))
        or not 0 < bounds_mhz[0] < bounds_mhz[1]
    ):
        raise ValueError(
            "logical-frequency-mhz requires two positive increasing values"
        )
    return tuple(float(value) / 1000 for value in bounds_mhz)


def initialize(logical_frequency_ghz=(0.14, 0.16)):
    """Configure numerical calculations with cyclic frequencies in GHz."""
    logical_frequency_ghz = tuple(map(float, logical_frequency_ghz))
    if (
        len(logical_frequency_ghz) != 2
        or not all(np.isfinite(logical_frequency_ghz))
        or not 0 < logical_frequency_ghz[0] < logical_frequency_ghz[1]
    ):
        raise ValueError(
            "logical_frequency_ghz requires two positive increasing values"
        )
    s.CONDITIONS = replace(
        s.CONDITIONS, logical_frequency_ghz=tuple(logical_frequency_ghz)
    )
    s.BASELINE = dict(s.BASELINE, logical_frequency=tuple(logical_frequency_ghz))


def capacitances(ec1, ec2, c12, cg):
    """Physical C1,C2 (fF) with EXACT prescribed inverse-matrix diagonals.

    Eliminate resonator: A=[[X,-b],[-b,Y]], b=C12+Cg^2/R,
    R=Cr+2Cg. If diag(A^-1)=(d1,d2), determinant D satisfies
    d1*d2*D^2-D-b^2=0. X=d2*D, Y=d1*D; subtract attached C.
    """
    cr = s.Device().Cr_fF
    r = cr + 2 * cg
    d1 = (ec1 / 1000) / s.CHARGE_CONSTANT
    d2 = (ec2 / 1000) / s.CHARGE_CONSTANT
    b = c12 + cg * cg / r
    determinant = (1 + np.sqrt(1 + 4 * d1 * d2 * b * b)) / (2 * d1 * d2)
    correction = c12 + cg - cg * cg / r
    return d2 * determinant - correction, d1 * determinant - correction


def point(ec1, ec2, x, basis):
    c1, c2 = capacitances(ec1, ec2, x[2], x[3])
    return s.point_from_x(c1, c2, x, basis)


def solve(task):
    bounds = task.get("logical_frequency_mhz")
    ghz = task.get("logical_frequency_ghz")
    # Explicitly labelled legacy MHz tasks are converted at this input boundary.
    if bounds is not None:
        converted = logical_range_ghz(bounds)
        if ghz is not None and not np.allclose(converted, ghz, rtol=0, atol=1e-12):
            raise ValueError("Conflicting MHz and GHz logical-frequency task inputs")
        ghz = converted
    initialize((0.14, 0.16) if ghz is None else ghz)
    ec1, ec2 = task["ec"]
    started = time.monotonic()
    attempts = []
    chosen = None
    closest = None
    lower, upper = map(np.array, s.nominal_search_bounds())
    cache = {}

    def evaluate(x):
        k = tuple(map(float, x))
        if k not in cache:
            cache[k] = point(ec1, ec2, x, s.SCREENING_BASIS)
        return cache[k]

    def residual(x):
        p = evaluate(x)
        d = p["device"]
        return np.r_[
            s.violations(p, margin=0.3),
            max(0, 36 - d["C1_shunt_fF"]) / 10,
            max(0, d["C1_shunt_fF"] - 128) / 10,
            max(0, 36 - d["C2_shunt_fF"]) / 10,
            max(0, d["C2_shunt_fF"] - 128) / 10,
        ]

    seeds = []
    for p in task["warm"]:
        x = np.clip(
            p["x"] if "x" in p else s.x_from_point(p), lower + 1e-6, upper - 1e-6
        )
        if not any(np.linalg.norm(x - y) < 1e-5 for y in seeds):
            seeds.append(x)
    with threadpool_limits(limits=1):
        for seed in seeds[:3]:
            if attempts and time.monotonic() - started > 25:
                break
            fit = least_squares(
                residual,
                seed,
                bounds=(lower, upper),
                x_scale=[1, 1, 2, 5],
                diff_step=2e-5,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=35,
            )
            p = evaluate(fit.x)
            attempts.append(
                dict(
                    seed=seed.tolist(),
                    x=fit.x.tolist(),
                    nfev=int(fit.nfev),
                    cost=float(fit.cost),
                    failures=s.failures(p),
                    endpoint=p,
                )
            )
            if closest is None or np.linalg.norm(s.violations(p)) < np.linalg.norm(
                s.violations(closest)
            ):
                closest = p
            candidates = [q for q in cache.values() if not s.failures(q)]
            if candidates:
                # First acceptable numerical witness; do not rank by readout.
                q = audit_point(candidates[0])
                d = q["device"]
                errs = [abs(q["loaded_ec1_mhz"] - ec1), abs(q["loaded_ec2_mhz"] - ec2)]
                failed = s.failures(q, require_sign_check=True)
                if any(v > 1e-6 for v in errs):
                    failed.append("loaded_EC_target")
                if not all(36 <= d[k] <= 128 for k in ("C1_shunt_fF", "C2_shunt_fF")):
                    failed.append("capacitance_search_range")
                attempts[-1]["audit_failures"] = failed
                if not failed:
                    chosen = q
                    break
    return dict(
        ec1_target_mhz=ec1,
        ec2_target_mhz=ec2,
        status="validated_pass" if chosen else "no_validated_solution_found",
        point=chosen,
        closest_screening_point=closest,
        attempts=attempts,
        seconds=time.monotonic() - started,
        completed_utc=utc(),
        readout_optimized=False,
    )


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--logical-frequency-mhz", type=float, nargs=2, default=(140, 160)
    )
    parser.add_argument("--ec1-range", type=float, nargs=2)
    parser.add_argument("--ec2-range", type=float, nargs=2)
    parser.add_argument("--ec-step-mhz", type=float, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--domain", choices=("hull", "rectangle"), default="hull")
    parser.add_argument(
        "--orientation-domain",
        choices=("full", "reduced"),
        default="reduced",
        help="Full includes label-swapped targets. Weekend regeneration uses full.",
    )
    parser.add_argument(
        "--hull-scale",
        type=float,
        default=1.0,
        help="Hull expansion. Weekend regeneration explicitly uses 1.5.",
    )
    a = parser.parse_args()
    OUT = a.output.resolve()
    source = a.source.resolve()
    if bool(a.ec1_range) != bool(a.ec2_range):
        parser.error("Provide both EC ranges or neither")
    if not np.isfinite(a.ec_step_mhz) or a.ec_step_mhz <= 0 or a.workers < 1:
        parser.error("Step and worker count must be positive")
    if not np.isfinite(a.hull_scale) or a.hull_scale < 1:
        parser.error("Hull scale must be at least one")
    logical_frequency_ghz = logical_range_ghz(a.logical_frequency_mhz)
    if not a.run:
        print("Use --run to calculate; no automatic calculation on import.")
        return
    if a.background:
        OUT.mkdir(parents=True, exist_ok=True)
        launch = OUT / "launch.json"
        if launch.exists() and process_alive(json.loads(launch.read_text())["pid"]):
            raise RuntimeError("Already running")
        command = [
            sys.executable,
            "-m",
            "readout_workflow.direct_ec_feasibility",
            "--run",
            "--output",
            str(OUT),
            "--source",
            str(source),
            "--workers",
            str(a.workers),
            "--logical-frequency-mhz",
            *map(str, a.logical_frequency_mhz),
            "--ec-step-mhz",
            str(a.ec_step_mhz),
            "--domain",
            a.domain,
            "--orientation-domain",
            a.orientation_domain,
            "--hull-scale",
            str(a.hull_scale),
        ]
        if a.ec1_range:
            command += [
                "--ec1-range",
                *map(str, a.ec1_range),
                "--ec2-range",
                *map(str, a.ec2_range),
            ]
        kwargs = dict(
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW
                | subprocess.DETACHED_PROCESS
                | subprocess.CREATE_BREAKAWAY_FROM_JOB
            )
        else:
            kwargs["start_new_session"] = True
        with (OUT / "run.log").open("ab") as log:
            kwargs["stdout"] = log
            child = subprocess.Popen(command, **kwargs)
        record = dict(
            pid=child.pid,
            started_utc=utc(),
            background=True,
            command=command,
            logical_frequency_mhz=a.logical_frequency_mhz,
            ec_step_mhz=a.ec_step_mhz,
            optimize_readout=False,
            failure_diagnosis_enabled=False,
        )
        atomic_json(launch, record)
        print(json.dumps(record))
        return

    initialize(logical_frequency_ghz)
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads(source.read_text())
    good = [
        row["point"]
        for row in old
        if row["status"] in ("validated_pass", "recovered_viable")
    ]
    if not good:
        raise ValueError("Source contains no validated warm starts")
    xy = np.array([[p["loaded_ec1_mhz"], p["loaded_ec2_mhz"]] for p in good])
    if a.ec1_range:
        intervals = (a.ec1_range, a.ec2_range)
        if any(
            not np.all(np.isfinite(interval)) or not 0 < interval[0] <= interval[1]
            for interval in intervals
        ):
            parser.error("EC ranges must be finite and increasing")
        xs = np.arange(a.ec1_range[0], a.ec1_range[1] + 1e-8, a.ec_step_mhz)
        ys = np.arange(a.ec2_range[0], a.ec2_range[1] + 1e-8, a.ec_step_mhz)
        targets = [
            (float(x), float(y))
            for x in xs
            for y in ys
            if a.orientation_domain == "full" or x < y
        ]
        original_targets = set(targets)
        mirrored_targets = set()
        selection = "Explicit EC rectangle; no contour restriction. Full keeps both orders, reduced keeps EC1<EC2."
    elif a.orientation_domain == "full" and a.domain == "hull":
        targets, original_targets, mirrored_targets = mirrored_target_grid(
            xy,
            step_mhz=a.ec_step_mhz,
            hull_scale=a.hull_scale,
            ec_bounds_mhz=s.CONDITIONS.loaded_ec_mhz,
        )
        selection = "Union of separately scaled original and label-swapped hulls. Membership is not assumed feasible."
    else:
        domain_xy = (
            np.vstack((xy, xy[:, ::-1])) if a.orientation_domain == "full" else xy
        )
        targets = target_grid(
            domain_xy,
            a.domain,
            step_mhz=a.ec_step_mhz,
            hull_scale=a.hull_scale,
            ec_bounds_mhz=s.CONDITIONS.loaded_ec_mhz,
        )
        if a.orientation_domain == "reduced":
            targets = [point for point in targets if point[0] < point[1]]
        original_targets = set(targets)
        mirrored_targets = set()
        selection = "Witness-derived rectangle or hull. Reduced keeps EC1<EC2. Membership is not assumed feasible."
    if not targets:
        raise ValueError("Selected EC domain contains no targets")
    source_xy = [tuple(value) for value in xy]
    warm_points = list(good)
    if a.orientation_domain == "full":
        # Mirror only the domain and trial coordinates. Recompute every target.
        source_xy.extend(tuple(value) for value in xy[:, ::-1])
        warm_points.extend(mirrored_warm_point(point) for point in good)
    source_xy = np.asarray(source_xy, float)
    policy = dict(
        conditions=asdict(s.CONDITIONS),
        logical_frequency_mhz=a.logical_frequency_mhz,
        ec_step_mhz=a.ec_step_mhz,
        ec1_range_mhz=a.ec1_range,
        ec2_range_mhz=a.ec2_range,
        target_domain="explicit_rectangle" if a.ec1_range else a.domain,
        orientation_domain=a.orientation_domain,
        hull_scale=a.hull_scale,
        readout_label_swap_symmetry=a.orientation_domain == "full",
        targets=targets,
        original_orientation_targets=len(original_targets),
        mirrored_orientation_targets=len(mirrored_targets),
        selection=selection,
        source_file=str(source),
        source_sha256=sha256(source.read_bytes()).hexdigest(),
        source_witnesses=len(good),
        effective_warm_witnesses=len(warm_points),
        capacitance_range_ff=[36, 128],
        budget=dict(starts=3, max_nfev=35, screening_soft_seconds=25),
        optimize_readout=False,
        retry_failed=False,
        failure_diagnosis_enabled=False,
    )
    from readout_workflow.validation import POLICY

    files = [
        Path(__file__),
        *[
            ROOT / "readout_workflow" / name
            for name in ("calculation.py", "model.py", "validation.py")
        ],
        *sorted((ROOT / "core").glob("*.py")),
    ]
    policy["validation_policy"] = POLICY
    policy["source_hashes"] = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files
    }
    settings_path = OUT / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != json.loads(
        json.dumps(policy)
    ):
        raise RuntimeError(
            "Saved settings/source/validation policy differ. Use a fresh output directory."
        )
    atomic_json(settings_path, policy)
    worst = 0
    for point in good:
        device = point["device"]
        c1, c2 = capacitances(
            point["loaded_ec1_mhz"],
            point["loaded_ec2_mhz"],
            device["C12_fF"],
            device["C1r_fF"],
        )
        worst = max(
            worst, abs(c1 - device["C1_shunt_fF"]), abs(c2 - device["C2_shunt_fF"])
        )
    assert worst < 1e-8
    atomic_json(
        OUT / "capacitance_reconstruction_check.json",
        dict(witnesses_checked=len(good), max_capacitance_error_ff=worst),
    )

    def filename(ec):
        return OUT / f"ec_{ec[0]:g}_{ec[1]:g}.json"

    rows = [
        json.loads(filename(ec).read_text()) for ec in targets if filename(ec).exists()
    ]
    done = {(row["ec1_target_mhz"], row["ec2_target_mhz"]) for row in rows}

    def progress(phase="running"):
        atomic_json(
            OUT / "progress.json",
            dict(
                phase=phase,
                total=len(targets),
                completed=len(rows),
                passed=sum(row["status"] == "validated_pass" for row in rows),
                updated_utc=utc(),
                pid=os.getpid(),
                logical_frequency_mhz=a.logical_frequency_mhz,
                ec_step_mhz=a.ec_step_mhz,
                optimize_readout=False,
                failure_diagnosis_enabled=False,
            ),
        )

    progress()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {}
        for ec in targets:
            if ec in done:
                continue
            near = np.argsort(np.sum((source_xy - np.array(ec)) ** 2, axis=1))[:3]
            membership = dict(
                in_original_scaled_hull=ec in original_targets,
                in_mirrored_scaled_hull=ec in mirrored_targets,
            )
            task = dict(
                ec=ec,
                warm=[warm_points[i] for i in near],
                logical_frequency_ghz=logical_frequency_ghz,
            )
            futures[pool.submit(solve, task)] = (ec, membership)
        for future in as_completed(futures):
            ec, membership = futures[future]
            row = future.result() | membership
            rows.append(row)
            atomic_json(filename(ec), row)
            progress()
            print(
                json.dumps(
                    dict(
                        ec=ec,
                        status=row["status"],
                        seconds=row["seconds"],
                        completed=len(rows),
                    )
                ),
                flush=True,
            )
    rows.sort(key=lambda row: (row["ec1_target_mhz"], row["ec2_target_mhz"]))
    atomic_json(OUT / "results.json", rows)
    progress("complete")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
