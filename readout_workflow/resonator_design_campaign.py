"""Checkpointed four-worker campaign over the one strict EC half-plane.

Each child invokes the standard resonator-robust-design optimizer. This module
only schedules work: it does not alter physics, thresholds, or optimization.
Original input data are read-only. Resume the same command after interruption.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from .storage import atomic_json, utc

ROOT = Path(__file__).resolve().parents[1]


def command_for(
    ec,
    output,
    starts,
    maxiter,
    max_halfwidth,
    quick=False,
    conditions=None,
    boundary_resolution=0.025,
    refine_from=None,
):
    x, y = ec
    command = [
        sys.executable,
        "-u",
        "-m",
        "readout_workflow.resonator_robust_design",
        "--run",
        "--mode",
        "tolerance",
        "--ec",
        str(x),
        str(y),
        "--max-halfwidth-ghz",
        str(max_halfwidth),
        "--starts",
        str(starts),
        "--maxiter",
        str(maxiter),
        "--frequency-step-ghz",
        "0.05",
        "--boundary-resolution-ghz",
        str(boundary_resolution),
        "--input",
        str(output / f"seed_{x}_{y}.json"),
        "--conditions",
        str(
            conditions
            or Path("readout_workflow/data/B_ec_resonator_12h_20261109/settings.json")
        ),
        "--output",
        str(output / f"ec_{x}_{y}"),
    ]
    if quick:
        command += ["--quick-tolerance"]
    if refine_from:
        coarse = Path(refine_from) / f"ec_{x}_{y}"
        complete = coarse / "completion.json"
        if (
            complete.exists()
            and json.loads(complete.read_text()).get("phase") == "complete"
        ):
            command += ["--refine-from", str(coarse)]
    return command


def launch(ec, args):
    path = args.output / f"ec_{ec[0]}_{ec[1]}"
    path.mkdir(exist_ok=True, parents=True)
    if (path / "completion.json").exists() and json.loads(
        (path / "completion.json").read_text()
    ).get("phase") == "complete":
        return dict(ec_mhz=ec, status="complete", reused=True)
    changed = [
        name
        for name, value in getattr(args, "expected_hashes", {}).items()
        if sha256(Path(name).read_bytes()).hexdigest() != value
    ]
    if changed:
        return dict(ec_mhz=ec, status="source_changed", files=changed)
    tic = time.monotonic()
    with (
        (path / "run.log").open("a", encoding="utf-8") as out,
        (path / "run.stderr.log").open("a", encoding="utf-8") as err,
    ):
        result = subprocess.run(
            command_for(
                ec,
                args.output,
                args.starts,
                args.maxiter,
                args.max_halfwidth_ghz,
                args.quick_tolerance,
                args.conditions,
                args.boundary_resolution_ghz,
                args.refine_from,
            ),
            cwd=ROOT,
            stdout=out,
            stderr=err,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    complete = (path / "completion.json").exists() and json.loads(
        (path / "completion.json").read_text()
    ).get("phase") == "complete"
    return dict(
        ec_mhz=ec,
        status="complete"
        if complete and result.returncode == 0
        else "computation_error",
        returncode=result.returncode,
        elapsed_seconds=time.monotonic() - tic,
        updated_utc=utc(),
    )


def selected_rows(data, orientation="ec2-gt-ec1", step=2):
    """Fold equivalent labels for seeds only; recalculate every selected design."""
    preferred = {}
    swapped = {}
    for row in data:
        point = row.get("nominal") or row.get("point")
        if point is None:
            continue
        x, y = row["ec1_target_mhz"], row["ec2_target_mhz"]
        if x == y:
            continue
        correct = (y > x) if orientation == "ec2-gt-ec1" else (x > y)
        if correct:
            preferred[x, y] = dict(row, nominal=point)
        else:
            q = deepcopy(point)
            for a, b in [("C1_shunt_fF", "C2_shunt_fF"), ("C1r_fF", "C2r_fF")]:
                q["device"][a], q["device"][b] = q["device"][b], q["device"][a]
            for a, b in [
                ("ej1_ghz", "ej2_ghz"),
                ("bare_f1_ghz", "bare_f2_ghz"),
                ("loaded_ec1_mhz", "loaded_ec2_mhz"),
                ("ej1_over_ec1", "ej2_over_ec2"),
                ("g1r_over_2pi_mhz", "g2r_over_2pi_mhz"),
                ("participation1", "participation2"),
            ]:
                if a in q and b in q:
                    q[a], q[b] = q[b], q[a]
            # Never label a mirrored seed as a recalculated result/certificate.
            for name in (
                "chi_sign_stability",
                "chi_sign_probe_basis",
                "validation",
                "x",
            ):
                q.pop(name, None)
            q = {
                k: q[k]
                for k in (
                    "device",
                    "ej1_ghz",
                    "ej2_ghz",
                    "bare_f1_ghz",
                    "bare_f2_ghz",
                    "loaded_ec1_mhz",
                    "loaded_ec2_mhz",
                )
            }
            q["label_swapped_seed_only"] = True
            swapped[y, x] = dict(
                ec1_target_mhz=y, ec2_target_mhz=x, nominal=q, source_ec_mhz=[x, y]
            )
    rows = {**swapped, **preferred}
    return [
        r
        for (x, y), r in rows.items()
        if (x - 150) % step == 0 and (y - 150) % step == 0
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument(
        "--stop-file",
        type=Path,
        help="Stop scheduling after this file appears; finish active tuple checkpoints",
    )
    p.add_argument(
        "--refine-from",
        type=Path,
        help="Coarse campaign root; refine matching completed tuples and search missing tuples normally",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--starts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=50)
    p.add_argument("--max-halfwidth-ghz", type=float, default=1.0)
    p.add_argument("--wait-for-pilots", nargs="*", default=[])
    p.add_argument("--pilot-root", type=Path)
    p.add_argument("--ec-step-mhz", type=int, default=8)
    p.add_argument(
        "--orientation", choices=["ec2-gt-ec1", "ec1-gt-ec2"], default="ec2-gt-ec1"
    )
    p.add_argument("--boundary-resolution-ghz", type=float, default=0.025)
    p.add_argument("--quick-tolerance", action="store_true")
    p.add_argument(
        "--source",
        type=Path,
        default=Path("readout_workflow/data/B_ec_resonator_12h_20261109/results.json"),
    )
    p.add_argument(
        "--conditions",
        type=Path,
        default=Path("readout_workflow/data/B_ec_resonator_12h_20261109/settings.json"),
    )
    a = p.parse_args()
    if not a.run:
        print("Use --run explicitly.")
        return
    if a.ec_step_mhz < 2 or a.ec_step_mhz % 2:
        raise ValueError("EC step must be a multiple of 2 MHz")
    if a.boundary_resolution_ghz <= 0:
        raise ValueError("Boundary resolution must be positive")
    if min(a.workers, a.starts, a.maxiter) < 1:
        raise ValueError("workers, starts and maxiter must be positive")
    source = a.source
    if not source.exists() or not a.conditions.exists():
        raise FileNotFoundError(
            "Provide available --source results.json and --conditions settings.json; generated study data are not shipped in Git."
        )
    if a.refine_from:
        if not a.refine_from.is_dir():
            raise ValueError("Coarse campaign directory does not exist")
        if a.refine_from.resolve() == a.output.resolve():
            raise ValueError("Refinement campaign must use a separate output directory")
    data = json.loads(source.read_text())
    rows = selected_rows(data, a.orientation, a.ec_step_mhz)
    # Coarse 4 MHz coverage first, then fill the remaining 2 MHz lattice phases.
    rows.sort(
        key=lambda r: (
            ((r["ec1_target_mhz"] - 150) % (2 * a.ec_step_mhz)) // a.ec_step_mhz
            + 2 * ((r["ec2_target_mhz"] - 150) % (2 * a.ec_step_mhz)) // a.ec_step_mhz,
            r["ec1_target_mhz"],
            r["ec2_target_mhz"],
        )
    )
    a.output.mkdir(exist_ok=True, parents=True)
    # The launcher may be started while explicitly requested timing pilots are
    # finishing. Never duplicate those tuples or exceed four numerical workers.
    if a.wait_for_pilots:
        pilot_root = a.pilot_root or a.output
        until = time.monotonic() + 1800
        while any(
            not (pilot_root / f"ec_{key}" / "completion.json").exists()
            for key in a.wait_for_pilots
        ):
            atomic_json(
                a.output / "campaign_progress.json",
                dict(
                    phase="waiting_for_pilots",
                    pilots=a.wait_for_pilots,
                    updated_utc=utc(),
                ),
            )
            if time.monotonic() > until:
                raise RuntimeError(
                    "Timing pilots did not finish within 30 minutes; inspect their logs before restart"
                )
            time.sleep(10)
        timings = [
            json.loads((pilot_root / f"ec_{key}" / "completion.json").read_text())[
                "elapsed_seconds"
            ]
            for key in a.wait_for_pilots
        ]
        atomic_json(
            a.output / "benchmark_summary.json",
            dict(
                pilot_keys=a.wait_for_pilots,
                seconds=timings,
                projected_total_hours=sum(timings)
                / len(timings)
                * len(rows)
                / a.workers
                / 3600,
                workers=a.workers,
                updated_utc=utc(),
            ),
        )
    dependencies = [
        "readout_workflow/resonator_design_campaign.py",
        "readout_workflow/resonator_robust_design.py",
        "readout_workflow/resonator_refinement.py",
        "readout_workflow/ec_design_improvement.py",
        "readout_workflow/ec_resonator_tolerance.py",
        "readout_workflow/direct_ec_feasibility.py",
        "readout_workflow/calculation.py",
        "readout_workflow/model.py",
        "readout_workflow/validation.py",
        "core/transmon.py",
        "core/electrostatics.py",
        "core/operating_point_dispersive.py",
        "core/circuits.py",
        "core/constants.py",
        "core/runtime.py",
        "core/tracking.py",
    ]
    coarse_manifest = {}
    if a.refine_from:
        for r in rows:
            key = f"ec_{r['ec1_target_mhz']}_{r['ec2_target_mhz']}"
            coarse = a.refine_from / key
            if (coarse / "completion.json").exists():
                coarse_manifest[key] = {
                    name: sha256((coarse / name).read_bytes()).hexdigest()
                    for name in ("settings.json", "completion.json", "results.json")
                    if (coarse / name).exists()
                }
    identity = dict(
        refinement_source=str(a.refine_from.resolve()) if a.refine_from else None,
        coarse_manifest=coarse_manifest,
        conditions_path=str(a.conditions.resolve()),
        conditions_sha256=sha256(a.conditions.read_bytes()).hexdigest(),
        source_hashes={
            f: sha256(Path(f).read_bytes()).hexdigest() for f in dependencies
        },
        targets=[[r["ec1_target_mhz"], r["ec2_target_mhz"]] for r in rows],
        workers=a.workers,
        ec_step_mhz=a.ec_step_mhz,
        orientation=a.orientation,
        quick_tolerance=a.quick_tolerance,
        boundary_resolution_ghz=a.boundary_resolution_ghz,
        starts=a.starts,
        maxiter=a.maxiter,
        max_halfwidth_ghz=a.max_halfwidth_ghz,
        source_sha256=sha256(source.read_bytes()).hexdigest(),
        driver_sha256=sha256(
            Path("readout_workflow/resonator_robust_design.py").read_bytes()
        ).hexdigest(),
        objective="Largest validated symmetric interval around bare resonator 6 GHz, flexible EJ/operation recalibration; chi>=1 MHz and H>=95% remain acceptance constraints only (no secondary strength maximization). One strict half-plane; mirrored seeds only, no symmetry-cloned outputs.",
    )
    saved = a.output / "campaign_settings.json"
    if saved.exists() and json.loads(saved.read_text()) != identity:
        raise ValueError("Campaign identity changed; use a fresh folder")
    atomic_json(saved, identity)
    a.expected_hashes = identity["source_hashes"]
    pending = []
    results = []
    for r in rows:
        ec = [r["ec1_target_mhz"], r["ec2_target_mhz"]]
        seed = a.output / f"seed_{ec[0]}_{ec[1]}.json"
        if not seed.exists():
            atomic_json(seed, dict(point=r.get("nominal") or r.get("point")))
        path = a.output / f"ec_{ec[0]}_{ec[1]}"
        if (path / "completion.json").exists() and json.loads(
            (path / "completion.json").read_text()
        ).get("phase") == "complete":
            results.append(dict(ec_mhz=ec, status="complete", reused=True))
        else:
            pending.append(ec)
    tic = time.monotonic()
    iterator = iter(pending)
    count_new = 0

    def progress(phase):
        elapsed = time.monotonic() - tic
        return dict(
            phase=phase,
            completed=sum(r["status"] == "complete" for r in results),
            errors=sum(r["status"] != "complete" for r in results),
            total=len(rows),
            workers=a.workers,
            elapsed_seconds=elapsed,
            estimated_remaining_hours=elapsed
            / max(1, count_new)
            * (len(rows) - len(results))
            / 3600
            if count_new
            else None,
            updated_utc=utc(),
        )

    atomic_json(a.output / "campaign_progress.json", progress("running"))
    halted_source = False
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {}

        def schedule():
            if halted_source or (a.stop_file and a.stop_file.exists()):
                return
            try:
                ec = next(iterator)
            except StopIteration:
                return
            futures[pool.submit(launch, ec, a)] = ec

        for _ in range(a.workers):
            schedule()
        while futures:
            done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for fut in done:
                ec = futures.pop(fut)
                try:
                    r = fut.result()
                except Exception as exc:
                    r = dict(ec_mhz=ec, status="computation_error", error=repr(exc))
                results.append(r)
                if r["status"] == "source_changed":
                    halted_source = True
                count_new += 1
                atomic_json(a.output / "campaign_results.json", results)
                print(json.dumps(r), flush=True)
                schedule()
            atomic_json(a.output / "campaign_progress.json", progress("running"))
    atomic_json(a.output / "campaign_results.json", results)
    atomic_json(
        a.output / "campaign_progress.json",
        progress(
            "blocked_source_changed"
            if halted_source
            else "paused"
            if len(results) < len(rows)
            else "complete"
            if all(r["status"] == "complete" for r in results)
            else "complete_with_errors"
        ),
    )


if __name__ == "__main__":
    main()
