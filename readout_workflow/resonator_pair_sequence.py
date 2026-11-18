"""Run B fine, A coarse, then refine A; checkpoint and stop between tasks.

Explicit --run only. Input snapshots and a strict code fingerprint protect
restart consistency. Stage campaigns finish active tuples when PAUSE_REQUESTED
appears. Forced interruption preserves each optimizer/validation checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from Readout_ParamSpace_Robustness import process_alive

from .storage import atomic_json, utc

ROOT = Path(__file__).resolve().parents[1]


def stage_plan(output, workers=4):
    """All stages use one half-plane and the complete original 2 MHz lattice."""
    output = Path(output)
    common = [
        sys.executable,
        "-u",
        "-m",
        "readout_workflow.resonator_design_campaign",
        "--run",
        "--workers",
        str(workers),
        "--ec-step-mhz",
        "2",
        "--orientation",
        "ec2-gt-ec1",
        "--starts",
        "3",
        "--maxiter",
        "50",
        "--stop-file",
        str(output / "PAUSE_REQUESTED"),
    ]
    specs = [
        ("B_fine", "B", 0.00625, None),
        ("A_coarse", "A", 0.025, None),
        ("A_refined", "A", 0.00625, "A_coarse"),
    ]
    stages = []
    for name, pair, resolution, coarse in specs:
        directory = output / name
        cmd = common + [
            "--source",
            str(output / "input_snapshot" / pair / "results.json"),
            "--conditions",
            str(output / "input_snapshot" / pair / "settings.json"),
            "--boundary-resolution-ghz",
            str(resolution),
            "--output",
            str(directory),
        ]
        if coarse:
            cmd += ["--refine-from", str(output / coarse)]
        stages.append(
            dict(
                name=name,
                pair=pair,
                boundary_resolution_ghz=resolution,
                output=str(directory),
                command=cmd,
            )
        )
    return stages


@contextmanager
def exclusive_run(folder):
    path = folder / "sequence.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        previous = json.loads(path.read_text())
        if process_alive(previous.get("pid")):
            raise RuntimeError("This sequence already has a live scheduler") from None
        path.unlink()
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, json.dumps(dict(pid=os.getpid(), started_utc=utc())).encode())
    os.close(descriptor)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def read_status(path):
    for _ in range(3):
        try:
            return json.loads(path.read_text()) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            time.sleep(0.1)
    return {}


def run(output, workers):
    output.mkdir(parents=True, exist_ok=True)
    stages = stage_plan(output, workers)
    paths = [
        Path(__file__),
        *[
            ROOT / "readout_workflow" / name
            for name in (
                "resonator_design_campaign.py",
                "resonator_robust_design.py",
                "resonator_refinement.py",
                "ec_design_improvement.py",
                "ec_resonator_tolerance.py",
                "direct_ec_feasibility.py",
                "calculation.py",
                "model.py",
                "validation.py",
                "storage.py",
            )
        ],
        *[
            ROOT / "core" / name
            for name in (
                "transmon.py",
                "electrostatics.py",
                "operating_point_dispersive.py",
                "circuits.py",
                "constants.py",
                "runtime.py",
                "tracking.py",
            )
        ],
    ]
    hashes = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths
    }
    inputs = {}
    for pair in ("A", "B"):
        origin = (
            ROOT
            / "Ec_tuple search workflow"
            / "readout"
            / "direct_ec"
            / f"rail_{pair.lower()}_step2"
        )
        for name in ("results.json", "settings.json"):
            src = origin / name
            dest = output / "input_snapshot" / pair / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(src, dest)
            inputs[f"{pair}/{name}"] = sha256(dest.read_bytes()).hexdigest()
    identity = dict(
        stages=stages,
        workers=workers,
        orientation="ec2-gt-ec1",
        ec_step_mhz=2,
        source_hashes=hashes,
        input_hashes=inputs,
    )
    plan = output / "sequence_plan.json"
    if plan.exists() and json.loads(plan.read_text()) != identity:
        raise ValueError(
            "Sequence source/input identity changed; restore saved source or use a fresh run"
        )
    atomic_json(plan, identity)
    for file in paths:
        dest = output / "source_snapshot" / file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(file, dest)
    atomic_json(
        output / "sequence_launch.json",
        dict(pid=os.getpid(), workers=workers, started_utc=utc()),
    )
    for stage in stages:
        if (output / "PAUSE_REQUESTED").exists():
            atomic_json(
                output / "sequence_progress.json",
                dict(phase="paused", stage=stage["name"], updated_utc=utc()),
            )
            return
        changed = [
            name
            for name, value in hashes.items()
            if sha256((ROOT / name).read_bytes()).hexdigest() != value
        ]
        if changed:
            atomic_json(
                output / "sequence_progress.json",
                dict(
                    phase="blocked_source_changed",
                    stage=stage["name"],
                    files=changed,
                    updated_utc=utc(),
                ),
            )
            return
        directory = Path(stage["output"])
        progress = directory / "campaign_progress.json"
        if (
            progress.exists()
            and json.loads(progress.read_text()).get("phase") == "complete"
        ):
            continue
        directory.mkdir(exist_ok=True, parents=True)
        with (
            (directory / "supervisor.log").open("a", encoding="utf-8") as log,
            (directory / "supervisor.stderr.log").open("a", encoding="utf-8") as err,
        ):
            child = subprocess.Popen(
                stage["command"],
                cwd=ROOT,
                stdout=log,
                stderr=err,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            while child.poll() is None:
                state = read_status(progress)
                atomic_json(
                    output / "sequence_progress.json",
                    dict(
                        phase="running",
                        stage=stage["name"],
                        child_pid=child.pid,
                        campaign=state,
                        pause_requested=(output / "PAUSE_REQUESTED").exists(),
                        updated_utc=utc(),
                    ),
                )
                time.sleep(10)
        state = read_status(progress)
        atomic_json(
            output / f"{stage['name']}_exit.json",
            dict(returncode=child.returncode, campaign=state, updated_utc=utc()),
        )
        if (
            child.returncode != 0
            or state.get("phase") != "complete"
            or state.get("errors", 0)
        ):
            atomic_json(
                output / "sequence_progress.json",
                dict(
                    phase="paused"
                    if state.get("phase") == "paused"
                    else "stage_requires_review",
                    stage=stage["name"],
                    campaign=state,
                    returncode=child.returncode,
                    updated_utc=utc(),
                ),
            )
            return
    atomic_json(
        output / "sequence_progress.json",
        dict(phase="complete", stages=[s["name"] for s in stages], updated_utc=utc()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not a.run:
        print("Use --run explicitly; no calculation started.")
        return
    if a.workers < 1:
        raise ValueError("Positive workers required")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(a.output):
        run(a.output, a.workers)


if __name__ == "__main__":
    main()
