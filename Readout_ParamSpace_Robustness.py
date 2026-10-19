"""Entry point for the single-junction readout parameter-space calculation.

Opening this file or running it without arguments only shows saved status.
The Hamiltonian/conditions are in readout_workflow/calculation.py; the nominal
strength solver is optimization.py; the resumable map is workflow.py.

Commands from the repository root:
    python Readout_ParamSpace_Robustness.py status
    python Readout_ParamSpace_Robustness.py run --workers 4 --background

Only the nominal map and its failure diagnosis are implemented here. No
robustness or report generation is automatically launched.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from readout_workflow.storage import atomic_json, status, utc

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "readout_workflow/data/current_map"


def process_alive(pid):
    """Query a process without sending it a signal or risking PID termination."""
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, 0, int(pid))
    if not handle:
        return False
    try:
        result = ctypes.c_uint32()
        return (
            bool(kernel.GetExitCodeProcess(handle, ctypes.byref(result)))
            and result.value == 259
        )
    finally:
        kernel.CloseHandle(handle)


def child_command(args, supervise=False):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--output",
        str(args.output),
        "--workers",
        str(args.workers),
    ]
    if supervise:
        command.append("--supervise")
    if args.pair:
        command += ["--pair", *map(str, args.pair)]
    if args.limit:
        command += ["--limit", str(args.limit)]
    return command


def launch(args):
    """Launch a hidden supervisor; preserve per-event checkpoints on interruption."""
    args.output.mkdir(parents=True, exist_ok=True)
    launch_file = args.output / "launch.json"
    if launch_file.exists():
        previous = json.loads(launch_file.read_text(encoding="utf-8"))
        if process_alive(previous.get("pid")):
            raise RuntimeError("A supervisor is already running for this output")
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    with (args.output / "run.log").open("ab") as log:
        settings = dict(
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
        flags = (
            subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            if os.name == "nt"
            else 0
        )
        breakaway = os.name == "nt"
        try:
            process = subprocess.Popen(
                child_command(args, True),
                **settings,
                creationflags=flags
                | (subprocess.CREATE_BREAKAWAY_FROM_JOB if os.name == "nt" else 0),
            )
        except OSError as error:
            if getattr(error, "winerror", None) != 5:
                raise
            breakaway = False
            process = subprocess.Popen(
                child_command(args, True), **settings, creationflags=flags
            )
    record = dict(
        pid=process.pid,
        started_utc=utc(),
        scope="map_only",
        breakaway_from_terminal_job=breakaway,
    )
    atomic_json(launch_file, record)
    print(json.dumps(record))


def supervise(args):
    """Bounded automatic retries; successful map completion ends the supervisor."""
    with (args.output / "supervisor.jsonl").open("a", encoding="utf-8") as log:
        for attempt in range(1, 11):
            result = subprocess.run(child_command(args), cwd=ROOT, check=False)
            log.write(
                json.dumps(
                    dict(utc=utc(), attempt=attempt, returncode=result.returncode)
                )
                + "\n"
            )
            log.flush()
            if result.returncode == 0:
                return
            time.sleep(30)
    raise RuntimeError("Retry budget exhausted; committed results are preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=("status", "run"), default="status"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pair", nargs=2, type=int, metavar=("C1", "C2"))
    parser.add_argument(
        "--limit", type=int, help="Small validation run; use a separate --output"
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.command == "status":
        print(json.dumps(status(args.output), indent=2))
    elif args.background:
        launch(args)
    elif args.supervise:
        supervise(args)
    else:
        from readout_workflow.workflow import run

        run(args)


if __name__ == "__main__":
    main()
