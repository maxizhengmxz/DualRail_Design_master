"""Read-only, three-hour health checks for the running map calculation.

Writes a compact local report/history and optionally requests a Windows
notification. This does not schedule ChatGPT messages, modify calculations,
restart jobs, or perform robustness. It exits on completion, pause or failure.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from Readout_ParamSpace_Robustness import DEFAULT_OUTPUT, ROOT, process_alive

from .storage import atomic_json

CHINA = timezone(timedelta(hours=8))
TERMINAL = {"map_complete", "paused", "paused_after_cleanup", "stopped_by_user"}


def inspect(output, previous=None, now=None):
    """Report health from committed events, process liveness and new log errors."""
    output = Path(output)
    now = now or datetime.now(UTC)
    snapshot = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    launch = json.loads((output / "launch.json").read_text(encoding="utf-8"))
    connection = sqlite3.connect(output / "samples.sqlite", timeout=10)
    try:
        connection.execute("BEGIN")
        counts = dict(
            connection.execute("SELECT phase,count(*) FROM events GROUP BY phase")
        )
        last = connection.execute(
            "SELECT json_extract(payload,'$.completed_utc') FROM events ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        connection.close()
    supervisor = process_alive(launch.get("pid"))
    calculator = process_alive(snapshot.get("pid"))
    age = (now - datetime.fromisoformat(last[0])).total_seconds() if last else None
    # A resumed job legitimately starts with old committed timestamps. Allow
    # its first new point to finish before treating that history as a stall.
    launch_age = (now - datetime.fromisoformat(launch["started_utc"])).total_seconds()
    phase = snapshot["phase"]
    total = sum(counts.values())
    delta = total - previous["total_events"] if previous else None
    log_path = output / "run.log"
    offset = log_path.stat().st_size if log_path.exists() else 0
    old_offset = previous.get("log_offset", 0) if previous else max(0, offset - 65536)
    if old_offset > offset:
        old_offset = 0
    new_errors = []
    if log_path.exists():
        with log_path.open("rb") as stream:
            stream.seek(old_offset)
            text = stream.read().decode("utf-8", errors="replace")
        needles = (
            "Traceback (most recent call last)",
            "PermissionError:",
            "BrokenProcessPool:",
            "RuntimeError:",
            "MemoryError:",
            "OSError:",
        )
        new_errors = [
            line.strip()
            for line in text.splitlines()
            if any(marker in line for marker in needles)
        ][-8:]
    issues = []
    if integrity != "ok":
        issues.append("SQLite integrity check failed")
    if counts.get("robustness", 0):
        issues.append("Unexpected robustness events")
    if phase not in TERMINAL:
        if not supervisor and not calculator:
            issues.append("Supervisor and calculator are no longer running")
        elif not calculator:
            issues.append("Calculator is not running; supervisor may be retrying")
        if age is not None and age > 1800 and launch_age > 1800:
            issues.append("No committed event in the last 30 minutes")
        if delta == 0:
            issues.append("No additional events since the previous check")
        if new_errors:
            issues.append("New errors appeared in the calculation log")
    if issues:
        state = "needs_attention"
    elif phase == "map_complete":
        state = "completed"
    elif phase in TERMINAL:
        state = "paused"
    else:
        state = "running_smoothly"
    return dict(
        checked_utc=now.isoformat(),
        checked_china=now.astimezone(CHINA).isoformat(),
        state=state,
        phase=phase,
        event_counts=counts,
        total_events=total,
        events_since_previous=delta,
        supervisor_alive=supervisor,
        calculator_alive=calculator,
        last_event_age_seconds=age,
        sqlite_integrity=integrity,
        issues=issues,
        new_errors=new_errors,
        log_offset=offset,
        robustness_samples=counts.get("robustness", 0),
    )


def notify(report):
    """Request a nonblocking Windows notification; never open a visible console."""
    if os.name != "nt":
        return False
    title = "Readout calculation: " + report["state"].replace("_", " ")
    counts = report["event_counts"]
    message = (
        f"Phase: {report['phase']}. First pass: {counts.get('feasibility_normal', 0)}/4278. "
        f"Saved events: {report['total_events']}. "
        + (
            "; ".join(report["issues"])
            if report["issues"]
            else "Checkpoints healthy; robustness disabled."
        )
    )[:240]

    def quote(s):
        return "'" + s.replace("'", "''") + "'"

    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "Add-Type -AssemblyName System.Drawing; "
        "$notice = New-Object System.Windows.Forms.NotifyIcon; "
        "$notice.Icon = [System.Drawing.SystemIcons]::Information; "
        "$notice.Visible = $true; "
        f"$notice.BalloonTipTitle = {quote(title)}; "
        f"$notice.BalloonTipText = {quote(message)}; "
        "$notice.ShowBalloonTip(10000); Start-Sleep -Seconds 10; "
        "$notice.Visible = $false; $notice.Dispose();"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-STA",
                "-EncodedCommand",
                encoded,
            ],
            creationflags=subprocess.CREATE_NO_WINDOW,
            capture_output=True,
            timeout=25,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def publish(output, report, notification=False):
    """Keep one readable current report and a small append-only check history."""
    if notification:
        report["windows_notification_requested"] = notify(report)
    atomic_json(output / "health.json", report)
    with (output / "health_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report) + "\n")
    counts = report["event_counts"]
    lines = [
        "# Readout calculation health",
        "",
        f"Checked (China time): {report['checked_china']}",
        f"Status: {report['state']}",
        f"Phase: {report['phase']}",
        f"First-pass results: {counts.get('feasibility_normal', 0)} / 4278",
        f"All committed events: {report['total_events']}",
        f"Events since previous check: {report['events_since_previous']}",
        f"Database integrity: {report['sqlite_integrity']}",
        f"Robustness events: {report['robustness_samples']}",
        "",
        "Issues: " + ("; ".join(report["issues"]) if report["issues"] else "none"),
        "",
        "This is a local automated check, not a scheduled ChatGPT message.",
    ]
    (output / "health.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


def watch(output, hours=3, notifications=False, once=False):
    output = Path(output)
    previous = None
    interval = hours * 3600
    while True:
        deadline = time.monotonic() + interval
        report = inspect(output, previous)
        publish(output, report, notifications)
        if once or report["state"] in {"completed", "paused", "needs_attention"}:
            return
        previous = report
        while time.monotonic() < deadline:
            time.sleep(min(60, deadline - time.monotonic()))
            # End-of-run changes deserve a final report even before three hours.
            try:
                phase = json.loads((output / "progress.json").read_text())["phase"]
                if phase in TERMINAL:
                    break
            except (OSError, ValueError, KeyError):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.hours <= 0:
        parser.error("hours must be positive")
    if args.background:
        file = args.output / "monitor_launch.json"
        if file.exists() and process_alive(json.loads(file.read_text()).get("pid")):
            raise RuntimeError("A health monitor is already running")
        command = [
            sys.executable,
            "-m",
            "readout_workflow.monitor",
            "--output",
            str(args.output),
            "--hours",
            str(args.hours),
        ]
        if args.notify:
            command.append("--notify")
        with (args.output / "monitor.log").open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    | subprocess.DETACHED_PROCESS
                    | subprocess.CREATE_BREAKAWAY_FROM_JOB
                ),
            )
        atomic_json(
            file,
            dict(
                pid=process.pid,
                interval_hours=args.hours,
                started_utc=datetime.now(UTC).isoformat(),
                notifications=args.notify,
                delivery="local health.md and Windows notification; no automatic chat messages",
            ),
        )
        print(f"Started three-hour health monitor, PID {process.pid}")
    else:
        watch(args.output, args.hours, args.notify, args.once)


if __name__ == "__main__":
    main()
