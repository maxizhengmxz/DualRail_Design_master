"""Optional capacitor-variation study. No work is launched on import."""

from __future__ import annotations
import json
import multiprocessing
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from floating_coupler.batch import _initialize, _lock, _write
from floating_coupler.fabrication import parameter_names
from .policy import campaign_source_signature
from .archive import iter_records
from .robustness_worker import worker


def usable(row):
    best = row.get("gate", {}).get("best", {})
    return all(
        np.isfinite(best.get(k, np.nan))
        for k in ("raw_cz_fidelity", "postselected_cz_fidelity")
    )


def draw_errors(names, seed, index, attempt, lower, upper):
    """Uniform maximum within a stratum, uniform signed cube face given max."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, attempt]))
    maximum = float(rng.uniform(lower, upper))
    values = rng.uniform(-maximum, maximum, len(names))
    face = int(rng.integers(len(names)))
    values[face] = maximum * rng.choice([-1.0, 1.0])
    return dict(zip(names, map(float, values)))


def max_deviation(row):
    return max(map(abs, row.get("errors", {}).values()), default=0.0)


def report(folder):
    from .robustness_report import generate

    return generate(folder)


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(args.input).read_text())
    bins = int(round(args.deviation_percent / args.bin_width_percent))
    if args.sampling == "max-bins" and not np.isclose(
        bins * args.bin_width_percent, args.deviation_percent
    ):
        raise ValueError("Deviation range must contain an integer number of bins")
    edges = np.linspace(0.0, args.deviation_percent / 100, bins + 1)
    names = parameter_names(base)
    initial = np.zeros(bins, int)
    if args.reuse:
        initial[:] = np.histogram(
            [max_deviation(r) for r in iter_records(args.reuse) if usable(r)],
            bins=edges,
        )[0]
    need = (
        np.maximum(args.target_per_bin - initial, 0)
        if args.sampling == "max-bins"
        else np.array([args.samples], dtype=int)
    )
    nslots = len(need)
    signature = campaign_source_signature()
    manifest_path = out / "campaign.json"
    with _lock(out):
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            immutable = dict(
                sampling=args.sampling,
                seed=args.seed,
                input=base,
                reuse=str(Path(args.reuse).resolve()) if args.reuse else None,
                deviation_percent=args.deviation_percent,
                bin_width_percent=args.bin_width_percent,
                target_per_bin=args.target_per_bin,
                samples=args.samples,
            )
            if (
                manifest["source_signature"] != signature
                or manifest["request"] != immutable
            ):
                raise ValueError("Source or inputs changed; use a fresh output folder")
        else:
            immutable = dict(
                sampling=args.sampling,
                seed=args.seed,
                input=base,
                reuse=str(Path(args.reuse).resolve()) if args.reuse else None,
                deviation_percent=args.deviation_percent,
                bin_width_percent=args.bin_width_percent,
                target_per_bin=args.target_per_bin,
                samples=args.samples,
            )
            manifest = dict(
                study="robustness",
                request=immutable,
                source_signature=signature,
                workers=args.workers,
                started_utc=datetime.now(timezone.utc).isoformat(),
                deadline_epoch=time.time() + args.hours * 3600,
                initial_finite_counts=initial.tolist(),
                required_new=need.tolist(),
                input_file="input.json",
                reuse_archive=immutable["reuse"],
                counting="Both gate fidelities finite; failed diagnostics retained",
                physics="Fabrication changes EC/J; no self-C compensation; standard recovery ladder",
            )
            _write(manifest_path, manifest)
            _write(out / "input.json", base)
        samples = out / "samples"
        samples.mkdir(exist_ok=True)
        attempts, counts = np.zeros(nslots, int), np.zeros(nslots, int)
        for path in samples.glob("*.json"):
            row = json.loads(path.read_text())
            i = row["sampling"]["bin_index"]
            attempts[i] += 1
            counts[i] += usable(row)
        caps = need + np.maximum(10, np.ceil(0.1 * need).astype(int))
        caps[need == 0] = 0
        pending, last_report, stop = {}, time.time(), None

        def status(state):
            _write(
                out / "status.json",
                dict(
                    state=state,
                    pid=__import__("os").getpid(),
                    updated_utc=datetime.now(timezone.utc).isoformat(),
                    workers=args.workers,
                    new_finite_points=int(counts.sum()),
                    completed_attempts=int(attempts.sum()),
                    initial_finite_counts=initial.tolist(),
                    new_finite_counts=counts.tolist(),
                    remaining_per_bin=np.maximum(need - counts, 0).tolist(),
                    active=[t["id"] for t in pending.values()],
                    stop_reason=stop,
                ),
            )

        def save(future, task):
            try:
                row = future.result()
            except Exception as error:
                row = dict(
                    id=task["id"],
                    kind=task["kind"],
                    errors=task["errors"],
                    status="worker_error",
                    error=repr(error),
                )
            row["sampling"] = task["sampling"]
            row["completed_utc"] = datetime.now(timezone.utc).isoformat()
            _write(samples / f"{row['id']:08d}.json", row)
            i = task["sampling"]["bin_index"]
            attempts[i] += 1
            counts[i] += usable(row)

        status("running")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
        ) as pool:
            while True:
                reserved = np.zeros(nslots, int)
                for task in pending.values():
                    reserved[task["sampling"]["bin_index"]] += 1
                if (out / "STOP").exists():
                    stop = "STOP file"
                elif time.time() >= manifest["deadline_epoch"] - 180:
                    stop = "report reserve"
                elif campaign_source_signature() != signature:
                    stop = "source changed"
                while not stop and len(pending) < args.workers:
                    choices = [
                        i
                        for i in range(nslots)
                        if counts[i] + reserved[i] < need[i]
                        and attempts[i] + reserved[i] < caps[i]
                    ]
                    if not choices:
                        break
                    i = min(
                        choices,
                        key=lambda k: ((counts[k] + reserved[k]) / max(1, need[k]), k),
                    )
                    attempt = int(attempts[i] + reserved[i])
                    if args.sampling == "max-bins":
                        errors = draw_errors(
                            names, args.seed, i, attempt, edges[i], edges[i + 1]
                        )
                        kind = "stratified-max"
                    else:
                        rng = np.random.default_rng(
                            np.random.SeedSequence([args.seed, attempt])
                        )
                        errors = dict(
                            zip(
                                names,
                                map(
                                    float,
                                    rng.uniform(
                                        -args.deviation_percent / 100,
                                        args.deviation_percent / 100,
                                        len(names),
                                    ),
                                ),
                            )
                        )
                        kind = "independent-all"
                    task = dict(
                        id=10_000_000 + i * 100_000 + attempt,
                        kind=kind,
                        errors=errors,
                        base=base,
                        sampling=dict(bin_index=i, attempt=attempt),
                        representative=attempt % 20 == 0,
                    )
                    pending[pool.submit(worker, task)] = task
                    reserved[i] += 1
                if not pending:
                    break
                done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    save(future, pending.pop(future))
                status("running")
                if time.time() - last_report > 3600:
                    for future in list(pending):
                        save(future, pending.pop(future))
                    report(out)
                    last_report = time.time()
        status("completed" if np.all(counts >= need) else "partial")
        return report(out)
