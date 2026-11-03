"""Lossless, versioned robustness data archives; readers never extract paths."""

from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

SCHEMA = "coupler-robustness-archive-v1"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_records(path):
    """Yield full records in stable archive order without filesystem extraction."""
    with tarfile.open(path, "r:xz") as archive:
        with archive.extractfile("records.jsonl") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def read_json(path, member):
    with tarfile.open(path, "r:xz") as archive:
        with archive.extractfile(member) as stream:
            return json.load(stream)


def summarize(records):
    import math

    counts = Counter(
        total=0,
        A_failed=0,
        B_failed=0,
        off_found=0,
        gate_found=0,
        gate_best_available=0,
    )
    for row in records:
        counts["total"] += 1
        for rail in ("A", "B"):
            counts[rail + "_failed"] += (
                not row.get("readout", {}).get(rail, {}).get("passed", False)
            )
        counts["off_found"] += bool(row.get("off", {}).get("found"))
        counts["gate_found"] += bool(row.get("gate", {}).get("found"))
        best = row.get("gate", {}).get("best", {})
        counts["gate_best_available"] += all(
            math.isfinite(best.get(k, float("nan")))
            for k in ("raw_cz_fidelity", "postselected_cz_fidelity")
        )
    return dict(counts)


def write_archive(path, records, inputs, provenance=None):
    """Write once, verify every member plus aggregate statistics, then publish."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.TemporaryDirectory(prefix="archive-", dir=path.parent) as temporary:
        temporary = Path(temporary)
        data = temporary / "records.jsonl"
        ids = set()
        with data.open("w", encoding="utf-8", newline="\n") as stream:
            for row in records:
                if row["id"] in ids:
                    raise ValueError(f"Duplicate record ID {row['id']}")
                ids.add(row["id"])
                stream.write(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                )
        summary = summarize(json.loads(line) for line in data.open(encoding="utf-8"))
        payloads = {
            "input.json": inputs,
            "provenance.json": provenance or {},
            "summary.json": summary,
        }
        entries = {
            "records.jsonl": dict(bytes=data.stat().st_size, sha256=file_hash(data))
        }
        for name, value in payloads.items():
            item = temporary / name
            item.write_text(
                json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
            )
            entries[name] = dict(bytes=item.stat().st_size, sha256=file_hash(item))
        manifest = {
            "schema": SCHEMA,
            "records": len(ids),
            "entries": entries,
            "description": "Full retained evaluations, not deduplicated physical settings. "
            "Failed/diagnostic outcomes and numerical history are retained.",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        candidate = temporary / "archive.tar.xz"
        with tarfile.open(candidate, "w:xz", preset=3) as archive:
            for name in ["manifest.json", *entries]:
                archive.add(temporary / name, arcname=name, recursive=False)
        verify_archive(candidate, expected=summary)
        candidate.replace(path)
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
        "summary": summary,
    }


def verify_archive(path, expected=None):
    with tarfile.open(path, "r:xz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        if manifest["schema"] != SCHEMA:
            raise ValueError("Unknown archive schema")
        for name, meta in manifest["entries"].items():
            digest, count = hashlib.sha256(), 0
            with archive.extractfile(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    count += len(block)
            if digest.hexdigest() != meta["sha256"] or count != meta["bytes"]:
                raise ValueError("Archive integrity mismatch: " + name)
    summary = summarize(iter_records(path))
    if summary["total"] != manifest["records"] or (
        expected is not None and summary != expected
    ):
        raise ValueError("Archive statistics mismatch")
    return summary
