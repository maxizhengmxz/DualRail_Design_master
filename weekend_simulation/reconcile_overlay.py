#!/usr/bin/env python3
"""Reconcile fixed hardware at every point of one readout/gate overlay.

This stage deliberately stops before the resonator-inclusive full-network check.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from readout_workflow.storage import atomic_json

HERE = Path(__file__).resolve().parent
A_RESULTS = HERE / "readout/direct_ec/rail_a_step5/results.json"
B_RESULTS = HERE / "readout/direct_ec/rail_b_step2/results.json"


def _reconcile_rail(task):
    ec, target_j, initial, logical_range = task
    from combined_feasibility_test.reconcile_and_plot import reconcile_rail
    return target_j, reconcile_rail(ec, target_j, initial, logical_range)


def _reconcile_point(task):
    item, a_result, path = task
    from combined_feasibility_test.reconcile_and_plot import reconcile_rail
    b_result = reconcile_rail(item["ec_b_mhz"], item["couplings_GHz"]["B1_B2"],
                              item["readout_b_point"], (.08, .10))
    passed = a_result["status"] == b_result["status"] == "validated_pass"
    result = {
        "ec_a_mhz": item["ec_a_mhz"], "ec_b_mhz": item["ec_b_mhz"],
        "gate": {key: item[key] for key in ("raw_fidelity", "postselected_fidelity", "time_ns", "fA_GHz", "fB_GHz", "EJ_GHz", "couplings_GHz")},
        "rail_a": a_result, "rail_b": b_result,
        "fixed_interrail_coupling_target_ghz": item["couplings_GHz"]["A2_B1"],
        "reduced_fixed_hardware_pass": passed,
    }
    atomic_json(Path(path), result)
    return result


def _parallel(function, tasks, workers, label):
    results = []
    if not tasks:
        return results
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(function, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                print(json.dumps({"stage": label, "completed": completed, "total": len(tasks)}), flush=True)
    return results


def load_inputs(repetition):
    config = json.loads((repetition / "config.json").read_text())
    ec_a = tuple(round(1000 * value) for value in config["tasks"][0]["ec_GHz"][:2])
    a_rows = json.loads(A_RESULTS.read_text())
    a_row = next((row for row in a_rows if (row["ec1_target_mhz"], row["ec2_target_mhz"]) == ec_a), None)
    if a_row is None or a_row["status"] != "validated_pass":
        raise RuntimeError(f"No validated rail-A readout witness for {ec_a}")
    b_rows = json.loads(B_RESULTS.read_text())
    b_valid = {(row["ec1_target_mhz"], row["ec2_target_mhz"]): row for row in b_rows if row["status"] == "validated_pass"}
    database = repetition / "gate_output/compact_results.sqlite"
    connection = sqlite3.connect(database)
    gate_rows = connection.execute(
        "SELECT ec_b1_mhz,ec_b2_mhz,result_status,raw_fidelity,postselected_fidelity,details_json FROM gate_results"
    ).fetchall()
    connection.close()
    if len(gate_rows) != len(b_rows):
        raise RuntimeError("This reconciler currently requires one logical-frequency sample per ordered rail-B EC point")
    overlay = []
    for b1, b2, status, raw, post, details_json in gate_rows:
        ec_b = (round(b1), round(b2))
        if ec_b not in b_valid or status != "calibrated" or raw is None or post is None or raw <= .95 or post <= .999:
            continue
        details = json.loads(details_json)
        overlay.append({"ec_a_mhz": list(ec_a), "ec_b_mhz": list(ec_b), "raw_fidelity": raw,
                        "postselected_fidelity": post, "time_ns": details["time_ns"],
                        "fA_GHz": details["fA_GHz"], "fB_GHz": details["fB_GHz"],
                        "EJ_GHz": details["EJ_GHz"], "couplings_GHz": details["couplings_GHz"],
                        "readout_b_point": b_valid[ec_b]["point"]})
    return ec_a, a_row, b_rows, overlay


def compact(output, results):
    database = output / "fixed_hardware_results.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE IF EXISTS reconciliation")
    connection.execute("CREATE TABLE reconciliation(ec_b1_mhz INTEGER,ec_b2_mhz INTEGER,reconciled_pass INTEGER,raw_fidelity REAL,postselected_fidelity REAL,failure_reasons TEXT,details_json TEXT,PRIMARY KEY(ec_b1_mhz,ec_b2_mhz))")
    rows = []
    for result in results:
        failures = result["rail_a"]["failures"] + result["rail_b"]["failures"]
        rows.append((*result["ec_b_mhz"], int(result["reduced_fixed_hardware_pass"]),
                     result["gate"]["raw_fidelity"], result["gate"]["postselected_fidelity"],
                     json.dumps(failures, separators=(",", ":")), json.dumps(result, separators=(",", ":"))))
    connection.executemany("INSERT INTO reconciliation VALUES(?,?,?,?,?,?,?)", rows)
    connection.commit(); count = connection.execute("SELECT count(*) FROM reconciliation").fetchone()[0]
    connection.close()
    if count != len(results):
        raise RuntimeError("Compact reconciliation database verification failed")
    return database


def plot_map(output, b_rows, results, ec_a):
    values = {(row["ec1_target_mhz"], row["ec2_target_mhz"]): 0 for row in b_rows}
    for result in results:
        values[tuple(result["ec_b_mhz"])] = 2 if result["reduced_fixed_hardware_pass"] else 1
    colors = ListedColormap(["#dedede", "#f4a261", "#2a9d8f"])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], 3)
    points = sorted(values)
    fig, ax = plt.subplots(figsize=(7.4, 6.4), layout="constrained")
    ax.scatter([point[0] for point in points], [point[1] for point in points],
               c=[values[point] for point in points], s=12, marker="s", cmap=colors, norm=norm)
    ax.set(title=f"Fixed-hardware reconciliation: rail A EC = {ec_a} MHz",
           xlabel=r"$E_{C,B1}/h$ (MHz)", ylabel=r"$E_{C,B2}/h$ (MHz)")
    ax.set_aspect("equal")
    ax.legend(handles=[Patch(color=colors(index), label=label) for index, label in enumerate(
        ("Outside readout/gate overlay", "Overlay: reconciliation failed", "Overlay: reconciliation passed"))], fontsize=8)
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    fig.savefig(plots / "fixed_hardware_reconciliation.png", dpi=220)
    fig.savefig(plots / "fixed_hardware_reconciliation.pdf")
    plt.close(fig)


def reconcile(repetition, workers=4, prune=False):
    ec_a, a_row, b_rows, overlay = load_inputs(repetition)
    output = repetition / "fixed_hardware_reconciliation"
    checkpoints = output / "points"; checkpoints.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "inputs.json", {"ec_a_mhz": ec_a, "overlay_points": len(overlay),
        "gate_thresholds": {"raw_strictly_greater_than": .95, "postselected_strictly_greater_than": .999},
        "scope": "Every point in the independent readout/gate overlay; resonator-inclusive full-network validation excluded."})

    results, pending_items = [], []
    for item in overlay:
        b1, b2 = item["ec_b_mhz"]; path = checkpoints / f"ec_{b1}_{b2}.json"
        if path.exists(): results.append(json.loads(path.read_text()))
        else: pending_items.append((item,path))
    # Rail A is identical for pending points sharing the same gate-required A1-A2 coupling.
    unique_a_targets = sorted({item["couplings_GHz"]["A1_A2"] for item,_ in pending_items})
    a_tasks = [(ec_a, target, a_row["point"], (.14, .16)) for target in unique_a_targets]
    a_cache = dict(_parallel(_reconcile_rail, a_tasks, workers, "rail_a_cache"))
    tasks = [(item,a_cache[item["couplings_GHz"]["A1_A2"]],str(path)) for item,path in pending_items]
    results.extend(_parallel(_reconcile_point, tasks, workers, "overlay_points"))
    results.sort(key=lambda row: tuple(row["ec_b_mhz"]))
    database = compact(output, results)
    plot_map(output, b_rows, results, ec_a)
    from plot_reconciled_metrics import create_plots
    metric_plots = create_plots(repetition)
    failures = Counter(reason for result in results for reason in result["rail_a"]["failures"] + result["rail_b"]["failures"])
    summary = {"ec_a_mhz": ec_a, "overlay_points": len(overlay), "points_checked": len(results),
        "reconciled_pass": sum(row["reduced_fixed_hardware_pass"] for row in results),
        "reconciled_fail": sum(not row["reduced_fixed_hardware_pass"] for row in results),
        "failure_reason_counts": dict(failures), "compact_database": str(database),
        "metric_plots": metric_plots,
        "full_network_validation": "not run; handed off to colleague"}
    atomic_json(output / "summary.json", summary)
    if prune:
        for path in checkpoints.glob("*.json"): path.unlink()
        checkpoints.rmdir()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repetition", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prune-checkpoints", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reconcile(args.repetition.resolve(), args.workers, args.prune_checkpoints), indent=2))


if __name__ == "__main__":
    main()
