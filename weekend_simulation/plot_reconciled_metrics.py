#!/usr/bin/env python3
"""Plot fidelity and idle-to-gate EJ shifts on a reconciled overlay."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
B_RESULTS = HERE / "readout/direct_ec/rail_b_step2/results.json"


def load_rows(repetition):
    database = repetition / "fixed_hardware_reconciliation/fixed_hardware_results.sqlite"
    connection = sqlite3.connect(database)
    records = connection.execute("SELECT details_json FROM reconciliation WHERE reconciled_pass=1").fetchall()
    connection.close()
    gate_connection = sqlite3.connect(repetition / "gate_output/compact_results.sqlite")
    gate_frequencies = {(round(b1),round(b2)):(json.loads(details)["fA_GHz"],json.loads(details)["fB_GHz"])
                        for b1,b2,details in gate_connection.execute("SELECT ec_b1_mhz,ec_b2_mhz,details_json FROM gate_results")}
    gate_connection.close()
    rows = []
    for (details_json,) in records:
        result = json.loads(details_json); gate_ej = result["gate"]["EJ_GHz"]
        gate_frequency_a, gate_frequency_b = gate_frequencies[tuple(result["ec_b_mhz"])]
        idle = [result["rail_a"]["point"]["ej1_ghz"], result["rail_a"]["point"]["ej2_ghz"],
                result["rail_b"]["point"]["ej1_ghz"], result["rail_b"]["point"]["ej2_ghz"]]
        rows.append({"ec_b_mhz": result["ec_b_mhz"], "raw_fidelity": result["gate"]["raw_fidelity"],
                     "postselected_fidelity": result["gate"]["postselected_fidelity"],
                     "idle_ej_ghz": dict(zip(("A1","A2","B1","B2"), idle)),
                     "gate_ej_ghz": dict(zip(("A1","A2","B1","B2"), gate_ej)),
                     "idle_frequency_a_ghz": result["rail_a"]["point"]["operating_frequency_ghz"],
                     "gate_frequency_a_ghz": result["gate"].get("fA_GHz",gate_frequency_a),
                     "idle_frequency_b_ghz": result["rail_b"]["point"]["operating_frequency_ghz"],
                     "gate_frequency_b_ghz": result["gate"].get("fB_GHz",gate_frequency_b),
                     **{f"delta_{name}_ghz": gate_ej[index] - idle[index]
                        for index, name in enumerate(("A1","A2","B1","B2"))}})
    return rows


def plot_pair(rows, fields, titles, output, filename, label, percent=False, diverging=False, panel_limits=None, shared_limit=None):
    arrays = [[row[field] * (100 if percent else 1) for row in rows] for field in fields]
    finite = np.concatenate(arrays)
    if shared_limit is not None:
        vmin, vmax, cmap = -shared_limit, shared_limit, "coolwarm"
    elif diverging:
        limit = max(abs(float(finite.min())), abs(float(finite.max()))); vmin, vmax, cmap = -limit, limit, "coolwarm"
    else:
        vmin, vmax, cmap = float(finite.min()), float(finite.max()), "viridis"
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.7), layout="constrained")
    for index, (axis, field, title, values) in enumerate(zip(axes, fields, titles, arrays, strict=True)):
        local_vmin, local_vmax = panel_limits[index] if panel_limits else (vmin, vmax)
        image = axis.scatter([row["ec_b_mhz"][0] for row in rows], [row["ec_b_mhz"][1] for row in rows],
                             c=values, marker="s", s=14, cmap=cmap, vmin=local_vmin, vmax=local_vmax)
        axis.set(title=f"{title}\nrange {min(values):.6g} to {max(values):.6g}",
                 xlabel=r"$E_{C,B1}/h$ (MHz)", ylabel=r"$E_{C,B2}/h$ (MHz)")
        axis.set_aspect("equal"); figure.colorbar(image, ax=axis, shrink=.9, label=label)
    figure.savefig(output / f"{filename}.png", dpi=220)
    figure.savefig(output / f"{filename}.pdf")
    plt.close(figure)


def create_plots(repetition):
    rows = load_rows(repetition)
    output = repetition / "fixed_hardware_reconciliation/plots"; output.mkdir(parents=True, exist_ok=True)
    data = repetition / "fixed_hardware_reconciliation/operating_point_ej_shifts.json"
    data.write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        filenames = ("reconciled_gate_fidelity", "reconciled_rail_a_ej_shifts", "reconciled_rail_b_ej_shifts",
                     "reconciled_rail_a_operating_frequencies", "reconciled_rail_b_operating_frequencies")
        for filename in filenames:
            figure, axis = plt.subplots(figsize=(8,5), layout="constrained")
            axis.text(.5,.5,"No fixed-hardware reconciliation-passing points",ha="center",va="center",fontsize=14)
            axis.set_axis_off(); figure.savefig(output/f"{filename}.png",dpi=220); figure.savefig(output/f"{filename}.pdf"); plt.close(figure)
        summary = {"reconciled_points_plotted": 0, "empty_final_region": True,
                   "ej_shift_definition": "gate minus idle", "postselected_fidelity_color_scale_percent": [99.9,99.99],
                   "operating_frequency_color_scale_ghz": [4,6], "data": str(data),
                   "plots": sorted(path.name for path in output.iterdir())}
        (repetition / "fixed_hardware_reconciliation/metric_plot_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        return summary
    all_ej_shifts = [row[f"delta_{name}_ghz"] for row in rows for name in ("A1","A2","B1","B2")]
    shared_ej_limit = max(abs(min(all_ej_shifts)), abs(max(all_ej_shifts)))
    plot_pair(rows, ("raw_fidelity","postselected_fidelity"), ("Raw CZ fidelity","Post-selected CZ fidelity"),
              output, "reconciled_gate_fidelity", "CZ fidelity (%)", percent=True,
              panel_limits=((100*min(row["raw_fidelity"] for row in rows),100*max(row["raw_fidelity"] for row in rows)),(99.9,99.99)))
    plot_pair(rows, ("delta_A1_ghz","delta_A2_ghz"),
              (r"Rail A: $\Delta E_{J,A1}$",r"Rail A: $\Delta E_{J,A2}$"), output,
              "reconciled_rail_a_ej_shifts", r"$E_{J,gate}-E_{J,idle}$ (GHz)", shared_limit=shared_ej_limit)
    plot_pair(rows, ("delta_B1_ghz","delta_B2_ghz"),
              (r"Rail B: $\Delta E_{J,B1}$",r"Rail B: $\Delta E_{J,B2}$"), output,
              "reconciled_rail_b_ej_shifts", r"$E_{J,gate}-E_{J,idle}$ (GHz)", shared_limit=shared_ej_limit)
    plot_pair(rows, ("idle_frequency_a_ghz","gate_frequency_a_ghz"),
              ("Rail A: idle operating frequency","Rail A: gate operating frequency"), output,
              "reconciled_rail_a_operating_frequencies", "Operating frequency (GHz)", panel_limits=((4,6),(4,6)))
    plot_pair(rows, ("idle_frequency_b_ghz","gate_frequency_b_ghz"),
              ("Rail B: idle operating frequency","Rail B: gate operating frequency"), output,
              "reconciled_rail_b_operating_frequencies", "Operating frequency (GHz)", panel_limits=((4,6),(4,6)))
    summary = {"reconciled_points_plotted": len(rows), "ej_shift_definition": "gate minus idle",
               "postselected_fidelity_color_scale_percent": [99.9,99.99],
               "shared_ej_shift_color_scale_ghz": [-shared_ej_limit,shared_ej_limit],
               "operating_frequency_color_scale_ghz": [4,6],
               "data": str(data), "plots": sorted(path.name for path in output.iterdir())}
    (repetition / "fixed_hardware_reconciliation/metric_plot_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("repetition",type=Path);args=parser.parse_args()
    print(json.dumps(create_plots(args.repetition.resolve()),indent=2))


if __name__ == "__main__": main()
