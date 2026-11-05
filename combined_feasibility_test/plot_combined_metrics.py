"""Plot gate fidelity and idle-to-gate EJ shifts on the combined feasible map."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent))
from readout_workflow.storage import atomic_json


def gate_records() -> dict[tuple[int, int], dict]:
    records = {}
    for path in (DATA / "gate_central_output" / "points").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        ec = tuple(round(1000 * value) for value in record["payload"]["ec_GHz"][2:])
        records[ec] = record
    return records


def grid_axes(rows):
    source = json.loads((DATA / "rail_b_readout_map_official" / "results.json").read_text())
    xs = sorted({row["ec1_target_mhz"] for row in source})
    ys = sorted({row["ec2_target_mhz"] for row in source})
    return xs, ys, {x: i for i, x in enumerate(xs)}, {y: i for i, y in enumerate(ys)}


def array(rows, field, xs, ys, xi, yi):
    values = np.full((len(ys), len(xs)), np.nan)
    for row in rows:
        x, y = row["ec_b_mhz"]
        values[yi[y], xi[x]] = row[field]
    return values


def plot_pair(rows, fields, titles, filename, colorbar_label, *, percent=False, diverging=False):
    xs, ys, xi, yi = grid_axes(rows)
    grids = [array(rows, field, xs, ys, xi, yi) for field in fields]
    finite = np.concatenate([grid[np.isfinite(grid)] for grid in grids])
    if percent:
        grids = [100 * grid for grid in grids]
        finite = 100 * finite
    if diverging:
        limit = max(abs(float(np.min(finite))), abs(float(np.max(finite))))
        vmin, vmax, cmap = -limit, limit, "coolwarm"
    else:
        vmin, vmax, cmap = float(np.min(finite)), float(np.max(finite)), "viridis"
        if vmin == vmax:
            vmin -= 1e-9
            vmax += 1e-9
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.7), constrained_layout=True)
    extent = [min(xs) - 2.5, max(xs) + 2.5, min(ys) - 2.5, max(ys) + 2.5]
    for axis, grid, title in zip(axes, grids, titles, strict=True):
        image = axis.imshow(
            grid,
            origin="lower",
            interpolation="nearest",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        local = grid[np.isfinite(grid)]
        axis.set(
            xlabel=r"$E_{C,B1}$ (MHz)",
            ylabel=r"$E_{C,B2}$ (MHz)",
            title=f"{title}\nrange {np.min(local):.5g} to {np.max(local):.5g}",
            xticks=xs,
            yticks=ys,
        )
        axis.tick_params(axis="x", rotation=45)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.9)
        colorbar.set_label(colorbar_label)
    output = DATA / "plots"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{filename}.png", dpi=220)
    figure.savefig(output / f"{filename}.pdf")
    plt.close(figure)


def main() -> int:
    reconciliation = json.loads((DATA / "reconciliation" / "results.json").read_text())
    gates = gate_records()
    rows = []
    for result in reconciliation:
        if not result["reduced_fixed_hardware_pass"]:
            continue
        ec = tuple(result["ec_b_mhz"])
        final = gates[ec]["result"]["final"]
        gate_ej = final["EJ_GHz"]
        idle_a = [result["rail_a"]["point"]["ej1_ghz"], result["rail_a"]["point"]["ej2_ghz"]]
        idle_b = [result["rail_b"]["point"]["ej1_ghz"], result["rail_b"]["point"]["ej2_ghz"]]
        rows.append(
            {
                "ec_b_mhz": ec,
                "raw_fidelity": final["raw_fidelity"],
                "postselected_fidelity": final["postselected_fidelity"],
                "idle_ej_ghz": {"A1": idle_a[0], "A2": idle_a[1], "B1": idle_b[0], "B2": idle_b[1]},
                "gate_ej_ghz": {"A1": gate_ej[0], "A2": gate_ej[1], "B1": gate_ej[2], "B2": gate_ej[3]},
                "delta_ej_ghz": {
                    "A1": gate_ej[0] - idle_a[0],
                    "A2": gate_ej[1] - idle_a[1],
                    "B1": gate_ej[2] - idle_b[0],
                    "B2": gate_ej[3] - idle_b[1],
                },
            }
        )
    for row in rows:
        row.update({f"delta_{name}_ghz": value for name, value in row["delta_ej_ghz"].items()})
    atomic_json(DATA / "combined_operating_point_ej_shifts.json", rows)
    plot_pair(
        rows,
        ("raw_fidelity", "postselected_fidelity"),
        ("Raw CZ fidelity", "Post-selected CZ fidelity"),
        "combined_gate_fidelity",
        "CZ fidelity (%)",
        percent=True,
    )
    plot_pair(
        rows,
        ("delta_A1_ghz", "delta_A2_ghz"),
        (r"Rail A: $\Delta E_{J,A1}$", r"Rail A: $\Delta E_{J,A2}$"),
        "rail_a_ej_shifts",
        r"$E_{J,gate}-E_{J,idle}$ (GHz)",
        diverging=True,
    )
    plot_pair(
        rows,
        ("delta_B1_ghz", "delta_B2_ghz"),
        (r"Rail B: $\Delta E_{J,B1}$", r"Rail B: $\Delta E_{J,B2}$"),
        "rail_b_ej_shifts",
        r"$E_{J,gate}-E_{J,idle}$ (GHz)",
        diverging=True,
    )
    print(json.dumps({"combined_feasible_points": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
