"""Create fidelity maps and the two separate count traces from checkpoints."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_stage(config_path: str | Path) -> dict:
    """Plot one fixed-A2 grid; never recompute or recalibrate its checkpoints."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = (config_path.parent / config["output_directory"]).resolve()
    tag = str(config["grid"]["B_MHz"]["step"]).replace(".", "p")
    plots = out / f"plots_{tag}MHz"
    plots.mkdir(exist_ok=True)
    records = [json.loads(p.read_text()) for p in (out / "points").glob("*.json")]
    points = {}
    for record in records:
        ec = record["payload"]["ec_GHz"]
        points[tuple(round(v * 1000, 7) for v in ec)] = record
    a1_values = sorted({key[0] for key in points})
    counts = []
    table = []
    for a1 in a1_values:
        subset = {key: value for key, value in points.items() if key[0] == a1}
        a2 = next(iter(subset))[1]
        b_values = sorted({key[2] for key in subset} | {key[3] for key in subset})
        index = {value: i for i, value in enumerate(b_values)}
        raw = np.full((len(b_values), len(b_values)), np.nan)
        conditional = raw.copy()
        boundary = []
        n_raw = n_post = n_calibrated = 0
        for key, record in subset.items():
            result = record.get("result", {})
            row = {
                "EC_A1_MHz": key[0],
                "EC_A2_MHz": key[1],
                "EC_B1_MHz": key[2],
                "EC_B2_MHz": key[3],
                "status": result.get("status", record["status"]),
            }
            if result.get("status") == "calibrated":
                final = result["final"]
                r, c = final["raw_fidelity"], final["postselected_fidelity"]
                raw[index[key[3]], index[key[2]]] = r * 100
                conditional[index[key[3]], index[key[2]]] = c * 100
                n_raw += r > 0.90
                n_post += c > 0.99
                n_calibrated += 1
                row.update(
                    raw_fidelity=r,
                    postselected_fidelity=c,
                    acceptance=final["acceptance"],
                    gate_time_ns=final["time_ns"],
                    S1=final["S1"],
                    S2=final["S2"],
                    frequency_boundary=result["frequency_boundary"],
                )
                if result["frequency_boundary"]:
                    boundary.append((key[2], key[3]))
            table.append(row)
        for values, name in [(raw, "raw"), (conditional, "postselected")]:
            fig, ax = plt.subplots(figsize=(7, 6))
            color = plt.get_cmap("viridis").copy()
            color.set_bad("#dedede")
            # Nearest cells only: missing upper-triangle samples remain visibly masked.
            mesh = ax.pcolormesh(
                b_values,
                b_values,
                np.ma.masked_invalid(values),
                shading="nearest",
                cmap=color,
                vmin=90 if name == "raw" else 99,
                vmax=100,
            )
            fig.colorbar(mesh, ax=ax, label=f"{name.capitalize()} CZ fidelity (%)")
            if boundary:
                x, y = zip(*boundary, strict=True)
                ax.scatter(
                    x, y, s=5, c="white", marker=".", label="Frequency-boundary optimum"
                )
                ax.legend(fontsize=7, loc="lower right")
            ax.set(
                xlabel="EC-B1/h (MHz)",
                ylabel="EC-B2/h (MHz)",
                title=f"EC-A2={a2:g}, EC-A1={a1:g} MHz; {name} fidelity",
            )
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(plots / f"A1_{a1:g}_{name}.png", dpi=160)
            plt.close(fig)
        # Keep both operating frequencies alongside the two fidelity maps.
        fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
        quantities = [
            ("raw_fidelity", "Raw CZ fidelity (%)", 100.0, 95.0, 100.0),
            (
                "postselected_fidelity",
                "Postselected CZ fidelity (%)",
                100.0,
                99.0,
                100.0,
            ),
            ("fA_GHz", "A common bare f01 (GHz)", 1.0, None, None),
            ("fB_GHz", "B common bare f01 after calibration (GHz)", 1.0, None, None),
        ]
        for ax, (name, title, scale, low, high) in zip(
            axes.flat, quantities, strict=True
        ):
            values = np.full_like(raw, np.nan)
            for key, record in subset.items():
                result = record.get("result", {})
                if result.get("status") != "calibrated":
                    continue
                value = result["fA_GHz"] if name == "fA_GHz" else result["final"][name]
                values[index[key[3]], index[key[2]]] = scale * value
            color = plt.get_cmap("inferno" if "fidelity" in name else "viridis").copy()
            color.set_bad("#bbbbbb")
            mesh = ax.pcolormesh(
                b_values,
                b_values,
                np.ma.masked_invalid(values),
                shading="nearest",
                cmap=color,
                vmin=low,
                vmax=high,
            )
            fig.colorbar(
                mesh, ax=ax, extend="both" if "fidelity" in name else "neither"
            )
            ax.set(title=title, xlabel="EC-B1/h (MHz)", ylabel="EC-B2/h (MHz)")
            ax.set_aspect("equal")
        fig.suptitle(f"EC-A2={a2:g}, EC-A1={a1:g} MHz | grey: no calibrated result")
        fig.savefig(plots / f"A1_{a1:g}_four_panel.png", dpi=160)
        fig.savefig(plots / f"A1_{a1:g}_four_panel.pdf")
        plt.close(fig)
        counts.append(
            {
                "EC_A1_MHz": a1,
                "sampled_points": len(subset),
                "calibrated": n_calibrated,
                "raw_above_90_percent": int(n_raw),
                "postselected_above_99_percent": int(n_post),
            }
        )
    for key, name, title in [
        ("raw_above_90_percent", "raw_counts", "Raw fidelity > 90%"),
        (
            "postselected_above_99_percent",
            "postselected_counts",
            "Postselected fidelity > 99%",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(
            [v["EC_A1_MHz"] for v in counts],
            [v[key] for v in counts],
            width=config["grid"]["A1_MHz"]["step"],
            edgecolor="white",
        )
        ax.set(
            xlabel="EC-A1/h (MHz)",
            ylabel="Number of sampled B-plane points",
            title=title,
        )
        fig.tight_layout()
        fig.savefig(plots / f"{name}.png", dpi=160)
        plt.close(fig)
    (plots / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    fields = sorted({key for row in table for key in row})
    with (plots / "point_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    return {
        "plots": str(plots),
        "points": len(table),
        "calibrated": sum(c["calibrated"] for c in counts),
    }
