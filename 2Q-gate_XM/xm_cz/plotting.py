"""Create fidelity maps and the two separate count traces from checkpoints."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIDELITY_LIMITS = {
    "raw_fidelity": (95.0, 100.0),
    "postselected_fidelity": (99.9, 99.99),
}


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
        splits = record["payload"].get("logical_splittings_GHz", [0.150, 0.090])
        key = tuple(round(v * 1000, 7) for v in ec)
        points[(*key, *[round(v, 9) for v in splits])] = record
    logical_axis_name = next(
        (
            name
            for name in ("logical_A_GHz", "logical_B_GHz")
            if name in config["grid"]
        ),
        None,
    )
    logical_grid = logical_axis_name is not None
    logical_index = 4 if logical_axis_name == "logical_A_GHz" else 5
    logical_label = "A" if logical_index == 4 else "B"
    group_values = sorted(
        {key[logical_index] if logical_grid else key[0] for key in points}
    )
    common_calibrated = None
    if logical_grid:
        calibrated_by_group = []
        for group_value in group_values:
            calibrated_by_group.append(
                {
                    (key[2], key[3])
                    for key, record in points.items()
                    if key[logical_index] == group_value
                    and record.get("result", {}).get("status") == "calibrated"
                }
            )
        common_calibrated = set.intersection(*calibrated_by_group)
    counts = []
    table = []
    for group_value in group_values:
        subset = {
            key: value
            for key, value in points.items()
            if (key[logical_index] if logical_grid else key[0]) == group_value
        }
        a1 = next(iter(subset))[0]
        a2 = next(iter(subset))[1]
        logical_value = next(iter(subset))[logical_index]
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
                "logical_A_GHz": key[4],
                "logical_B_GHz": key[5],
                "status": result.get("status", record["status"]),
            }
            if result.get("status") == "calibrated":
                final = result["final"]
                r, c = final["raw_fidelity"], final["postselected_fidelity"]
                if common_calibrated is None or (key[2], key[3]) in common_calibrated:
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
                if result["frequency_boundary"] and (
                    common_calibrated is None
                    or (key[2], key[3]) in common_calibrated
                ):
                    boundary.append((key[2], key[3]))
            table.append(row)
        for values, name in [(raw, "raw"), (conditional, "postselected")]:
            metric = f"{name}_fidelity"
            low, high = FIDELITY_LIMITS[metric]
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
                vmin=low,
                vmax=high,
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
                title=(
                    f"EC-A2={a2:g}, EC-A1={a1:g} MHz; "
                    f"logical {logical_label}={logical_value:g} GHz; {name} fidelity"
                ),
            )
            ax.set_aspect("equal")
            fig.tight_layout()
            prefix = (
                f"logical_{logical_label}_{logical_value:g}"
                if logical_grid
                else f"A1_{a1:g}"
            )
            fig.savefig(plots / f"{prefix}_{name}.png", dpi=160)
            plt.close(fig)
        # Keep both operating frequencies alongside the two fidelity maps.
        fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
        quantities = [
            (
                "raw_fidelity",
                "Raw CZ fidelity (%)",
                100.0,
                *FIDELITY_LIMITS["raw_fidelity"],
            ),
            (
                "postselected_fidelity",
                "Postselected CZ fidelity (%)",
                100.0,
                *FIDELITY_LIMITS["postselected_fidelity"],
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
                if common_calibrated is not None and (
                    key[2],
                    key[3],
                ) not in common_calibrated:
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
        fig.suptitle(
            f"EC-A2={a2:g}, EC-A1={a1:g} MHz, "
            f"logical {logical_label}={logical_value:g} GHz"
            " | grey: unavailable at any swept logical frequency"
        )
        prefix = (
            f"logical_{logical_label}_{logical_value:g}"
            if logical_grid
            else f"A1_{a1:g}"
        )
        fig.savefig(plots / f"{prefix}_four_panel.png", dpi=160)
        fig.savefig(plots / f"{prefix}_four_panel.pdf")
        plt.close(fig)
        counts.append(
            {
                "EC_A1_MHz": a1,
                f"logical_{logical_label}_GHz": logical_value,
                "sampled_points": len(subset),
                "calibrated": n_calibrated,
                "raw_above_90_percent": int(n_raw),
                "postselected_above_99_percent": int(n_post),
            }
        )
    if logical_grid and group_values:
        b_values = sorted({key[2] for key in points} | {key[3] for key in points})
        index = {value: i for i, value in enumerate(b_values)}
        cubes = {}
        for metric in ("raw_fidelity", "postselected_fidelity"):
            cube = np.full(
                (len(group_values), len(b_values), len(b_values)), np.nan
            )
            for key, record in points.items():
                result = record.get("result", {})
                if result.get("status") == "calibrated" and (
                    key[2],
                    key[3],
                ) in common_calibrated:
                    cube[
                        group_values.index(key[logical_index]),
                        index[key[3]],
                        index[key[2]],
                    ] = 100.0 * result["final"][metric]
            cubes[metric] = cube

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
        for ax, (metric, label) in zip(
            axes,
            [
                ("raw_fidelity", "Minimum raw CZ fidelity"),
                (
                    "postselected_fidelity",
                    "Minimum postselected CZ fidelity",
                ),
            ],
            strict=True,
        ):
            low, high = FIDELITY_LIMITS[metric]
            minimum = np.ma.masked_invalid(cubes[metric]).min(axis=0).filled(np.nan)
            color = plt.get_cmap("viridis").copy()
            color.set_bad("#dedede")
            mesh = ax.pcolormesh(
                b_values,
                b_values,
                np.ma.masked_invalid(minimum),
                shading="nearest",
                cmap=color,
                vmin=low,
                vmax=high,
            )
            fig.colorbar(mesh, ax=ax, label=f"{label} (%)")
            ax.set(
                title=f"{label} across logical {logical_label}",
                xlabel="EC-B1/h (MHz)",
                ylabel="EC-B2/h (MHz)",
            )
            ax.set_aspect("equal")
        fig.suptitle(
            f"Worst-case fidelity for logical {logical_label} = "
            f"{min(group_values):.3f}-{max(group_values):.3f} GHz; "
            "grey is the union of unavailable points"
        )
        fig.savefig(plots / f"logical_{logical_label}_minimum_fidelity.png", dpi=180)
        fig.savefig(plots / f"logical_{logical_label}_minimum_fidelity.pdf")
        plt.close(fig)

        for metric, label in [
            ("raw_fidelity", "Raw CZ fidelity"),
            ("postselected_fidelity", "Postselected CZ fidelity"),
        ]:
            low, high = FIDELITY_LIMITS[metric]
            fig, axes = plt.subplots(
                2, 3, figsize=(16, 10), sharex=True, sharey=True, layout="constrained"
            )
            color = plt.get_cmap("viridis").copy()
            color.set_bad("#dedede")
            mesh = None
            for ax, logical_value, values in zip(
                axes.flat, group_values, cubes[metric], strict=False
            ):
                mesh = ax.pcolormesh(
                    b_values,
                    b_values,
                    np.ma.masked_invalid(values),
                    shading="nearest",
                    cmap=color,
                    vmin=low,
                    vmax=high,
                )
                ax.set_title(f"logical {logical_label} = {logical_value:g} GHz")
                ax.set_aspect("equal")
            for ax in axes[-1, :]:
                ax.set_xlabel("EC-B1/h (MHz)")
            for ax in axes[:, 0]:
                ax.set_ylabel("EC-B2/h (MHz)")
            for ax in axes.flat[len(group_values) :]:
                ax.set_visible(False)
            fig.colorbar(mesh, ax=axes, label=f"{label} (%)")
            fig.suptitle(
                f"{label} versus logical {logical_label} splitting; "
                f"EC-A1={a1:g}, EC-A2={a2:g} MHz"
            )
            fig.savefig(
                plots / f"logical_{logical_label}_{metric}_comparison.png", dpi=180
            )
            fig.savefig(plots / f"logical_{logical_label}_{metric}_comparison.pdf")
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
        logical_axis = np.asarray(group_values)
        for row, (metric, label) in enumerate(
            [
                ("raw_fidelity", "Raw fidelity"),
                ("postselected_fidelity", "Postselected fidelity"),
            ]
        ):
            cube = cubes[metric]
            complete = np.all(np.isfinite(cube), axis=0)
            fidelity_range = np.full(cube.shape[1:], np.nan)
            fidelity_range[complete] = np.ptp(cube[:, complete], axis=0)
            centered = logical_axis - logical_axis.mean()
            slope = np.full(cube.shape[1:], np.nan)
            slope[complete] = np.sum(
                centered[:, None] * cube[:, complete], axis=0
            ) / np.sum(centered**2)
            for ax, values, title, cmap in [
                (
                    axes[row, 0],
                    fidelity_range,
                    f"{label} range (percentage points)",
                    "magma",
                ),
                (
                    axes[row, 1],
                    slope / 1000.0,
                    f"{label} slope (percentage points/MHz)",
                    "coolwarm",
                ),
            ]:
                limit = np.nanmax(np.abs(values)) if "slope" in title else None
                mesh = ax.pcolormesh(
                    b_values,
                    b_values,
                    np.ma.masked_invalid(values),
                    shading="nearest",
                    cmap=cmap,
                    vmin=-limit if limit else 0.0,
                    vmax=limit,
                )
                fig.colorbar(mesh, ax=ax)
                ax.set(
                    title=title,
                    xlabel="EC-B1/h (MHz)",
                    ylabel="EC-B2/h (MHz)",
                )
                ax.set_aspect("equal")
        fig.suptitle(
            f"Sensitivity across logical {logical_label} = "
            f"{min(group_values):.3f}-{max(group_values):.3f} GHz; "
            "grey requires a calibrated result at all five values"
        )
        fig.savefig(plots / f"logical_{logical_label}_sensitivity.png", dpi=180)
        fig.savefig(plots / f"logical_{logical_label}_sensitivity.pdf")
        plt.close(fig)
    for key, name, title in [
        ("raw_above_90_percent", "raw_counts", "Raw fidelity > 90%"),
        (
            "postselected_above_99_percent",
            "postselected_counts",
            "Postselected fidelity > 99%",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x_key = logical_axis_name if logical_grid else "EC_A1_MHz"
        axis_key = logical_axis_name if logical_grid else "A1_MHz"
        ax.bar(
            [v[x_key] for v in counts],
            [v[key] for v in counts],
            width=config["grid"][axis_key]["step"],
            edgecolor="white",
        )
        ax.set(
            xlabel=(
                f"Logical {logical_label} splitting (GHz)"
                if logical_grid
                else "EC-A1/h (MHz)"
            ),
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
