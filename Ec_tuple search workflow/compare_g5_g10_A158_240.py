#!/usr/bin/env python3
"""Compare matched corrected-cavity g=5 and g=10 MHz rail-B campaigns."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
G5 = HERE / "g5_corrected_cavity_A158_240_2026-11-17"
G10 = HERE / "g10_corrected_cavity_A158_240_2026-11-17"
B_RESULTS = HERE / "cavity_split_campaign_2026-11-09/readout/direct_ec/rail_b_step2_fr5p95/results.json"
OUT = HERE / "g5_vs_g10_A158_240_comparison_2026-11-17"
SOURCES = ("00", "01", "10", "11")
SOURCE_COLORS = ("#E69F00", "#377EB8", "#2CA25F", "#984EA3")


def load(campaign: Path):
    database = campaign / "gate_output/compact_anchor_results.sqlite"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    saved = connection.execute(
        "SELECT ec_b1_mhz,ec_b2_mhz,status,raw_fidelity,postselected_fidelity,"
        "selected_anchor_id,result_json FROM gate_results"
    ).fetchall()
    connection.close()
    reconciliation = {}
    rec_db = campaign / "fixed_hardware_reconciliation/fixed_hardware_results.sqlite"
    if rec_db.exists():
        connection = sqlite3.connect(f"file:{rec_db}?mode=ro", uri=True)
        reconciliation = {
            (round(a), round(b)): bool(value)
            for a, b, value in connection.execute(
                "SELECT ec_b1_mhz,ec_b2_mhz,reconciled_pass FROM reconciliation"
            )
        }
        connection.close()
    rows = {}
    for b1, b2, status, raw, post, anchor, text in saved:
        result = json.loads(text)
        final = result.get("final") or {}
        channel = result.get("channel_selection") or final.get("channel_selection") or {}
        coupling = (final.get("couplings_GHz") or {}).get("A2_B1")
        key = (round(b1), round(b2))
        raw_value = float(final.get("raw_fidelity", raw or float("nan")))
        post_value = float(final.get("postselected_fidelity", post or float("nan")))
        passed = status == "calibrated" and raw_value > 0.95 and post_value > 0.999
        rows[key] = {
            "status": status,
            "gate_pass": passed,
            "reconciled_pass": reconciliation.get(key),
            "raw_fidelity": raw_value if status == "calibrated" else None,
            "postselected_fidelity": post_value if status == "calibrated" else None,
            "source": result.get("logical_source") or final.get("logical_source"),
            "anchor": anchor,
            "gate_time_ns": final.get("time_ns"),
            "j_req_mhz": None if coupling is None else 1000.0 * float(coupling),
            "aggregate_s1": channel.get("s1", final.get("S1")),
            "maximum_logical_ratio": channel.get("maximum_logical_ratio", final.get("maximum_logical_ratio")),
            "gate_frequency_ghz": final.get("fB_GHz"),
        }
    return rows


def readout_grid():
    rows = json.loads(B_RESULTS.read_text())
    xs = sorted({round(row["ec1_target_mhz"]) for row in rows})
    ys = sorted({round(row["ec2_target_mhz"]) for row in rows})
    valid = np.zeros((len(ys), len(xs)), dtype=bool)
    for row in rows:
        if row.get("status") == "validated_pass":
            valid[ys.index(round(row["ec2_target_mhz"])), xs.index(round(row["ec1_target_mhz"]))] = True
    return xs, ys, valid


def grid(rows, xs, ys, key, predicate=lambda row: True):
    value = np.full((len(ys), len(xs)), np.nan)
    for (x, y), row in rows.items():
        item = row.get(key)
        if item is not None and predicate(row):
            value[ys.index(y), xs.index(x)] = float(item)
    return value


def contour(ax, xs, ys, valid):
    ax.contour(xs, ys, valid.astype(float), levels=[0.5], colors="white", linewidths=2.2)
    ax.contour(xs, ys, valid.astype(float), levels=[0.5], colors="black", linewidths=0.8)


def quantiles(values):
    data = np.asarray(values, dtype=float)
    return {
        "count": len(data), "minimum": float(data.min()), "q05": float(np.quantile(data, .05)),
        "median": float(np.median(data)), "q95": float(np.quantile(data, .95)), "maximum": float(data.max()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g5, g10 = load(G5), load(G10)
    if set(g5) != set(g10):
        raise RuntimeError("The matched campaigns do not contain identical rail-B grids")
    xs, ys, readout_valid = readout_grid()
    keys = sorted(g5)

    pass_category = {}
    rec_category = {}
    for key in keys:
        p5, p10 = g5[key]["gate_pass"], g10[key]["gate_pass"]
        pass_category[key] = (1 if p5 else 0) + (2 if p10 else 0)
        r5, r10 = g5[key]["reconciled_pass"] is True, g10[key]["reconciled_pass"] is True
        rec_category[key] = (1 if r5 else 0) + (2 if r10 else 0)

    category_colors = ("#D9D9D9", "#E76F51", "#4575B4", "#2A9D8F")
    category_labels = ("neither", "5 MHz only", "10 MHz only", "both")
    cmap = ListedColormap(category_colors)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)
    extent = [min(xs)-1, max(xs)+1, min(ys)-1, max(ys)+1]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 12.0), layout="constrained", sharex=True, sharey=True)
    for ax, categories, title in (
        (axes[0, 0], pass_category, "Gate-fidelity feasibility"),
        (axes[0, 1], rec_category, "Fixed-hardware reconciled feasibility"),
    ):
        data = np.zeros((len(ys), len(xs)), dtype=int)
        for (x, y), value in categories.items():
            data[ys.index(y), xs.index(x)] = value
        ax.imshow(data, origin="lower", interpolation="nearest", extent=extent, aspect="equal", cmap=cmap, norm=norm)
        contour(ax, xs, ys, readout_valid)
        ax.set_title(title)

    source_cmap = ListedColormap(SOURCE_COLORS)
    source_norm = BoundaryNorm(np.arange(-0.5, 4.5), source_cmap.N)
    for ax, rows, title in (
        (axes[1, 0], g5, "Selected CZ source: g/h=5 MHz"),
        (axes[1, 1], g10, "Selected CZ source: g/h=10 MHz"),
    ):
        for source_index, source in enumerate(SOURCES):
            points = [key for key, row in rows.items() if row["gate_pass"] and row["source"] == source]
            ax.scatter([x for x, _ in points], [y for _, y in points], s=10, marker="s",
                       color=SOURCE_COLORS[source_index], linewidths=0)
        contour(ax, xs, ys, readout_valid)
        ax.set_title(title)
    for ax in axes.flat:
        ax.set_xlabel(r"$E_{C,B1}/h$ (MHz)")
        ax.set_ylabel(r"$E_{C,B2}/h$ (MHz)")
        ax.set_aspect("equal")
    fig.legend(handles=[Patch(color=color, label=label) for color, label in zip(category_colors, category_labels)] +
               [Patch(color=SOURCE_COLORS[i], label=f"source {source}") for i, source in enumerate(SOURCES)],
               loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Effect of doubling intended CZ coupling at fixed EC-A=(158,240) MHz", fontsize=14)
    fig.savefig(OUT / "feasibility_and_cz_source_comparison.png", dpi=220)
    fig.savefig(OUT / "feasibility_and_cz_source_comparison.pdf")
    plt.close(fig)

    metrics = (
        ("gate_time_ns", "Calibrated gate time (ns)", "viridis_r", False),
        ("j_req_mhz", r"Required physical $J_{req}/h$ (MHz)", "viridis", False),
        ("raw_fidelity", "Raw CZ fidelity", "viridis", True),
        ("postselected_fidelity", "Post-selected CZ fidelity", "viridis", True),
        ("aggregate_s1", r"Aggregate $S_1$", "magma", False),
        ("maximum_logical_ratio", r"Maximum logical $|g/\Delta|$", "magma", False),
        ("gate_frequency_ghz", "Rail-B gate frequency (GHz)", "viridis", False),
    )
    common = [key for key in keys if g5[key]["gate_pass"] and g10[key]["gate_pass"]]
    fig, axes = plt.subplots(len(metrics), 3, figsize=(15.5, 4.2*len(metrics)), layout="constrained")
    metric_summary = {}
    for row_index, (key, label, cmap_name, fidelity) in enumerate(metrics):
        values5 = np.asarray([g5[p][key] for p in common], dtype=float)
        values10 = np.asarray([g10[p][key] for p in common], dtype=float)
        if fidelity:
            values5 *= 100.0
            values10 *= 100.0
            label += " (%)"
        shared_min = float(min(values5.min(), values10.min()))
        shared_max = float(max(values5.max(), values10.max()))
        for column, (values, title) in enumerate(((values5, "g/h=5 MHz"), (values10, "g/h=10 MHz"))):
            data = np.full((len(ys), len(xs)), np.nan)
            for point, value in zip(common, values):
                data[ys.index(point[1]), xs.index(point[0])] = value
            image = axes[row_index, column].imshow(data, origin="lower", interpolation="nearest", extent=extent,
                                                   aspect="equal", cmap=cmap_name, vmin=shared_min, vmax=shared_max)
            contour(axes[row_index, column], xs, ys, readout_valid)
            axes[row_index, column].set_title(title + "\n" + label)
            fig.colorbar(image, ax=axes[row_index, column], shrink=.76)
        delta = values10 - values5
        data = np.full((len(ys), len(xs)), np.nan)
        for point, value in zip(common, delta):
            data[ys.index(point[1]), xs.index(point[0])] = value
        limit = float(max(abs(delta.min()), abs(delta.max()))) or 1.0
        image = axes[row_index, 2].imshow(data, origin="lower", interpolation="nearest", extent=extent,
                                         aspect="equal", cmap="coolwarm", norm=TwoSlopeNorm(0, -limit, limit))
        contour(axes[row_index, 2], xs, ys, readout_valid)
        axes[row_index, 2].set_title("10 MHz minus 5 MHz\n" + label)
        fig.colorbar(image, ax=axes[row_index, 2], shrink=.76)
        metric_summary[key] = {
            "g5": quantiles(values5), "g10": quantiles(values10), "g10_minus_g5": quantiles(delta),
        }
    for ax in axes.flat:
        ax.set_xlabel(r"$E_{C,B1}/h$ (MHz)")
        ax.set_ylabel(r"$E_{C,B2}/h$ (MHz)")
    fig.suptitle("Pointwise metric changes on points passing at both target couplings", fontsize=14)
    fig.savefig(OUT / "pointwise_metric_comparison.png", dpi=180)
    fig.savefig(OUT / "pointwise_metric_comparison.pdf")
    plt.close(fig)

    source_changed = sum(g5[p]["source"] != g10[p]["source"] for p in common)
    anchor_changed = sum(g5[p]["anchor"] != g10[p]["anchor"] for p in common)
    output = {
        "schema": "matched-g5-g10-fixed-a-comparison-v1",
        "ec_a_mhz": [158, 240],
        "cavity_provenance_ghz": {"rail_a": 6.05, "rail_b": 5.95},
        "grid_point_count": len(keys),
        "gate_pass_counts": Counter(category_labels[value] for value in pass_category.values()),
        "reconciled_pass_counts": Counter(category_labels[value] for value in rec_category.values()),
        "common_gate_pass_count": len(common),
        "source_changed_on_common_pass_count": source_changed,
        "anchor_changed_on_common_pass_count": anchor_changed,
        "g5_source_counts": Counter(row["source"] for row in g5.values() if row["gate_pass"]),
        "g10_source_counts": Counter(row["source"] for row in g10.values() if row["gate_pass"]),
        "metrics_on_common_gate_pass_points": metric_summary,
        "figures": ["feasibility_and_cz_source_comparison.png", "pointwise_metric_comparison.png"],
    }
    (OUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
