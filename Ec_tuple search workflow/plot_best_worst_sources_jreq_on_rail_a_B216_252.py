#!/usr/bin/env python3
"""Plot CZ source and J_req on rail-A grid at fixed B=(216,252)."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
BEST_SELECTIONS = HERE / "combined_overlay_selection_2026-11-12/combined_overlay_best_configurations.json"
WORST_STATUS = HERE / "worst_case_fractional_anchor_corrected_cavities_2026-11-13/campaign_status.json"
A_READOUT = (
    HERE / "cavity_split_campaign_2026-11-09/readout/direct_ec"
    / "rail_a_step2_fr6p05/results.json"
)
OUTPUT = (
    HERE / "worst_case_fractional_anchor_corrected_cavities_2026-11-13"
    / "campaign_summary/B216_252_best_vs_worst_on_rail_a"
)
FIXED_B = (216, 252)
SELECTED_A = (154, 208)
SOURCES = ("00", "01", "10", "11")
COLORS = {"00": "#e69f00", "01": "#377eb8", "10": "#2ca25f", "11": "#984ea3"}


def readout_grid():
    rows = json.loads(A_READOUT.read_text())
    xs = sorted({round(row["ec1_target_mhz"]) for row in rows})
    ys = sorted({round(row["ec2_target_mhz"]) for row in rows})
    valid = np.zeros((len(ys), len(xs)), dtype=bool)
    for row in rows:
        if row.get("status") == "validated_pass":
            x, y = round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"])
            valid[ys.index(y), xs.index(x)] = True
    return xs, ys, valid


def repetitions():
    best_payload = json.loads(BEST_SELECTIONS.read_text())
    best = {
        tuple(map(round, row["ec_a_mhz"])): Path(row["source_repetition"])
        for row in best_payload["selections"]
    }
    worst_payload = json.loads(WORST_STATUS.read_text())
    worst = {
        tuple(map(round, row["ec_a_mhz"])): Path(row["directory"])
        for row in worst_payload["jobs"]
        if row["state"] == "complete"
    }
    return best, worst


def load_map(directories: dict, xs: list[int], ys: list[int]):
    source_grid = np.zeros((len(ys), len(xs)), dtype=int)
    j_grid = np.full((len(ys), len(xs)), np.nan)
    failures = []
    source_counts = Counter()
    status_counts = Counter()
    for (a1, a2), directory in directories.items():
        database = directory / "gate_output/compact_anchor_results.sqlite"
        if not database.exists():
            failures.append((a1, a2))
            status_counts["missing database"] += 1
            continue
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        saved = connection.execute(
            "SELECT status,raw_fidelity,postselected_fidelity,result_json "
            "FROM gate_results WHERE round(ec_b1_mhz)=? AND round(ec_b2_mhz)=?",
            FIXED_B,
        ).fetchone()
        connection.close()
        if saved is None:
            failures.append((a1, a2))
            status_counts["missing B target"] += 1
            continue
        status, raw, post, text = saved
        result = json.loads(text)
        source = result.get("logical_source")
        final = result.get("final") or {}
        coupling = (final.get("couplings_GHz") or {}).get("A2_B1")
        ix, iy = xs.index(a1), ys.index(a2)
        if source in SOURCES:
            source_grid[iy, ix] = SOURCES.index(source) + 1
            source_counts[source] += 1
        else:
            source_counts["none"] += 1
        if coupling is not None:
            j_grid[iy, ix] = 1000.0 * float(coupling)
        fidelity_pass = (
            status == "calibrated"
            and (raw or 0.0) > 0.95
            and (post or 0.0) > 0.999
        )
        if source not in SOURCES or coupling is None or not fidelity_pass:
            failures.append((a1, a2))
            status_counts["no source, J_req, or fidelity pass"] += 1
        else:
            status_counts["complete"] += 1
    return source_grid, j_grid, failures, source_counts, status_counts


def mark_selected(ax):
    ax.scatter([SELECTED_A[0]], [SELECTED_A[1]], marker="*", s=235,
               facecolor="#fff176", edgecolor="#d62728", linewidth=1.8, zorder=6)
    ax.annotate("(154, 208)", SELECTED_A, xytext=(8, 8), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#8b0000",
                bbox={"boxstyle":"round,pad=0.2", "facecolor":"white", "alpha":.82,
                      "edgecolor":"#d62728", "linewidth":.7}, zorder=7)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    xs, ys, readout_valid = readout_grid()
    best_dirs, worst_dirs = repetitions()
    best_source, best_j, best_fail, best_counts, best_status = load_map(best_dirs, xs, ys)
    worst_source, worst_j, worst_fail, worst_counts, worst_status = load_map(worst_dirs, xs, ys)

    source_cmap = ListedColormap(["#dfe4e8", *(COLORS[source] for source in SOURCES)])
    source_norm = BoundaryNorm(np.arange(-0.5, 5.0, 1.0), source_cmap.N)
    j_values = np.concatenate([best_j[np.isfinite(best_j)], worst_j[np.isfinite(worst_j)]])
    j_norm = Normalize(float(j_values.min()), float(j_values.max()))
    extent = [min(xs)-1, max(xs)+1, min(ys)-1, max(ys)+1]

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 13.2), layout="constrained", sharex=True, sharey=True)
    columns = [
        (0, "Best case: minimum-S1 anchor with fidelity fallback", best_source, best_j, best_fail, best_counts),
        (1, "Worst case: maximum-S1 anchor, no fallback", worst_source, worst_j, worst_fail, worst_counts),
    ]
    j_image = None
    for column, title, source_grid, j_grid, failures, counts in columns:
        source_ax = axes[0, column]
        source_ax.imshow(source_grid, origin="lower", interpolation="nearest", extent=extent,
                         aspect="equal", cmap=source_cmap, norm=source_norm)
        source_ax.contour(xs, ys, readout_valid.astype(float), levels=[.5], colors="black", linewidths=1.1)
        if failures:
            source_ax.scatter([x for x, _ in failures], [y for _, y in failures], marker="x",
                              s=14, linewidths=.6, color="#222222", zorder=4)
        count_text = "  ".join(f"{source}: {counts[source]}" for source in SOURCES)
        if counts["none"]:
            count_text += f"  none: {counts['none']}"
        source_ax.set_title(title + "\nSelected CZ source — " + count_text, fontsize=10.5)
        mark_selected(source_ax)

        j_ax = axes[1, column]
        j_image = j_ax.imshow(j_grid, origin="lower", interpolation="nearest", extent=extent,
                              aspect="equal", cmap="viridis", norm=j_norm)
        j_ax.contour(xs, ys, readout_valid.astype(float), levels=[.5], colors="white", linewidths=2.5)
        j_ax.contour(xs, ys, readout_valid.astype(float), levels=[.5], colors="black", linewidths=1.0)
        if failures:
            j_ax.scatter([x for x, _ in failures], [y for _, y in failures], marker="x",
                         s=14, linewidths=.6, color="#222222", zorder=4)
        finite = j_grid[np.isfinite(j_grid)]
        j_ax.set_title(
            rf"$J_{{\rm req}}/h$ — median {np.median(finite):.3f} MHz; "
            rf"range {finite.min():.3f}-{finite.max():.3f} MHz",
            fontsize=10.5,
        )
        mark_selected(j_ax)

    for ax in axes.flat:
        ax.set_xlabel(r"$E_{C,A1}/h$ (MHz)")
        ax.set_ylabel(r"$E_{C,A2}/h$ (MHz)")
        ax.set_aspect("equal")
    fig.colorbar(j_image, ax=axes[1, :], label=r"$J_{\mathrm{req}}/h$ (MHz)", shrink=.82)
    legend = [
        Patch(facecolor=COLORS[source], edgecolor="none",
              label=rf"$|{source}\rangle\leftrightarrow|\widetilde{{0020}}\rangle$")
        for source in SOURCES
    ]
    legend.extend([
        Line2D([], [], color="black", lw=1.1, label="rail-A readout-feasible boundary"),
        Line2D([], [], color="#222222", marker="x", linestyle="none", markersize=5,
               label="missing source/J_req or failed fidelity"),
        Line2D([], [], marker="*", linestyle="none", markersize=12,
               markerfacecolor="#fff176", markeredgecolor="#d62728",
               label=r"selected rail A: $(154,208)$ MHz"),
    ])
    fig.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        r"CZ collision source and required inter-rail coupling on the rail-A grid at "
        r"$(E_{C,B1},E_{C,B2})=(216,252)$ MHz",
        fontsize=14,
    )
    png = OUTPUT / "cz_sources_and_Jreq_best_vs_worst_on_rail_a.png"
    pdf = OUTPUT / "cz_sources_and_Jreq_best_vs_worst_on_rail_a.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    ix, iy = xs.index(SELECTED_A[0]), ys.index(SELECTED_A[1])
    summary = {
        "fixed_ec_b_mhz": list(FIXED_B),
        "marked_ec_a_mhz": list(SELECTED_A),
        "best_case": {
            "source_counts": dict(best_counts), "status_counts": dict(best_status),
            "j_req_count": int(np.isfinite(best_j).sum()),
            "j_req_minimum_mhz": float(np.nanmin(best_j)),
            "j_req_median_mhz": float(np.nanmedian(best_j)),
            "j_req_maximum_mhz": float(np.nanmax(best_j)),
            "marked_source": SOURCES[best_source[iy, ix]-1] if best_source[iy, ix] else None,
            "marked_j_req_mhz": float(best_j[iy, ix]),
        },
        "worst_case": {
            "source_counts": dict(worst_counts), "status_counts": dict(worst_status),
            "j_req_count": int(np.isfinite(worst_j).sum()),
            "j_req_minimum_mhz": float(np.nanmin(worst_j)),
            "j_req_median_mhz": float(np.nanmedian(worst_j)),
            "j_req_maximum_mhz": float(np.nanmax(worst_j)),
            "marked_source": SOURCES[worst_source[iy, ix]-1] if worst_source[iy, ix] else None,
            "marked_j_req_mhz": float(worst_j[iy, ix]),
        },
        "figure_png": str(png.resolve()), "figure_pdf": str(pdf.resolve()),
        "provenance_note": (
            "The historical best-case campaign predates corrected downstream cavity propagation; "
            "the worst-case campaign uses corrected 6.05/5.95 GHz provenance."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
