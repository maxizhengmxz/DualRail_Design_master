#!/usr/bin/env python3
"""Compare required inter-rail coupling on the B grid at A=(154,208)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
BEST = (
    HERE / "cavity_split_campaign_2026-11-09/sampled_runs/rail_a_lt_partition"
    / "A_LT_0086_ecA_154_208/gate_output/compact_anchor_results.sqlite"
)
WORST = (
    HERE / "worst_case_fractional_anchor_corrected_cavities_2026-11-13"
    / "split_6p05_5p95/A_LT_0086_ecA_154_208"
    / "gate_output/compact_anchor_results.sqlite"
)
B_READOUT = (
    HERE / "cavity_split_campaign_2026-11-09/readout/direct_ec"
    / "rail_b_step2_fr5p95/results.json"
)
OUTPUT = (
    HERE / "worst_case_fractional_anchor_corrected_cavities_2026-11-13"
    / "campaign_summary/A154_208_best_vs_worst_Jreq"
)
SELECTED_B = (216, 252)


def readout_grid():
    rows = json.loads(B_READOUT.read_text())
    xs = sorted({round(row["ec1_target_mhz"]) for row in rows})
    ys = sorted({round(row["ec2_target_mhz"]) for row in rows})
    feasible = np.zeros((len(ys), len(xs)), dtype=bool)
    for row in rows:
        if row.get("status") == "validated_pass":
            x, y = round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"])
            feasible[ys.index(y), xs.index(x)] = True
    return xs, ys, feasible


def required_coupling(path: Path, xs: list[int], ys: list[int]):
    grid = np.full((len(ys), len(xs)), np.nan)
    missing = []
    source = np.full((len(ys), len(xs)), "", dtype="U2")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT ec_b1_mhz,ec_b2_mhz,result_json FROM gate_results"
    ).fetchall()
    connection.close()
    for ec1, ec2, text in rows:
        x, y = round(ec1), round(ec2)
        result = json.loads(text)
        final = result.get("final") or {}
        coupling = (final.get("couplings_GHz") or {}).get("A2_B1")
        source[ys.index(y), xs.index(x)] = result.get("logical_source") or ""
        if coupling is None:
            missing.append((x, y))
        else:
            grid[ys.index(y), xs.index(x)] = 1000.0 * float(coupling)
    return grid, source, missing


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    xs, ys, feasible = readout_grid()
    best, best_source, best_missing = required_coupling(BEST, xs, ys)
    worst, worst_source, worst_missing = required_coupling(WORST, xs, ys)
    values = np.concatenate([best[np.isfinite(best)], worst[np.isfinite(worst)]])
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    extent = [min(xs)-1, max(xs)+1, min(ys)-1, max(ys)+1]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), layout="constrained", sharex=True, sharey=True)
    panels = [
        (axes[0], best, best_missing, "Best case: minimum-S1 anchor with fidelity fallback"),
        (axes[1], worst, worst_missing, "Worst case: maximum-S1 anchor, no fallback"),
    ]
    image = None
    for ax, grid, missing, title in panels:
        image = ax.imshow(grid, origin="lower", interpolation="nearest", extent=extent,
                          aspect="equal", cmap="viridis", norm=norm)
        ax.contour(xs, ys, feasible.astype(float), levels=[.5], colors="white", linewidths=2.5)
        ax.contour(xs, ys, feasible.astype(float), levels=[.5], colors="black", linewidths=1.05)
        if missing:
            ax.scatter([x for x, _ in missing], [y for _, y in missing], marker="x",
                       s=13, linewidths=.55, color="#222222", zorder=3)
        ax.scatter([SELECTED_B[0]], [SELECTED_B[1]], marker="*", s=235,
                   facecolor="#fff176", edgecolor="#d62728", linewidth=1.8, zorder=5)
        ax.annotate("(216, 252)", SELECTED_B, xytext=(8, 8), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#8b0000",
                    bbox={"boxstyle":"round,pad=0.2", "facecolor":"white", "alpha":.82,
                          "edgecolor":"#d62728", "linewidth":.7}, zorder=6)
        finite = grid[np.isfinite(grid)]
        ax.set_title(
            title + "\n" +
            rf"median {np.median(finite):.3f} MHz; range {finite.min():.3f}-{finite.max():.3f} MHz",
            fontsize=10.5,
        )
        ax.set_xlabel(r"$E_{C,B1}/h$ (MHz)")
        ax.set_ylabel(r"$E_{C,B2}/h$ (MHz)")
        ax.set_aspect("equal")
    fig.colorbar(image, ax=axes, label=r"$J_{\mathrm{req}}/h$ (MHz)", shrink=.82)
    fig.legend(
        handles=[
            Line2D([], [], color="black", lw=1.1, label="rail-B readout-feasible boundary"),
            Line2D([], [], color="#222222", marker="x", linestyle="none", markersize=5,
                   label=r"no calibrated $J_{\mathrm{req}}$"),
            Line2D([], [], marker="*", linestyle="none", markersize=12,
                   markerfacecolor="#fff176", markeredgecolor="#d62728",
                   label=r"selected tuple: $(E_{C,B1},E_{C,B2})=(216,252)$ MHz"),
        ],
        loc="outside lower center", ncol=3, frameon=False,
    )
    fig.suptitle(
        r"Required physical inter-rail coupling for $g_{\mathrm{target}}/h=5$ MHz at "
        r"$(E_{C,A1},E_{C,A2})=(154,208)$ MHz",
        fontsize=14,
    )
    png = OUTPUT / "Jreq_best_vs_worst.png"
    pdf = OUTPUT / "Jreq_best_vs_worst.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    ix, iy = xs.index(SELECTED_B[0]), ys.index(SELECTED_B[1])
    summary = {
        "ec_a_mhz": [154, 208],
        "marked_ec_b_mhz": list(SELECTED_B),
        "target_effective_g_mhz": 5.0,
        "definition": "J_req = g_target / |<logical source|n_A2 n_B1|~0020>|",
        "shared_color_scale_mhz": [float(values.min()), float(values.max())],
        "best_case": {
            "stored_point_count": int(np.isfinite(best).sum()),
            "missing_count": len(best_missing),
            "minimum_mhz": float(np.nanmin(best)),
            "median_mhz": float(np.nanmedian(best)),
            "maximum_mhz": float(np.nanmax(best)),
            "selected_tuple_j_req_mhz": float(best[iy, ix]),
            "selected_tuple_logical_source": best_source[iy, ix],
        },
        "worst_case": {
            "stored_point_count": int(np.isfinite(worst).sum()),
            "missing_count": len(worst_missing),
            "minimum_mhz": float(np.nanmin(worst)),
            "median_mhz": float(np.nanmedian(worst)),
            "maximum_mhz": float(np.nanmax(worst)),
            "selected_tuple_j_req_mhz": float(worst[iy, ix]),
            "selected_tuple_logical_source": worst_source[iy, ix],
        },
        "figure_png": str(png.resolve()),
        "figure_pdf": str(pdf.resolve()),
        "provenance_note": (
            "The historical best-case split campaign predates the corrected-cavity downstream "
            "rerun; the worst-case campaign uses corrected 6.05/5.95 GHz provenance."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
