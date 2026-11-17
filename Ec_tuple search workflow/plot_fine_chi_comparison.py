"""Compare fine C1-C2 readout maps at two mean-chi requirements."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D


PASS = {"validated_pass", "recovered_viable"}


def load(path: Path):
    return json.loads(path.read_text())


def mean_chi_mhz(row):
    point = row.get("point")
    if not point:
        return np.nan
    return abs(500.0 * (point["chi_m_ghz"] + point["chi_p_ghz"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi1", type=Path, required=True)
    parser.add_argument("--chi5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old = load(args.chi1 / "results.json")
    new = load(args.chi5 / "results.json")
    diagnostics = load(args.chi5 / "diagnostic_results.json")
    old_pass = [row for row in old if row["status"] in PASS]
    new_pass = [row for row in new if row["status"] in PASS]
    old_chi = np.array([mean_chi_mhz(row) for row in old_pass])

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.2), constrained_layout=True)
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xlim(35, 129)
        ax.set_ylim(35, 129)
        ax.set_xlabel(r"$C_1$ (fF)")
        ax.set_ylabel(r"$C_2$ (fF)")
        ax.plot([36, 128], [36, 128], color="0.75", lw=0.8)

    for ax, rows, passed, title in (
        (axes[0, 0], old, old_pass, r"(a) $|\bar{\chi}|\geq1$ MHz"),
        (axes[0, 1], new, new_pass, r"(b) $|\bar{\chi}|\geq5$ MHz"),
    ):
        ax.scatter([r["c1"] for r in rows], [r["c2"] for r in rows], s=7,
                   color="#dedede", marker="s", linewidths=0, rasterized=True)
        if passed:
            ax.scatter([r["c1"] for r in passed], [r["c2"] for r in passed], s=12,
                       color="#16865c", marker="s", linewidths=0, rasterized=True)
        ax.set_title(f"{title}\n{len(passed):,} / {len(rows):,} feasible")
    axes[0, 0].legend(handles=[
        Line2D([], [], marker="s", ls="", color="#16865c", label="Feasible"),
        Line2D([], [], marker="s", ls="", color="#dedede", label="Not feasible"),
    ], loc="upper left", frameon=False)

    scatter = axes[1, 0].scatter(
        [r["c1"] for r in old_pass], [r["c2"] for r in old_pass], c=old_chi,
        s=18, marker="s", linewidths=0, cmap="viridis", vmin=1.0, vmax=5.0,
        rasterized=True,
    )
    axes[1, 0].set_title(
        r"(c) $|\bar{\chi}|$ of old feasible witnesses"
        + f"\nrange {old_chi.min():.3f}-{old_chi.max():.3f} MHz"
    )
    fig.colorbar(scatter, ax=axes[1, 0], label=r"$|\bar{\chi}|$ (MHz)")

    dominant = [row.get("dominant", "unspecified") for row in diagnostics]
    order = [name for name, _ in Counter(dominant).most_common()]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(order)))
    lookup = {name: index for index, name in enumerate(order)}
    codes = np.array([lookup[name] for name in dominant])
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(order) + 0.5), len(order))
    axes[1, 1].scatter(
        [r["c1"] for r in diagnostics], [r["c2"] for r in diagnostics], c=codes,
        s=13, marker="s", linewidths=0, cmap=cmap, norm=norm, rasterized=True,
    )
    axes[1, 1].set_title("(d) Dominant failure at 5 MHz\n2 fF diagnostic lattice")
    axes[1, 1].legend(
        handles=[Line2D([], [], marker="s", ls="", color=colors[i],
                        label=f"{name.replace('_', ' ')} ({dominant.count(name)})")
                 for i, name in enumerate(order)],
        loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8,
    )

    fig.suptitle("Rail-A fine C1-C2 readout comparison (logical frequency 140-160 MHz)",
                 fontsize=15)
    args.output.mkdir(parents=True, exist_ok=True)
    png = args.output / "rail_a_fine_c1_c2_chi1_vs_chi5.png"
    pdf = args.output / "rail_a_fine_c1_c2_chi1_vs_chi5.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    summary = {
        "chi1_directory": str(args.chi1),
        "chi5_directory": str(args.chi5),
        "logical_frequency_mhz": [140.0, 160.0],
        "grid_points_each": len(old),
        "chi1_feasible": len(old_pass),
        "chi5_feasible": len(new_pass),
        "old_witness_abs_mean_chi_mhz": {
            "min": float(old_chi.min()),
            "median": float(np.median(old_chi)),
            "max": float(old_chi.max()),
            "count_ge_5": int(np.count_nonzero(old_chi >= 5.0)),
        },
        "chi5_diagnostic_status_counts": dict(Counter(r["status"] for r in diagnostics)),
        "chi5_dominant_failure_counts": dict(Counter(dominant)),
        "comparison_note": (
            "Grid and declared numerical policies match, except abs_mean_chi_min_mhz. "
            "Source hashes differ between campaign dates, so the old-witness threshold "
            "recheck is the strict threshold-only comparison."
        ),
    }
    (args.output / "rail_a_fine_c1_c2_chi1_vs_chi5_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
