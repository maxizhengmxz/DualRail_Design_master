"""Compare original 6 GHz and split-cavity direct-EC readout maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load(path: Path) -> dict[tuple[float, float], dict]:
    rows = json.loads(path.read_text())
    return {
        (float(row["ec1_target_mhz"]), float(row["ec2_target_mhz"])): row
        for row in rows
    }


def feasible(rows: dict[tuple[float, float], dict]) -> set[tuple[float, float]]:
    return {point for point, row in rows.items() if row["status"] == "validated_pass"}


def points(ax, values, **kwargs):
    values = sorted(values)
    if values:
        ax.scatter(
            [p[0] for p in values], [p[1] for p in values],
            marker="s", linewidths=0, rasterized=True, **kwargs,
        )


def map_panel(ax, rows, passed, title):
    points(ax, rows, s=8, color="#dedede")
    points(ax, passed, s=13, color="#16865c")
    ax.set_title(f"{title}\n{len(passed):,} / {len(rows):,} feasible")


def difference_panel(ax, old_rows, new_rows, old_pass, new_pass, title):
    points(ax, set(old_rows) | set(new_rows), s=8, color="#e5e5e5")
    points(ax, old_pass & new_pass, s=13, color="#4575b4")
    points(ax, old_pass - new_pass, s=19, color="#d73027")
    points(ax, new_pass - old_pass, s=19, color="#1a9850")
    ax.set_title(
        f"{title}\nshared {len(old_pass & new_pass):,}; "
        f"old only {len(old_pass - new_pass):,}; new only {len(new_pass - old_pass):,}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-a", type=Path, required=True)
    parser.add_argument("--new-a", type=Path, required=True)
    parser.add_argument("--old-b", type=Path, required=True)
    parser.add_argument("--new-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    datasets = {
        "a_old": load(args.old_a), "a_new": load(args.new_a),
        "b_old": load(args.old_b), "b_new": load(args.new_b),
    }
    passed = {name: feasible(rows) for name, rows in datasets.items()}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.3), constrained_layout=True)
    map_panel(axes[0, 0], datasets["a_old"], passed["a_old"], "(a) Rail A: original 6.00 GHz cavity")
    map_panel(axes[0, 1], datasets["a_new"], passed["a_new"], "(b) Rail A: new 6.05 GHz cavity")
    difference_panel(axes[0, 2], datasets["a_old"], datasets["a_new"], passed["a_old"], passed["a_new"], "(c) Rail-A feasibility change")
    map_panel(axes[1, 0], datasets["b_old"], passed["b_old"], "(d) Rail B: original 6.00 GHz cavity")
    map_panel(axes[1, 1], datasets["b_new"], passed["b_new"], "(e) Rail B: new 5.95 GHz cavity")
    difference_panel(axes[1, 2], datasets["b_old"], datasets["b_new"], passed["b_old"], passed["b_new"], "(f) Rail-B feasibility change")

    all_coordinates = set().union(*(set(rows) for rows in datasets.values()))
    low = min(min(point) for point in all_coordinates) - 3
    high = max(max(point) for point in all_coordinates) + 3
    for row, rail in zip(axes, ("A", "B")):
        for ax in row:
            ax.set_aspect("equal")
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.plot([low, high], [low, high], color="0.7", lw=0.8, zorder=0)
            ax.set_xlabel(rf"$E_{{C,{rail}1}}/h$ (MHz)")
            ax.set_ylabel(rf"$E_{{C,{rail}2}}/h$ (MHz)")

    axes[0, 0].legend(handles=[
        Line2D([], [], marker="s", ls="", color="#16865c", label="Readout feasible"),
        Line2D([], [], marker="s", ls="", color="#dedede", label="No validated witness"),
    ], loc="upper left", frameon=False, fontsize=8)
    axes[0, 2].legend(handles=[
        Line2D([], [], marker="s", ls="", color="#4575b4", label="Feasible in both"),
        Line2D([], [], marker="s", ls="", color="#d73027", label="6.00 GHz only"),
        Line2D([], [], marker="s", ls="", color="#1a9850", label="New cavity only"),
        Line2D([], [], marker="s", ls="", color="#e5e5e5", label="Not feasible"),
    ], loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Effect of split bare-cavity frequencies on independent EC readout maps", fontsize=15)

    args.output.mkdir(parents=True, exist_ok=True)
    png = args.output / "split_cavity_ec_readout_map_comparison.png"
    pdf = args.output / "split_cavity_ec_readout_map_comparison.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    summary = {}
    for rail in ("a", "b"):
        old, new = passed[f"{rail}_old"], passed[f"{rail}_new"]
        summary[rail] = {
            "old_feasible": len(old), "new_feasible": len(new),
            "shared_feasible": len(old & new),
            "old_only": len(old - new), "new_only": len(new - old),
        }
    (args.output / "split_cavity_ec_readout_map_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
