#!/usr/bin/env python3
"""Find and plot the largest 4-connected rail-A component sharing one rail-B EC."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "combined_overlay_selection_2026-11-12" / "idle_above_all_interactions_200MHz"
SOURCE = ANALYSIS / "reselected_configurations.json"
OUTPUT = ANALYSIS / "largest_same_ecb_component"
BASE_PATH = HERE / "select_combined_overlay_configurations.py"
SPEC = importlib.util.spec_from_file_location("combined_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)


def components(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[tuple(row["ec_b_mhz"])][tuple(row["ec_a_mhz"])] = row
    found = []
    for ec_b, mapping in grouped.items():
        unseen = set(mapping)
        while unseen:
            seed = unseen.pop()
            component = {seed}
            stack = [seed]
            while stack:
                x, y = stack.pop()
                for neighbor in ((x-2, y), (x+2, y), (x, y-2), (x, y+2)):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            found.append((len(component), ec_b, component, mapping))
    return sorted(found, key=lambda item: item[0], reverse=True)


def b_background(ax):
    rows = json.loads(BASE.SPLIT_B.read_text())
    xs = sorted({round(r["ec1_target_mhz"]) for r in rows})
    ys = sorted({round(r["ec2_target_mhz"]) for r in rows})
    import numpy as np
    values = np.full((len(ys), len(xs)), np.nan)
    for row in rows:
        x, y = round(row["ec1_target_mhz"]), round(row["ec2_target_mhz"])
        values[ys.index(y), xs.index(x)] = float(row["status"] == "validated_pass")
    ax.imshow(values, origin="lower", interpolation="nearest",
              extent=[min(xs)-1,max(xs)+1,min(ys)-1,max(ys)+1], aspect="equal",
              cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
              norm=BoundaryNorm([-.5,.5,1.5],2))


def main() -> None:
    rows = json.loads(SOURCE.read_text())["selections"]
    size, ec_b, component, mapping = components(rows)[0]
    selected = [mapping[point] for point in sorted(component, key=lambda p: (p[1], p[0]))]
    anchors = sorted({row["selected_anchor_id"] for row in selected})
    colors = {anchor: color for anchor, color in zip(anchors, ("#2a6fbb", "#d95f02"))}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), layout="constrained")

    xs, ys, grid = BASE.combined_a_background()
    axes[0].imshow(grid, origin="lower", interpolation="nearest",
                   extent=[min(xs)-1,max(xs)+1,min(ys)-1,max(ys)+1], aspect="equal",
                   cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
                   norm=BoundaryNorm([-.5,.5,1.5],2))
    for index, row in enumerate(selected, 1):
        x, y = row["ec_a_mhz"]
        axes[0].scatter(x, y, c=colors[row["selected_anchor_id"]], s=62,
                        edgecolors="black", linewidths=.5, zorder=4)
        axes[0].annotate(str(index), (x,y), xytext=(3,3), textcoords="offset points", fontsize=7)
    axes[0].legend(handles=[Line2D([],[],marker="o",ls="",color=color,label=anchor)
                            for anchor,color in colors.items()], title="Selected rail-A anchor", fontsize=8)
    axes[0].set(title=f"Largest same-rail-B component: {size} rail-A points",
                xlabel=r"$E_{C,A1}/h$ (MHz)", ylabel=r"$E_{C,A2}/h$ (MHz)",
                xlim=(147,171), ylim=(231,239))
    axes[0].set_aspect("equal")

    b_background(axes[1])
    radii = [row["anchor_consistent_robustness_margin_mhz"] for row in selected]
    radius = min(radii)
    axes[1].add_patch(Circle(ec_b, radius, facecolor="none", edgecolor="#7b3294", lw=2))
    axes[1].scatter(*ec_b, c="#7b3294", s=100, edgecolors="black", zorder=4)
    axes[1].annotate(f"selected by {size} rail-A points\nminimum radius {radius:.2f} MHz",
                     ec_b, xytext=(8,8), textcoords="offset points", fontsize=9)
    axes[1].legend(handles=[Patch(facecolor="#d8efe8",edgecolor=".5",label="Rail-B readout valid"),
                            Patch(facecolor="#e2e6e9",edgecolor=".5",label="Rail-B readout invalid")],
                   loc="upper right", fontsize=8)
    axes[1].set(title=rf"Shared ordered rail-B configuration: $({ec_b[0]},{ec_b[1]})$ MHz",
                xlabel=r"$E_{C,B1}/h$ (MHz)", ylabel=r"$E_{C,B2}/h$ (MHz)",
                xlim=(245,283), ylim=(180,220))
    axes[1].set_aspect("equal")
    fig.suptitle("Largest 4-connected rail-A region with one exact selected rail-B EC pair", fontsize=13)
    fig.savefig(OUTPUT / "largest_same_ecb_component.png", dpi=220)
    fig.savefig(OUTPUT / "largest_same_ecb_component.pdf")
    plt.close(fig)

    # Full rail-A map: show where this connected component sits in the complete
    # independent readout-feasible domain rather than only in the zoomed view.
    fig, ax = plt.subplots(figsize=(10.5, 8.8), layout="constrained")
    ax.imshow(grid, origin="lower", interpolation="nearest",
              extent=[min(xs)-1,max(xs)+1,min(ys)-1,max(ys)+1], aspect="equal",
              cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
              norm=BoundaryNorm([-.5,.5,1.5],2))
    # A one-cell mask gives a clean contour around the union of selected 2 MHz
    # grid cells, including the component's non-rectangular end points.
    component_mask = np.zeros((len(ys), len(xs)), dtype=float)
    for x, y in component:
        component_mask[ys.index(y), xs.index(x)] = 1.0
    ax.contour(
        xs, ys, component_mask, levels=[.5], colors=["#d62728"],
        linewidths=2.2, zorder=5,
    )
    for row in selected:
        x, y = row["ec_a_mhz"]
        ax.scatter(x, y, c=colors[row["selected_anchor_id"]], s=34,
                   edgecolors="black", linewidths=.35, zorder=4)
    ax.annotate(
        rf"16-point component" "\n" rf"shared $E_{{C,B}}=(272,200)$ MHz",
        xy=(160,235), xytext=(178,250),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
        fontsize=9, color="#a50f15", fontweight="bold",
    )
    ax.plot([min(xs), max(xs)], [min(ys), max(ys)], "k--", lw=.7, alpha=.35)
    ax.legend(
        handles=[
            Patch(facecolor="#d8efe8", edgecolor=".5", label="Rail-A readout valid"),
            Patch(facecolor="#e2e6e9", edgecolor=".5", label="Rail-A readout invalid"),
            *[Line2D([],[],marker="o",ls="",color=color,label=anchor)
              for anchor,color in colors.items()],
            Line2D([],[],color="#d62728",lw=2.2,label="Largest same-$E_{C,B}$ component"),
        ],
        loc="upper right", fontsize=8,
    )
    ax.set(
        title="Largest connected same-rail-B region on the full rail-A readout map",
        xlabel=r"$E_{C,A1}/h$ (MHz)", ylabel=r"$E_{C,A2}/h$ (MHz)",
    )
    ax.set_aspect("equal")
    fig.savefig(OUTPUT / "component_marked_on_full_rail_a_map.png", dpi=220)
    fig.savefig(OUTPUT / "component_marked_on_full_rail_a_map.pdf")
    plt.close(fig)

    record = {
        "connectivity": "4-neighbor on 2 MHz rail-A grid",
        "component_size": size,
        "shared_ec_b_mhz": list(ec_b),
        "rail_a_points": [list(point) for point in sorted(component)],
        "anchor_counts": dict(Counter(row["selected_anchor_id"] for row in selected)),
        "minimum_robustness_margin_mhz": min(radii),
        "clearance_range_mhz": [
            min(row["neighborhood_minimum_idle_above_interactions_clearance_mhz"] for row in selected),
            max(row["neighborhood_minimum_idle_above_interactions_clearance_mhz"] for row in selected),
        ],
        "idle_minus_selected_gate_frequency_range_mhz": [
            min(row["idle_operating_minus_selected_gate_frequency_mhz"] for row in selected),
            max(row["idle_operating_minus_selected_gate_frequency_mhz"] for row in selected),
        ],
    }
    (OUTPUT / "component_details.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), **record}, indent=2))


if __name__ == "__main__":
    main()
