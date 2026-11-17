#!/usr/bin/env python3
"""Plot the largest rail-A component whose selected rail-B EC lies in a 5 MHz ball."""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
from collections import Counter
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
OUTPUT = ANALYSIS / "largest_ecb_5mhz_ball_component"
RADIUS = 5.0

spec = importlib.util.spec_from_file_location("same_ecb", HERE / "plot_largest_same_ecb_component.py")
helper = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helper)


def search(rows):
    b_points = sorted({tuple(row["ec_b_mhz"]) for row in rows})
    centers = [(float(x), float(y)) for x, y in b_points]
    for first, second in itertools.combinations(b_points, 2):
        dx, dy = second[0]-first[0], second[1]-first[1]
        distance = math.hypot(dx, dy)
        if not 0 < distance <= 2*RADIUS + 1e-9:
            continue
        mx, my = (first[0]+second[0])/2, (first[1]+second[1])/2
        height = math.sqrt(max(0.0, RADIUS**2-distance**2/4))
        ux, uy = -dy/distance, dx/distance
        centers.extend([(mx+height*ux, my+height*uy), (mx-height*ux, my-height*uy)])
    best = None
    for center in centers:
        allowed = {point for point in b_points if math.dist(point, center) <= RADIUS+1e-7}
        mapping = {tuple(row["ec_a_mhz"]): row for row in rows if tuple(row["ec_b_mhz"]) in allowed}
        unseen = set(mapping)
        while unseen:
            seed = unseen.pop(); component = {seed}; stack = [seed]
            while stack:
                x, y = stack.pop()
                for neighbor in ((x-2,y),(x+2,y),(x,y-2),(x,y+2)):
                    if neighbor in unseen:
                        unseen.remove(neighbor); component.add(neighbor); stack.append(neighbor)
            candidate = (len(component), center, allowed, component, mapping)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def main() -> None:
    rows = json.loads(SOURCE.read_text())["selections"]
    size, center, allowed, component, mapping = search(rows)
    selected = [mapping[point] for point in sorted(component)]
    used = Counter(tuple(row["ec_b_mhz"]) for row in selected)
    colors = {point: color for point, color in zip(sorted(used), ("#1b9e77", "#d95f02", "#7570b3"))}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), layout="constrained")

    xs, ys, grid = helper.BASE.combined_a_background()
    axes[0].imshow(grid, origin="lower", interpolation="nearest",
                   extent=[min(xs)-1,max(xs)+1,min(ys)-1,max(ys)+1], aspect="equal",
                   cmap=ListedColormap(["#e2e6e9", "#d8efe8"]), norm=BoundaryNorm([-.5,.5,1.5],2))
    mask = np.zeros((len(ys),len(xs)))
    for x,y in component: mask[ys.index(y),xs.index(x)] = 1
    axes[0].contour(xs,ys,mask,levels=[.5],colors=["#d62728"],linewidths=2.2,zorder=5)
    for row in selected:
        b = tuple(row["ec_b_mhz"]); x,y = row["ec_a_mhz"]
        axes[0].scatter(x,y,c=colors[b],s=34,edgecolors="black",linewidths=.35,zorder=4)
    axes[0].legend(handles=[Line2D([],[],marker="o",ls="",color=color,label=f"B EC={point}, n={used[point]}")
                            for point,color in colors.items()],fontsize=8,loc="upper right")
    axes[0].set(title=f"Largest 5 MHz-ball component: {size} rail-A points",
                xlabel=r"$E_{C,A1}/h$ (MHz)",ylabel=r"$E_{C,A2}/h$ (MHz)",
                xlim=(147,173),ylim=(229,241)); axes[0].set_aspect("equal")

    helper.b_background(axes[1])
    axes[1].add_patch(Circle(center,RADIUS,facecolor="#7b3294",edgecolor="#542788",alpha=.15,lw=2))
    axes[1].scatter(*center,marker="+",c="#542788",s=100,lw=2,zorder=5)
    for point,count in sorted(used.items()):
        axes[1].scatter(*point,c=colors[point],s=85,edgecolors="black",zorder=5)
        axes[1].annotate(f"{point}\nn={count}",point,xytext=(5,5),textcoords="offset points",fontsize=8)
    axes[1].legend(handles=[Patch(facecolor="#d8efe8",edgecolor=".5",label="Rail-B readout valid"),
                            Patch(facecolor="#e2e6e9",edgecolor=".5",label="Rail-B readout invalid"),
                            Patch(facecolor="#7b3294",alpha=.15,edgecolor="#542788",label="5 MHz EC ball")],
                   fontsize=8,loc="upper right")
    axes[1].set(title=rf"Selected rail-B EC points inside radius 5 MHz ball centered at ({center[0]:.1f},{center[1]:.1f})",
                xlabel=r"$E_{C,B1}/h$ (MHz)",ylabel=r"$E_{C,B2}/h$ (MHz)",
                xlim=(258,278),ylim=(190,208)); axes[1].set_aspect("equal")
    fig.suptitle("Largest connected rail-A region after relaxing exact rail-B EC equality",fontsize=13)
    fig.savefig(OUTPUT / "largest_ecb_5mhz_ball_component.png",dpi=220)
    fig.savefig(OUTPUT / "largest_ecb_5mhz_ball_component.pdf")
    plt.close(fig)

    # Full rail-A readout-map view of the relaxed component.
    fig, ax = plt.subplots(figsize=(10.5, 8.8), layout="constrained")
    ax.imshow(grid, origin="lower", interpolation="nearest",
              extent=[min(xs)-1,max(xs)+1,min(ys)-1,max(ys)+1], aspect="equal",
              cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
              norm=BoundaryNorm([-.5,.5,1.5],2))
    ax.contour(xs,ys,mask,levels=[.5],colors=["#d62728"],linewidths=2.2,zorder=5)
    for row in selected:
        b = tuple(row["ec_b_mhz"]); x,y = row["ec_a_mhz"]
        ax.scatter(x,y,c=colors[b],s=34,edgecolors="black",linewidths=.35,zorder=4)
    ax.annotate(
        "33-point connected component\nselected B ECs fit inside a 5 MHz ball",
        xy=(160,235), xytext=(178,251),
        arrowprops=dict(arrowstyle="->",color="#d62728",lw=1.5),
        fontsize=9,color="#a50f15",fontweight="bold",
    )
    ax.plot([min(xs),max(xs)],[min(ys),max(ys)],"k--",lw=.7,alpha=.35)
    ax.legend(
        handles=[
            Patch(facecolor="#d8efe8",edgecolor=".5",label="Rail-A readout valid"),
            Patch(facecolor="#e2e6e9",edgecolor=".5",label="Rail-A readout invalid"),
            *[Line2D([],[],marker="o",ls="",color=color,
                     label=f"B EC={point}, n={used[point]}")
              for point,color in colors.items()],
            Line2D([],[],color="#d62728",lw=2.2,label="Largest 5 MHz-ball component"),
        ],
        loc="upper right",fontsize=8,
    )
    ax.set(
        title="Largest 5 MHz rail-B-ball component on the full rail-A readout map",
        xlabel=r"$E_{C,A1}/h$ (MHz)",ylabel=r"$E_{C,A2}/h$ (MHz)",
    )
    ax.set_aspect("equal")
    fig.savefig(OUTPUT / "component_marked_on_full_rail_a_map.png",dpi=220)
    fig.savefig(OUTPUT / "component_marked_on_full_rail_a_map.pdf")
    plt.close(fig)

    record = {
        "connectivity": "4-neighbor on 2 MHz rail-A grid",
        "rail_b_ball_center_mhz": list(center), "rail_b_ball_radius_mhz": RADIUS,
        "component_size": size, "rail_a_points": [list(point) for point in sorted(component)],
        "used_rail_b_counts": {f"{x},{y}": count for (x,y),count in sorted(used.items())},
        "selected_anchor_counts": dict(Counter(row["selected_anchor_id"] for row in selected)),
        "minimum_robustness_margin_mhz": min(row["anchor_consistent_robustness_margin_mhz"] for row in selected),
        "neighborhood_clearance_range_mhz": [min(row["neighborhood_minimum_idle_above_interactions_clearance_mhz"] for row in selected),max(row["neighborhood_minimum_idle_above_interactions_clearance_mhz"] for row in selected)],
    }
    (OUTPUT / "component_details.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"output":str(OUTPUT),**record},indent=2))


if __name__ == "__main__": main()
