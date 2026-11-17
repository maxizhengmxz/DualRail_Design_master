#!/usr/bin/env python3
"""Plot the largest rail-A component selecting ordered rail-B EC=(262,186) MHz."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "combined_overlay_selection_2026-11-12" / "idle_above_all_interactions_200MHz"
SOURCE = ANALYSIS / "reselected_configurations.json"
OUTPUT = ANALYSIS / "ecb_262_186_component"
TARGET_B = (262, 186)

spec = importlib.util.spec_from_file_location("base", HERE / "select_combined_overlay_configurations.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def main() -> None:
    rows = [
        row for row in json.loads(SOURCE.read_text())["selections"]
        if tuple(row["ec_b_mhz"]) == TARGET_B
    ]
    mapping = {tuple(row["ec_a_mhz"]): row for row in rows}
    unseen = set(mapping)
    components = []
    while unseen:
        seed = unseen.pop(); component = {seed}; stack = [seed]
        while stack:
            x, y = stack.pop()
            for neighbor in ((x-2,y),(x+2,y),(x,y-2),(x,y+2)):
                if neighbor in unseen:
                    unseen.remove(neighbor); component.add(neighbor); stack.append(neighbor)
        components.append(component)
    component = max(components, key=len)
    selected = [mapping[point] for point in sorted(component)]
    anchors = sorted({row["selected_anchor_id"] for row in selected})
    palette = plt.get_cmap("tab10")
    colors = {anchor: palette(index) for index, anchor in enumerate(anchors)}

    xs, ys, grid = base.combined_a_background()
    mask = np.zeros((len(ys), len(xs)))
    for x, y in component:
        mask[ys.index(y), xs.index(x)] = 1
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 8.8), layout="constrained")
    ax.imshow(grid, origin="lower", interpolation="nearest",
              extent=[min(xs)-1,max(xs)+1,min(ys)-1,max(ys)+1], aspect="equal",
              cmap=ListedColormap(["#e2e6e9", "#d8efe8"]),
              norm=BoundaryNorm([-.5,.5,1.5],2))
    ax.contour(xs,ys,mask,levels=[.5],colors=["#d62728"],linewidths=2.2,zorder=5)
    for row in selected:
        x,y = row["ec_a_mhz"]
        ax.scatter(x,y,c=[colors[row["selected_anchor_id"]]],s=38,
                   edgecolors="black",linewidths=.4,zorder=4)
    ax.annotate(
        rf"{len(component)}-point component" "\n" rf"shared $E_{{C,B}}=(262,186)$ MHz",
        xy=(160,239),xytext=(178,253),
        arrowprops=dict(arrowstyle="->",color="#d62728",lw=1.5),
        fontsize=9,color="#a50f15",fontweight="bold",
    )
    ax.plot([min(xs),max(xs)],[min(ys),max(ys)],"k--",lw=.7,alpha=.35)
    ax.legend(handles=[
        Patch(facecolor="#d8efe8",edgecolor=".5",label="Rail-A readout valid"),
        Patch(facecolor="#e2e6e9",edgecolor=".5",label="Rail-A readout invalid"),
        *[Line2D([],[],marker="o",ls="",color=colors[a],label=a) for a in anchors],
        Line2D([],[],color="#d62728",lw=2.2,label=r"Shared $E_{C,B}=(262,186)$ component"),
    ],loc="upper right",fontsize=8)
    ax.set(title=r"Rail-A connected region selecting ordered $E_{C,B}=(262,186)$ MHz",
           xlabel=r"$E_{C,A1}/h$ (MHz)",ylabel=r"$E_{C,A2}/h$ (MHz)")
    ax.set_aspect("equal")
    fig.savefig(OUTPUT / "component_on_full_rail_a_map.png",dpi=220)
    fig.savefig(OUTPUT / "component_on_full_rail_a_map.pdf")
    plt.close(fig)

    record = {
        "connectivity": "4-neighbor on 2 MHz rail-A grid",
        "shared_ec_b_mhz": list(TARGET_B),
        "component_size": len(component),
        "rail_a_points": [list(point) for point in sorted(component)],
        "anchor_counts": dict(Counter(row["selected_anchor_id"] for row in selected)),
        "minimum_robustness_margin_mhz": min(row["anchor_consistent_robustness_margin_mhz"] for row in selected),
        "neighborhood_clearance_range_mhz": [
            min(row["neighborhood_minimum_idle_above_interactions_clearance_mhz"] for row in selected),
            max(row["neighborhood_minimum_idle_above_interactions_clearance_mhz"] for row in selected),
        ],
    }
    (OUTPUT / "component_details.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"output":str(OUTPUT),**record},indent=2))


if __name__ == "__main__": main()
