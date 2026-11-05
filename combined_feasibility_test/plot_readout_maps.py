"""Plot the independent rail-A and rail-B direct-EC readout maps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load(name: str) -> list[dict]:
    return json.loads((DATA / name / "results.json").read_text(encoding="utf-8"))


def main() -> int:
    maps = [
        ("Rail A readout: 140–160 MHz", load("rail_a_readout_map")),
        ("Rail B readout: 80–100 MHz", load("rail_b_readout_map_official")),
    ]
    targets = sorted(
        {
            (row["ec1_target_mhz"], row["ec2_target_mhz"])
            for _, rows in maps
            for row in rows
        }
    )
    xs = sorted({x for x, _ in targets})
    ys = sorted({y for _, y in targets})
    xi = {x: index for index, x in enumerate(xs)}
    yi = {y: index for index, y in enumerate(ys)}
    cmap = ListedColormap(["#d95f5f", "#2a9d8f"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), constrained_layout=True)
    for axis, (title, rows) in zip(axes, maps, strict=True):
        grid = np.full((len(ys), len(xs)), np.nan)
        for row in rows:
            x, y = row["ec1_target_mhz"], row["ec2_target_mhz"]
            grid[yi[y], xi[x]] = row["status"] == "validated_pass"
        axis.imshow(
            grid,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[min(xs) - 2.5, max(xs) + 2.5, min(ys) - 2.5, max(ys) + 2.5],
            aspect="auto",
        )
        passed = sum(row["status"] == "validated_pass" for row in rows)
        axis.set(
            xlabel=r"$E_{C,1}$ (MHz)",
            ylabel=r"$E_{C,2}$ (MHz)",
            title=f"{title}\n{passed}/{len(rows)} validated",
            xticks=xs,
            yticks=ys,
        )
        axis.tick_params(axis="x", rotation=45)
        axis.grid(False)
    axes[1].legend(
        handles=[
            Patch(facecolor=cmap(1), label="Validated readout witness"),
            Patch(facecolor=cmap(0), label="No validated witness found"),
        ],
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
    )
    figure.suptitle("Independent readout feasibility on the direct-EC hull", fontsize=14)
    output = DATA / "plots"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "independent_readout_maps.png", dpi=220)
    figure.savefig(output / "independent_readout_maps.pdf")
    plt.close(figure)
    print(output / "independent_readout_maps.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
