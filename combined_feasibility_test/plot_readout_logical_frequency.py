"""Plot audited logical frequency over the independent readout-feasible maps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load(directory: str) -> list[dict]:
    return json.loads((DATA / directory / "results.json").read_text(encoding="utf-8"))


def main() -> int:
    maps = [
        ("Rail A", "rail_a_readout_map", (140.0, 160.0)),
        ("Rail B", "rail_b_readout_map_official", (80.0, 100.0)),
    ]
    rows_by_map = [(title, load(directory), limits) for title, directory, limits in maps]
    targets = {
        (row["ec1_target_mhz"], row["ec2_target_mhz"])
        for _, rows, _ in rows_by_map
        for row in rows
    }
    xs = sorted({x for x, _ in targets})
    ys = sorted({y for _, y in targets})
    xi = {x: index for index, x in enumerate(xs)}
    yi = {y: index for index, y in enumerate(ys)}
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.7), constrained_layout=True)
    for axis, (title, rows, limits) in zip(axes, rows_by_map, strict=True):
        grid = np.full((len(ys), len(xs)), np.nan)
        values = []
        for row in rows:
            point = row.get("point")
            if row["status"] != "validated_pass" or point is None:
                continue
            value = 1000.0 * point["splitting_ghz"]
            values.append(value)
            grid[yi[row["ec2_target_mhz"]], xi[row["ec1_target_mhz"]]] = value
        image = axis.imshow(
            grid,
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=limits[0],
            vmax=limits[1],
            extent=[min(xs) - 2.5, max(xs) + 2.5, min(ys) - 2.5, max(ys) + 2.5],
            aspect="auto",
        )
        axis.set(
            xlabel=r"$E_{C,1}$ (MHz)",
            ylabel=r"$E_{C,2}$ (MHz)",
            title=f"{title}: audited logical frequency\nfound range {min(values):.2f}–{max(values):.2f} MHz",
            xticks=xs,
            yticks=ys,
        )
        axis.tick_params(axis="x", rotation=45)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.9)
        colorbar.set_label("Logical frequency (MHz)")
    figure.suptitle("Logical frequency on the independent readout-feasible direct-EC hull", fontsize=14)
    output = DATA / "plots"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "readout_logical_frequency_maps.png", dpi=220)
    figure.savefig(output / "readout_logical_frequency_maps.pdf")
    plt.close(figure)
    print(output / "readout_logical_frequency_maps.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
