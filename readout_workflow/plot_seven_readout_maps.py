"""Export and plot completed seven-mode grids; never launch a quantum solve."""

import argparse
import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from readout_workflow.readout_shift_map import three_state_contours


def intervals(x, values, overlap):
    """Linear grid interpolation, explicitly a screening contour estimate."""
    edges = [float(x[0]), float(x[-1])]
    for target in (-0.01, 0.01):
        for i in range(len(x) - 1):
            a, b = values[i] - target, values[i + 1] - target
            if a * b < 0 and min(overlap[i], overlap[i + 1]) >= 0.95:
                edges.append(float(x[i] + (x[i + 1] - x[i]) * (-a) / (b - a)))
    edges.sort()
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = (lo + hi) / 2
        if abs(np.interp(mid, x, values)) < 0.01 and np.interp(mid, x, overlap) >= 0.95:
            result.append([lo, hi])
    return result


def main():
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    caches = {}
    summaries = {}
    for rail in "AB":
        caches[rail] = np.load(a.output / f"{rail}-seven-map.npz")
        assert caches[rail]["done"].all(), "Map is incomplete"
        summaries[rail] = json.loads(
            (a.output / f"{rail}-seven-summary.json").read_text()
        )
    for spectator in (0, 1, "both"):
        folder = a.output / (
            f"spectator-{spectator}L" if spectator != "both" else "both-spectators"
        )
        folder.mkdir(exist_ok=True)
        matching = {}
        for rail in "AB":
            d = caches[rail]
            m = summaries[rail]
            x, y = d["ej1_ghz"], d["ej2_ghz"]
            s = d["shift_mhz"]
            chi = s[:, 1:] - s[:, 0, None]
            gaps = np.stack(
                (abs(chi[:, 1] - chi[:, 0]), abs(chi[:, 1]), abs(chi[:, 0])), axis=1
            )
            band_by_spectator = []
            for spec in (0, 1):
                band_by_spectator.append(
                    [
                        intervals(
                            x,
                            chi[spec, 0, j] - chi[spec, 1, j],
                            np.minimum(
                                d["overlap"][spec, j], d["matter_overlap"][spec, j]
                            ),
                        )
                        for j in range(len(y))
                    ]
                )
            if spectator == "both":
                z = gaps.min(axis=0)
                weight = np.minimum(d["overlap"], d["matter_overlap"]).min(axis=0)
                bands = []
                for b0, b1 in zip(*band_by_spectator):
                    bands.append(
                        sorted(
                            [
                                [max(lo0, lo1), min(hi0, hi1)]
                                for lo0, hi0 in b0
                                for lo1, hi1 in b1
                                if max(lo0, lo1) < min(hi0, hi1)
                            ]
                        )
                    )
                scope = f"Seven modes: thresholds hold for BOTH logical spectator states; {len(x)}×{len(y)} grid"
            else:
                z = gaps[spectator]
                weight = np.minimum(
                    d["overlap"][spectator], d["matter_overlap"][spectator]
                )
                bands = band_by_spectator[spectator]
                scope = f"Seven modes: spectator {spectator}L; fixed nominal OFF EJ; {len(x)}×{len(y)} grid"
            np.savez_compressed(
                folder / f"{rail}-maps.npz",
                ej1_ghz=x,
                ej2_ghz=y,
                pairwise_gaps_mhz=z,
                min_assignment_overlap=weight,
            )
            meta = dict(
                m,
                plot_scope=scope,
                matching_label="Estimated logical gap <10 kHz (left)",
                assignment_display="Minimum of dressed-matter/product and seven-mode/matter assignment weights; underlying overlaps retained separately",
                interpolation_note="Narrow 10 kHz bands estimated by linear grid interpolation; not a calibrated narrow-band boundary",
            )
            (folder / f"{rail}-summary.json").write_text(json.dumps(meta, indent=2))
            matching[rail] = dict(
                matching_bands=[
                    dict(ej2_ghz=float(ej2), intervals_ej1_ghz=b)
                    for ej2, b in zip(y, bands)
                ]
            )
        (folder / "logical-readout-chi-contours.json").write_text(
            json.dumps(matching, indent=2)
        )
        three_state_contours(folder, 0.5, frequency_axes=True, dual_threshold=True)
        print("Plotted", spectator, flush=True)
    # Compact numeric summary includes both spectator states at nominal idle.
    nominal = {}
    for rail, m in summaries.items():
        s = np.array(m["nominal_shift_mhz"])
        chi = s[:, 1:] - s[:, 0, None]
        nominal[rail] = dict(
            chi_relative_ground_mhz=chi.tolist(),
            logical_gap_khz=(1000 * abs(chi[:, 1] - chi[:, 0])).tolist(),
            assignment_overlap=m["nominal_overlap"],
            matter_reference_overlap=m["nominal_matter_reference_overlap"],
            max_C3_C5_shift_change_khz=max(
                abs(v)
                for row in m["audit"]
                for values in row["change_khz"]
                for v in values
            ),
        )
    (a.output / "nominal-summary.json").write_text(json.dumps(nominal, indent=2))
    print(json.dumps(nominal, indent=2))


if __name__ == "__main__":
    main()
