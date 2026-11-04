"""Fixed-hardware EJ1/EJ2 cavity-frequency maps for separate readout triples.

00=pair ground; 10/01=lower/upper isolated-pair one-excitation branches used
as dressed-state overlap references (NOT unmixed physical product states).
Plot delta fr(state)=fr(state)-fr(bare loaded LC), in MHz, with no extra 2.
No capacitor optimization, coupler, opposite pair or feasibility filtering.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
COUPLER = HERE.parent / "Gnd-coupler+Floating-T"
sys.path[:0] = [str(HERE.parent), str(COUPLER)]


def shifts(point, fr_ghz):
    ground = 1000 * (point["dressed_r_ghz"] - fr_ghz)
    return [
        ground,
        ground + 1000 * point["chi_m_ghz"],
        ground + 1000 * point["chi_p_ghz"],
    ]


def calculate(args):
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    import numpy as np
    from core.transmon import Basis
    from floating_coupler.fabrication import readout_device
    from floating_coupler.design import ej_for_frequency
    from readout_workflow.model import ReadoutModel

    started = perf_counter()
    base = json.loads(args.input.read_text(encoding="utf-8-sig"))
    ro = base["readouts"][args.pair]
    device = readout_device(ro, {})
    basis = Basis(ncut=25, transmon_levels=6, resonator_levels=3, eigenlevels=30)
    model = ReadoutModel(device, basis)
    frequency_bounds = tuple(args.frequency_bounds_ghz)
    axes = [
        np.linspace(
            ej_for_frequency(frequency_bounds[0], ec, 40),
            ej_for_frequency(frequency_bounds[1], ec, 40),
            args.points,
        )
        for ec in model.ec_ghz[:2]
    ]
    maps = np.empty((3, args.points, args.points))
    overlap = np.empty((args.points, args.points))
    f1, f2 = np.empty_like(overlap), np.empty_like(overlap)
    for y, ej2 in enumerate(axes[1]):
        for x, ej1 in enumerate(axes[0]):
            point = model.evaluate(float(ej1), float(ej2))
            maps[:, y, x] = shifts(point, model.fr_ghz)
            overlap[y, x] = point["min_assignment_overlap"]
            f1[y, x], f2[y, x] = point["bare_f1_ghz"], point["bare_f2_ghz"]
        if y % 10 == 0:
            print(
                json.dumps(
                    dict(
                        pair=args.pair,
                        rows_done=y + 1,
                        total=args.points,
                        elapsed_seconds=perf_counter() - started,
                    )
                ),
                flush=True,
            )
    nominal = ro["nominal"]
    idle = [nominal["ej1_ghz"], nominal["ej2_ghz"]]
    idle_point = model.evaluate(*idle)
    high_basis = Basis(ncut=35, transmon_levels=8, resonator_levels=3, eigenlevels=40)
    high = ReadoutModel(device, high_basis)
    audit = []
    for ej1, ej2 in [
        *[
            (axes[0][i], axes[1][j])
            for i in (0, args.points // 2, args.points - 1)
            for j in (0, args.points // 2, args.points - 1)
        ],
        idle,
    ]:
        low, large = (
            model.evaluate(float(ej1), float(ej2)),
            high.evaluate(float(ej1), float(ej2)),
        )
        audit.append(
            dict(
                ej1_ghz=float(ej1),
                ej2_ghz=float(ej2),
                change_khz=(
                    1000
                    * (np.array(shifts(large, high.fr_ghz)) - shifts(low, model.fr_ghz))
                ).tolist(),
                overlap_low=float(low["min_assignment_overlap"]),
                overlap_high=float(large["min_assignment_overlap"]),
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / f"{args.pair}-maps.npz",
        ej1_ghz=axes[0],
        ej2_ghz=axes[1],
        shift_mhz=maps,
        min_assignment_overlap=overlap,
        bare_f1_ghz=f1,
        bare_f2_ghz=f2,
    )
    result = dict(
        pair=args.pair,
        fixed_device=asdict(device),
        basis=asdict(basis),
        audit_basis=asdict(high_basis),
        points=args.points,
        frequency_bounds_ghz=frequency_bounds,
        ej_bounds_ghz=[[float(a[0]), float(a[-1])] for a in axes],
        bare_loaded_resonator_ghz=model.fr_ghz,
        nominal_idle_ej_ghz=idle,
        nominal_shift_mhz=shifts(idle_point, model.fr_ghz),
        nominal_chi_relative_00_mhz=[
            1000 * idle_point["chi_m_ghz"],
            1000 * idle_point["chi_p_ghz"],
        ],
        nominal_overlap=float(idle_point["min_assignment_overlap"]),
        state_order=[
            "00 pair ground",
            "10 lower hybridized branch",
            "01 upper hybridized branch",
        ],
        shift_definition="(E_s,1-E_s,0)/h - loaded uncoupled LC fr; MHz, no factor 2",
        chi_definition="chi10=deltafr10-deltafr00; chi01=deltafr01-deltafr00",
        labeling="Unique pointwise overlap assignment to isolated pair branches x photon 0/1; not a sweep-history adiabatic branch tracker",
        diagnostics="Overlap below 0.95 is marked; all calculated values retained. No readout feasibility thresholds remove pixels.",
        audit=audit,
        elapsed_seconds=perf_counter() - started,
    )
    (args.output / f"{args.pair}-summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def plot(folder, relative00=False):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axs = plt.subplots(
        2,
        2 if relative00 else 3,
        figsize=(11.5 if relative00 else 15, 8.7),
        layout="constrained",
    )
    for row, rail in enumerate(("A", "B")):
        meta = json.loads((folder / f"{rail}-summary.json").read_text())
        data = np.load(folder / f"{rail}-maps.npz")
        z = data["shift_mhz"]
        if relative00:
            z = z[1:] - z[0]
        norm = Normalize(vmin=float(z.min()), vmax=float(z.max()))
        labels = (
            ("10: lower branch", "01: upper branch")
            if relative00
            else ("00: pair ground", "10: lower branch", "01: upper branch")
        )
        for col, label in enumerate(labels):
            ax = axs[row, col]
            mesh = ax.pcolormesh(
                data["ej1_ghz"],
                data["ej2_ghz"],
                z[col],
                shading="auto",
                cmap="viridis",
                norm=norm,
                rasterized=True,
            )
            bad = data["min_assignment_overlap"] < 0.95
            if bad.any():
                ax.contourf(
                    data["ej1_ghz"],
                    data["ej2_ghz"],
                    bad.astype(float),
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["///"],
                )
            ax.plot(
                *meta["nominal_idle_ej_ghz"],
                "*",
                color="white",
                markeredgecolor="black",
                ms=11,
            )
            ax.set(
                xlabel=rf"$E_{{J,{rail}1}}/h$ (GHz)",
                ylabel=rf"$E_{{J,{rail}2}}/h$ (GHz)",
                title=f"{rail} rail | {label}",
            )
        fig.colorbar(
            mesh,
            ax=axs[row, :],
            label=(
                r"$f_r(s)-f_r(00)$ (MHz)"
                if relative00
                else rf"$\delta f_r$ (MHz), relative to {meta['bare_loaded_resonator_ghz']:.6f} GHz"
            ),
            shrink=0.9,
        )
    fig.suptitle(
        (
            "Ground-referenced cavity shifts: 00 subtracted from each state\n"
            if relative00
            else "Fixed nominal readout circuits: state-dependent cavity shifts\n"
        )
        + "Separate pair + resonator; star = nominal idle; hatching = assignment overlap < 0.95",
        fontsize=14,
    )
    stem = "cavity-shifts-relative00-EJ1-EJ2" if relative00 else "cavity-shifts-EJ1-EJ2"
    fig.savefig(folder / (stem + ".png"), dpi=180)
    fig.savefig(folder / (stem + ".svg"))
    plt.close(fig)


def separability(folder):
    """Relative spectral contrast, NOT a measurement-error probability.

    D=|fr10-fr01|, G=(|fr10-fr00|+|fr01-fr00|)/2, R=D/G.
    Absolute values in G avoid a false divergence when mean signed chi vanishes.
    The 1% criterion is the existing chi-matching engineering convention.
    """
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    import numpy as np
    from scipy.optimize import brentq
    from core.transmon import Basis, transmon_spectrum
    from readout_workflow.model import ReadoutModel, Device
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    summary = {}
    for row, rail in enumerate("AB"):
        data = np.load(folder / f"{rail}-maps.npz")
        meta = json.loads((folder / f"{rail}-summary.json").read_text())
        ej1, ej2 = data["ej1_ghz"], data["ej2_ghz"]
        chi = data["shift_mhz"][1:] - data["shift_mhz"][0]
        ground = (abs(chi[0]) + abs(chi[1])) / 2
        delta = abs(chi[0] - chi[1])
        ratio = delta / ground
        model = ReadoutModel(Device(**meta["fixed_device"]), Basis(**meta["basis"]))

        # Refine the narrow equality corridor: a 61-point 2D grid alone can
        # miss the sub-percent strip. Only this contour refinement is new work.
        def contours_at(y):
            cache = {}

            def at(x):
                x = float(x)
                if x not in cache:
                    p = model.evaluate(x, float(y))
                    a, b = p["chi_m_ghz"], p["chi_p_ghz"]
                    cache[x] = (
                        (a - b) / ((abs(a) + abs(b)) / 2),
                        p["min_assignment_overlap"],
                    )
                return cache[x]

            values = [at(x)[0] for x in ej1]
            roots = {}
            for target in (-1.0, -0.01, 0.0, 0.01, 1.0):
                found = []
                for left, right, vl, vr in zip(
                    ej1[:-1], ej1[1:], values[:-1], values[1:]
                ):
                    if (vl - target) * (vr - target) < 0:
                        x = brentq(
                            lambda x: at(x)[0] - target,
                            float(left),
                            float(right),
                            xtol=1e-9,
                        )
                        value, overlap = at(x)
                        if abs(value - target) < 1e-5 and overlap >= 0.95:
                            found.append(float(x))
                roots[str(target)] = found
            return roots

        bands = [contours_at(y) for y in ej2]
        for col, z in enumerate((delta, ratio)):
            ax = axs[row, col]
            lo, hi = (0.001, max(1.0, float(delta.max()))) if col == 0 else (0.001, 2.0)
            im = ax.pcolormesh(
                ej1,
                ej2,
                np.maximum(z, lo),
                shading="auto",
                cmap="viridis",
                norm=LogNorm(lo, hi),
                rasterized=True,
            )
            fig.colorbar(
                im,
                ax=ax,
                label=r"$|f_r(10)-f_r(01)|$ (MHz)"
                if col == 0
                else r"$R=|f_r(10)-f_r(01)|/G$",
            )
            for target in ("-0.01", "0.0", "0.01"):
                points = [
                    (root, float(y)) for y, b in zip(ej2, bands) for root in b[target]
                ]
                if points:
                    points = np.array(points)
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        s=3 if target != "0.0" else 6,
                        c="white",
                        marker=".",
                        linewidths=0,
                    )
            ax.contour(ej1, ej2, ratio, levels=[1.0], colors="black", linewidths=1.4)
            bad = data["min_assignment_overlap"] < 0.95
            if bad.any():
                ax.contourf(
                    ej1,
                    ej2,
                    bad.astype(float),
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["///"],
                )
            ax.plot(
                *meta["nominal_idle_ej_ghz"],
                "*",
                c="white",
                markeredgecolor="black",
                ms=12,
            )
            ax.set(
                xlabel=rf"$E_{{J,{rail}1}}/h$ (GHz)",
                ylabel=rf"$E_{{J,{rail}2}}/h$ (GHz)",
                title=f"{rail}: "
                + (
                    "logical-state frequency gap"
                    if col == 0
                    else "gap relative to average ground contrast"
                ),
            )
        c = np.array(meta["nominal_chi_relative_00_mhz"])
        cut = contours_at(meta["nominal_idle_ej_ghz"][1])
        cut_f = {
            key: [
                float(transmon_spectrum(x, float(model.ec_ghz[0]), model.basis)[0][1])
                for x in xs
            ]
            for key, xs in cut.items()
        }
        summary[rail] = dict(
            nominal_logical_gap_khz=float(1000 * abs(c[0] - c[1])),
            nominal_ground_gaps_mhz=abs(c).tolist(),
            nominal_R=float(abs(c[0] - c[1]) / np.mean(abs(c))),
            nominal_ej2_cut=dict(
                ej2_ghz=meta["nominal_idle_ej_ghz"][1],
                signed_ratio_roots_ej1_ghz=cut,
                signed_ratio_roots_bare_f1_ghz=cut_f,
            ),
            refined_contours=[
                dict(ej2_ghz=float(y), roots=b) for y, b in zip(ej2, bands)
            ],
        )
        print(
            rail,
            {k: v for k, v in summary[rail].items() if k != "refined_contours"},
            flush=True,
        )
    fig.suptitle(
        "Readout frequency contrast: logical 10 versus 01\n"
        "White dots: R <= 1% corridor boundaries/equality; black curve: R = 1; star: nominal idle\n"
        "G = average of the two absolute gaps to 00; hatching: assignment overlap < 0.95",
        fontsize=12,
    )
    fig.savefig(folder / "logical-readout-separability.png", dpi=180)
    fig.savefig(folder / "logical-readout-separability.svg")
    plt.close(fig)
    (folder / "logical-readout-separability.json").write_text(
        json.dumps(summary, indent=2)
    )


def contrast_contours(folder):
    """User-specified absolute chi-gap contours, with refined 10 kHz bands."""
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    import numpy as np
    from scipy.optimize import brentq
    from core.transmon import Basis
    from readout_workflow.model import ReadoutModel, Device
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.9), layout="constrained")
    summary = {}
    for ax, rail in zip(axs, "AB"):
        data = np.load(folder / f"{rail}-maps.npz")
        meta = json.loads((folder / f"{rail}-summary.json").read_text())
        x, y = data["ej1_ghz"], data["ej2_ghz"]
        signed = data["shift_mhz"][1] - data["shift_mhz"][2]
        gap = abs(signed)
        valid = data["min_assignment_overlap"] >= 0.95
        ax.set_facecolor("#eeeeee")
        z = np.ma.masked_where(~valid, gap)
        ax.contourf(
            x,
            y,
            z,
            levels=[0.0, 1.0, max(2.01, float(gap.max()) + 0.01)],
            colors=["white", "#ffe0b2"],
        )
        cs = ax.contour(
            x,
            y,
            z,
            levels=[0.5, 1.0, 2.0],
            colors=["#238b45", "#d94801", "#762a83"],
            linewidths=1.5,
        )
        ax.clabel(cs, inline=True, fmt=lambda v: f"{v:g} MHz", fontsize=9)
        model = ReadoutModel(Device(**meta["fixed_device"]), Basis(**meta["basis"]))
        bands = []
        for j, ej2 in enumerate(y):
            cache = {}

            def at(ej1):
                ej1 = float(ej1)
                if ej1 not in cache:
                    p = model.evaluate(ej1, float(ej2))
                    cache[ej1] = (
                        1000 * (p["chi_m_ghz"] - p["chi_p_ghz"]),
                        p["min_assignment_overlap"],
                    )
                return cache[ej1]

            boundaries = []
            for target in (-0.01, 0.01):
                for left, right, vl, vr in zip(
                    x[:-1], x[1:], signed[j, :-1], signed[j, 1:]
                ):
                    if (vl - target) * (vr - target) < 0:
                        root = brentq(
                            lambda f: at(f)[0] - target,
                            float(left),
                            float(right),
                            xtol=1e-9,
                        )
                        value, overlap = at(root)
                        if abs(value - target) < 1e-6 and overlap >= 0.95:
                            boundaries.append(float(root))
            edges = [float(x[0]), *sorted(boundaries), float(x[-1])]
            row = []
            for left, right in zip(edges[:-1], edges[1:]):
                val, overlap = at((left + right) / 2)
                if abs(val) < 0.01 and overlap >= 0.95:
                    row.append([left, right])
            bands.append(row)
        for branch in range(max(map(len, bands), default=0)):
            lower = [row[branch][0] if len(row) > branch else np.nan for row in bands]
            upper = [row[branch][1] if len(row) > branch else np.nan for row in bands]
            ax.fill_betweenx(y, lower, upper, color="#2166ac", zorder=3)
            ax.plot(lower, y, color="#2166ac", lw=0.6, zorder=3)
            ax.plot(upper, y, color="#2166ac", lw=0.6, zorder=3)
        ax.plot(
            *meta["nominal_idle_ej_ghz"],
            "*",
            c="#ffed00",
            markeredgecolor="black",
            ms=13,
            zorder=5,
        )
        ax.set(
            xlim=(x[0], x[-1]),
            ylim=(y[0], y[-1]),
            xlabel=rf"$E_{{J,{rail}1}}/h$ (GHz)",
            ylabel=rf"$E_{{J,{rail}2}}/h$ (GHz)",
            title=f"Pair {rail}",
        )
        summary[rail] = dict(
            nominal_gap_khz=1000 * abs(np.diff(meta["nominal_chi_relative_00_mhz"])[0]),
            matching_bands=[
                dict(ej2_ghz=float(ej2), intervals_ej1_ghz=row)
                for ej2, row in zip(y, bands)
            ],
        )
    fig.suptitle(
        r"Logical readout contrast: $|\chi_{01}-\chi_{10}|$"
        + "\nOrange: >1 MHz; blue: <10 kHz",
        fontsize=14,
    )
    fig.legend(
        handles=[
            Patch(fc="#2166ac", label="<10 kHz"),
            Patch(fc="#ffe0b2", label=">1 MHz"),
            Patch(fc="#eeeeee", label="Assignment overlap <95%"),
            Line2D(
                [],
                [],
                marker="*",
                color="none",
                markerfacecolor="#ffed00",
                markeredgecolor="black",
                ms=12,
                label="Nominal idle",
            ),
        ],
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.savefig(folder / "logical-readout-chi-contours.png", dpi=190)
    fig.savefig(folder / "logical-readout-chi-contours.svg")
    plt.close(fig)
    (folder / "logical-readout-chi-contours.json").write_text(
        json.dumps(summary, indent=2)
    )


def matching_band_tracks(bands, max_step):
    """Connect neighboring row intervals without joining disappearing branches."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    tracks = []
    active = []
    for row, band in enumerate(bands):
        current = band["intervals_ej1_ghz"]
        matched = {}
        if active and current:
            previous = [np.mean(tracks[t][row - 1]) for t in active]
            cost = abs(np.array(previous)[:, None] - np.mean(current, axis=1))
            ii, jj = linear_sum_assignment(cost)
            matched = {
                int(j): active[i] for i, j in zip(ii, jj) if cost[i, j] <= max_step
            }
        active = []
        for j, interval in enumerate(current):
            if j in matched:
                index = matched[j]
            else:
                index = len(tracks)
                tracks.append(np.full((len(bands), 2), np.nan))
            tracks[index][row] = interval
            active.append(index)
    return tracks


def three_state_contours(
    folder, threshold_mhz=1.0, frequency_axes=False, dual_threshold=False
):
    """All three absolute pairwise gaps, shared >1 MHz intersection shading."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from core.transmon import Basis, transmon_spectrum
    from readout_workflow.model import ReadoutModel, Device

    matching = json.loads((folder / "logical-readout-chi-contours.json").read_text())
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    summary = {}
    suffix = (
        ""
        if threshold_mhz == 1.0
        else "-all-gt-" + f"{threshold_mhz:g}".replace(".", "p") + "MHz"
    )
    if dual_threshold:
        suffix = "-all-gt-0p5-and-1MHz"
    if frequency_axes:
        suffix += "-bare-f01"
    for row, rail in enumerate("AB"):
        d = np.load(folder / f"{rail}-maps.npz")
        meta = json.loads((folder / f"{rail}-summary.json").read_text())
        x, y = d["ej1_ghz"], d["ej2_ghz"]
        source_x, source_y = x.copy(), y.copy()
        nominal = meta["nominal_idle_ej_ghz"]

        def coordinate(value, index):
            return value

        if frequency_axes:
            ecs = meta.get("loaded_transmon_ec_ghz")
            if ecs is None:
                model = ReadoutModel(
                    Device(**meta["fixed_device"]), Basis(**meta["basis"])
                )
                ecs = model.ec_ghz[:2]

            def coordinate(value, index):
                return (
                    float(
                        transmon_spectrum(
                            float(value),
                            float(ecs[index]),
                            Basis(ncut=40, transmon_levels=3),
                        )[0][1]
                    )
                    if np.isfinite(value)
                    else np.nan
                )

            x = np.array([coordinate(v, 0) for v in source_x])
            y = np.array([coordinate(v, 1) for v in source_y])
            nominal = [coordinate(v, i) for i, v in enumerate(nominal)]
        if "pairwise_gaps_mhz" in d:
            gaps = list(d["pairwise_gaps_mhz"])
        else:
            chi10, chi01 = d["shift_mhz"][1:] - d["shift_mhz"][0]
            gaps = [abs(chi01 - chi10), abs(chi01), abs(chi10)]
        minimum = np.minimum.reduce(gaps)
        valid = d["min_assignment_overlap"] >= 0.95
        joint = (minimum > threshold_mhz) & valid
        for col, (gap, title) in enumerate(
            zip(
                gaps,
                [
                    r"$|\chi_{01}-\chi_{10}|$",
                    r"$|\chi_{01}-\chi_{00}|$",
                    r"$|\chi_{10}-\chi_{00}|$",
                ],
            )
        ):
            ax = axs[row, col]
            ax.set_facecolor("#eeeeee")
            ax.contourf(
                x,
                y,
                np.ma.masked_where(~valid, minimum),
                levels=(
                    [0.0, 0.5, 1.0, max(1.01, float(minimum.max()) + 0.01)]
                    if dual_threshold
                    else [
                        0.0,
                        threshold_mhz,
                        max(threshold_mhz + 0.01, float(minimum.max()) + 0.01),
                    ]
                ),
                colors=(
                    ["white", "#ffcc80", "#d95f9e"]
                    if dual_threshold
                    else ["white", "#ffcc80"]
                ),
            )
            contours = ax.contour(
                x,
                y,
                np.ma.masked_where(~valid, gap),
                levels=[0.5, 1.0, 2.0],
                colors=["#238b45", "#d94801", "#762a83"],
                linewidths=1.3,
            )
            ax.clabel(contours, inline=True, fmt=lambda v: f"{v:g} MHz", fontsize=8)
            if col == 0:
                bands = matching[rail]["matching_bands"]
                for track in matching_band_tracks(bands, 3 * max(np.diff(source_x))):
                    lower, upper = track.T
                    yy = [q["ej2_ghz"] for q in bands]
                    if frequency_axes:
                        lower = [coordinate(v, 0) for v in lower]
                        upper = [coordinate(v, 0) for v in upper]
                        yy = [coordinate(v, 1) for v in yy]
                    ax.fill_betweenx(yy, lower, upper, color="#2166ac", zorder=3)
                    ax.plot(lower, yy, c="#2166ac", lw=0.5)
                    ax.plot(upper, yy, c="#2166ac", lw=0.5)
            ax.plot(
                *nominal,
                "*",
                color="#ffed00",
                markeredgecolor="black",
                ms=12,
                zorder=5,
            )
            ax.set(
                xlim=(x[0], x[-1]),
                ylim=(y[0], y[-1]),
                title=f"Pair {rail}: " + title,
                xlabel=rf"Bare $f_{{01,{rail}1}}$ (GHz)"
                if frequency_axes
                else rf"$E_{{J,{rail}1}}/h$ (GHz)",
                ylabel=rf"Bare $f_{{01,{rail}2}}$ (GHz)"
                if frequency_axes
                else rf"$E_{{J,{rail}2}}/h$ (GHz)",
            )
        np.savez_compressed(
            folder / f"{rail}-three-state-contrast{suffix}.npz",
            ej1_ghz=source_x,
            ej2_ghz=source_y,
            plot_x=x,
            plot_y=y,
            pairwise_gaps_mhz=gaps,
            minimum_gap_mhz=minimum,
            all_three_above_threshold=joint,
            all_three_gt_0p5mhz=(minimum > 0.5) & valid,
            all_three_gt_1mhz=(minimum > 1.0) & valid,
            threshold_mhz=threshold_mhz,
            assignment_valid=valid,
        )
        j, i = np.unravel_index(
            np.argmax(np.where(valid, minimum, -np.inf)), minimum.shape
        )
        summary[rail] = dict(
            threshold_mhz=threshold_mhz,
            valid_all_three_above_threshold_grid_points=int(joint.sum()),
            best_minimum_gap_mhz=float(minimum[j, i]),
            best_grid_ej_ghz=[float(source_x[i]), float(source_y[j])],
            axes="bare f01 GHz" if frequency_axes else "EJ/h GHz",
            shading_thresholds_mhz=[0.5, 1.0] if dual_threshold else [threshold_mhz],
            source_grid_points=len(x),
            note="Saved-grid contours; no new optimization or feasibility screening beyond state assignment",
        )
    fig.suptitle(
        meta.get("plot_scope", "Three-state cavity readout contrasts")
        + "\n"
        + (
            "Orange: all three gaps >0.5 MHz; pink: all three >1 MHz"
            if dual_threshold
            else f"Orange shading: ALL three pairwise gaps >{threshold_mhz:g} MHz (same region repeated across each row)"
        ),
        fontsize=14,
    )
    fig.legend(
        handles=[
            Patch(
                fc="#ffcc80",
                label="All three >0.5 MHz"
                if dual_threshold
                else f"All three gaps >{threshold_mhz:g} MHz",
            ),
            *(
                [Patch(fc="#d95f9e", label="All three >1 MHz")]
                if dual_threshold
                else []
            ),
            Patch(
                fc="#2166ac",
                label=meta.get("matching_label", "Logical gap <10 kHz (left column)"),
            ),
            Patch(fc="#eeeeee", label="Assignment overlap <95%"),
            Line2D(
                [],
                [],
                marker="*",
                color="none",
                markerfacecolor="#ffed00",
                markeredgecolor="black",
                ms=12,
                label="Nominal idle",
            ),
        ],
        loc="outside lower center",
        ncol=3 if dual_threshold else 4,
        frameon=False,
        fontsize=9,
    )
    fig.savefig(
        folder / ("three-state-readout-chi-contours" + suffix + ".png"), dpi=190
    )
    fig.savefig(folder / ("three-state-readout-chi-contours" + suffix + ".svg"))
    # Portrait report pages stack individual axes; retain full-size labels.
    fig.canvas.draw()
    fig.set_layout_engine(None)
    for legend in fig.legends:
        legend.set_visible(False)
    renderer = fig.canvas.get_renderer()
    for row, rail in enumerate("AB"):
        for col, name in enumerate(("logical", "01-ground", "10-ground")):
            box = (
                axs[row, col]
                .get_tightbbox(renderer)
                .transformed(fig.dpi_scale_trans.inverted())
            )
            fig.savefig(
                folder / f"{rail}-contrast-{name}.png",
                bbox_inches=box.expanded(1.02, 1.025),
                dpi=190,
            )
    plt.close(fig)
    (folder / ("three-state-readout-chi-contours" + suffix + ".json")).write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pair", choices=["A", "B"])
    parser.add_argument("--points", type=int, default=61)
    parser.add_argument(
        "--frequency-bounds-ghz",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Required for a new calculation: supplied bare-f01 tuning bounds for this pair",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--frequency-axes",
        action="store_true",
        help="Use full-cosine bare f01 coordinates on three-state maps",
    )
    parser.add_argument(
        "--dual-threshold",
        action="store_true",
        help="Shade the all-three >0.5 and >1 MHz regions separately",
    )
    parser.add_argument(
        "--all-gap-threshold-mhz",
        type=float,
        default=1.0,
        help="Shared threshold for all three gaps in contour shading",
    )
    parser.add_argument(
        "--three-state-contours",
        action="store_true",
        help="All three pairwise chi gaps and their shared >1 MHz region, using saved data",
    )
    parser.add_argument(
        "--contrast-contours",
        action="store_true",
        help="Absolute chi gap: <10 kHz band and 0.5/1/2 MHz contours",
    )
    parser.add_argument(
        "--separability",
        action="store_true",
        help="Compare 10/01 frequency contrast and refine chi-matching contours",
    )
    parser.add_argument(
        "--relative00",
        action="store_true",
        help="Plot only 10/01 with 00 subtracted; uses saved maps",
    )
    args = parser.parse_args()
    if args.three_state_contours:
        if args.all_gap_threshold_mhz <= 0:
            parser.error("The gap threshold must be positive")
        three_state_contours(
            args.output,
            args.all_gap_threshold_mhz,
            args.frequency_axes,
            args.dual_threshold,
        )
    elif args.contrast_contours:
        contrast_contours(args.output)
    elif args.separability:
        separability(args.output)
    elif args.plot:
        plot(args.output, args.relative00)
    else:
        if not args.input or not args.pair or args.points < 3:
            parser.error("Require --input, --pair and at least three grid points")
        if (
            args.frequency_bounds_ghz is None
            or not 0 < args.frequency_bounds_ghz[0] < args.frequency_bounds_ghz[1]
        ):
            parser.error("Require positive increasing --frequency-bounds-ghz LOW HIGH")
        calculate(args)
