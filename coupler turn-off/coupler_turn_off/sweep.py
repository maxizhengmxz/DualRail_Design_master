"""Optional stage 6: bare-frequency maps with B fixed and A1 rebalanced.

This characterization does not enforce/reoptimize readout at every point.
State-overlap masks indicate where a four-state logical interpretation fails.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from coupler_turn_off.engine import objective_mhz, save
from readout_workflow.calculation import Basis, _transmon


def f01(ej, ec):
    """Exact local transmon transition in GHz, before mode hybridization."""
    return float(_transmon(ej, ec, Basis(25, 8, 3, 12))[0][1])


def ej_for_frequency(frequency, ec, bounds):
    return float(brentq(lambda ej: f01(ej, ec) - frequency, *bounds, xtol=1e-10))


def axis(spec):
    """Explicit [low GHz, high GHz, number of samples]; endpoint inclusive."""
    lo, hi, count = spec
    if lo >= hi or int(count) != count or count < 2:
        raise ValueError("Sweep axes require [low, high, integer count >= 2]")
    return np.linspace(lo, hi, int(count))


def sweep(case, seed, output):
    config = case.config
    spec = config.get("sweep")
    if not isinstance(spec, dict):
        raise ValueError(
            "Optional sweep needs explicit fa_bare_ghz and fc_bare_ghz axes"
        )
    fa, fc = axis(spec["fa_bare_ghz"]), axis(spec["fc_bare_ghz"])
    path = Path(output) / "sweep.json"
    # Refuse to reuse a mismatched grid or operating point. Config identity is
    # already in the output path, and the seed is checked here independently.
    identity = dict(
        seed_ej_ghz=list(seed), fa_bare_ghz=fa.tolist(), fc_bare_ghz=fc.tolist()
    )
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError("Sweep seed/grid differs from the saved checkpoint")
        rows = saved["rows"]
    else:
        rows = []
    done = {(r["ia"], r["ic"]) for r in rows}
    ec1, ec2 = (case.point[k] / 1000 for k in ("loaded_ec1_mhz", "loaded_ec2_mhz"))
    fb = f01(seed[2], ec1)
    for ia, frequency_a in enumerate(fa):
        for ic, frequency_c in enumerate(fc):
            if (ia, ic) in done:
                continue
            row = dict(
                ia=ia,
                ic=ic,
                fa_bare_ghz=float(frequency_a),
                fc_bare_ghz=float(frequency_c),
            )
            try:
                ej = np.asarray(seed, dtype=float).copy()
                ej[1] = ej_for_frequency(frequency_a, ec2, case.bounds[1])
                ej[4] = ej_for_frequency(
                    frequency_c, case.candidate["ec_ghz"], case.bounds[4]
                )

                def negative_balance(value, ej=ej):
                    ej[0] = value
                    return -case.model(ej).evaluate(ej[4])[
                        "single_excitation_a_hybridization"
                    ]

                # Maximize near equal bare rail transitions. A bounded local
                # search is not a proof of a globally maximal hybridization.
                center = ej_for_frequency(frequency_a, ec1, case.bounds[0])
                width = spec["a1_ej_search_half_width_ghz"]
                fit = minimize_scalar(
                    negative_balance,
                    method="bounded",
                    bounds=(
                        max(case.bounds[0, 0], center - width),
                        min(case.bounds[0, 1], center + width),
                    ),
                    options=dict(xatol=2e-6, maxiter=35),
                )
                if not fit.success:
                    raise ValueError("A1 hybridization search did not converge")
                ej[0] = fit.x
                p = case.evaluate(ej, readout=False)
                keys = (
                    "energy_basis_logical_exchange_mhz",
                    "energy_basis_logical_zz_mhz",
                    "physical_signed_exchange_mhz",
                    "physical_sigma_min",
                    "sigma_min",
                    "single_excitation_subspace_sigma_min",
                    "single_excitation_a_hybridization",
                    "leakage_gap_mhz",
                )
                row.update(
                    ej_ghz=ej.tolist(), result={k: p[k] for k in keys}, error=None
                )
            except ValueError as exc:
                row.update(result=None, error=str(exc))
            rows.append(row)
            save(
                path,
                dict(
                    identity=identity,
                    b1_bare_ghz=fb,
                    readout_reoptimized=False,
                    rows=rows,
                ),
            )
        print(f"Sweep row {ia + 1}/{len(fa)} complete", flush=True)


def plot(case, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    output = Path(output)
    saved = json.loads((output / "sweep.json").read_text())
    fa = np.asarray(saved["identity"]["fa_bare_ghz"])
    fc = np.asarray(saved["identity"]["fc_bare_ghz"])
    if len(saved["rows"]) != len(fa) * len(fc):
        raise ValueError("Finish the sweep before plotting its minima")
    rows = [r for r in saved["rows"] if r["result"] is not None]
    reliable = [
        r
        for r in rows
        if min(
            r["result"]["sigma_min"],
            r["result"]["single_excitation_subspace_sigma_min"],
        )
        >= case.config["screens"]["logical_subspace_min"]
    ]
    if not reliable:
        raise ValueError("No reliable logical points in this sweep")
    physical_reliable = [
        r
        for r in rows
        if r["result"]["physical_sigma_min"] >= case.config["screens"]["tracking_min"]
    ]
    if not physical_reliable:
        raise ValueError("No reliable reduced physical-port points in this sweep")
    # Sampled minima are numerical candidates, not continuous optima. Save
    # their larger-basis checks rather than claiming sub-Hz precision.
    minima = []
    for kind in ("physical", "logical", "combined"):
        pool = physical_reliable if kind == "physical" else reliable
        best = min(pool, key=lambda r: objective_mhz(r["result"], kind))
        check = case.evaluate(best["ej_ghz"], case.config["audit_basis"], readout=False)
        minima.append(
            dict(
                objective=kind,
                point=best,
                audit_objective_mhz=objective_mhz(check, kind),
                audit_J1_mhz=check["energy_basis_logical_exchange_mhz"],
                audit_ZZ_mhz=check["energy_basis_logical_zz_mhz"],
            )
        )
    save(output / "sweep_minima.json", minima)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), constrained_layout=True)
    keys = (
        "energy_basis_logical_exchange_mhz",
        "energy_basis_logical_zz_mhz",
        "physical_signed_exchange_mhz",
    )
    for panel, (ax, key, title) in enumerate(
        zip(
            axes,
            keys,
            (
                r"Logical exchange $|J_1|$",
                r"Logical ZZ $|\zeta|$",
                r"Physical-port $|J_{\rm port}|$",
            ),
            strict=True,
        )
    ):
        values = np.full((len(fc), len(fa)), np.nan)
        for row in rows:
            p = row["result"]
            valid = (
                p["physical_sigma_min"] >= case.config["screens"]["tracking_min"]
                if panel == 2
                else min(p["sigma_min"], p["single_excitation_subspace_sigma_min"])
                >= case.config["screens"]["logical_subspace_min"]
            )
            if valid:
                values[row["ic"], row["ia"]] = 1000 * abs(p[key])  # MHz -> kHz
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#bfbfbf")
        im = ax.pcolormesh(
            fa,
            fc,
            np.ma.masked_invalid(values),
            norm=LogNorm(1, 1000),
            cmap=cmap,
            shading="auto",
        )
        ax.axvline(saved["b1_bare_ghz"], color="white", ls="--")
        for marker, label, record in zip(
            ("s", "o", "*"), ("P", "L", "Q"), minima, strict=True
        ):
            r = record["point"]
            ax.scatter(
                r["fa_bare_ghz"],
                r["fc_bare_ghz"],
                marker=marker,
                facecolor="white",
                edgecolor="black",
                s=65,
            )
            offset = {"P": (-14, 12), "L": (10, 12), "Q": (10, -16)}[label]
            ax.annotate(
                label,
                (r["fa_bare_ghz"], r["fc_bare_ghz"]),
                xytext=offset,
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", lw=0.5),
            )
        ax.set(
            title=title,
            xlabel=r"$f_{A2}^{\rm bare}$ (GHz)",
            ylabel=r"$f_C^{\rm bare}$ (GHz)",
        )
        ax.set_xlim(fa[0], fa[-1])
        ax.set_ylim(fc[0], fc[-1])
        fig.colorbar(im, ax=ax, label="Magnitude (kHz)", extend="both")
    fig.suptitle(
        "B fixed; A1 rebalanced; readout not reoptimized\nP: physical minimum; L: logical minimum; Q: combined minimum; dashed: bare A2-B1 resonance\nGray: unreliable logical overlap or failed point. Shared color range 1 kHz-1 MHz.",
        fontsize=11,
    )
    fig.savefig(output / "coupling-map.png", dpi=220)
    plt.close(fig)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ic in dict.fromkeys((0, len(fc) // 2, len(fc) - 1)):
        selected = sorted((r for r in rows if r["ic"] == ic), key=lambda r: r["ia"])
        if selected:
            a.plot(
                [r["fa_bare_ghz"] for r in selected],
                [r["ej_ghz"][0] for r in selected],
                "--",
                label=f"A1 at fC={fc[ic]:.3f}",
            )
    selected = sorted((r for r in rows if r["ic"] == 0), key=lambda r: r["ia"])
    a.plot(
        [r["fa_bare_ghz"] for r in selected],
        [r["ej_ghz"][1] for r in selected],
        color="black",
        label="A2",
    )
    selected = sorted((r for r in rows if r["ia"] == 0), key=lambda r: r["ic"])
    b.plot([r["fc_bare_ghz"] for r in selected], [r["ej_ghz"][4] for r in selected])
    a.axvline(saved["b1_bare_ghz"], ls=":", color="gray")
    a.set_xlim(fa[0], fa[-1])
    a.set(xlabel="Bare A2 frequency (GHz)", ylabel="EJ/h (GHz)")
    a.legend(fontsize=8)
    b.set(xlabel="Bare coupler frequency (GHz)", ylabel="Coupler EJ/h (GHz)")
    fig.savefig(output / "ej-frequency.png", dpi=220)
    plt.close(fig)
