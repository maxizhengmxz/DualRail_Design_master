"""Random-only explanatory robustness graphics; no circuit recalculation."""

from pathlib import Path
import csv, json
import numpy as np
from scipy.stats import spearmanr, fisher_exact
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def name(s):
    return (
        s.replace("ground:", "Ground ")
        .replace("self:", "Self ")
        .replace("resonator:", "Resonator ")
        .replace("link:", "Link ")
        .replace(":", "–")
    )


def bh(p):
    order = np.argsort(p)
    q = np.ones(len(p))
    v = 1.0
    for k in range(len(order) - 1, -1, -1):
        v = min(v, p[order[k]] * len(p) / (k + 1))
        q[order[k]] = v
    return q


def build(output):
    OUT = Path(output)
    rows = [
        r
        for r in csv.DictReader((OUT / "all-points.csv").open(encoding="utf-8"))
        if r["kind"] in ["random-all", "new-random-all", "independent-all"]
        and r["pending"] == "False"
    ]
    errors = [json.loads(r["errors_json"]) for r in rows]
    features = sorted(set(k for e in errors for k in e))
    X = 100 * np.array([[e.get(f, 0.0) for f in features] for e in errors])
    raw = np.array([float(r["raw"]) for r in rows])
    sel = np.array([float(r["selected"]) for r in rows])
    Bfail = np.array([r["B_pass"] == "False" for r in rows])
    available = np.isfinite(raw) & np.isfinite(sel)
    ys = [
        Bfail.astype(float),
        1 - raw,
        1 - sel,
        np.where(available, (sel < 0.999).astype(float), np.nan),
        (~available).astype(float),
    ]
    labels = [
        "B readout\nrejected",
        "Raw\ninfidelity",
        "Selected\ninfidelity",
        "Selected\n<99.9%",
        "Gate result\nunavailable",
    ]
    R = np.zeros((len(features), len(ys)))
    Q = np.ones_like(R)
    for j, y in enumerate(ys):
        ok = np.isfinite(y)
        p = []
        for i in range(len(features)):
            rho, pv = spearmanr(X[ok, i], y[ok])
            R[i, j] = 0 if not np.isfinite(rho) else rho
            p.append(1.0 if not np.isfinite(pv) else pv)
        Q[:, j] = bh(np.array(p))
    order = np.argsort(-np.max(abs(R), axis=1))
    plt.rcParams.update({"font.size": 10})
    top = order[:8]
    fig, axs = plt.subplots(1, 3, figsize=(12, 6.7), sharey=True, layout="constrained")
    profile_counts = {}
    for ax, y, key, title in zip(
        axs,
        [Bfail, raw < 0.99, sel < 0.999],
        ["B_readout", "raw", "selected"],
        [
            "B-readout outcome",
            "Raw fidelity: 99% split",
            "Selected fidelity: 99.9% split",
        ],
    ):
        valid = np.ones(len(rows), bool) if key == "B_readout" else available
        profile_counts[key] = dict(
            good=int(sum(valid & ~y)),
            bad=int(sum(valid & y)),
            unavailable_excluded=int(sum(~valid)),
        )
        for adverse, c, offset, label in [
            (False, "#2c7f97", -0.12, "Accepted / higher fidelity"),
            (True, "#b6423a", 0.12, "Rejected / lower fidelity"),
        ]:
            group = valid & (y == adverse)
            vals = X[group][:, top]
            if not len(vals):
                ax.plot([], [], color=c, label=label + " (n=0)")
                continue
            quart = np.percentile(vals, [25, 50, 75], axis=0)
            ax.errorbar(
                quart[1],
                np.arange(len(top)) + offset,
                xerr=np.stack([quart[1] - quart[0], quart[2] - quart[1]]),
                fmt="o",
                ms=4,
                c=c,
                capsize=3,
                label=label + f" (n={sum(group)})",
            )
        ax.axvline(0, c="gray", ls=":", lw=0.8)
        ax.set(xlim=(-5, 5), xlabel="Capacitance deviation (%)", title=title)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        ax.grid(alpha=0.2, axis="x")
    axs[0].set_yticks(range(len(top)), [name(features[i]) for i in top])
    axs[0].invert_yaxis()
    fig.suptitle(
        "Conditional deviation profiles: median and interquartile range", fontsize=12
    )
    fig.savefig(OUT / "random-success-failure-profiles.png", dpi=200)
    fig.savefig(OUT / "random-success-failure-profiles.svg")
    plt.close(fig)

    def pairmap(xkey, ykey, stem, title):
        xx = X[:, features.index(xkey)]
        yy = X[:, features.index(ykey)]
        limit = max(0.25, float(np.ceil(np.max(np.abs(X)) * 4) / 4))
        edges = np.linspace(-limit, limit, 6)
        shape = (5, 5)
        count = np.zeros(shape, int)
        BF = np.zeros(shape)
        RM = np.full(shape, np.nan)
        SM = RM.copy()
        RB = RM.copy()
        SB = RM.copy()
        NA = np.zeros(shape)
        G = np.zeros(shape, int)
        for i in range(5):
            for j in range(5):
                mask = (
                    (xx >= edges[i])
                    & (xx < (edges[i + 1] if i < 4 else edges[i + 1] + 1e-8))
                    & (yy >= edges[j])
                    & (yy < (edges[j + 1] if j < 4 else edges[j + 1] + 1e-8))
                )
                count[j, i] = sum(mask)
                finite = mask & available
                G[j, i] = sum(finite)
                if count[j, i]:
                    BF[j, i] = 100 * np.mean(Bfail[mask])
                    NA[j, i] = 100 * np.mean(~available[mask])
                if G[j, i]:
                    RM[j, i] = 100 * np.median(raw[finite])
                    SM[j, i] = 100 * np.median(sel[finite])
                    RB[j, i] = 100 * np.mean(raw[finite] < 0.99)
                    SB[j, i] = 100 * np.mean(sel[finite] < 0.999)
        fig, axs = plt.subplots(2, 3, figsize=(13, 8.8), layout="constrained")
        arrays = [BF, RM, SM, RB, SB, NA]
        titles = [
            "B-readout rejection fraction",
            "Median raw fidelity",
            "Median selected fidelity",
            "Raw fidelity <99% fraction",
            "Selected fidelity <99.9% fraction",
            "Gate result unavailable fraction",
        ]
        for k, (ax, z, tt) in enumerate(zip(axs.ravel(), arrays, titles)):
            n = count if k in [0, 5] else G
            masked = np.ma.masked_where((n < 10) | ~np.isfinite(z), z)
            if k in [1, 2]:
                low = min(np.nanmin(z), 99.0 if k == 1 else 99.9)
                low = np.floor(low * 100) / 100
                vmin, vmax = low, 100.0
                cm = "viridis"
            else:
                vmin, vmax = 0.0, 100.0
                cm = "magma_r"
            im = ax.pcolormesh(
                edges, edges, masked, vmin=vmin, vmax=vmax, cmap=cm, shading="flat"
            )
            fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02, label="%")
            ax.set(
                title=tt,
                xlabel=name(xkey) + " deviation (%)",
                ylabel=name(ykey) + " deviation (%)",
                xlim=(edges[0], edges[-1]),
                ylim=(edges[0], edges[-1]),
            )
            ax.set_xticks(edges)
            ax.set_yticks(edges)
            ax.set_facecolor("#dddddd")
            for j in range(5):
                for i in range(5):
                    if n[j, i] >= 10:
                        text = "n=" + str(n[j, i])
                        color = (
                            "white"
                            if (k not in [1, 2] and z[j, i] > 50)
                            or (k in [1, 2] and z[j, i] < (vmin + vmax) / 2)
                            else "black"
                        )
                        ax.text(
                            (edges[i] + edges[i + 1]) / 2,
                            (edges[j] + edges[j + 1]) / 2,
                            text,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=color,
                        )
        fig.suptitle(title + f" | n={len(rows)} independent random draws", fontsize=12)
        fig.savefig(OUT / (stem + ".png"), dpi=200)
        fig.savefig(OUT / (stem + ".svg"))
        plt.close(fig)
        return dict(
            x=xkey,
            y=ykey,
            total_count=count.tolist(),
            finite_gate_count=G.tolist(),
            B_rejection_percent=BF.tolist(),
            median_raw_percent=RM.tolist(),
            median_selected_percent=SM.tolist(),
            raw_low_percent=RB.tolist(),
            selected_low_percent=SB.tolist(),
            gate_unavailable_percent=NA.tolist(),
            edges_percent=edges.tolist(),
        )

    readout = pairmap(
        "resonator:r2",
        "self:B1",
        "random-readout-capacitor-map",
        "B-readout association: resonator capacitance versus B1 self-capacitance",
    )
    gate = pairmap(
        "link:A12:A21",
        "self:B2",
        "random-gate-capacitor-map",
        "Gate association: A intra-rail link versus B2 self-capacitance",
    )
    fcorr = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(fcorr, 0)
    strongest = []
    for j, label in enumerate(labels):
        ii = np.argsort(-abs(R[:, j]))[:3]
        strongest.append(
            dict(
                outcome=label.replace("\n", " "),
                features=[
                    dict(feature=features[i], rho=float(R[i, j]), q=float(Q[i, j]))
                    for i in ii
                ],
            )
        )
    cont = np.zeros((2, 2), int)
    for b, lo in zip(Bfail[available], sel[available] < 0.999):
        cont[int(b), int(lo)] += 1
    odds, pvalue = fisher_exact(cont)
    meta = dict(
        random_only_readout_vs_low_selected=dict(
            contingency=cont.tolist(), odds_ratio=float(odds), fisher_p=float(pvalue)
        ),
        random_sample_count=len(rows),
        unique_error_vectors=len(
            set(tuple(e.get(k, 0) for k in features) for e in errors)
        ),
        B_readout_rejected=int(sum(Bfail)),
        gate_unavailable=int(sum(~available)),
        low_raw_below99=int(sum(available & (raw < 0.99))),
        low_selected_below99p9=int(sum(available & (sel < 0.999))),
        profile_counts=profile_counts,
        largest_input_Pearson_abs=float(np.max(abs(fcorr))),
        strongest=strongest,
        readout_pair=readout,
        gate_pair=gate,
        notes="All-capacitor independent draws only. Deterministic, link-only, targeted draws and rescues excluded. Gate correlations include finite diagnostics and rejected-readout positions; missing gates kept separate. Visual pair choices selected from same data, exploratory only. All fidelity thresholds descriptive, not acceptance cuts. Median selected differences smaller than0.01percentage point are below agreed numerical precision.",
    )
    (OUT / "random-correlation-visualization-data.json").write_text(
        json.dumps(meta, indent=2)
    )
    return meta
