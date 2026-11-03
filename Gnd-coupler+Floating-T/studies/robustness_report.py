"""Read-only all-point robustness reporting; never filters out rejected diagnostics."""

import os

for k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[k] = "1"
import json, csv, pathlib, datetime
import numpy as np
from scipy.stats import spearmanr, fisher_exact
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
import pymupdf

ROOT = pathlib.Path(__file__).resolve().parents[1]
from .archive import iter_records


def load(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def get(r, keys, default=np.nan):
    try:
        for k in keys.split("."):
            r = r[k]
        return float(r) if r is not None else default
    except (KeyError, TypeError, ValueError):
        return default


def flat(r):
    d = dict(
        id=r["id"],
        campaign=r.get("_campaign", "additional"),
        origin=r.get("_origin", "new-policy"),
        kind=r["kind"],
        status=r.get("status"),
        pending=r.get("_origin") == "pending",
    )
    for pair in ["A", "B"]:
        ro = r.get("readout", {}).get(pair, {})
        p = ro.get("point") or ro.get("diagnostic_point") or {}
        d[pair + "_pass"] = bool(ro.get("passed"))
        d[pair + "_failures"] = ";".join(ro.get("nominal_failures", []))
        d[pair + "_diagnostic"] = not d[pair + "_pass"]
        cp, cm = get(p, "chi_p_ghz"), get(p, "chi_m_ghz")
        d[pair + "_chi_ratio"] = get(p, "ratio")
        d[pair + "_max_chi_MHz"] = max(abs(cp), abs(cm)) * 1000
        d[pair + "_mean_chi_MHz"] = abs((cp + cm) / 2) * 1000
        d[pair + "_H"] = get(p, "hybridization")
        d[pair + "_lambda"] = get(p, "lambda_max")
        d[pair + "_f_GHz"] = get(p, "operating_frequency_ghz")
        d[pair + "_EJ1_GHz"] = get(p, "ej1_ghz")
    d.update(
        off_found=bool(r.get("off", {}).get("found")),
        gate_found=bool(r.get("gate", {}).get("found")),
        gate_approximate=bool(r.get("gate", {}).get("approximate")),
        off_GHz=get(r, "off.fc_ghz"),
        physical_off_Hz=get(r, "off.metrics.physical_exchange_hz"),
        logical_exchange_Hz=get(r, "off.metrics.logical_exchange_hz"),
        logical_double_flip_Hz=get(r, "off.metrics.logical_double_flip_hz"),
        ZZ_Hz=get(r, "off.metrics.secular_logical_zz_hz"),
        physical_at10_Hz=get(r, "off.reference_10ghz.metrics.physical_exchange_hz"),
        raw=get(r, "gate.best.raw_cz_fidelity"),
        selected=get(r, "gate.best.postselected_cz_fidelity"),
        duration_ns=get(r, "gate.best.time_ns"),
        phase_error_rad=get(r, "gate.best.phase_error_rad"),
        gate_status=r.get("gate", {}).get("status", ""),
    )
    if not np.isfinite(d["off_GHz"]):
        d["off_GHz"] = get(r, "off.best.fc_ghz")
    d["errors_json"] = json.dumps(r.get("errors", {}), sort_keys=True)
    return d


def fmt(v, n=4):
    return "NA" if not np.isfinite(v) else f"{v:.{n}f}"


def unavailable_report(out, summary):
    """Keep failed evaluations visible when no finite gate result exists yet."""
    target = out / (
        "robustness-summary-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        + ".pdf"
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Robustness summary: gate results unavailable", styles["Heading1"]),
        Spacer(1, 12),
        Paragraph(
            "All completed evaluations remain in all-points.csv. No fidelity histogram is drawn because no finite gate result exists; unavailable results are not assigned zero or unit fidelity.",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]
    story.append(Table([[k.replace("_", " "), str(v)] for k, v in summary.items()]))
    SimpleDocTemplate(str(target), pagesize=A4).build(story)
    result = dict(pdf=str(target), pages=1, summary=summary)
    (out / "report-status.json").write_text(json.dumps(result, indent=2))
    return result


def generate(folder):
    folder = pathlib.Path(folder).resolve()
    out = folder / "report"
    out.mkdir(exist_ok=True)
    manifest = load(folder / "campaign.json")
    reuse = manifest.get("reuse_archive")
    rows = list(iter_records(reuse)) if reuse else []
    for p in sorted((folder / "samples").glob("*.json")):
        r = load(p)
        r["_campaign"] = "pooled"
        rows.append(r)
    if not rows:
        raise ValueError("No completed evaluations to report yet")
    flatrows = [flat(r) for r in rows]
    with (out / "all-points.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flatrows[0]))
        w.writeheader()
        w.writerows(flatrows)
    done = [d for d in flatrows if not d["pending"]]
    pending = len(flatrows) - len(done)
    summary = dict(
        total=len(rows),
        processed=len(done),
        pending=pending,
        A_failed=sum(not d["A_pass"] for d in done),
        B_failed=sum(not d["B_pass"] for d in done),
        off_found=sum(d["off_found"] for d in done),
        gate_found=sum(d["gate_found"] for d in done),
        gate_best_available=sum(np.isfinite(d["selected"]) for d in done),
        readout_rejected_with_gate=sum(
            not (d["A_pass"] and d["B_pass"]) and np.isfinite(d["selected"])
            for d in done
        ),
    )
    summary = {k: int(v) for k, v in summary.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if not any(np.isfinite(d["raw"]) and np.isfinite(d["selected"]) for d in done):
        return unavailable_report(out, summary)
    random = [
        (r, d)
        for r, d in zip(rows, flatrows)
        if r["kind"] in ["random-all", "new-random-all", "independent-all"]
        and not d["pending"]
    ]
    features = sorted(set(k for r, d in random for k in r.get("errors", {})))
    X = np.array(
        [[r.get("errors", {}).get(k, 0.0) / 0.05 for k in features] for r, d in random]
    )
    corr = []
    targets = {
        "B_readout_failure": np.array([not d["B_pass"] for r, d in random], float),
        "selected_gate_error": np.array([1 - d["selected"] for r, d in random]),
        "OFF_frequency": np.array([d["off_GHz"] for r, d in random]),
    }
    featurecols = [(k, X[:, i]) for i, k in enumerate(features)]
    featurecols += [
        (features[i] + " * " + features[j], X[:, i] * X[:, j])
        for i in range(len(features))
        for j in range(i + 1, len(features))
    ]
    for target, y in targets.items():
        block = []
        for name, x in featurecols:
            valid = np.isfinite(x) & np.isfinite(y)
            if sum(valid) < 12 or np.ptp(y[valid]) == 0 or np.ptp(x[valid]) == 0:
                continue
            rho, p = spearmanr(x[valid], y[valid])
            block.append(
                dict(
                    target=target,
                    feature=name,
                    rho=float(rho),
                    p=float(p),
                    n=int(sum(valid)),
                )
            )
        order = np.argsort([v["p"] for v in block])
        last = 1.0
        for rank in range(len(order) - 1, -1, -1):
            q = block[order[rank]]
            last = min(last, q["p"] * len(block) / (rank + 1))
            q["q_FDR"] = last
        corr.extend(block)
    if corr:
        with (out / "correlations.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(corr[0]))
            w.writeheader()
            w.writerows(corr)
    styles = getSampleStyleSheet()
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 12
    story = []
    W = A4[0] - 76

    def txt(s):
        story.extend([Paragraph(s, styles["Normal"]), Spacer(1, 8)])

    def title(s):
        story.extend([Paragraph(s, styles["Heading1"]), Spacer(1, 10)])

    def page():
        story.append(PageBreak())

    def table(headers, data, widths=None):
        cells = [
            [Paragraph(str(c), styles["Normal"]) for c in row]
            for row in [headers] + data
        ]
        t = Table(
            cells,
            colWidths=[W / len(headers)] * len(headers) if widths is None else widths,
            repeatRows=1,
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edf4")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.2, colors.lightgrey),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([t, Spacer(1, 9)])

    def plot(fig, name):
        fig.tight_layout()
        f = out / (name + ".png")
        fig.savefig(f, dpi=160)
        plt.close(fig)
        story.append(
            Image(str(f), width=W, height=W * fig.get_figheight() / fig.get_figwidth())
        )
        story.append(Spacer(1, 7))

    title("Capacitance robustness: summary report")
    txt(
        "Updated "
        + datetime.datetime.now(datetime.timezone.utc)
        .astimezone(datetime.timezone(datetime.timedelta(hours=8)))
        .strftime("%Y-%m-%d %H:%M China time")
    )
    txt(
        "Physical capacitor variations under the explicit study manifest. Separate A/B readout triples and five-transmon OFF/CZ recovery. No seven-mode calculation or noise budget. No fabrication-time EC compensation. All saved points are included, including rejected readout positions, imperfect/diagnostic gates and unavailable results. This PDF presents summaries and figures; complete per-record data are saved in all-points.csv. Missing values are NA, never zero."
    )
    table(["Measure", "Count"], [[k.replace("_", " "), v] for k, v in summary.items()])
    txt(
        "All results are pooled into one robustness study. The sample set includes independent capacitor draws, deterministic sweeps and targeted failure rescues. It is intentionally failure-enriched and is not a manufacturing-yield estimate. The same physical component errors are shared across independently synthesized subsystem representations."
    )
    page()
    title("Combined robustness results")
    run = load(folder / "status.json")
    original = {}

    txt(
        "<b>Study status: "
        + run["state"].upper()
        + ".</b> All "
        + str(len(flatrows))
        + " evaluated records are presented together. Pending records: "
        + str(pending)
        + ". Calculation errors: "
        + str(run.get("calculation_errors", 0) + original.get("calculation_errors", 0))
        + "."
    )
    available = [d for d in done if np.isfinite(d["selected"])]
    accepted = [d for d in available if d["gate_found"]]
    diagnostic = [d for d in available if not d["gate_found"]]
    table(
        ["Completed-run outcome", "Records"],
        [
            ["Total evaluated records", len(flatrows)],
            ["Pending records", pending],
            ["A readout rejected", summary["A_failed"]],
            ["B readout rejected", summary["B_failed"]],
            ["Physical OFF accepted", summary["off_found"]],
            ["Gate solution accepted", summary["gate_found"]],
            ["Finite gate values, including diagnostics", len(available)],
            ["Finite diagnostic gate, not accepted", len(diagnostic)],
            ["No finite gate result", len(done) - len(available)],
        ],
    )
    txt(
        "The combined study contains independent random draws, deterministic sweeps, targeted retries and nearby failure-oriented settings. Repeated physical settings are retained as separate evaluation records. All outcomes and distributions are reported for the pooled dataset; rejection counts are NOT fabrication failure probabilities."
    )
    txt(
        "Max-stratified sampling, when requested, uses the bin width and per-bin target recorded in campaign.json and retains supplied archive records. Within each new draw, choose its maximum uniformly inside its assigned bin, a random signed limiting capacitor, and uniform deviations for all other capacitors inside that maximum. Such draws are not unrestricted independent +/-5% samples and are excluded from random-only correlation analyses. Rejected/failed attempts remain in the appendix."
    )
    page()
    title("Combined fidelity and OFF summary")
    stats = []
    for key, label, scale in [
        ("raw", "Raw gate fidelity (%)", 100),
        ("selected", "Postselected gate fidelity (%)", 100),
        ("duration_ns", "Gate duration (ns)", 1),
        ("off_GHz", "Physical OFF frequency (GHz)", 1),
    ]:
        x = np.array([d[key] * scale for d in done])
        x = x[np.isfinite(x)]
        stats.append(
            [label, len(x), *[fmt(v, 4) for v in np.percentile(x, [0, 50, 100])]]
        )
    table(["All finite values", "n", "Minimum", "Median", "Maximum"], stats)
    txt(
        "Fidelity statistics include accepted and diagnostic/fallback gates, and retain results at rejected readout positions. They are not fidelity statistics conditioned on all constraints passing. Accepted phase-pi gate results, finite diagnostics and missing results are separated in the preceding table and flagged per point in the CSV. A high postselected score can coexist with reduced raw fidelity; both are reported."
    )
    txt(
        "The physical OFF criterion is exchange below 1 Hz. Every record has an accepted physical OFF in the saved run, but some require the authorized 10-11 GHz rescue interval. This does not imply cancellation of logical exchange or secular ZZ; those remain separate diagnostics. The maximum OFF position is below 11 GHz."
    )
    txt(
        "The two-dimensional fidelity-density plots and capacitor correlations in the following sections use the completed results. Maximum-deviation sampling clusters near 5% for independent multi-capacitor draws; gray histogram bins are unsampled, not failures. A future tolerance-scale study should use separate deviation budgets with comparable sample counts, rather than interpret this sampled density as a uniform tolerance curve. No new tolerance-budget sweep is launched by this report update."
    )
    page()
    title("Readout: accepted and rejected diagnostics")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pair in zip(axs, ["A", "B"]):
        for accepted, c, label in [
            (True, "#327f91", "Accepted"),
            (False, "#ba443e", "Rejected diagnostic"),
        ]:
            q = [d for d in done if d[pair + "_pass"] == accepted]
            ax.scatter(
                [d[pair + "_mean_chi_MHz"] for d in q],
                [100 * d[pair + "_chi_ratio"] for d in q],
                s=7,
                alpha=0.5,
                c=c,
                label=label,
            )
        ax.axvline(1, c="black", ls="--", lw=0.8)
        ax.axhline(5, c="black", ls=":", lw=0.8)
        ax.set(
            xlabel="Absolute mean chi (MHz)",
            ylabel="Chi mismatch (%)",
            title="Pair " + pair,
        )
        ax.legend(fontsize=7)
    plot(fig, "readout-all")
    txt(
        "The 1 MHz line is the hard mean-chi floor; 5% is the last rescue mismatch limit, not the baseline. Max chi in the appendix is max(|chi+|,|chi-|), and is reported separately from mean chi."
    )
    failures = {}
    for pair in ["A", "B"]:
        for d in done:
            if not d[pair + "_pass"]:
                failures[pair + ": " + d[pair + "_failures"]] = (
                    failures.get(pair + ": " + d[pair + "_failures"], 0) + 1
                )
    table(
        ["Failed constraint set", "Count"],
        sorted(failures.items(), key=lambda v: -v[1])[:12] or [["None", 0]],
    )
    bad = [d for d in done if not d["B_pass"]]
    txt(
        "B-rejected diagnostics at EJ1 >24.9 GHz: "
        + str(sum(d["B_EJ1_GHz"] > 24.9 for d in bad))
        + " / "
        + str(len(bad))
        + ". Such saturation suggests insufficient available readout-strength margin; a bounded miss is not a proof of infeasibility."
    )
    page()
    title("CZ and OFF: all available results")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for accepted, c, label in [
        (True, "#327f91", "Readout accepted"),
        (False, "#ba443e", "Readout rejected"),
    ]:
        q = [d for d in done if (d["A_pass"] and d["B_pass"]) == accepted]
        axs[0].scatter(
            [100 * d["raw"] for d in q],
            [100 * d["selected"] for d in q],
            s=8,
            alpha=0.5,
            c=c,
            label=label,
        )
        axs[1].scatter(
            [d["duration_ns"] for d in q],
            [d["off_GHz"] for d in q],
            s=8,
            alpha=0.5,
            c=c,
            label=label,
        )
    axs[0].set(xlabel="Raw fidelity (%)", ylabel="Selected fidelity (%)")
    axs[1].set(xlabel="Gate duration (ns)", ylabel="Coupler OFF frequency (GHz)")
    axs[1].axhline(10, c="black", ls="--", lw=0.8)
    for ax in axs:
        ax.legend(fontsize=7)
    plot(fig, "gate-off-all")
    stats = []
    for key, label, scale in [
        ("raw", "Raw fidelity (%)", 100),
        ("selected", "Selected fidelity (%)", 100),
        ("duration_ns", "Gate duration (ns)", 1),
        ("off_GHz", "OFF frequency (GHz)", 1),
        ("physical_off_Hz", "Physical OFF residual (Hz)", 1),
        ("logical_exchange_Hz", "Residual logical exchange (Hz)", 1),
        ("ZZ_Hz", "Secular logical ZZ (Hz)", 1),
    ]:
        x = np.array([d[key] * scale for d in done])
        x = x[np.isfinite(x)]
        stats.append(
            [label, *[fmt(v, 4) for v in np.percentile(x, [0, 50, 95, 100])]]
            if len(x)
            else [label, *["NA"] * 4]
        )
    table(["All available", "Min", "Median", "95th pct", "Max"], stats)
    txt(
        "Physical OFF residual below 1 Hz defines OFF acceptance; logical exchange and ZZ remain residual diagnostics. OFF >10 GHz uses the authorized rescue interval below 11 GHz; 10 GHz reference residuals are retained in CSV. Fidelity has no pass threshold; phase-pi and hardware screens determine gate acceptance. Approximate candidates remain flagged."
    )
    page()
    title("Gate error conditioned on maximum deviation")
    from matplotlib.colors import LogNorm

    xedges = np.linspace(0, 5, 21)
    densities = []
    density_meta = []
    error_edges = np.geomspace(1e-6, 0.1, 61)
    log_widths = np.diff(np.log10(error_edges))
    for field in ["raw", "selected"]:
        pairs = [
            (
                100 * max([abs(v) for v in r.get("errors", {}).values()] or [0.0]),
                1 - d[field],
            )
            for r, d in zip(rows, flatrows)
            if not d["pending"] and np.isfinite(d[field])
        ]
        xx, ee = np.array(pairs).T
        assert np.all((ee >= 0) & (ee <= 1)), (
            "Nonpositive or invalid gate error requires explicit handling"
        )
        folded = np.clip(ee, error_edges[0], error_edges[-1])
        counts, xe, ye = np.histogram2d(xx, folded, bins=[xedges, error_edges])
        assert int(counts.sum()) == len(ee)
        low_counts = np.histogram(xx[ee < error_edges[0]], bins=xedges)[0]
        high_counts = np.histogram(xx[ee > error_edges[-1]], bins=xedges)[0]
        n = counts.sum(axis=1)
        hist = np.divide(
            counts,
            n[:, None] * log_widths[None, :],
            out=np.zeros_like(counts),
            where=n[:, None] > 0,
        )
        integrals = np.sum(hist * log_widths[None, :], axis=1)
        assert np.allclose(integrals[n > 0], 1.0)
        densities.append((hist, xe, ye, n))
        density_meta.append(
            dict(
                metric=field,
                n=len(pairs),
                missing=len(flatrows) - len(pairs),
                counts_per_deviation_bin=n.astype(int).tolist(),
                conditional_integrals=integrals.tolist(),
                error_definition="e=1-F, dimensionless",
                error_binning="60 uniform log10(error) display bins from1e-6 to0.1; all underflow mass folded into bottom bin, all overflow mass into top bin",
                underflow_counts_per_deviation_bin=low_counts.tolist(),
                overflow_counts_per_deviation_bin=high_counts.tolist(),
                displayed_error_range=[1e-6, 0.1],
                normalization="p(log10(e)|maximum deviation bin); each supported vertical column integrates to1 over log10(error); units inverse decade",
            )
        )
    positive = np.concatenate([h[h > 0] for h, xe, ye, n in densities])
    decade_low = int(np.floor(np.log10(min(positive))))
    decade_high = int(np.ceil(np.log10(max(positive))))
    decade_high = max(decade_high, decade_low + 1)
    from matplotlib.colors import ListedColormap, to_rgb
    import colorsys

    decade_palette = ["#3753a4", "#158b99", "#398b3e", "#c38a16", "#c34f24", "#9b2868"]
    shade_colors = []
    for k in range(decade_high - decade_low):
        hue, saturation, value = colorsys.rgb_to_hsv(
            *to_rgb(decade_palette[k % len(decade_palette)])
        )
        for f in np.linspace(0, 1, 256):
            shade_colors.append(
                colorsys.hsv_to_rgb(hue, 0.36 + 0.60 * f, 0.99 - 0.25 * f)
            )
    density_cmap = ListedColormap(shade_colors, name="density_decades")
    norm = LogNorm(vmin=10.0**decade_low, vmax=10.0**decade_high)
    fig = plt.figure(figsize=(11, 5.7), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1.2], width_ratios=[1, 1, 0.045])
    axs = [fig.add_subplot(gs[0, i]) for i in range(2)]
    support = [fig.add_subplot(gs[1, i], sharex=axs[i]) for i in range(2)]
    cax = fig.add_subplot(gs[0, 2])
    for ax, sax, (hist, xe, ye, n), meta, label in zip(
        axs, support, densities, density_meta, ["Raw CZ error", "Postselected CZ error"]
    ):
        mesh = ax.pcolormesh(
            xe,
            ye,
            np.ma.masked_where(hist.T <= 0, hist.T),
            norm=norm,
            cmap=density_cmap,
            shading="flat",
        )
        ax.set(
            xlim=(0, 5),
            ylim=(ye[0], ye[-1]),
            yscale="log",
            ylabel="Gate error 1 - F",
            title=label + "; n=" + str(meta["n"]),
        )
        ax.set_yticks(
            [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1],
            labels=["≤10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "≥10⁻¹"],
        )
        ax.set_facecolor("#eeeeee")
        ax.tick_params(labelbottom=False)
        ax.axhline(1e-4, color="#c31924", ls="--", lw=1.15, zorder=8)
        ax.text(
            0.12,
            1.25e-4,
            "Numerical accuracy",
            color="#9f101b",
            fontsize=7.5,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
        )
        centers = (xe[:-1] + xe[1:]) / 2
        sax.bar(centers, n, width=np.diff(xe) * 0.9, color="#467f94")
        sax.set(
            yscale="log",
            xlabel="Maximum capacitor deviation (%)",
            ylabel="Samples",
            ylim=(0.7, max(n) * 3),
        )
        sax.grid(alpha=0.2, axis="y")
        for xc, count in zip(centers, n):
            if count > 0:
                sax.text(
                    xc,
                    max(1.0, count) * 1.08,
                    str(int(count)),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
    cb = fig.colorbar(
        mesh, cax=cax, ticks=10.0 ** np.arange(decade_low, decade_high + 1)
    )
    cb.set_label(
        "Conditional density per error decade\nNew hue each density decade; shades within"
    )
    for decade in range(decade_low + 1, decade_high):
        cb.ax.axhline(10.0**decade, color="white", lw=1.2)
    density_plot = out / "fidelity-vs-max-deviation-pdf.png"
    fig.savefig(density_plot, dpi=200)
    fig.savefig(out / "fidelity-vs-max-deviation-pdf.svg")
    fig.savefig(out / "error-vs-max-deviation-conditional-pdf.png", dpi=200)
    fig.savefig(out / "error-vs-max-deviation-conditional-pdf.svg")
    plt.close(fig)
    story.extend(
        [Image(str(density_plot), width=W, height=W * 5.7 / 11), Spacer(1, 10)]
    )
    (out / "fidelity-density-metadata.json").write_text(
        json.dumps(density_meta, indent=2)
    )
    np.savez(
        out / "fidelity-vs-max-deviation-pdf.npz",
        raw_density=densities[0][0],
        raw_x_edges=densities[0][1],
        raw_error_edges=densities[0][2],
        raw_counts=densities[0][3],
        selected_density=densities[1][0],
        selected_x_edges=densities[1][1],
        selected_error_edges=densities[1][2],
        selected_counts=densities[1][3],
    )
    txt(
        "Vertical axis: gate error e=1-F, displayed logarithmically from 1e-6 (0.0001% error) to 0.1 (10% error), with 60 uniformly log-spaced display bins. ALL probability below 1e-6 is added to the bottom bin; ALL probability above 0.1 is added to the top bin. Those endpoint bins also include their ordinary within-range samples. Thus no finite result is discarded. Original error values in the saved records are unchanged. The dashed red line at 1e-4 marks the agreed absolute fidelity accuracy of 0.01 percentage point; it is not a failure threshold or physical error floor. Exact values below this accuracy are not certified at that tolerance."
    )
    txt(
        "Each supported deviation column integrates to ONE over log10(display error), including both folded tails. Colors show conditional bin probability divided by the display-bin width in decades, on a logarithmic color scale. At the endpoints this is a folded histogram representation, not a claim that all tail errors equal the plotted boundary. The bottom/top bins are cumulative-tail bins. Each factor-of-ten interval in density has its own hue; light-to-dark shades distinguish values within that decade, with white separators at decade boundaries."
    )
    txt(
        "Bottom panels show the sample count per deviation bin, with nonzero counts labeled. Gray means an empty observed cell or unsupported column; count panels distinguish them. Sparse columns remain statistically weak. Every column uses all available finite gate results, including readout-rejected and non-phase-pi diagnostic gates."
    )
    txt(
        "All "
        + str(density_meta[0]["n"])
        + " finite raw and selected gate results contribute. Unavailable results remain NA in the CSV. Maximum deviation is the largest absolute fractional capacitor error relative to nominal, in percent. These empirical samples pool deterministic sweeps, independent draws, targeted rescues and max-stratified draws; conditional normalization does not make them a fabrication probability model."
    )
    page()
    title("Logical residual couplings at coupler OFF")
    fields = [
        ("logical_exchange_Hz", r"$|g_{01_L,10_L}|$"),
        ("logical_double_flip_Hz", r"$|g_{00_L,11_L}|$"),
    ]
    coupling_data = []
    coupling_meta = []
    for field, label in fields:
        pairs = [
            (
                100 * max([abs(v) for v in r.get("errors", {}).values()] or [0.0]),
                abs(d[field]),
            )
            for r, d in zip(rows, flatrows)
            if not d["pending"] and np.isfinite(d[field])
        ]
        xx, gg = np.array(pairs).T
        coupling_data.append((xx, gg))
    allg = np.concatenate([gg for xx, gg in coupling_data])
    gmin = max(1.0, 100 * np.floor(min(allg) / 100))
    gmax = max(gmin * 1.1, 100 * np.ceil(max(allg) / 100))
    gedges = np.geomspace(gmin, gmax, 61)
    gwidths = np.diff(np.log10(gedges))
    gres = []
    for (field, label), (xx, gg) in zip(fields, coupling_data):
        cc, xe, ye = np.histogram2d(xx, np.clip(gg, gmin, gmax), bins=[xedges, gedges])
        nn = cc.sum(axis=1)
        assert int(cc.sum()) == len(gg)
        hh = np.divide(
            cc,
            nn[:, None] * gwidths[None, :],
            out=np.zeros_like(cc),
            where=nn[:, None] > 0,
        )
        assert np.allclose(np.sum(hh * gwidths[None, :], axis=1)[nn > 0], 1.0)
        gres.append((hh, xe, ye, nn))
        coupling_meta.append(
            dict(
                field=field,
                n=len(gg),
                missing=len(flatrows) - len(gg),
                absolute_coupling_hz=dict(
                    minimum=float(min(gg)),
                    median=float(np.median(gg)),
                    maximum=float(max(gg)),
                ),
                counts_per_bin=nn.astype(int).tolist(),
                definition="Absolute effective Hamiltonian matrix element H/h in Hz; no extra2pi or factor2; at each record optimized physical exchange OFF, idle rails at recovered readout position. Signed coefficients retained in CSV.",
                normalization="p(log10(abs(g)Hz)|maximum deviation bin); each supported column integrates1 per logarithmic coupling coordinate",
            )
        )
    positive = np.concatenate([h[h > 0] for h, xe, ye, n in gres])
    dl = int(np.floor(np.log10(min(positive))))
    dh = max(dl + 1, int(np.ceil(np.log10(max(positive)))))
    ccolors = []
    for k in range(dh - dl):
        hue, saturation, value = colorsys.rgb_to_hsv(
            *to_rgb(decade_palette[k % len(decade_palette)])
        )
        for shade in np.linspace(0, 1, 256):
            ccolors.append(
                colorsys.hsv_to_rgb(hue, 0.36 + 0.60 * shade, 0.99 - 0.25 * shade)
            )
    gcmap = ListedColormap(ccolors)
    gnorm = LogNorm(vmin=10.0**dl, vmax=10.0**dh)
    fig = plt.figure(figsize=(11, 5.7), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1.2], width_ratios=[1, 1, 0.045])
    axes = [fig.add_subplot(gs[0, k]) for k in range(2)]
    bottom = [fig.add_subplot(gs[1, k], sharex=axes[k]) for k in range(2)]
    cax = fig.add_subplot(gs[0, 2])
    from matplotlib.ticker import NullFormatter

    for ax, bax, (hh, xe, ye, nn), (field, label), meta in zip(
        axes, bottom, gres, fields, coupling_meta
    ):
        im = ax.pcolormesh(
            xe,
            ye,
            np.ma.masked_where(hh.T <= 0, hh.T),
            cmap=gcmap,
            norm=gnorm,
            shading="flat",
        )
        ax.set(
            xlim=(0, 5),
            ylim=(gmin, gmax),
            yscale="log",
            ylabel="Residual coupling magnitude (Hz)",
            title=label + "; n=" + str(meta["n"]),
        )
        ticks = np.linspace(gmin, gmax, 5)
        ax.set_yticks(ticks, labels=[f"{v:g}" for v in ticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelbottom=False)
        ax.set_facecolor("#eeeeee")
        bax.bar((xe[:-1] + xe[1:]) / 2, nn, width=np.diff(xe) * 0.9, color="#467f94")
        bax.set(
            yscale="log",
            ylim=(0.7, max(nn) * 3),
            xlabel="Maximum capacitor deviation (%)",
            ylabel="Samples",
        )
        for xc, count in zip((xe[:-1] + xe[1:]) / 2, nn):
            if count > 0:
                bax.text(
                    xc,
                    count * 1.08,
                    str(int(count)),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
    cb = fig.colorbar(im, cax=cax, ticks=10.0 ** np.arange(dl, dh + 1))
    cb.set_label(
        "Conditional density per coupling decade\nNew hue each density decade; shades within"
    )
    for decade in range(dl + 1, dh):
        cb.ax.axhline(10.0**decade, color="white", lw=1.2)
    cf = out / "logical-off-couplings-vs-max-deviation-pdf.png"
    fig.savefig(cf, dpi=200)
    fig.savefig(out / "logical-off-couplings-vs-max-deviation-pdf.svg")
    plt.close(fig)
    story.extend([Image(str(cf), width=W, height=W * 5.7 / 11), Spacer(1, 10)])
    (out / "logical-off-couplings-density-metadata.json").write_text(
        json.dumps(coupling_meta, indent=2)
    )
    np.savez(
        out / "logical-off-couplings-density.npz",
        exchange=gres[0][0],
        double_flip=gres[1][0],
        deviation_edges=xedges,
        coupling_edges_hz=gedges,
        counts_exchange=gres[0][3],
        counts_double_flip=gres[1][3],
    )
    txt(
        "Both panels use the saved five-transmon Hamiltonian at EACH sample optimized coupler-OFF bias and recovered rail idle position. Plotted values are absolute logical matrix elements H/h in Hz: 01L-10L exchange and 00L-11L double-flip coupling. They are not twice-the-coupling oscillation frequencies. Signed coefficients remain in all-points.csv."
    )
    txt(
        "OFF is still defined by physical 0100-0010 exchange below 1 Hz; the two LOGICAL couplings are residual diagnostics and need not vanish. Readout-rejected diagnostic positions are included and remain flagged in the CSV. All available OFF results are used, including records with no finite gate fidelity; per-panel n and count strips specify support."
    )
    txt(
        "Same conditional-density convention as the gate-error figure: each supported deviation column integrates to one over log10(coupling magnitude). Coupling bins are uniformly log-spaced over the common observed range; hues distinguish density decades and shades distinguish values within them. Gray denotes zero observed density or unsupported columns. These are pooled empirical distributions, not manufacturing-yield probabilities. No new OFF searches were run."
    )
    table(
        ["Logical residual magnitude (Hz)", "Minimum", "Median", "Maximum"],
        [
            [
                m["field"],
                *[
                    fmt(m["absolute_coupling_hz"][k], 2)
                    for k in ["minimum", "median", "maximum"]
                ],
            ]
            for m in coupling_meta
        ],
    )
    page()
    title("Failure and capacitance correlations")
    txt(
        "Associations are computed only within independent random-all samples (n="
        + str(len(random))
        + "), excluding deterministic sweeps, link-only draws, targeted failure-enriched rescues and max-stratified draws. Spearman correlations test individual fractional capacitor errors and pairwise products. Benjamini-Hochberg false-discovery correction is applied separately to each outcome. Correlation is not causation; optimization settings, basis choices and finite-search censoring remain confounders."
    )
    for target in targets:
        top = sorted(
            [q for q in corr if q["target"] == target],
            key=lambda q: abs(q["rho"]),
            reverse=True,
        )[:5]
        table(
            [target, "rho", "FDR q", "n"],
            [
                [q["feature"], fmt(q["rho"], 3), f"{q['q_FDR']:.2g}", q["n"]]
                for q in top
            ],
            [W * 0.62, W * 0.12, W * 0.14, W * 0.12],
        )
    valid = [d for d in done if np.isfinite(d["selected"])]
    cont = np.zeros((2, 2), int)
    for d in valid:
        cont[int(not d["B_pass"]), int(d["selected"] < 0.999)] += 1
    odds, pval = fisher_exact(cont)
    txt(
        "Association check, descriptive selected fidelity below 99.9% (not an acceptance rule): contingency rows B-pass/B-fail, columns >=99.9%/<99.9% = "
        + str(cont.tolist())
        + ". Fisher odds ratio "
        + str(odds)
        + ", p="
        + f"{pval:.3g}"
        + ". This pooled comparison is descriptive; random-only capacitor correlations above are the controlled subset."
    )
    if len(random) >= 20 and any(np.isfinite(d["selected"]) for _, d in random):
        from .correlation_views import build as build_correlation_views

        vm = build_correlation_views(out)
        page()
        title("Failure and low-fidelity deviation profiles")
        story.extend(
            [
                Image(
                    str(out / "random-success-failure-profiles.png"),
                    width=W,
                    height=W * 6.7 / 12,
                ),
                Spacer(1, 8),
            ]
        )
        txt(
            "Markers show median capacitor deviation; bars span the 25th-75th percentiles within each outcome group. The eight displayed capacitors were selected using the saved rank-association screening. B-readout accepted/rejected groups include all random samples. Fidelity profiles exclude the "
            + str(vm["gate_unavailable"])
            + " unavailable gate results, which remain a separate outcome in the two-capacitor maps. Raw 99% and selected 99.9% splits are descriptive display choices, not acceptance conditions."
        )
        table(
            ["Random-only outcome", "Records"],
            [
                ["B-readout accepted", vm["profile_counts"]["B_readout"]["good"]],
                ["B-readout rejected", vm["B_readout_rejected"]],
                ["Raw fidelity below 99%", vm["low_raw_below99"]],
                ["Selected fidelity below 99.9%", vm["low_selected_below99p9"]],
                ["Gate result unavailable", vm["gate_unavailable"]],
            ],
        )
        vc = vm["random_only_readout_vs_low_selected"]
        txt(
            "Readout/gate association within the same random subset: contingency rows B-pass/B-fail and columns selected fidelity >=99.9%/<99.9% = "
            + str(vc["contingency"])
            + ". Fisher odds ratio "
            + fmt(vc["odds_ratio"], 3)
            + ", p="
            + f"{vc['fisher_p']:.3g}"
            + ". This comparison excludes unavailable gates and includes finite diagnostics at rejected readout positions. The surrounding 22 capacitor dimensions remain varied; these profiles are conditional distributions, not isolated single-capacitor scans."
        )
        page()
        title("B-readout: two-capacitor failure map")
        story.extend(
            [
                Image(
                    str(out / "random-readout-capacitor-map.png"),
                    width=W,
                    height=W * 8.8 / 13,
                ),
                Spacer(1, 8),
            ]
        )
        txt(
            "Axes: resonator-r2 self-capacitance versus B1 self-capacitance, each relative to nominal. Bin edges are shown on the axes and saved in the map metadata. Color is the empirical rejection fraction, median fidelity, low-fidelity fraction or unavailable-gate fraction as labeled; it is NOT probability density. Every cell is annotated with its denominator n. Rejection/unavailable fractions use all records in that cell; fidelity statistics use only finite gate results. Gray means fewer than ten supported records, not failure."
        )
        rp = vm["readout_pair"]
        high = rp["B_rejection_percent"][-1][0]
        low = rp["B_rejection_percent"][0][-1]
        nh = rp["total_count"][-1][0]
        nl = rp["total_count"][0][-1]
        txt(
            "Opposite-corner comparison: larger B1 self-C with smaller resonator-r2 C gives "
            + fmt(high, 1)
            + "% rejected readout (n="
            + str(nh)
            + "); the opposite corner gives "
            + fmt(low, 1)
            + "% (n="
            + str(nl)
            + "). These are observed conditional outcomes, not guaranteed yields or proof of infeasibility. Higher B1 self-C can limit reachable frequency under the EJ ceiling; smaller resonator C raises resonator frequency, reducing readout-strength margin."
        )

        txt(
            "Median selected colors span at least 99.9-100%, avoiding visual amplification of tiny numerical changes. Median differences below 0.01 percentage point are unresolved at the agreed gate precision. The low-fidelity-fraction panels expose tails that a median would hide."
        )
        page()
        title("Gate fidelity: two-capacitor association map")
        story.extend(
            [
                Image(
                    str(out / "random-gate-capacitor-map.png"),
                    width=W,
                    height=W * 8.8 / 13,
                ),
                Spacer(1, 8),
            ]
        )
        txt(
            "Axes: physical A12-A21 intra-rail link versus B2 self-capacitance. These are the retained reference comparison axes; do not assume they are the strongest predictors in a new study. The maps are exploratory associations, not independently validated predictors. B-readout rejection is shown separately from gate outcomes."
        )
        txt(
            "Compare the observed profiles and maps rather than infer a fixed direction from the reference case. Overlapping one-dimensional profiles do not rule out multivariate or nonlinear effects. All other capacitor errors and their retuned controls are marginalized within each bin. Cell sample counts and unavailable-gate fractions should be considered before interpreting any apparent safe region. No sampled region is certified failure-free outside its observed support."
        )
        txt(
            "Full map matrices, support counts, profile group counts and strongest associations are saved in random-correlation-visualization-data.json. The profile and two-capacitor figures are also saved as individual PNG and SVG files beside this PDF. Numerical circuit calculations were not rerun for this visualization update."
        )
    else:
        txt(
            "Independent-random association views omitted: fewer than20 supported independent draws or no finite gate results. Structured/max-stratified points are not substituted for random draws."
        )
    page()
    title("Flags and interpretation")
    table(
        ["Flag", "Count"],
        [
            [
                "Readout rejected but gate values retained",
                summary["readout_rejected_with_gate"],
            ],
            ["OFF above 10 GHz", sum(d["off_GHz"] > 10 for d in done)],
            [
                "Best gate available but phase-pi not accepted",
                sum(np.isfinite(d["selected"]) and not d["gate_found"] for d in done),
            ],
            [
                "No finite gate best recorded",
                sum(not np.isfinite(d["selected"]) for d in done),
            ],
            [
                "Selected fidelity <99.9%, descriptive only",
                sum(d["selected"] < 0.999 for d in done),
            ],
            ["Raw fidelity <99%, descriptive only", sum(d["raw"] < 0.99 for d in done)],
            [
                "Duration at 130 ns upper bound",
                sum(abs(d["duration_ns"] - 130) < 1e-4 for d in done),
            ],
        ],
    )
    txt(
        "Small selected error can coexist with low raw fidelity: inspect survival/return before drawing a performance conclusion. Cases hitting the duration bound may prefer slower gates; the 70-130 ns bounds are retained. Readout chi strength at the EJ ceiling is not rescued by relaxing only chi matching. No unavailable gate is represented as a perfect gate or omitted from the detailed CSV."
    )
    txt(
        "Deliverables beside this PDF: all-points.csv (full diagnostics including mean/max chi, mismatch, H, dispersive ratio, phase error, acceptance and capacitor-error vector); correlations.csv (all tested individual and product associations); summary.json. The PDF omits the per-record listing. The CSV retains every saved point, with NA for unavailable values and explicit flags. Numerical records remain unchanged."
    )

    def footer(c, d):
        c.setFont("Helvetica", 7)
        c.drawString(
            38, 20, "Robustness summary | separate readout triples + five transmons"
        )
        c.drawRightString(A4[0] - 38, 20, str(d.page))

    temp = out / "summary-pages.pdf"
    SimpleDocTemplate(
        str(temp),
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=34,
        bottomMargin=34,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    doc = pymupdf.open(temp)
    target = out / (
        "robustness-summary-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        + ".pdf"
    )
    doc.save(target, garbage=4, deflate=True)
    doc.close()
    qa = out / "qa"
    qa.mkdir(exist_ok=True)
    doc = pymupdf.open(target)
    for i, page in enumerate(doc):
        if i < 6 or i == len(doc) - 1:
            page.get_pixmap(matrix=pymupdf.Matrix(1.1, 1.1)).save(
                qa / f"page-{i + 1:03d}.png"
            )
    result = dict(
        pdf=str(target),
        pages=len(doc),
        summary=summary,
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    (out / "report-status.json").write_text(json.dumps(result, indent=2))
    return result
