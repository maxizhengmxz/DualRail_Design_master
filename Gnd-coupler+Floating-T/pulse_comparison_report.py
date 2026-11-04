"""Insert the optional saved five-transmon EJ-pulse study into a design PDF.

Report-only: no Hamiltonian solver or optimizer imports. Earlier report content
and numerical-status statements are preserved. Repeated edits replace this insert.
"""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

PREFIX = "Optional EJ-pulse CZ"


def insert_report(report, study, output, date, after_page=6):
    import pymupdf
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Paragraph, Table, TableStyle
    from matplotlib.mathtext import math_to_image
    from matplotlib.font_manager import FontProperties

    data = json.loads((study / "comparison.json").read_text())
    rows = data["rows"]
    if len(rows) != 10 or not all(r["phase_pi_calibrated"] for r in rows):
        raise ValueError("Require the ten completed calibrated comparison records")
    cases = [json.loads(p.read_text()) for p in sorted(study.glob("*/result.json"))]
    if len(cases) != 10:
        raise ValueError("Missing per-case records")
    figure = study / "pulse-shape-comparison.png"
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=12)
    bold = ParagraphStyle("bold", parent=body, fontName="Helvetica-Bold")
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=A4)
    width, height = A4
    y = height - 42

    def title(suffix):
        nonlocal y
        canvas.setFont("Helvetica-Bold", 17)
        canvas.drawString(42, height - 42, PREFIX + " | " + suffix)
        y = height - 62

    def para(text, heading=False):
        nonlocal y
        item = Paragraph(text, bold if heading else body)
        _, h = item.wrap(width - 84, height)
        item.drawOn(canvas, 42, y - h)
        y -= h + 8

    def table(contents, widths):
        nonlocal y
        table = Table(
            [
                [Paragraph(str(cell), bold if i == 0 else body) for cell in row]
                for i, row in enumerate(contents)
            ],
            colWidths=widths,
            repeatRows=1,
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eff5")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.35, colors.HexColor("#cdd6dd")),
                ]
            )
        )
        _, h = table.wrap(width - 84, height)
        table.drawOn(canvas, 42, y - h)
        y -= h + 12

    def equation(text):
        nonlocal y
        image = BytesIO()
        math_to_image(
            text,
            image,
            prop=FontProperties(size=10),
            dpi=220,
            format="png",
            color="#183448",
        )
        picture = ImageReader(image)
        iw, ih = picture.getSize()
        scale = min(72 / 220, (width - 84) / iw)
        canvas.drawImage(
            picture,
            42,
            y - ih * scale,
            width=iw * scale,
            height=ih * scale,
            mask="auto",
        )
        y -= ih * scale + 12

    def end_page():
        if y < 45:
            raise ValueError(f"Optional pulse page overflows: y={y}")
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            42,
            24,
            f"Grounded coupler design | Optional five-transmon pulse study | Added {date}",
        )
        canvas.showPage()

    title("definition and scope")
    para(
        f"Added {date}; optional post-design study, requested separately. Earlier static-gate and "
        "integrated-readout sections retain their original controls, dates and validation status. "
        "This section reads completed calculations; no optimization was run during reporting."
    )
    para("What is calculated", True)
    para(
        "Coherent evolution of the nominal FIVE-transmon A1-A2-C-B1-B2 circuit. Start at the saved "
        "five-mode OFF position (C f01 = 9.800131 GHz), with A at 4.61 GHz and B at 5.39 GHz. "
        "Freeze all capacitances, loaded ECs and charge couplings; A EJs remain fixed. "
        "Move C, B1 and B2 simultaneously to gate settings, hold, then return symmetrically. "
        "The C ON endpoint is fixed at 7.05 GHz. Calibrate the common bare B gate frequency "
        "and hold duration at conditional phase pi, retaining the selected logical 10 to B20 CZ channel."
    )
    para(
        "Loaded EC/h = (165, 250, 450.285, 160, 260) MHz in physical order A1,A2,C,B1,B2. "
        "Use the SAME four full-system dressed IDLE logical states as both inputs and final projections. "
        "The full rise, hold, fall and simulated ringing act on them. This differs from static ON-state "
        "preparation. No resonators, T1, flux noise or measurement dynamics are included."
    )
    para("Rise time means 10% to 90% of the commanded EJ excursion", True)
    para(
        "For normalized rising envelope s(t), the quoted rise is t(s=0.9)-t(s=0.1). It is NOT the "
        "full edge duration T. Falling edges use the same width. Both shapes interpolate EJ directly; "
        "local frequencies follow the full-cosine spectrum at fixed loaded EC. These are not tanh/cosine "
        "frequency pulses or calibrated SQUID-flux waveforms."
    )
    equation(
        r"$s_{\rm tanh}(t)=\frac{\tanh[(t-T/2)/\tau]+\tanh 4}{2\tanh 4},\quad T=8\tau$"
    )
    equation(
        r"$t_{10-90}=2\tau\,\mathrm{atanh}(0.8\tanh 4),\quad T=3.645897\,t_{10-90}$"
    )
    equation(
        r"$s_{\cos}(t)=\frac{1-\cos(\pi t/T)}{2},\quad T=1.693955\,t_{10-90}\quad(0\leq t\leq T)$"
    )
    table(
        [
            ["Commanded rise", "Tanh full edge T", "Cosine full edge T"],
            ["10 ns", "36.46 ns", "16.94 ns"],
            ["1 ns", "3.65 ns", "1.69 ns"],
        ],
        [150, 180, width - 84 - 330],
    )
    para("1 GHz sharp cutoff and timing convention", True)
    para(
        "Filter the ENTIRE EJ envelope with H(f)=1 for |f|&lt;1 GHz and zero above it, assuming zero "
        "phase (a pure delay is removed). Retain overshoot and pre/post-ringing without clipping. "
        "The line model acts on EJ; a physical flux line requires a separate nonlinear SQUID map."
    )
    equation(
        r"$E_{J,i}(t)=E^{\rm idle}_{J,i}+\left(E^{\rm gate}_{J,i}-E^{\rm idle}_{J,i}\right)s_{\rm filtered}(t)$"
    )
    para(
        "Programmed gate duration = 2T + hold. Propagation initially adds 16 ns before and after that "
        "span, and the fastest-edge audit tests 32 ns per side. This padding is a numerical window, "
        "not a mandatory hardware guard. An ideal brick wall has infinite sinc tails and no exact "
        "finite settling time. The programmed and simulated durations are both retained in the data."
    )
    end_page()

    title("frequency tuning in time")
    para(
        "Representative saved 10 ns tanh and cosine gates. Colored intervals mark the complete "
        "EJ ramp, calibrated hold and return; gray lines mark command boundaries. The 10 ns "
        "arrows refer to 10%-90% of EJ excursion, not to the full shaded ramp."
    )
    picture = ImageReader(str(study / "frequency-tuning-schematic.png"))
    iw, ih = picture.getSize()
    pw = width - 84
    ph = pw * ih / iw
    canvas.drawImage(picture, 42, y - ph, width=pw, height=ph)
    y -= ph + 14
    para(
        "These are BARE local transmon frequencies obtained from each filtered EJ waveform and "
        "its loaded EC. They are not dressed logical splittings or instantaneous coupled-system "
        "eigenbranches. A1/A2 remain fixed at 4.61 GHz; B1/B2 go from 5.39 GHz to approximately "
        "4.8713 GHz; C goes from its five-mode OFF setting at 9.8001 GHz to 7.05 GHz. "
        "The two B curves nearly coincide; their underlying EJ excursions differ."
    )
    table(
        [
            ["10 ns command", "Full ramp (ns)", "Hold (ns)", "Programmed total (ns)"],
            ["tanh", "36.46", "85.28", "158.20"],
            ["cosine", "16.94", "101.93", "135.81"],
        ],
        [110, 120, 120, width - 84 - 350],
    )
    para(
        "The 1 GHz filter and pre/post-ringing are included in the curves. The shown traces "
        "use saved calibrated endpoints; no gate optimization or coupled time propagation was rerun."
    )
    end_page()

    title("calibration and results")
    para(
        "Each row is one locally calibrated phase-pi pulse. The bounded search maximizes RAW fidelity; "
        "the postselected score and survival are evaluated at that SAME pulse. Common B gate f01 is "
        "about 4.8713 GHz, with the precise per-row values saved. This is not a global optimum proof."
    )
    contents = [
        [
            "Shape",
            "Rise<br/>(ns)",
            "Hold<br/>(ns)",
            "Programmed<br/>total (ns)",
            "Raw<br/>(%)",
            "Selected<br/>(%)",
        ]
    ]
    ordered = sorted(rows, key=lambda r: (r["shape"] != "tanh", -r["input_rise_ns"]))
    for r in ordered:
        contents.append(
            [
                r["shape"],
                f"{r['input_rise_ns']:g}",
                f"{r['hold_ns']:.2f}",
                f"{r['programmed_total_ns']:.2f}",
                f"{100 * r['raw_fidelity']:.3f}",
                f"{100 * r['postselected_fidelity']:.3f}",
            ]
        )
    table(contents, [65, 55, 70, 110, 100, width - 84 - 400])
    para("Fidelity and postselection convention", True)
    para(
        "Let M be the four-by-four final propagator projected onto the dressed idle logical basis. "
        "After the same permitted local-Z phase corrections used throughout the CZ workflow, "
        "raw fidelity includes nonlogical loss. The selected value is the established "
        "survival-normalized score, not a separately optimized gate."
    )
    equation(
        r"$\bar p=\frac{\mathrm{Tr}(M^\dagger M)}{4},\quad F_{\rm raw}=\frac{\mathrm{Tr}(M^\dagger M)+|\mathrm{Tr}(U_{\rm CZ}^{\dagger} ZM)|^2}{20},\quad F_{\rm sel}=\frac{F_{\rm raw}}{\bar p}$"
    )
    para(
        "The 10 ns cosine gate takes 135.81 ns versus 158.20 ns for 10 ns tanh: 22.39 ns shorter. "
        "Their postselected scores differ by about 0.0052 percentage point, below the adopted "
        "0.01-point probability-accuracy target; the raw-score difference is about 0.0574 point. "
        "At similar total duration, 5 ns tanh has visibly lower selected fidelity than 10 ns cosine."
    )
    para(
        "Shorter edges increase coherent wrong-logical-state population as well as leakage. "
        "Postselection removes nonlogical population but cannot remove logical mixing. From 1 to 0.5 ns "
        "the programmed gate shortens by only 1-2 ns. The delivered 0.5 ns edges broaden to about "
        "0.59 ns (tanh) and 0.57 ns (cosine). The 1 GHz filter barely changes the 10 ns tanh result."
    )
    para("Numerical scope", True)
    para(
        "Start with six levels per rail transmon, five coupler levels and a total local-index cutoff "
        "of six (287 even-parity states). Transfer phase at fixed hold/C to eight rail levels, seven "
        "coupler levels and cutoff eight (782 states). Cutoff is a numerical truncation, not excitation-number "
        "conservation. Propagation keeps full charge interactions and off-diagonal EJ-dependent cosine terms "
        "in one fixed basis. Filter FFT: 4096 ns span, 1/256 ns sample. The final 0.5 ns rows use a 0.0375 ns "
        "integration step; other rows use approximately 0.075 ns."
    )
    audit_note = (
        "Representative 0.5 ns cosine checks: step halving changed raw fidelity by 0.01457 percentage "
        "point and selected fidelity by 0.000157 point; the finer row was adopted and rephased. "
        "Larger excitation cutoff changed raw by 0.000078 point; doubled tail window changed raw "
        "by 0.000429 point. These latter checks used the earlier coarse controls/step. Some worst-input "
        "leakage differences exceeded 10<super>-4</super>. A full independent 10<super>-4</super> certificate for every row, or a "
        "further step audit at each final fine control, was not completed. The 135 regression tests passed."
    )
    end_page()

    title("comparison figure")
    para(
        "Same five-transmon hardware and dressed idle frame; ideal 1 GHz EJ cutoff. "
        "Original four-panel layout, scaled to fit this portrait page."
    )
    picture = ImageReader(str(figure))
    iw, ih = picture.getSize()
    pw = width - 84
    ph = pw * ih / iw
    canvas.drawImage(picture, 42, y - ph, width=pw, height=ph)
    y -= ph + 14
    end_page()

    width, height = A4
    canvas.setPageSize(A4)
    title("checks and timing")
    para("Recorded costs from the first comparison", True)
    contents = [
        [
            "Shape / rise",
            "Main case<br/>(s)",
            "Filtered<br/>trials",
            "Screened pulse<br/>propagation (s)",
        ]
    ]
    for d in sorted(cases, key=lambda d: (d["shape"] != "tanh", -d["rise_ns"])):
        path = study / f"{d['shape']}-{d['rise_ns']:g}ns" / "progress.json"
        progress = json.loads(path.read_text())
        contents.append(
            [
                f"{d['shape']} / {d['rise_ns']:g} ns",
                f"{d['elapsed_seconds']:.1f}"
                if "elapsed_seconds" in d
                else "not fully timed",
                str(len(progress)),
                f"{sum(r['seconds'] for r in progress):.1f}",
            ]
        )
    table(contents, [140, 100, 85, width - 84 - 325])
    para(
        "Main case includes unfiltered seeding, filtered calibration and larger-basis phase transfer, "
        "but excludes later focused checks. The two shape workers ran concurrently; summed durations "
        "are not wall time. Exact end-to-end wall time was not recorded. The last tanh search was stopped "
        "after 30 trials; its logged trial propagation totals 275.6 s and fixed-hold completion 139.0 s. "
        "The focused cosine audit took 197.2 s; its final fine-step phase correction took 43.6 s."
    )
    para("Why it was slow", True)
    para(
        "The ideal line response adds ringing throughout the pulse, including the plateau. Changing hold "
        "duration changes the filtered waveform, so the filtered calculation cannot simply reuse a "
        "constant ON spectrum. Every trial propagated four logical inputs through half the complete "
        "time-symmetric waveform. A screened trial took roughly 8-13 s; a logged larger-basis propagation "
        "typically took 26-49 s. Larger-basis transfer repeated such propagations to enforce phase pi. "
        "The long 0.5 ns tanh line search kept sampling a fidelity plateau despite a five-iteration limit."
    )
    para("Representative numerical checks", True)
    para(audit_note)
    end_page()
    title("speedup options")
    para("Already implemented after this run", True)
    para(
        "Hard 12-trial filtered search limit; fidelity/survival/leakage plateau stop at 10<super>-4</super> after eight "
        "trials; bounded phase finish; cached phase-slope/secant updates instead of a fresh derivative "
        "probe each iteration. Two shape workers, one BLAS thread each, exact waveform caches and "
        "time symmetry were already used. Default integration step is now 0.0375 ns for rises&lt;=1 ns "
        "and 0.075 ns otherwise. The exact as-run driver and interrupted-case completion are preserved. "
        "The revised routine has not been rebenchmarked across the full sweep."
    )
    para("Further proposals - not implemented in this report", True)
    para(
        "1. Reuse the closest calibrated edge as the next seed, instead of repeating a seven-frequency "
        "scan plus scalar polish for every case.<br/>2. Test the filtered pulse at the unfiltered optimum "
        "first. If differences are below 10<super>-4</super>, try phase-only transfer before a new two-control search.<br/>"
        "3. Reuse identical fixed-basis Hamiltonian matrices and idle states across rise times; benchmark "
        "the smallest adequate production basis on representative slow/fast edges.<br/>"
        "4. Use a separately checked adaptive propagation step to concentrate work near edges/ringing. "
        "Do not replace a ringing plateau by a constant Hamiltonian without validating that approximation."
    )
    para(
        "These proposals need separate agreement and measurement. No new basis reduction or pulse "
        "approximation was applied for this PDF, and no speed factor is promised. Reproduction inputs, "
        "case records, comparison.csv/json, figure and provenance are in output/pulse-shape-comparison-1GHz/."
    )
    end_page()
    canvas.save()

    source = pymupdf.open(report)
    metadata = source.metadata
    for i in range(len(source) - 1, -1, -1):
        if PREFIX in source[i].get_text():
            source.delete_page(i)
    if not 0 <= after_page <= len(source):
        raise ValueError("Invalid insertion page")
    addition = pymupdf.open(stream=stream.getvalue(), filetype="pdf")
    source.insert_pdf(addition, start_at=after_page)
    for i, page in enumerate(source):
        box = pymupdf.Rect(
            page.rect.width - 67,
            page.rect.height - 35,
            page.rect.width - 36,
            page.rect.height - 14,
        )
        page.add_redact_annot(box, fill=(1, 1, 1))
        page.apply_redactions(images=0, graphics=0)
        page.insert_textbox(box, str(i + 1), fontsize=8, align=2)
    source.set_metadata(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".updating.pdf")
    source.save(temp, garbage=4, deflate=True)
    source.close()
    temp.replace(output)
    with pymupdf.open(output) as document:
        count = len(document)
    record = dict(
        report=str(output.resolve()),
        pages=count,
        date=date,
        optional=True,
        figure_sha256=sha256(figure.read_bytes()).hexdigest(),
        tuning_figure_sha256=sha256(
            (study / "frequency-tuning-schematic.png").read_bytes()
        ).hexdigest(),
        data_sha256=sha256((study / "comparison.json").read_bytes()).hexdigest(),
        scope="Report-only insertion; historical design validation preserved; no calculations rerun",
    )
    output.with_suffix(".pulse-update.json").write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument("--after-page", type=int, default=6)
    a = parser.parse_args()
    print(
        json.dumps(
            insert_report(a.report, a.study, a.output, a.date, a.after_page), indent=2
        )
    )
