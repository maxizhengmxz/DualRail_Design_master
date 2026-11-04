"""Insert saved standalone readout maps into a design PDF; no physics solves.

The two portrait pages retain their model/basis scope and do not replace historical
integrated acceptance or gate results. Repeated calls replace this insert.
"""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

TITLE = "6d | Standalone three-state readout contrast"
FIGURE = "three-state-readout-chi-contours-all-gt-0p5-and-1MHz-bare-f01.png"


def insert_maps(report, maps, output, date, after_page=None):
    import pymupdf
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    summaries = {
        rail: json.loads((maps / f"{rail}-summary.json").read_text()) for rail in "AB"
    }
    figure = maps / FIGURE
    assert figure.is_file(), figure
    for summary in summaries.values():
        if "fixed_device" not in summary or "spectators" in summary:
            raise ValueError(
                "Require standalone pair+resonator maps, not seven-mode maps"
            )
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=12)
    header = ParagraphStyle("header", parent=body, fontName="Helvetica-Bold")
    buffer = BytesIO()
    canvas = Canvas(buffer, pagesize=A4)
    width, height = A4
    top = height - 42

    def paragraph(text, y, bold=False):
        item = Paragraph(text, header if bold else body)
        _, item_height = item.wrap(width - 84, height)
        item.drawOn(canvas, 42, y - item_height)
        return y - item_height - 9

    def table(rows, y, widths):
        data = [
            [Paragraph(str(cell), header if i == 0 else body) for cell in row]
            for i, row in enumerate(rows)
        ]
        item = Table(data, colWidths=widths, repeatRows=1)
        item.setStyle(
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
        _, item_height = item.wrap(width - 84, height)
        item.drawOn(canvas, 42, y - item_height)
        return y - item_height - 14

    def footer():
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            42, 24, f"Grounded coupler design | Standalone readout maps added {date}"
        )

    canvas.setFont("Helvetica-Bold", 17)
    canvas.drawString(42, top, TITLE)
    y = paragraph(
        f"Added {date}. Standard readout characterization after the independent readout links "
        "are fitted. Existing design, integrated-readout and gate sections retain their original "
        "controls, dates and validation status; this addition does not rerun them.",
        top - 18,
    )
    y = paragraph("Models, axes and state definitions", y, True)
    y = paragraph(
        "Separate A1-r1-A2 and B1-r2-B2 systems; no coupler and no opposite rail. "
        "All capacitors and resonator inductances are fixed at the saved nominal designs. "
        "Each pair's two EJs vary independently on a 61 x 61 grid, with axes converted to "
        "bare single-transmon f01 using the loaded EC and full-cosine spectrum. "
        "Ranges are A: 4-5 GHz and B: 4.4-5.8 GHz. Stars mark the standalone nominal idle points.",
        y,
    )
    y = paragraph(
        "00 denotes pair ground; 10 and 01 denote the lower and upper isolated-pair "
        "one-excitation branches used as overlap references. They are not unmixed physical "
        "occupations. The cavity frequency in state s is (E[s,1]-E[s,0])/h. "
        "All plotted contrasts are absolute differences of these frequencies, with no factor of two.",
        y,
    )
    y = paragraph("Nominal standalone contrasts (screening basis)", y, True)
    rows = [["Pair", "10 - 00 (MHz)", "01 - 00 (MHz)", "|01 - 10| (kHz)"]]
    for rail, summary in summaries.items():
        c10, c01 = summary["nominal_chi_relative_00_mhz"]
        rows.append([rail, f"{c10:.6f}", f"{c01:.6f}", f"{abs(c01 - c10) * 1000:.3f}"])
    y = table(rows, y, [45, 155, 155, width - 84 - 355])
    y = paragraph("Contours and interpretation", y, True)
    y = paragraph(
        "The following six-panel figure shows |01-10|, |01-00| and |10-00| at 0.5, 1 and 2 MHz. "
        "Orange means ALL three contrasts exceed 0.5 MHz; pink means ALL exceed 1 MHz. "
        "Blue marks the logical contrast below 10 kHz in the left column. "
        "Gray flags minimum reference assignment overlap below 95%; raw values remain saved. "
        "These are spectral-contrast diagnostics, not additional readout acceptance constraints "
        "or predictions of measurement fidelity. The existing H-score, chi-mismatch and "
        "dispersive constraints still apply separately.",
        y,
    )
    y = paragraph("Basis checks and reproduction", y, True)
    rows = [["Pair", "Grid + audits (s)", "Largest audited cavity-shift change (kHz)"]]
    for rail, summary in summaries.items():
        change = max(abs(v) for audit in summary["audit"] for v in audit["change_khz"])
        rows.append([rail, f"{summary['elapsed_seconds']:.2f}", f"{change:.3f}"])
    y = table(rows, y, [45, 170, width - 84 - 215])
    y = paragraph(
        "Screening uses six local transmon levels, resonator states 0/1/2, charge cutoff 25 "
        "and 30 assigned eigenlevels. Ten fixed-control checks per pair (nine grid locations "
        "plus nominal idle) use eight local levels, charge cutoff 35 and 40 eigenlevels, "
        "with the same three resonator states. Finite-grid contours and these spot checks "
        "do not certify every narrow boundary or Hz-level residual coupling. "
        "Nominal screening values above need not equal earlier larger-basis acceptance tables.",
        y,
    )
    y = paragraph(
        "Source: readout-cavity-EJ-maps/{A,B}-summary.json and {A,B}-maps.npz, from "
        "nominal-design/fabrication-input.json readout hardware. Effective modal capacitors "
        "in the map snapshots are reductions of the physical floating-pad circuit, not "
        "fabricated link values. Replot with python -m readout_workflow.readout_shift_map; no optimization is run "
        "during report assembly. Seven-mode 2D maps are optional and excluded here.",
        y,
    )
    if y < 45:
        raise ValueError(f"Report insert overflow: bottom={y}")
    footer()
    canvas.showPage()
    canvas.setFont("Helvetica-Bold", 17)
    canvas.drawString(42, height - 42, TITLE + " | maps")
    canvas.setFont("Helvetica", 9)
    canvas.drawString(
        42,
        height - 60,
        "Original six-panel layout | Standalone triples | Bare f01 axes",
    )
    picture = ImageReader(str(figure))
    iw, ih = picture.getSize()
    pw = width - 84
    ph = pw * ih / iw
    canvas.drawImage(picture, 42, height - 86 - ph, width=pw, height=ph)
    footer()
    canvas.save()

    source = pymupdf.open(report)
    original_metadata = source.metadata
    # Idempotent insert: do not accumulate duplicate map pages on later edits.
    for i in range(len(source) - 1, -1, -1):
        if TITLE in source[i].get_text():
            source.delete_page(i)
    if after_page is None:
        # Optional pulse sections shift pagination; follow the readout section,
        # rather than inserting maps into an unrelated fixed page number.
        anchors = [
            i
            for i, page in enumerate(source)
            if "6c | Readout at calibrated idle centers" in page.get_text()
        ]
        if len(anchors) != 1:
            raise ValueError("Readout section anchor missing; specify --after-page")
        after_page = anchors[0] + 1
    if not 0 <= after_page <= len(source):
        raise ValueError("Invalid insertion page")
    addition = pymupdf.open(stream=buffer.getvalue(), filetype="pdf")
    source.insert_pdf(addition, start_at=after_page)
    for i, page in enumerate(source):
        # Existing page number is right aligned in the footer, outside body text.
        box = pymupdf.Rect(
            page.rect.width - 67,
            page.rect.height - 35,
            page.rect.width - 36,
            page.rect.height - 14,
        )
        page.add_redact_annot(box, fill=(1, 1, 1))
        page.apply_redactions(images=0, graphics=0)
        page.insert_textbox(box, str(i + 1), fontsize=8, align=2)
    source.set_metadata(original_metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".updating.pdf")
    source.save(temp, garbage=4, deflate=True)
    source.close()
    temp.replace(output)
    record = dict(
        report=str(output.resolve()),
        date=date,
        pages=after_page,
        map_figure_sha256=sha256(figure.read_bytes()).hexdigest(),
        standalone_map_directory=str(maps.resolve()),
        scope="Standalone readout triple maps only; no numerical reruns",
    )
    with pymupdf.open(output) as document:
        record["pages"] = len(document)
    output.with_suffix(".readout-map-update.json").write_text(
        json.dumps(record, indent=2)
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--after-page",
        type=int,
        default=None,
        help="Default: after the calibrated-idle readout section",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            insert_maps(
                args.report, args.maps, args.output, args.date, args.after_page
            ),
            indent=2,
        )
    )
