#!/usr/bin/env python3
"""Create the concise combined-feasibility PDF report."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "data" / "plots"
OUTPUT = ROOT.parent / "output" / "pdf" / "combined_readout_xm_cz_feasibility_report.pdf"

PAGE = landscape(A4)
PAGE_W, PAGE_H = PAGE
NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2878B5")
TEAL = colors.HexColor("#2A9D8F")
ORANGE = colors.HexColor("#E07A3F")
LIGHT = colors.HexColor("#EEF3F7")
MID = colors.HexColor("#667788")
DARK = colors.HexColor("#17212B")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=27, leading=32, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=9 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=14, leading=20, textColor=MID, spaceAfter=9 * mm,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, textColor=NAVY, spaceAfter=5 * mm,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=13, leading=16, textColor=BLUE, spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10.2, leading=14.5, textColor=DARK, spaceAfter=3 * mm,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.5, leading=11, textColor=MID,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9, leading=12, textColor=MID, alignment=TA_CENTER,
            spaceBefore=2 * mm,
        ),
        "metric": ParagraphStyle(
            "Metric", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=16, leading=19, textColor=NAVY, alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.5, leading=11, textColor=MID, alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=11, leading=16, textColor=NAVY, spaceAfter=0,
        ),
    }


S = styles()


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E1E8"))
    canvas.line(16 * mm, 12 * mm, PAGE_W - 16 * mm, 12 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID)
    canvas.drawString(16 * mm, 7.5 * mm, "Combined Readout and XM-CZ Feasibility")
    canvas.drawRightString(PAGE_W - 16 * mm, 7.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def scaled_image(path: Path, max_w: float, max_h: float) -> Image:
    reader = ImageReader(str(path))
    width, height = reader.getSize()
    scale = min(max_w / width, max_h / height)
    return Image(str(path), width=width * scale, height=height * scale)


def metric_cards(items):
    cells = []
    for value, label in items:
        cells.append([Paragraph(value, S["metric"]), Paragraph(label, S["metric_label"])])
    table = Table([cells], colWidths=[(PAGE_W - 40 * mm) / len(cells)] * len(cells))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CDD9E2")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
    ]))
    return table


def figure_page(title: str, image_name: str, caption: str, max_h=142 * mm):
    return [
        Paragraph(title, S["h1"]),
        scaled_image(PLOTS / image_name, PAGE_W - 36 * mm, max_h),
        Paragraph(caption, S["caption"]),
        PageBreak(),
    ]


def build_report():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=PAGE,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=16 * mm,
        title="Combined Readout and XM-CZ Feasibility Report",
        author="Dual-rail simulation study",
        subject="Readout, gate, fixed-hardware, and resonator-inclusive validation",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=footer))

    story = [
        Spacer(1, 13 * mm),
        Paragraph("Combined Readout and XM-CZ Feasibility", S["title"]),
        Paragraph(
            "A staged design-space study for two dual-rail qubits, independent idle readout, "
            "and a central-frequency CZ gate.", S["subtitle"]),
        metric_cards([
            ("48 / 48", "readout-feasible targets for each rail"),
            ("28", "independent readout-gate overlay points"),
            ("16", "fixed-hardware reconciled candidates"),
            ("0 / 16", "full-network endpoint passes"),
        ]),
        Spacer(1, 9 * mm),
        Paragraph("Executive conclusion", S["h1"]),
        Paragraph(
            "The independent screens identify a substantial promising region, and 16 candidates "
            "can be represented by one fixed fabricated capacitance network. In the final "
            "resonator-inclusive model, all 16 remain dispersive and pass the updated S1 screen. "
            "However, none retain both the complete conditional readout requirements and the strict "
            "CZ thresholds at the inherited operating points.", S["body"]),
        Table([[Paragraph(
            "This is an endpoint validation failure, not a proof that the fixed hardware is globally "
            "infeasible. The next useful calculation is a local joint repair of idle and gate controls "
            "while keeping each solved capacitance matrix fixed.", S["callout"]) ]],
            colWidths=[PAGE_W - 44 * mm], style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F4F2")),
                ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
            ])),
        Spacer(1, 8 * mm),
        Paragraph("Study scope", S["h2"]),
        Paragraph(
            "Rail A uses a 140-160 MHz logical-frequency requirement and rail B uses 80-100 MHz. "
            "Readout is evaluated at idle; the gate uses a separate operating point. Josephson "
            "energies may change between the two points, but fabricated capacitances must agree.",
            S["body"]),
        PageBreak(),
    ]

    story += figure_page(
        "1. Independent idle-readout feasibility",
        "independent_readout_maps.png",
        "Both independently simulated rails validate all 48 loaded-EC targets in the direct-EC hull.",
    )
    story += figure_page(
        "2. Logical-frequency witnesses",
        "readout_logical_frequency_maps.png",
        "The selected readout witnesses satisfy their respective logical-frequency windows. Rail A "
        "solutions cluster near the lower boundary because the search stops at the first validated witness.",
    )
    story += figure_page(
        "3. Independent-map overlay and hardware reconciliation",
        "agreed_test_overlay.png",
        "The central-frequency gate map is overlaid with the independent rail-B readout map. Of 28 "
        "overlap points, 16 admit consistent reduced fixed hardware.",
    )
    story += figure_page(
        "4. Gate fidelity on the reconciled region",
        "combined_gate_fidelity.png",
        "Before full-network loading, the 16 reconciled candidates have raw fidelity of 95.60-98.49% "
        "and post-selected fidelity of 99.908-99.978%.",
    )

    story += [
        Paragraph("5. Idle-to-gate Josephson-energy movement", S["h1"]),
        Paragraph(
            "The same fabricated hardware is used at idle and gate operation, but the four EJs are "
            "allowed to change. The maps below show signed shifts, defined as gate EJ minus idle EJ.",
            S["body"]),
        Table([
            [scaled_image(PLOTS / "rail_a_ej_shifts.png", 123 * mm, 92 * mm),
             scaled_image(PLOTS / "rail_b_ej_shifts.png", 123 * mm, 92 * mm)],
            [Paragraph("Rail A: A1 -3.356 to +0.013 GHz; A2 -2.313 to +0.040 GHz", S["caption"]),
             Paragraph("Rail B: B1 -5.653 to +1.113 GHz; B2 -3.443 to +0.868 GHz", S["caption"])],
        ], colWidths=[(PAGE_W - 40 * mm) / 2] * 2,
            style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("ALIGN", (0, 0), (-1, -1), "CENTER")])),
        PageBreak(),
    ]

    story += [
        Paragraph("6. Resonator-inclusive full-network validation", S["h1"]),
        Paragraph(
            "Each reconciled candidate was promoted to a six-node network containing A1, A2, the "
            "rail-A resonator, B1, B2, and the rail-B resonator. The same physical capacitance matrix "
            "was frozen at idle and gate operation. Two basis sizes were used to check numerical stability.",
            S["body"]),
        scaled_image(PLOTS / "full_network_validation.png", PAGE_W - 36 * mm, 120 * mm),
        Paragraph(
            "All 16 physical realizations pass dispersiveness and updated S1. No inherited endpoint "
            "passes the complete readout or gate criteria. Larger-basis gate ranges are 81.09-97.07% "
            "raw and 92.55-99.28% post-selected fidelity.", S["caption"]),
        PageBreak(),
    ]

    data = [
        ["Stage", "Candidates passing", "Interpretation"],
        ["Independent readout", "48/48 per rail", "Each rail is feasible in isolation"],
        ["Readout-gate overlay", "28", "Independent requirements overlap"],
        ["Reduced fixed hardware", "16", "Loaded EC and intra-rail coupling agree"],
        ["Full physical realization", "16/16", "Six-node capacitance solves converge"],
        ["Dispersive and updated S1", "16/16", "No failure from these screens"],
        ["Conditional idle readout", "0/16", "Spectator dependence and chi mismatch dominate"],
        ["Full-network gate thresholds", "0/16", "Inherited gate controls do not survive loading"],
        ["Complete endpoint", "0/16", "Local joint retuning is required"],
    ]
    summary_table = Table(data, colWidths=[58 * mm, 45 * mm, 137 * mm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D5DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.7 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7 * mm),
    ]))
    story += [
        Paragraph("7. Conclusions and recommended next step", S["h1"]),
        summary_table,
        Spacer(1, 3 * mm),
        Paragraph("What the study establishes", S["h2"]),
        Paragraph(
            "The staged overlay efficiently reduced the design space before invoking the expensive "
            "complete model. It also confirmed that fixed-hardware realization, dispersiveness, and S1 "
            "are not the present bottlenecks.", S["body"]),
        Paragraph("What remains unresolved", S["h2"]),
        Paragraph(
            "Independent readout and gate witnesses are not automatically compatible after both "
            "resonators and all inverse-capacitance cross couplings are included. A duration-only scan "
            "over 60-140 ns did not recover the 99.9% post-selected gate requirement.", S["body"]),
        Paragraph("Recommended next calculation", S["h2"]),
        Paragraph(
            "For each of the 16 fixed capacitance matrices, locally re-optimize idle EJs for conditional "
            "readout and jointly re-optimize gate EJs, physical operating frequency, and duration. Rank "
            "solutions by robust margin, then repeat the logical-frequency sensitivity sweep.", S["body"]),
        Paragraph(
            "Modeling note: this validation represents the effective A2-B1 gate coefficient with a "
            "fixed direct capacitance; an explicit tunable-coupler design requires a revised final network.",
            S["small"]),
    ]

    doc.build(story)


if __name__ == "__main__":
    build_report()
    print(OUTPUT)
