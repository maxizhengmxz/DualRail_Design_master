#!/usr/bin/env python3
"""Create the current strict-4D fixed-A campaign report."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf"
PDF = OUT / "current_fixed_a_strict_4d_campaign_report.pdf"
FIGURES = OUT / "combined_feasibility_report_figures"
FOUR_D = HERE / "four_dimensional_feasibility_joint_anchor_range_5MHz_no_jump_limit_2026-11-12"
CORRECTED = HERE / "corrected_cavity_strict_tuple_A154_208_B216_252_2026-11-13"


def image(path: Path, width=174 * mm, max_height=205 * mm):
    with PILImage.open(path) as source:
        w, h = source.size
    height = width * h / w
    if height > max_height:
        height = max_height
        width = height * w / h
    return Image(str(path), width=width, height=height)


def table(rows, widths):
    body = [
        [Paragraph(str(value), STYLES["TableHead"] if i == 0 else STYLES["TableCell"]) for value in row]
        for i, row in enumerate(rows)
    ]
    result = Table(body, colWidths=widths, repeatRows=1, hAlign="CENTER")
    result.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#81909a")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173f5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f7")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#687680"))
    canvas.drawString(17 * mm, 9 * mm, "Fixed-rail-A strict 4D feasibility campaign")
    canvas.drawRightString(A4[0] - 17 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=21, leading=25, textColor=colors.HexColor("#17324d"), spaceAfter=8))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=10.5, leading=14, textColor=colors.HexColor("#52636f"), spaceAfter=12))
styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=9.3, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=10, leading=14, backColor=colors.HexColor("#e7f1f6"), borderColor=colors.HexColor("#6e9bb7"), borderWidth=.7, borderPadding=8, spaceAfter=8))
styles.add(ParagraphStyle(name="Warning", parent=styles["BodyText"], fontSize=9.2, leading=13, backColor=colors.HexColor("#fff2d5"), borderColor=colors.HexColor("#d29b3d"), borderWidth=.7, borderPadding=7, spaceAfter=8))
styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontSize=7.5, leading=9.2))
styles.add(ParagraphStyle(name="TableHead", parent=styles["TableCell"], textColor=colors.white, fontName="Helvetica-Bold"))
STYLES = styles


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corrected = json.loads((CORRECTED / "corrected_tuple_summary.json").read_text())["corrected"]
    strict = json.loads((FOUR_D / "summary.json").read_text())
    doc = SimpleDocTemplate(
        str(PDF), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="Current Fixed-Rail-A Strict 4D Campaign Report",
    )
    story = [
        Spacer(1, 18 * mm),
        Paragraph("Fixed-Rail-A Strict 4D Feasibility Campaign", styles["ReportTitle"]),
        Paragraph("Independent readout and gate maps, common-anchor neighborhood certification, and corrected-cavity center validation", styles["Subtitle"]),
        Spacer(1, 7 * mm),
        Paragraph("Executive summary", styles["Heading1"]),
        Paragraph("The current workflow searches for a high-fidelity CZ design while holding rail A at one validated idle-readout bias. Rail B is tuned downward from an independently validated idle-readout point to the CZ interaction point. A saved-data four-dimensional EC analysis then requires every sampled neighbor to satisfy readout, gate, clearance, common-anchor and logical-source conditions.", styles["Body2"]),
        Paragraph("The strict 5 MHz analysis leaves two rail-A projections. The preferred exploratory tuple is EC=(154,208,216,252) MHz. A fresh center-tuple rerun using rail-A/rail-B cavity frequencies 6.05/5.95 GHz passes raw and post-selected fidelity and fixed-hardware reconciliation.", styles["Callout"]),
        table([
            ["Current item", "Result"],
            ["Grid", "2 MHz ordered EC lattice"],
            ["Strict neighborhood", "5 MHz-radius 4D ball; 233 sampled offsets"],
            ["Fidelity thresholds", "Raw >95%; post-selected >99.9%"],
            ["Signed idle clearance", "Rail-B idle > all retained CZ interactions by 200 MHz"],
            ["Strict 4D centers", f"{strict['five_mhz_ball_center_count']:,}"],
            ["Rail-A projections", "(208,154) and (154,208) MHz"],
            ["Corrected center tuple", "(154,208,216,252) MHz: pass"],
        ], [58 * mm, 105 * mm]),
        Spacer(1, 8 * mm),
        Paragraph("Scope warning", styles["Heading2"]),
        Paragraph("The corrected rerun certifies the selected center tuple, not every neighbor. The historical split-cavity campaign used correct independent readout maps but leaked a 6.00 GHz default into rail-A anchor generation and rail-B reconciliation. Its neighborhood map remains exploratory until corrected neighbors are rerun.", styles["Warning"]),
        PageBreak(),
        Paragraph("1. Why the procedure evolved", styles["Heading1"]),
        table([
            ["Version", "Change", "Reason"],
            ["Independent maps", "Separate rail-A/rail-B readout maps; independent ordered gate map; overlay afterward.", "Idle readout and gate operation may use different EJs."],
            ["Per-point reconciliation", "Check fabrication-fixed hardware at every overlay point.", "One witness cannot certify a region."],
            ["Fixed rail A", "Scan connected rail-A physical-frequency range; retain complete readout anchors; keep EJ-A fixed.", "Emulate one applied rail-A bias from idle to gate."],
            ["Aggregate S1", "Compare all logical sources using whole-gate aggregate S1, then calibrate and evaluate fidelity.", "A clean targeted oscillation can hide spectator mixing."],
            ["Split cavities", "Use 6.05 GHz for A and 5.95 GHz for B; assert provenance downstream.", "Test asymmetric cavity design and prevent silent 6.00 GHz defaults."],
            ["Signed clearance", "Require rail-B idle above every retained CZ interaction by >200 MHz.", "The experimental trajectory tunes downward into the interaction."],
            ["Strict 4D ball", "Require every saved neighbor to pass and share a usable stored-anchor protocol.", "Convert point feasibility into fabrication-neighborhood robustness."],
        ], [26 * mm, 70 * mm, 66 * mm]),
        Spacer(1, 6 * mm),
        Paragraph("Readout and gate maps remain independent throughout. Gate computation is never restricted to currently readout-valid rail-B points. The final full-network resonator check remains collaborator-owned.", styles["Callout"]),
        PageBreak(),
        Paragraph("2. Independent readout preparation", styles["Heading1"]),
        Paragraph("Separate fine C1-C2 maps support the 140-160 MHz rail-A logical band and the 80-100 MHz rail-B band. Direct EC maps use mirrored target construction, a 1.5-scaled hull, full ordered orientation and a 2 MHz EC lattice. Gate symmetry reduction is not used.", styles["Body2"]),
        image(FIGURES / "fine_c1_c2_independent_maps.png", 174 * mm, 100 * mm),
        Spacer(1, 3 * mm),
        image(HERE / "cavity_split_campaign_2026-11-09/readout/comparison/split_cavity_ec_readout_map_comparison.png", 174 * mm, 105 * mm),
        PageBreak(),
        Paragraph("3. Per-rail-A fixed-bias gate workflow", styles["Heading1"]),
        Paragraph("For one ordered rail-A EC pair, the readout optimizer scans physical operating frequency, finds the largest connected validated component, and samples five complete hardware witnesses at configurable fractional positions. Every anchor is screened over all four logical sources. Aggregate S1 ranks clean intended CZ excursions while suppressing unwanted transfer from every logical input. Exact phase-pi calibration and fidelity decide acceptance; ranked fallback is retained because S1 is not a monotonic fidelity predictor.", styles["Body2"]),
        table([
            ["Stage", "Hard requirement / retained record"],
            ["Rail-A readout", "Logical splitting 140-160 MHz; complete EJ, C12, Cg, bare frequencies and cavity provenance"],
            ["Rail-B gate", "All ordered EC-B targets; fB search 4.0-5.5 GHz; representative isolated B splitting 80 MHz"],
            ["CZ channel", "Minimum aggregate-S1 feasible resonance at common intended coupling; all four logical sources compared"],
            ["Calibration", "Exact conditional phase pi; raw >95%; post-selected >99.9%"],
            ["Overlay", "Independent rail-B idle readout AND independent gate acceptance"],
            ["Reconciliation", "Rail-A EJ fixed; rail-B capacitance hardware consistent; rail-B EJ may move"],
        ], [48 * mm, 114 * mm]),
        Spacer(1, 5 * mm),
        image(CORRECTED / "readout_frequency_scan/physical_frequency_feasibility.png", 160 * mm, 103 * mm),
        PageBreak(),
        Paragraph("4. Strict saved-data 4D neighborhood", styles["Heading1"]),
        Paragraph("A 4D center is accepted only when all 233 sampled offsets within radius 5 MHz have complete hard-valid records, a nonempty intersection of connected rail-A operating ranges, stored selected anchors inside that shared range, and one logical source throughout. The gate-frequency jump limit was removed because rail B is tunable; its span remains diagnostic.", styles["Body2"]),
        image(FOUR_D / "four_dimensional_feasibility_projections.png", 174 * mm, 104 * mm),
        Spacer(1, 3 * mm),
        table([
            ["Filtering stage", "Rail-A survivors", "Unique rail-B survivors"],
            ["At least one hard-valid tuple", "556", "2,824"],
            ["Complete hard-valid ball", "262", "1,610"],
            ["Nonempty joint rail-A range", "262", "1,610"],
            ["Stored anchors inside joint range", "2", "980"],
            ["Logical-source-consistent ball", "2", "816"],
        ], [69 * mm, 45 * mm, 48 * mm]),
        PageBreak(),
        Paragraph("5. The two strict rail-A projections", styles["Heading1"]),
        image(FOUR_D / "rail_a_two_surviving_candidates.png", 165 * mm, 145 * mm),
        Spacer(1, 4 * mm),
        Paragraph("Both projections have an individual connected range 4.025-4.650 GHz and a neighborhood-shared range 4.025-4.225 GHz. Their stored selected anchor is 4.100 GHz. The shared width itself is not unique: many rejected candidates have 200 MHz or wider intersections. They fail because their already computed selected anchors do not all lie inside the shared interval.", styles["Body2"]),
        Paragraph("The circles show the rail-A-plane cross-section of the 4D condition. On the 2 MHz lattice, the largest sampled 4D radius is 4.899 MHz and the largest one-axis displacement is 4 MHz.", styles["Warning"]),
        PageBreak(),
        Paragraph("6. Corrected-cavity validation of the preferred tuple", styles["Heading1"]),
        Paragraph("The tuple (154,208,216,252) MHz was recomputed in a fresh directory. Rail-A anchors were solved at a 6.05 GHz cavity; rail-B reconciliation used the independent 5.95 GHz readout witness. The corrected connected range is 4.025-4.725 GHz and fractional anchors are 4.100, 4.225, 4.375, 4.525 and 4.650 GHz. The selected anchor remains 4.100 GHz and the selected logical source is 10 coupled to the B20-like auxiliary.", styles["Body2"]),
        image(CORRECTED / "gate_output/plots/anchor_screening_s1.png", 174 * mm, 103 * mm),
        Spacer(1, 3 * mm),
        table([
            ["Corrected center metric", "Value"],
            ["Raw CZ fidelity", f"{100*corrected['raw_fidelity']:.4f}%"],
            ["Post-selected CZ fidelity", f"{100*corrected['postselected_fidelity']:.4f}%"],
            ["Acceptance", f"{100*corrected['acceptance']:.4f}%"],
            ["Gate duration", f"{corrected['gate_time_ns']:.3f} ns"],
            ["Aggregate S1", f"{corrected['S1']:.6f}"],
            ["Fixed-hardware reconciliation", "Pass"],
        ], [82 * mm, 60 * mm]),
        PageBreak(),
        Paragraph("7. Operating frequencies and tuning burden", styles["Heading1"]),
        image(CORRECTED / "fixed_hardware_reconciliation/plots/rail_b_bare_frequency_shifts.png", 174 * mm, 96 * mm),
        Spacer(1, 4 * mm),
        table([
            ["Frequency quantity", "Value"],
            ["Rail-A gate operating frequency", "4.1000 GHz"],
            ["Rail-A gate bare frequencies", "4.1076, 4.1018 GHz"],
            ["Rail-B idle operating frequency", f"{corrected['rail_b_idle_operating_frequency_ghz']:.4f} GHz"],
            ["Rail-B idle bare frequencies", ", ".join(f"{x:.4f}" for x in corrected['rail_b_idle_bare_frequencies_ghz']) + " GHz"],
            ["Rail-B gate frequency", f"{corrected['gate_frequency_ghz']:.4f} GHz"],
            ["Maximum B bare-frequency shift", f"{corrected['rail_b_maximum_bare_frequency_shift_mhz']:.2f} MHz"],
            ["Gate rail-center B-A detuning", "about 293 MHz"],
            ["Closest gate bare-frequency separation", "about 285 MHz"],
            ["Inter-rail coupling", "5.56 MHz"],
        ], [89 * mm, 67 * mm]),
        Spacer(1, 4 * mm),
        Paragraph("The direct single-excitation separation is large compared with the inter-rail coupling, while the higher-excitation logical-to-auxiliary CZ channel is intentionally resonant. The full resonator-inclusive network may contain additional collisions and is not certified here.", styles["Callout"]),
        PageBreak(),
        Paragraph("8. Conclusions and next actions", styles["Heading1"]),
        Paragraph("The fixed-rail-A workflow now has explicit readout/gate independence, complete hardware anchors, aggregate-S1 channel selection, per-point reconciliation, signed idle clearance and strict 4D saved-data neighborhood checks. The corrected center tuple remains high fidelity and hardware-consistent after cavity provenance is enforced.", styles["Body2"]),
        Paragraph("Recommended next actions", styles["Heading2"]),
        Paragraph("1. If strict neighborhood certification is required, rerun only the corrected-cavity offsets around (154,208,216,252) MHz rather than the entire campaign.<br/>2. Send the promoted complete witness to the collaborator's resonator-inclusive full-network validator, update S1 and dispersiveness, and recompute fidelity.<br/>3. Retain the exact center as the current corrected reference, but do not treat the historical split-cavity 5 MHz projection as final production certification.<br/>4. Add pulse-ramp and decoherence results only through the collaborator's separate workflow.", styles["Body2"]),
        Spacer(1, 5 * mm),
        Paragraph("Primary artifacts", styles["Heading2"]),
        table([
            ["Artifact", "Repository-relative location"],
            ["Detailed campaign history", "Ec_tuple search workflow/SIMULATION_CAMPAIGN_EVOLUTION.md"],
            ["Strict 4D outputs", "Ec_tuple search workflow/four_dimensional_feasibility_joint_anchor_range_5MHz_no_jump_limit_2026-11-12/"],
            ["Corrected center run", "Ec_tuple search workflow/corrected_cavity_strict_tuple_A154_208_B216_252_2026-11-13/"],
            ["Implementation authority", "Ec_tuple search workflow/DEVELOPMENT_MEMORY.md and 2Q-gate_XM/WORKFLOW.md"],
        ], [48 * mm, 114 * mm]),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps({"pdf": str(PDF), "pages_expected": 9}, indent=2))


if __name__ == "__main__":
    # The current promoted report is owned by the corrected low-J candidate
    # generator. Keep this historical entry point as a compatibility wrapper so
    # it cannot silently overwrite the current report with the superseded tuple.
    runpy.run_path(
        str(Path(__file__).with_name("create_promoted_low_j_candidate_report.py")),
        run_name="__main__",
    )
