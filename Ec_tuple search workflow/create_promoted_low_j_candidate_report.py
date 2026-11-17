#!/usr/bin/env python3
"""Create the current report for the promoted corrected low-J candidate."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "output/pdf"
PDF = OUT / "current_fixed_a_strict_4d_campaign_report.pdf"
FIGURES = OUT / "combined_feasibility_report_figures"
CANDIDATE = HERE / "corrected_cavity_full_neighborhood_A158_240_B158_226_2026-11-17"
ANALYSIS = CANDIDATE / "candidate_analysis"
CENTER_REP = CANDIDATE / "repetitions/A_11_ecA_158_240"
CROSS = HERE / "relaxed_adaptive_4d_best_case_2026-11-17/three_tuple_orthogonal_cross_sections"
HISTORICAL = HERE / "relaxed_adaptive_4d_best_case_2026-11-17/relaxed_centers.csv"


def image(path: Path, width=174 * mm, max_height=220 * mm):
    with PILImage.open(path) as source:
        w, h = source.size
    height = width * h / w
    if height > max_height:
        height = max_height
        width = height * w / h
    return Image(str(path), width=width, height=height)


def crop_comparison():
    FIGURES.mkdir(parents=True, exist_ok=True)
    source_path = CROSS / "three_tuple_best_case_A_and_B_cross_sections.png"
    with PILImage.open(source_path) as source:
        w, h = source.size
        # Remove a little title/legend whitespace while preserving panel labels.
        top = source.crop((0, int(0.025 * h), w, int(0.465 * h)))
        bottom = source.crop((0, int(0.515 * h), w, int(0.955 * h)))
        top.save(FIGURES / "three_tuple_fixed_a_cross_sections.png")
        bottom.save(FIGURES / "three_tuple_fixed_b_cross_sections.png")


def table(rows, widths, font=7.5):
    body = [
        [Paragraph(str(value), STYLES["TableHead"] if index == 0 else STYLES["TableCell"]) for value in row]
        for index, row in enumerate(rows)
    ]
    result = Table(body, colWidths=widths, repeatRows=1, hAlign="CENTER")
    result.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#81909a")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173f5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return result


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#687680"))
    canvas.drawString(17 * mm, 9 * mm, "Corrected-cavity fixed-rail-A EC optimization")
    canvas.drawRightString(A4[0] - 17 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
                          fontSize=20, leading=24, textColor=colors.HexColor("#17324d"), spaceAfter=8))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["BodyText"], alignment=TA_CENTER,
                          fontSize=10.3, leading=14, textColor=colors.HexColor("#52636f"), spaceAfter=12))
styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=9.2, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=9.8, leading=13.5,
                          backColor=colors.HexColor("#e7f1f6"), borderColor=colors.HexColor("#6e9bb7"),
                          borderWidth=.7, borderPadding=8, spaceAfter=8))
styles.add(ParagraphStyle(name="Warning", parent=styles["BodyText"], fontSize=9.1, leading=12.5,
                          backColor=colors.HexColor("#fff2d5"), borderColor=colors.HexColor("#d29b3d"),
                          borderWidth=.7, borderPadding=7, spaceAfter=8))
styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontSize=7.4, leading=9.1))
styles.add(ParagraphStyle(name="TableHead", parent=styles["TableCell"], textColor=colors.white,
                          fontName="Helvetica-Bold"))
STYLES = styles


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crop_comparison()
    complete = json.loads((ANALYSIS / "complete_corrected_configuration.json").read_text())
    metrics = complete["corrected_neighborhood_metrics"]
    gate = complete["gate_result"]
    derived = complete["derived_physical_metrics"]
    anchor = complete["selected_anchor"]
    j = metrics["required_interrail_coupling_mhz"]

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
                            topMargin=16 * mm, bottomMargin=15 * mm)
    story = [
        Paragraph("Corrected-Cavity Four-EC Candidate Optimization", STYLES["ReportTitle"]),
        Paragraph("Promotion of EC=(158,240,158,226) MHz after full 5 MHz neighborhood validation",
                  STYLES["Subtitle"]),
        Paragraph(
            "The preferred toy-model design is now <b>EC=(158,240,158,226) MHz</b>. "
            "A fresh corrected-cavity campaign evaluates every one of the 233 lattice points in its "
            "5 MHz-radius four-dimensional neighborhood. All 233 pass gate thresholds and all 233 "
            "pass fixed-hardware reconciliation, with one consistent 10-to-B20 CZ channel.",
            STYLES["Callout"],
        ),
        table([
            ["Promoted result", "Corrected value"],
            ["Cavity provenance", "Rail A 6.05 GHz; rail B 5.95 GHz"],
            ["Center raw / post-selected CZ", f"{100*gate['raw_fidelity']:.4f}% / {100*gate['postselected_fidelity']:.4f}%"],
            ["Center gate duration", f"{gate['time_ns']:.3f} ns"],
            ["Center J_req/h", f"{1000*gate['couplings_GHz']['A2_B1']:.4f} MHz"],
            ["Neighborhood J_req/h", f"{j['minimum']:.4f}-{j['maximum']:.4f} MHz; median {j['median']:.4f} MHz"],
            ["Neighborhood gate time", f"{metrics['gate_time_ns']['minimum']:.3f}-{metrics['gate_time_ns']['maximum']:.3f} ns"],
            ["Gate / reconciliation pass", "233/233 / 233/233"],
            ["Selected channel", "logical 10 to B20-like |~0020>"],
        ], [62 * mm, 100 * mm]),
        Spacer(1, 6 * mm),
        Paragraph("Scope", STYLES["Heading2"]),
        Paragraph(
            "This report covers independent readout witnesses, the static toy CZ model, aggregate-S1 "
            "channel selection, calibrated fidelity, signed idle clearance and fixed-hardware "
            "reconciliation. The collaborator-owned resonator-inclusive full-network validation, "
            "pulse ramps and decoherence remain separate downstream checks.", STYLES["Warning"]),
        PageBreak(),
        Paragraph("1. Why the selection criterion changed", STYLES["Heading1"]),
        Paragraph(
            "The initial strict saved-data analysis emphasized a complete 4D neighborhood, common "
            "stored anchor and common logical source. The later calibration-adaptive interpretation "
            "retains the physical hard checks but permits each fabricated tuple to use its own valid "
            "bias calibration. The first relaxed ranking maximized radius and only constrained the "
            "relative spread of J_req. That produced EC=(234,160,240,170) MHz, but did not penalize "
            "the absolute physical coupling required to realize the common 5 MHz intended matrix element.",
            STYLES["Body2"]),
        Paragraph(
            "For a common target effective coupling, J_req is a hardware demand rather than a simulated "
            "speed difference. If two devices instead use the same fixed J_hardware, effective coupling "
            "is proportional to J_hardware/J_req and gate duration is proportional to J_req. Absolute "
            "J_req must therefore be included when comparing realizable designs.", STYLES["Callout"]),
        table([
            ["Tuple", "Role", "Neighborhood median J_req/h", "5 MHz status", "Interpretation"],
            ["(154,208,216,252)", "Previous reference", "5.558 MHz", "Historical then corrected reference", "Lower radius; higher coupling"],
            ["(234,160,240,170)", "Radius-first relaxed optimum", "5.876 MHz", "Historical", "Largest radius; highest coupling"],
            ["(158,240,158,226)", "Promoted low-J design", f"{j['median']:.3f} MHz", "Fully corrected: 233/233", "Best coupling/robustness balance"],
        ], [37 * mm, 39 * mm, 33 * mm, 30 * mm, 35 * mm]),
        Spacer(1, 5 * mm),
        Paragraph(
            f"At common J_hardware, the promoted neighborhood median implies about "
            f"{100*(1-j['median']/5.557553204051501):.1f}% less interaction time than the previous "
            f"reference and {100*(1-j['median']/5.875601639708695):.1f}% less than the radius-first design.",
            STYLES["Body2"]),
        PageBreak(),
        Paragraph("2. Fixed-rail-A cross-sections through the three tuples", STYLES["Heading1"]),
        Paragraph(
            "Each column fixes the rail-A EC pair of one candidate and scans the complete rail-B grid. "
            "Unlike a collapsed projection, every colored point is one definite four-valued EC tuple. "
            "The star is the selected rail-B point and the dashed circle shows a 5 MHz rail-plane section.",
            STYLES["Body2"]),
        image(FIGURES / "three_tuple_fixed_a_cross_sections.png", 174 * mm, 215 * mm),
        PageBreak(),
        Paragraph("3. Fixed-rail-B cross-sections through the three tuples", STYLES["Heading1"]),
        Paragraph(
            "The complementary rows fix each selected rail-B pair and scan rail A. These two orthogonal "
            "views show that the promoted low-J point is locally smooth in both rail subspaces; the sharp "
            "edge in the earlier 2D projection was caused by changing the hidden representative pair.",
            STYLES["Body2"]),
        image(FIGURES / "three_tuple_fixed_b_cross_sections.png", 174 * mm, 215 * mm),
        PageBreak(),
        Paragraph("4. Corrected-cavity neighborhood result", STYLES["Heading1"]),
        Paragraph(
            "The corrected campaign uses the independent 6.05 GHz rail-A and 5.95 GHz rail-B readout "
            "datasets throughout anchor generation, gate calculation and reconciliation. It contains "
            "233 offsets on the 2 MHz lattice inside a 5 MHz four-dimensional ball, with no missing records.",
            STYLES["Body2"]),
        image(ANALYSIS / "corrected_neighborhood_metric_distributions.png", 170 * mm, 128 * mm),
        Spacer(1, 3 * mm),
        table([
            ["Neighborhood metric", "Minimum", "Median", "Maximum"],
            ["Raw fidelity", f"{100*metrics['raw_fidelity']['minimum']:.4f}%", f"{100*metrics['raw_fidelity']['median']:.4f}%", f"{100*metrics['raw_fidelity']['maximum']:.4f}%"],
            ["Post-selected fidelity", f"{100*metrics['postselected_fidelity']['minimum']:.4f}%", f"{100*metrics['postselected_fidelity']['median']:.4f}%", f"{100*metrics['postselected_fidelity']['maximum']:.4f}%"],
            ["Gate time", f"{metrics['gate_time_ns']['minimum']:.3f} ns", f"{metrics['gate_time_ns']['median']:.3f} ns", f"{metrics['gate_time_ns']['maximum']:.3f} ns"],
            ["J_req/h", f"{j['minimum']:.4f} MHz", f"{j['median']:.4f} MHz", f"{j['maximum']:.4f} MHz"],
            ["Aggregate S1", f"{metrics['aggregate_s1']['minimum']:.5f}", f"{metrics['aggregate_s1']['median']:.5f}", f"{metrics['aggregate_s1']['maximum']:.5f}"],
            ["Maximum logical ratio", f"{metrics['maximum_logical_ratio']['minimum']:.4f}", f"{metrics['maximum_logical_ratio']['median']:.4f}", f"{metrics['maximum_logical_ratio']['maximum']:.4f}"],
        ], [52 * mm, 37 * mm, 37 * mm, 37 * mm]),
        PageBreak(),
        Paragraph("5. Corrected rail-A anchor and CZ calibration", STYLES["Heading1"]),
        image(CENTER_REP / "readout_frequency_scan/physical_frequency_feasibility.png", 170 * mm, 102 * mm),
        Spacer(1, 3 * mm),
        image(CENTER_REP / "gate_output/plots/anchor_screening_s1.png", 170 * mm, 103 * mm),
        PageBreak(),
        Paragraph("6. Complete center hardware witness", STYLES["Heading1"]),
        table([
            ["Center quantity", "Corrected value"],
            ["EC tuple", "(158,240,158,226) MHz"],
            ["Selected rail-A anchor", f"{anchor['readout_operating_frequency_GHz']:.4f} GHz ({anchor['anchor_id']})"],
            ["Rail-A readout / gate logical splitting", f"{1000*anchor['readout_logical_frequency_GHz']:.3f} / {1000*anchor['gate_logical_frequency_GHz']:.3f} MHz"],
            ["Rail-A EJ/h", ", ".join(f"{value:.6f}" for value in anchor['fixed_A']['EJ_GHz']) + " GHz"],
            ["Rail-A C12 / Cg", f"{anchor['fixed_A']['readout_C12_fF']:.4f} / {anchor['fixed_A']['readout_Cg_fF']:.4f} fF"],
            ["Rail-A bare f01", ", ".join(f"{value:.6f}" for value in anchor['bare_f01_GHz']) + " GHz"],
            ["Rail-B idle operating frequency", f"{derived['rail_b_idle_operating_frequency_ghz']:.6f} GHz"],
            ["Rail-B gate operating frequency", f"{derived['rail_b_gate_operating_frequency_ghz']:.6f} GHz"],
            ["Rail-B idle bare f01", ", ".join(f"{value:.6f}" for value in derived['rail_b_idle_bare_frequencies_ghz']) + " GHz"],
            ["Rail-B gate bare f01", ", ".join(f"{value:.6f}" for value in derived['rail_b_gate_bare_frequencies_ghz']) + " GHz"],
            ["Rail-B bare-frequency shifts", ", ".join(f"{value:.2f}" for value in derived['rail_b_bare_frequency_shifts_gate_minus_idle_mhz']) + " MHz"],
            ["Idle B-above-A bare detuning", f"{derived['minimum_rail_a_b_idle_bare_detuning_mhz']:.2f} MHz"],
            ["Closest gate bare separation", f"{derived['minimum_absolute_rail_a_b_gate_bare_separation_mhz']:.2f} MHz"],
            ["Fixed-hardware reconciliation", "Pass"],
        ], [70 * mm, 92 * mm]),
        Spacer(1, 5 * mm),
        Paragraph(
            "Rail A keeps the same applied EJ values from idle to gate. Rail B is tuned downward by about "
            "466-469 MHz in bare frequency. The direct gate-point bare separation remains about 214 MHz, "
            "well above the 4.37 MHz inter-rail coupling, while the intended higher-excitation CZ channel "
            "is deliberately brought into resonance.", STYLES["Callout"]),
        PageBreak(),
        Paragraph("7. Recommendation and retained artifacts", STYLES["Heading1"]),
        Paragraph(
            "EC=(158,240,158,226) MHz is the current preferred toy-model design. It combines a fully "
            "passing corrected 5 MHz neighborhood with substantially lower required physical coupling, "
            "high fidelity, one stable CZ partner and a smooth calibrated gate-time distribution. The "
            "previous tuple remains a useful corrected reference; the radius-first tuple remains a "
            "robustness comparison, but neither provides the same coupling advantage.", STYLES["Callout"]),
        table([
            ["Artifact", "Repository-relative location"],
            ["Corrected neighborhood", "Ec_tuple search workflow/corrected_cavity_full_neighborhood_A158_240_B158_226_2026-11-17/"],
            ["Complete center record", ".../candidate_analysis/complete_corrected_configuration.json"],
            ["Neighborhood summary", ".../corrected_neighborhood_summary.json"],
            ["Three-tuple comparison", ".../relaxed_adaptive_4d_best_case_2026-11-17/three_tuple_orthogonal_cross_sections/"],
            ["Campaign history", "Ec_tuple search workflow/SIMULATION_CAMPAIGN_EVOLUTION.md"],
            ["Implementation memory", "Ec_tuple search workflow/DEVELOPMENT_MEMORY.md"],
        ], [52 * mm, 110 * mm]),
        Spacer(1, 5 * mm),
        Paragraph(
            "Next: pass the complete corrected witness to the collaborator's resonator-inclusive full-network "
            "validator, update dispersiveness and aggregate S1 in that network, and only then promote the "
            "design beyond the present toy-model campaign.", STYLES["Warning"]),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps({"pdf": str(PDF), "pages_expected": 8}, indent=2))


if __name__ == "__main__":
    main()
