#!/usr/bin/env python3
"""Create a report-style PDF for the fixed-rail-A anchor workflow campaign."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "sampled_runs" / "tonight_first12_B4_5p5"
SELECTION = CAMPAIGN / "best_configuration_selection"
A4_DETAIL = SELECTION / "A4_detailed_validation"
READOUT = HERE / "readout" / "plots"
OUT = ROOT / "output" / "pdf"
PDF = OUT / "fixed_rail_a_anchor_workflow_report.pdf"


def img(path: Path, width=175*mm, max_height=210*mm):
    with PILImage.open(path) as source:
        w, h = source.size
    height = width * h / w
    if height > max_height:
        height = max_height
        width = height * w / h
    return Image(str(path), width=width, height=height)


def table(rows, widths, font=8, header=True):
    body = [[Paragraph(str(v), STYLES["TableHead"] if header and i == 0 else STYLES["TableCell"]) for v in row] for i, row in enumerate(rows)]
    result = Table(body, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    commands = [
        ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#80909b")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244f73")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f8")])]
    result.setStyle(TableStyle(commands))
    return result


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#6d7880"))
    canvas.drawString(17*mm, 9*mm, "Fixed-rail-A anchor workflow campaign")
    canvas.drawRightString(A4_PAGE[0] - 17*mm, 9*mm, f"Page {doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=22, leading=27, textColor=colors.HexColor("#17324d"), spaceAfter=10))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Heading2"], alignment=TA_CENTER, fontSize=11, leading=15, textColor=colors.HexColor("#52636f"), spaceAfter=12))
styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=9.4, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=10, leading=14, backColor=colors.HexColor("#e8f2f7"), borderColor=colors.HexColor("#6b9abb"), borderWidth=.7, borderPadding=8, spaceAfter=8))
styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontSize=7.4, leading=9))
styles.add(ParagraphStyle(name="TableHead", parent=styles["TableCell"], textColor=colors.white, fontName="Helvetica-Bold"))
STYLES = styles
A4_PAGE = A4


def fallback_statistics():
    """Aggregate anchor-fallback outcomes across the completed 12-run campaign."""
    ranks = {rank: 0 for rank in range(1, 6)}
    total_points = 0
    screened_points = 0
    additional_attempts = 0
    rescued_after_no_calibration = 0
    rescued_after_low_fidelity = 0
    reconciled_rescues = 0

    for repetition in sorted(CAMPAIGN.glob("A*_ecA_*")):
        gate_db = repetition / "gate_output" / "compact_anchor_results.sqlite"
        reconciliation_db = repetition / "fixed_hardware_reconciliation" / "fixed_hardware_results.sqlite"
        if not gate_db.exists():
            continue
        connection = sqlite3.connect(gate_db)
        total_points += connection.execute("SELECT COUNT(*) FROM gate_results").fetchone()[0]
        screened_points += connection.execute(
            "SELECT COUNT(*) FROM gate_results WHERE status != 'no_anchor_passed_screening'"
        ).fetchone()[0]
        additional_attempts += connection.execute(
            """SELECT COALESCE(SUM(CASE
                   WHEN json_array_length(json_extract(result_json, '$.fidelity_attempts')) > 1
                   THEN json_array_length(json_extract(result_json, '$.fidelity_attempts')) - 1
                   ELSE 0 END), 0)
               FROM gate_results"""
        ).fetchone()[0]
        for rank, count in connection.execute(
            """SELECT json_extract(result_json, '$.selected_anchor_rank'), COUNT(*)
               FROM gate_results
               WHERE status = 'calibrated'
               GROUP BY json_extract(result_json, '$.selected_anchor_rank')"""
        ):
            if rank:
                ranks[int(rank)] += count
        for first_status, count in connection.execute(
            """SELECT json_extract(result_json, '$.fidelity_attempts[0].status'), COUNT(*)
               FROM gate_results
               WHERE status = 'calibrated'
                 AND json_extract(result_json, '$.selected_anchor_rank') > 1
               GROUP BY json_extract(result_json, '$.fidelity_attempts[0].status')"""
        ):
            if first_status == "no_exact_pi_calibration":
                rescued_after_no_calibration += count
            else:
                rescued_after_low_fidelity += count
        if reconciliation_db.exists():
            connection.execute("ATTACH DATABASE ? AS fixed_hardware", (str(reconciliation_db),))
            reconciled_rescues += connection.execute(
                """SELECT COUNT(*)
                   FROM gate_results AS g
                   JOIN fixed_hardware.reconciliation AS r
                     USING (ec_b1_mhz, ec_b2_mhz)
                   WHERE json_extract(g.result_json, '$.selected_anchor_rank') > 1
                     AND r.reconciled_pass = 1"""
            ).fetchone()[0]
        connection.close()

    return {
        "ranks": ranks,
        "total_points": total_points,
        "screened_points": screened_points,
        "additional_attempts": additional_attempts,
        "rescued_after_no_calibration": rescued_after_no_calibration,
        "rescued_after_low_fidelity": rescued_after_low_fidelity,
        "reconciled_rescues": reconciled_rescues,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    choices = json.loads((SELECTION / "best_configurations.json").read_text())
    rows = choices["selections"]
    detail = json.loads((A4_DETAIL / "complete_configuration.json").read_text())
    a4 = next(r for r in rows if r["selection_id"] == "A4")
    a4_dir = Path(a4["source_repetition"])
    anchor = json.loads((a4_dir / "anchor_selection.json").read_text())
    campaign = json.loads((a4_dir / "gate_output" / "anchor_campaign_summary.json").read_text())
    reconcile = json.loads((a4_dir / "fixed_hardware_reconciliation" / "summary.json").read_text())
    status = json.loads((CAMPAIGN / "campaign_status.json").read_text())
    fallback = fallback_statistics()
    rescued = sum(fallback["ranks"][rank] for rank in range(2, 6))

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=17*mm, leftMargin=17*mm, topMargin=15*mm, bottomMargin=15*mm, title="Fixed-Rail-A Physical-Frequency Anchor Workflow")
    story = [
        Spacer(1, 18*mm), Paragraph("Fixed-Rail-A Physical-Frequency Anchor Workflow", styles["ReportTitle"]),
        Paragraph("Method, 12-repetition campaign, robust design selection, and A4 time-domain validation", styles["Subtitle"]),
        Spacer(1, 8*mm),
        Paragraph("Executive summary", styles["Heading1"]),
        Paragraph("This campaign evaluates whether rail A can remain at one applied bias from idle readout through the CZ operating point. For each fixed ordered rail-A EC pair, the readout workflow finds a connected interval of viable physical operating frequencies. Five validated rail-A hardware witnesses are sampled from that interval. The gate search then keeps each rail-A EJ pair fixed, scans the entire ordered rail-B EC grid independently, selects the best collision source and anchor, and finally reconciles the selected gate result with the independently generated rail-B readout map.", styles["Body2"]),
        Paragraph("All 12 ordered rail-A repetitions completed over 4,955 rail-B EC targets each. Eight designs contain a selected point with at least 10 MHz geometric feasibility margin. A4, EC-A=(170,250) MHz, provides the strongest overall balance: raw CZ 99.3185%, post-selected CZ 99.99387%, 12.65 MHz margin, and 210.40 MHz maximum rail-B bare-frequency shift.", styles["Callout"]),
        table([["Campaign item", "Result"], ["Ordered rail-A repetitions", "12 (A1/A1s through A6/A6s)"], ["Rail-B targets per repetition", "4,955"], ["Gate frequency bounds for rail B", "4.0 to 5.5 GHz"], ["Fidelity requirements", "Raw >95%; post-selected >99.9%"], ["Robust selection target", "Margin >=10 MHz, then minimum max |bare-frequency shift of B|"], ["Stored output", "Plots plus compact SQLite results; duplicated point JSON pruned"]], [57*mm, 105*mm]),
        PageBreak(),
        Paragraph("1. Why the workflow changed", styles["Heading1"]),
        Paragraph("The earlier workflow allowed both rails to be retuned between idle and gate operation. The revised design goal is more restrictive and experimentally useful: rail A is the lower-physical-frequency rail and should retain the same EJ values at idle and gate operation. Rail B supplies the operating-point motion needed to form the selected logical-to-auxiliary collision.", styles["Body2"]),
        Paragraph("A single rail-A EC pair can have many readout-valid EJ solutions. Therefore, choosing one arbitrary EJ pair would hide sensitivity to rail-A physical frequency. The new workflow represents this freedom explicitly with a small set of complete readout anchors sampled from the largest connected readout-valid physical-frequency interval.", styles["Body2"]),
        table([["Stage", "Technical action", "Reason"],
               ["1. Fix EC-A", "Select one ordered (EC-A1, EC-A2) pair from the independent rail-A readout map.", "Fabrication parameters are fixed per repetition."],
               ["2. Scan physical frequency", "Run readout optimization from 4 to 8 GHz at 25 MHz spacing while enforcing logical A in 140-160 MHz and the established C12/Cg ranges.", "Find all complete rail-A readout witnesses, including EJ, C12 and Cg."],
               ["3. Preserve connectedness", "Identify the largest connected validated frequency component.", "Avoid sampling across an infeasible gap."],
               ["4. Sample anchors", "Use user-configurable fractional positions; this campaign used 0.1, 0.3, 0.5, 0.7 and 0.9.", "Automates sampling before the interval is known."],
               ["5. Independent gate map", "For every ordered EC-B point, screen all anchors and all four logical sources; calibrate ranked candidates.", "Readout at idle and the gate calculation remain independent."],
               ["6. Reconcile hardware", "Keep EC, C12 and Cg consistent with the idle witnesses; rail-A EJ is fixed, while rail-B EJ may move.", "Certify a realizable idle/gate pair at every surviving point."],
               ["7. Rank designs", "Require fidelity, calculate geometric margin, then minimize maximum absolute rail-B bare-frequency shift.", "Balance low tuning with robustness to EC placement."]], [24*mm, 78*mm, 60*mm]),
        PageBreak(),
        Paragraph("2. Independent readout preparation", styles["Heading1"]),
        Paragraph("Separate fine C1-C2 campaigns were retained for the two logical bands. The EC maps use a 1.5-scaled hull and mirrored readout treatment. Gate EC points remain ordered; no gate symmetry reduction is assumed.", styles["Body2"]),
        img(READOUT / "fine_c1_c2_independent_maps.png", 174*mm, 102*mm),
        Spacer(1, 3*mm), img(READOUT / "rail_a_independent_ec_readout_map.png", 83*mm, 94*mm),
        PageBreak(),
        Paragraph("3. Rail-A EC sampling", styles["Heading1"]),
        Paragraph("The first 12 ordered configurations from the prior weekend selection were used. Each base pair and its exact label swap were evaluated separately.", styles["Body2"]),
        img(READOUT / "rail_a_selected_ec_repetitions.png", 174*mm, 148*mm),
        Spacer(1, 2*mm),
        table([["ID", "EC-A (MHz)", "ID", "EC-A (MHz)"], ["A1", "(160,240)", "A1s", "(240,160)"], ["A2", "(155,215)", "A2s", "(215,155)"], ["A3", "(155,260)", "A3s", "(260,155)"], ["A4", "(170,250)", "A4s", "(250,170)"], ["A5", "(160,225)", "A5s", "(225,160)"], ["A6", "(160,250)", "A6s", "(250,160)"]], [22*mm, 55*mm, 22*mm, 55*mm]),
        PageBreak(),
        Paragraph("4. Example anchor construction: A4", styles["Heading1"]),
        Paragraph(f"For A4, the largest sampled connected readout-valid interval is {anchor['largest_sampled_component_ghz'][0]:.3f} to {anchor['largest_sampled_component_ghz'][1]:.3f} GHz. The five validated anchors are 4.775, 4.800, 4.825, 4.850 and 4.875 GHz. Each anchor retains its complete readout witness and fixed rail-A EJ pair.", styles["Body2"]),
        img(a4_dir / "readout_frequency_scan" / "physical_frequency_feasibility.png", 170*mm, 100*mm),
        Spacer(1, 3*mm),
        Paragraph("The gate stage screened all five anchors at every EC-B point. Anchor 4.875 GHz dominated the calibrated map, but other anchors remained available and were selected where their source-specific score and calibrated fidelity were better.", styles["Body2"]),
        table([["A4 gate-map outcome", "Count"], ["Calibrated and fidelity-passing", f"{campaign['gate_pass']:,}"], ["Readout/gate overlay", f"{campaign['overlay_counts']['overlay']:,}"], ["Gate-only", f"{campaign['overlay_counts']['gate_only']:,}"], ["Readout-only", f"{campaign['overlay_counts']['readout_only']:,}"], ["Neither", f"{campaign['overlay_counts']['neither']:,}"]], [80*mm, 45*mm]),
        PageBreak(),
        Paragraph("Anchor-ranking fallback check", styles["Heading1"]),
        Paragraph("The gate evaluator ranks the five complete rail-A readout anchors by the collision-screening score, attempts full CZ calibration at the best-ranked anchor, and falls back only when that candidate cannot calibrate or misses the fidelity thresholds. A retrospective check across all 12 repetitions shows that the score is a useful screen but is not a monotonic predictor of calibrated fidelity.", styles["Body2"]),
        table([["Selected anchor rank", "Fidelity-passing points", "Interpretation"],
               ["1 (best score)", f"{fallback['ranks'][1]:,}", "Passed without fallback"],
               ["2", f"{fallback['ranks'][2]:,}", "Rescued by first fallback"],
               ["3", f"{fallback['ranks'][3]:,}", "Rescued by second fallback"],
               ["4", f"{fallback['ranks'][4]:,}", "Rescued by third fallback"],
               ["5", f"{fallback['ranks'][5]:,}", "Rescued by final fallback"]], [44*mm, 49*mm, 69*mm]),
        Spacer(1, 4*mm),
        table([["Fallback diagnostic", "Measured result"],
               ["Rail-B points evaluated across 12 repetitions", f"{fallback['total_points']:,}"],
               ["Additional fidelity-passing points recovered", f"{rescued:,}"],
               ["Rescues after rank 1 had no exact pi calibration", f"{fallback['rescued_after_no_calibration']:,}"],
               ["Rescues after rank 1 calibrated below threshold", f"{fallback['rescued_after_low_fidelity']:,}"],
               ["Fallback rescues surviving readout overlay and reconciliation", f"{fallback['reconciled_rescues']:,}"],
               ["Additional full-fidelity attempts caused by fallback", f"{fallback['additional_attempts']:,}"]], [100*mm, 62*mm]),
        Spacer(1, 5*mm),
        Paragraph(f"Fallback recovered {rescued:,} gate-valid points, including {fallback['reconciled_rescues']:,} points that remained valid after the independent rail-B readout overlay and fixed-hardware reconciliation. Most rescues followed a failure to obtain exact pi calibration at the lowest-score anchor. This confirms that a low screening score does not by itself guarantee a calibratable, fidelity-passing CZ solution. The production workflow therefore retains all five fallback opportunities.", styles["Callout"]),
        Paragraph("The extra attempts are a small fraction of the total gate workload, so removing fallback would save only modest runtime while creating holes in the final reconciled feasibility regions and potentially reducing neighborhood robustness margins.", styles["Body2"]),
        PageBreak(),
        Paragraph("5. Independent gate map and overlay: A4", styles["Heading1"]),
        img(a4_dir / "gate_output" / "plots" / "selected_anchor_gate_maps.png", 175*mm, 108*mm),
        Spacer(1, 3*mm),
        img(a4_dir / "gate_output" / "plots" / "readout_gate_overlay.png", 145*mm, 95*mm),
        PageBreak(),
        Paragraph("6. Fixed-hardware reconciliation", styles["Heading1"]),
        Paragraph(f"A4 has {campaign['overlay_counts']['overlay']:,} independent readout/gate overlay points. Per-point reconciliation retains {reconcile['reconciled_pass']:,} and rejects {reconcile['reconciled_fail']:,}. Rail-A fixed-bias audits pass at all reconciled candidates; failures arise from the rail-B readout realization and its constraints.", styles["Body2"]),
        img(a4_dir / "fixed_hardware_reconciliation" / "plots" / "rail_b_bare_frequency_shifts.png", 175*mm, 91*mm),
        Spacer(1, 3*mm),
        img(a4_dir / "fixed_hardware_reconciliation" / "plots" / "rail_b_operating_frequencies.png", 175*mm, 91*mm),
        PageBreak(),
        Paragraph("7. Selection across all 12 repetitions", styles["Heading1"]),
        Paragraph("A geometric margin is the Euclidean distance on the 2 MHz EC-B lattice to the nearest point outside the fidelity-passing reconciled region, including the sampled-domain exterior. A point must first satisfy both fidelity thresholds. The preferred pool has margin at least 10 MHz; within it, the maximum absolute rail-B bare-frequency shift is minimized.", styles["Body2"]),
        img(SELECTION / "best_configurations_on_rail_a_readout_map.png", 174*mm, 145*mm),
        PageBreak(),
        Paragraph("8. Selected configurations", styles["Heading1"]),
        img(SELECTION / "best_configurations_table.png", 174*mm, 73*mm),
        Spacer(1, 4*mm),
        Paragraph("A3, A3s, A6 and A6s have no fidelity-passing candidate with 10 MHz margin. Their reported points are best-available fallbacks and are marked with asterisks/crosses. A4 and A4s have nearly identical outcomes, supporting weak sensitivity to the rail-A label ordering at this EC pair. A2 is more robust but requires substantially greater rail-B tuning; A3 and A6 tune less but lie close to the feasibility boundary.", styles["Body2"]),
        Paragraph("A4 is therefore the most balanced candidate rather than the optimum of any single scalar metric.", styles["Callout"]),
        PageBreak(),
        Paragraph("9. Complete selected A4 configuration", styles["Heading1"]),
    ]

    ra, rb = detail["rail_A"], detail["rail_B"]
    idle, gate = detail["interrail_detuning"]["idle"], detail["interrail_detuning"]["gate"]
    story += [
        table([["Quantity", "Rail A", "Rail B"],
               ["EC/h (MHz)", str(tuple(ra["ec_mhz"])), str(tuple(rb["ec_mhz"]))],
               ["Idle EJ/h (GHz)", ", ".join(f"{v:.6f}" for v in ra["ej_idle_and_gate_ghz"]), ", ".join(f"{v:.6f}" for v in rb["idle_ej_ghz"])],
               ["Gate EJ/h (GHz)", "unchanged", ", ".join(f"{v:.6f}" for v in rb["gate_ej_ghz"])],
               ["Idle bare f01 (GHz)", ", ".join(f"{v:.6f}" for v in ra["bare_f01_idle_and_gate_ghz"]), ", ".join(f"{v:.6f}" for v in rb["idle_bare_f01_ghz"])],
               ["Gate bare f01 (GHz)", "unchanged", ", ".join(f"{v:.6f}" for v in rb["gate_bare_f01_ghz"])],
               ["Readout operating frequency (GHz)", f"{ra['readout_operating_frequency_ghz']:.6f}", f"{rb['idle_readout_operating_frequency_ghz']:.6f}"],
               ["Readout logical frequency (MHz)", f"{ra['readout_logical_frequency_mhz']:.3f}", f"{rb['idle_readout_logical_frequency_mhz']:.3f}"],
               ["Readout C12 (fF)", f"{ra['readout_C12_fF']:.4f}", f"{rb['idle_readout_device']['C12_fF']:.4f}"],
               ["Readout Cg per transmon (fF)", f"{ra['readout_Cg_fF']:.4f}", f"{rb['idle_readout_device']['C1r_fF']:.4f}"]], [58*mm, 52*mm, 52*mm]),
        Spacer(1, 4*mm),
        table([["Bare-frequency detuning", "Idle", "Gate"], ["Rail-center B-A (MHz)", f"{idle['center_B_minus_A_mhz']:.2f}", f"{gate['center_B_minus_A_mhz']:.2f}"], ["Closest inter-rail separation (MHz)", f"{idle['closest_absolute_interrail_detuning_mhz']:.2f}", f"{gate['closest_absolute_interrail_detuning_mhz']:.2f}"], ["Same-index B1-A1 (MHz)", f"{idle['same_index_B_minus_A_mhz'][0]:.2f}", f"{gate['same_index_B_minus_A_mhz'][0]:.2f}"], ["Same-index B2-A2 (MHz)", f"{idle['same_index_B_minus_A_mhz'][1]:.2f}", f"{gate['same_index_B_minus_A_mhz'][1]:.2f}"]], [70*mm, 45*mm, 45*mm]),
        Spacer(1, 4*mm),
        Paragraph("Rail A remains at the same EJ and bare transmon frequencies. Rail B shifts downward by 210.40 and 207.97 MHz. The closest bare-frequency separation decreases from 431.87 MHz at idle to 223.90 MHz at the gate point.", styles["Callout"]),
        PageBreak(),
        Paragraph("10. Exact static-gate time evolution", styles["Heading1"]),
        Paragraph("The selected A4 point dynamically couples logical source |10> to the dressed auxiliary state |~0020>. The exact 256-state static Hamiltonian was reconstructed from the archived EJ and coupling values. Population transfers almost completely to the auxiliary state near half the interaction time, returns at 98.726 ns, and accumulates a conditional phase of -pi.", styles["Body2"]),
        img(A4_DETAIL / "gate_time_evolution.png", 175*mm, 121*mm),
        Spacer(1, 3*mm),
        table([["Verification metric", "Archived", "Reconstructed"], ["Raw CZ fidelity", f"{100*detail['gate']['archived']['raw_fidelity']:.6f}%", f"{100*detail['gate']['reconstructed']['raw_fidelity']:.6f}%"], ["Post-selected CZ fidelity", f"{100*detail['gate']['archived']['postselected_fidelity']:.6f}%", f"{100*detail['gate']['reconstructed']['postselected_fidelity']:.6f}%"], ["Acceptance", f"{100*detail['gate']['archived']['acceptance']:.6f}%", f"{100*detail['gate']['reconstructed']['acceptance']:.6f}%"], ["Conditional phase error", f"{detail['gate']['archived']['phase_error_rad']:.3e} rad", f"{detail['gate']['reconstructed']['phase_error_rad']:.3e} rad"]], [75*mm, 44*mm, 44*mm]),
        PageBreak(),
        Paragraph("11. Interpretation and next steps", styles["Heading1"]),
        Paragraph("The campaign demonstrates that fixing rail-A EJ is compatible with high-fidelity CZ solutions over a nontrivial final feasibility region. The main design tradeoff is between the rail-B tuning excursion and geometric margin. A4 is a strong next-stage candidate because it retains 12.65 MHz margin, avoids the low-margin behavior of A3/A6, and reduces the tuning burden relative to A2.", styles["Body2"]),
        Paragraph("The time-evolution calculation verifies the static gate model used in the map, but it is not yet a complete pulse-level or resonator-inclusive certification. The model assumes an ideal static interaction hold at the calibrated point. Ramp dynamics, flux-line distortions, decoherence, and the full two-resonator network remain outside this report.", styles["Body2"]),
        Paragraph("Recommended next checks", styles["Heading2"]),
        Paragraph("1. Pass the complete A4 hardware witness to the collaborator's full-network validator. Include both resonators, update S1, and recompute the CZ result.<br/>2. Add finite ramps and evaluate nonadiabatic leakage around the 98.726 ns interaction window.<br/>3. Test fabrication and bias perturbations around A4 and A4s, especially near the 12.65 MHz EC-B margin.<br/>4. Repeat the anchor scan with a finer physical-frequency mesh around 4.875 GHz if A4 is promoted to fabrication-level optimization.", styles["Body2"]),
        Spacer(1, 5*mm),
        Paragraph("Data provenance", styles["Heading2"]),
        table([["Artifact", "Repository-relative location"], ["Campaign status and 12 repetitions", "weekend_simulation/sampled_runs/tonight_first12_B4_5p5/"], ["Best-configuration JSON/CSV and maps", "weekend_simulation/sampled_runs/tonight_first12_B4_5p5/best_configuration_selection/"], ["Complete A4 configuration", ".../best_configuration_selection/A4_detailed_validation/complete_configuration.json"], ["Time-series data", ".../best_configuration_selection/A4_detailed_validation/gate_time_evolution.npz"], ["Gate/reconciliation databases", "Each repetition: gate_output/compact_anchor_results.sqlite and fixed_hardware_reconciliation/fixed_hardware_results.sqlite"]], [52*mm, 110*mm], font=7),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps({"pdf": str(PDF), "pages_expected": 13, "repetitions": len(status["jobs"])}, indent=2))


if __name__ == "__main__":
    main()
