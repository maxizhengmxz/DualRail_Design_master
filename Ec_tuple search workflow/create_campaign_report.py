#!/usr/bin/env python3
"""Create the combined-feasibility campaign report and companion Markdown note."""

from __future__ import annotations

import json, sqlite3, statistics, shutil
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE/"combined_runs/production_logical_A140_B80"
MANIFEST=HERE/"selections/rail_a_ec_selections.json"
OUT=ROOT/"output/pdf"
FIG=OUT/"combined_feasibility_report_figures"
PDF=OUT/"combined_feasibility_campaign_report.pdf"
NOTE=OUT/"combined_feasibility_campaign_report.md"
READOUT_PLOTS=HERE/"readout/plots"
PREPARATORY={
    "fine":READOUT_PLOTS/"fine_c1_c2_independent_maps.png",
    "mirrored":READOUT_PLOTS/"fine_c1_c2_mirrored_maps.png",
    "ec_a":READOUT_PLOTS/"rail_a_independent_ec_readout_map.png",
    "ec_b":READOUT_PLOTS/"rail_b_independent_ec_readout_map.png",
    "selection":READOUT_PLOTS/"rail_a_selected_ec_repetitions.png",
}


def load():
    manifest=json.loads(MANIFEST.read_text()); reps=[]; points=[]
    for selection in manifest["selections"]:
        sid=selection["selection_id"];a=tuple(selection["ec_a_mhz"]);directory=CAMPAIGN/f"{sid}_ecA_{a[0]}_{a[1]}"
        connection=sqlite3.connect(directory/"fixed_hardware_reconciliation/fixed_hardware_results.sqlite")
        records=connection.execute("SELECT ec_b1_mhz,ec_b2_mhz,reconciled_pass,details_json FROM reconciliation").fetchall();connection.close()
        failed_a=Counter();failed_b=Counter();passed=[]
        for b1,b2,ok,payload in records:
            row=json.loads(payload)
            if ok:
                idle=[row["rail_a"]["point"]["ej1_ghz"],row["rail_a"]["point"]["ej2_ghz"],row["rail_b"]["point"]["ej1_ghz"],row["rail_b"]["point"]["ej2_ghz"]]
                gate=row["gate"]["EJ_GHz"];delta=[gate[i]-idle[i] for i in range(4)]
                point={"id":sid,"ec_a":a,"ec_b":(int(b1),int(b2)),"delta":delta,
                       "a_tune":sum(abs(x) for x in delta[:2]),"b_tune":sum(abs(x) for x in delta[2:]),
                       "raw":row["gate"]["raw_fidelity"],"post":row["gate"]["postselected_fidelity"],
                       "idle_freq":(row["rail_a"]["point"]["operating_frequency_ghz"],row["rail_b"]["point"]["operating_frequency_ghz"]),
                       "gate_freq":(row["gate"].get("fA_GHz"),row["gate"].get("fB_GHz"))}
                point["dominance"]=point["b_tune"]/max(point["a_tune"],1e-12);passed.append(point);points.append(point)
            else:
                failed_a.update(row["rail_a"]["failures"]);failed_b.update(row["rail_b"]["failures"])
        reps.append({"id":sid,"ec":a,"group":selection.get("selection_group","original"),"directory":directory,
                     "overlay":len(records),"pass":len(passed),"fraction":len(passed)/len(records),"points":passed,
                     "min_a":min((x["a_tune"] for x in passed),default=np.nan),
                     "median_a":statistics.median([x["a_tune"] for x in passed]) if passed else np.nan,
                     "fail_a":failed_a,"fail_b":failed_b})
    return manifest,reps,points


def summary_maps(reps):
    FIG.mkdir(parents=True,exist_ok=True)
    x=np.array([r["ec"][0] for r in reps]);y=np.array([r["ec"][1] for r in reps])
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout="constrained")
    plots=[([100*r["fraction"] for r in reps],"Final reconciled fraction (%)","viridis",0,80),
           ([r["min_a"] for r in reps],r"Minimum $|\Delta E_{J,A1}|+|\Delta E_{J,A2}|$ (GHz)","magma_r",1.8,5)]
    for ax,(value,label,cmap,vmin,vmax) in zip(axes,plots,strict=True):
        im=ax.scatter(x,y,c=value,s=58,marker="s",cmap=cmap,vmin=vmin,vmax=vmax,edgecolors="white",linewidths=.4)
        for r in reps:
            if r["id"] in ("A30","A10","A27","A11","A14","A15"): ax.annotate(r["id"],r["ec"],xytext=(3,3),textcoords="offset points",fontsize=7)
        ax.set(xlabel=r"$E_{C,A1}/h$ (MHz)",ylabel=r"$E_{C,A2}/h$ (MHz)");ax.set_aspect("equal");fig.colorbar(im,ax=ax,label=label)
    axes[0].set_title("Final fixed-hardware feasible fraction")
    axes[1].set_title("Best rail-A tuning at each repetition")
    path=FIG/"rail_a_global_summary.png";fig.savefig(path,dpi=220);plt.close(fig);return path


def recommendation_plot(shortlist):
    labels=[f"{p['id']}\nA{p['ec_a']}\nB{p['ec_b']}" for p in shortlist]
    a=[p["a_tune"] for p in shortlist];b=[p["b_tune"] for p in shortlist]
    pos=np.arange(len(labels));fig,ax=plt.subplots(figsize=(9,4.8),layout="constrained")
    width=.36;ax.bar(pos-width/2,a,width,label="Rail A total |Delta EJ|",color="#457b9d");ax.bar(pos+width/2,b,width,label="Rail B total |Delta EJ|",color="#e76f51")
    ax.set_xticks(pos,labels);ax.set_ylabel("Summed absolute EJ shift (GHz)");ax.set_title("Representative feasible designs: tuning burden");ax.legend();ax.grid(axis="y",alpha=.25)
    path=FIG/"representative_tuning_burden.png";fig.savefig(path,dpi=220);plt.close(fig);return path


def failure_plot(reps):
    selected=[r for r in reps if r["id"] in ("A14","A14s","A15","A15s")]
    reasons=("dispersive","hybridization","logical_frequency","chi_matching","absolute_mean_chi","EJ_over_EC")
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax,rail,key in ((axes[0],"Rail A","fail_a"),(axes[1],"Rail B","fail_b")):
        width=.19;pos=np.arange(len(reasons))
        for index,r in enumerate(selected):
            vals=[100*r[key][reason]/r["overlay"] for reason in reasons]
            ax.bar(pos+(index-1.5)*width,vals,width,label=r["id"])
        ax.set_xticks(pos,[x.replace("_","\n") for x in reasons],fontsize=8);ax.set_ylabel("Occurrences per failed overlay point (%)");ax.set_title(f"{rail} failure criteria");ax.legend();ax.grid(axis="y",alpha=.2)
    path=FIG/"zero_region_failure_reasons.png";fig.savefig(path,dpi=220);plt.close(fig);return path


def workflow_plot():
    labels=["Separate fine C1-C2\nreadout campaigns","1.5 scaled, mirrored\nEC exploration hulls","Independent readout\nand gate maps","Boolean overlay at\nevery ordered EC point","Fixed-hardware\nreconciliation","Collaborator: resonator-\ninclusive full network"]
    fig,ax=plt.subplots(figsize=(12,2.6),layout="constrained");ax.set_xlim(-.3,len(labels)-.7);ax.set_ylim(-.8,.8);ax.axis("off")
    palette=("#457b9d","#2a9d8f","#e9c46a","#f4a261","#e76f51","#8d6e63")
    for index,(label,color) in enumerate(zip(labels,palette,strict=True)):
        ax.text(index,0,label,ha="center",va="center",fontsize=9,color="white" if index not in (2,) else "#222",
                bbox=dict(boxstyle="round,pad=.55",fc=color,ec="white",lw=1.2))
        if index<len(labels)-1:ax.annotate("",xy=(index+.62,0),xytext=(index+.38,0),arrowprops=dict(arrowstyle="->",lw=1.6,color="#455a64"))
    path=FIG/"workflow_overview.png";fig.savefig(path,dpi=220,transparent=False);plt.close(fig);return path


def copy_preparatory():
    copied={}
    for name,source in PREPARATORY.items():
        destination=FIG/source.name;shutil.copy2(source,destination);copied[name]=destination
    return copied


def image(path,width=175*mm):
    from PIL import Image as PILImage
    with PILImage.open(path) as source:w,h=source.size
    return Image(str(path),width=width,height=width*h/w)


def build_pdf(manifest,reps,points,shortlist,figures,prep):
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name="TitleCenter",parent=styles["Title"],alignment=TA_CENTER,textColor=colors.HexColor("#17324d"),spaceAfter=12))
    styles.add(ParagraphStyle(name="Small",parent=styles["BodyText"],fontSize=8.5,leading=11))
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=17*mm,leftMargin=17*mm,topMargin=16*mm,bottomMargin=15*mm,title="Combined Dual-Rail Feasibility Campaign")
    story=[Paragraph("Combined Dual-Rail Readout/Gate Feasibility Campaign",styles["TitleCenter"]),Paragraph("68 ordered rail-A repetitions; independent readout/gate overlay followed by per-point fixed-hardware reconciliation",styles["Heading2"]),Spacer(1,5*mm)]
    story += [Paragraph("Executive summary",styles["Heading1"]),Paragraph(
        f"All 68 gate maps completed over 4,955 ordered rail-B EC targets each. Fixed-hardware reconciliation produced {sum(r['pass'] for r in reps):,} passing four-EC designs across 64 repetitions. Four repetitions (A14/A14s and A15/A15s) have an empty final region. The smallest observed rail-A EJ motion occurs in A30 at rail-B EC (150,192) MHz: total rail-A |Delta EJ|={shortlist[0]['a_tune']:.3f} GHz versus rail-B {shortlist[0]['b_tune']:.3f} GHz.",styles["BodyText"]),Spacer(1,3*mm),image(figures[0]),PageBreak()]
    story += [Paragraph("Current workflow and design rationale",styles["Heading1"]),image(figures[3],175*mm),Spacer(1,3*mm),Paragraph(
        "Rail A denotes the lower-physical-frequency rail in the gate design, even though its logical splitting is the higher band (140-160 MHz versus 80-100 MHz for rail B). Readout and gate maps are deliberately independent: gate-valid points are not prefiltered by readout, because idle and gate EJs may differ. Their Boolean intersection is formed only after both maps exist, and every surviving ordered EC point is reconciled independently.",styles["BodyText"]),Spacer(1,3*mm)]
    rationale=[["Choice","Reasoning used in this campaign"],["Separate C1-C2 campaigns","The two logical bands produce materially different readout regions; the rail-B map was recomputed rather than reusing rail A."],["Mirrored readout treatment","Readout label swapping is exact and was used to display both orientations without duplicated calculation; gate EC points remain ordered because gate symmetry is not assumed."],["1.5 scaled EC hull","A single fixed expansion factor explores beyond the direct C1-C2-derived range while keeping the campaign bounded and reproducible."],["5 MHz A / 2 MHz B EC grids","Rail B has the broader feasible region and is the dense gate-map axis; it received the finer lattice."],["Gate sample 140/80 MHz","Fine-readout witnesses concentrate near the lower ends of the allowed bands. One pair controlled runtime; it is a campaign approximation, not a general sensitivity proof."],["Raw >95%, post >99.9%","These strict thresholds define gate validity before overlay."],["Base points plus exact swaps","Ordered simulations test the possible breakdown of A1/A2 label symmetry; added boundary and gap-filling points densify the final rail-A map."],["Per-point reconciliation","No first-success shortcut: every overlay-valid B point is checked for a compatible fixed fabrication realization."],["Full network deferred","The resonator-inclusive network, updated S1, and final CZ recomputation remain the collaborator-owned next stage."]]
    cell=ParagraphStyle(name="RationaleCell",parent=styles["Small"],fontSize=7.2,leading=8.7,spaceAfter=0,spaceBefore=0)
    header=ParagraphStyle(name="RationaleHeader",parent=cell,textColor=colors.white,fontName="Helvetica-Bold")
    wrapped=[[Paragraph(str(value),header if row_index==0 else cell) for value in row] for row_index,row in enumerate(rationale)]
    rt=Table(wrapped,colWidths=[42*mm,120*mm],repeatRows=1);rt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244f73")),("GRID",(0,0),(-1,-1),.3,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef4f8")])]))
    story += [rt,PageBreak(),Paragraph("Preparatory fine C1-C2 readout maps",styles["Heading1"]),Paragraph("Independent fine campaigns were run at the specified logical-frequency bands. The selected-witness distributions cluster near 140 MHz for rail A and 80 MHz for rail B, which motivated the 140/80 MHz gate sample used in this campaign.",styles["BodyText"]),Spacer(1,2*mm),image(prep["fine"],175*mm),PageBreak(),Paragraph("Mirrored readout treatment",styles["Heading1"]),Paragraph("The triangular C1-C2 calculation is mirrored by exact readout label swap for visualization and EC-domain construction. No duplicate readout solve is implied. This symmetry reduction was not carried into the ordered gate EC map.",styles["BodyText"]),Spacer(1,3*mm),image(prep["mirrored"],175*mm),PageBreak(),Paragraph("Direct EC readout maps",styles["Heading1"]),Paragraph("Both maps use the 1.5 scaled two-orientation hull. Rail A uses a 5 MHz lattice; rail B uses a 2 MHz lattice.",styles["BodyText"]),Spacer(1,2*mm),image(prep["ec_a"],175*mm),Spacer(1,2*mm),image(prep["ec_b"],175*mm),PageBreak(),Paragraph("Rail-A repetition selection",styles["Heading1"]),Paragraph("The 34 base points combine the original interior selection with validated boundary points and gap-filling points. Every base point is simulated again with its exact A1/A2 ordering swapped, giving 68 repetitions.",styles["BodyText"]),Spacer(1,2*mm),image(prep["selection"],172*mm),PageBreak(),Paragraph("Campaign quantities and interpretation",styles["Heading1"])]
    data=[["Quantity","Value"],["Rail-A logical target","140 MHz"],["Rail-B logical target","80 MHz"],["Gate thresholds","raw >95%; post-selected >99.9%"],["Rail-A repetitions","68 ordered (34 base + swaps)"],["Gate targets per repetition","4,955"],["Repetitions with final region","64"],["Total reconciled passing points",f"{sum(r['pass'] for r in reps):,}"]]
    table=Table(data,colWidths=[55*mm,105*mm]);table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244f73")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.35,colors.grey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef4f8")])]))
    story += [table,Spacer(1,5*mm),Paragraph("Selection criterion",styles["Heading2"]),Paragraph("For a concrete design, rail-A tuning burden is defined as |Delta EJ,A1|+|Delta EJ,A2|. Rail-B-dominant tuning means the corresponding rail-B sum is larger; the shortlist emphasizes ratios above two where possible. This metric ranks operating-point motion, not flux-line voltage, slew rate, or calibration complexity.",styles["BodyText"]),PageBreak()]
    story += [Paragraph("Recommended concrete designs",styles["Heading1"]),image(figures[1]),Spacer(1,3*mm)]
    rows=[["Design","EC A (MHz)","EC B (MHz)","A tune","B tune","B/A","Raw","Post"]]
    for p in shortlist: rows.append([p["id"],str(p["ec_a"]),str(p["ec_b"]),f"{p['a_tune']:.3f}",f"{p['b_tune']:.3f}",f"{p['dominance']:.2f}",f"{100*p['raw']:.3f}%",f"{100*p['post']:.4f}%"])
    t=Table(rows,colWidths=[15*mm,25*mm,25*mm,18*mm,18*mm,14*mm,19*mm,22*mm],repeatRows=1);t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244f73")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7.5),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [t,Spacer(1,3*mm),Paragraph("A30 is the primary low-rail-A-motion candidate. A10 offers higher raw fidelity at nearly the same A burden. A27 is intentionally B-dominant but requires a large B excursion. A11 represents the robust-region tradeoff: it has 1,622 passing B points, but its typical rail-A motion is larger than the point-optimal designs.",styles["Small"]),PageBreak()]
    a30=next(r for r in reps if r["id"]=="A30");a11=next(r for r in reps if r["id"]=="A11")
    story += [Paragraph("Representative repetition A30: minimum rail-A tuning",styles["Heading1"]),Paragraph(f"Rail-A EC={a30['ec']} MHz; {a30['pass']} of {a30['overlay']} overlay points survive reconciliation ({100*a30['fraction']:.1f}%).",styles["BodyText"]),image(a30["directory"]/"fixed_hardware_reconciliation/plots/fixed_hardware_reconciliation.png",160*mm),Spacer(1,2*mm),image(a30["directory"]/"fixed_hardware_reconciliation/plots/reconciled_gate_fidelity.png",175*mm),PageBreak(),Paragraph("A30 tuning maps",styles["Heading1"]),image(a30["directory"]/"fixed_hardware_reconciliation/plots/reconciled_rail_a_ej_shifts.png",175*mm),Spacer(1,2*mm),image(a30["directory"]/"fixed_hardware_reconciliation/plots/reconciled_rail_b_ej_shifts.png",175*mm),PageBreak()]
    story += [Paragraph("Representative repetition A11: robust feasible region",styles["Heading1"]),Paragraph(f"Rail-A EC={a11['ec']} MHz; {a11['pass']} of {a11['overlay']} overlay points survive ({100*a11['fraction']:.1f}%). A11 illustrates the tradeoff between feasible-area robustness and minimizing rail-A motion.",styles["BodyText"]),image(a11["directory"]/"fixed_hardware_reconciliation/plots/fixed_hardware_reconciliation.png",160*mm),Spacer(1,2*mm),image(a11["directory"]/"fixed_hardware_reconciliation/plots/reconciled_rail_a_ej_shifts.png",175*mm),PageBreak()]
    story += [Paragraph("Failure analysis",styles["Heading1"]),Paragraph("A14/A14s (150,195)/(195,150) MHz and A15/A15s (165,215)/(215,165) MHz have zero reconciliation-passing points. Rail A fails at every overlay point. The universal rail-A blockers are the dispersive and hybridization criteria; logical-frequency and chi-matching failures are additional contributors. Rail-B failures occur at only about 23% of these overlay points and therefore are not the cause of the empty final regions.",styles["BodyText"]),Spacer(1,3*mm),image(figures[2]),Spacer(1,2*mm),Paragraph("Counts are criterion occurrences divided by failed overlay points. Multiple criteria may fail at one point, so bars do not sum to 100%.",styles["Small"]),PageBreak()]
    story += [Paragraph("Conclusions and next validation step",styles["Heading1"]),Paragraph("Prioritize A30 (160,255; B=150,192 MHz) and A10 (165,255; B=278,180 MHz) for full-network validation when minimizing motion of the lower-physical-frequency rail A is the main objective. Retain A11 as a robustness reference. Exclude A14/A14s and A15/A15s from downstream validation unless the readout reconciliation constraints or physical hardware parameterization are changed.",styles["BodyText"]),Spacer(1,4*mm),Paragraph("The final full-network step should include both resonators, verify dispersive matter-resonator coupling, update S1, and recompute CZ fidelity using the reconciled fixed hardware. That step is outside this campaign and remains assigned to the collaborator's validator.",styles["BodyText"])]
    def footer(canvas,doc):canvas.saveState();canvas.setFont("Helvetica",8);canvas.setFillColor(colors.grey);canvas.drawString(17*mm,9*mm,"Combined dual-rail feasibility campaign");canvas.drawRightString(A4[0]-17*mm,9*mm,f"Page {doc.page}");canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def build_note(manifest,reps,shortlist,figures,prep):
    zero=[r for r in reps if not r["pass"]]
    embed=lambda path:f"![[{FIG.name}/{path.name}]]"
    lines=["# Combined dual-rail feasibility campaign report","","> [!summary] Outcome",f"> 68 ordered rail-A repetitions completed; {sum(r['pass'] for r in reps):,} reconciled four-EC designs survive across 64 repetitions. Four repetitions have empty final regions.","","## Current workflow",embed(figures[3]),"","1. Run separate fine C1-C2 readout campaigns for the 140-160 MHz and 80-100 MHz logical bands.","2. Mirror readout by exact label swap, then construct 1.5-scaled EC exploration hulls.","3. Compute independent readout and gate maps; do not prefilter the gate domain by readout.","4. Overlay the maps only after both are complete.","5. Reconcile fixed fabrication hardware at every overlay-valid point; idle and gate EJs may differ.","6. Hand surviving points to the collaborator's resonator-inclusive full-network validator.","","### Parameter rationale","","- Rail A is the lower-physical-frequency gate rail but has the higher logical band.","- Rail A uses 5 MHz EC spacing; rail B uses 2 MHz because its feasible region is broader and it is the dense gate-map axis.","- The gate map uses 140/80 MHz because readout witnesses concentrate near the lower band edges; this controls runtime but does not replace a future sensitivity study.","- Gate thresholds are raw >95% and post-selected >99.9%.","- Base points and exact swaps retain ordered-label information; boundary and interior additions densify the final map.","","## Preparatory maps",embed(prep["fine"]),"",embed(prep["mirrored"]),"",embed(prep["ec_a"]),"",embed(prep["ec_b"]),"",embed(prep["selection"]),"","## Global results",embed(figures[0]),"","## Recommended concrete points","","| Rep | EC A (MHz) | EC B (MHz) | Sum abs dEJ A (GHz) | Sum abs dEJ B (GHz) | B/A | Raw | Post-selected |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in shortlist:lines.append(f"| {p['id']} | {p['ec_a']} | {p['ec_b']} | {p['a_tune']:.3f} | {p['b_tune']:.3f} | {p['dominance']:.2f} | {100*p['raw']:.3f}% | {100*p['post']:.4f}% |")
    lines += ["","The primary recommendation is A30 with B=(150,192) MHz. A10 is the higher-raw-fidelity alternative. A27 is strongly B-dominant but requires a large B excursion. A11 is retained as a large-feasible-region reference.","",embed(figures[1]),"","## Failure regions",embed(figures[2]),"","A14/A14s and A15/A15s have zero passing points. Rail A fails at every overlay point, universally on both dispersiveness and hybridization; rail-B failures are secondary.","","## Data and result locations","",f"- Campaign: `{CAMPAIGN}`",f"- Selection manifest: `{MANIFEST}`","- Per repetition gate archive: `<repetition>/gate_output/compact_results.sqlite`","- Per repetition reconciliation data: `<repetition>/fixed_hardware_reconciliation/fixed_hardware_results.sqlite`","- Per repetition figures: `<repetition>/fixed_hardware_reconciliation/plots/`",f"- PDF: `{PDF}`","","## Scope","","Readout and gate maps are independent. Every overlay point is reconciled independently. The resonator-inclusive full-network validation is not included and remains the next collaborator-owned stage."]
    NOTE.write_text("\n".join(lines)+"\n")


def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest,reps,points=load()
    def pick(sid,ecb):return next(p for p in points if p["id"]==sid and p["ec_b"]==ecb)
    shortlist=[pick("A30",(150,192)),pick("A10",(278,180)),pick("A27",(244,156)),pick("A11",(150,192))]
    figures=(summary_maps(reps),recommendation_plot(shortlist),failure_plot(reps),workflow_plot());prep=copy_preparatory()
    build_pdf(manifest,reps,points,shortlist,figures,prep);build_note(manifest,reps,shortlist,figures,prep)
    print(json.dumps({"pdf":str(PDF),"note":str(NOTE),"pages_source_repetitions":68,"passing_points":len(points),"shortlist":[{"id":p["id"],"ec_a":p["ec_a"],"ec_b":p["ec_b"]} for p in shortlist]},indent=2))


if __name__=="__main__":main()
