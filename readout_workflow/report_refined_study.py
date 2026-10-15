"""Uniform-pixel maps and an auditable report for the refined study."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from run_refined_study import OUT, UNIFORM
from refined_pair_search import SETTINGS
from reference_devices import original_preset


def load():
    db=sqlite3.connect(OUT/"study.sqlite")
    rows=[json.loads(r[0]) for r in db.execute("SELECT result FROM points")]
    meta={key:json.loads(value) for key,value in db.execute("SELECT key,value FROM meta")}
    trials=db.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    return rows,meta,trials


def figures(rows,meta):
    grid=UNIFORM;edges=np.r_[grid-.5,grid[-1]+.5]
    shape=(len(grid),len(grid));status=np.full(shape,-1)
    maps={key:np.full(shape,np.nan) for key in ("H","frequency","Cg","gap","EC1","EC2","chi","lambda","EJ_EC","overlap")}
    idx={float(c):i for i,c in enumerate(grid)}
    requested={tuple(p) for p in meta.get("uniform_requested",[])}
    for c1,c2 in requested:
        if c1 in idx and c2 in idx: status[idx[c2],idx[c1]]=2
    uniform=[]
    for row in rows:
        pair=(row["c1"],row["c2"])
        if pair not in requested or pair[0] not in idx or pair[1] not in idx: continue
        i,j=idx[pair[0]],idx[pair[1]]
        if row["status"]=="validated_pass":
            status[j,i]=1;uniform.append(row);p=row["point"]
            values={"H":p["hybridization"],"frequency":p["operating_frequency_ghz"],
                "Cg":p["device"]["C1r_fF"],"gap":1000*p["splitting_ghz"],
                "EC1":p["loaded_ec1_mhz"],"EC2":p["loaded_ec2_mhz"],
                "chi":100*p["ratio"],"lambda":p["lambda_max"],
                "EJ_EC":min(p["ej1_over_ec1"],p["ej2_over_ec2"]),
                "overlap":p["min_assignment_overlap"]}
            for key,value in values.items(): maps[key][j,i]=value
        elif row["status"]=="no_witness_found": status[j,i]=0
        else: status[j,i]=2
    def draw(ax,key,title,limits=None):
        ax.pcolormesh(edges,edges,np.ma.masked_where(status!=0,np.zeros(shape)),
            cmap=ListedColormap(["#e6e6e6"]),shading="flat",vmin=0,vmax=1)
        ax.pcolormesh(edges,edges,np.ma.masked_where(status!=2,np.zeros(shape)),
            cmap=ListedColormap(["#f7cf99"]),shading="flat",vmin=0,vmax=1)
        kwargs={} if limits is None else dict(vmin=limits[0],vmax=limits[1])
        mesh=ax.pcolormesh(edges,edges,np.ma.masked_invalid(maps[key]),cmap="viridis",shading="flat",**kwargs)
        ax.plot([edges[0],edges[-1]],[edges[0],edges[-1]],ls=":",color="black",lw=.8)
        ax.set(xlabel="C1 (fF)",ylabel="C2 (fF)",title=title,
               xlim=(edges[0],edges[-1]),ylim=(edges[0],edges[-1]))
        ax.set_aspect("equal")
        return mesh
    note="Only C1 > C2; uniform 1 fF pixels. Gray: searched, no witness found; orange: unresolved; white: excluded/not sampled."
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout="constrained")
    specs=[("H","Hybridization of selected solution",(.95,1.)),
        ("frequency","Operation frequency (GHz)",(4,8)),
        ("Cg","Equal readout capacitance Cg (fF)",None),
        ("gap","Logical gap (MHz)",(90,110))]
    for ax,(key,title,limits) in zip(axes.flat,specs): fig.colorbar(draw(ax,key,title,limits),ax=ax,shrink=.85)
    fig.suptitle("Refined feasible-design map | 6 GHz bare resonator\n"+note,fontsize=11)
    fig.savefig(OUT/"refined_feasible_maps.png",dpi=180);fig.savefig(OUT/"refined_feasible_maps.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout="constrained")
    for ax,key,title in zip(axes[:2],("EC1","EC2"),(r"Loaded $E_{C1}/h$ (MHz)",r"Loaded $E_{C2}/h$ (MHz)")):
        fig.colorbar(draw(ax,key,title,(150,400)),ax=ax,shrink=.85)
    cdf=axes[2];stats={}
    for index in (1,2):
        vals=np.sort([r["point"][f"loaded_ec{index}_mhz"] for r in uniform])
        label="Larger-C rail (transmon 1)" if index==1 else "Smaller-C rail (transmon 2)"
        if len(vals):
            cdf.step(np.r_[150,vals,400],np.r_[0,np.arange(1,len(vals)+1)/len(vals),1],where="post",label=label)
            stats[f"EC{index}"]={"min":float(min(vals)),"median":float(np.median(vals)),"max":float(max(vals))}
    pooled=np.sort([r["point"][key] for r in uniform for key in ("loaded_ec1_mhz","loaded_ec2_mhz")])
    if len(pooled): cdf.step(np.r_[150,pooled,400],np.r_[0,np.arange(1,len(pooled)+1)/len(pooled),1],
        where="post",color="black",ls="--",lw=1,label="Either label after symmetric relabeling")
    cdf.set(xlim=(150,400),ylim=(0,1.02),xlabel=r"Loaded $E_C/h$ (MHz)",ylabel="Empirical cumulative fraction",
            title=f"CDF of {len(uniform)} passing grid centers")
    cdf.grid(alpha=.25);cdf.legend(fontsize=7)
    fig.suptitle("Actual loaded charging energies | C1 > C2\nCDF is weighted equally per passing grid center, not a fabrication probability.",fontsize=11)
    fig.savefig(OUT/"refined_charging_energies.png",dpi=180);fig.savefig(OUT/"refined_charging_energies.pdf");plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout="constrained")
    specs=[("chi","Worst chi-mismatch (%)",(0,1)),("lambda","Maximum multilevel dispersive ratio",(0,.1)),
        ("EJ_EC","Minimum EJ/EC of the two rails",None),("overlap","Minimum reference-state overlap",(.95,1))]
    for ax,(key,title,limits) in zip(axes.flat,specs): fig.colorbar(draw(ax,key,title,limits),ax=ax,shrink=.85)
    fig.suptitle("Constraint diagnostics of selected solutions\n"+note,fontsize=11)
    fig.savefig(OUT/"refined_constraint_maps.png",dpi=180);fig.savefig(OUT/"refined_constraint_maps.pdf");plt.close(fig)
    (OUT/"final_grid_definition.json").write_text(json.dumps({"spacing_fF":1.,"centers_fF":grid.tolist(),
        "requested_centers":len(requested),"validated_pass_centers":int(sum((status==1).ravel())),
        "no_witness_centers":int(sum((status==0).ravel())),"unresolved_centers":int(sum((status==2).ravel())),
        "only_C1_gt_C2":True,"CDF":stats},indent=2))
    return uniform,stats,status


def contiguous(values):
    groups=[]
    for value in sorted(values):
        if not groups or value!=groups[-1][-1]+1: groups.append([value])
        else: groups[-1].append(value)
    return [[g[0],g[-1]] for g in groups]


def summary(rows,uniform,meta,trials,stats,status):
    ranges=[]
    for c1 in UNIFORM:
        vals=[r["c2"] for r in uniform if r["c1"]==c1]
        if vals: ranges.append({"C1_fF":float(c1),"C2_passing_integer_intervals_fF":contiguous(vals),"count":len(vals)})
    (OUT/"uniform_ranges.json").write_text(json.dumps(ranges,indent=2))
    good=[r for r in rows if r["status"]=="validated_pass"]
    keys=("hybridization","operating_frequency_ghz","splitting_ghz","ratio","lambda_max","loaded_ec1_mhz","loaded_ec2_mhz")
    metrics={key:{"min":min(r["point"][key] for r in uniform),"median":float(np.median([r["point"][key] for r in uniform])),
                  "max":max(r["point"][key] for r in uniform)} for key in keys} if uniform else {}
    result={"updated_utc":datetime.now(timezone.utc).isoformat(),"settings":meta.get("settings",SETTINGS),
        "trials":trials,"unique_pairs":len(rows),"validated_total":len(good),"validated_uniform":len(uniform),
        "no_witness_uniform":int(sum((status==0).ravel())),"unresolved_uniform":int(sum((status==2).ravel())),
        "metrics":metrics,"CDF_statistics":stats,"ranges":ranges}
    if (OUT/'audit.json').exists(): result['audit']=json.loads((OUT/'audit.json').read_text())
    (OUT/"summary.json").write_text(json.dumps(result,indent=2))
    lines=["# Refined capacitance-pair study", "",
        "Only C1>C2 was computed. Final maps use uniform 1 fF square pixels. Each colored pixel is a separately computed, larger-basis-validated example; no interior is filled by interpolation. Noninteger historical/original anchors are recorded separately and do not create irregular map pixels.","",
        "## Applied constraints","",
        "- Logical gap: 90-110 MHz, free during joint optimization (not targeted to exactly 100 MHz).",
        "- Worst chi normalization: max_i |(chi+ - chi-)/chi_i| <=1%.",
        "- Maximum upward isolated-pair dispersive ratio from ground and logical states <=0.1.",
        "- Hybridization >=0.95; minimum branch-reference overlap >=0.95.",
        "- Both loaded EC/h >=150 MHz and strictly <400 MHz; both EJ/h <25 GHz; both EJ/EC >50.",
        "- Bare resonator 6 GHz, 50 ohm equivalent-mode impedance; dressed carrier 4-8 GHz.",
        "- Equal readout capacitors, variable common Cg; C12 and both EJs adjustable.","",
        "Numerical design bounds: Cg=0.1-25 fF; C12=0.005-15 fF; C1,C2 grid centers=36-128 fF. The coarse above-resonator branch reached C2=44 fF, so the final grid was extended downward to resolve that boundary. These finite bounds define this study's coverage.","",
        "Hybridization is a charge-transition balance score, not a state overlap: S(i,s)=|<G|n_i|s>|^2, p(i,s)=S(i,s)/sum_s S(i,s), H=1-2 mean(|p-1/2|). The separate 0.95 reference-overlap screen checks state assignment reliability.","",
        "The 90% participation estimate initializes the capacitance range only. Final charging energies and all couplings use the complete physical capacitance matrix. Sampling extends to C1=128 fF to avoid an artificial cutoff at the approximate 116 fF upper bound and to include the 118 fF design-A anchor. Equal Cg does not enforce equal numerical g.","",
        "## Coverage and validation","",
        f"Unique shunt pairs evaluated: {len(rows)}; nonlinear search/validation records: {trials}; validated passing uniform-grid centers: {len(uniform)}. Uniform-grid no-witness centers: {result['no_witness_uniform']}; unresolved centers: {result['unresolved_uniform']}.","",
        "Screening basis: 15 charge cutoff / 6 transmon levels / 5 resonator levels / 20 eigenstates. Every accepted plotted solution is recomputed at unchanged physical component values in basis 20/9/8/40. Search repair uses the larger basis where necessary. Source hashes, stages, attempts, and checkpoints are retained in study.sqlite and settings.json.","",
        "A searched point without a witness is not a proof of physical infeasibility. It reflects a bounded multi-start nonlinear search. Orange denotes unresolved numerical/budget/validation cases and must not be treated as infeasible. White denotes excluded or uncomputed cells. Carrier, gap, and coupling colors show one selected feasible design, not a global optimum at that pair.","",
        "## Feasible integer-grid intervals","",
        "Each interval below includes passing sampled integer C2 centers only. It does not certify noninteger points between them.","",
        "| C1 (fF) | Passing C2 centers (fF) |","|---:|---|"]
    for r in ranges:
        if r["C1_fF"]%5==0 or r in (ranges[0],ranges[-1]):
            display=", ".join(str(int(a)) if a==b else f"{int(a)}-{int(b)}" for a,b in r["C2_passing_integer_intervals_fF"])
            lines.append(f"| {r['C1_fF']:g} | {display} |")
    lines += ["","![Feasible designs](refined_feasible_maps.png)","",
        "![Charging energies](refined_charging_energies.png)","",
        "CDF interpretation: because only C1>C2 is sampled, the separate curves describe the larger- and smaller-capacitance rails. If both mirror orientations are given equal weight, the two transmon-label distributions are identical; the pooled dashed curve shows that common distribution. Uniform integer centers have equal weight, which is not a measured fabrication distribution.","",
        "![Constraint diagnostics](refined_constraint_maps.png)","",
        "## Scope","",
        "This is a design-existence study, not a fabrication-tolerance study. C12 and Cg can change between pixels; only the EJs are in-situ knobs within a fixed circuit. No minimum absolute chi, readout speed, finite-drive transition, Purcell-filter or frequency-noise requirement is imposed. The 50 ohm impedance is the equivalent LC-mode impedance, not automatically the CPW characteristic impedance."]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return result


def pdf_report(result,rows):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,Image,PageBreak
    fonts=Path(matplotlib.get_data_path())/"fonts"/"ttf"
    pdfmetrics.registerFont(TTFont("RefDV",str(fonts/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("RefDV-Bold",str(fonts/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("RefDV",normal="RefDV",bold="RefDV-Bold")
    blue=colors.HexColor("#173B59")
    styles={"body":ParagraphStyle("b",fontName="RefDV",fontSize=9,leading=12,spaceAfter=7),
        "title":ParagraphStyle("t",fontName="RefDV-Bold",fontSize=19,leading=23,textColor=blue,spaceAfter=12),
        "h":ParagraphStyle("h",fontName="RefDV-Bold",fontSize=12,leading=16,textColor=blue,spaceBefore=8,spaceAfter=6),
        "small":ParagraphStyle("s",fontName="RefDV",fontSize=7.6,leading=10,spaceAfter=6)}
    width=A4[0]-80;story=[]
    def add(text,style="body"): story.append(Paragraph(text,styles[style]))
    def picture(filename,w=width):
        im=Image(str(OUT/filename));im.drawHeight=w*im.imageHeight/im.imageWidth;im.drawWidth=w;story.append(im)
    def table(data,widths):
        items=[[Paragraph(str(v),styles["small"]) for v in r] for r in data]
        t=Table(items,colWidths=widths,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dfeaf3")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f2f5f7")]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5)]));story.extend([t,Spacer(1,6)])
    add("Refined dual-rail design study","title")
    add("Run started 14 October 2026 | Updated constraints | Strict C1 &gt; C2 symmetry reduction","small")
    add(f"<b>{result['validated_uniform']} validated passing grid centers</b> were found on a uniform 1 fF lattice. All accepted map pixels were independently recomputed in the larger numerical basis. This is a map of discovered feasible designs, not a proof of a global exclusion boundary.")
    table([["Constraint","Applied value"],["Logical gap","90-110 MHz; variable throughout optimization"],
        ["Readout mismatch","max_i |(χ+ - χ-)/χ_i| ≤1%"],["Dispersive condition","Maximum upward-pair |g/Δ| ≤0.1"],
        ["States","Hybridization ≥0.95; reference overlap ≥0.95"],
        ["Transmon energies","Both loaded EC/h ≥150 and &lt;400 MHz; EJ/h &lt;25 GHz; EJ/EC &gt;50"],
        ["Resonator / operation","6 GHz bare resonator, 50 Ω equivalent-mode impedance; carrier 4-8 GHz"],
        ["Design variables","Both EJs, C12 and equal C1r=C2r=Cg; C1&gt;C2 only"]],[140,width-140])
    add("Workflow and reliability","h")
    add("A 2 fF coarse sweep, continuation from nearby passing designs, multi-start retries and a final uniform 1 fF sweep were used. Local transmon frequencies parameterize the EJs through numerical inversion at the actual loaded ECs. All constraints participate jointly; neither exact χ matching nor maximal hybridization is required. The logical gap is not fixed to 100 MHz.")
    add(f"{result['unique_pairs']} distinct pairs and {result['trials']} search/validation records are checkpointed. Final-grid no-witness centers: {result['no_witness_uniform']}; unresolved centers: {result['unresolved_uniform']}. Gray denotes a bounded search without a witness; orange denotes unresolved; white is excluded/uncomputed. No interpolation supplies passing pixels.")
    add("90% participation is only an initial capacitance-to-EC estimate. The actual inverse capacitance matrix determines every reported EC and coupling. High-side capacitance coverage extends to 128 fF to avoid artificially excluding valid low-EC circuits or the original 118 fF anchor.")
    add("Finite design bounds: Cg=0.1-25 fF; C12=0.005-15 fF; integer C1,C2 centers=36-128 fF. The high-EC branch touched the coarse 44 fF edge, prompting the lower extension. Hybridization measures charge-transition balance, not Bell-state overlap.","small")
    add("Feasible sampled intervals","h")
    ranges=result["ranges"]
    selected=[r for r in ranges if r["C1_fF"]%5==0 or r in (ranges[0],ranges[-1])]
    add("The sampled intervals and full one-fF results are listed with the charging-energy figure. The complete per-row ranges are retained in uniform_ranges.json.","small")
    story.append(PageBreak())
    add("Selected feasible solutions","title")
    picture("refined_feasible_maps.png",width)
    add("Each pixel represents one selected design. Operation frequency, logical gap and coupling may differ between pixels. These colors do not claim a global optimum of hybridization or readout performance.","small")
    metrics=result['metrics']
    if metrics:
        add(f"Selected operation frequencies span {metrics['operating_frequency_ghz']['min']:.3f}-{metrics['operating_frequency_ghz']['max']:.3f} GHz. Logical gaps span {1000*metrics['splitting_ghz']['min']:.3f}-{1000*metrics['splitting_ghz']['max']:.3f} MHz. Hybridization spans {metrics['hybridization']['min']:.5f}-{metrics['hybridization']['max']:.5f}.")
    story.append(PageBreak())
    add("Loaded energies and checks","title")
    picture("refined_charging_energies.png",width)
    add("The ordered-rail CDFs compare the larger-C rail (1) and smaller-C rail (2). With equal weighting of both mirror orientations, the two transmon-label CDFs coincide; the dashed pooled curve represents that common distribution. These are equal-weight sampled-grid distributions, not fabrication probabilities.")
    add("Feasible sampled intervals","h")
    table([["C1 (fF)","Passing sampled integer C2 intervals (fF)"]]+[[f"{r['C1_fF']:g}",
        ", ".join(str(int(a)) if a==b else f"{int(a)}-{int(b)}" for a,b in r["C2_passing_integer_intervals_fF"])] for r in selected],
        [100,width-100])
    add("Intervals refer to computed integer centers, not a guarantee for every continuous point between them.","small")
    if metrics:
        add(f"Loaded EC1/h spans {metrics['loaded_ec1_mhz']['min']:.2f}-{metrics['loaded_ec1_mhz']['max']:.2f} MHz; EC2/h spans {metrics['loaded_ec2_mhz']['min']:.2f}-{metrics['loaded_ec2_mhz']['max']:.2f} MHz. Extending the search ceiling to 400 MHz does not guarantee feasible solutions reach that ceiling.")
    story.append(PageBreak())
    add("Constraint checks and scope","title")
    picture("refined_constraint_maps.png",width*.92)
    add("Original and optimized anchors","h")
    anchor_pairs={(118.,78.),(115.,80.)}
    anchor_pairs.update((original_preset(name)['C1_shunt_fF'],original_preset(name)['C2_shunt_fF']) for name in ('tdt1','tdt2'))
    anchors=[r for r in rows if (r['c1'],r['c2']) in anchor_pairs]
    table([["C1 / C2 (fF)","Search status","Failed checks, if any"]]+[
        [f"{r['c1']:.3f} / {r['c2']:.3f}",r["status"],", ".join(r.get("failures",[])) or "None"] for r in anchors],
        [125,165,width-290])
    add("Anchor rows fix C1/C2 only: C12, equal Cg and EJs are reoptimized. They do not test the original devices with every component held unchanged.","small")
    audit=result.get('audit',{})
    if audit:
        add(f"Additional convergence audit: {audit['extra_basis_samples']} extreme-case witnesses checked at basis 25/12/10/60; {len(audit['extra_basis_failures'])} crossed an acceptance threshold. Full changes are recorded in audit.json.","small")
    add("C12 and Cg are redesigned between points; this is design feasibility, not fabrication robustness. No minimum |χ|, measurement speed, finite-drive, Purcell-filter or frequency-noise condition is imposed. CPW parameters must map consistently to the equivalent-mode impedance.","small")
    add("Reproduce and audit: settings.json, study.sqlite, uniform_points.json, uniform_ranges.json, final_grid_definition.json and report.md in readout_workflow/data/refined_shunt_map_2026_10_14/.","small")
    def footer(canvas,doc):
        canvas.setFont("RefDV",7);canvas.setFillColor(colors.HexColor("#526372"))
        canvas.drawString(40,24,"Refined numerical study | 90-110 MHz gap | EC/h <400 MHz | C1 > C2")
        canvas.drawRightString(A4[0]-40,24,str(doc.page))
    SimpleDocTemplate(str(OUT/"refined_study_report.pdf"),pagesize=A4,leftMargin=40,rightMargin=40,
        topMargin=36,bottomMargin=43,title="Refined dual-rail readout design study").build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    rows,meta,trials=load()
    uniform,stats,status=figures(rows,meta)
    result=summary(rows,uniform,meta,trials,stats,status)
    pdf_report(result,rows)
    print(json.dumps({key:result[key] for key in ("unique_pairs","validated_uniform","no_witness_uniform","unresolved_uniform")},indent=2))


if __name__=="__main__": main()
