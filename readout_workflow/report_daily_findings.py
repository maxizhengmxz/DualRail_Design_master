"""Concise, source-data-driven daily PDF report with rendered page checks."""
import json
from pathlib import Path
import fitz
import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
DATA=BASE/"data"
OUT=ROOT/"output"/"pdf"/"dual_rail_readout_findings_2026-10-14.pdf"
PREVIEW=ROOT/"tmp"/"pdfs"/"daily_findings"
BLUE=colors.HexColor("#163A59")
GRAY=colors.HexColor("#53616F")
LIGHT=colors.HexColor("#EEF3F7")


def load(path): return json.loads((DATA/path).read_text())


def main():
    OUT.parent.mkdir(parents=True,exist_ok=True);PREVIEW.mkdir(parents=True,exist_ok=True)
    fonts=Path(matplotlib.get_data_path())/"fonts"/"ttf"
    pdfmetrics.registerFont(TTFont("DV",str(fonts/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-Bold",str(fonts/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DV",normal="DV",bold="DV-Bold",italic="DV",boldItalic="DV-Bold")
    styles={
        "title":ParagraphStyle("title",fontName="DV-Bold",fontSize=20,leading=24,textColor=BLUE,spaceAfter=9),
        "subtitle":ParagraphStyle("subtitle",fontName="DV",fontSize=9,leading=12,textColor=GRAY,spaceAfter=12),
        "h":ParagraphStyle("h",fontName="DV-Bold",fontSize=12,leading=16,textColor=BLUE,spaceBefore=10,spaceAfter=6),
        "body":ParagraphStyle("body",fontName="DV",fontSize=9,leading=12.4,spaceAfter=7),
        "small":ParagraphStyle("small",fontName="DV",fontSize=7.8,leading=10.3,textColor=GRAY,spaceAfter=5),
        "cell":ParagraphStyle("cell",fontName="DV",fontSize=8,leading=10.7),
        "header":ParagraphStyle("header",fontName="DV-Bold",fontSize=8,leading=10.7,textColor=colors.white),
    }
    story=[]
    def p(text,style="body"): return Paragraph(text,styles[style])
    def add(text,style="body"): story.append(p(text,style))
    def table(rows,widths):
        cells=[[p(str(value),"header" if i==0 else "cell") for value in row] for i,row in enumerate(rows)]
        t=Table(cells,colWidths=widths,hAlign="LEFT",repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLUE),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),
            ("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LINEBELOW",(0,0),(-1,0),.6,BLUE)]))
        story.extend([t,Spacer(1,7)])
    def image(path,width):
        doc=fitz.open(path) if str(path).endswith(".pdf") else None
        if doc:
            pix=doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
            target=PREVIEW/(Path(path).stem+"_source.png");pix.save(target);doc.close();path=target
        im=Image(str(path));im.drawHeight=width*im.imageHeight/im.imageWidth;im.drawWidth=width
        story.append(im)
    width=A4[0]-84

    originals=load(Path("original_design_audit/operating_points.json"))
    old=[next(p for p in r["points"] if p["resonator_case"]=="6_GHz") for r in originals]
    designs=load(Path("equal_g/validated.json"))
    cdf=load(Path("equal_g/robustness/samples_2pct_limit_0.01/cdf_statistics.json"))
    ranges=load(Path("shunt_pair_map/ranges.json"))
    witnesses=load(Path("shunt_pair_map/validated.json"))
    w=next(r["large_basis"] for r in witnesses if r["c1"]==115 and r["c2"]==80)

    # PAGE 1: the contract and original-versus-new distinction.
    add("Transmon dual-rail readout", "title")
    add("Daily findings | 14 October 2026 | Numerical single-junction workflow", "subtitle")
    add("<b>Main result.</b> We found circuits with χ-mismatch below 1%, a logical gap near 100 MHz and the requested low-photon dispersive cutoff. The original presets match χ well, but fail the added dispersive and E<sub>J</sub>/E<sub>C</sub> requirements at their approximately 4.4 GHz operating points.")
    add("Constraints and definitions", "h")
    table([
        ["Quantity","Applied in the reported calculations","Agreed for future work"],
        ["Logical gap","95-105 MHz; design solver targets 100 MHz","<b>90-110 MHz</b> (±10%)"],
        ["χ-mismatch","R = max<sub>i=±</sub> |(χ<sub>+</sub>-χ<sub>-</sub>)/χ<sub>i</sub>| ≤ 0.01","Same"],
        ["Dispersive screen","max |g<sub>ij</sub>/Δ<sub>ij</sub>| ≤ 0.1, upward isolated-pair transitions from ground and both logical states","Same; low-photon screen"],
        ["Transmon energies","Each E<sub>J</sub>/h &lt; 25 GHz; loaded E<sub>C</sub>/h = 150-350 MHz; each E<sub>J</sub>/E<sub>C</sub> &gt; 50","Same"],
        ["State diagnostics","Hybridization H ≥ 0.95; reference-state overlap ≥ 0.95","Same"],
        ["Resonator / tuning","6 GHz <b>bare</b> resonator; 50 Ω equivalent-mode impedance; both EJs tunable","Same; pair map allows carrier 4-8 GHz"],
    ],[105,267,width-372])
    add("The ±10% gap window has <b>not</b> been used to recompute these results. E<sub>C</sub> = e²(C<sup>-1</sup>)<sub>ii</sub>/2 includes all capacitive loading. χ<sub>±</sub> are full conditional resonator-frequency shifts relative to the ground state. No minimum |χ| or readout-speed requirement was imposed.","small")
    add("Original circuits versus new optimized A/B", "h")
    table([
        ["Original preset at ~4.4 GHz","tdt1","tdt2","Requirement"],
        ["C1 / C2 (fF)","64.72 / 54.63","75.89 / 63.21","Original physical shunts"],
        ["R / logical gap",*[f"{100*max(x['ratio_m'],x['abs_delta_chi_over_chi_p']):.3f}% / {1000*x['splitting_ghz']:.1f} MHz" for x in old],"≤1% / 95-105 MHz"],
        ["EJ1/EC1 / EJ2/EC2",*[f"{x['ej1_over_ec1']:.1f} / {x['ej2_over_ec2']:.1f}" for x in old],"Both &gt;50"],
        ["Dispersive maximum",*[f"{x['lambda_max']:.3f}" for x in old],"≤0.1"],
    ],[155,114,114,width-383])
    add("<b>Naming:</b> original tdt1/tdt2 are not the later designs A/B. Larger new shunts lower E<sub>C</sub>; smaller readout caps reduce coupling. Reducing readout capacitance alone removes loading and raises E<sub>C</sub>.","small")
    table([["Later equal-g design","C1 / C2 (fF)","C1r / C2r (fF)","Loaded EC1 / EC2 (MHz)"]]+[
        [label,f"{r['device']['C1_shunt_fF']:g} / {r['device']['C2_shunt_fF']:g}",
         f"{r['device']['C1r_fF']:.3f} / {r['device']['C2r_fF']:.3f}",
         f"{r['loaded_ec1_mhz']:.2f} / {r['loaded_ec2_mhz']:.2f}"] for label,r in zip(("A","B"),designs)],
        [125,105,120,width-350])
    add("<b>Equal g is not sufficient by itself.</b> In the new A/B circuits, the unconstrained hybridization maxima have H≈0.987/0.986 but mismatch≈3.18%/2.86%. Allowing EJ detuning meets the 1% ceiling at nominal H≈0.9574/0.9595. Equal-g designs were feasible, but did not automatically outperform equal-capacitance designs; anharmonicities and higher-transition matrix elements also affect matching.")

    # PAGE 2: recalibration, CDF, tolerance and parameter sensitivity.
    story.append(PageBreak())
    add("Robustness with EJ recalibration", "title")
    add("Fixed fabricated capacitances | Both EJs retuned | Carrier held at 4.4 GHz", "subtitle")
    add("<b>Recalibration permits non-maximal hybridization.</b> The search chooses the highest H within the χ-mismatch ceiling, rather than requiring the unconstrained hybridization maximum. Equal g is a nominal design condition; it is not reimposed after capacitor errors.")
    add("Each of C1, C2, C12, C1r and C2r varies independently within ±2% of its own nominal value. The same 300 perturbation vectors are applied to A and B: 258 Latin-hypercube interior samples, 32 extreme corners and 10 isolated ±2% endpoints. Cr and Lr stay fixed.")
    table([["Passing samples","A: R ≤0.9%","A: R ≤1%","B: R ≤0.9%","B: R ≤1%"],
        ["Interior (258)","216 / 258",f"{cdf['A']['interior_new_pass']} / 258","222 / 258",f"{cdf['B']['interior_new_pass']} / 258"],
        ["Extreme corners (32)","22 / 32","24 / 32","22 / 32","24 / 32"],
        ["Isolated endpoints (10)","8 / 10","10 / 10","10 / 10","10 / 10"]],
        [120]+[(width-120)/4]*4)
    image(DATA/"equal_g/robustness/samples_2pct_limit_0.01/matching_cdf.png",width)
    add("<b>Reading the CDF:</b> each sample contributes its best attainable H under the mismatch ceiling. At H=0.95, the 1% curves give failure fractions of 8.1% (A) and 5.8% (B); passing fractions are <b>91.9% and 94.2%</b>. All remaining failures are H&lt;0.95, not dispersive or energy failures. Relaxing the ceiling never reduced H in the paired samples.","small")
    add("What hybridization measures", "h")
    add("For rail i and logical branch s, S<sub>is</sub> = |&lt;G|n<sub>i</sub>|s&gt;|² and p<sub>is</sub> = S<sub>is</sub>/(S<sub>i-</sub>+S<sub>i+</sub>). The score is H = 1 - 2 mean |p<sub>is</sub>-1/2|. It measures balanced charge-transition strength, <b>not state fidelity</b>. In the ideal two-level model H=0.95 corresponds to 52.5%/47.5% rail populations. Resonator admixture needs the separate overlap and dispersive checks.")
    add("Sensitivity: percentage versus absolute capacitance", "h")
    add("The least robust isolated percentage variation was C1, followed by C2r, C1r, C2 and C12. However, A's ±2% means ±2.36 fF for C1 but only ±1.56 fF for C2. Their recalibrated sensitivities per fF are nearly equal: |dH/dC1|≈0.002041 and |dH/dC2|≈0.002013 (1.4% apart). B differs by only 2.6% per fF. This is not an intrinsic weakness of rail 1.")
    add("Harmful combinations tend toward C1 low, C2 high, C1r high and C2r low. Inputs were independently sampled; these are output associations, not assumed fabrication correlations. The combined counts are not yield estimates.","small")

    # PAGE 3: design freedom, discovered band and interpretation limits.
    story.append(PageBreak())
    add("Acceptable capacitance-pair examples", "title")
    add("Equal readout capacitors C1r=C2r | EJ1, EJ2, C12 and common readout cap adjustable", "subtitle")
    add("This map answers a <b>design-existence</b> question, not fabrication robustness. At each C1/C2 pair, the other circuit parameters may be redesigned. A 90% participation estimate converts EC/h=150-350 MHz into C≈49.8-116.2 fF; final checks use the full capacitance matrix. Actual participation in passing solutions spans 87.9-96.5%.")
    image(DATA/"shunt_pair_map/feasible_pairs.png",width)
    add("134 ordered pairs searched; <b>40 passing solutions</b> survived larger-basis checks. The plotted selected carriers are 4.08-5.35 GHz; 4-8 GHz was searched, but these examples do not establish exclusion at other frequencies. Black dots are refined passing samples between the original grid points.","small")
    table([["Larger C (fF)","Smaller C: passing sampled span (fF)"]]+[
        [f"{r['larger_shunt_fF']:.1f}",f"{r['smallest_found_smaller_shunt_fF']:.1f}-{r['largest_found_smaller_shunt_fF']:.1f}"] for r in ranges],
        [145,width-145])
    add("Example: C1/C2=115/80 fF, equal readout caps 6.5859 fF, C12=2.4947 fF and EJ1/EJ2=16.6804/12.3224 GHz give a 4.4 GHz carrier, 100 MHz gap, R=0.980%, H=0.9776 and dispersive maximum 0.0857. Both EJ/EC ratios exceed 50.","small")
    add("Interpretation and limits", "h")
    add("<b>Symmetry:</b> swapping the complete rails mirrors the map across C1=C2. Over both orientations, EC1 and EC2 have identical CDFs. The earlier CDF over only C1≥C2 compared larger- and smaller-capacitance rails, not two intrinsically different transmons. Loaded EC is calculated from the complete inverse capacitance matrix.")
    add("<b>No rigorous boundary:</b> table endpoints are sampled passing examples, not certified continuous intervals. Gray means no solution found by a bounded search. Exact original shunt pairs were also searched with adjustable couplings/EJs; no passing examples were found, which is not proof of impossibility. Search branches targeted R=0.98% to leave numerical margin below 1%.","small")
    add("<b>Physical scope:</b> no finite-drive transitions, readout speed, minimum |χ|, Purcell-filter dynamics or frequency-noise protection were verified. The 50 Ω specification is equivalent-mode impedance, not automatically CPW characteristic impedance. Fourteen representative robustness points retained classifications in a larger basis; workflow regression tests passed.","small")
    add('AWS context: the <link href="https://arxiv.org/html/2604.16292v1" color="#163A59">2026 symmetric-readout experiment</link> reports a 94.15 MHz gap and excellent matching, but does not publish enough EC/EJ/transverse-coupling data to verify all our numerical cutoffs. Its reported resonator frequency is 6.7916 GHz in the ground state, not our 6 GHz bare input.',"small")
    add("Reproducibility: readout_workflow/data/original_design_audit/; equal_g/robustness/samples_2pct_limit_0.01/; shunt_pair_map/. These folders retain input parameters, attempted solutions, validation data and detailed reports.","small")

    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#CED8E0"))
        canvas.line(42,35,A4[0]-42,35)
        canvas.setFont("DV",7);canvas.setFillColor(GRAY)
        canvas.drawString(42,23,"14 October 2026 | Numerical readout study | Historical results: 95-105 MHz gap")
        canvas.drawRightString(A4[0]-42,23,f"{doc.page}")
        canvas.restoreState()
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=45,
        title="Transmon dual-rail readout: daily findings, 14 October 2026",author="Readout workflow study")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    pdf=fitz.open(OUT)
    for i,page in enumerate(pdf):
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(PREVIEW/f"page_{i+1}.png")
    print(json.dumps({"pdf":str(OUT),"pages":len(pdf),"page_characters":[len(p.get_text()) for p in pdf]},indent=2))
    pdf.close()

if __name__=="__main__": main()
