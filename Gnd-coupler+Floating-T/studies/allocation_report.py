"""Render saved self/ground-allocation results; never launch physics searches."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
def load(path):return json.loads(path.read_text())

def build(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    common=load(ROOT/'output/common-rail-allocation-20261118/result.json')
    off=load(ROOT/'output/common-allocation-off-20261118/result.json')
    a2=load(ROOT/'output/a2-residual-rms-20261118/sweep.json')
    move=load(ROOT/'output/a2-matched-allocation-20261118/sweep.json')
    asym=load(ROOT/'output/a2-asymmetric-eta05-rms-loading-20261118/sweep.json')
    nominal=load(ROOT/'configs/Design_V1_20261105.json');ref=common['reference']
    for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyA',fontName='Arial',fontSize=9.4,leading=13,spaceAfter=7))
    styles.add(ParagraphStyle(name='TitleA',fontName='Arial-Bold',fontSize=20,leading=24,spaceAfter=12,textColor=colors.HexColor('#173b55')))
    styles.add(ParagraphStyle(name='HeadingA',fontName='Arial-Bold',fontSize=13,leading=16,spaceAfter=9,textColor=colors.HexColor('#173b55')))
    styles.add(ParagraphStyle(name='SmallA',fontName='Arial',fontSize=8,leading=10.5,spaceAfter=5))
    story=[]
    def p(t,small=False):story.append(Paragraph(t,styles['SmallA' if small else 'BodyA']))
    def h(t):story.append(Paragraph(t,styles['HeadingA']))
    def table(rows,widths=None):
        wrapped=[[Paragraph(str(v),styles['SmallA']) for v in row] for row in rows]
        t=Table(wrapped,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f9')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),
            ('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8ba6b8'))]))
        story.extend([t,Spacer(1,9)])
    def picture(path,width=490):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:w,hh=im.size
        story.append(Image(str(path),width=width,height=width*hh/w))
    def page():story.append(PageBreak())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # Convert saved historical controls to the same physical self fraction as
    # the all-rail study. Never use 1-eta: its loaded-budget denominator differs.
    def self_fraction(row):
        cs=row['A2_self_ff']
        if row.get('A2_ground_each_ff') is not None:
            ground=row['A2_ground_each_ff']/2
        else:
            a,b=row['A21_ground_ff'],row['A22_ground_ff'];ground=a*b/(a+b)
        return cs/(cs+ground)
    ph=nominal['physical'];cs0=ph['pad_mutual_capacitances_ff']['A2']
    nominal_p=cs0/(cs0+ph['ground_capacitances_ff']['A21']/2)
    fig,axes=plt.subplots(1,2,figsize=(10,3.45))
    for d,label in [(move,'Minimize capacitor movement'),(a2,'Minimize residual changes')]:
        rr=sorted([r for r in d['records'] if r['status']=='matched'],key=self_fraction)
        x=[self_fraction(r)*100 for r in rr]
        axes[0].plot(x,[abs(r['g01_A1_C_mhz']) for r in rr],label=label)
        keys=[f'g01_{a}_{b}_mhz' for a,b in d['remaining_pairs']]
        axes[1].plot(x,[np.sqrt(np.mean([(r[k]-d['reference'][k])**2 for k in keys])) for r in rr],label=label)
    for ax in axes:
        ax.axvline(nominal_p*100,color='.5',ls=':',label='Nominal A2 allocation');ax.set_xlabel('A2 physical self fraction p (%)');ax.grid(alpha=.2)
    axes[0].set_ylabel('|g(A1,C)| (MHz)');axes[1].set_ylabel('RMS residual change (MHz)')
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'a2-only.png',dpi=190);plt.close(fig)
    rr=[r for r in common['records'] if r['status']=='matched']
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for k,label in [('g01_A1_C_mhz','A1-C'),('g01_C_B2_mhz','C-B2'),('g01_C_r1_mhz','C-r1'),('g01_C_r2_mhz','C-r2')]:
        axes[0].plot([r['self_fraction']*100 for r in rr],[abs(r[k]) for r in rr],label=label)
    axes[0].set_ylabel('Local exchange magnitude (MHz)');axes[0].legend(fontsize=8)
    axes[1].plot([r['self_fraction']*100 for r in rr],[r['rms_unintended_exchange_mhz'] for r in rr],label='All 12 unintended terms')
    axes[1].axhline(common['nominal_rms_unintended_exchange_mhz'],color='.5',ls=':',label='Original nominal RMS')
    axes[1].set_ylabel('RMS local exchange (MHz)');axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axvspan(87,90,color='#f2dfdc',alpha=.8);ax.set_xlim(10,90);ax.set_xlabel('Common physical self fraction p (%)');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'all-rails.png',dpi=190);plt.close(fig)
    rr_off=[r for r in off['records'] if r['label']!='nominal'];base=off['records'][0]['audit']['recalibrated']
    fig,axes=plt.subplots(1,2,figsize=(10,3.2));x=[float(r['label'])*100 for r in rr_off]
    axes[0].plot(x,[(r['audit']['recalibrated']['fc_ghz']-base['fc_ghz'])*1000 for r in rr_off],'o-')
    axes[0].set_ylabel('OFF-frequency shift from nominal (MHz)')
    for k,label in [('logical_01_10_g_hz','01L - 10L'),('logical_00_11_g_hz','00L - 11L')]:
        axes[1].plot(x,[r['audit']['recalibrated'][k]-base[k] for r in rr_off],'o-',label=label)
    axes[1].set_ylabel('Logical exchange change at retuned OFF (Hz)');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Common physical self fraction p (%)');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'off-response.png',dpi=190);plt.close(fig)

    story.append(Paragraph('Self-capacitance versus ground-capacitance allocation',styles['TitleA']))
    p('Matched intended circuit; A2-only and common-allocation studies | 18 November 2026',True)
    p('<b>Main finding.</b> Compensating physical capacitors can preserve loaded ECs and the intended interactions while changing self/ground allocation. Unintended interactions remain allocation-dependent. Imposing the same physical self fraction on all four rails gives much smaller overall local residuals at lower self fraction, while five-body OFF performance changes only slightly after coupler retuning.')
    h('1. Reference circuit and matching method')
    p('Reference: Design_V1_20261105, grounded C2 and floating A1, A2, B1, B2; no parasitic capacitors. Loaded rail EC tuple: (165, 250, 160, 260) MHz; coupler EC: 450.285 MHz. All five ECs are held fixed. Nominal rail EJs and resonator inductances remain fixed.')
    p('Construct the full nodal capacitance matrix, transform to transmon differential/common coordinates, and eliminate the floating common coordinates. Let C<sub>m</sub> be the reduced modal capacitance matrix and K = C<sub>m</sub><super>-1</super>. Match seven diagonal entries of K and the nine intended off-diagonal entries. The transmon diagonals determine EC; the off-diagonals determine charge coefficients J (J = 4e<super>2</super>K/h). Thus, with unchanged EC and EJ, intended local charge-transition couplings are recovered. Resonator diagonals and fixed inductances preserve the uncoupled modal resonator frequencies.')
    table([['Intended interaction','Matched channels'],['Within rails','A1-A2; B1-B2'],['Coupler links','A2-C; C-B1'],['Direct inter-rail','A2-B1'],['Readout links','A1-r1; A2-r1; B1-r2; B2-r2']],[135,355])
    p('Matching these entries does not match the entire matrix: 12 unintended off-diagonals remain. Local exchange coefficients quoted here include fixed nominal idle/OFF charge matrix elements; they are distinct from effective logical exchange after diagonalization.',True)
    h('One allocation definition throughout this report')
    p('<b>p = Cs / (Cs + Cg,series)</b>, with Cg,series = Cg,each/2 for equal pad grounds. For unequal grounds use Cg,series = Cg1 Cg2 / (Cg1 + Cg2). A larger p always means more self capacitance and less series-ground contribution. The denominator excludes external loading.')
    table([['Study','Displayed allocation coordinate','Other rail allocations'],['A2-only','A2 physical self fraction p','Not tied; may compensate'],['All four rails','Same physical self fraction p on all four rails','Tied to common p']],[80,245,165])
    p('All A2-only axes and numerical labels have been converted from their saved historical ground/loaded-budget control using actual capacitor values. That conversion is not a simple complement. The two studies now use the same fraction and direction. No EC-only sweep is included.',True)
    page()
    h('2. A2-only redistribution: matched EC and intended links')
    p('Only A2\'s allocation coordinate is prescribed. Compensation can change self/ground capacitances of the other rails and existing links; therefore this does not mean that every physical capacitor outside A2 was frozen. Each rail retains equal grounds in the main comparison.')
    p('The earlier optimizer first minimized compensation-capacitor movement. A subsequent version minimized the RMS <b>change from nominal</b> of the 12 unintended exchange coefficients. This differs from the all-rail study, which minimizes their <b>absolute magnitudes</b>.')
    picture(out/'a2-only.png')
    rows=[['A2 self p (%)','|A1-C| MHz','|C-B2| MHz','RMS change MHz']]
    selected=[r for r in a2['records'] if r['status']=='matched' and any(abs(r['eta']-v)<1e-8 for v in [.3,.5,.7,.85,.9])]
    for r in sorted(selected,key=self_fraction):
        rows.append([f'{self_fraction(r)*100:.2f}',f"{abs(r['g01_A1_C_mhz']):.3f}",f"{abs(r['g01_C_B2_mhz']):.3f}",f"{r['rms_unrestrained_exchange_change_mhz']:.3f}"])
    table(rows,[115,125,125,125])
    p(f'Nominal A2 has physical self fraction p={nominal_p*100:.2f}%. At p=44.71%, matching preserves the intended matrix entries but increases A1-C from 0.389 to 4.583 MHz; C-B2 remains near 33.65 MHz. The residual-change objective improves compensation tradeoffs without suppressing the already-large nominal B-side residual.')
    ar=next(r for r in asym['records'] if abs(r['eta']-.5)<1e-8)
    p(f"<b>Approved asymmetric-ground test, physical p={self_fraction(ar)*100:.2f}%.</b> With the loading limits enforced, A21/A22 ground caps became {ar['A21_ground_ff']:.3f}/{ar['A22_ground_ff']:.3f} fF and A2 self-C {ar['A2_self_ff']:.3f} fF. |A1-C| decreased modestly to {abs(ar['g01_A1_C_mhz']):.3f} MHz; RMS residual change was {ar['rms_unrestrained_exchange_change_mhz']:.3f} MHz. The largest compensation-capacitor change was {ar['max_other_cap_change_pct']:.1f}%. This test preserved the selected historical ground contribution, not exactly the same physical p as the equal-ground point.")
    p('The historical equal-ground curves are matrix-matching diagnostics: their saved run did not enforce the later loading-budget acceptance checks. The asymmetric test above and the all-rail results below explicitly enforce the saved loading limits. Do not treat these different objectives and acceptance scopes as a controlled one-variable comparison.',True)
    page()
    h('3. All rails share one physical allocation ratio')
    p('Prescribe common p on A1, A2, B1 and B2, retaining equal pad grounds. Adjust four rail self caps, coupler self-C, nine physical links and two resonator self caps: 16 variables for 16 matching conditions. The equality Jacobian has rank 16 at every accepted point, leaving no continuous freedom at fixed p. Minimize absolute residual exchange among the feasible roots found; vary p to reduce residuals.')
    picture(out/'all-rails.png')
    rows=[['Self p','|A1-C|','|C-B2|','|C-r1|','|C-r2|','RMS (12)']]
    rows.append(['Nominal']+[f'{abs(ref[k]):.3f}' for k in ['g01_A1_C_mhz','g01_C_B2_mhz','g01_C_r1_mhz','g01_C_r2_mhz']]+[f"{common['nominal_rms_unintended_exchange_mhz']:.3f}"])
    for v in [.1,.3,.5,.7,.85]:
        r=next(r for r in rr if r['self_fraction']==v)
        rows.append([f'{v*100:.0f}%']+[f'{abs(r[k]):.3f}' for k in ['g01_A1_C_mhz','g01_C_B2_mhz','g01_C_r1_mhz','g01_C_r2_mhz']]+[f"{r['rms_unintended_exchange_mhz']:.3f}"])
    table(rows,[70,84,84,84,84,84])
    p('All coupling columns are local exchange magnitudes in MHz at nominal idle/OFF controls. Lower self fraction suppresses the overall residuals on this matched family. At p=10%, RMS falls by about 33-fold, but A1-C and C-r1 individually increase slightly relative to nominal.')
    p('<b>Feasibility.</b> Of 81 samples (1 percentage point spacing), 78 matched positive-capacitance circuits were found: p=0.10-0.87. No accepted match was found at 0.88-0.90. Continuing the matched branch with signed capacitance requires A12-r1 = -0.154, -2.393 and -6.863 fF, respectively. These are rejected nonphysical diagnostics; finite local search is not a global infeasibility proof.')
    p('All accepted solutions satisfy the 10% rail branch-loading limit and at least 55% coupler self contribution. Worst relative intended-matrix mismatch: 3.3 x 10<super>-11</super>. At p=10%, reduced C differs by 0.166% in Frobenius norm; small global matrix change can still correspond to large relative changes of small unintended entries.',True)
    page()
    h('4. Five-transmon OFF response and candidate hardware')
    p('For representative common-ratio circuits, freeze all four nominal rail EJs and synthesized EC/J. Exclude resonators dynamically. First compare at the common nominal five-body OFF bias, then retune only coupler EJ within 7-10 GHz. OFF nulls physical 0100-0010 using all four physical single-excitation references. Logical exchanges are residual diagnostics.')
    picture(out/'off-response.png')
    rows=[['Self p','Fixed-bias physical g (kHz)','Retuned OFF (GHz)','01L-10L (Hz)','00L-11L (Hz)']]
    for label in ['nominal','0.1','0.5','0.87']:
        x=next(x for x in off['records'] if x['label']==label);a=x['audit']['recalibrated'];b=x['audit']['common_nominal_five_mode_off']
        rows.append([label,f"{b['physical_g_hz']/1000:+.3f}",f"{a['fc_ghz']:.6f}",f"{a['logical_01_10_g_hz']:.1f}",f"{a['logical_00_11_g_hz']:.1f}"])
    table(rows,[65,115,110,100,100])
    p('The p=10% OFF shift is +5.407 MHz; retuned logical residual changes are only +3.95/+4.30 Hz. A physical root survives at every sampled ratio. This does not mean that local residual suppression produces a logical null. Secular ZZ changes from +12.0 Hz nominal to -8.9 Hz at p=10% (audit basis).')
    best=common['solutions']['0.1']['physical'];rows=[['p=10% rail','Across-pad Cs (fF)','Ground cap per pad (fF)']]
    for n in ['A1','A2','B1','B2']:
        rows.append([n,f"{best['pad_mutual_capacitances_ff'][n]:.3f}",f"{best['ground_capacitances_ff'][best['pad_pairs'][n][0]]:.3f}"])
    table(rows,[90,200,200])
    p('At p=10%: coupler self-C 23.816 fF, self contribution 55.36%; A22-C1 6.680 fF, C1-B11 13.640 fF, direct A22-B11 0.745 fF. A2/B1 full coupling-branch loading is 2.38%/2.14%. This candidate has not replaced the nominal design.')
    p('<b>Numerical scope.</b> OFF production basis: local/pair/coupler levels 6/12/5; audit 7/15/6. Audit controls re-root after fixed-control comparison. Larger endpoint checks (8/18/7) change physical exchange by about 2.5 Hz and logical exchanges by 1.5 Hz. Results support few-Hz comparisons, not certified 1-Hz convergence. No new gate-fidelity or integrated-readout validation was performed.',True)
    p('<b>Saved-data provenance.</b> configs/Design_V1_20261105.json; output/a2-matched-allocation-20261118 and a2-residual-rms-20261118/sweep.json; a2-asymmetric-eta05-rms-loading-20261118/sweep.json; common-rail-allocation-20261118/result.json; common-allocation-off-20261118/result.json. This renderer reads saved data only.',True)
    page()
    h('5. Protocol for changing one rail pair\'s allocation')
    p('<b>Allocation convention.</b> Use p = Cs / (Cs + Cg,series), where Cg,series = Cg1 Cg2 / (Cg1 + Cg2). With equal grounds, Cg,series = Cg,each/2. A larger p always means a larger self-capacitance contribution.')
    h('Step 1. Prescribe the pair and compensation freedom')
    p('Select the pair and target fraction, for example p(A1)=p(A2)=0.30. By default retain equal grounds on each rail transmon and preserve the other pair\'s nominal physical self fractions. Existing physical capacitors may compensate within the agreed loading limits. The other pair\'s self and ground capacitors may scale together without changing its fractions. Do not introduce new links or asymmetric grounds without authorization.')
    h('Step 2. Match the intended effective circuit as hard constraints')
    p('Rebuild the nodal matrix, eliminate floating common coordinates, and invert the reduced modal capacitance matrix at every candidate. Preserve all five loaded ECs, both resonator frequencies with unchanged inductances, and nine intended charge coefficients: A1-A2, B1-B2, A2-C, C-B1, A2-B1, A1-r1, A2-r1, B1-r2 and B2-r2. Enforce the prescribed fractions, positive physical capacitors and loading limits. Never use EJ to compensate an EC mismatch.')
    h('Step 3. Minimize residuals, then physical capacitor movement')
    p('<b>Primary objective:</b> minimize R = sqrt[(1/12) sum |g(i,j)|<super>2</super>] over the 12 unintended off-diagonal interactions. Evaluate every local exchange coefficient at the same nominal idle/OFF controls, using fixed reference charge matrix elements. Minimize absolute residual magnitudes, rather than only changes from nominal; the latter can preserve already-large unwanted terms.')
    p('<b>Secondary objective:</b> among candidates with comparably small R, minimize D = max |(Ck-Ck,nom)/Ck,nom|. Then minimize the sum of squared fractional capacitor changes as a tie-breaker. Prescribed self/ground changes are included when reporting total physical movement; also report compensation-only movement separately. Record the worst individual residual so that an RMS reduction cannot hide a bad channel. Fix the numerical allowance for comparable R before the tie-break search, rather than changing it after seeing the result.')
    h('Step 4. Continue from nominal and retain unsuccessful fits')
    p('Approach the requested ratio in small steps, warm-starting each fit from the previous solution; include independent starts. Check the matching Jacobian rank. If no continuous freedom remains, residual reduction requires choosing another allocation or explicitly authorizing more geometry freedom. If no accepted match is found, retain the closest candidate and identify its limiting capacitor or constraint. Do not silently relax requirements or interpret a finite search as proof of infeasibility.')
    h('Step 5. Validate before adopting the compensated design')
    p('Freeze the synthesized capacitances and nominal rail EJs. Compare physical and both logical exchange channels at the original OFF bias; retune only coupler EJ to recover physical OFF and report the logical residuals, secular ZZ and numerical drift. Then recalibrate the selected CZ channel using the authorized ON controls, reporting duration, raw and postselected fidelity. Finally check integrated readout before replacing the nominal design. Smaller local residuals alone do not certify improved gate fidelity.')
    p('<b>Priority order:</b> hard intended-circuit matching; minimum absolute residual interactions; minimum capacitor movement; explicit OFF, gate and readout validation. This page specifies a future procedure and launches no new calculation.',True)
    def footer(c,doc):
        c.setFont('Arial',8);c.setFillColor(colors.HexColor('#637484'));c.drawString(43,24,'Self/ground allocation | matched nominal circuit | 2026-11-18');c.drawRightString(A4[0]-43,24,str(doc.page))
    pdf=out/'self-ground-capacitance-allocation.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=38,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    print(pdf)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args();build(args.output)
