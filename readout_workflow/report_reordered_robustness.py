"""EC-first robustness; logical-frequency relaxation before H relaxation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reordered_robustness import TIERS,reasons
from run_reordered_robustness import OUT,OLD,BASE

NAMES=['Baseline',r'$E_C/h$ floor'+'\n120 MHz','Free\noperation\nfrequency','Logical\nfrequency\n80-120 MHz',r'$H$ floor'+'\n90%',r'$E_J/E_C$ floor'+'\n40']


def main():
    rows=json.loads((OUT/'samples.json').read_text());designs=json.loads((OUT/'designs.json').read_text())
    old=json.loads((OLD/'analysis.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
    assert len(rows)==900
    def passed(r,i): return r['first_pass'] is not None and r['first_pass']<=i
    results={}
    for label in designs:
        group=[r for r in rows if r['design']==label]
        counts={kind:dict(total=sum(r['kind']==kind for r in group),
            cumulative=[sum(passed(r,i) for r in group if r['kind']==kind) for i in range(6)])
            for kind in ('interior','corner','one_at_a_time')}
        gains=[]
        for i in range(6):
            chosen=[r for r in group if r['first_pass']==i];points=[r['stages'][-1]['point'] for r in chosen]
            gains.append(dict(tier=TIERS[i]['name'],count=len(chosen),
                H_range=[min(p['hybridization'] for p in points),max(p['hybridization'] for p in points)] if points else None,
                gap_mhz_range=[min(p['splitting_ghz']*1000 for p in points),max(p['splitting_ghz']*1000 for p in points)] if points else None,
                min_EC_mhz=min(min(p['loaded_ec1_mhz'],p['loaded_ec2_mhz']) for p in points) if points else None))
        results[label]=dict(counts=counts,cumulative=[sum(passed(r,i) for r in group) for i in range(6)],gains=gains,
            unresolved=[r['id'] for r in group if r['first_pass'] is None],numerical_errors=[r for r in group if r.get('error')])
        if label in ('1','2'):
            results[label]['same_constraint_research_gain']=results[label]['cumulative'][2]-sum(
                old[label]['counts'][kind]['cumulative'][1] for kind in ('interior','corner','one_at_a_time'))
    (OUT/'analysis.json').write_text(json.dumps(results,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
    cdf_statistics={}
    for label in designs:
        y=results[label]['counts']['interior'];axes[0].plot(range(6),np.array(y['cumulative'])/y['total']*100,'o-',label=f'Design {label}')
        values=[]
        for row in rows:
            if row['design']!=label or row['kind']!='interior': continue
            candidates=[s['point'] for s in row['stages'] if s['tier'] in ('baseline','EC120')]
            eligible=[p for p in candidates if not [v for v in reasons(p,TIERS[1],designs[label]['operating_frequency_ghz']) if v!='hybridization']]
            if eligible: values.append(max(p['hybridization'] for p in eligible))
        values=np.sort(values)
        cdf_statistics[label]=dict(eligible=len(values),total_interior=224,hybridization_values=values.tolist(),
            criterion='EC >=120 MHz; fixed nominal operation frequency; chi <=1%; all other baseline screens except H')
        if len(values): axes[1].step(values,np.arange(1,len(values)+1)/len(values),where='post',
            label=f'Design {label}: {len(values)}/224 eligible')
    axes[0].set(xticks=range(6),xticklabels=NAMES,ylabel='Cumulative interior pass fraction (%)',
        ylim=(0,105),xlim=(-.2,5.2),title='Cumulative rescue ladder',xlabel='Successive cumulative relaxations')
    axes[0].tick_params(axis='x',labelsize=13,pad=8)
    axes[1].axvline(.95,color='black',ls='--',lw=1)
    axes[1].set(xlabel='Hybridization score H',ylabel='Conditional empirical CDF',ylim=(0,1.02),
        title='Fixed operation frequency\nEC floor 120 MHz')
    for ax in axes:
        ax.grid(alpha=.25);ax.legend(fontsize=11,loc='upper left')
        ax.tick_params(axis='y',labelsize=12)
        ax.xaxis.label.set_size(13);ax.yaxis.label.set_size(13);ax.title.set_size(13)
    axes[1].tick_params(axis='x',labelsize=12)
    (OUT/'cdf_statistics.json').write_text(json.dumps(cdf_statistics,indent=2))
    fig.savefig(OUT/'reordered_comparison.png',dpi=200);plt.close(fig)
    make_pdf(results,designs,audit)
    print(json.dumps({k:{'cumulative':v['cumulative'],'counts':v['counts']} for k,v in results.items()},indent=2))


def make_pdf(results,designs,audit):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Image,Spacer,PageBreak
    fonts=Path(matplotlib.get_data_path())/'fonts'/'ttf'
    pdfmetrics.registerFont(TTFont('RO',str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('RO-Bold',str(fonts/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('RO',normal='RO',bold='RO-Bold')
    blue=colors.HexColor('#173B59');width=A4[0]-80;story=[]
    styles={k:ParagraphStyle(k,fontName='RO-Bold' if k in ('title','h') else 'RO',fontSize=s,leading=l,
        spaceAfter=7,textColor=blue if k in ('title','h') else colors.black)
        for k,s,l in (('title',18,23),('h',11,15),('body',8.7,12),('small',7.3,10))}
    def add(text,k='body'): story.append(Paragraph(text,styles[k]))
    def table(data,widths):
        t=Table([[Paragraph(str(v),styles['small']) for v in r] for r in data],colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfeaf3')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f7')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([t,Spacer(1,8)])
    add('Reordered robustness rescue study','title')
    add('15 October 2026 | Same 3 designs and 900 fabricated circuits | Logical-frequency relaxation before H relaxation','small')
    add('Operation frequency preserves the previous mean dressed logical-transition frequency, as confirmed by the user. Only the terminology changes; it is not redefined as the mean loaded bare-transmon frequency.')
    table([['Design','C1 / C2 (fF)','Nominal operation frequency (GHz)']]+[
        [k,f"{p['device']['C1_shunt_fF']:g} / {p['device']['C2_shunt_fF']:g}",f"{p['operating_frequency_ghz']:.6f}"]
        for k,p in designs.items()],[65,150,width-215])
    add('Identical perturbations and unchanged hard constraints','h')
    add('For each design: 224 Latin-hypercube interior samples, all 64 six-parameter corners, and 12 isolated endpoints. C1, C2, C12, C1r, C2r and bare resonator frequency independently vary within +/-2%; the same vectors are reused exactly. The nominal bare resonator is 6 GHz at 50 ohm equivalent-mode impedance; Cr and Lr both scale inversely with the perturbed bare frequency.')
    add('Both EJs may recalibrate; all perturbed physical components remain fixed. Every tier retains worst-normalized chi mismatch <=1%, multilevel dispersive ratio <=0.1, reference overlap >=0.95, both EJ/h <25 GHz, and both loaded EC/h <400 MHz. The EC lower bound, logical-frequency window, H floor and EJ/EC floor follow this cumulative ladder:')
    table([['Tier','EC floor (MHz)','Operation frequency','Logical frequency (MHz)','H floor','EJ/EC']]+[
        [i,f"{t['ec_min']:g}",'Nominal' if t['fixed'] else '4-8 GHz',f"{1000*t['gap'][0]:g}-{1000*t['gap'][1]:g}",
         f"{t['H']:.2f}",f">{t['ej_ec']:g}"] for i,t in enumerate(TIERS)],
        [35,90,140,100,65,width-430])
    add('Cumulative passes out of all 300 samples','h')
    table([['Design','Baseline','+ EC120','+ free operation frequency','+ logical frequency, H95','+ H90','+ EJ/EC40']]+[
        [k,*r['cumulative']] for k,r in results.items()],[55,*([float((width-55)/6)]*6)])
    add('Unchanged baseline calculations are reused. Earlier accepted points are tested against the new tiers and recomputed at unchanged EJs in the larger basis. New searches are performed where the feasible set changes. A later-discovered witness is credited to the earliest tier it actually satisfies. This is a bounded design search, not a proof of global infeasibility or optimality.','small')
    story.append(PageBreak());add('Comparison and interpretation','title')
    im=Image(str(OUT/'reordered_comparison.png'));im.drawHeight=width*im.imageHeight/im.imageWidth;im.drawWidth=width;story.append(im)
    add('CDF: the 224 interior samples per design are screened at fixed nominal operation frequency, EC/h >=120 MHz and chi mismatch <=1%, with all other baseline constraints except H retained. Each eligible sample contributes its best-found score; the legend gives the eligible count. The dashed line marks H=95%. Excluded samples receive no invented H value.','small')
    add('Separate random interiors from stress tests','h')
    data=[['Design / samples','Baseline','EC120','Free operation frequency','Logical frequency, H95','H90','EJ/EC40']]
    for k,r in results.items():
        for kind,short in (('interior','interior'),('corner','corners'),('one_at_a_time','endpoints')):
            c=r['counts'][kind];data.append([f"{k}: {short} ({c['total']})",*[str(v) for v in c['cumulative']]])
    table(data,[130,*([float((width-130)/6)]*6)])
    add('What changed before lowering H?','h')
    for label in ('1','2'):
        r=results[label];g=r['gains'][3]
        detail=f" Newly rescued logical frequencies: {g['gap_mhz_range'][0]:.2f}-{g['gap_mhz_range'][1]:.2f} MHz." if g['count'] else ''
        add(f"Design {label}: {r['cumulative'][2]}/300 pass with free operation frequency and H>=95%; widening the logical-frequency window while retaining H>=95% raises that to {r['cumulative'][3]}/300. Another {r['cumulative'][4]-r['cumulative'][3]} are rescued after lowering H to 90%."+detail)
        if label=='2':
            add('Representative Design 2 sample: operation frequency 4.00017 GHz, logical frequency 90.917 MHz and H=94.823%. The logical frequency is already inside the original window, and the operation frequency is near its 4 GHz lower limit; widening the logical-frequency window does not remove this sample\'s H deficit.','small')
        if r.get('same_constraint_research_gain',0):
            add(f"Additional warm starts found {r['same_constraint_research_gain']} more solutions in the unchanged free-operation-frequency feasible set than the earlier bounded search. That search gain is already included before the logical-frequency step.",'small')
    r=results['3'];g=r['gains'][1]
    ec_text=f"The newly EC-rescued witnesses reach a minimum loaded EC/h of {g['min_EC_mhz']:.3f} MHz; these +/-2% samples do not explore EC values all the way to 120 MHz." if g['count'] else 'No sample was rescued by lowering the EC floor at fixed operation frequency.'
    add(f"Design 3: lowering the EC floor alone raises acceptance from {r['cumulative'][0]}/300 to {r['cumulative'][1]}/300, with H still >=95% and the original logical-frequency and EJ/EC limits. "+ec_text)
    add(f"All accepted witnesses use validation basis 20/9/8/40. Independent audit checked {audit['accepted_checks']} accepted witnesses and {audit['extra_basis_samples']} additional 25/12/10/60 cases; {len(audit['extra_basis_failures'])} convergence threshold changes, {len(audit['numerical_errors'])} numerical errors.",'small')
    add('Combined counts include deliberately oversampled stress corners and are not fabrication yields. No minimum absolute chi, measurement speed, finite-drive, Purcell-filter or frequency-noise condition is imposed. Full perturbations, physical circuits, EJs, searches and per-witness relaxation requirements are saved in the data directory.','small')
    path=BASE.parent/'output'/'pdf'/'reordered_robustness_2026-10-15.pdf'
    def footer(canvas,doc):
        canvas.setFont('RO',7);canvas.setFillColor(colors.HexColor('#526372'))
        canvas.drawString(40,24,'EC120 -> free operation frequency -> logical frequency 80-120 -> H90 -> EJ/EC40')
        canvas.drawRightString(A4[0]-40,24,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=36,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    import pymupdf
    render=OUT/'rendered';render.mkdir(exist_ok=True)
    with pymupdf.open(path) as doc:
        for i,p in enumerate(doc): p.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(render/f'report_{i+1}.png')


if __name__=='__main__': main()
