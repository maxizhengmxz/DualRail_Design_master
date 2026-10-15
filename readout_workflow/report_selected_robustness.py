"""Summaries, figures and a concise PDF for the selected robustness study."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from run_selected_robustness import OUT,BASE
from robustness_rescue import PARAMETERS,TIERS,reasons

LABELS=['Fixed carrier','Carrier 4-8 GHz','H >=90%','Gap 80-120 MHz','EJ/EC >40']
NAMES=['C1','C2','C12','C1r','C2r','Bare fr']


def passing(row,index):
    return row.get('first_pass') is not None and row['first_pass']<=index


def baseline_point(row):
    if row.get('first_pass')==0 and row.get('stages'): return row['stages'][-1]['point']
    return row['stages'][0]['point'] if row.get('stages') else None


def summarize(rows,designs):
    analyses={}
    for label,nominal in designs.items():
        group=[r for r in rows if r['design']==label]
        interior=[r for r in group if r['kind']=='interior']
        counts={kind:dict(total=sum(r['kind']==kind for r in group),
            cumulative=[sum(passing(r,i) for r in group if r['kind']==kind) for i in range(5)],
            first_pass=[sum(r.get('first_pass')==i for r in group if r['kind']==kind) for i in range(5)])
            for kind in ('interior','corner','one_at_a_time')}
        causes=Counter()
        for row in group:
            if row.get('first_pass') is None:
                if row.get('error'): causes['numerical_error']+=1
                elif row.get('structural_failure'): causes[row['structural_failure']]+=1
                elif row['stages']: causes.update(row['stages'][-1]['failures'])
        conditional=[r for r in interior if baseline_point(r) is not None and
            not [v for v in reasons(baseline_point(r),TIERS[0],nominal['operating_frequency_ghz']) if v!='hybridization']]
        associations={}
        for i,key in enumerate(PARAMETERS):
            x=np.array([r['errors'][i] for r in interior])
            base=np.array([passing(r,0) for r in interior],float)
            final=np.array([passing(r,4) for r in interior],float)
            correlations={}
            for name,y in (('baseline_pass',base),('final_pass',final)):
                correlations[name]=float(spearmanr(x,y).statistic) if np.std(y)>0 else None
            failed=[r['errors'][i]*100 for r in interior if not passing(r,4)]
            associations[key]=dict(**correlations,failed_mean_error_percent=float(np.mean(failed)) if failed else None)
        oat=[]
        for i,key in enumerate(PARAMETERS):
            values=sorted([r for r in group if r['kind']=='one_at_a_time' and r['parameter']==key],key=lambda r:r['errors'][i])
            oat.append(dict(parameter=key,minus_first_pass=values[0].get('first_pass'),plus_first_pass=values[1].get('first_pass'),
                minus_structural=values[0].get('structural_failure'),plus_structural=values[1].get('structural_failure')))
        rescues=[]
        for i in range(1,5):
            rescued=[r for r in group if r.get('first_pass')==i]
            points=[r['stages'][-1]['point'] for r in rescued]
            rescues.append(dict(tier=TIERS[i]['name'],count=len(points),
                actual_relaxation_patterns=dict(Counter(' + '.join(r.get('required_relaxations_for_witness',[])) or 'none' for r in rescued)),
                H_range=[min(p['hybridization'] for p in points),max(p['hybridization'] for p in points)] if points else None,
                gap_mhz_range=[1000*min(p['splitting_ghz'] for p in points),1000*max(p['splitting_ghz'] for p in points)] if points else None,
                min_ej_ec=min(min(p['ej1_over_ec1'],p['ej2_over_ec2']) for p in points) if points else None))
        analyses[label]=dict(nominal=nominal,counts=counts,unrescued_causes=dict(causes),
            structural_EC_failures=sum(r.get('structural_failure')=='EC_range' for r in group),
            conditional_CDF_count=len(conditional),associations=associations,oat=oat,rescues=rescues,
            baseline_total=sum(passing(r,0) for r in group),final_total=sum(passing(r,4) for r in group))
    (OUT/'analysis.json').write_text(json.dumps(analyses,indent=2))
    return analyses


def figures(rows,designs,analyses):
    colors=['tab:blue','tab:orange','tab:green']
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for color,(label,nominal) in zip(colors,designs.items()):
        a=analyses[label];counts=a['counts']['interior']
        axes[0].plot(range(5),np.array(counts['cumulative'])/counts['total']*100,'o-',color=color,label=f'Design {label}')
        vals=[]
        for row in rows:
            if row['design']!=label or row['kind']!='interior': continue
            p=baseline_point(row)
            if p is not None and not [v for v in reasons(p,TIERS[0],nominal['operating_frequency_ghz']) if v!='hybridization']:
                vals.append(p['hybridization'])
        vals=np.sort(vals)
        if len(vals): axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=color,
            label=f'Design {label}: {len(vals)}/224 eligible')
    axes[0].set(xticks=range(5),xticklabels=['Fixed\ncarrier','Free\ncarrier','H floor\n90%','Gap\n80-120','EJ/EC\n>40'],
        ylabel='Cumulative interior pass fraction (%)',ylim=(0,103),title='Same samples; cumulative rescue ladder')
    axes[1].axvline(.95,color='black',ls='--',lw=1)
    axes[1].set(xlabel='Best-found fixed-carrier hybridization',ylabel='Conditional empirical CDF',
        ylim=(0,1.02),title='1% matching; all other baseline screens met')
    for ax in axes: ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(OUT/'rescue_and_cdf.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,color,(label,nominal) in zip(axes,colors,designs.items()):
        group=[r for r in rows if r['design']==label and r['kind']=='interior']
        for category,marker,col in (('Baseline','o',color),('Rescued','^','tab:purple'),('Unrescued','x','tab:red')):
            subset=[r for r in group if ('Baseline' if passing(r,0) else 'Rescued' if passing(r,4) else 'Unrescued')==category]
            ax.scatter([100*r['errors'][0] for r in subset],[100*r['errors'][5] for r in subset],
                marker=marker,color=col,s=20,alpha=.65,label=f'{category} ({len(subset)})')
        ax.set(xlabel='C1 error (%)',ylabel='Bare resonator frequency error (%)',xlim=(-2.1,2.1),ylim=(-2.1,2.1),
            title=f"Design {label}: {nominal['device']['C1_shunt_fF']:g}, {nominal['device']['C2_shunt_fF']:g} fF")
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'failure_locations.png',dpi=190);plt.close(fig)


def report(rows,designs,analyses):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Image,Spacer,PageBreak
    fontdir=Path(matplotlib.get_data_path())/'fonts'/'ttf'
    for suffix,filename in (('','DejaVuSans.ttf'),('-Bold','DejaVuSans-Bold.ttf')):
        pdfmetrics.registerFont(TTFont('SR'+suffix,str(fontdir/filename)))
    pdfmetrics.registerFontFamily('SR',normal='SR',bold='SR-Bold')
    blue=colors.HexColor('#173B59');width=A4[0]-80
    styles={k:ParagraphStyle(k,fontName='SR-Bold' if k in ('title','h') else 'SR',
        fontSize=s,leading=l,spaceAfter=7,textColor=blue if k in ('title','h') else colors.black)
        for k,s,l in (('title',18,23),('h',11,15),('body',8.6,12),('small',7.2,9.5))}
    story=[]
    def add(text,k='body'): story.append(Paragraph(text,styles[k]))
    def table(data,widths):
        t=Table([[Paragraph(str(v),styles['small']) for v in row] for row in data],colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfeaf3')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f7')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([t,Spacer(1,8)])
    def picture(name):
        im=Image(str(OUT/name));im.drawHeight=width*im.imageHeight/im.imageWidth;im.drawWidth=width;story.append(im)
    add('Selected-design robustness and rescue','title')
    add('15 October 2026 | 3 designs x 300 paired perturbations | Both EJs recalibrated','small')
    add('Nominal physical circuits','h')
    data=[['Design','C1 / C2 (fF)','C12 / Cg (fF)','Operation (GHz)','EC1 / EC2 (MHz)']]
    for label,p in designs.items():
        d=p['device'];data.append([label,f"{d['C1_shunt_fF']:g} / {d['C2_shunt_fF']:g}",f"{d['C12_fF']:.3f} / {d['C1r_fF']:.3f}",
            f"{p['operating_frequency_ghz']:.6f}",f"{p['loaded_ec1_mhz']:.2f} / {p['loaded_ec2_mhz']:.2f}"])
    table(data,[42,110,110,105,width-367])
    add('Each nominal circuit comes from the validated map. Picks were stratified into approximately equal viable-area C1 bands, then deliberately redrawn toward lower capacitances at the user\'s request; they are not an unbiased three-design yield survey.')
    add('Perturbation and recalibration protocol','h')
    add('Each of C1, C2, C12, C1r, C2r and bare resonator frequency is independently allowed a relative error from -2% to +2%. The same 300 six-dimensional vectors are applied to all designs: 224 Latin-hypercube interior samples, all 64 extreme corners, and 12 single-parameter endpoints (seed 20261015). Interior counts are separate from stress tests.')
    add('Bare fr = 6 GHz nominal; mode impedance Z = 50 ohm remains fixed. For a frequency error epsilon, Cr and Lr are both divided by 1+epsilon, preserving sqrt(Lr/Cr). Readout capacitors vary independently, so equal Cg is nominal only. Actual inverse capacitance matrices set loaded ECs and all couplings.')
    add('Only the EJs are recalibrated after perturbation. Fixed-carrier calibration first seeks the highest hybridization under chi matching and the other screens, at each design\'s own nominal operation frequency (100 Hz carrier tolerance). Failures then enter the cumulative ladder below. Numerical search outcomes are not global impossibility proofs.')
    table([['Tier','Carrier','H floor','Gap (MHz)','EJ/EC floor'],
        ['0: baseline','Nominal','.95','90-110','>50'],['1: carrier rescue','4-8 GHz','.95','90-110','>50'],
        ['2: hybridization rescue','4-8 GHz','.90','90-110','>50'],['3: gap rescue','4-8 GHz','.90','80-120','>50'],
        ['4: energy-ratio rescue','4-8 GHz','.90','80-120','>40']],[140,100,60,100,width-400])
    add('Unchanged at every tier: max_i |(chi+ - chi-)/chi_i| <=1%; maximum multilevel dispersive ratio <=0.1; reference overlap >=0.95; both loaded EC/h >=150 and <400 MHz; both EJ/h <25 GHz. Reducing the EJ/EC floor never relaxes the EC range or EJ ceiling. H is a charge-transition balance score, not Bell-state fidelity.','small')
    story.append(PageBreak());add('Pass fractions and rescue gains','title')
    data=[['Design / samples','Baseline','+ carrier','+ H90','+ gap','+ EJ/EC40']]
    for label,a in analyses.items():
        for kind,short in (('interior','interior'),('corner','corners'),('one_at_a_time','endpoints')):
            c=a['counts'][kind];data.append([f'{label}: {short} (n={c["total"]})',*[f'{n}/{c["total"]}' for n in c['cumulative']]])
    table(data,[145,*([float((width-145)/5)]*5)])
    picture('rescue_and_cdf.png')
    add('Counts are cumulative on the same samples. A solution found later is credited to the earliest tier it actually satisfies, preventing false rescue gains. The conditional CDF includes only interior samples with fixed-carrier points meeting chi matching and all other baseline screens except H. Best-found H is not a certified global maximum; the dashed line marks 95%.','small')
    add('Remaining failures','h')
    for label,a in analyses.items():
        causes=', '.join(f"{key.replace('_',' ')}: {value}" for key,value in a['unrescued_causes'].items()) or 'none'
        add(f"Design {label}: {a['baseline_total']}/300 baseline passes and {a['final_total']}/300 after all rescues. Remaining failure causes: {causes}.")
    add('Constraint margins help interpret these counts: Design 1 starts at a 90.10 MHz gap and minimum EJ/EC=50.28, close to two baseline floors. Design 3 starts at EC1/h=151.38 MHz, only 0.92% above the EC floor. Its exact nominal sensitivity d ln(EC1)/d ln(C1) is -0.930, so a positive C1 error of about 1% can breach that floor. EJ recalibration cannot repair a capacitance-set EC.','small')
    add('The combined 300-point fraction is not a fabrication yield: extreme corners and endpoint controls are deliberately overrepresented. Even the interior fraction describes the chosen independent uniform-error sampling convention, not a measured process distribution.','small')
    story.append(PageBreak());add('Sensitivity and actual rescue witnesses','title')
    picture('failure_locations.png')
    add('Each panel uses only the 224 interior vectors. Other four errors also vary; these projections show associations, not causal isolation. Single-parameter endpoints below provide controlled comparisons.','small')
    def display(index,structural):
        return 'EC failure' if structural else 'No rescue' if index is None else str(index)
    data=[['Varied parameter','Design 1: -2% / +2%','Design 2: -2% / +2%','Design 3: -2% / +2%']]
    for i,name in enumerate(NAMES):
        line=[name]
        for label in designs:
            r=analyses[label]['oat'][i]
            line.append(display(r['minus_first_pass'],r['minus_structural'])+' / '+display(r['plus_first_pass'],r['plus_structural']))
        data.append(line)
    table(data,[100,*([float((width-100)/3)]*3)])
    add('Endpoint cells give the first successful tier (0-4); EC failure cannot be repaired by any allowed relaxation. All other errors are zero for these endpoints.','small')
    data=[['Design / newly rescued tier','Count','Achieved H range','Gap range (MHz)','Lowest EJ/EC']]
    for label,a in analyses.items():
        for i,r in enumerate(a['rescues'],1):
            if not r['count']: continue
            data.append([f'{label} / tier {i}',r['count'],f"{r['H_range'][0]:.4f}-{r['H_range'][1]:.4f}",
                f"{r['gap_mhz_range'][0]:.2f}-{r['gap_mhz_range'][1]:.2f}",f"{r['min_ej_ec']:.3f}"])
    table(data,[145,45,105,120,width-415])
    add('Table values are rounded; acceptance is checked before rounding. These witnesses do not prove the minimum relaxation required. Later tiers permit earlier relaxations, but a witness need not use all of them; a gap-rescued point may still have H above 95%. Actual relaxation patterns are recorded per sample and in analysis.json.','small')
    audit=json.loads((OUT/'audit.json').read_text()) if (OUT/'audit.json').exists() else {}
    add('Numerical validation and scope','h')
    add(f"Every accepted witness is recomputed at unchanged component values and EJs in basis 20/9/8/40, versus screening 15/6/5/20; larger-basis repair is attempted if necessary. Independent audit: {audit.get('accepted_checks','pending')} accepted constraints checked; {audit.get('extra_basis_samples','pending')} extra 25/12/10/60 checks, with {len(audit.get('extra_basis_failures',[]))} threshold changes.", 'small')
    add('No minimum absolute chi, readout speed, finite-drive, Purcell-filter or frequency-noise condition is imposed. This is an equivalent-mode LC model; CPW characteristic impedance requires a consistent mode conversion. Full samples, device values, EJs, failed checks and solver attempts are retained in the data directory.','small')
    path=BASE.parent/'output'/'pdf'/'selected_robustness_2026-10-15.pdf';path.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('SR',7);canvas.setFillColor(colors.HexColor('#526372'))
        canvas.drawString(40,24,'Paired +/-2% component errors | 1% chi matching | EJ-only recalibration')
        canvas.drawRightString(A4[0]-40,24,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=36,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    import pymupdf
    rendered=OUT/'rendered';rendered.mkdir(exist_ok=True)
    with pymupdf.open(path) as doc:
        for i,page in enumerate(doc): page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(rendered/f'report_{i+1}.png')
    return path


def main():
    rows=json.loads((OUT/'samples.json').read_text());designs=json.loads((OUT/'designs.json').read_text())
    assert len(rows)==900 and len({(r['design'],r['id']) for r in rows})==900
    a=summarize(rows,designs);figures(rows,designs,a);path=report(rows,designs,a)
    (OUT/'report.md').write_text('# Selected-design robustness\n\nPDF: '+str(path)+'\n\n'+
        '\n'.join(f"Design {k}: baseline {v['baseline_total']}/300; final {v['final_total']}/300; interior cumulative {v['counts']['interior']['cumulative']}; remaining causes {v['unrescued_causes']}" for k,v in a.items()),encoding='utf-8')
    print(json.dumps({k:{'counts':v['counts'],'unrescued':v['unrescued_causes']} for k,v in a.items()},indent=2))


if __name__=='__main__': main()
