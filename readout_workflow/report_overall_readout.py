"""Concise overall report: viability, sampled designs and robustness."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import re
import hashlib
from matplotlib.mathtext import math_to_image

BASE=Path(__file__).resolve().parent
MAP=BASE/'data'/'refined_shunt_map_2026_10_14'
ROBUST=BASE/'data'/'reordered_robustness_2026_10_15'
OUT=BASE.parent/'output'/'pdf'
TMP=BASE.parent/'tmp'/'pdfs'/'overall_readout'


def figures(rows,designs):
    TMP.mkdir(parents=True,exist_ok=True)
    grid=np.arange(36.,129.);edges=np.r_[grid-.5,grid[-1]+.5]
    status=np.full((len(grid),len(grid)),-1);values=np.full_like(status,np.nan,dtype=float)
    for r in rows:
        a,b=r['c1'],r['c2']
        if a%1 or b%1 or not 36<=b<a<=128: continue
        i,j=int(a-36),int(b-36)
        if r['status']=='validated_pass': status[j,i]=1;values[j,i]=r['point']['operating_frequency_ghz']
        elif r['status']=='no_witness_found': status[j,i]=0
        else: status[j,i]=2
    fig,ax=plt.subplots(figsize=(6.8,5.4),layout='constrained')
    for code,color in ((0,'#e6e6e6'),(2,'#f7cf99')):
        ax.pcolormesh(edges,edges,np.ma.masked_where(status!=code,np.zeros_like(status)),
                      cmap=ListedColormap([color]),vmin=0,vmax=1,shading='flat')
    mesh=ax.pcolormesh(edges,edges,np.ma.masked_invalid(values),vmin=4,vmax=8,cmap='viridis',shading='flat')
    cb=fig.colorbar(mesh,ax=ax,shrink=.91,pad=.035);cb.set_label('Operation frequency (GHz)',fontsize=11)
    ax.plot([35.5,128.5],[35.5,128.5],':',color='black',lw=.8)
    offsets={'1':(-24,11),'2':(10,12),'3':(9,-19)}
    for label,p in designs.items():
        x,y=p['device']['C1_shunt_fF'],p['device']['C2_shunt_fF']
        ax.scatter(x,y,s=190,marker='*',color='#d21b3c',edgecolors='white',linewidths=.9,zorder=5)
        ax.annotate(label,(x,y),xytext=offsets[label],textcoords='offset points',fontsize=11,weight='bold',
            bbox=dict(boxstyle='round,pad=.15',fc='white',ec='#303030',lw=.6),
            arrowprops=dict(arrowstyle='-',lw=.7,color='black'),zorder=6)
    ax.set(xlim=(35.5,128.5),ylim=(35.5,128.5),xlabel='C1 (fF)',ylabel='C2 (fF)',
           title=r'Validated viable $C_1$-$C_2$ pairs and robustness samples')
    ax.set_xlabel(r'$C_1$ (fF)');ax.set_ylabel(r'$C_2$ (fF)')
    ax.set_aspect('equal');ax.tick_params(labelsize=10)
    fig.savefig(TMP/'viable_pairs_marked.png',dpi=230);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,1.8),layout='constrained')
    h=np.linspace(.8,1,301)
    ax.plot(h*100,(1-h/2)*100,label=r'Larger bare-state population $1-H/2$')
    ax.plot(h*100,h/2*100,label=r'Smaller bare-state population $H/2$')
    ax.axvline(95,ls='--',color='gray',lw=.9)
    ax.scatter([95,95],[52.5,47.5],s=16,zorder=5)
    ax.set(xlabel=r'Hybridization score $H$ (%)',ylabel='Bare-state population (%)',xlim=(80,100),ylim=(38,62),xticks=[80,85,90,95,100])
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='center left')
    fig.savefig(TMP/'mixing.png',dpi=230);plt.close(fig)
    stats=json.loads((ROBUST/'cdf_statistics.json').read_text())
    analysis=json.loads((ROBUST/'analysis.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
    names=['Baseline',r'$E_C/h$ floor'+'\n120 MHz','Flexible\noperation\nfrequency','Logical\nfrequency\n80-120 MHz',r'$H$ floor'+'\n90%',r'$E_J/E_C$ floor'+'\n40']
    for label in designs:
        counts=analysis[label]['counts']['interior']
        axes[0].plot(range(6),np.array(counts['cumulative'])/counts['total']*100,'o-',label=f'Design {label}')
        v=stats[label]['hybridization_values']
        axes[1].step(v,np.arange(1,len(v)+1)/len(v),where='post',label=f'Design {label}: {len(v)}/224 eligible')
    axes[0].set(xticks=range(6),xticklabels=names,ylabel='Cumulative interior pass fraction (%)',ylim=(0,105),xlim=(-.2,5.2),title='Cumulative rescue ladder',xlabel='Successive cumulative relaxations')
    axes[0].tick_params(axis='x',labelsize=13,pad=8)
    axes[1].axvline(.95,color='black',ls='--',lw=1)
    axes[1].set(xlabel=r'Hybridization score $H$',ylabel='Conditional empirical CDF',ylim=(0,1.02),title='Fixed operation frequency\n'+r'$E_C/h$ floor 120 MHz')
    for ax in axes:
        ax.grid(alpha=.25);ax.legend(fontsize=11,loc='upper left')
        ax.tick_params(axis='y',labelsize=12)
        ax.xaxis.label.set_size(13);ax.yaxis.label.set_size(13);ax.title.set_size(13)
    fig.savefig(TMP/'comparison.png',dpi=220);plt.close(fig)


def main():
    rows=json.loads((MAP/'uniform_points.json').read_text());designs=json.loads((ROBUST/'designs.json').read_text())
    analysis=json.loads((ROBUST/'analysis.json').read_text());audit=json.loads((ROBUST/'audit.json').read_text())
    sensitivity=json.loads((ROBUST/'overall_component_sensitivity.json').read_text())
    definitions=json.loads((MAP/'final_grid_definition.json').read_text())
    figures(rows,designs)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Image,Spacer,PageBreak
    fonts=Path(matplotlib.get_data_path())/'fonts'/'ttf'
    pdfmetrics.registerFont(TTFont('OV',str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('OV-Bold',str(fonts/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('OV',normal='OV',bold='OV-Bold')
    blue=colors.HexColor('#173B59');width=A4[0]-80;story=[]
    styles={k:ParagraphStyle(k,fontName='OV-Bold' if k in ('title','h') else 'OV',fontSize=s,leading=l,
        spaceAfter=7,textColor=blue if k in ('title','h') else colors.black)
        for k,s,l in (('title',17.5,22),('h',11,15),('body',8.5,11.5),('small',7.2,9.5))}
    symbols={'C1':'C_1','C2':'C_2','C12':'C_{12}','C1r':'C_{1r}','C2r':'C_{2r}','Cr':'C_r','Lr':'L_r','Cg':'C_g',
             'EJ1':'E_{J1}','EJ2':'E_{J2}','EC1':'E_{C1}','EC2':'E_{C2}','EJ':'E_J','EC':'E_C','EJs':'E_J',
             'ECs':'E_C','H':'H','R':'R','f_op':'f_{\\mathrm{op}}','f_L':'f_L','chi+':r'\chi_+','chi-':r'\chi_-','chi':r'\chi'}
    pattern=re.compile(r'(?<![A-Za-z0-9_])('+ '|'.join(re.escape(t) for t in sorted(symbols,key=len,reverse=True))+r')(?![A-Za-z0-9_])')
    def notation(text,k):
        text=str(text).replace('Free operation frequency','Flexible operation frequency').replace('free operation frequency','flexible operation frequency')
        text=text.replace('EC120',r'$E_C/h\geq120$ MHz').replace('H95',r'$H\geq95\%$').replace('H90',r'$H\geq90\%$').replace('EJ/EC40',r'$E_J/E_C>40$')
        size=styles[k].fontSize
        def formula(expr):
            key=hashlib.sha1((expr+str(size)).encode()).hexdigest()[:16]
            path=TMP/f'math_{key}.png'
            if not path.exists():
                from matplotlib.font_manager import FontProperties
                math_to_image('$'+expr+'$',path,prop=FontProperties(size=size),dpi=240,color='black')
            from PIL import Image as PILImage
            with PILImage.open(path) as im: w,h=im.size
            return f'<img src="{path.as_posix()}" width="{w*72/240:.3f}" height="{h*72/240:.3f}" valign="middle"/>'
        parts=re.split(r'(\$[^$]+\$)',str(text))
        return ''.join(formula(part[1:-1]) if part.startswith('$') else pattern.sub(lambda m:formula(symbols[m[0]]),part) for part in parts)
    def add(text,k='body'): story.append(Paragraph(notation(text,k),styles[k]))
    def table(data,widths):
        t=Table([[Paragraph(notation(v,'small'),styles['small']) for v in r] for r in data],colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfeaf3')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f7')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([t,Spacer(1,6)])
    def picture(path,w=width):
        im=Image(str(path));im.drawHeight=w*im.imageHeight/im.imageWidth;im.drawWidth=w;story.append(im)
    add('Dual-rail readout','title')
    add('15 October 2026 | Viable capacitance pairs, selected circuits and robustness','small')
    add('1. Viable C1-C2 region: definitions and conditions','h')
    add(r'Two transmons, each specified by capacitance $C_i$ and single-junction energy $E_{Ji}$, couple to each other through $C_{12}$ and to a shared resonator through $C_{ir}$. The resonator mode has capacitance $C_r$ and inductance $L_r$. Junction parasitic capacitances are omitted. Loaded charging energies and interactions follow from the full inverse capacitance matrix.')
    add('Nominal selection thresholds used for the map','h')
    table([['Condition','Applied threshold'],
        ['Resonator','Bare frequency 6 GHz; equivalent-mode impedance 50 ohm'],
        ['Matching',r'$R=\max_{i\in\{-,+\}}\left|(\chi_+-\chi_-)/\chi_i\right|\leq 1\%$'],
        ['Dispersive condition',r'$\max_{ij}|g_{ij}/\Delta_{ij}|\leq 0.1$ over upward isolated-pair transitions from ground and the two logical states; $\Delta_{ij}$ is transition-resonator detuning'],
        ['Hybridization / state assignment',r'$H\geq95\%$; minimum six-branch reference-state overlap $\geq95\%$'],
        ['Logical / operation frequency','Logical frequency 90-110 MHz; operation frequency 4-8 GHz'],
        ['Transmon energies',r'Both loaded $150\leq E_C/h<400$ MHz; both $E_J/h<25$ GHz; both $E_J/E_C>50$'],
        ['Other design variables',r'Both junction energies, $C_{12}=0.005$-15 fF, and equal nominal $C_{1r}=C_{2r}=C_g=0.1$-25 fF can vary between map pixels']],[135,width-135])
    add('The earlier 90% capacitance-participation assumption only initializes the search range; it is not enforced as an equality. Every accepted point uses its actual capacitance matrix. The map covers uniform 1 fF centers from 36 to 128 fF, restricted to C1>C2. Mirror designs swap the transmon labels.','small')
    add('What the hybridization score means','h')
    add(r'For transmon $i$ and logical eigenstate $s\in\{-,+\}$, $S_{is}$ is its ground-to-logical charge-transition strength. Normalizing over the two logical transitions gives $p_{is}$. H measures the balance between the two branches: H=1 means equal strengths.')
    add(r'$S_{is}=|\langle G|n_i|s\rangle|^2,\qquad p_{is}=S_{is}/(S_{i-}+S_{i+}),\qquad H=1-2\,\mathrm{mean}_{i,s}|p_{is}-1/2|$')
    add(r'$|-\rangle=\sqrt{p}\,|10\rangle-\sqrt{1-p}\,|01\rangle,\qquad |+\rangle=\sqrt{1-p}\,|10\rangle+\sqrt{p}\,|01\rangle$')
    add(r'$H=1-|2p-1|$ in the ideal two-state model. Frequencies: $f_{\mathrm{op}}=(f_++f_-)/2$, $f_L=|f_+-f_-|$.')
    add(r'In the ideal single-excitation model, $|10\rangle$ and $|01\rangle$ denote one excitation in either loaded bare transmon, with no resonator excitation. H then directly measures their mixing: H=95% gives 52.5%/47.5% populations, H=90% gives 55%/45%, and H=100% gives 50%/50%.')
    picture(TMP/'mixing.png',440)
    add('The plot assumes ideal two-state mixing. In the full multilevel calculation, H measures normalized charge-transition balance, not state overlap or Bell-state fidelity; it alone does not bound resonator or higher-level participation. Separate dispersive and reference-overlap checks are retained. f_op is the mean dressed logical-transition frequency; f_L is the logical frequency.','small')
    story.append(PageBreak())
    add('1. Viable pairs and the three sampled circuits','title')
    picture(TMP/'viable_pairs_marked.png',405)
    add(f"{definitions['validated_pass_centers']:,} validated viable centers out of {definitions['requested_centers']:,}. Gray: searched without finding a witness ({definitions['no_witness_centers']:,}); orange: unresolved ({definitions['unresolved_centers']}); white: excluded C1<=C2. Colors show one feasible operation frequency per pair. No interpolation supplies passing pixels. Red stars identify the three robustness designs.",'small')
    data=[['ID','C1 / C2 (fF)','C12 (fF)','Cg (fF)','EJ1/h / EJ2/h (GHz)','EC1/h / EC2/h (MHz)','H (%)','f_op (GHz)']]
    for label,p in designs.items():
        d=p['device'];data.append([label,f"{d['C1_shunt_fF']:g} / {d['C2_shunt_fF']:g}",f"{d['C12_fF']:.3f}",f"{d['C1r_fF']:.3f}",
            f"{p['ej1_ghz']:.6f} / {p['ej2_ghz']:.6f}",f"{p['loaded_ec1_mhz']:.2f} / {p['loaded_ec2_mhz']:.2f}",
            f"{100*p['hybridization']:.2f}",f"{p['operating_frequency_ghz']:.6f}"])
    table(data,[24,64,42,42,110,95,50,88])
    add('Selections began with approximately equal viable-area C1 bands; user-directed redraws chose the final positions. These three circuits are illustrative, not an unbiased survey of fabrication yield. Parameters above are rounded; full-precision witnesses are saved in designs.json.','small')
    add('Why C12 and resonator/readout capacitances matter','h')
    add('The shaded region is an existence map: each C1-C2 pair has its own chosen C12, Cg and EJs. Here Cr=530.52 fF and Lr=1.32629 nH are fixed by the specified 6 GHz and 50 ohm mode. C1-C2 alone therefore does not specify a matched circuit.')
    add('All loaded ECs and charge couplings depend on the inverse capacitance matrix. C12 changes exchange, logical frequency and loading. Cr changes resonator frequency and zero-point fluctuations. Cg changes both readout couplings and loading; Cr is distinct from the coupling capacitors C1r and C2r. A component change at fixed C1/C2 can move the chosen point outside chi matching:')
    examples=[r for r in sensitivity if (r['design'],r['change']) in (('2','C12 +10%'),('2','Cr +10%'),('3','Cg +10%'))]
    table([['Fixed-point example','Nominal R (%)','R after change (%)']]+[
        [f"Design {r['design']}: {r['change']}",f"{r['nominal_mismatch_percent']:.3f}",f"{r['new_mismatch_percent']:.3f}"] for r in examples],
        [240,135,width-375])
    add('Examples keep all other components and EJs fixed, including Lr for the Cr change. A failed chosen point can sometimes be recovered by EJ recalibration. The map does not guarantee matching for arbitrary C12 or Cr. Bare resonator frequency and mode impedance together determine Cr and Lr.','small')
    story.append(PageBreak())
    add('2. Robustness with EJ recalibration','title')
    add('Each circuit receives the same 300 six-parameter error vectors: 224 Latin-hypercube interior samples, 64 extreme corners and 12 isolated +/-2% endpoints. C1, C2, C12, C1r, C2r and bare resonator frequency independently vary within +/-2%. Cr and Lr both scale inversely with bare frequency to keep the mode impedance at 50 ohm. Physical components remain at their perturbed values; only the EJs recalibrate.')
    add('Fixed versus flexible operation frequency','h')
    add('Fixed means both EJs may change while f_op remains at that design\'s nominal value in the table, within +/-100 Hz (a numerical solver tolerance, not an experimental stability requirement). Relative EJ tuning can alter mixing and logical frequency. Free operation frequency removes that equality and allows f_op anywhere in 4-8 GHz. The resonator remains at its sampled frequency during recalibration.')
    add('The current cumulative rescue order is: EC floor 150 ->120 MHz; free operation frequency; logical-frequency window 90-110 ->80-120 MHz while H remains >=95%; H floor 95 ->90%; EJ/EC floor >50 ->>40. All stages retain chi matching <=1%, dispersive ratio <=0.1, reference overlap >=95%, EC/h <400 MHz and EJ/h <25 GHz. These relaxations are robustness tests; the region on page 2 still uses the original selection thresholds.')
    table([['Design','Baseline','+ EC120','+ free operation frequency','+ logical frequency, H95','+ H90','+ EJ/EC40']]+[
        [label,*r['cumulative']] for label,r in analysis.items()],[55,*([float((width-55)/6)]*6)])
    add('Cumulative counts are out of 300, including stress tests. The figure below uses only the 224 interior samples per design.','small')
    picture(TMP/'comparison.png')
    add('CDF: fixed nominal operation frequency, EC floor 120 MHz and chi matching <=1%, with all other original screens except H retained. The legend gives eligible counts; each eligible interior sample contributes its best-found hybridization. The dashed line is H=95%. Excluded samples receive no invented score.','small')
    add(f"Validation: all 1,608 colored map witnesses and all {audit['accepted_checks']} accepted robustness solutions were checked in basis 20/9/8/40. The robustness audit includes {audit['extra_basis_samples']} extra 25/12/10/60 checks with no classification changes. Negative search results are not global impossibility proofs. Interior errors assume independent uniform marginals; combined counts are not fabrication yields.",'small')
    add('Scope: no minimum absolute chi, measurement speed, finite-drive, Purcell-filter or frequency-noise condition is imposed. A distributed CPW requires a consistent equivalent-mode conversion; its characteristic impedance is not automatically the mode impedance used here. Sources: refined_shunt_map_2026_10_14 and reordered_robustness_2026_10_15 data directories under readout_workflow/data/.','small')
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'overall_readout_findings_2026-10-15.pdf'
    def footer(canvas,doc):
        canvas.setFont('OV',7);canvas.setFillColor(colors.HexColor('#526372'))
        canvas.drawString(40,24,'Dual-rail readout | 15 October 2026')
        canvas.drawRightString(A4[0]-40,24,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=42,
        title='Dual-rail readout').build(story,onFirstPage=footer,onLaterPages=footer)
    import pymupdf
    with pymupdf.open(path) as doc:
        for i,page in enumerate(doc): page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(TMP/f'page_{i+1}.png')
        print(json.dumps(dict(pdf=str(path),pages=len(doc),map_points=definitions['validated_pass_centers'],robustness_samples=900),indent=2))


if __name__=='__main__': main()
