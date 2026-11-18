"""Tied-p=0.10 EC/coupling targets and annotated modal capacitance matrices.

Report generation only. Reconstruct K from saved tied-allocation capacitors, verify
saved reference targets, and distinguish K matching constraints from residuals.
"""
import argparse,json,sys,subprocess
from pathlib import Path
import numpy as np
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT.parent),str(ROOT)]
from studies.capacitance_model import circuit,coefficients
from studies.capacitance_model import INTENDED
from floating_coupler.model import SCALE
from core.transmon import Basis,transmon_spectrum

def build(output):
    nominal=json.loads((ROOT/'configs/Design_V1_20261105.json').read_text())
    study=json.loads((ROOT/'output/common-rail-allocation-20261118/result.json').read_text())
    ph=study['solutions']['0.1']['physical'];value=circuit(ph);names=value['modal_order'];k=value['K'];cm=np.linalg.inv(k)
    assert np.allclose(k,np.asarray(ph['K_per_ff']),rtol=1e-10,atol=1e-15)
    idle=nominal['operating_points']['idle_off']
    n01={n:abs(transmon_spectrum(idle['ej_ghz'][n],ph['loaded_ec_ghz'][n],Basis(ncut=35,transmon_levels=5))[1][0,1]) for n in names[:5]}
    ref=coefficients(ph,k,n01)
    for a,b in INTENDED:
        key=f'g01_{a}_{b}_mhz'
        assert abs(ref[key]-study['reference'][key])<1e-6
    for i,n in enumerate(names[:5]):assert abs(SCALE*k[i,i]-ph['loaded_ec_ghz'][n])<1e-12
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    classes=[['D' if i==j else 'I' if (a,b) in INTENDED or (b,a) in INTENDED else 'R'
              for j,b in enumerate(names)] for i,a in enumerate(names)]
    assets=out/'assets';assets.mkdir(exist_ok=True)
    diagram=dict(show_transmon_parameters=False,
        physical=dict(pad_pairs=ph['pad_pairs'],grounds_ff=ph['ground_capacitances_ff'],shunts_ff=ph['pad_mutual_capacitances_ff'],links=ph['physical_links']),
        rail_labels={n:{} for n in ('A1','A2','B1','B2')},coupler_labels={},
        readouts={rail:dict(target_frequency_ghz=ref[f'f_{r}_ghz'],links=[e for e in ph['physical_links'] if r in (e['from'],e['to'])]) for rail,r in [('A','r1'),('B','r2')]})
    (assets/'circuit-input.json').write_text(json.dumps(diagram,indent=2))
    template=(ROOT/'templates/circuit-schematic.py.in').read_text().replace('{{PARAMETERS_JSON}}',str((assets/'circuit-input.json').resolve())).replace('{{OUTPUT_DIRECTORY}}',str(assets.resolve()))
    template=template.replace('Readout branches are contextual; decoherence calculations exclude resonator dynamics.','Full seven-mode capacitive network; all parasitic capacitors are zero.')
    (assets/'draw-circuit.py').write_text(template)
    subprocess.run([sys.executable,str(assets/'draw-circuit.py')],check=True)
    (out/'target-matrices.json').write_text(json.dumps(dict(reference='Tied p=0.10; matched to Design_V1_20261105',
        modal_order=names,K_per_ff=k.tolist(),Cmodal_ff=cm.tolist(),entry_classes=classes,
        class_legend=dict(D='Matched K diagonal',I='Matched intended K off-diagonal',R='Unmatched residual K off-diagonal'),
        note='Cmodal entries are not independently fixed: the classification is a constraint mask on K only.'),indent=2))
    for mat,label in [(k,'inverse-capacitance-per-fF'),(cm,'modal-capacitance-fF')]:
        with (out/(label+'.csv')).open('w') as f:
            f.write(','+','.join(names)+'\n')
            for n,row in zip(names,mat):f.write(n+','+','.join(f'{v:.12g}' for v in row)+'\n')
    for n,fn in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf')]:pdfmetrics.registerFont(TTFont(n,'C:/Windows/Fonts/'+fn))
    st=getSampleStyleSheet();st.add(ParagraphStyle(name='BodyA',fontName='Arial',fontSize=9.5,leading=13,spaceAfter=8))
    st.add(ParagraphStyle(name='SmallA',fontName='Arial',fontSize=8,leading=11,spaceAfter=6))
    st.add(ParagraphStyle(name='TitleA',fontName='Arial-Bold',fontSize=20,leading=24,spaceAfter=13,textColor=colors.HexColor('#173b55')))
    st.add(ParagraphStyle(name='HeadA',fontName='Arial-Bold',fontSize=13,leading=17,spaceAfter=10,textColor=colors.HexColor('#173b55')))
    story=[]
    def p(t,small=False):story.append(Paragraph(t,st['SmallA' if small else 'BodyA']))
    def heading(t):story.append(Paragraph(t,st['HeadA']))
    def tab(rows,widths):
        t=Table([[Paragraph(str(v),st['SmallA']) for v in row] for row in rows],colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([t,Spacer(1,10)])
    def matrix(mat,annotate):
        rows=[['']+names]
        for i,n in enumerate(names):
            vals=[]
            for j,v in enumerate(mat[i]):
                text=f'{v:.3f}' if abs(v)>=.01 else f'{v:.2e}'
                vals.append(text+'\n'+classes[i][j] if annotate else text)
            rows.append([n]+vals)
        t=Table(rows,colWidths=[35]+[65]*7,rowHeights=[28]+[44]*7)
        commands=[('FONTNAME',(0,0),(-1,-1),'Arial'),('FONTSIZE',(0,0),(-1,-1),8.5),('LEADING',(0,0),(-1,-1),13),('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8edf1')),('BACKGROUND',(0,1),(0,-1),colors.HexColor('#e8edf1')),('GRID',(0,0),(-1,-1),.4,colors.white)]
        palette={'D':'#dbe9fa','I':'#d9efe5','R':'#fce6cf'}
        if annotate:
            for i in range(7):
                for j in range(7):commands.append(('BACKGROUND',(j+1,i+1),(j+1,i+1),colors.HexColor(palette[classes[i][j]])))
        else:commands.append(('ROWBACKGROUNDS',(1,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')]))
        t.setStyle(TableStyle(commands));story.extend([t,Spacer(1,13)])
    story.append(Paragraph('Tied 10% self-capacitance circuit',st['TitleA']))
    p('All four rail transmons: p = 0.10 | grounded C2 | two shared readout resonators',True)
    from PIL import Image as PILImage
    with PILImage.open(assets/'transmon-circuit.png') as im:iw,ih=im.size
    story.append(Image(str(assets/'transmon-circuit.png'),width=500,height=500*ih/iw));story.append(Spacer(1,10))
    p('The diagram and all matrices in this report use the <b>tied p=0.10 synthesized physical capacitors</b>, not the original nominal capacitors. p = Cs / (Cs + Cg,each/2); grounds are equal on each rail transmon. C2 is grounded and both coupler links attach to C1. No parasitics are drawn or included.')
    rows=[['Mode','Loaded EC/h (MHz)','Self Cs (fF)','Ground per pad (fF)','Self p']]
    for n in names[:5]:
        ground=ph['ground_capacitances_ff'][ph['pad_pairs'][n][0]]
        rows.append([n,f"{ph['loaded_ec_ghz'][n]*1000:.3f}",f"{ph['pad_mutual_capacitances_ff'][n]:.3f}",'C2 grounded' if n=='C' else f'{ground:.3f}','--' if n=='C' else '0.10'])
    tab(rows,[55,120,100,140,75])
    p('Physical loading: coupler self contribution 55.36%; full coupler-branch loading on A2/B1 is 2.38%/2.14%. These fractions use the study\'s complete three-link branch-removal definition. The all-rail ratio condition does not apply to the grounded coupler.',True)
    p('Five-body coupler-only OFF recalibration gives 9.907940 GHz in the audit basis, with residual logical exchanges about 865.8 Hz (01L-10L) and 853.7 Hz (00L-11L). Larger-basis drift is a few Hz. This is not a seven-mode OFF certification or a recalibrated gate result.',True)
    story.append(PageBreak())
    story.append(Paragraph('EC and intended coupling targets',st['TitleA']))
    p('Tied p=0.10 design | coefficients matched to Design_V1_20261105 | 18 November 2026',True)
    p('The following ECs and nine intended coefficients are preserved by synthesis; residual entries are specific to the tied p=0.10 circuit. All ECs are loaded values after including the complete capacitive network.')
    heading('1. Loaded ECs and reference controls')
    rows=[['Mode','EC/h (MHz)','Bare f01 (GHz)','EJ/h (GHz)']]
    for n in names[:5]:rows.append([n,f"{ph['loaded_ec_ghz'][n]*1000:.3f}",f"{idle['bare_f01_ghz'][n]:.6f}",f"{idle['ej_ghz'][n]:.6f}"])
    tab(rows,[65,125,150,150])
    p(f"Resonators: r1 = {ref['f_r1_ghz']:.6f} GHz; r2 = {ref['f_r2_ghz']:.6f} GHz. These are the local modal frequencies set by K diagonals and fixed inductances, before coupled-system diagonalization.",True)
    heading('2. Nine intended interactions to preserve')
    rows=[['Channel','Charge J (MHz)','Local exchange g (MHz)']]
    for a,b in INTENDED:
        key=f'{a}_{b}';j=ref.get('J_'+key+'_mhz')
        rows.append([a+' - '+b,'--' if j is None else f'{j:+.3f}',f"{ref['g01_'+key+'_mhz']:+.3f}"])
    tab(rows,[125,170,195])
    p('For transmon pairs, H/h contains J n(i)n(j), with dimensionless Cooper-pair number n. Local g = J |n01(i)| |n01(j)| in the saved phase convention. Readout g includes the resonator zero-point charge. Frequencies are in cycles per second, not angular frequency; no extra 2 pi is implied.',True)
    p('All exchange values above use the listed inherited nominal idle/OFF reference controls. The 9.896743 GHz coupler setting is a comparison reference, not the recalibrated OFF of this candidate. Matching charge coefficients preserves local g at fixed EC/EJ; tuning EJ changes the matrix elements and g. Direct A2-B1 remains nonzero and is canceled by virtual coupler paths at effective OFF.',True)
    story.append(PageBreak())
    heading('3. Matrix that is actually matched: K = Cmodal inverse')
    p('Order: A1, A2, C, B1, B2, r1, r2. Displayed numerical matrix is <b>1000 K</b>, where K is in fF<super>-1</super>. Thus an entry shown as 8.518 represents 0.008518 fF<super>-1</super>. Scientific notation retains small residual entries.')
    matrix(1000*k,True)
    tab([['Mark / color','Meaning','Independent entries'],['D / blue','Matched diagonal: five ECs + two resonator scales','7'],['I / green','Matched intended interaction coefficient','9'],['R / orange','Residual interaction; not an equality target','12']],[95,325,70])
    p('The matrix is symmetric: mirrored entries are the same constraint, not two independent constraints. The 12 residual pairs are A1-C, A1-B1, A1-B2, A2-B2, C-B2, C-r1, C-r2, A1-r2, A2-r2, B1-r1, B2-r1 and r1-r2.')
    p('For transmons, EC(i)/h = e<super>2</super>K(i,i)/(2h), and J(i,j) = 4e<super>2</super>K(i,j)/h, after unit conversion. For transmon-resonator links, the fixed resonator diagonal and inductance fix its zero-point charge; matching K(i,r) therefore preserves its charge coupling.')
    p('Residual colors identify local Hamiltonian interactions, not effective logical exchange rates. Their dynamical consequences depend on detuning, virtual paths and state hybridization; they require the separate OFF/gate validation described in the allocation report.',True)
    story.append(PageBreak())
    heading('4. Equivalent reduced capacitance matrix Cmodal (fF)')
    p('This is the tied p=0.10 seven-mode equivalent capacitance matrix, obtained as Cmodal = K<super>-1</super>. It describes differential transmon modes and grounded resonator modes after eliminating floating common coordinates. It is not the list of physical port-to-port capacitors.')
    matrix(cm,False)
    p('<b>Important:</b> individual entries of Cmodal are not the hard matching targets. The blue/green/orange classification on the previous page applies to its inverse K, where charging energies and charge interactions are defined. Inversion mixes all entries: Cmodal can change while all intended K entries remain fixed. A Cmodal off-diagonal cannot generally be identified with one physical coupling capacitor.')
    heading('Reproducible reduction and compensation')
    p('Build the physical nodal Maxwell matrix Cnode using ground, across-pad and link capacitors; C2 is ground and omitted. Let B have +1/-1 entries on each floating transmon\'s two pads, +1 on grounded coupler C1, and +1 on each resonator node. The retained charge projection gives K = B<super>T</super> Cnode<super>-1</super> B; then invert K to obtain Cmodal. All matrices use the same listed modal order and charge orientation.')
    p('During redistribution, reconstruct these matrices from every candidate\'s physical capacitors. Enforce the 7 D and 9 I entries of K, retain positive capacitances and loading bounds, and minimize the absolute RMS local exchange over the 12 R entries. Among comparable residual objectives, minimize fractional capacitor changes. Preserving every entry of Cmodal would additionally preserve all residuals and is a different, stronger constraint.')
    p('Provenance: output/common-rail-allocation-20261118/result.json, solutions[0.1].physical supplies all actual capacitors; configs/Design_V1_20261105.json supplies inherited EJ reference controls only. The mixed electrostatics backend rebuilds K independently; intended coefficients are verified against the nominal targets. Five-body OFF results come from output/common-allocation-off-20261118/result.json. Full-precision matrices and classification mask are saved beside this PDF.',True)
    def footer(c,doc):
        c.setFont('Arial',8);c.setFillColor(colors.HexColor('#637484'));c.drawString(43,24,'Tied p=0.10 circuit | EC, intended interactions and residual matrix entries');c.drawRightString(A4[0]-43,24,str(doc.page))
    pdf=out/'tied-p010-ec-coupling-matrix-targets.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=38,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    print(pdf)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args();build(args.output)
