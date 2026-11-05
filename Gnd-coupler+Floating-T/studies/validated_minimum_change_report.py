"""Report completed quantum validation of frozen minimum-change candidates."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Flowable

WF=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(WF),str(WF.parent)]
from studies.minimum_change_report import append_protocol
from studies.parasitic_absorption_report import compensation_scale,draw_compensation_map,compensation_matrix
from studies.minimum_change_validation import secular


def render(campaign,candidates,circuit,output,temporary):
    campaign,candidates,circuit,output,temporary=map(Path,(campaign,candidates,circuit,output,temporary))
    manifest=json.loads((campaign/'manifest.json').read_text())
    audit=json.loads((candidates/'study.json').read_text())
    source=json.loads(Path(audit['source']).read_text())
    rows=[]
    for candidate in audit['cases']:
        name=f"g{candidate['group']}-{candidate['ports']}-{str(candidate['amount_ff']).replace('.','p')}ff"
        row=json.loads((campaign/'cases'/(name+'.json')).read_text())
        if row['identity']!=manifest['identity'] or row['status']!='complete':
            raise ValueError('Complete every candidate validation before final reporting')
        if row['fit']!=candidate['fit']:
            raise ValueError('Quantum results use a different capacitor allocation')
        if 'logical_basis_check' not in row:
            raise ValueError('Complete fixed-bias logical basis checks before reporting')
        rows.append(row)
    if len(rows)!=24:
        raise ValueError('Expected all 24 candidate validations')
    temporary.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles['Normal'].fontName='Helvetica'
    styles['Normal'].fontSize=9
    styles['Normal'].leading=12
    styles['Heading1'].fontSize=17
    styles['Heading1'].leading=20
    compact=styles['Normal'].clone('CompactValidation')
    compact.fontSize=7.5
    compact.leading=10
    width=A4[0]-84
    story=[]
    toc=[]
    page_number=1

    def text(value):
        story.extend([Paragraph(value,styles['Normal']),Spacer(1,8)])

    def title(value):
        story.extend([Paragraph(value,styles['Heading1']),Spacer(1,10)])

    def new_page(value):
        nonlocal page_number
        story.append(PageBreak())
        page_number+=1
        title(value)
        toc.append([1,value,page_number])

    def table(headers,data):
        cells=[[Paragraph(str(v),compact)for v in row]for row in [headers,*data]]
        item=Table(cells,colWidths=[width/len(headers)]*len(headers),repeatRows=1)
        item.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff5')),
                                  ('VALIGN',(0,0),(-1,-1),'TOP'),
                                  ('LINEBELOW',(0,0),(-1,-1),.25,colors.lightgrey),
                                  ('TOPPADDING',(0,0),(-1,-1),5),
                                  ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([item,Spacer(1,9)])

    def gate(row):
        return row['quantum']['phase_recalibrated_gate']

    def maximum(row):
        return max(abs(c['percent_change']) for c in row['fit']['changes'])

    baseline=manifest['baseline']['phase_recalibrated_gate']
    nominal_off=manifest['nominal_off_ghz']
    nominal_logical=manifest['nominal_off']['logical']
    nominal_secular=secular(nominal_logical)
    passed=sum(row['off_pass'] for row in rows)
    phases=sum(row['phase_pi_pass'] for row in rows)
    title('Parasitic compensation | validated results')
    toc.append([1,'Circuit and validation scope',1])
    text('Grounded coupler with floating rail transmons and both readout resonators. '
         'The report uses the selected 1-ppt minimum-change geometries for all '
         '24 cases, each containing one 0.1 or 0.2 fF parasitic. Every gate and '
         'OFF result below was recalculated for that same capacitor allocation.')
    drawing_box=[]

    class Circuit(Flowable):
        def __init__(self):
            super().__init__()
            self.width=width
            with pymupdf.open(circuit)as reference:
                self.height=width*reference[0].rect.height/reference[0].rect.width

        def draw(self):
            matrix=self.canv._currentMatrix
            drawing_box[:]=[matrix[4],A4[1]-matrix[5]-self.height,
                            matrix[4]+self.width,A4[1]-matrix[5]]

    story.extend([Circuit(),Spacer(1,8)])
    text('The circuit shows the nominal physical capacitor values. Green '
         'Cp1-Cp6/Cp11-Cp12 form group 1, and red Cp7-Cp10 form group 2. '
         'Actual compensations are shown later in percent and fF. The unrounded '
         'nominal input, fitted circuits and complete validation data are embedded.')
    table(['Input or comparison control','Value'],[
        ['Loaded rail EC/h','A1,A2,B1,B2 = 165,250,160,260 MHz'],
        ['Loaded coupler EC/h',f"{source['physical']['loaded_ec_ghz']['C']*1000:.3f} MHz"],
        ['Nominal audit-basis OFF',f'{nominal_off:.8f} GHz'],
        ['Static gate controls',f"Coupler {manifest['gate_point']['fc_ghz']:.2f} GHz, duration {manifest['gate_point']['time_ns']:.1f} ns. Common B frequency corrects conditional phase to pi."],
        ['Computed validation outcomes',f'{passed}/24 OFF points below 10 GHz meet the 1-Hz within-basis threshold. {phases}/24 gate calibrations meet phase pi.'],
    ])

    append_protocol(story,title,text,quantum_status=
        'All 24 quantum validations have been computed for these geometries. '
        'OFF and phase-pi acceptance counts are on page 1. Basis comparisons '
        'and unresolved numerical qualifications are reported explicitly on the final page.')
    page_number+=1
    toc.append([1,'Minimum-change compensation protocol',2])

    for amount in (.1,.2):
        new_page(f'Gate and OFF results | {amount:.1f} fF')
        text('Geometry and idle rail EJs stay fixed. Each OFF point tunes '
             'coupler EJ alone and targets physical 0100-0010 exchange. Static '
             'CZ uses consistently dressed ON logical references, fixed '
             f"coupler {manifest['gate_point']['fc_ghz']:.2f} GHz and "
             f"{manifest['gate_point']['time_ns']:.1f} ns, with common-B phase correction. "
             'Raw and postselected scores use the same phase-calibrated operating point.')
        data=[]
        for row in rows:
            if row['amount_ff']!=amount:
                continue
            metric=gate(row)
            off=row['off']['point']
            label=row['ports']+(' *' if not row['off_pass'] or not row['phase_pi_pass'] else '')
            data.append([label,f'{maximum(row):.1f}%',
                         f"{row['fit']['matched_relative_frobenius_K_error']*1e6:.1f}",
                         f"{off['frequency_ghz']:.8f}",
                         f"{100*metric['raw_cz_fidelity']:.3f}%",
                         f"{100*metric['postselected_cz_fidelity']:.4f}%"])
        table(['Parasitic','Max cap<br/>change','Matched K<br/>error (ppm)',
               'OFF f01<br/>(GHz)','Raw CZ','Postselected<br/>CZ'],data)
        text(f"Nominal raw / postselected scores are {100*baseline['raw_cz_fidelity']:.3f}% / "
             f"{100*baseline['postselected_cz_fidelity']:.4f}%. Postselected score equals "
             'raw fidelity divided by mean logical survival. Gate propagation '
             'is the coherent static-ON benchmark. Full scores, survival, '
             'per-input leakage, frequency trims and unrounded controls are embedded.')
        bad=[r['ports'] for r in rows if r['amount_ff']==amount and (not r['off_pass'] or not r['phase_pi_pass'])]
        if bad:
            text('<b>Acceptance flags.</b> '+', '.join(bad)+
                 '. An asterisk denotes a failed OFF or phase condition. The '
                 'embedded case record identifies the failed condition and retains '
                 'the best evaluated allowed bias.')
        text('All rates use the recorded numerical bases. Differences below '
             '0.01 percentage point in gate fidelity carry the agreed basis '
             'precision qualification. OFF frequencies are explicitly evaluated '
             'settings. The final page separates within-basis cancellation '
             'from cross-basis convergence.')

    new_page('Logical interactions at individual OFF points')
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for amount,color in ((.1,'#2166ac'),(.2,'#b2182b')):
        subset=[r for r in rows if r['amount_ff']==amount]
        values=([r['logical_at_off']['logical_channels'][0]['g_hz']for r in subset],
                [r['logical_at_off']['logical_channels'][1]['g_hz']for r in subset],
                [r['secular_at_off']['secular_zz_hz']for r in subset],
                [(r['off']['point']['frequency_ghz']-nominal_off)*1000 for r in subset])
        for axis,points in zip(axes.flat,values):
            axis.plot(range(12),points,'o-',color=color,label=f'{amount:.1f} fF')
    references=(nominal_logical['logical_channels'][0]['g_hz'],
                nominal_logical['logical_channels'][1]['g_hz'],nominal_secular['secular_zz_hz'],0.)
    labels=(r'$01_L\leftrightarrow10_L$ g (Hz)',r'$00_L\leftrightarrow11_L$ g (Hz)',
            'Secular logical ZZ (Hz)','OFF shift from nominal (MHz)')
    case_names=[r['ports']for r in rows if r['amount_ff']==.1]
    for axis,label,reference in zip(axes.flat,labels,references):
        axis.axhline(reference,color='black',ls='--',lw=.8,label='Nominal')
        axis.set_xticks(range(12),case_names,rotation=60,ha='right',fontsize=7)
        axis.set_ylabel(label)
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    figure=temporary/'validated-logical-interactions.png'
    fig.savefig(figure,dpi=180)
    plt.close(fig)
    story.append(Image(str(figure),width=width,height=width*.7))
    text('All logical interactions are evaluated at the individually retuned '
         'physical OFF point, with saved idle rail EJs. Logical exchanges use '
         'the canonical four-logical-state projection followed by local '
         'single-rail energy-axis alignment. Secular ZZ = '
         '(E11-E10-E01+E00)/h uses tracked dressed eigenenergies. Virtual '
         'auxiliary, coupler and resonator effects are retained.')
    text(f'OFF shift is fC,OFF(case) - {nominal_off:.8f} GHz. Positive '
         'means a higher required coupler frequency. The frequencies are '
         'verified by direct exchange evaluation. Logical exchange and ZZ '
         'remain residual diagnostics alongside the physical cancellation target.')

    for amount in (.1,.2):
        new_page(f'Idle residuals and detunings | {amount:.1f} fF')
        data=[]
        for row in rows:
            if row['amount_ff']!=amount:
                continue
            channels=row['logical_at_off']['logical_channels']
            data.append([row['ports'],f"{row['off_at_nominal']['A2_B1_hz']/1000:.3f}",
                         f"{channels[0]['g_hz']:.1f}",f"{abs(channels[0]['detuning_mhz']):.3f}",
                         f"{channels[1]['g_hz']:.1f}",f"{abs(channels[1]['detuning_mhz']):.3f}",
                         f"{row['secular_at_off']['secular_zz_hz']:.1f}"])
        table(['Parasitic','Physical g<br/>at nominal<br/>OFF (kHz)',
               '01L-10L<br/>g (Hz)','|detuning|<br/>(MHz)',
               '00L-11L<br/>g (Hz)','|detuning|<br/>(MHz)','Secular<br/>ZZ (Hz)'],data)
        text('The physical-exchange column keeps the common nominal audit-basis '
             'OFF frequency. The logical-exchange and ZZ columns use each '
             'candidate\'s individually retuned OFF frequency. Every idle rail '
             'EJ remains fixed. These two bias conventions are kept separate '
             'to show both the effect before retuning and the recovered result.')
        table(['Parasitic','Retuned physical<br/>g (Hz)','Mean logical<br/>survival (%)',
               'Worst gate nonlogical<br/>population (%)','B phase trim<br/>(MHz)'],[
            [row['ports'],f"{row['off']['point']['A2_B1_hz']:.3f}",
             f"{100*gate(row)['logical_survival']:.3f}",
             f"{100*gate(row)['worst_input_nonlogical_population']:.3f}",
             f"{1000*(gate(row)['fb_ghz']-baseline['fb_ghz']):+.3f}"]
            for row in rows if row['amount_ff']==amount])
        text('Survival, nonlogical population and the B trim refer to the '
             'static CZ operating point. Mean logical survival averages over '
             'the four logical inputs. The worst nonlogical population is the '
             'largest leakage among those inputs. The B trim is relative to '
             'the nominal phase-calibrated B setting.')

    new_page('Group 2 | physical idle admixture')
    nominal_states=manifest['nominal_participation']['states']
    group_two=[r for r in rows if r['group']==2]
    fig,axes=plt.subplots(2,2,figsize=(10,6.2),constrained_layout=True)
    for i,state in enumerate(('B0L','B1L')):
        for j,(occupation,owner)in enumerate((('1000','A1'),('0100','A2'))):
            axis=axes[i,j]
            for amount,color in ((.1,'#2166ac'),(.2,'#b2182b')):
                subset=[r for r in group_two if r['amount_ff']==amount]
                values=[r['off']['point']['state_participation']['states'][state]['physical_weights'][occupation]
                        -nominal_states[state]['physical_weights'][occupation] for r in subset]
                axis.plot(range(4),values,'o-',color=color,label=f'{amount:.1f} fF')
            axis.set_xticks(range(4),[r['ports']for r in group_two if r['amount_ff']==.1],fontsize=8)
            axis.set_title(f'{state}: {owner} component')
            axis.set_ylabel('Change in absolute probability')
            axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            axis.axhline(0,color='black',ls='--',lw=.8)
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
    figure=temporary/'validated-idle-participation.png'
    fig.savefig(figure,dpi=180)
    plt.close(fig)
    story.append(Image(str(figure),width=width,height=width*.62))
    text('For dressed B0L/B1L, rail A is tracked from its pair-ground reference. '
         'A1 weight = |&lt;1000;0C;0r1;0r2|psi&gt;|<super>2</super> and A2 '
         'weight uses 0100, in rail order A1,A2,B1,B2. These are components '
         'of normalized full-system eigenvectors at the individual OFF bias. '
         'The plot shows compensated minus nominal weight. Negative changes '
         'mean reduced A-rail participation. Four-state renormalization is absent.')
    table(['Nominal state','A1 weight','A2 weight','A1+A2 weight'],[
        [state,f"{nominal_states[state]['physical_weights']['1000']:.3e}",
         f"{nominal_states[state]['physical_weights']['0100']:.3e}",
         f"{nominal_states[state]['A_weight']:.3e}"]for state in ('B0L','B1L')])
    text('All four rail-state components for both rails and all 24 cases '
         'are embedded. Participation changes retain the numerical-basis '
         'qualification of the underlying state calculation.')

    for group in (1,2):
        studies=[]
        for amount in (.1,.2):
            cases={}
            for row in rows:
                if row['group']!=group or row['amount_ff']!=amount:
                    continue
                fit=deepcopy(row['fit'])
                categories={'ground':0,'self':1,'resonator':2,'link':3}
                fit['changes'].sort(key=lambda c:categories[c['component'].split(':',1)[0]])
                cases[row['ports']]=dict(fit=fit)
            studies.append(dict(amount_ff=amount,cases=cases))
        percentage_scale=compensation_scale(studies,'percent_change')
        magnitudes=np.concatenate([abs(compensation_matrix(s,'delta_ff')[2]).ravel()for s in studies])
        thresholds=np.unique(np.r_[np.round(np.percentile(magnitudes,[25,50,75,90,95]),3),
                                   np.ceil(max(magnitudes)*1000)/1000])
        thresholds=thresholds[thresholds>=.001]
        boundaries=np.r_[-thresholds[::-1],-.0005,.0005,thresholds]
        palette=[matplotlib.colors.to_hex(plt.get_cmap('RdBu_r')(v))for v in np.linspace(.05,.95,len(boundaries)-1)]
        ff_scale=dict(boundaries=boundaries,ticks=np.r_[-thresholds[::-1],0,thresholds],
                      colors=palette,extend='neither',spacing='uniform')
        for study in studies:
            for field,unit,scale in (('percent_change','%',percentage_scale),('delta_ff','fF',ff_scale)):
                new_page(f"Group {group} compensation | {study['amount_ff']:.1f} fF | {unit}")
                figure=temporary/f"validated-g{group}-{study['amount_ff']}-{field}.png"
                draw_compensation_map(study,field,figure,scale)
                story.append(Image(str(figure),width=width,height=width*10.4/9.5))
                text('Columns are independent single-parasitic cases. Rows '
                     'are the compensating physical capacitors of the same '
                     'geometries used for the quantum results. Percentages '
                     'use one decimal and fF values use three. Percentage '
                     'bands are 0, +/-1, +/-3, +/-5, +/-8 and +/-10%, with '
                     'end colors covering larger changes. The two amplitudes '
                     'share each unit\'s color scale within the same group.')

    new_page('Numerical accuracy and execution time')
    fields=('raw_cz_fidelity','postselected_cz_fidelity','logical_survival')
    worst={field:max(abs(row['calibrated_gate_basis_check']['metrics'][field]-gate(row)[field])for row in rows)
           for field in fields}
    physical_change=max(abs(row['off_production_check']['A2_B1_hz']-row['off']['point']['A2_B1_hz'])for row in rows)
    logical_change=max(abs(v)for row in rows for v in row['logical_basis_check']['exchange_changes_hz'])
    zz_change=max(abs(row['logical_basis_check']['secular_change_hz'])for row in rows)
    table(['Check','Worst measured change or result'],[
        ['Raw gate fidelity basis change',f"{worst['raw_cz_fidelity']:.3e} probability"],
        ['Selected gate score basis change',f"{worst['postselected_cz_fidelity']:.3e} probability"],
        ['Mean survival basis change',f"{worst['logical_survival']:.3e} probability"],
        ['Physical OFF coupling at identical bias',f'{physical_change:.2f} Hz between production and audit'],
        ['Logical exchange at identical bias',f'{logical_change:.2f} Hz between production and audit'],
        ['Secular ZZ at identical bias',f'{zz_change:.2f} Hz between production and audit'],
    ])
    text('Production basis is local6/pair12/C5 with charge cutoff ncut35. '
         'Audit basis is local7/pair15/C6 with ncut40. Each resonator retains '
         'states 0,1,2. All basis comparisons hold the reported controls '
         'fixed. The gate probability target is 1e-4, or 0.01 percentage point. '
         'Within-basis OFF acceptance is |physical g| below 1 Hz. A finite '
         'production/audit difference above 1 Hz leaves cross-basis residual '
         'certification unresolved. Values remain reported with that qualification.')
    failures=[row['ports']+f" ({row['amount_ff']:.1f} fF)"for row in rows
              if any(abs(row['calibrated_gate_basis_check']['metrics'][field]-gate(row)[field])>1e-4 for field in fields)]
    if failures:
        text('<b>Gate precision flags.</b> '+', '.join(failures)+'.')
    stages={key:sum(row['timings'].get(key,0)for row in rows)for key in rows[0]['timings']}
    table(['Execution measure','Minutes'],[
        ['Campaign elapsed wall time',f"{manifest.get('wall_seconds',0)/60:.2f}"],
        ['Workers',str(manifest['workers'])],
        *[[name.replace('_seconds','').replace('_',' '),f'{seconds/60:.2f}']for name,seconds in stages.items()],
        ['Fixed-bias logical postchecks, summed',f"{sum(row['logical_basis_check']['seconds']for row in rows)/60:.2f}"],
    ])
    text('Stage times sum worker execution time. Parallel worker durations '
         'overlap and exceed elapsed wall time. The frozen candidate geometry '
         'and case identity are checked before reporting. Nominal inputs, '
         'minimum-change fits, all quantum case records, basis checks and '
         'unrounded controls are embedded for reproduction. Local convergence '
         'of the capacitor fit carries the qualification stated on page 2.')

    def footer(canvas,doc):
        canvas.setFont('Helvetica',7)
        canvas.drawString(42,23,'Minimum-change parasitic validation | 2026-11-05')
        canvas.drawRightString(A4[0]-42,23,str(doc.page))

    staged=temporary/'validated-content.pdf'
    SimpleDocTemplate(str(staged),pagesize=A4,leftMargin=42,rightMargin=42,
                      topMargin=38,bottomMargin=38).build(story,onFirstPage=footer,onLaterPages=footer)
    with pymupdf.open(staged)as document,pymupdf.open(circuit)as drawing:
        if len(document)!=page_number:
            raise ValueError(f'Report overflow changed pagination: {len(document)} versus {page_number}')
        document[0].show_pdf_page(pymupdf.Rect(drawing_box),drawing,0)
        document.set_toc(toc)
        document.xref_set_key(document.pdf_catalog(),'OpenAction',f'[{document[0].xref} 0 R /Fit]')
        document.xref_set_key(document.pdf_catalog(),'PageMode','/UseOutlines')
        document.embfile_add('nominal-compensation-input.json',Path(audit['source']).read_bytes())
        document.embfile_add('minimum-change-fits.json',(candidates/'study.json').read_bytes())
        document.embfile_add('quantum-validation.json',json.dumps(dict(manifest=manifest,cases=rows),indent=2).encode())
        document.embfile_add('bounded_parasitic_compensation.py',(WF/'studies/bounded_parasitic_compensation.py').read_bytes())
        output.parent.mkdir(parents=True,exist_ok=True)
        new=output.with_suffix('.new.pdf')
        document.save(new,garbage=4,deflate=True)
    new.replace(output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--circuit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--temporary',type=Path,required=True)
    args=parser.parse_args()
    print(render(args.campaign,args.candidates,args.circuit,args.output,args.temporary))
