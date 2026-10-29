"""Frozen five-body physical OFF refinement and wide logical-residual report.

Only coupler EJ varies. Capacitors, loaded ECs and rail EJs remain fixed.
A plot interval outside the allowed OFF interval is diagnostic, never acceptance.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
from floating_coupler.channel_policy import off_target_definition
from floating_coupler.mixed_electrostatics import mixed_model
from floating_coupler.pairs import PairModel
from floating_coupler.paired_design import off_channel
from floating_coupler.off_search import exchange_roots


def sweep(model,off_range,plot_range,basis,points=81):
    """No geometry fitting or speed approximation; directly evaluate each bias."""
    if not 0<off_range[0]<off_range[1] or not 0<plot_range[0]<plot_range[1]:
        raise ValueError('Frequency intervals must be positive and increasing')
    if plot_range[0]>off_range[0] or plot_range[1]<off_range[1]:
        raise ValueError('Diagnostic plot must cover the allowed OFF interval')
    if points<3:raise ValueError('At least three sweep points are required')
    system=PairModel(model,**basis);cache={}
    def at(fc):
        fc=float(fc)
        if fc not in cache:
            detail=system.evaluate(fc);physical=off_channel(detail)
            logical=detail['logical_exchange'];h=np.asarray(logical['local_energy_hamiltonian_ghz'])
            cache[fc]=dict(fc_ghz=fc,coupler_ej_ghz=detail['coupler_ej_ghz'],physical_exchange=physical,
                          logical_exchange_hz=float(h[1,2]*1e9),logical_double_flip_hz=float(h[0,3]*1e9),
                          logical_detuning_mhz=float((h[1,1]-h[2,2])*1000),
                          logical_double_flip_detuning_mhz=float((h[0,0]-h[3,3])*1000),
                          logical_overlap=logical['minimum_subspace_overlap'])
        return cache[fc]
    grid=np.linspace(*off_range,points)
    roots=exchange_roots(lambda f:off_channel(system.physical_off(float(f)))['g_mhz'],grid,tolerance=1e-6)
    roots=[f for f in roots if off_range[0]<f<off_range[1]]
    root_rows=[at(f) for f in roots]
    for fc in np.linspace(*plot_range,points):at(fc)
    return dict(off_definition=off_target_definition('five'),basis=basis,off_range_ghz=list(off_range),
                plot_range_ghz=list(plot_range),roots=root_rows,accepted_off_found=bool(roots),
                design_certified=False,fixed_control_convergence_audit_required=True,
                rows=sorted(cache.values(),key=lambda row:row['fc_ghz']),
                scope='Frozen hardware; tune C EJ only. Extended plot range diagnostic only. Logical residuals are not the temporary OFF objective.')


def plot(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    rows=data['rows'];x=[r['fc_ghz'] for r in rows]
    axes[0].plot(x,[r['physical_exchange']['g_mhz']*1000 for r in rows],label='Physical 0100 ↔ 0010')
    for key,label in [('logical_exchange_hz','Logical 01 ↔ 10'),('logical_double_flip_hz','Logical 00 ↔ 11')]:
        axes[1].plot(x,[r[key] for r in rows],label=label)
    lo,hi=data['off_range_ghz'];xmin,xmax=data['plot_range_ghz']
    for ax in axes:
        ax.axhline(0,color='black',lw=.7);ax.axvspan(xmin,lo,color='gray',alpha=.15);ax.axvspan(hi,xmax,color='gray',alpha=.15)
        for root in data['roots']:ax.axvline(root['fc_ghz'],color='black',ls='--',lw=1)
        ax.set(xlabel='Bare coupler f01 (GHz)',xlim=(xmin,xmax));ax.grid(alpha=.2);ax.legend()
    axes[0].set(ylabel='Signed physical coupling (kHz)',title='Temporary OFF target')
    axes[1].set(ylabel='Signed logical coupling (Hz)',title='Residuals; gray intervals diagnostic only')
    axes[1].set_yscale('symlog',linthresh=1.)
    values=[r[key] for r in rows for key in ('logical_exchange_hz','logical_double_flip_hz')]
    if min(values)>=0:axes[1].set_ylim(0,max(1.,max(values)*1.2))
    fig.savefig(output/'off-and-logical-residuals.png',dpi=170)
    fig.savefig(output/'off-and-logical-residuals.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circuit',required=True);parser.add_argument('--off-range',nargs=2,type=float,required=True)
    parser.add_argument('--plot-range',nargs=2,type=float,required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--points',type=int,default=81)
    args=parser.parse_args();record=json.loads(Path(args.circuit).read_text());p=record.get('physical',record)
    basis=dict(local_levels=6,pair_states=15,coupler_levels=6,ncut=35)
    model,_=mixed_model(p['pad_pairs'],p['grounds_ff'],p['shunts_ff'],p['links'],p['idle_frequencies_ghz'],6,35)
    if 'ej_ghz' in record:
        model.ej=np.array([record['ej_ghz'][n] for n in model.names],dtype=float)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'sweep.json').exists():raise ValueError('Use a fresh output directory; do not overwrite a different policy/run')
    data=sweep(model,args.off_range,args.plot_range,basis,args.points);data['source']=args.circuit
    (out/'sweep.json').write_text(json.dumps(data,indent=2));plot(data,out)
    print(json.dumps(dict(accepted_off_found=data['accepted_off_found'],roots=data['roots']),indent=2))

if __name__=='__main__':main()
