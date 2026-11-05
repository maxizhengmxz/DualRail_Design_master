"""Fixed-bias production/audit logical residual comparison for completed cases."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

WF=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(WF),str(WF.parent)]


def check(path):
    import numpy as np
    from scipy.linalg import eigh
    from threadpoolctl import threadpool_limits
    from studies.parasitic_absorption import PRODUCTION,rebuild,write
    from studies.minimum_change_validation import secular
    from integrated_readout_range import frozen_model
    from seven_readout import SevenReadout
    from floating_coupler.channel_policy import logical_exchange_from_eigenpairs
    result=json.loads(path.read_text())
    if result['status']!='complete':
        return dict(case=path.name,status='incomplete')
    if 'logical_basis_check' in result:
        return dict(case=path.name,status='cached')
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        physical,_=rebuild(result['fit']['physical'])
        frequency=result['off']['point']['frequency_ghz']
        system=SevenReadout(frozen_model(physical),physical,frequency,basis=PRODUCTION,off_only=True)
        sector=np.flatnonzero(system.parity==1)
        energy,vectors=eigh(system.ham[sector,:][:,sector].toarray(),
                            subset_by_index=(0,system.eigenlevels-1),check_finite=False)
        positions=np.searchsorted(sector,[system.refs[(label,0,0)] for label in ('00','01','10','11')])
        exchange=logical_exchange_from_eigenpairs(energy,vectors[positions,:])
        matrix=np.asarray(exchange['local_energy_hamiltonian_ghz'])
        channels=[dict(g_hz=float(matrix[i,j]*1e9)) for i,j in ((1,2),(0,3))]
        logical=dict(logical_exchange=exchange,logical_channels=channels)
        coarse_secular=secular(logical)
    differences=[float(a['g_hz']-b['g_hz']) for a,b in
                 zip(channels,result['logical_at_off']['logical_channels'])]
    result['logical_basis_check']=dict(basis=PRODUCTION,at_fixed_off_ghz=frequency,
        logical_channels=channels,secular=coarse_secular,
        exchange_changes_hz=differences,
        secular_change_hz=float(coarse_secular['secular_zz_hz']-result['secular_at_off']['secular_zz_hz']),
        exchange_agrees_to_1hz=max(abs(v) for v in differences)<=1,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seconds=time.perf_counter()-start)
    write(path,result)
    return dict(case=path.name,status='checked',changes_hz=differences)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(check,path) for path in (args.root/'cases').glob('*.json')]):
            print(json.dumps(future.result()),flush=True)
