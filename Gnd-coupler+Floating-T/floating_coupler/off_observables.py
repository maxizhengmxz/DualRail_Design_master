"""Physical OFF reference construction shared by five- and seven-mode models.

TEMPORARY-OFF: these physical single-excitation references own OFF optimization.
The encoded two-excitation logical exchanges remain independent diagnostics.
See ../OFF_POLICY.md for the open question and exact restoration map.
"""
import numpy as np


def physical_single_excitation_references(pair):
    """Rows are bare rail occupations 1000,0100,0010,0001 in the pair basis.

    Transform the complete physical reference vectors into the retained isolated
    pair eigenbasis. Do not substitute individual hybridized rail eigenmodes.
    """
    levels=pair.basis['local_levels'];p,c,_=pair.dims
    vacuum=np.eye(c)[0]
    rows=[]
    for a,b in [((1,0),(0,0)),((0,1),(0,0)),((0,0),(1,0)),((0,0),(0,1))]:
        av=pair.a['vectors'][np.ravel_multi_index(a,(levels,levels)),:p]
        bv=pair.b['vectors'][np.ravel_multi_index(b,(levels,levels)),:p]
        rows.append(np.kron(np.kron(av,vacuum),bv))
    return np.asarray(rows)
