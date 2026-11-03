"""Signed and tangential zeros for effective exchange; no extrapolated roots."""
import numpy as np
from scipy.optimize import brentq,minimize_scalar


def exchange_roots(function,grid,tolerance=1e-6):
    grid=np.asarray(grid,dtype=float)
    if len(grid)<3 or np.any(np.diff(grid)<=0):raise ValueError('Supply at least three increasing scan points')
    values=np.array([function(float(x)) for x in grid]);roots=[]
    def add(x):
        if np.isfinite(function(x)) and abs(function(x))<=tolerance and not any(abs(x-r)<1e-6 for r in roots):roots.append(float(x))
    for x,g in zip(grid,values):
        if np.isfinite(g) and abs(g)<=tolerance:add(float(x))
    for a,b,ga,gb in zip(grid[:-1],grid[1:],values[:-1],values[1:]):
        if np.isfinite(ga+gb) and ga*gb<0:add(brentq(function,a,b,xtol=1e-11))
    intervals=[(grid[0],grid[1]),(grid[-2],grid[-1])]
    for i in range(1,len(grid)-1):
        if np.isfinite(values[i-1:i+2]).all() and abs(values[i])<=min(abs(values[i-1]),abs(values[i+1])):
            intervals.append((grid[i-1],grid[i+1]))
    for a,b in intervals:
        result=minimize_scalar(lambda x:abs(function(float(x))),bounds=(a,b),method='bounded',options={'xatol':1e-11})
        add(float(result.x))
    return sorted(roots)

def threshold_off_search(function,grid,nominal,threshold_hz=1.):
    """Accept |physical g| below a magnitude threshold; an exact zero is optional.

    function is cyclic MHz. This tests magnitude in the supplied basis, not
    cross-basis numerical convergence. Grid samples and interior minima count.
    """
    if threshold_hz<=0:raise ValueError('Require a positive residual threshold')
    seen={}
    def at(f):
        f=float(f)
        if f not in seen:seen[f]=float(function(f))
        return seen[f]
    grid=np.asarray(grid,dtype=float)
    for f in grid:at(f)
    candidates=[f for f,g in seen.items() if np.isfinite(g) and abs(g)*1e6<threshold_hz]
    if not candidates:
        exchange_roots(at,grid,tolerance=threshold_hz/1e6)
        candidates=[f for f,g in seen.items() if np.isfinite(g) and abs(g)*1e6<threshold_hz]
    finite=[f for f,g in seen.items() if np.isfinite(g)]
    best=min(candidates,key=lambda f:abs(f-nominal)) if candidates else min(finite,key=lambda f:abs(seen[f])) if finite else None
    return dict(passed=bool(candidates),fc_ghz=best,residual_hz=seen[best]*1e6 if best is not None else None,
                threshold_hz=threshold_hz,scan=[[float(f),at(f) if np.isfinite(at(f)) else None] for f in grid])
