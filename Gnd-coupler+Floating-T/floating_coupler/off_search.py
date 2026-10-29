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
