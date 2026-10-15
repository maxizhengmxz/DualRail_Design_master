"""Compare resonator baselines and check basis convergence at three EJ2 values."""
import numpy as np
from threadpoolctl import threadpool_limits
from model import Basis, Device, ReadoutModel


if __name__ == "__main__":
    d = Device()
    m = ReadoutModel(d)
    bare = 1/(2*np.pi*np.sqrt(d.Lr_nH*1e-9*d.Cr_fF*1e-15))/1e9
    print(f"Cr: {d.Cr_fF:.6f} fF; effective Cr: {m.cr_eff*1e15:.6f} fF")
    print(f"Bare fr: {bare:.9f} GHz; local loaded fr: {m.fr_ghz:.9f} GHz")
    big = ReadoutModel(d, Basis(ncut=20, transmon_levels=9, resonator_levels=8, eigenlevels=30))
    keys = ("bare_f1_ghz", "bare_f2_ghz", "dressed_r_ghz", "chi_m_ghz", "chi_p_ghz", "min_assignment_overlap")
    with threadpool_limits(limits=1):
        for ej2 in (10, 20, 30):
            p, q = m.evaluate(20, ej2), big.evaluate(20, ej2)
            print(f"EJ1=20, EJ2={ej2} GHz:", {k: round(p[k], 6) for k in keys})
            print("Larger-basis changes (MHz):", {
                k: round(1000*(q[k]-p[k]), 6)
                for k in ("dressed_r_ghz", "chi_m_ghz", "chi_p_ghz")
            })
