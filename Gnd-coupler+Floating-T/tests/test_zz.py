"""Consolidated zz regression suite; original coverage preserved."""


# ---- Original coverage: test_physical_zz ----
import unittest
import numpy as np
from core.transmon import Basis
from floating_coupler.model import Model
from floating_coupler.design import ej_for_frequency
from floating_coupler.zz_diagnostics import physical_zz

class PhysicalZZTests(unittest.TestCase):
    def test_decoupled_B_has_zero_conditional_shift_despite_A_coupler_dressing(self):
        ec=np.array([.25,.44,.16]);ej=np.array([ej_for_frequency(f,e,35) for f,e in zip([4.61,10.7,5.39],ec)])
        j=np.zeros((3,3));j[0,1]=j[1,0]=.12
        model=Model(('A2','C','B1'),ec,ej,j,Basis(ncut=35,transmon_levels=6),{})
        row=physical_zz(model,10.7)
        self.assertLess(abs(row['physical_zz_hz']),1e-3)
        self.assertGreater(row['minimum_assignment_weight'],.98)


# ---- Original coverage: test_physical_zz_derivatives ----
import unittest
import numpy as np
from core.transmon import Basis
from floating_coupler.model import Model
from floating_coupler.design import ej_for_frequency
from floating_coupler.zz_diagnostics import physical_zz
from floating_coupler.zz_diagnostics import zz_derivatives

class PhysicalZZDerivativeTests(unittest.TestCase):
    def test_eigenvector_derivative_matches_independent_energy_differences(self):
        ec=np.array([.25,.44,.16]);ej=np.array([ej_for_frequency(f,e,35) for f,e in zip([4.61,10.6,5.39],ec)])
        j=np.array([[0.,-.2,-.013],[-.2,0.,.16],[-.013,.16,0.]])
        model=Model(('A2','C','B1'),ec,ej,j,Basis(ncut=35,transmon_levels=5),{})
        f=10.6;h=.002;z=lambda x:physical_zz(model,x)['physical_zz_hz']
        independent=(z(f-2*h)-8*z(f-h)+8*z(f+h)-z(f+2*h))/(12*h)
        derivative=zz_derivatives(model,f)['dzz_df_hz_per_ghz']
        self.assertAlmostEqual(derivative,independent,delta=max(.02,abs(independent)*1e-5))


if __name__=='__main__':unittest.main()
