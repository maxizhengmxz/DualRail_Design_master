"""Physical invariance and basis-aware sign-certificate regression tests."""
import unittest
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
from threadpoolctl import threadpool_limits
from readout_workflow import calculation as calc, model as module
from readout_workflow.validation import FAST, AUDIT

class ValidationRepresentationTests(unittest.TestCase):
    def test_real_and_original_complex_phase_gauges_agree(self):
        basis=calc.Basis(12,5,4,20)
        m=calc.ReadoutModel(calc.Device(),basis)
        eigh=module.eigh
        def complex_gauge(a,*args,**kwargs):
            if 'subset_by_index' not in kwargs:return eigh(a,*args,**kwargs)
            phase=np.tile(1j**np.arange(basis.resonator_levels),basis.transmon_levels**2)
            w,v=eigh(phase[:,None]*a*phase.conj()[None,:],*args,**kwargs)
            return w,phase.conj()[:,None]*v
        with threadpool_limits(limits=1):
            real=m.evaluate(14,12)
            with patch.object(module,'eigh',complex_gauge):old=m.evaluate(14,12)
        for k in ('chi_m_ghz','chi_p_ghz','splitting_ghz','hybridization','min_assignment_overlap'):
            self.assertAlmostEqual(real[k],old[k],places=10)

    def test_probe_basis_is_part_of_certificate_identity(self):
        with threadpool_limits(limits=1):
            p=calc.annotate(calc.ReadoutModel(calc.Device(),FAST),14,12)
            p=calc.check_chi_sign_stability(p,early_exit=True)
        self.assertTrue(calc.chi_sign_certificate_valid(p))
        changed=p|{'chi_sign_probe_basis':asdict(AUDIT)}
        self.assertFalse(calc.chi_sign_certificate_valid(changed))

    def test_legacy_feasibility_entry_routes_to_adaptive_validator(self):
        with threadpool_limits(limits=1):
            p=calc.annotate(calc.ReadoutModel(calc.Device(),FAST),14,12)
            checked=calc.validate({'point':p})
        self.assertEqual(checked['point']['validation']['policy'],'adaptive-readout-v1')
        self.assertEqual(checked['point']['device'],p['device'])
        self.assertEqual(checked['point']['ej1_ghz'],p['ej1_ghz'])
        self.assertEqual(checked['point']['ej2_ghz'],p['ej2_ghz'])

if __name__=='__main__':unittest.main()
