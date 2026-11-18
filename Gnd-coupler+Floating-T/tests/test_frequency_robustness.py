"""Physical frequency-scenario reconstruction and independent sampling."""
import unittest
import numpy as np
from pathlib import Path
import tempfile
from floating_coupler.fabrication import resonator_components
from studies.robustness import draw_resonator_frequencies,identity_offset,usable


class FrequencyRobustnessTests(unittest.TestCase):
    def test_main_and_diagnostic_gate_recovery_use_input_lower_bound(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from floating_coupler.fabrication import gate_fit
        from studies.robustness_worker import missing_gate_diagnostic
        seen=[]
        class NoRoot:
            def __init__(self,model,fc,basis,fb_bounds,**kwargs):
                self.fc=fc;seen.append(fc)
            def at(self,fb,t):
                return dict(fc_ghz=self.fc,fb_ghz=fb,time_ns=t,
                    phase_error_rad=.3+(fb-4.8)**2,postselected_cz_fidelity=.9,
                    raw_cz_fidelity=.8,logical_survival=.88,
                    actual_bias_score_diagnostics=dict(maximum_logical_ratio=.1),
                    minimum_dressed_reference_overlap=.99)
            def phase_root(self,t,**kwargs):raise ValueError('No root')
        model=SimpleNamespace(ej=np.ones(5)*20,ec=np.ones(5)*.2)
        seed=dict(fc_ghz=6.4,fb_ghz=4.8,time_ns=100.)
        off=dict(fc_ghz=9.8,gate_on_lower_ghz=6.5)
        for function,target in [(gate_fit,'floating_coupler.fabrication.PhaseCalibration'),
                                (missing_gate_diagnostic,'floating_coupler.calibrate_on.PhaseCalibration')]:
            seen.clear()
            with patch(target,NoRoot):function(model,seed,off)
            self.assertTrue(seen)
            self.assertAlmostEqual(min(seen),6.5)
            self.assertTrue(all(6.5<=f<9.8 for f in seen))

    def test_frequency_and_impedance_reconstruct_from_lc(self):
        for frequency in (5.8,5.9,6.,6.1,6.2):
            ro=dict(case=dict(resonator_node='r'),physical={},
                    resonator_frequency_scenario=dict(frequency_ghz=frequency,impedance_ohm=50.))
            cr,lr=resonator_components(ro,{})
            self.assertAlmostEqual(1/(2*np.pi*np.sqrt(cr*1e-15*lr*1e-9))/1e9,frequency)
            self.assertAlmostEqual(np.sqrt(lr*1e-9/(cr*1e-15)),50.)

    def test_original_capacitor_policy_unchanged(self):
        ro=dict(case=dict(resonator_node='r'),physical=dict(
            resonator_self_capacitance_ff=530.,resonator_inductance_nh=1.3))
        self.assertEqual(resonator_components(ro,{'resonator:r':.05}),(556.5,1.3))

    def test_double_variation_rejected(self):
        ro=dict(case=dict(resonator_node='r'),physical={},
                resonator_frequency_scenario=dict(frequency_ghz=6.,impedance_ohm=50.))
        with self.assertRaises(ValueError):resonator_components(ro,{'resonator:r':.01})

    def test_new_run_requires_explicit_frequency_ranges(self):
        with self.assertRaisesRegex(ValueError,'New robustness runs require'):
            draw_resonator_frequencies({},41,0)

    def test_frequency_draws_independent_reproducible_in_requested_ranges(self):
        base=dict(resonator_frequency_study=dict(ranges_ghz={'A':[6.,6.2],'B':[5.8,6.]},impedance_ohm=50.))
        samples=[draw_resonator_frequencies(base,41,i) for i in range(1000)]
        self.assertEqual(samples[0],draw_resonator_frequencies(base,41,0))
        self.assertTrue(all(6<=r['A']<6.2 and 5.8<=r['B']<6 for r in samples))
        self.assertLess(abs(np.corrcoef([[r['A'] for r in samples],[r['B'] for r in samples]])[0,1]),.1)

    def test_reuse_does_not_clone_prior_sample_identities(self):
        retained=set(range(10_000_000,10_000_200))
        offset=identity_offset(retained)
        new={offset+i*100_000+j for i in range(20) for j in range(200)}
        self.assertFalse(retained&new)
        self.assertEqual(len(new),4000)

    def test_unavailable_fidelity_never_counts_as_a_finite_gate(self):
        self.assertFalse(usable({'gate':{'best':{'raw_cz_fidelity':None,'postselected_cz_fidelity':.999}}}))
        self.assertFalse(usable({'gate':{}}))
        self.assertTrue(usable({'gate':{'best':{'raw_cz_fidelity':.95,'postselected_cz_fidelity':.999}}}))

    def test_archive_report_preserves_frequency_policy_and_partial_coverage(self):
        from studies.archive import write_archive,report_metadata
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'study.tar.xz'
            inputs=dict(resonator_frequency_study=dict(ranges_ghz={'A':[6.,6.2],'B':[5.8,6.]},impedance_ohm=50))
            write_archive(path,[],inputs,dict(status=dict(state='partial',completed_attempts=0)))
            campaign,status=report_metadata(path)
            self.assertEqual(status['state'],'partial')
            self.assertEqual(campaign['request']['input']['resonator_frequency_study'],inputs['resonator_frequency_study'])


if __name__=='__main__':unittest.main()
