"""Consolidated cz regression suite; original coverage preserved."""


# ---- Original coverage: test_gate ----
import unittest

import numpy as np

from floating_coupler.gate import gate_metrics


class GateMetricTests(unittest.TestCase):
    def test_cz_local_z_and_leakage(self):
        phases = np.array([0.3, 0.7, -0.2, 0.7 - 0.2 - 0.3 + np.pi])
        ideal = np.diag(np.exp(1j * phases))
        metrics = gate_metrics(ideal)
        self.assertAlmostEqual(metrics["raw_cz_fidelity"], 1)
        self.assertAlmostEqual(metrics["phase_error_rad"], 0)
        leaky = gate_metrics(np.sqrt(0.9) * ideal)
        self.assertAlmostEqual(leaky["raw_cz_fidelity"], 0.9)
        self.assertAlmostEqual(leaky["postselected_cz_fidelity"], 1)
        self.assertAlmostEqual(leaky["average_leakage"], 0.1)

    def test_identity_does_not_score_as_cz(self):
        self.assertAlmostEqual(gate_metrics(np.eye(4))["raw_cz_fidelity"], 0.4)




# ---- Original coverage: test_on_gate ----
import unittest

import numpy as np
from scipy.linalg import eigh

from floating_coupler.on_gate import dressed_on_metrics, secular_logical_shifts


class DressedOnGateTests(unittest.TestCase):
    def test_auxiliary_shift_is_included_in_secular_zz(self):
        ham = np.diag([0., 0.1, 0.2, 0.3, 0.105])
        ham[1, 4] = ham[4, 1] = 0.0003
        detail = {"effective_hamiltonian_ghz": ham.tolist(),
                  "reference_labels": ["00", "01", "10", "11", "B20"],
                  "logical_local_energy_frame": {"local_rotations": [np.eye(2).tolist(), np.eye(2).tolist()]}}
        result = secular_logical_shifts(detail)
        expected = (np.sqrt(0.005**2 + 4 * 0.0003**2) - 0.005) / 2 * 1e9
        self.assertAlmostEqual(result["secular_zz_hz"], expected, places=6)
        self.assertTrue(result["well_identified"])

    def test_cz_excursion_is_retained_and_not_postselected_away(self):
        # Four logical references PLUS a resonant auxiliary. A full return on
        # logical 01 is CZ-equivalent up to local Z, while a half cycle leaks.
        ham = np.zeros((5, 5))
        ham[1, 4] = ham[4, 1] = 0.005
        energy, vectors = eigh(ham)
        coefficients = vectors[:4, :].T
        full = dressed_on_metrics(energy, coefficients, 100)
        half = dressed_on_metrics(energy, coefficients, 50)
        self.assertAlmostEqual(full["raw_cz_fidelity"], 1, places=12)
        self.assertAlmostEqual(full["logical_survival"], 1, places=12)
        self.assertAlmostEqual(half["average_leakage"], 0.25, places=12)
        self.assertLess(half["raw_cz_fidelity"], full["raw_cz_fidelity"])




# ---- Original coverage: test_calibrate_on ----
import unittest

import numpy as np
from scipy.linalg import eigh

from floating_coupler.calibrate_on import PhaseCalibration
from floating_coupler.channel_policy import S1Candidate,select_channel


class ResonantCycleCalibration(PhaseCalibration):
    def __init__(self, source=1, minimum_midpoint_b20=None,selected_source='01'):
        selection=select_channel([S1Candidate(selected_source,'B20',5.,.001,5.,.05,'synthetic-5-state','fixture',tracking_verified=True)],moved_pair='B',idle_frequency_ghz=5.2,target_g_mhz=5.,auxiliary='B20')
        super().__init__(None, 7.2, None, [4.999, 5.001], minimum_midpoint_b20=minimum_midpoint_b20,selection=selection)
        self.source = source

    def spectrum(self, fb):
        ham = np.zeros((5, 5))
        ham[self.source, 4] = ham[4, self.source] = 0.005
        ham[4, 4] = fb - 5
        energy, vectors = eigh(ham)
        return energy, vectors[:4, :].T, {"overlap": 1., "auxiliary_coefficients": vectors[4, :]}


class PhaseCalibrationTests(unittest.TestCase):
    def test_offset_frequency_uses_cached_return_then_exact_phase_correction(self):
        from floating_coupler.calibrate_on import cached_duration_search
        from floating_coupler.precision import correct_gate_phase
        cal=ResonantCycleCalibration()
        scan=cached_duration_search(cal.duration_evaluator(5.0002),[95.,105.],100.)
        self.assertIsNone(scan['best'])
        self.assertIsNotNone(scan['best_return'])
        candidate=scan['best_return']
        point=correct_gate_phase(lambda fb:cal.at(fb,candidate['time_ns']),candidate)
        self.assertLess(abs(point['phase_error_rad']),1e-7)
        self.assertEqual(point['time_ns'],candidate['time_ns'])

    def test_warm_start_transfers_phase_when_exact_seed_has_no_pi_duration(self):
        from floating_coupler.calibrate_on import warm_start_gate
        cal=ResonantCycleCalibration()
        class Backend:
            selection=cal.selection
            def duration_evaluator(self,fc,fb):return cal.duration_evaluator(fb)
            def at(self,fc,fb,t):return cal.at(fb,t)
        seed=cal.at(5.0002,100.)
        result=warm_start_gate(Backend(),seed,[7.,9.9],[95.,105.])
        self.assertEqual(result['attempts'][1]['method'],'local five-body phase transfer')
        self.assertLess(abs(result['best']['phase_error_rad']),1e-7)
        self.assertAlmostEqual(result['best']['fb_ghz'],5.,places=7)

    def test_cached_duration_search_does_not_request_another_spectrum(self):
        from floating_coupler.calibrate_on import cached_duration_search
        from unittest.mock import patch
        cal=ResonantCycleCalibration()
        with patch.object(cal,'spectrum',wraps=cal.spectrum) as spectrum:
            evaluate=cal.duration_evaluator(5.)
            result=cached_duration_search(evaluate,[95.,105.],100.,points=41)
            self.assertEqual(spectrum.call_count,1)
        self.assertAlmostEqual(result['best']['time_ns'],100.,places=5)
        self.assertGreater(result['best']['raw_cz_fidelity'],1-1e-8)
        self.assertLess(abs(result['best']['phase_error_rad']),1e-7)

    def test_warm_start_tries_five_body_bias_before_any_frequency_grid(self):
        from floating_coupler.calibrate_on import warm_start_gate
        cal=ResonantCycleCalibration()
        seen=[]
        class Backend:
            selection=cal.selection
            def duration_evaluator(self,fc,fb):
                seen.append((fc,fb))
                return cal.duration_evaluator(fb)
        seed=cal.at(5.,100.)
        result=warm_start_gate(Backend(),seed,[7.,9.9],[95.,105.])
        self.assertEqual(seen,[(seed['fc_ghz'],seed['fb_ghz'])])
        self.assertEqual(result['best']['channel_selection'],seed['channel_selection'])

    def test_seven_mode_batch_matches_direct_spectral_propagation(self):
        from final_seven_validation import Gate
        from floating_coupler.calibrate_on import cached_duration_search
        cal=ResonantCycleCalibration()
        en,coeff,meta=cal.spectrum(5.)
        backend=object.__new__(Gate);backend.selection=cal.selection
        seen=[]
        def spectrum(fc,fb):
            seen.append((fc,fb))
            return en,coeff,meta['auxiliary_coefficients'],dict(
                minimum_subspace_overlap=1.,_gate_scores={'maximum_logical_ratio':.05})
        backend.spectrum=spectrum
        rows=backend.at_times(7.2,5.,[95.,100.,105.])
        self.assertEqual(len(seen),1)
        for row,t in zip(rows,[95.,100.,105.]):
            direct=cal.at(5.,t)
            for key in ['raw_cz_fidelity','postselected_cz_fidelity','logical_survival','phase_error_rad']:
                self.assertAlmostEqual(row[key],direct[key],places=12)
        evaluate=backend.duration_evaluator(7.2,5.)
        count=len(seen)
        result=cached_duration_search(evaluate,[95.,105.],100.)
        self.assertEqual(len(seen),count)
        self.assertGreater(result['best']['raw_cz_fidelity'],.9999)

    def test_phase_only_correction_keeps_duration_and_coupler_fixed(self):
        from floating_coupler.precision import correct_gate_phase
        cal=ResonantCycleCalibration()
        point=cal.at(5.0002,100.)
        corrected=correct_gate_phase(lambda fb:cal.at(fb,100.),point,cal.fb_bounds)
        self.assertLess(abs(corrected['phase_error_rad']),1e-7)
        self.assertEqual(corrected['time_ns'],100.)
        self.assertEqual(corrected['fc_ghz'],7.2)
        self.assertLessEqual(corrected['phase_correction_evaluations'],5)

    def test_flat_resonant_peak_stops_before_sub_accuracy_polishing(self):
        from floating_coupler.precision import refine_gate_scalar
        cal=ResonantCycleCalibration()
        result=refine_gate_scalar(lambda t:cal.at(5.,t),(99.9,100.1),100.,xatol=.02)
        self.assertEqual(result['stop_reason'],'metric_plateau_1e-4')
        self.assertLessEqual(result['evaluations'],5)
        self.assertGreater(result['best']['raw_cz_fidelity'],1-1e-4)
        self.assertLess(abs(result['best']['phase_error_rad']),1e-7)

    def test_non_pi_endpoint_is_not_accepted_when_budget_expires(self):
        from floating_coupler.precision import refine_gate_controls
        cal=ResonantCycleCalibration()
        def invalid(x):return dict(cal.at(5.,100.),phase_error_rad=.2)
        with self.assertRaisesRegex(ValueError,'phase-pi feasible'):
            refine_gate_controls(invalid,[0.],[(None,None)],max_evaluations=2)

    def test_survival_change_prevents_false_fidelity_plateau(self):
        from floating_coupler.precision import GateRefinement
        cal=ResonantCycleCalibration()
        def evaluate(x):
            return dict(cal.at(5.,100.),logical_survival=.99+.001*x[0])
        track=GateRefinement(evaluate)
        # Postselected/raw scores alone are unchanged, but survival is not.
        for value in [0.,1.,2.]:track.accepted([value])
        self.assertEqual(track.reason,'optimizer_complete')

    def test_batch_uses_one_spectrum_and_preserves_auxiliary_population(self):
        from unittest.mock import patch
        cal=ResonantCycleCalibration(minimum_midpoint_b20=.5)
        with patch.object(cal,'spectrum',wraps=cal.spectrum) as spectrum:
            batch=cal.at_times(5.,[95.,100.,105.])
            self.assertEqual(spectrum.call_count,1)
        for row,t in zip(batch,[95.,100.,105.]):
            ordinary=cal.at(5.,t)
            self.assertAlmostEqual(row['raw_cz_fidelity'],ordinary['raw_cz_fidelity'],places=12)
            self.assertAlmostEqual(row['midpoint_selected_auxiliary_from_selected_source'],
                                   ordinary['midpoint_selected_auxiliary_from_selected_source'],places=12)
    def test_S1_selected_10_source_is_not_replaced_by_01(self):
        correct=ResonantCycleCalibration(source=2,minimum_midpoint_b20=.5,selected_source='10')
        row=correct.phase_root(100)
        self.assertEqual(row['channel_selection']['source_logical'],'10')
        self.assertGreater(row['midpoint_selected_auxiliary_from_selected_source'],.99)
        self.assertNotIn('midpoint_b20_from_logical01',row)
    def test_requested_source_branch_is_enforced(self):
        correct = ResonantCycleCalibration(source=1, minimum_midpoint_b20=0.5)
        wrong = ResonantCycleCalibration(source=2, minimum_midpoint_b20=0.5)
        self.assertGreater(correct.phase_root(100)["midpoint_b20_from_logical01"], 0.99)
        with self.assertRaises(ValueError):
            wrong.phase_root(100)

    def test_conditional_phase_root_and_return_refinement(self):
        cal = ResonantCycleCalibration()
        phase = cal.phase_root(100)
        self.assertAlmostEqual(phase["fb_ghz"], 5, places=10)
        self.assertLess(abs(phase["phase_error_rad"]), 1e-9)
        optimum = cal.optimize_time([95, 105], points=11)["best"]
        self.assertAlmostEqual(optimum["time_ns"], 100, places=4)
        self.assertAlmostEqual(optimum["raw_cz_fidelity"], 1, places=9)




# ---- Original coverage: test_idle_gate ----
from dataclasses import replace
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from core.transmon import Basis
from floating_coupler.design import ej_for_frequency
from floating_coupler.idle_gate import IdleDressedGate, charge_vectors
from floating_coupler.model import local_spectrum
from test_off_physics import synthetic


class IdleGateTests(unittest.TestCase):
    def test_charge_basis_phase_matches_shared_local_operators(self):
        vectors = charge_vectors(18, 0.24, 20, 4)
        charges = np.arange(-20, 21)
        projected_n = vectors.T @ (charges[:, None] * vectors)
        _, n = local_spectrum(18, 0.24, Basis(ncut=20, transmon_levels=4))
        np.testing.assert_allclose(projected_n, n, atol=1e-12)

    def test_no_switch_preserves_exact_dressed_idle_encoding(self):
        model = synthetic()
        ej = model.ej.copy()
        for i in (3, 4):
            ej[i] = ej_for_frequency(5.2, model.ec[i], 20)
        model = replace(model, ej=ej)
        basis = dict(local_levels=3, pair_states=9, coupler_levels=3, ncut=20)
        with threadpool_limits(limits=1):
            gate = IdleDressedGate(model, 9.0, basis)
            zero = gate.at(5.2, 9.0, 0)
            hold = gate.at(5.2, 9.0, 137)
            populations = gate.leakage_populations(5.2, 9.0, 137)
        self.assertAlmostEqual(zero["raw_cz_fidelity"], 0.4, places=10)
        self.assertLess(zero["maximum_idle_state_projection_loss"], 1e-12)
        self.assertLess(abs(hold["average_leakage"]), 1e-12)
        self.assertLess(abs(hold["average_wrong_logical_population"]), 1e-12)
        self.assertLess(populations["outside_idle_logical_probability"], 1e-12)




if __name__=='__main__':unittest.main()
