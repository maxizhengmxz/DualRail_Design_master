"""Consolidated runtime regression suite; original coverage preserved."""


# ---- Original coverage: test_performance ----
"""Exact accelerations and practical accuracy guards, without production scans."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from core.transmon import Basis,transmon_spectrum
from floating_coupler.model import local_spectrum
from floating_coupler.pairs import PairModel,pair_spectrum
from floating_coupler.precision import compare,verify_fixed_controls
from floating_coupler.batch import run_tasks
from floating_coupler.on_gate import dressed_on_metrics,dressed_on_metrics_many
from test_off_physics import synthetic


def independent_job(task):
    return dict(value=task['n']**2,pid=os.getpid())


class PerformanceTests(unittest.TestCase):
    def test_tridiagonal_spectrum_matches_dense_full_cosine_with_same_charge_gauge(self):
        for ej,ec in [(16.,.165),(12.,.25),(130.,.1)]:
            basis=Basis(ncut=25,transmon_levels=6)
            e,q=transmon_spectrum(ej,ec,basis)
            signs=np.ones(len(e))
            for i in range(1,len(e)):signs[i]=signs[i-1]*(1 if q[i-1,i]>=0 else -1)
            q=q*signs[:,None]*signs[None,:]
            actual=local_spectrum(ej,ec,basis)
            np.testing.assert_allclose(actual[0],e,atol=1e-10,rtol=0)
            np.testing.assert_allclose(actual[1],q,atol=1e-10,rtol=0)
            with self.assertRaises(ValueError):actual[0][0]=1

    def test_pair_cache_invalidates_for_charge_coefficient_and_cannot_be_corrupted(self):
        model=synthetic();first=pair_spectrum(model,(0,1),3,20)
        first['auxiliary']['20']=-1
        again=pair_spectrum(model,(0,1),3,20)
        self.assertNotEqual(again['auxiliary']['20'],-1)
        model.j[0,1]*=1.1;model.j[1,0]=model.j[0,1]
        changed=pair_spectrum(model,(0,1),3,20)
        self.assertNotEqual(again['splitting_mhz'],changed['splitting_mhz'])

    def test_odd_only_off_matches_full_model(self):
        system=PairModel(synthetic(),local_levels=3,pair_states=9,coupler_levels=3,ncut=20)
        full=system.evaluate(8.5)['physical_exchange']
        fast=system.physical_off(8.5)['physical_exchange']
        self.assertAlmostEqual(full['g_mhz'],fast['g_mhz'],places=9)

    def test_batched_gate_metrics_match_independent_times(self):
        e=np.array([0.,.09,.15,.24,.7]);q,_=np.linalg.qr(np.arange(25.).reshape(5,5)+np.eye(5))
        coeff=q[:,:4];times=[95.,100.,105.]
        rows=dressed_on_metrics_many(e,coeff,times)
        for row,t in zip(rows,times):
            original=dressed_on_metrics(e,coeff,t)
            for key in ('raw_cz_fidelity','postselected_cz_fidelity','logical_survival','phase_error_rad'):
                self.assertAlmostEqual(row[key],original[key],places=11)

    def test_hz_and_gate_percentage_point_precision_and_early_stop(self):
        old=dict(physical_exchange_hz=0.,logical_exchange_hz=1000.,logical_double_flip_hz=900.)
        self.assertTrue(compare(old,{k:v+.9 for k,v in old.items()},'residual')['converged'])
        self.assertFalse(compare(old,{k:v+1.1 for k,v in old.items()},'residual')['converged'])
        gate=dict(raw_cz_fidelity=.99,postselected_cz_fidelity=.999,logical_survival=.991)
        self.assertTrue(compare(gate,{k:v+5e-5 for k,v in gate.items()},'gate')['converged'])
        self.assertFalse(compare(gate,{k:v+2e-4 for k,v in gate.items()},'gate')['converged'])
        seen=[]
        def evaluate(b):seen.append(b);return {k:v+.2/b['level'] for k,v in old.items()}
        result=verify_fixed_controls(evaluate,[{'level':4},{'level':4},{'level':5},{'level':8}],'residual')
        self.assertTrue(result['converged']);self.assertEqual(len(seen),2)
        self.assertEqual(result['final_basis'],{'level':5})

    def test_spawned_checkpoint_resume_and_source_guard(self):
        with TemporaryDirectory() as folder:
            tasks=[dict(n=2),dict(n=3)]
            first=run_tasks(independent_job,tasks,folder,workers=2,signature='version-a')
            self.assertEqual([r['value'] for r in first],[4,9])
            again=run_tasks(independent_job,tasks,folder,workers=1,signature='version-a')
            self.assertEqual(again,first)
            with self.assertRaises(ValueError):
                run_tasks(independent_job,tasks,folder,workers=1,signature='version-b')
            self.assertFalse((Path(folder)/'writer.lock').exists())

    def test_budget_exhaustion_and_nonfinite_results_are_not_convergence(self):
        def metrics(b):
            return dict(raw_cz_fidelity=.9+b['level']*.001,
                        postselected_cz_fidelity=.95,logical_survival=.96)
        self.assertFalse(verify_fixed_controls(metrics,[{'level':4},{'level':5}],'gate')['converged'])
        good=metrics({'level':4});bad=dict(good,raw_cz_fidelity=float('nan'))
        self.assertFalse(compare(good,bad,'gate')['converged'])

    def test_stable_nonzero_residual_is_not_accepted_as_a_null(self):
        from search_correct_channel_caps import optimize
        x=[10.,12.,.9,.4,7.1,9.8,4.85]
        metrics=dict(g_on_mhz=-5.,detuning_on_mhz=0.,g_off_mhz=81e-6,
            own_coupler_fraction=.8,on_maximum_unintended_ratio=.1,
            on_minimum_overlap=.99,off_minimum_overlap=.99,rail_loading_sum=.05)
        physical=dict(full_branch_budget={n:dict(specified_links_loading_fraction=.025) for n in ('A2','B1')})
        class ConstantEvaluator:
            calls=0
            def __call__(self,values):
                self.calls+=1
                return dict(x=list(values),metrics=metrics,physical=physical)
        with TemporaryDirectory() as folder:
            result=optimize(ConstantEvaluator(),x,'cancel',Path(folder)/'progress.json',maxiter=2)
        self.assertFalse(result['feasible'])

    def test_excess_loading_rejected_before_any_pair_diagonalization(self):
        from floating_coupler.refine_capacitors import DesignEvaluator
        from floating_coupler.channel_policy import select_channel,S1Candidate
        selection=select_channel([S1Candidate('10','B20',4.85,.001,5.,.05,'test','fixture',tracking_verified=True)],
            moved_pair='B',idle_frequency_ghz=5.39,target_g_mhz=5.,auxiliary='B20')
        ecs=dict(A1=.165,A2=.25,B1=.16,B2=.26)
        physical=dict(full_branch_budget={n:dict(specified_links_loading_fraction=.11) for n in ('A2','B1')})
        circuit=SimpleNamespace(request=dict(loaded_ec_ghz=ecs),frequencies=dict(A1=4.61,A2=4.61,B1=5.39,B2=5.39),
            fit=lambda caps,ec:(np.array([.165,.25,ec,.16,.26]),np.zeros((5,5)),physical))
        evaluator=DesignEvaluator(circuit,selection,dict(local_levels=3,pair_states=9,coupler_levels=3,ncut=20))
        with patch('floating_coupler.refine_capacitors.PairModel') as quantum:
            with self.assertRaisesRegex(ValueError,'loading'):
                evaluator([5.,10.,.5,.4,7.1,9.8,4.85])
            quantum.assert_not_called()




if __name__=='__main__':unittest.main()
