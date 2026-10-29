"""Consolidated readout regression suite; original coverage preserved."""


# ---- Original coverage: test_explicit_readout_circuit ----
import unittest,numpy as np
from standalone_pad_readout import build_device
from floating_coupler.pad_electrostatics import build_pad_circuit
from floating_coupler.mixed_electrostatics import mixed_circuit,fit_mixed_shunts

class ExplicitReadoutCircuitTests(unittest.TestCase):
    def test_single_pad_readout_equivalence_and_equal_physical_links(self):
        case={'nodes':['A1','A2'],'pad_pairs':{'A1':['a','b'],'A2':['c','d']},'loaded_ec_ghz':[.165,.25],'resonator_node':'r'}
        dev,p=build_device(case,[4.61,4.61,16.,10.,6.08,.3,.65])
        circuit=build_pad_circuit(p['pad_pairs'],p['ground_capacitances_ff'],{'r':p['resonator_self_capacitance_ff']},p['physical_links'],p['pad_mutual_capacitances_ff'])
        gauge=np.diag([-1,1,1]);np.testing.assert_allclose(np.linalg.inv(dev.capacitance_matrix())*1e-15,gauge@circuit.inverse_modal_capacitance_per_ff@gauge,atol=1e-13)
        self.assertEqual([e['capacitance_ff'] for e in p['physical_links'][1:]],[10.,10.])
        np.testing.assert_allclose(list(circuit.loaded_ec_ghz.values()),[.165,.25],atol=1e-10)
    def test_grounded_C_with_retained_resonator_matches_direct_node_reduction(self):
        pads={'A':['a','b'],'C':['c','g']};grounds={'a':80.,'b':80.,'c':0.,'g':0.};links=[{'from':'b','to':'c','capacitance_ff':10.},{'from':'a','to':'r','capacitance_ff':8.}];targets={'A':.25,'C':.4}
        shunts,c=fit_mixed_shunts(pads,grounds,links,targets,grounded_pads=('g',),resonator_caps_ff={'r':530.})
        self.assertEqual(c['modal_order'],['A','C','r']);self.assertNotIn('g',c['node_order'])
        np.testing.assert_allclose(c['loaded_ec_ghz'],[.25,.4],atol=1e-10)
        np.testing.assert_allclose(c['K'],c['embedding'].T@np.linalg.inv(c['nodal_capacitance_ff'])@c['embedding'],atol=1e-13)


# ---- Original coverage: test_pair_readout ----
"""Physical floating-to-differential reduction and exact loaded-EC budget."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from independent_readout import device_for_ec, float_physical, policy, equal_coordinates, evaluate
from independent_readout import audit
from core.transmon import Basis
from readout_workflow import calculation as study
from readout_workflow.calculation import ReadoutModel


class PairReadoutTests(unittest.TestCase):
    def test_equal_family_and_independent_energy_loops(self):
        x=equal_coordinates([4.61,.4,1.9,4.9,6.04])
        self.assertEqual(x[3],x[4])
        basis=Basis(15,4,4,15)
        p=evaluate({'ec':[165.,250.]},x,basis)
        independent=audit(p,basis)
        self.assertTrue(independent['equal_readout_capacitors'])
        np.testing.assert_allclose(independent['agreement_ghz'],[0,0],atol=2e-11)
        self.assertEqual(len({v['index'] for v in independent['states'].values()}),6)

    def test_case_policy_updates_both_acceptance_paths(self):
        old_conditions,old_baseline=study.CONDITIONS,study.BASELINE
        try:
            policy({'window':(.08,.10)})
            self.assertEqual(study.CONDITIONS.ej_over_ec_min,45.)
            self.assertEqual(study.BASELINE['ej_ec'],45.)
            self.assertEqual(study.BASELINE['logical_frequency'],(.08,.10))
        finally:
            study.CONDITIONS,study.BASELINE=old_conditions,old_baseline

    def test_floating_common_mode_elimination(self):
        d=device_for_ec([165.,250.],1.7,4.1,2.8,6.01)
        c=np.zeros((5,5))  # A+,A-,B+,B-,resonator signal; return is ground
        def link(i,j,value):
            c[i,i]+=value
            if j is not None:
                c[j,j]+=value;c[i,j]-=value;c[j,i]-=value
        link(0,1,d.C1_shunt_fF);link(2,3,d.C2_shunt_fF)
        link(0,2,2*d.C12_fF);link(1,3,2*d.C12_fF)
        link(0,4,2*d.C1r_fF);link(1,None,2*d.C1r_fF)
        link(2,4,2*d.C2r_fF);link(3,None,2*d.C2r_fF)
        link(4,None,d.Cr_fF)
        t=np.array([[.5,0,0,1,0],[-.5,0,0,1,0],
                    [0,.5,0,0,1],[0,-.5,0,0,1],[0,0,1,0,0]])
        block=t.T@c@t
        reduced=block[:3,:3]-block[:3,3:]@np.linalg.solve(block[3:,3:],block[3:,:3])
        np.testing.assert_allclose(reduced,d.capacitance_matrix()*1e15,atol=2e-13)
        np.testing.assert_allclose(ReadoutModel(d).ec_ghz[:2]*1000,[165,250],atol=1e-10)

    def test_impedance_and_physical_link_reporting(self):
        d=device_for_ec([160,260],2.6,.7,2.8,6.02)
        self.assertAlmostEqual(np.sqrt(d.Lr_nH*1e-9/(d.Cr_fF*1e-15)),50)
        physical=float_physical(d)
        self.assertEqual(physical['transmon_resonator_link_each_ff'],[1.4,5.6])
        self.assertEqual(physical['pair_link_each_ff'],5.2)




# ---- Original coverage: test_seven_readout ----
import unittest,numpy as np
from core.transmon import Basis
from floating_coupler.model import Model,FULL
from floating_coupler.design import ej_for_frequency
from seven_readout import SevenReadout

class SevenModeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ec=np.array([.165,.25,.4,.16,.26]);ej=np.array([ej_for_frequency(f,e,20) for f,e in zip([4.61,4.61,9.8,5.39,5.39],ec)]);j=np.zeros((5,5))
        for a,b,value in [(0,1,-.05),(3,4,-.027),(1,2,-.15),(2,3,.12)]:j[a,b]=j[b,a]=value
        model=Model(FULL,ec,ej,j,Basis(ncut=20,transmon_levels=4),{})
        physical={'K_per_ff':np.zeros((7,7)).tolist(),'resonator_parameters':{r:{'fr_ghz':f,'q_zpf_coulomb':1e-19,'gamma_ghz_per_n':[0.]*5} for r,f in [('r1',5.99),('r2',6.01)]}}
        cls.s=SevenReadout(model,physical,9.8,basis={'local_levels':4,'pair_states':6,'coupler_levels':3,'ncut':20},photons=3,eigenlevels=100).solve()
    def test_resonator_decoupling_gives_zero_chi_and_unit_dressed_reference_overlap(self):
        for pair in ['A','B']:
            for row in self.s.readout(pair):
                self.assertAlmostEqual(row['chi_m_ghz'],0.,places=10)
                self.assertAlmostEqual(row['chi_p_ghz'],0.,places=10)
                self.assertGreater(row['min_assignment_overlap'],1-1e-9)
    def test_physical_off_and_logical_residual_match_five_modes_when_resonators_decouple(self):
        five=self.s.pair.evaluate(9.8)
        seven=self.s.matter_detail()
        self.assertEqual(seven['physical_exchange']['off_definition']['target'],'physical_0100_0010_exchange_zero')
        self.assertEqual(seven['physical_exchange']['off_definition']['model'],'seven')
        self.assertAlmostEqual(five['physical_exchange']['g_mhz'],seven['physical_exchange']['g_mhz'],places=8)
        self.assertAlmostEqual(five['logical_exchange']['g_mhz'],seven['logical_exchange']['g_mhz'],places=8)
        self.assertNotIn('off_definition',seven['logical_exchange'])

    def test_full_hamiltonian_preserves_global_parity(self):
        h=self.s.ham.tocoo();wrong=self.s.parity[h.row]!=self.s.parity[h.col];self.assertLess(float(np.max(abs(h.data[wrong]),initial=0)),1e-11)
    def test_off_only_skips_even_sector_and_matches_full_physical_exchange(self):
        fast=SevenReadout(self.s.model,self.s.physical,9.8,basis=self.s.basis,
                          photons=3,eigenlevels=100,off_only=True).solve()
        self.assertEqual(set(fast.canonical),{-1})
        self.assertIsNone(fast.matter_energy)
        self.assertAlmostEqual(fast.physical_off_detail()['physical_exchange']['g_mhz'],
                               self.s.matter_detail()['physical_exchange']['g_mhz'],places=8)
        with self.assertRaises(ValueError):fast.matter_detail()
    def test_factorized_active_A_readout_matches_shared_three_mode_solver(self):
        from scipy.constants import elementary_charge as e,h
        from readout_workflow.model import Device,ReadoutModel
        dev=Device(115.,75.,3.,5.,5.,530.,1.33);basis=Basis(20,3,3,20);reference=ReadoutModel(dev,basis);ec=np.array([reference.ec_ghz[0],reference.ec_ghz[1],.4,.16,.26]);ej=np.array([ej_for_frequency(f,x,20) for f,x in zip([4.61,4.61,9.8,5.39,5.39],ec)]);j=np.zeros((5,5));j[0,1]=j[1,0]=reference.J_ghz[0,1];j[3,4]=j[4,3]=.027
        model=Model(FULL,ec,ej,j,basis,{})
        gammas=[2*e*reference.qr_zpf*reference.cinv[i,2]/(h*1e9) for i in [0,1]]+[0.,0.,0.]
        physical={'K_per_ff':np.zeros((7,7)).tolist(),'resonator_parameters':{'r1':{'fr_ghz':reference.fr_ghz,'q_zpf_coulomb':reference.qr_zpf,'gamma_ghz_per_n':gammas},'r2':{'fr_ghz':6.01,'q_zpf_coulomb':1e-19,'gamma_ghz_per_n':[0.]*5}}}
        seven=SevenReadout(model,physical,9.8,basis={'local_levels':3,'pair_states':9,'coupler_levels':3,'ncut':20},photons=3,eigenlevels=150).solve();expected=reference.evaluate(ej[0],ej[1])
        for row in seven.readout('A'):
            self.assertAlmostEqual(row['chi_m_ghz'],expected['chi_m_ghz'],places=9)
            self.assertAlmostEqual(row['chi_p_ghz'],expected['chi_p_ghz'],places=9)
            self.assertAlmostEqual(row['hybridization'],expected['hybridization'],places=8)


# ---- Original coverage: test_final_seven_validation ----
import unittest
import numpy as np
from core.transmon import Basis
from floating_coupler.model import Model,FULL
from floating_coupler.design import ej_for_frequency
from seven_readout import SevenReadout
from final_seven_validation import gate_detail

class FinalSevenGateTests(unittest.TestCase):
    def test_even_gate_extraction_matches_full_seven_mode_matter_detail(self):
        ec=np.array([.165,.25,.29,.16,.26]);freq=[4.61,4.61,7.05,4.79,4.79]
        ej=np.array([ej_for_frequency(f,e,20) for f,e in zip(freq,ec)]);j=np.zeros((5,5))
        for a,b,value in [(0,1,-.03),(3,4,-.02),(1,2,-.10),(2,3,.08),(1,3,-.005)]:j[a,b]=j[b,a]=value
        model=Model(FULL,ec,ej,j,Basis(ncut=20,transmon_levels=4),{})
        resonators={r:dict(fr_ghz=f,q_zpf_coulomb=1e-18,gamma_ghz_per_n=g) for r,f,g in [('r1',5.99,[.004,-.006,0,0,0]),('r2',6.01,[0,0,0,.004,-.006])]}
        physical=dict(resonator_parameters=resonators,K_per_ff=np.diag(np.ones(7)).tolist())
        system=SevenReadout(model,physical,7.05,basis=dict(local_levels=4,pair_states=6,coupler_levels=3,ncut=20),photons=2).solve()
        full=system.matter_detail();even=gate_detail(system);indices=[full['reference_labels'].index(label) for label in even['reference_labels']]
        np.testing.assert_allclose(np.array(full['effective_hamiltonian_ghz'])[np.ix_(indices,indices)],even['effective_hamiltonian_ghz'],atol=2e-11,rtol=0)





class ReadoutSweepBasisTests(unittest.TestCase):
    def test_centers_endpoints_and_thresholds_trigger_audit(self):
        from readout_common_sweep import audit_reasons,START_BASIS
        m=dict(chi_ratio_percent=.3,mean_chi_min_abs_mhz=1.2,H_min=.98,
               overlap_min=.98,lambda_max=.07,logical_min_mhz=150,logical_max_mhz=150)
        self.assertEqual(START_BASIS['coupler_levels'],3)
        self.assertTrue(audit_reasons(m,[.14,.16],0,[-100,-50,0,50,100]))
        self.assertFalse(audit_reasons(m,[.14,.16],50,[-100,-50,0,50,100]))
        self.assertIn('chi_ratio_percent',audit_reasons(dict(m,chi_ratio_percent=.99),[.14,.16],50,[-100,0,50,100]))

    def test_hybridization_guard_covers_observed_basis_shift(self):
        from readout_common_sweep import audit_reasons
        m=dict(chi_ratio_percent=.3,mean_chi_min_abs_mhz=1.2,H_min=.95025,
               overlap_min=.98,lambda_max=.07,logical_min_mhz=150,logical_max_mhz=150)
        self.assertIn('H_min',audit_reasons(m,[.14,.16],50,[-100,0,50,100]))


class CalibratedProbeControlsTests(unittest.TestCase):
    def test_probe_preserves_stationary_differential_idle_unless_overridden(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from coupler_readout_probe import probe
        cell=dict(physical=dict(bare_frequencies_ghz=dict(A1=4.60,A2=4.60)),
                  request=dict(idle_bare_frequencies_ghz=dict(A1=4.612,A2=4.625)))
        base=SimpleNamespace(ej=np.ones(5),ec=np.array([.165,.25,.4,.16,.26]))
        # Capture the Hamiltonian controls before any expensive diagonalization.
        for override,expected in [(None,[4.612,4.625]),(4.61,[4.61,4.61])]:
            captured=[]
            def stop(model,*args,**kwargs):
                captured.append(model.ej.copy())
                raise RuntimeError('captured controls')
            with patch('coupler_readout_probe.frozen_model',return_value=base), \
                 patch('coupler_readout_probe.replace',side_effect=lambda model,ej:SimpleNamespace(ej=ej)), \
                 patch('coupler_readout_probe.SevenReadout',side_effect=stop):
                with self.assertRaisesRegex(RuntimeError,'captured controls'):
                    probe(cell,7.05,4.87,override)
            np.testing.assert_allclose(captured[0][:2],[ej_for_frequency(f,e,25) for f,e in zip(expected,base.ec[:2])])


if __name__ == '__main__':
    unittest.main()
