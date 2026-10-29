"""Consolidated off_physics regression suite; original coverage preserved."""


# ---- Original coverage: test_pairs ----
import unittest

import numpy as np
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits

from floating_coupler.model import FULL, build_model
from floating_coupler.pairs import PairModel, local_energy_frame
from floating_coupler.dressing import rail_dressing


def synthetic(cross=True):
    links = [{"from": "A1", "to": "A2", "charge_coupling_ghz": 0.025},
             {"from": "B1", "to": "B2", "charge_coupling_ghz": 0.035}]
    if cross:
        links.extend({"from": a, "to": b, "charge_coupling_ghz": j} for a, b, j in (
            ("A1", "C", 0.002), ("A2", "C", 0.08), ("C", "B1", 0.09),
            ("C", "B2", 0.003), ("A1", "B1", 0.0001), ("A2", "B1", 0.005),
            ("A1", "B2", 0.00001), ("A2", "B2", 0.00015)))
    return build_model({"nodes": list(FULL), "transmons": {
        n: {"ec_ghz": ec, "ej_ghz": ej} for n, ec, ej in zip(
            FULL, (0.17, 0.24, 0.35, 0.16, 0.27), (16, 12, 30, 24, 15), strict=True)},
        "electrostatics": {"mode": "strengths", "links": links},
        "basis": {"levels": 3, "ncut": 20}})


class PairTests(unittest.TestCase):
    def test_decoupled_coupler_does_not_shift_rails_when_tuned(self):
        model = synthetic(False)
        basis = dict(local_levels=3, pair_states=9, coupler_levels=3, ncut=20)
        with threadpool_limits(limits=1):
            off = rail_dressing(model, 9.0, basis)
            on = rail_dressing(model, 7.5, basis)
        np.testing.assert_allclose(off["local_self_frequencies_ghz"], on["local_self_frequencies_ghz"], atol=1e-12)
        self.assertLess(max(m["coupler_excited_probability"] for m in on["modes"]), 1e-12)

    def test_pair_transform_preserves_full_five_transmon_spectrum(self):
        model = synthetic()
        transformed = PairModel(model, local_levels=3, pair_states=9, coupler_levels=3, ncut=20)
        ham, ej = transformed.hamiltonian(8.5)
        full, _ = model.hamiltonian(ej)
        with threadpool_limits(limits=1):
            left, right = eigvalsh(ham), eigvalsh(full)
        np.testing.assert_allclose(left - left[0], right - right[0], atol=2e-12)
        # Exact charge-parity selection is retained without an RWA.
        forbidden = transformed.parity[:, None] != transformed.parity[None, :]
        self.assertLess(np.max(abs(ham[forbidden])), 1e-12)

    def test_disconnected_pairs_have_no_nonlocal_residual(self):
        system = PairModel(synthetic(False), local_levels=3, pair_states=9, coupler_levels=3, ncut=20)
        with threadpool_limits(limits=1):
            row = system.evaluate(8.5)
        self.assertLess(abs(row["cz_0020"]["matrix_element_mhz"]), 1e-8)
        self.assertLess(abs(row["logical_zz_mhz"]), 1e-8)
        self.assertLess(abs(row["logical_local_energy_frame"]["exchange_mhz"]), 1e-8)
        self.assertGreater(row["minimum_subspace_overlap"], 0.999999)

    def test_local_rotations_are_not_interqubit_coupling(self):
        ha = np.array([[0, 0.012], [0.012, 0.09]])
        hb = np.array([[0, -0.02], [-0.02, 0.15]])
        row = local_energy_frame(np.kron(ha, np.eye(2)) + np.kron(np.eye(2), hb))
        self.assertGreater(abs(row["nominal_frame_pauli_mhz"]["XI"]), 1)
        self.assertLess(abs(row["exchange_mhz"]), 1e-10)
        self.assertLess(abs(row["zz_mhz"]), 1e-10)




# ---- Original coverage: test_physical_off ----
"""Regression coverage for the temporary physical OFF objective.

These test virtual dressing and preservation of strong within-rail mixing,
not just the policy label or the value of a raw Hamiltonian entry.
"""
import unittest
import numpy as np
from scipy.linalg import eigh
from floating_coupler.channel_policy import physical_exchange_from_eigenpairs
from floating_coupler.paired_design import off_channel


class PhysicalOffTests(unittest.TestCase):
    def test_four_references_preserve_rail_mixing_and_target_the_physical_link(self):
        h=np.diag([4.61,4.61,5.39,5.39])
        for i,j,g in [(0,1,-.075),(2,3,-.045),(1,2,.002)]:
            h[i,j]=h[j,i]=g
        e,v=eigh(h)
        result=physical_exchange_from_eigenpairs(e,v)
        self.assertAlmostEqual(result['g_mhz'],2.,places=9)
        np.testing.assert_allclose(result['effective_hamiltonian_ghz'],h,atol=1e-12)
        self.assertEqual(off_channel({'physical_exchange':result})['g_mhz'],result['g_mhz'])

    def test_virtual_paths_are_included_before_extracting_physical_exchange(self):
        h=np.diag([4.5,4.6,5.4,5.5,10.])
        h[1,4]=h[4,1]=.01;h[2,4]=h[4,2]=.02
        e,v=eigh(h)
        result=physical_exchange_from_eigenpairs(e,v[:4])
        expected=.5*.01*.02*(1/(4.6-10)+1/(5.4-10))*1000
        self.assertEqual(h[1,2],0.)
        self.assertAlmostEqual(result['g_mhz'],expected,delta=abs(expected)*1e-4)

    def test_logical_residual_cannot_be_used_as_physical_off_proof(self):
        with self.assertRaises(ValueError):
            off_channel({'logical_exchange':{'g_mhz':0.,'valid':True}})




# ---- Original coverage: test_logical_off ----
"""OFF isolation includes virtual auxiliary paths and differs from CZ nulls."""
import unittest
import numpy as np

from floating_coupler.channel_policy import (effective_logical_exchange,
    off_target_definition,require_off_definition,logical_exchange_from_eigenpairs,logical_exchange_definition)
from floating_coupler.off_search import exchange_roots
from floating_coupler.paired_design import off_channel


class LogicalOffTests(unittest.TestCase):
    def test_virtual_auxiliary_exchange_survives_zero_raw_logical_entry(self):
        h=np.diag([10.,10.1,10.15,10.25,11.])
        h[1,4]=h[4,1]=.002
        h[2,4]=h[4,2]=.003
        result=effective_logical_exchange(h,[0,1,2,3])
        expected=.5*.002*.003*(1/(10.1-11)+1/(10.15-11))*1000
        self.assertEqual(h[1,2],0.)
        self.assertAlmostEqual(result['g_mhz'],expected,delta=abs(expected)*1e-4)
        self.assertTrue(result['valid'])
        self.assertEqual(result['observable_definition'],logical_exchange_definition('five'))

    def test_logical_exchange_is_not_auxiliary_matrix_element(self):
        h=np.diag([0.,.09,.15,.24,.8])
        h[1,2]=h[2,1]=.001
        result=effective_logical_exchange(h,[0,1,2,3])
        self.assertAlmostEqual(result['g_mhz'],1.,places=9)
        self.assertEqual(h[2,4],0.)
        with self.assertRaises(ValueError):off_channel({'selected_cz_channel':{'g_mhz':0.}})

    def test_local_rotations_do_not_create_pair_exchange(self):
        ha=np.array([[0.,.008],[.008,.15]])
        hb=np.array([[0.,-.006],[-.006,.09]])
        h=np.kron(ha,np.eye(2))+np.kron(np.eye(2),hb)
        result=effective_logical_exchange(h,range(4))
        self.assertLess(abs(result['g_mhz']),1e-10)
        self.assertAlmostEqual(np.linalg.eigvalsh(h)[-1],np.linalg.eigvalsh(result['effective_hamiltonian_ghz'])[-1],places=12)

    def test_old_off_proofs_rejected_independent_of_valid_on_record(self):
        for old in (None,{'target':'selected_cz_channel_zero'},off_target_definition('three'),{'version':'logical-exchange-off-v1'}):
            with self.assertRaises(ValueError):require_off_definition(old,'five')
        self.assertEqual(require_off_definition(off_target_definition('five')),off_target_definition('five'))

    def test_root_search_finds_tangential_null_without_endpoint_sign_change(self):
        roots=exchange_roots(lambda x:(x-.37)**2,np.linspace(0,1,11),tolerance=1e-10)
        self.assertTrue(any(abs(x-.37)<1e-6 for x in roots),roots)
        self.assertEqual(exchange_roots(lambda x:(x-.37)**2+.01,np.linspace(0,1,11)),[])
        self.assertEqual(exchange_roots(lambda x:x-2,np.linspace(0,1,11)),[])




# ---- Original coverage: test_off_sweep ----
"""The reporting command must not confuse diagnostic range with acceptance."""
import unittest
from off_sweep import sweep



class FrozenSweepTests(unittest.TestCase):
    def test_separate_physical_objective_and_wide_logical_residuals(self):
        basis=dict(local_levels=3,pair_states=9,coupler_levels=3,ncut=20)
        model=synthetic(False)
        result=sweep(model,[7.1,9.9],[7.,12.],basis,points=3)
        self.assertEqual(result['off_definition']['target'],'physical_0100_0010_exchange_zero')
        self.assertEqual(result['plot_range_ghz'],[7.,12.])
        for root in result['roots']:
            self.assertTrue(7.1<root['fc_ghz']<9.9)
        for row in result['rows']:
            self.assertLess(abs(row['logical_exchange_hz']),1e-3)
            self.assertLess(abs(row['logical_double_flip_hz']),1e-3)

    def test_plot_range_must_cover_the_acceptance_range(self):
        with self.assertRaises(ValueError):
            sweep(synthetic(False),[7.,12.],[7.,10.],{},points=3)




# ---- Original coverage: test_channel_policy ----
import unittest
import numpy as np

from floating_coupler import channel_policy as circuit
from transmon_cz import channel_policy as toy
from floating_coupler.paired_design import selected_channel


class SharedChannelPolicyTests(unittest.TestCase):
    def test_circuit_imports_same_authority_objects(self):
        for name in ('ChannelSelection','S1Candidate','select_channel','one_excitation_indices','interaction_scores',
                     'off_target_definition','effective_logical_exchange','logical_exchange_from_eigenpairs',
                     'physical_exchange_from_eigenpairs','logical_exchange_definition'):
            self.assertIs(getattr(circuit,name),getattr(toy,name))

    def test_selected_source_is_not_inferred_from_auxiliary(self):
        chosen=toy.select_channel(
            [toy.S1Candidate('10','B20',4.9,.001,5.,.05,'synthetic','fixture',tracking_verified=True)],
            moved_pair='B',idle_frequency_ghz=5.39,target_g_mhz=5.,auxiliary='B20')
        ham=np.diag([0.,.1,.2,.3,.4])
        ham[1,4]=ham[4,1]=.001
        ham[2,4]=ham[4,2]=.007
        detail=dict(reference_labels=['00','01','10','11','B20'],
                    effective_hamiltonian_ghz=ham.tolist(),
                    logical_local_energy_frame={'local_rotations':[np.eye(2),np.eye(2)]})
        with self.assertRaises(ValueError): selected_channel(detail)
        detail['channel_selection']=chosen.to_dict()
        row=selected_channel(detail)
        self.assertEqual(row['source_logical'],'10')
        self.assertAlmostEqual(row['g_mhz'],7.)
        with self.assertRaises(ValueError):
            selected_channel(detail,source='01',target='B20',selection=chosen)




# ---- Original coverage: test_staged_three_body ----
import unittest
import numpy as np
from staged_three_body import two_excitation,single_excitation,off_roots
from floating_coupler.model import Model
from floating_coupler.design import ej_for_frequency
from core.transmon import Basis

class ThreeBodyChannelTests(unittest.TestCase):
    def test_off_seed_uses_single_excitation_not_cz_exchange(self):
        m=self.model();m.j[0,2]=m.j[2,0]=1e-5
        row=single_excitation(m,8.)
        from floating_coupler.model import local_spectrum
        qa=local_spectrum(m.ej[0],m.ec[0],m.basis)[1]
        qb=local_spectrum(m.ej[2],m.ec[2],m.basis)[1]
        expected=1000*m.j[0,2]*qa[1,0]*qb[0,1]
        self.assertAlmostEqual(row['g_mhz'],expected,delta=abs(expected)*1e-5)
        self.assertEqual(row['off_definition']['target'],'physical_01_10_exchange_zero')
    def model(self):
        ec=np.array([.25,.35,.16]);ej=np.array([ej_for_frequency(f,e,30) for f,e in zip([4.61,8,5.39],ec)])
        return Model(('A2','C','B1'),ec,ej,np.zeros((3,3)),Basis(ncut=30,transmon_levels=4),{})
    def test_disconnected_two_excitation_channel(self):
        row=two_excitation(self.model(),8.)
        self.assertAlmostEqual(row['g_mhz'],0.,places=9)
        self.assertAlmostEqual(row['minimum_overlap'],1.,places=9)
    def test_transition_selection_is_B1_two_excitation(self):
        m=self.model();m.j[0,2]=m.j[2,0]=1e-5
        row=two_excitation(m,8.)
        from floating_coupler.model import local_spectrum
        qa=local_spectrum(m.ej[0],m.ec[0],m.basis)[1]
        qb=local_spectrum(m.ej[2],m.ec[2],m.basis)[1]
        expected=1000*m.j[0,2]*qa[1,0]*qb[1,2]
        self.assertAlmostEqual(row['g_mhz'],expected,delta=abs(expected)*1e-5)


if __name__=='__main__':unittest.main()
