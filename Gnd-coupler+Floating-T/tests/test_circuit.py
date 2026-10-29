"""Consolidated circuit regression suite; original coverage preserved."""


# ---- Original coverage: test_model ----
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from floating_coupler.model import (
    FULL, PORT, SCALE, build_model, floating_inverse, port_exchange, scan_coupler,
)


def synthetic(names=PORT):
    return {
        "nodes": list(names),
        "transmons": {n: {"ec_ghz": 0.25, "ej_ghz": 18 if n != "C" else 30} for n in names},
        "electrostatics": {"mode": "strengths", "links": []},
        "basis": {"ncut": 12, "levels": 3},
    }


class FloatingElectrostaticsTests(unittest.TestCase):
    def test_disconnected_and_single_bridge(self):
        expected = np.diag([1 / 70, 1 / 90])
        disconnected, _ = floating_inverse(("A2", "C"), [70, 90], [])
        bridge, _ = floating_inverse(("A2", "C"), [70, 90], [
            {"from": "A2+", "to": "C-", "capacitance_ff": 8},
        ])
        np.testing.assert_allclose(disconnected, expected, atol=1e-15)
        np.testing.assert_allclose(bridge, expected, atol=1e-15)

    def test_balanced_links_analytic_and_polarity(self):
        caps = [{"from": "A2" + s, "to": "C" + s, "capacitance_ff": 6} for s in ("+", "-")]
        inverse, laplacian = floating_inverse(("A2", "C"), [70, 90], caps)
        expected = np.linalg.inv([[73, -3], [-3, 93]])
        np.testing.assert_allclose(inverse, expected, atol=1e-15)
        np.testing.assert_allclose(laplacian.sum(axis=0), 0, atol=1e-15)
        crossed = copy.deepcopy(caps)
        for item in crossed:
            item["to"] = "C" + ("-" if item["to"].endswith("+") else "+")
        flipped, _ = floating_inverse(("A2", "C"), [70, 90], crossed)
        np.testing.assert_allclose(flipped, inverse * [[1, -1], [-1, 1]], atol=1e-15)

    def test_loaded_ec_reconstruction_and_induced_terms(self):
        config = synthetic()
        caps = []
        for a, b in (("A2", "C"), ("C", "B1")):
            caps.extend({"from": a + s, "to": b + s, "capacitance_ff": 5} for s in ("+", "-"))
        expected_junctions = np.array([70., 85., 90.])
        inverse, _ = floating_inverse(PORT, expected_junctions, caps)
        for name, ec in zip(PORT, SCALE * inverse.diagonal(), strict=True):
            config["transmons"][name]["ec_ghz"] = float(ec)
        config["electrostatics"] = {"mode": "island_capacitors", "capacitors": caps}
        model = build_model(config)
        np.testing.assert_allclose(list(model.electrostatics["junction_capacitances_ff"].values()), expected_junctions, rtol=1e-8)
        self.assertGreater(model.j[0, 2], 0)  # induced without a direct capacitor

    def test_impossible_capacitance_budget_rejected(self):
        config = synthetic()
        config["electrostatics"] = {"mode": "island_capacitors", "capacitors": [
            {"from": "A2" + s, "to": "C" + s, "capacitance_ff": 10000} for s in ("+", "-")
        ]}
        config["transmons"]["A2"]["ec_ghz"] = 1
        config["transmons"]["C"]["ec_ghz"] = 0.1
        with self.assertRaises(ValueError):
            build_model(config)

    def test_strength_conversion_and_positive_definiteness(self):
        config = synthetic()
        config["electrostatics"]["links"] = [
            {"from": "A2", "to": "B1", "signed_j01_mhz": -7}
        ]
        model = build_model(config)
        _, spectra = model.hamiltonian()
        self.assertAlmostEqual(model.j[0, 2] * spectra[0][1][0, 1] * spectra[2][1][0, 1] * 1000, -7)
        config["electrostatics"]["links"] = [
            {"from": "A2", "to": "B1", "charge_coupling_ghz": 10}
        ]
        with self.assertRaises(ValueError):
            build_model(config)

    def test_templates_cannot_launch(self):
        for path in (Path(__file__).resolve().parents[1] / "configs").glob("*template.json"):
            if path.name not in ("template.json", "strengths-template.json"):
                continue  # Other entry points use different schemas.
            with self.assertRaises(ValueError):
                build_model(json.loads(path.read_text()))


class ExchangeTests(unittest.TestCase):
    def test_decoupled_exact_and_subset(self):
        full = build_model(synthetic(FULL))
        subset = full.subset()
        np.testing.assert_array_equal(subset.ec, full.ec[[1, 2, 3]])
        self.assertEqual(subset.names, PORT)
        row = port_exchange(subset, 30)
        self.assertAlmostEqual(row["exchange_mhz"], 0, places=10)
        self.assertAlmostEqual(row["minimum_subspace_overlap"], 1, places=10)

    def test_weak_exchange_matches_input_and_full_hamiltonian(self):
        config = synthetic(FULL)
        config["electrostatics"]["links"] = [{"from": "A2", "to": "B1", "signed_j01_mhz": 1}]
        model = build_model(config)
        ham, _ = model.hamiltonian()
        np.testing.assert_allclose(ham, ham.T, atol=1e-14)
        full = port_exchange(model, 30)
        reduced = port_exchange(model.subset(), 30)
        self.assertAlmostEqual(full["exchange_mhz"], 1, delta=0.001)
        self.assertAlmostEqual(full["exchange_mhz"], reduced["exchange_mhz"], places=8)

    def test_mediated_cancellation_root_and_fixed_hardware(self):
        config = synthetic()
        config["electrostatics"]["links"] = [
            {"from": "A2", "to": "C", "signed_j01_mhz": 60},
            {"from": "C", "to": "B1", "signed_j01_mhz": 60},
            {"from": "A2", "to": "B1", "signed_j01_mhz": 3},
        ]
        model = build_model(config)
        before = model.j.copy()
        result = scan_coupler(model, 22, 40, 15)
        self.assertEqual(len(result["physical_port_roots"]), 1)
        root = result["physical_port_roots"][0]
        self.assertLess(abs(root["exchange_mhz"]), 1e-5)
        self.assertTrue(root["valid"])
        np.testing.assert_array_equal(model.j, before)
        np.testing.assert_array_equal(model.ej, [18, 30, 18])




# ---- Original coverage: test_pad_electrostatics ----
import unittest
import numpy as np
from floating_coupler.model import SCALE
from floating_coupler.pad_electrostatics import build_pad_circuit,fit_balanced_pad_grounds,fit_pad_mutual_capacitances,fit_pad_shunts_with_own_ground_fraction,pad_capacitance_budget
from floating_coupler.model import build_model,port_exchange
from floating_coupler.design import ej_for_frequency
from threadpoolctl import threadpool_limits
from floating_coupler.pad_fanout import replicate_couplers,fanout_report


class ExplicitPadTests(unittest.TestCase):
    def test_own_ratio_fit_keeps_equal_ground_and_loaded_ec(self):
        pads={'A':('a1','a2'),'C':('c1','c2'),'B':('b1','b2')};ec={'A':.25,'C':.35,'B':.16}
        ground={p:SCALE/ec[n] for n in pads for p in pads[n]}
        links=[{'from':'a2','to':'c1','capacitance_ff':6.},{'from':'c1','to':'b1','capacitance_ff':4.}]
        for eta in [.8,.995]:
            c=fit_pad_shunts_with_own_ground_fraction(pads,ec,ground,links,{'C':eta})
            np.testing.assert_allclose(list(c.loaded_ec_ghz.values()),list(ec.values()),rtol=1e-9)
            g=c.ground_capacitances_ff['c1'];m=c.pad_mutual_capacitances_ff['C']
            self.assertEqual(g,c.ground_capacitances_ff['c2'])
            self.assertAlmostEqual((g/2)/(m+g/2),eta,places=11)
            self.assertGreater(m,0.)
            self.assertEqual(c.ground_capacitances_ff['a1'],ground['a1'])

    def test_four_coupler_network_and_ec_compensation(self):
        pads={'A1':('A11','A12'),'A2':('A21','A22'),'C':('C1','C2'),'B1':('B11','B12'),'B2':('B21','B22')}
        target=dict(zip(pads,[.165,.25,.35,.16,.26]))
        grounds={p:SCALE/target[n] for n in pads for p in pads[n]}
        links=[{'from':a,'to':b,'capacitance_ff':c} for a,b,c in
               [('A12','A21',4.),('A22','C1',5.),('C1','B11',3.),('A22','B11',.1),('B12','B21',3.)]]
        fitted=fit_pad_mutual_capacitances(pads,target,grounds,{},links)
        config={'transmons':{n:{'ec_ghz':e} for n,e in target.items()},'electrostatics':{
          'pad_pairs':pads,'ground_capacitances_ff':grounds,'pad_mutual_capacitances_ff':fitted.junction_capacitances_ff,'physical_links':links}}
        one=fanout_report(config,'A2',1)
        self.assertAlmostEqual(one['fixed_physical_hardware']['central_pair']['A2']['loaded_ec_ghz'],.25,places=10)
        four=fanout_report(config,'A2',4)
        self.assertLess(four['fixed_physical_hardware']['central_pair']['A2']['loaded_ec_ghz'],.25)
        self.assertAlmostEqual(four['central_pair_mutuals_compensated']['central_pair']['A2']['loaded_ec_ghz'],.25,places=10)
        self.assertAlmostEqual(four['central_pair_mutuals_compensated']['central_pair']['A2']['ground_fraction'],.5,places=10)
        c,_,_=replicate_couplers(config,'A2',4)
        for k in range(4):self.assertEqual(c.junction_capacitances_ff[f'C@{k}'],fitted.junction_capacitances_ff['C'])
        # Net external loading is recalculated; physical copies cannot be
        # replaced by four times the single-cell percentage.
        b1=one['fixed_physical_hardware']['central_pair']['A2']['all_external_fraction']
        b4=four['fixed_physical_hardware']['central_pair']['A2']['all_external_fraction']
        self.assertGreater(abs(b4-4*b1),1e-4)

    def test_fixed_symmetric_grounds_fitted_mutual_and_budget(self):
        pads={'A':('a1','a2'),'B':('b1','b2')};target={'A':.2,'B':.25}
        grounds={p:SCALE/target[n] for n in pads for p in pads[n]}
        links=[{'from':'a2','to':'b1','capacitance_ff':6.}]
        c=fit_pad_mutual_capacitances(pads,target,grounds,{},links)
        np.testing.assert_allclose(list(c.loaded_ec_ghz.values()),list(target.values()),rtol=1e-10)
        self.assertEqual(c.ground_capacitances_ff,grounds)
        budget=pad_capacitance_budget(c,links)
        for n in pads:
            self.assertAlmostEqual(budget[n]['ground_fraction'],.5,places=10)
            self.assertAlmostEqual(budget[n]['ground_fraction']+budget[n]['pad_mutual_fraction']+budget[n]['all_external_fraction'],1.)
            self.assertAlmostEqual(budget[n]['all_external_fraction'],budget[n]['specified_links_net_loading_fraction'],places=10)
        isolated=fit_pad_mutual_capacitances(pads,target,grounds,{},[])
        for n in pads:self.assertAlmostEqual(isolated.junction_capacitances_ff[n],.5*SCALE/target[n],places=8)

    def test_coupler_pad_choice_changes_above_rail_cancellation(self):
        pads={'A2':('A21','A22'),'C':('C1','C2'),'B1':('B11','B12')}
        ec={'A2':.25,'C':.35,'B1':.16};names=list(pads)
        with threadpool_limits(limits=1):
            for port in ['C1','C2']:
                links=[{'from':a,'to':b,'capacitance_ff':c} for a,b,c in
                       [('A22','C1',5.),(port,'B11',5.),('A22','B11',.01)]]
                r=fit_balanced_pad_grounds(pads,ec,{},links)
                config={'nodes':names,'transmons':{n:{'ec_ghz':ec[n],'ej_ghz':ej_for_frequency(f,ec[n],25)}
                          for n,f in zip(names,[4.61,9.95,5.39])},
                        'electrostatics':{'mode':'strengths','links':[{'from':names[i],'to':names[j],
                          'charge_coupling_ghz':float(r.charge_coefficients_ghz[i,j])}
                          for i in range(3) for j in range(i+1,3)]},'basis':{'levels':4,'ncut':25}}
                m=build_model(config)
                exchanges=[port_exchange(m,ej_for_frequency(f,ec['C'],25))['exchange_mhz'] for f in [6.5,8.]]
                if port=='C1':self.assertLess(exchanges[0]*exchanges[1],0.)
                else:self.assertTrue(all(g<0 for g in exchanges))

    def test_isolated_symmetric_pad_capacitance_is_not_half_loaded_budget(self):
        ec=.25
        result=fit_balanced_pad_grounds({'Q':('q1','q2')},{'Q':ec},{},[])
        self.assertAlmostEqual(result.ground_capacitances_ff['q1'],2*SCALE/ec,places=8)
        self.assertAlmostEqual(result.ground_capacitances_ff['q2'],2*SCALE/ec,places=8)
        self.assertAlmostEqual(result.loaded_ec_ghz['Q'],ec,places=11)
        junction=7.
        result=fit_balanced_pad_grounds({'Q':('q1','q2')},{'Q':ec},{},[],{'Q':junction})
        self.assertAlmostEqual(result.ground_capacitances_ff['q1'],2*(SCALE/ec-junction),places=8)

    def test_full_user_topology_balances_physical_loads_and_loaded_ec(self):
        pads={'A1':('A11','A12'),'A2':('A21','A22'),'C':('C1','C2'),
              'B1':('B11','B12'),'B2':('B21','B22')}
        ec=dict(zip(pads,[.165,.25,.35,.16,.26]))
        # Synthetic test links, not selected fabrication values.
        links=[dict(zip(('from','to','capacitance_ff'),edge)) for edge in
               [('A12','r1',3.),('A21','r1',3.),('A12','A21',2.),
                ('A22','C1',8.),('C1','B11',6.),('A22','B11',.3),
                ('B12','r2',2.),('B21','r2',2.),('B12','B21',1.5)]]
        result=fit_balanced_pad_grounds(pads,ec,{'r1':530.,'r2':531.},links)
        np.testing.assert_allclose(list(result.loaded_ec_ghz.values()),list(ec.values()),rtol=1e-10)
        c=result.nodal_capacitance_ff
        for a,b in pads.values():
            self.assertAlmostEqual(c[result.nodes.index(a),result.nodes.index(a)],c[result.nodes.index(b),result.nodes.index(b)],places=10)
        self.assertTrue(all(v>0 for v in result.ground_capacitances_ff.values()))
        self.assertAlmostEqual(result.ground_capacitances_ff['A11']-result.ground_capacitances_ff['A12'],5.,places=10)
        # Both rail links terminate on C1. Equal total pad loading requires
        # C2's ground capacitor to exceed C1's by their sum, not equal grounds.
        self.assertAlmostEqual(result.ground_capacitances_ff['C2']-result.ground_capacitances_ff['C1'],14.,places=10)
        self.assertEqual(result.inverse_modal_capacitance_per_ff.shape,(7,7))
        self.assertEqual(result.nodes,('A11','A12','A21','A22','C1','C2','B11','B12','B21','B22','r1','r2'))

    def test_reduced_matrix_matches_common_coordinate_schur_complement(self):
        pads={'A':('a1','a2'),'B':('b1','b2')}
        edges=[{'from':'a2','to':'r','capacitance_ff':4.},
               {'from':'b1','to':'r','capacitance_ff':5.}]
        r=fit_balanced_pad_grounds(pads,{'A':.2,'B':.25},{'r':530.},edges)
        # Columns are A differential, B differential, resonator, A common, B common.
        t=np.array([[.5,0,0,1,0],[-.5,0,0,1,0],[0,.5,0,0,1],[0,-.5,0,0,1],[0,0,1,0,0]])
        c=t.T@r.nodal_capacitance_ff@t
        schur=c[:3,:3]-c[:3,3:]@np.linalg.solve(c[3:,3:],c[3:,:3])
        np.testing.assert_allclose(r.effective_modal_capacitance_ff,schur,rtol=1e-12,atol=1e-12)
        # No fabricated direct capacitor. The nonzero kinetic coefficient is
        # induced by the complete capacitance network, not a physical edge.
        self.assertEqual(r.nodal_capacitance_ff[r.nodes.index('a2'),r.nodes.index('b1')],0.)
        self.assertGreater(abs(r.inverse_modal_capacitance_per_ff[0,1]),1e-10)

    def test_impossible_positive_ground_budget_rejected(self):
        with self.assertRaisesRegex(ValueError,'Cannot match'):
            fit_balanced_pad_grounds({'Q':('q1','q2')},{'Q':.25},{},[],{'Q':100.})
        with self.assertRaises(ValueError):
            build_pad_circuit({'Q':('q1','q2')},{'q1':0.,'q2':0.},{},[])




# ---- Original coverage: test_mixed_electrostatics ----
import unittest
import numpy as np
from floating_coupler.mixed_electrostatics import mixed_circuit,fit_mixed_shunts,mixed_budget
from grounded_c2_search import mixed_matrix
from floating_coupler.model import SCALE

class MixedCircuitTests(unittest.TestCase):
    def setUp(self):
        self.pads={'A2':['A21','A22'],'C':['C1','C2'],'B1':['B11','B12']}
        self.grounds={'A21':77.,'A22':77.,'C1':0.,'C2':0.,'B11':121.,'B12':121.}
        self.links=[{'from':'A22','to':'C1','capacitance_ff':10.},{'from':'C1','to':'B11','capacitance_ff':13.},{'from':'A22','to':'B11','capacitance_ff':.8}]
    def test_general_builder_matches_independent_grounded_kernel(self):
        shunts={'A2':36.,'C':24.,'B1':58.}
        c=mixed_circuit(self.pads,self.grounds,shunts,self.links)
        _,k=mixed_matrix(self.grounds,self.links,list(shunts.values()))
        np.testing.assert_allclose(c['K'],k,rtol=1e-13)
        self.assertNotIn('C2',c['node_order'])
    def test_loaded_ECs_and_link_removal_budget(self):
        targets={'A2':.25,'C':.44,'B1':.16}
        shunts,c=fit_mixed_shunts(self.pads,self.grounds,self.links,targets)
        np.testing.assert_allclose(c['loaded_ec_ghz'],list(targets.values()),atol=1e-10)
        b=mixed_budget(self.pads,self.grounds,shunts,self.links,self.links[:2])
        self.assertAlmostEqual(b['C']['own_fraction'],shunts['C']/(SCALE/.44),places=9)
        self.assertAlmostEqual(b['C']['specified_links_loading_fraction'],1-b['C']['own_fraction'],places=9)


# ---- Original coverage: test_loading ----
import unittest

import numpy as np

from floating_coupler.loading import coupler_loading
from floating_coupler.model import PORT, SCALE, floating_inverse


class LoadingTests(unittest.TestCase):
    def test_known_network_keeps_shunts_fixed_and_distinguishes_metrics(self):
        shunts = [70., 25., 110.]
        caps = [{"from": a + s, "to": b + s, "capacitance_ff": c}
                for a, b, c in [("A2", "C", 8.), ("C", "B1", 12.), ("A2", "B1", .5)]
                for s in ("+", "-")]
        inv, _ = floating_inverse(PORT, shunts, caps)
        ec = SCALE * np.diag(inv)
        cfg = dict(nodes=list(PORT), transmons={n: dict(ec_ghz=e, ej_ghz=15.)
            for n, e in zip(PORT, ec)}, electrostatics=dict(mode="island_capacitors", capacitors=caps))
        row = coupler_loading(cfg)
        c = row["nodes"]["C"]
        self.assertAlmostEqual(c["without_C_links_effective_capacitance_ff"], 25., places=8)
        self.assertAlmostEqual(c["physical_across_terminal_capacitance_ff"], 25., places=8)
        self.assertAlmostEqual(c["self_capacitance_fraction"], 25. * ec[1] / SCALE)
        self.assertAlmostEqual(c["self_capacitance_fraction"] + c["net_effective_loading_fraction"], 1.)
        self.assertAlmostEqual(c["differential_C_link_capacitance_ff"], 10.)
        self.assertAlmostEqual(c["net_effective_loading_fraction"], 1 - 25. * ec[1] / SCALE)
        self.assertGreater(c["link_budget_fraction"], c["net_effective_loading_fraction"])
        self.assertAlmostEqual(row["nodes"]["A2"]["differential_C_link_capacitance_ff"], 4.)
        cfg["electrostatics"]["capacitors"][0]["capacitance_ff"] = 9.
        with self.assertRaisesRegex(ValueError, "equal"):
            coupler_loading(cfg)




# ---- Original coverage: test_ground_ratio ----
import unittest
from floating_coupler.pad_electrostatics import fit_pad_shunts_with_own_ground_fraction

class GroundRatioTests(unittest.TestCase):
    def test_asymmetric_own_ratio_and_loaded_ec(self):
        pads={'A':['a','b'],'C':['c','d']}
        target={'A':.25,'C':.35}
        circuit=fit_pad_shunts_with_own_ground_fraction(pads,target,{'a':70.,'b':70.,'c':60.,'d':60.},[{'from':'b','to':'c','capacitance_ff':12.}],{'C':.8},{'C':4.})
        g=circuit.ground_capacitances_ff
        self.assertAlmostEqual(g['d']/g['c'],4.,places=10)
        series=g['c']*g['d']/(g['c']+g['d'])
        self.assertAlmostEqual(series/(series+circuit.pad_mutual_capacitances_ff['C']),.8,places=10)
        for n in target:self.assertAlmostEqual(circuit.loaded_ec_ghz[n],target[n],places=9)

    def test_negative_ratio_rejected(self):
        with self.assertRaises(ValueError):
            fit_pad_shunts_with_own_ground_fraction({'C':['c','d']},{'C':.35},{'c':50.,'d':50.},[],{'C':.8},{'C':-1.})


# ---- Original coverage: test_grounded_c2 ----
import unittest
import numpy as np
from grounded_c2_search import mixed_matrix,fit_circuit
from floating_coupler.pad_electrostatics import build_pad_circuit
from floating_coupler.model import SCALE

class GroundedC2Tests(unittest.TestCase):
    def test_matches_infinite_C2_ground_limit(self):
        pads={'A2':['A21','A22'],'C':['C1','C2'],'B1':['B11','B12']}
        grounds={'A21':77.,'A22':77.,'C1':25.,'C2':1e9,'B11':121.,'B12':121.}
        links=[{'from':'A22','to':'C1','capacitance_ff':10.},{'from':'C1','to':'B11','capacitance_ff':13.},{'from':'A22','to':'B11','capacitance_ff':.7}]
        c=build_pad_circuit(pads,grounds,{},links,{'A2':36.,'C':3.,'B1':58.})
        _,k=mixed_matrix(grounds,links,[36.,28.,58.])
        np.testing.assert_allclose(c.inverse_modal_capacitance_per_ff,k,atol=2e-9,rtol=0)
    def test_loaded_ECs_and_own_budget(self):
        request={'rail_ground_fraction':.5,'ncut':30,'levels':4}
        model,loads,physical=fit_circuit(10.,13.,.8,.4,request)
        np.testing.assert_allclose(model.ec,[.25,.4,.16],atol=1e-9)
        _,k=mixed_matrix(physical['rail_ground_capacitances_ff'],physical['physical_links'],[physical['A2_across_pad_ff'],physical['C1_total_own_capacitance_to_ground_ff'],physical['B1_across_pad_ff']])
        np.testing.assert_allclose(SCALE*np.diag(k),model.ec,atol=1e-9)
        self.assertAlmostEqual(loads[1],1-physical['own_coupler_fraction'],places=8)


# ---- Original coverage: test_grounded_five ----
import unittest,json
from pathlib import Path
import numpy as np
from grounded_five import construct
from floating_coupler.mixed_electrostatics import mixed_fanout

class GroundedFiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request=json.loads((Path(__file__).resolve().parent/'fixtures'/'grounded-five-request.json').read_text())
        cls.seed=cls.request['seed_grounded_three_body']
        cls.model,cls.record=construct(cls.seed,cls.request)
    def test_loaded_ecs_splittings_and_frozen_links(self):
        np.testing.assert_allclose(self.model.ec,list(self.request['loaded_ec_ghz'].values()),atol=1e-10)
        np.testing.assert_allclose(self.record['pair_splittings_mhz'],[150,90],atol=1e-6)
        self.assertEqual(self.record['links'][:3],self.seed['physical']['physical_links'])
        self.assertLess(self.record['budget']['C']['specified_links_loading_fraction'],.45)
    def test_four_branch_budget_restores_only_shared_ECs(self):
        result=mixed_fanout(self.record,'A2',4)
        compensated=result['shared_pair_EC_compensated']['shared_rail_pair']
        self.assertAlmostEqual(compensated['A2']['loaded_ec_ghz'],.25,places=9)
        self.assertAlmostEqual(compensated['A1']['loaded_ec_ghz'],.165,places=9)
        self.assertEqual(len(result['fixed_hardware']['branch_loaded_ec_ghz']),12)


# ---- Original coverage: test_pad_design_constraints ----
import unittest
from floating_coupler.pad_design import validate_pad_design_constraints


class PadFrequencyConstraintTests(unittest.TestCase):
    def test_upper_off_branch_enforced_before_synthesis(self):
        spec=dict(on_frequency_lower_exclusive_ghz=6.5,off_frequency_upper_exclusive_ghz=10.,
                  on_fc_ghz=6.501,off_fc_ghz=9.)
        validate_pad_design_constraints(spec)
        for off in [5.7,6.501,10.,10.1]:
            with self.subTest(off=off),self.assertRaises(ValueError):
                validate_pad_design_constraints(dict(spec,off_fc_ghz=off))
        with self.assertRaises(ValueError):
            validate_pad_design_constraints(dict(spec,on_fc_ghz=6.5))
        with self.assertRaises(ValueError):
            validate_pad_design_constraints(dict(spec,on_frequency_lower_exclusive_ghz=7.))
        validate_pad_design_constraints(dict(spec,on_frequency_lower_exclusive_ghz=7.,on_fc_ghz=7.001))

    def test_missing_bounds_cannot_silently_inherit_seed_constraints(self):
        with self.assertRaises(ValueError):
            validate_pad_design_constraints(dict(on_fc_ghz=6.501,off_fc_ghz=9.95))




# ---- Original coverage: test_refine_capacitors ----
"""Independent reconstruction of the immutable-EC capacitor compensation."""
import json
from pathlib import Path
import unittest

import numpy as np

from grounded_five import construct
from floating_coupler.model import Model,FULL,SCALE
from floating_coupler.mixed_electrostatics import mixed_circuit
from floating_coupler.pairs import pair_spectrum
from floating_coupler.refine_capacitors import CompensatedCircuit


class CapacitorRefinementTests(unittest.TestCase):
    def test_capacitor_changes_preserve_loaded_rail_ec_and_idle_splittings(self):
        request=json.loads((Path(__file__).resolve().parent/'fixtures/grounded-five-on70-request.json').read_text())
        model,record=construct(request['seed_grounded_three_body'],request)
        fitter=CompensatedCircuit(request,record,model)
        ec,j,physical=fitter.fit([7.2,12.8,.72],.43)
        reconstructed=mixed_circuit(physical['pad_pairs'],physical['grounds_ff'],
                                     physical['shunts_ff'],physical['links'])
        np.testing.assert_allclose(reconstructed['loaded_ec_ghz'],[.165,.25,.43,.16,.26],atol=1e-11,rtol=0)
        reconstructed_j=8*SCALE*reconstructed['K'];np.fill_diagonal(reconstructed_j,0)
        np.testing.assert_allclose(j,reconstructed_j,atol=1e-12,rtol=0)
        candidate=Model(FULL,ec,model.ej.copy(),j,model.basis,{})
        splits=[pair_spectrum(candidate,pair,6,35)['splitting_mhz'] for pair in ((0,1),(3,4))]
        np.testing.assert_allclose(splits,[150,90],atol=1e-6,rtol=0)
        self.assertTrue(all(c>0 for c in physical['shunts_ff'].values()))
        self.assertNotEqual(physical['shunts_ff']['A2'],record['shunts_ff']['A2'])




# ---- Original coverage: test_design ----
from dataclasses import replace
import unittest
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from core.transmon import Basis, transmon_spectrum
from floating_coupler.design import (
    computational_residuals, ej_for_frequency, synthesize,
)
from floating_coupler.model import PORT, SCALE, build_model


class DesignTests(unittest.TestCase):
    def test_blank_design_template_rejected(self):
        template = Path(__file__).resolve().parents[1] / "configs" / "design-template.json"
        with self.assertRaises(ValueError):
            synthesize(json.loads(template.read_text())[0])

    def test_numerical_frequency_inversion(self):
        basis = Basis(ncut=25, transmon_levels=3)
        for ec, frequency in ((0.25, 4.61), (0.16, 5.39), (0.3, 7.2)):
            ej = ej_for_frequency(frequency, ec)
            self.assertAlmostEqual(transmon_spectrum(ej, ec, basis)[0][1], frequency, places=9)

    def test_decoupled_residuals(self):
        config = {"nodes": list(PORT), "transmons": {
            n: {"ec_ghz": ec, "ej_ghz": ej} for n, ec, ej in
            zip(PORT, (0.25, 0.3, 0.16), (12, 30, 24), strict=True)},
            "electrostatics": {"mode": "strengths", "links": []}}
        row = computational_residuals(build_model(config))
        self.assertAlmostEqual(row["physical_zz_mhz"], 0, places=9)
        self.assertAlmostEqual(row["double_flip_mhz"], 0, places=9)
        np.testing.assert_allclose(row["assigned_eigenstate_coupler_mean_local_level"], 0, atol=1e-12)

    def test_fixed_capacitor_off_on_synthesis(self):
        # Synthetic regression fixture; not a saved user case or physical default.
        spec = {"rail_ec_ghz": [0.24, 0.17], "fa_ghz": 4.5,
                "off_fb_ghz": 5.3, "on_fb_ghz": 4.7,
                "off_fc_ghz": 11, "on_fc_ghz": 6.8,
                "link_loading_fractions": [0.09, 0.09],
                "coupler_ec_bounds_ghz": [0.1, 1.1], "direct_each_max_ff": 5,
                "target_exchange_mhz": 4, "search_levels": 3, "search_ncut": 20}
        with threadpool_limits(limits=1):
            result = synthesize(spec)
        self.assertLess(abs(result["off"]["exchange_mhz"]), 1e-6)
        self.assertAlmostEqual(abs(result["on"]["exchange_mhz"]), 4, places=6)
        np.testing.assert_allclose(result["loading_audit"]["link_differential_loading_fractions"], 0.045)
        self.assertLess(max(result["loading_audit"]["link_physical_island_loading_fractions"]), 0.1)
        model = build_model(result["config"])
        # No EC/J change between off and on; only B1/C Josephson energies differ.
        on = replace(model, ej=np.asarray(result["on"]["ej_ghz"]))
        np.testing.assert_array_equal(model.ec, on.ec)
        np.testing.assert_array_equal(model.j, on.j)
        self.assertEqual(model.ej[0], on.ej[0])
        caps = result["config"]["electrostatics"]["capacitors"]
        self.assertEqual(len(caps), 6)
        self.assertAlmostEqual(caps[0]["capacitance_ff"], SCALE / 0.24 * 0.09)




# ---- Original coverage: test_paired_design ----
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from floating_coupler.model import SCALE
from floating_coupler.paired_design import synthesize_paired, selected_channel, off_channel
from floating_coupler.channel_policy import S1Candidate,select_channel


class PairedDesignTests(unittest.TestCase):
    def test_asymmetric_budget_and_dressed_channel_constraints(self):
        spec = dict(rail_ec_ghz={"A1": .18, "A2": .24, "B1": .17, "B2": .25},
                    coupler_ec_ghz=.35, idle_fa_ghz=4.4, idle_fb_ghz=5.2,
                    pair_splittings_mhz=[80, 120], on_fc_ghz=7.2, off_fc_ghz=14.,
                    on_g_target_mhz=1.5, maximum_loading_fraction=.09,
                    gate_fb_bounds_ghz=[4.5, 5.0], loading_ratio_a_over_b=1.2,
                    basis=dict(local_levels=3, pair_states=9, coupler_levels=3, ncut=20))
        spec["maximum_coupler_loading_fraction"] = .4
        spec['channel_selection']=select_channel([S1Candidate('01','B20',4.65,.001,5.,.05,'synthetic','fixture',tracking_verified=True)],moved_pair='B',idle_frequency_ghz=5.2,target_g_mhz=5.,auxiliary='B20').to_dict()
        with threadpool_limits(limits=1):
            result = synthesize_paired(spec, [.04, .05, 3., 4., 4.65])
        self.assertTrue(result["feasible"], result["message"])
        a, b = result["differential_loading_fractions"]
        self.assertAlmostEqual(a / b, 1.2, places=9)
        self.assertLess(max(a, b), .09)
        self.assertLessEqual(result["coupler_link_loading_fraction"], .4)
        self.assertAlmostEqual(result["combined_rail_loading_fraction"], a + b)
        np.testing.assert_allclose(result["coupler_link_each_ff"],
                                   2 * np.array([a, b]) * SCALE / [.24, .17], rtol=1e-10)
        self.assertLess(abs(off_channel(result["off"])["g_mhz"]), 1e-5)
        self.assertAlmostEqual(selected_channel(result["on"])["g_mhz"], 1.5, places=5)
        self.assertLess(abs(selected_channel(result["on"])["detuning_mhz"]), 1e-4)
        # A tiny coupler budget must bind the solve, rather than accepting the
        # unconstrained geometry and merely printing a warning afterwards.
        spec["maximum_coupler_loading_fraction"] = .03
        with threadpool_limits(limits=1):
            limited = synthesize_paired(spec)
        self.assertLessEqual(limited["coupler_link_loading_fraction"], .03 + 1e-12)
        self.assertFalse(limited["feasible"])




# ---- Original coverage: test_off_seed ----
import unittest
from floating_coupler.off_seed import same_pad_off_frequency,own_ground_fraction_for_off,seed_record


class OffSeedTests(unittest.TestCase):
    def test_seed_matches_independent_weak_link_pad_circuit(self):
        import numpy as np
        from scipy.optimize import brentq
        from floating_coupler.model import SCALE
        from floating_coupler.pad_electrostatics import build_pad_circuit
        pads={'A':('a1','a2'),'C':('c1','c2'),'B':('b1','b2')}
        own_ec={'A':.25,'C':.35,'B':.16};names=list(pads)
        for eta_c in [.5,.9]:
            fractions={'A':.5,'C':eta_c,'B':.5}
            ground={p:2*fractions[n]*SCALE/own_ec[n] for n in pads for p in pads[n]}
            mutual={n:(1-fractions[n])*SCALE/own_ec[n] for n in pads}
            circuit=build_pad_circuit(pads,ground,{},[
                {'from':'a2','to':'c1','capacitance_ff':.01},
                {'from':'c1','to':'b1','capacitance_ff':.01}],mutual)
            j=circuit.charge_coefficients_ghz;ec=circuit.loaded_ec_ghz
            fa,fb=4.61,5.39
            na,nb=np.sqrt(fa/(16*ec['A'])),np.sqrt(fb/(16*ec['B']))
            def exchange(fc):
                nc=np.sqrt(fc/(16*ec['C']))
                kernel=.5*sum(1/(fc-f)+1/(fc+f) for f in [fa,fb])
                return j[0,2]*na*nb-(j[0,1]*na*nc)*(j[1,2]*nc*nb)*kernel
            numerical=brentq(exchange,5.5,8.)
            self.assertAlmostEqual(numerical,same_pad_off_frequency(fa,fb,eta_c),delta=.003)

    def test_equal_qubit_limit_and_detuned_inverse(self):
        for eta in [.1,.25,.5,.75,1.]:
            self.assertAlmostEqual(same_pad_off_frequency(5.,5.,eta),5*(1+eta)**.5,places=12)
            f=same_pad_off_frequency(4.61,5.39,eta)
            self.assertGreater(f,5.39)
            self.assertAlmostEqual(own_ground_fraction_for_off(4.61,5.39,f),eta,places=12)

    def test_upper_branch_and_unsupported_seed_rejected(self):
        self.assertTrue(seed_record(4.61,5.39,6.501,6.8)['frequency_order_valid'])
        self.assertFalse(seed_record(4.61,5.39,7.001,6.8)['frequency_order_valid'])
        self.assertFalse(seed_record(4.61,5.39,7.001,9.95)['within_minimal_weak_link_family'])




if __name__=='__main__':unittest.main()
