"""Integrated geometry must honor the case's approved idle splitting targets."""
import unittest
from standalone_pad_readout import build_device
from integrated_readout import construct
from floating_coupler.pairs import pair_spectrum


class IntegrationSplittingTests(unittest.TestCase):
    def test_nondefault_idle_targets_are_preserved_with_loaded_ECs(self):
        readouts={}
        for pair,names,ec,f,fr in [('A',['A1','A2'],[.154,.208],4.3,6.1),
                                  ('B',['B1','B2'],[.216,.252],4.86,5.9)]:
            case=dict(nodes=names,pad_pairs={n:[n+'1',n+'2'] for n in names},loaded_ec_ghz=ec,
                      resonator_node='r1' if pair=='A' else 'r2')
            coordinates=[f,f,12.,5.,fr,.5,.5]
            _,physical=build_device(case,coordinates)
            readouts[pair]=dict(case=case,coordinates=coordinates,physical=physical)
        request=dict(loaded_ec_ghz={'A1':.154,'A2':.208,'C':.3,'B1':.216,'B2':.252},
                     idle_bare_frequencies_ghz={'C':9.},idle_isolated_splittings_mhz=[155.,95.],
                     seed_grounded_three_body=dict(physical=dict(physical_links=[
                         {'from':'A22','to':'C1','capacitance_ff':1.},
                         {'from':'C1','to':'B11','capacitance_ff':1.},
                         {'from':'A22','to':'B11','capacitance_ff':.01}])))
        model,physical=construct(request,readouts)
        self.assertAlmostEqual(pair_spectrum(model,(0,1),6,35)['splitting_mhz'],155.,places=5)
        self.assertAlmostEqual(pair_spectrum(model,(3,4),6,35)['splitting_mhz'],95.,places=5)
        self.assertEqual(physical['isolated_pair_splitting_targets_mhz'],[155.,95.])
        for n,ec in request['loaded_ec_ghz'].items():self.assertAlmostEqual(physical['loaded_ec_ghz'][n],ec,places=10)
