"""Shared minimum-S1 selection and explicit state-identity regressions."""
from dataclasses import replace
import unittest
import numpy as np

from transmon_cz.channel_policy import (
    S1Candidate, Crossing, select_channel, require_selection, require_same_channel,
    one_excitation_indices, channel_indices, interaction_scores,
)


def candidate(source='10', score=.002, frequency=4.865):
    return S1Candidate(source,'B20',frequency,score,5.,.07,
                       'full-toy-256','same-inputs-g5',tracking_verified=True)


def selection(source='10'):
    return select_channel([candidate(source)],moved_pair='B',
                          idle_frequency_ghz=5.39,target_g_mhz=5.,auxiliary='B20')


class ChannelPolicyTests(unittest.TestCase):
    def test_logical_zero_remains_lower_mode(self):
        self.assertEqual(one_excitation_indices([0,5.2,4.9,9],[0,.97,.96,.01]),(2,1))

    def test_minimum_s1_wins_even_when_encountered_later(self):
        events=[candidate('11',.012,4.963),candidate('10',.002,4.865),candidate('01',.045,4.799)]
        chosen=select_channel(events,moved_pair='B',idle_frequency_ghz=5.39,target_g_mhz=5.,auxiliary='B20')
        self.assertEqual(chosen.source_logical,'10')
        # Reversing approach direction or candidate order does not change S1 ranking.
        reverse=select_channel(events[::-1],moved_pair='B',idle_frequency_ghz=4.7,target_g_mhz=5.,auxiliary='B20')
        self.assertEqual(reverse.source_logical,'10')

    def test_screen_is_applied_before_ranking(self):
        chosen=select_channel([replace(candidate('11',.0001),maximum_logical_ratio=.21),candidate()],
                              moved_pair='B',target_g_mhz=5.,auxiliary='B20')
        self.assertEqual(chosen.source_logical,'10')

    def test_unequal_g_or_estimate_cannot_certify_selection(self):
        for bad in (replace(candidate(),intended_g_mhz=8.3),
                    replace(candidate(),score_kind='uniformly_rescaled_estimate')):
            with self.assertRaises(ValueError):
                select_channel([bad],moved_pair='B',target_g_mhz=5.,auxiliary='B20')

    def test_mixed_bases_conditions_and_unverified_candidates_fail(self):
        for bad in (replace(candidate('11'),score_basis_id='retained-15'),
                    replace(candidate('11'),comparison_id='other-hardware'),
                    replace(candidate('11'),tracking_verified=False)):
            with self.assertRaises(ValueError):
                select_channel([candidate(),bad],moved_pair='B',target_g_mhz=5.,auxiliary='B20')

    def test_old_path_rule_and_crossing_only_input_are_rejected(self):
        with self.assertRaises(ValueError):
            require_selection(dict(selection().to_dict(),definition_version='dual-rail-cz-path-v1'))
        with self.assertRaises(ValueError):
            select_channel([Crossing('11','B20',.4,4.963,tracking_verified=True)],
                           moved_pair='B',target_g_mhz=5.,auxiliary='B20')

    def test_unresolved_equal_s1_does_not_revert_to_first_crossing(self):
        with self.assertRaises(ValueError):
            select_channel([candidate('11'),candidate('10')],moved_pair='B',target_g_mhz=5.,auxiliary='B20')

    def test_auxiliary_and_bit_order_are_explicit(self):
        with self.assertRaises(ValueError):require_selection('0020')
        for extra in ({'physical_auxiliary':'0200'},{'logical_order':['B','A']}):
            with self.assertRaises(ValueError):require_selection(dict(selection().to_dict(),**extra))

    def test_on_off_use_identical_selected_channel_record(self):
        chosen=selection()
        self.assertEqual(require_same_channel(chosen,chosen.to_dict()),chosen)
        self.assertEqual(channel_indices(['00','01','10','11','B20'],chosen),(2,4))
        with self.assertRaises(ValueError):require_same_channel(chosen,selection('01'))

    def test_s1_counts_unintended_edges_and_s2_is_separate(self):
        energy=np.array([0.,.1,.2,.3,.2,.8])
        h=np.zeros((6,6))
        for i,j,g in [(2,4,.005),(0,4,.02),(1,5,.014),(4,5,.03)]:h[i,j]=h[j,i]=g
        score=interaction_scores(energy,h,[0,1,2,3],2,4)
        self.assertAlmostEqual(score['S1'],(.1**2+.02**2)/4)
        self.assertAlmostEqual(score['S2'],.1**2+.05**2)
        changed=h.copy();changed[4,5]=changed[5,4]=.06
        self.assertEqual(score['S1'],interaction_scores(energy,changed,[0,1,2,3],2,4)['S1'])
        with self.assertRaises(ValueError):interaction_scores(energy,h,[0,1,2,3],1,4)


if __name__=='__main__':unittest.main()
