import unittest

import numpy as np

from dtc_dualrail.tracking import track_levels_by_overlap


def _symmetric_operator(size, edges):
    operator = np.zeros((size, size))
    for left, right, value in edges:
        operator[left, right] = value
        operator[right, left] = value
    return operator


def _tracking_fixture():
    size = 12
    # Raw ordering deliberately differs from the logical branch ordering.
    level = {
        "g": 0,
        "p": 2,
        "m": 1,
        "r": 3,
        "pp": 6,
        "pm": 5,
        "mm": 4,
        "pr": 8,
        "mr": 7,
    }
    p_operator = _symmetric_operator(
        size,
        [
            (level["g"], level["p"], 1.0),
            (level["p"], level["pp"], 1.4),
            (level["m"], level["pm"], 1.0),
            (level["r"], level["pr"], 1.0),
        ],
    )
    m_operator = _symmetric_operator(
        size,
        [
            (level["g"], level["m"], 1.1),
            (level["m"], level["mm"], 1.5),
            (level["p"], level["pm"], 1.0),
            (level["r"], level["mr"], 1.0),
        ],
    )
    q3_operator = _symmetric_operator(
        size,
        [
            (level["g"], level["r"], 2.0),
            (level["p"], level["pr"], 2.0),
            (level["m"], level["mr"], 2.0),
        ],
    )
    n1_operator = (p_operator + m_operator) / 2
    n2_operator = (p_operator - m_operator) / 2
    expected = [
        level[name] for name in ("g", "p", "m", "r", "pp", "pm", "mm", "pr", "mr")
    ]
    return np.eye(size), q3_operator, n1_operator, n2_operator, expected


class DressedStateTrackingTests(unittest.TestCase):
    def test_labels_complete_two_excitation_manifold(self):
        states, q3, n1, n2, expected = _tracking_fixture()
        energies = np.arange(states.shape[1], dtype=float)

        tracked_level, score, tracked_energy = track_levels_by_overlap(
            states,
            energies,
            q3_operator=q3,
            n1_operator=n1,
            n2_operator=n2,
        )

        self.assertEqual(tracked_level[0].tolist(), expected)
        np.testing.assert_array_equal(tracked_energy[0], energies[expected])
        np.testing.assert_allclose(score, 1.0)

    def test_pm_uses_both_parent_paths_instead_of_greedy_duplicate(self):
        states, q3, n1, n2, expected = _tracking_fixture()
        p_operator = n1 + n2
        m_operator = n1 - n2
        m = expected[2]
        mm = expected[6]
        # A one-path greedy rule would now select mm as pm. The second path
        # remains absent, so the two-path score still selects the true pm state.
        p_operator[m, mm] = p_operator[mm, m] = 3.0
        n1 = (p_operator + m_operator) / 2
        n2 = (p_operator - m_operator) / 2

        tracked_level, _ = track_levels_by_overlap(
            states,
            q3_operator=q3,
            n1_operator=n1,
            n2_operator=n2,
        )

        self.assertEqual(tracked_level[0].tolist(), expected)

    def test_accepts_one_operator_matrix_per_parameter_point(self):
        states, q3, n1, n2, expected = _tracking_fixture()

        tracked_level, _ = track_levels_by_overlap(
            [states, states],
            q3_operator=[q3, q3],
            n1_operator=[n1, n1],
            n2_operator=[n2, n2],
        )

        self.assertEqual(tracked_level.tolist(), [expected, expected])

    def test_falls_back_to_physical_rails_when_hybridization_is_weak(self):
        states, q3, hybridized_n1, hybridized_n2, expected = _tracking_fixture()
        n1 = hybridized_n1 + hybridized_n2
        n2 = hybridized_n1 - hybridized_n2

        tracked_level, score = track_levels_by_overlap(
            states,
            q3_operator=q3,
            n1_operator=n1,
            n2_operator=n2,
            hybridization_threshold=0.5,
        )

        self.assertEqual(tracked_level[0].tolist(), expected)
        np.testing.assert_allclose(score, 1.0)

    def test_fallback_can_be_disabled(self):
        states, q3, hybridized_n1, hybridized_n2, expected = _tracking_fixture()
        n1 = hybridized_n1 + hybridized_n2
        n2 = hybridized_n1 - hybridized_n2

        tracked_level, _ = track_levels_by_overlap(
            states,
            q3_operator=q3,
            n1_operator=n1,
            n2_operator=n2,
            hybridization_threshold=None,
        )

        self.assertNotEqual(tracked_level[0].tolist(), expected)

    def test_requires_all_operators_and_at_least_nine_candidates(self):
        with self.assertRaisesRegex(ValueError, "n1_operator"):
            track_levels_by_overlap(
                np.eye(10),
                q3_operator=np.eye(10),
            )
        with self.assertRaisesRegex(ValueError, "at least 9"):
            track_levels_by_overlap(
                np.eye(8),
                q3_operator=np.eye(8),
                n1_operator=np.eye(8),
                n2_operator=np.eye(8),
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            track_levels_by_overlap(
                np.eye(10),
                q3_operator=np.eye(10),
                n1_operator=np.eye(10),
                n2_operator=np.eye(10),
                hybridization_threshold=1.1,
            )


if __name__ == "__main__":
    unittest.main()
