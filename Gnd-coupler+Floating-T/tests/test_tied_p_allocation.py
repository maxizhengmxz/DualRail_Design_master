import pytest

from studies.tied_p_allocation import allocate, self_fraction


def test_self_fraction_uses_equal_ground_series_capacitance():
    physical = {
        "pad_pairs": {"A1": ["A11", "A12"]},
        "ground_capacitances_ff": {"A11": 180.0, "A12": 180.0},
        "pad_mutual_capacitances_ff": {"A1": 10.0},
    }
    assert self_fraction(physical, "A1") == pytest.approx(0.1)


@pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 1.1])
def test_target_p_must_be_open_unit_interval(target):
    with pytest.raises(ValueError, match="target p"):
        allocate({}, target)
