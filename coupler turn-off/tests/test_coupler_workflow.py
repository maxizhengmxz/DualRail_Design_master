"""New-case contracts, physical rate conversion and fixed-capacitance invariants.

Fixtures are constructed from circuit inputs; no saved study results are read.
"""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from coupler_turn_off.config import fingerprint, validate
from coupler_turn_off.engine import (
    Case,
    objective_mhz,
    resolve_direct_capacitance,
)
from threadpoolctl import threadpool_limits

from readout_workflow.calculation import Device, ReadoutModel


def inputs():
    device = Device(C1_shunt_fF=95, C2_shunt_fF=80, C12_fF=2, C1r_fF=6, C2r_fF=6)
    ec = ReadoutModel(device).ec_ghz
    return dict(
        schema="coupler-off-v1",
        case_id="test",
        pair=dict(
            ej1_ghz=14.0,
            ej2_ghz=12.0,
            device=vars(device),
            loaded_ec1_mhz=1000 * ec[0],
            loaded_ec2_mhz=1000 * ec[1],
            splitting_ghz=0.09,
            chi_m_ghz=-0.001,
            chi_p_ghz=-0.00101,
        ),
        direct_a2_b1=dict(capacitance_ff=0.12, bare_exchange_mhz=None),
        coupler_candidates=[dict(id="geometry", ec_ghz=0.28, links_ff=[2.5, 2.7])],
        screens=dict(
            chi_ratio_max=0.01,
            abs_mean_chi_min_mhz=1.0,
            hybridization_min=0.95,
            dispersive_max=0.15,
            logical_frequency_bounds_mhz=[81, 99],
            operation_frequency_bounds_ghz=[4, 8],
            tracking_min=0.95,
            logical_subspace_min=0.95,
            readout_subspace_min=0.98,
            coupler_frequency_bounds_ghz=[7, 8.5],
            rail_ec_bounds_mhz=[100, 400],
            rail_ej_max_ghz=25,
            rail_ej_over_ec_min=40,
            coupler_ec_max_mhz=300,
            coupler_ej_max_ghz=None,
        ),
        resonator_inductances_nh=[device.Lr_nH] * 2,
        ej_search_bounds_ghz=[[9, 24]] * 4 + [[22, 34]],
        basis=[4, 3, 16, 3],
        audit_basis=[5, 4, 20, 4],
        readout_window_ghz=[4.5, 7],
        spectral_ceiling_ghz=18.0,
        sweep=None,
    )


class WorkflowTests(unittest.TestCase):
    def test_requires_direct_input_and_rejects_ambiguous_rate_and_capacitor(self):
        config = inputs()
        self.assertEqual(validate(config), [])
        config["direct_a2_b1"]["bare_exchange_mhz"] = 0.1
        self.assertTrue(any("exactly one" in e for e in validate(config)))
        config["pair"] = None
        self.assertTrue(any("externally readout" in e for e in validate(config)))

    def test_invalid_bounds_and_candidate_path_are_rejected(self):
        config = inputs()
        config["ej_search_bounds_ghz"][0] = [20, 10]
        config["coupler_candidates"][0]["id"] = "../outside"
        errors = validate(config)
        self.assertTrue(any("lower" in e for e in errors))
        self.assertTrue(any("Candidate ids" in e for e in errors))

    def test_fingerprint_changes_with_numerical_and_physical_inputs(self):
        config = inputs()
        other = copy.deepcopy(config)
        other["audit_basis"][0] += 1
        self.assertNotEqual(fingerprint(config), fingerprint(other))

    def test_rate_to_capacitance_roundtrip_and_ej_independence(self):
        from coupler_turn_off.circuit import capacitance_matrix
        from scipy.constants import e, h

        from readout_workflow.calculation import Basis, _transmon

        config = inputs()
        point = config["pair"]
        matrix, _ = capacitance_matrix(point, 0, 0.28, 0.12)
        basis = Basis(25, 8, 3, 12)
        charges = [
            abs(
                _transmon(
                    point[f"ej{i}_ghz"], point[f"loaded_ec{i}_mhz"] / 1000, basis
                )[1][0, 1]
            )
            for i in (1, 2)
        ]
        rate = abs(
            1000
            * 4
            * e**2
            * np.linalg.inv(matrix * 1e-15)[1, 4]
            * np.prod(charges)
            / (h * 1e9)
        )
        config["direct_a2_b1"] = dict(capacitance_ff=None, bare_exchange_mhz=rate)
        self.assertAlmostEqual(resolve_direct_capacitance(config), 0.12, places=9)
        with threadpool_limits(limits=1):
            case = Case(config, config["coupler_candidates"][0])
            first = case.model([14, 12, 14, 12, 27])
            second = case.model([14.1, 12.2, 13.9, 12.1, 28])
        np.testing.assert_array_equal(first.cap, second.cap)
        self.assertIsNone(config["screens"]["coupler_ej_max_ghz"])
        self.assertEqual(case.bounds[4, 1], 34)

    def test_objectives_are_distinct_and_combined_has_equal_weights(self):
        result = dict(
            physical_signed_exchange_mhz=-9,
            energy_basis_logical_exchange_mhz=3,
            energy_basis_logical_zz_mhz=4,
        )
        self.assertEqual(objective_mhz(result, "physical"), 9)
        self.assertEqual(objective_mhz(result, "logical"), 3)
        self.assertEqual(objective_mhz(result, "combined"), 5)
        with self.assertRaises(ValueError):
            objective_mhz(result, "implicit")

    def test_cli_requires_explicit_objective_before_reading_or_running(self):
        from coupler_turn_off.__main__ import main

        with patch(
            "sys.argv",
            [
                "workflow",
                "optimize",
                "--config",
                "missing.json",
                "--candidate",
                "geometry",
            ],
        ):
            with self.assertRaises(SystemExit) as exc:
                main()
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
