"""Verify physical capacitance budgeting and the disconnected logical limit."""

import unittest

import numpy as np
from coupler_turn_off.circuit import (
    BudgetModel,
    capacitance_matrix,
    restricted_hamiltonian,
)
from coupler_turn_off.dispersive import network_dispersive
from scipy.constants import elementary_charge as e
from scipy.constants import h
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from readout_workflow.calculation import Basis, Device, ReadoutModel, annotate


class CouplerBudgetTests(unittest.TestCase):
    def test_restricted_hamiltonian_matches_explicit_kronecker_matrix(self):
        rng = np.random.default_rng(42)
        dims = (3, 2, 3)
        operators = []
        for dim in dims:
            a = rng.normal(size=(dim, dim))
            operators.append(a + a.T)
        blocks = [([i], [op]) for i, op in enumerate(operators)]
        coeff = rng.normal(size=(3, 3))
        coeff = (coeff + coeff.T) / 2
        energies = np.arange(np.prod(dims)) * 0.07
        full = np.diag(energies)
        for left in range(3):
            for right in range(left + 1, 3):
                factors = [np.eye(dim) for dim in dims]
                factors[left] = operators[left]
                factors[right] = operators[right]
                full += coeff[left, right] * np.kron(
                    np.kron(factors[0], factors[1]), factors[2]
                )
        indices = np.array([0, 2, 3, 7, 10, 16])
        np.testing.assert_allclose(
            restricted_hamiltonian(energies, blocks, dims, coeff, indices),
            full[np.ix_(indices, indices)],
            atol=1e-13,
            rtol=0,
        )

    def setUp(self):
        device = Device(
            C1_shunt_fF=90,
            C2_shunt_fF=75,
            C12_fF=2.047654044728067,
            C1r_fF=7.753209768978391,
            C2r_fF=7.753209768978391,
        )
        ec = ReadoutModel(device).ec_ghz
        self.point = dict(
            device=vars(device),
            ej1_ghz=13.256379949548528,
            ej2_ghz=11.44217292117331,
            loaded_ec1_mhz=1000 * ec[0],
            loaded_ec2_mhz=1000 * ec[1],
        )

    def test_preserves_loaded_ec_and_positive_physical_shunts(self):
        matrix, shunt = capacitance_matrix(self.point, 3, 0.3)
        ec = e**2 * np.diag(np.linalg.inv(matrix * 1e-15)) / (2 * h * 1e9)
        target = [
            self.point["loaded_ec1_mhz"] / 1000,
            self.point["loaded_ec2_mhz"] / 1000,
        ]
        np.testing.assert_allclose(ec[[0, 1, 4, 5]], target * 2, atol=1e-12)
        self.assertAlmostEqual(ec[3], 0.3, places=12)
        np.testing.assert_allclose(matrix.sum(axis=1)[3], shunt)
        self.assertTrue(np.all(matrix.sum(axis=1) > 0))
        self.assertLess(matrix.sum(axis=1)[1], 75)
        self.assertLess(matrix.sum(axis=1)[4], 90)
        # The inverse matrix induces an A2--B1 term even without a direct capacitor.
        self.assertEqual(matrix[1, 4], 0)
        self.assertGreater(np.linalg.inv(matrix)[1, 4], 0)

    def test_disconnected_pairs_have_zero_nonlocal_interaction(self):
        with threadpool_limits(limits=1):
            result = BudgetModel(self.point, 0, 0.3).evaluate(20)
        self.assertLess(result["residual_mhz"], 1e-8)
        self.assertLess(result["nonlocal_pauli_norm_mhz"], 1e-8)
        self.assertGreater(result["sigma_min"], 0.999999)
        self.assertLess(
            np.max(np.abs(result["single_excitation_exchange_matrix_mhz"])), 1e-8
        )

    def test_equal_exchange_uses_different_link_capacitors(self):
        with threadpool_limits(limits=1):
            model = BudgetModel(self.point, 1, 0.6)
            result = model.evaluate(24.9999)
        self.assertGreater(result["link_c_b1_ff"], result["link_a2_c_ff"])
        self.assertAlmostEqual(result["g_a2_c_mhz"], result["g_c_b1_mhz"], places=9)

    def test_direct_links_preserve_ec_and_produce_above_rail_zeros(self):
        for direct in (0.1, 0.2):
            with self.subTest(direct=direct), threadpool_limits(limits=1):
                model = BudgetModel(self.point, 3, 0.3, direct_ff=direct)
                self.assertAlmostEqual(model.cap[1, 4], -direct, places=12)
                target = [
                    self.point["loaded_ec1_mhz"],
                    self.point["loaded_ec2_mhz"],
                ] * 2
                np.testing.assert_allclose(
                    1000 * model.ec[[0, 1, 4, 5]], target, atol=1e-9
                )
                low = model.physical_exchange(13)
                high = model.physical_exchange(24.9999)
                self.assertLess(low["physical_signed_exchange_mhz"], 0)
                self.assertGreater(high["physical_signed_exchange_mhz"], 0)
                ej = brentq(
                    lambda x, model=model: model.physical_exchange(x)[
                        "physical_signed_exchange_mhz"
                    ],
                    13,
                    24.9999,
                    xtol=1e-9,
                )
                root = model.physical_exchange(ej)
                self.assertLess(abs(root["physical_signed_exchange_mhz"]), 1e-6)
                self.assertGreater(root["coupler_f01_ghz"], max(root["rail_f01_ghz"]))
                self.assertGreater(root["physical_sigma_min"], 0.9)

    def test_cancellation_above_7p5_with_relaxed_coupler_ej(self):
        for direct, bracket in ((0.1, (3.5, 4.0)), (0.2, (5.0, 5.7))):
            with self.subTest(direct=direct), threadpool_limits(limits=1):
                ec, ej = 0.285, 28.74861170457883

                def residual(cap, direct=direct, ec=ec, ej=ej):
                    return BudgetModel(
                        self.point, cap, ec, direct_ff=direct
                    ).physical_exchange(ej)["physical_signed_exchange_mhz"]

                cap = brentq(residual, *bracket, xtol=1e-9)
                model = BudgetModel(self.point, cap, ec, direct_ff=direct)
                result = model.physical_exchange(ej)
                self.assertLess(ej, 30)
                self.assertLess(ec, 0.3)
                self.assertGreater(result["coupler_f01_ghz"], 7.5)
                self.assertAlmostEqual(result["coupler_f01_ghz"], 7.8, places=8)
                self.assertLess(abs(result["physical_signed_exchange_mhz"]), 1e-6)

    def test_ej_recalibration_keeps_fabricated_capacitances_fixed(self):
        with threadpool_limits(limits=1):
            nominal = BudgetModel(self.point, 3, 0.285, direct_ff=0.1)
            ej = [
                self.point["ej1_ghz"] * 1.001,
                self.point["ej2_ghz"],
                self.point["ej1_ghz"] * 0.999,
                self.point["ej2_ghz"] * 1.001,
            ]
            trimmed = BudgetModel(
                self.point,
                3,
                0.285,
                direct_ff=0.1,
                fixed_links_ff=nominal.links,
                rail_ej_ghz=ej,
            )
        np.testing.assert_allclose(trimmed.cap, nominal.cap, atol=1e-12, rtol=0)
        np.testing.assert_allclose(trimmed.ec, nominal.ec, atol=1e-12, rtol=0)
        np.testing.assert_allclose(trimmed.rail_ej_ghz, ej)
        self.assertNotAlmostEqual(
            trimmed.pairs[0][0][1], trimmed.pairs[1][0][1], places=6
        )

    def test_parity_low_spectrum_matches_full_diagonalization(self):
        with threadpool_limits(limits=1):
            model = BudgetModel(self.point, 3, 0.285, direct_ff=0.1)
            full = model.evaluate(28)
            model.eigenenergy_max_ghz = 18
            fast = model.evaluate(28)
        for key in (
            "exchange_mhz",
            "zz_mhz",
            "residual_mhz",
            "sigma_min",
            "leakage_gap_mhz",
        ):
            self.assertAlmostEqual(fast[key], full[key], places=7, msg=key)

    def test_detuned_readout_matches_independent_isolated_pairs(self):
        basis = Basis(25, 4, 3, 16)
        inductances = [
            self.point["device"]["Lr_nH"] * (6 / f) ** 2 for f in (6.01, 5.99)
        ]
        with threadpool_limits(limits=1):
            model = BudgetModel(
                self.point,
                0,
                0.285,
                4,
                3,
                16,
                3,
                eigenenergy_max_ghz=18,
                resonator_inductances_nh=inductances,
            )
            result = model.evaluate(28, readout=True)
            for side, inductance in zip(("A", "B"), inductances, strict=True):
                device = dict(self.point["device"], Lr_nH=inductance)
                expected = ReadoutModel(Device(**device), basis).evaluate(
                    self.point["ej1_ghz"], self.point["ej2_ghz"]
                )
                for check in result["energy_mode_readout_checks"]:
                    if check["resonator"] == side:
                        self.assertAlmostEqual(
                            check["chi_minus_mhz"],
                            1000 * expected["chi_m_ghz"],
                            places=6,
                        )
                        self.assertAlmostEqual(
                            check["resonator_frequency_ghz"],
                            expected["dressed_r_ghz"],
                            places=6,
                        )

    def test_network_dispersive_recovers_original_screen_without_links(self):
        ej = [self.point["ej1_ghz"], self.point["ej2_ghz"]] * 2
        ej[2:] = [value * 1.003 for value in ej[2:]]
        with threadpool_limits(limits=1):
            model = BudgetModel(self.point, 0, 0.285, rail_ej_ghz=ej)
            actual = network_dispersive(model, 28, levels=6, pair_keep=10, c_levels=4)
            for i, report in enumerate(actual):
                isolated = ReadoutModel(
                    Device(**self.point["device"]), Basis(25, 6, 4, 30)
                )
                expected = annotate(isolated, *ej[2 * i : 2 * i + 2])
                self.assertAlmostEqual(
                    report["dispersive_ratio"], expected["lambda_max"], places=8
                )

    def test_energy_mode_readout_reduces_to_isolated_pair_without_links(self):
        basis = Basis(25, 4, 3, 16)
        with threadpool_limits(limits=1):
            model = BudgetModel(
                self.point, 0, 0.285, 4, 3, 16, 3, eigenenergy_max_ghz=18
            )
            result = model.evaluate(28, readout=True)
            expected = ReadoutModel(Device(**self.point["device"]), basis).evaluate(
                self.point["ej1_ghz"], self.point["ej2_ghz"]
            )
        for check in result["energy_mode_readout_checks"]:
            self.assertAlmostEqual(
                check["chi_minus_mhz"], 1000 * expected["chi_m_ghz"], places=6
            )
            self.assertAlmostEqual(
                check["chi_plus_mhz"], 1000 * expected["chi_p_ghz"], places=6
            )
            self.assertAlmostEqual(
                check["hybridization"], expected["hybridization"], places=6
            )


if __name__ == "__main__":
    unittest.main()
