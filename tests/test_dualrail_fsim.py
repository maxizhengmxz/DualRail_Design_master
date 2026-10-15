import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.linalg import expm

from dtc_dualrail.dualrail_fsim import (
    DYNAMIC_DUALRAIL_FSIM_SCHEMA,
    LOGICAL_STATE_LABELS,
    POPULATION_TRAJECTORY_SCHEMA,
    STATIC_DUALRAIL_FSIM_SCHEMA,
    DualRailCouplingConfig,
    DualRailPulse,
    DualRailPulseConstraints,
    FiveTransmonConfig,
    LogicalContour,
    LogicalModeMapping,
    StaticCoherentCeilingResult,
    assign_dressed_logical_states,
    average_gate_fidelity,
    build_dualrail_fsim_model,
    build_logical_contour,
    calibrate_reduced_pair,
    closest_unitary,
    compare_static_and_dynamic_rates,
    dualrail_idle_score,
    evaluate_static_coherent_ceiling,
    extract_fsim_angles,
    load_or_compute_static_interaction_landscape,
    local_z_optimized_fidelity,
    pauli_decompose,
    rank_static_gate_candidates,
    refine_static_gate_candidates,
    run_static_architecture_loop,
    run_static_interaction_landscape,
    select_dualrail_idle_point,
    track_logical_branches,
    validate_fixed_basis_along_path,
)
from dtc_dualrail.fsim import SolverConfig, TunableTransmonConfig, ideal_fsim


def contour_fixture(frequency=(90.0, 100.0, 110.0)):
    frequency = np.asarray(frequency, dtype=float)
    count = frequency.size
    return xr.Dataset(
        data_vars={
            "phi_1": ("operating_point", np.linspace(0.1, 0.2, count)),
            "phi_2": ("operating_point", np.linspace(0.2, 0.3, count)),
            "splitting_ghz": ("operating_point", frequency / 1e3),
            "operating_frequency_ghz": (
                "operating_point",
                np.linspace(4.8, 4.7, count),
            ),
            "hybridization": ("operating_point", np.full(count, 0.99)),
            "polarization": ("operating_point", np.zeros(count)),
        },
        attrs={"flux_param_1": "Phi_A1", "flux_param_2": "Phi_A2"},
    )


def tiny_config(*, inter_pair=True):
    def transmon(EJmax, EC, *, flux=0.0):
        return TunableTransmonConfig(
            EJmax=EJmax,
            EC=EC,
            d=0.15,
            flux=flux,
            ncut=4,
            truncated_dim=2,
        )

    coupling = DualRailCouplingConfig(
        a1_a2=0.05,
        a2_c=0.04 if inter_pair else 0.0,
        c_b1=0.04 if inter_pair else 0.0,
        b1_b2=0.05,
    )
    return FiveTransmonConfig(
        rail_a1=transmon(12.0, 0.24),
        rail_a2=transmon(11.8, 0.24),
        coupler=transmon(18.0, 0.5, flux=0.2),
        rail_b1=transmon(11.4, 0.24),
        rail_b2=transmon(11.2, 0.24),
        coupling=coupling,
        parameter_source="deterministic unit-test values",
    )


def pulse_contours(config):
    def contour(rail_1, rail_2):
        return LogicalContour(
            logical_frequency_mhz=np.asarray([90.0, 100.0, 110.0]),
            phi_1=np.asarray([rail_1.flux - 0.01, rail_1.flux, rail_1.flux + 0.01]),
            phi_2=np.asarray([rail_2.flux - 0.01, rail_2.flux, rail_2.flux + 0.01]),
            carrier_frequency_ghz=np.asarray([4.5, 4.5, 4.5]),
            hybridization=np.asarray([1.0, 1.0, 1.0]),
            polarization=np.asarray([0.0, 0.0, 0.0]),
        )

    return (
        contour(config.rail_a1, config.rail_a2),
        contour(config.rail_b1, config.rail_b2),
    )


class LogicalConventionTests(unittest.TestCase):
    def test_energy_and_character_mappings_are_explicit(self):
        mapping = LogicalModeMapping(
            logical_zero="upper",
            logical_one="lower",
            lower_character="-",
        )

        self.assertEqual(mapping.mode_for_bit(0), "upper")
        self.assertEqual(mapping.mode_for_bit(1), "lower")
        self.assertEqual(mapping.character_for_bit(0), "+")
        self.assertEqual(mapping.character_for_bit(1), "-")

    def test_mapping_rejects_duplicate_modes(self):
        with self.assertRaisesRegex(ValueError, "map lower and upper once"):
            LogicalModeMapping(logical_zero="lower", logical_one="lower").validate()


class LogicalContourTests(unittest.TestCase):
    def test_interpolation_inverse_and_derivatives(self):
        contour = build_logical_contour(contour_fixture())

        phi_1, phi_2 = contour.flux(95.0)
        self.assertAlmostEqual(float(phi_1), 0.125)
        self.assertAlmostEqual(float(phi_2), 0.225)
        self.assertAlmostEqual(contour.inverse_contour(phi_1, phi_2), 95.0, places=4)
        dataset = contour.to_dataset()
        self.assertIn("dphi_1_df", dataset)
        self.assertEqual(dataset.phi_1.attrs["units"], "flux quantum")

    def test_fold_requires_an_explicit_segment(self):
        dataset = contour_fixture((90.0, 100.0, 110.0, 100.0, 90.0))

        with self.assertRaisesRegex(ValueError, "folded into 2"):
            build_logical_contour(dataset)

        first = build_logical_contour(dataset, segment=0)
        second = build_logical_contour(dataset, segment=1)
        np.testing.assert_allclose(first.logical_frequency_mhz, [90, 100, 110])
        np.testing.assert_allclose(second.logical_frequency_mhz, [90, 100, 110])

    def test_rejects_uncovered_frequency_band(self):
        with self.assertRaisesRegex(ValueError, r"not \[90, 150\]"):
            build_logical_contour(
                contour_fixture(),
                frequency_range_mhz=(90.0, 150.0),
            )

    def test_requested_band_inserts_exact_interpolated_boundaries(self):
        contour = build_logical_contour(
            contour_fixture(),
            frequency_range_mhz=(92.5, 107.5),
        )

        self.assertEqual(contour.frequency_range_mhz, (92.5, 107.5))
        phi_1, phi_2 = contour.flux([92.5, 107.5])
        np.testing.assert_allclose(phi_1, [0.1125, 0.1875])
        np.testing.assert_allclose(phi_2, [0.2125, 0.2875])

    def test_rejects_disconnected_path(self):
        dataset = contour_fixture((90.0, 95.0, 100.0, 105.0))
        dataset["phi_1"] = ("operating_point", [0.1, 0.11, 0.5, 0.51])
        dataset["phi_2"] = ("operating_point", [0.2, 0.21, 0.6, 0.61])

        with self.assertRaisesRegex(ValueError, "disconnected"):
            build_logical_contour(dataset)


class LogicalAssignmentTests(unittest.TestCase):
    def test_hungarian_assignment_is_global_and_distinct(self):
        references = np.eye(3, dtype=complex)
        eigenstates = references[:, [2, 0, 1]].T

        assignment, probability = assign_dressed_logical_states(
            references,
            eigenstates,
        )

        np.testing.assert_array_equal(assignment, [1, 2, 0])
        np.testing.assert_allclose(probability, 1.0)

    def test_tracks_diabatic_and_phase_aligned_adiabatic_labels_separately(self):
        references = np.eye(3, dtype=complex)[:, :2].T
        leakage = np.eye(3, dtype=complex)[:, 2:3].T
        state_table = []
        for angle, phases in (
            (0.0, (0.0, 0.0)),
            (np.deg2rad(40), (0.3, -0.2)),
            (np.deg2rad(80), (0.7, -0.5)),
        ):
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=complex,
            )
            rotation[:, 0] *= np.exp(1j * phases[0])
            rotation[:, 1] *= np.exp(1j * phases[1])
            state_table.append(rotation)

        result = track_logical_branches(
            references,
            state_table,
            leakage_reference_states=leakage,
            leakage_labels=["leak"],
            minimum_continuity_overlap=0.5,
        )

        np.testing.assert_array_equal(result.diabatic_level[-1], [1, 0])
        np.testing.assert_array_equal(result.adiabatic_level, [[0, 1], [0, 1], [0, 1]])
        self.assertTrue(np.all(result.adiabatic_overlap[1:] > 0.5))
        self.assertTrue(np.all(result.logical_weight > 0.0))
        self.assertEqual(result.dominant_leakage_label.shape, (3, 2))
        self.assertTrue(result.continuity_valid)
        report = result.to_dataset()
        self.assertIn("phase_alignment_rad", report)
        self.assertIn("dominant_leakage_probability", report)


class GateMetricTests(unittest.TestCase):
    def test_static_logical_propagators_project_full_target_dynamics(self):
        model = build_dualrail_fsim_model(tiny_config())
        duration_ns = np.asarray([0.0, 0.2])

        actual = model.static_logical_propagators(duration_ns)

        evolution = (-2j * np.pi * model.hamiltonian_ghz * duration_ns[1]).expm()
        expected = np.asarray(
            [
                [left.overlap(evolution * right) for right in model.logical_references]
                for left in model.logical_references
            ]
        )
        np.testing.assert_allclose(actual[0], np.eye(4), atol=1e-12)
        np.testing.assert_allclose(actual[1], expected, atol=1e-12)

    def test_pauli_decomposition_recovers_known_term(self):
        z = np.diag([1.0, -1.0])

        coefficients = pauli_decompose(np.kron(z, z))

        self.assertAlmostEqual(coefficients["ZZ"], 1.0)
        for term, value in coefficients.items():
            if term != "ZZ":
                self.assertAlmostEqual(value, 0.0)

    def test_known_fsim_matrix_recovers_angles(self):
        expected_theta = np.pi / 3
        expected_phi = -np.pi / 4

        theta, phi, swap_probability = extract_fsim_angles(
            ideal_fsim(expected_theta, expected_phi)
        )

        self.assertAlmostEqual(theta, expected_theta)
        self.assertAlmostEqual(phi, expected_phi)
        self.assertAlmostEqual(swap_probability, np.sin(expected_theta) ** 2)

    def test_polar_factor_separates_uniform_leakage(self):
        target = ideal_fsim(np.pi / 2, 0.0)
        projected = np.sqrt(0.98) * target

        polar = closest_unitary(projected)

        np.testing.assert_allclose(polar, target, atol=1e-12)
        self.assertAlmostEqual(average_gate_fidelity(polar, target), 1.0)
        self.assertAlmostEqual(average_gate_fidelity(projected, target), 0.98)

    def test_local_z_optimization_removes_virtual_z_phases(self):
        target = ideal_fsim(np.pi / 2, 0.0)
        before = np.diag(np.exp(1j * np.asarray([0.2, -0.3, 0.3, -0.2])))
        after = np.diag(np.exp(1j * np.asarray([-0.1, 0.4, -0.4, 0.1])))
        actual = after @ target @ before

        fidelity, _ = local_z_optimized_fidelity(actual, target)

        self.assertAlmostEqual(fidelity, 1.0, places=9)

    def test_static_coherent_ceiling_finds_iswap_and_cz_times(self):
        iswap_hamiltonian = np.zeros((4, 4), dtype=complex)
        iswap_hamiltonian[1, 2] = iswap_hamiltonian[2, 1] = 0.001
        cz_hamiltonian = np.diag([0.0, 0.0, 0.0, 0.001])

        iswap = evaluate_static_coherent_ceiling(
            iswap_hamiltonian,
            "iswap",
            maximum_duration_ns=300.0,
            time_samples=31,
        )
        cz = evaluate_static_coherent_ceiling(
            cz_hamiltonian,
            "cz",
            maximum_duration_ns=600.0,
            time_samples=31,
        )

        self.assertIsInstance(iswap, StaticCoherentCeilingResult)
        self.assertAlmostEqual(iswap.fidelity, 1.0, places=8)
        self.assertAlmostEqual(iswap.duration_ns, 250.0, places=5)
        self.assertAlmostEqual(cz.fidelity, 1.0, places=8)
        self.assertAlmostEqual(cz.duration_ns, 500.0, places=5)
        self.assertEqual(iswap.virtual_z_rad.shape, (4,))

    def test_flat_top_dynamic_rates_recover_rates_with_static_nuisance_terms(self):
        exchange_mhz = 2.5
        zz_mhz = -1.2
        duration_ns = np.linspace(0.0, 80.0, 33)
        static_hamiltonian_ghz = np.asarray(
            [
                [0.0, 0.001, 0.0, 0.0],
                [0.001, 0.007, exchange_mhz / 1e3, 0.0],
                [0.0, exchange_mhz / 1e3, -0.004, 0.0007],
                [0.0, 0.0, 0.0007, (zz_mhz + 3.0) / 1e3],
            ],
            dtype=complex,
        )
        unitaries = np.asarray(
            [
                expm(-2j * np.pi * static_hamiltonian_ghz * duration)
                for duration in duration_ns
            ]
        )

        result = compare_static_and_dynamic_rates(
            duration_ns,
            unitaries,
            static_hamiltonian_ghz=static_hamiltonian_ghz,
        )

        self.assertAlmostEqual(result.dynamic_exchange_mhz, exchange_mhz, places=6)
        self.assertAlmostEqual(result.dynamic_zz_mhz, zz_mhz, places=6)
        self.assertLess(result.exchange_relative_error, 0.05)
        self.assertLess(result.zz_relative_error, 0.05)
        self.assertTrue(result.rates_agree)

    def test_rate_gate_normalizes_small_nuisance_to_relevant_gate_rate(self):
        duration_ns = np.linspace(0.0, 160.0, 17)
        static_hamiltonian = np.zeros((4, 4), dtype=complex)
        static_hamiltonian[1, 2] = static_hamiltonian[2, 1] = 0.04e-3
        static_hamiltonian[3, 3] = -3.2e-3
        dynamic_hamiltonian = static_hamiltonian.copy()
        dynamic_hamiltonian[1, 2] = dynamic_hamiltonian[2, 1] = 0.07e-3
        unitaries = np.asarray(
            [
                expm(-2j * np.pi * dynamic_hamiltonian * duration)
                for duration in duration_ns
            ]
        )

        result = compare_static_and_dynamic_rates(
            duration_ns,
            unitaries,
            static_hamiltonian_ghz=static_hamiltonian,
            normalization_rate_mhz=3.2,
        )

        self.assertAlmostEqual(result.dynamic_exchange_mhz, 0.07, places=6)
        self.assertEqual(result.normalization_rate_mhz, 3.2)
        self.assertLess(result.exchange_relative_error, 0.05)
        self.assertTrue(result.rates_agree)


class FixedBasisValidationTests(unittest.TestCase):
    def test_overlap_matches_logical_energies_and_leakage_gaps_along_path(self):
        config = tiny_config()
        reference = build_dualrail_fsim_model(config)

        result = validate_fixed_basis_along_path(
            reference,
            [config, config],
            path_coordinate=[0.0, 1.0],
            levels=12,
            relevant_rate_mhz=2.0,
        )

        self.assertEqual(result.sizes["path_point"], 2)
        self.assertEqual(result.sizes["matched_level"], 12)
        np.testing.assert_allclose(result.state_overlap, 1.0, atol=1e-10)
        np.testing.assert_allclose(result.energy_error_mhz, 0.0, atol=1e-9)
        np.testing.assert_allclose(result.logical_energy_error_mhz, 0.0, atol=1e-9)
        np.testing.assert_allclose(result.leakage_gap_error_mhz, 0.0, atol=1e-9)
        self.assertIn("closest_leakage_approach_path_point", result.attrs)
        self.assertEqual(result.attrs["overlap_matching"], "instantaneous-to-fixed")
        self.assertAlmostEqual(result.attrs["error_tolerance_mhz"], 0.2)


class ReducedStaticModelTests(unittest.TestCase):
    @staticmethod
    def idle_landscape_fixture():
        dimensions = (
            "coupler_flux",
            "logical_frequency_a_mhz",
            "logical_frequency_b_mhz",
        )
        return xr.Dataset(
            data_vars={
                "effective_exchange_mhz": (
                    dimensions,
                    [[[0.5], [0.1]], [[0.2], [0.05]]],
                ),
                "effective_zz_mhz": (
                    dimensions,
                    [[[0.1], [0.3]], [[0.2], [0.04]]],
                ),
                "logical_assignment_valid": (
                    dimensions,
                    [[[True], [True]], [[True], [False]]],
                ),
                "spectrally_isolated": (
                    dimensions,
                    np.ones((2, 2, 1), dtype=bool),
                ),
            },
            coords={
                "coupler_flux": [0.2, 0.3],
                "logical_frequency_a_mhz": [95.0, 100.0],
                "logical_frequency_b_mhz": [97.0],
            },
        )

    def test_idle_score_uses_exchange_matrix_element_and_dressed_zz(self):
        landscape = self.idle_landscape_fixture()

        score = dualrail_idle_score(landscape)
        qualified = select_dualrail_idle_point(landscape)
        unrestricted = select_dualrail_idle_point(
            landscape,
            require_qualified=False,
        )

        np.testing.assert_allclose(
            score,
            np.hypot(
                landscape.effective_exchange_mhz,
                landscape.effective_zz_mhz,
            ),
        )
        self.assertAlmostEqual(float(qualified.coupler_flux), 0.3)
        self.assertAlmostEqual(float(qualified.logical_frequency_a_mhz), 95.0)
        self.assertAlmostEqual(float(unrestricted.logical_frequency_a_mhz), 100.0)
        self.assertIn("idle_residual_coupling_mhz", qualified)

    def test_idle_selection_rejects_missing_valid_candidate(self):
        landscape = self.idle_landscape_fixture()
        landscape.logical_assignment_valid[:] = False

        with self.assertRaisesRegex(ValueError, "no qualified idle candidate"):
            select_dualrail_idle_point(landscape)
        with self.assertRaisesRegex(ValueError, "weights"):
            dualrail_idle_score(landscape, exchange_weight=0.0, zz_weight=0.0)

    def test_reduced_pair_calibration_matches_carrier_and_splitting(self):
        contour = build_logical_contour(contour_fixture())

        calibration = calibrate_reduced_pair(
            contour,
            100.0,
            ejmax_ghz=(18.0, 18.0),
            asymmetry=(0.1, 0.1),
            ncut=8,
            truncated_dim=3,
        )

        self.assertAlmostEqual(calibration.achieved_logical_frequency_mhz, 100.0)
        self.assertAlmostEqual(calibration.anchor_carrier_frequency_ghz, 4.75)
        self.assertGreater(calibration.coupling_ghz, 0.0)

    def test_model_assigns_four_states_and_reports_static_gate_metrics(self):
        model = build_dualrail_fsim_model(tiny_config())
        result = model.static_interaction()

        self.assertEqual(result.projected_hamiltonian_ghz.shape, (4, 4))
        self.assertEqual(result.effective_hamiltonian_ghz.shape, (4, 4))
        np.testing.assert_allclose(
            result.effective_hamiltonian_ghz,
            result.effective_hamiltonian_ghz.conj().T,
            atol=1e-12,
        )
        self.assertEqual(result.logical_energy_ghz.shape, (4,))
        self.assertEqual(result.logical_assignment_probability.shape, (4,))
        self.assertEqual(result.logical_singular_values.shape, (4,))
        self.assertAlmostEqual(
            result.logical_subspace_sigma_min,
            float(np.min(result.logical_singular_values)),
        )
        self.assertTrue(np.all(np.isfinite(result.logical_energy_ghz)))
        self.assertGreaterEqual(result.nearest_leakage_gap_mhz, 0.0)
        self.assertIsInstance(result.logical_assignment_valid, bool)

    def test_zero_inter_pair_coupling_has_no_gate_action(self):
        result = build_dualrail_fsim_model(
            tiny_config(inter_pair=False)
        ).static_interaction()

        self.assertAlmostEqual(result.direct_exchange_mhz, 0.0, places=9)
        self.assertAlmostEqual(result.effective_exchange_mhz, 0.0, places=7)
        self.assertAlmostEqual(result.logical_exchange_mhz, 0.0, places=7)
        self.assertAlmostEqual(result.effective_zz_mhz, 0.0, places=7)
        self.assertAlmostEqual(result.logical_zz_mhz, 0.0, places=7)

    def test_landscape_coordinates_metrics_and_fingerprint_cache(self):
        contour = build_logical_contour(contour_fixture())
        config = tiny_config()
        grids = {
            "logical_frequency_a_mhz": [105.0],
            "logical_frequency_b_mhz": [90.0],
            "coupler_flux_values": [0.2],
        }

        dataset = run_static_interaction_landscape(
            config,
            contour,
            contour,
            **grids,
        )

        self.assertEqual(
            dataset.logical_energy_ghz.dims,
            (
                "coupler_flux",
                "logical_frequency_a_mhz",
                "logical_frequency_b_mhz",
                "logical_state",
            ),
        )
        self.assertEqual(
            dataset.logical_state.values.tolist(), list(LOGICAL_STATE_LABELS)
        )
        self.assertEqual(
            dataset.projected_hamiltonian_real_ghz.shape,
            (1, 1, 1, 4, 4),
        )
        self.assertEqual(
            dataset.effective_hamiltonian_real_ghz.shape,
            (1, 1, 1, 4, 4),
        )
        self.assertIn("logical_assignment_valid", dataset)
        self.assertIn("logical_subspace_sigma_min", dataset)
        self.assertIn("model_logical_detuning_mhz", dataset)
        self.assertEqual(dataset.logical_singular_values.shape, (1, 1, 1, 4))
        self.assertEqual(dataset.pauli_coefficient_real_mhz.shape, (1, 1, 1, 16))
        self.assertEqual(dataset.pauli_term.size, 16)
        for name in (
            "iswap_exchange_component_mhz",
            "iswap_pair_creation_component_mhz",
            "iswap_nuisance_pauli_norm_mhz",
            "cz_offdiagonal_pauli_norm_mhz",
            "cz_offdiagonal_to_zz_ratio",
            "maximum_coupler_participation",
            "iswap_static_coherent_ceiling",
            "iswap_static_coherent_time_ns",
            "cz_static_coherent_ceiling",
            "cz_static_coherent_time_ns",
            "nearest_leakage_level",
            "nearest_leakage_logical_state",
            "nearest_leakage_branch_label",
            "nearest_leakage_branch_probability",
        ):
            self.assertIn(name, dataset)
        self.assertTrue(dataset.nearest_leakage_branch_label.item())
        self.assertAlmostEqual(dataset.logical_detuning_mhz.item(), 15.0)
        self.assertEqual(
            dataset.attrs["static_dualrail_fsim_schema"],
            STATIC_DUALRAIL_FSIM_SCHEMA,
        )
        refined = refine_static_gate_candidates(
            config,
            contour,
            contour,
            dataset,
            "iswap",
            seed_candidates=1,
            minimum_coherent_ceiling=0.0,
            maximum_candidates=1,
            minimum_sigma_min=0.0,
            minimum_leakage_gap_mhz=0.0,
        )
        self.assertEqual(refined.attrs["adaptive_refined_regions"], 1)
        self.assertEqual(refined.sizes["candidate"], 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "static.nc"
            computed, _ = load_or_compute_static_interaction_landscape(
                config,
                contour,
                contour,
                path=path,
                force=True,
                **grids,
            )
            loaded, _ = load_or_compute_static_interaction_landscape(
                config,
                contour,
                contour,
                path=path,
                **grids,
            )
            np.testing.assert_allclose(loaded.logical_zz_mhz, computed.logical_zz_mhz)
            loaded.close()

            changed_grids = {**grids, "coupler_flux_values": [0.21]}
            with self.assertRaisesRegex(ValueError, "force=True"):
                load_or_compute_static_interaction_landscape(
                    config,
                    contour,
                    contour,
                    path=path,
                    **changed_grids,
                )

    def test_parallel_landscape_matches_serial_result(self):
        contour = build_logical_contour(contour_fixture())
        config = tiny_config()
        grids = {
            "logical_frequency_a_mhz": [95.0, 100.0],
            "logical_frequency_b_mhz": [100.0],
            "coupler_flux_values": [0.2, 0.21],
        }

        serial = run_static_interaction_landscape(
            config,
            contour,
            contour,
            cpus=1,
            **grids,
        )
        parallel = run_static_interaction_landscape(
            config,
            contour,
            contour,
            cpus=2,
            **grids,
        )

        for name in serial.data_vars:
            if serial[name].dtype.kind in "OUS":
                np.testing.assert_array_equal(parallel[name], serial[name])
            else:
                np.testing.assert_allclose(parallel[name], serial[name], atol=1e-12)
        self.assertEqual(serial.attrs["worker_count"], 1)
        self.assertEqual(parallel.attrs["worker_count"], 2)

    def test_gate_candidates_are_pareto_filtered_and_hard_ceiling_gated(self):
        point = "atlas_point"
        landscape = xr.Dataset(
            data_vars={
                "logical_assignment_valid": (point, [True, True, True]),
                "spectrally_isolated": (point, [True, True, True]),
                "logical_subspace_sigma_min": (point, [0.99, 0.98, 0.99]),
                "nearest_leakage_gap_mhz": (point, [20.0, 30.0, 40.0]),
                "nearest_leakage_branch_label": (point, ["L0", "L1", "L2"]),
                "maximum_coupler_participation": (point, [0.1, 0.2, 0.05]),
                "model_logical_detuning_mhz": (point, [0.0, 0.1, 0.0]),
                "effective_exchange_mhz": (point, [2.0, 3.0, 4.0]),
                "effective_zz_mhz": (point, [0.1, 0.1, 0.1]),
                "iswap_exchange_component_mhz": (point, [1.0, 1.5, 2.0]),
                "iswap_pair_creation_component_mhz": (point, [0.01, 0.02, 0.01]),
                "iswap_nuisance_pauli_norm_mhz": (point, [0.1, 0.2, 0.05]),
                "cz_offdiagonal_pauli_norm_mhz": (point, [1.0, 1.0, 1.0]),
                "cz_offdiagonal_to_zz_ratio": (point, [10.0, 10.0, 10.0]),
                "iswap_static_coherent_ceiling": (point, [0.999, 0.994, 0.999]),
                "cz_static_coherent_ceiling": (point, [0.9, 0.9, 0.9]),
            },
            coords={
                point: [0, 1, 2],
                "coupler_flux": (point, [0.2, 0.21, 0.22]),
                "logical_frequency_a_mhz": (point, [100.0, 101.0, 102.0]),
                "logical_frequency_b_mhz": (point, [100.0, 101.0, 102.0]),
            },
        )

        candidates = rank_static_gate_candidates(
            landscape,
            "iswap",
            minimum_coherent_ceiling=0.995,
            maximum_candidates=10,
        )

        self.assertEqual(candidates.sizes["candidate"], 1)
        self.assertEqual(candidates.atlas_point.item(), 2)
        self.assertEqual(candidates.gate_mechanism.item(), "exchange")
        self.assertEqual(candidates.limiting_leakage_branch.item(), "L2")
        self.assertGreaterEqual(candidates.static_coherent_ceiling.item(), 0.995)
        self.assertEqual(candidates.pareto_rank.item(), 0)

    def test_architecture_loop_tracks_cases_and_blocks_parasitics(self):
        config = tiny_config()
        contour_a, contour_b = pulse_contours(config)
        architecture = run_static_architecture_loop(
            {"baseline": (config, contour_a, contour_b)},
            [100.0],
            [100.0],
            [0.2],
            coherent_ceiling_max_time_ns=1.0,
            coherent_ceiling_time_samples=3,
        )

        self.assertEqual(architecture.architecture.values.tolist(), ["baseline"])
        self.assertTrue(architecture.architecture_fingerprint.item())
        parasitic = replace(
            config,
            coupling=replace(config.coupling, a2_b1=0.001),
        )
        with self.assertRaisesRegex(ValueError, "parasitic couplings"):
            run_static_architecture_loop(
                {"parasitic": (parasitic, contour_a, contour_b)},
                [100.0],
                [100.0],
                [0.2],
                coherent_ceiling_max_time_ns=1.0,
                coherent_ceiling_time_samples=3,
            )


class PulseEvolutionTests(unittest.TestCase):
    def test_rectangular_population_trajectory_tracks_five_outcomes(self):
        config = tiny_config()
        model = build_dualrail_fsim_model(config)
        contour_a, contour_b = pulse_contours(config)
        pulse = DualRailPulse(
            coupler_peak_flux=0.22,
            duration_ns=4.0,
            rise_time_ns=0.0,
            output_samples=11,
        )

        trajectory = model.simulate_trajectory(
            pulse,
            contour_a,
            contour_b,
            initial_state="10",
        )
        final_result = model.simulate(pulse, contour_a, contour_b)

        self.assertEqual(trajectory.initial_state, "10")
        self.assertEqual(trajectory.solver_method, "exact")
        self.assertEqual(trajectory.logical_population.shape, (11, 4))
        self.assertEqual(trajectory.leakage.shape, (11,))
        np.testing.assert_allclose(
            trajectory.logical_population[0],
            [0.0, 0.0, 1.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory.logical_population.sum(axis=1) + trajectory.leakage,
            np.ones(11),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory.logical_population[-1],
            np.abs(final_result.projected_unitary[:, 2]) ** 2,
            atol=1e-12,
        )
        dataset = trajectory.to_dataset()
        self.assertEqual(
            dataset.attrs["population_trajectory_schema"],
            POPULATION_TRAJECTORY_SCHEMA,
        )
        self.assertEqual(list(dataset.logical_state.values), list(LOGICAL_STATE_LABELS))

    def test_population_trajectory_includes_idle_padding_and_flux_return(self):
        config = tiny_config()
        model = build_dualrail_fsim_model(config)
        contour_a, contour_b = pulse_contours(config)
        pulse = DualRailPulse(
            coupler_peak_flux=0.22,
            duration_ns=4.0,
            padding_ns=2.0,
            rise_time_ns=0.0,
            output_samples=21,
        )

        trajectory = model.simulate_trajectory(
            pulse,
            contour_a,
            contour_b,
            initial_state="10",
        )
        final_result = model.simulate(pulse, contour_a, contour_b)
        idle_flux = np.asarray(
            [subsystem.flux for subsystem in config.subsystems()],
            dtype=float,
        )
        before_active = trajectory.time_ns <= pulse.padding_ns
        after_active = trajectory.time_ns >= pulse.padding_ns + pulse.duration_ns

        self.assertAlmostEqual(trajectory.time_ns[0], 0.0)
        self.assertAlmostEqual(trajectory.time_ns[-1], pulse.total_duration_ns)
        np.testing.assert_allclose(
            trajectory.subsystem_flux[before_active],
            np.broadcast_to(
                idle_flux,
                trajectory.subsystem_flux[before_active].shape,
            ),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory.subsystem_flux[after_active],
            np.broadcast_to(
                idle_flux,
                trajectory.subsystem_flux[after_active].shape,
            ),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            trajectory.logical_population[before_active],
            np.broadcast_to(
                [0.0, 0.0, 1.0, 0.0],
                trajectory.logical_population[before_active].shape,
            ),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            trajectory.logical_population[-1],
            np.abs(final_result.projected_unitary[:, 2]) ** 2,
            atol=1e-11,
        )

    def test_pulse_rejects_negative_idle_padding(self):
        with self.assertRaisesRegex(ValueError, "padding_ns"):
            DualRailPulse(
                coupler_peak_flux=0.22,
                duration_ns=4.0,
                padding_ns=-1.0,
            ).validate()

    def test_shaped_population_trajectory_uses_solver(self):
        config = tiny_config()
        model = build_dualrail_fsim_model(config)
        contour_a, contour_b = pulse_contours(config)
        pulse = DualRailPulse(
            coupler_peak_flux=0.22,
            logical_frequency_a_peak_mhz=105.0,
            logical_frequency_b_peak_mhz=95.0,
            duration_ns=2.0,
            rise_time_ns=1.0,
            output_samples=21,
        )

        trajectory = model.simulate_trajectory(
            pulse,
            contour_a,
            contour_b,
            initial_state=1,
        )

        self.assertEqual(trajectory.initial_state, "01")
        self.assertEqual(trajectory.solver_method, "solver")
        self.assertTrue(np.all(np.isfinite(trajectory.logical_population)))
        self.assertTrue(np.all((trajectory.leakage >= 0) & (trajectory.leakage <= 1)))

    def test_population_trajectory_rejects_unknown_initial_state(self):
        config = tiny_config()
        model = build_dualrail_fsim_model(config)
        contour_a, contour_b = pulse_contours(config)

        with self.assertRaisesRegex(ValueError, "initial_state"):
            model.simulate_trajectory(
                DualRailPulse(coupler_peak_flux=0.22, duration_ns=1.0),
                contour_a,
                contour_b,
                initial_state="12",
            )

    def test_rectangular_exact_evolution_returns_metrics_and_dataset(self):
        config = tiny_config()
        model = build_dualrail_fsim_model(config)
        contour_a, contour_b = pulse_contours(config)
        pulse = DualRailPulse(
            coupler_peak_flux=0.22,
            duration_ns=4.0,
            rise_time_ns=0.0,
            output_samples=11,
        )

        result = model.simulate(pulse, contour_a, contour_b)

        self.assertEqual(result.solver_method, "exact")
        self.assertEqual(result.projected_unitary.shape, (4, 4))
        self.assertEqual(result.subsystem_flux.shape, (11, 5))
        np.testing.assert_allclose(
            result.subsystem_flux[[0, -1]],
            [
                [
                    config.rail_a1.flux,
                    config.rail_a2.flux,
                    config.coupler.flux,
                    config.rail_b1.flux,
                    config.rail_b2.flux,
                ],
            ]
            * 2,
        )
        self.assertTrue(
            np.all((result.leakage_by_state >= 0) & (result.leakage_by_state <= 1))
        )
        dataset = result.to_dataset()
        self.assertEqual(
            dataset.attrs["dynamic_dualrail_fsim_schema"],
            DYNAMIC_DUALRAIL_FSIM_SCHEMA,
        )
        self.assertEqual(dataset.projected_unitary_real.shape, (4, 4))

    def test_rectangular_solver_matches_exact_propagation(self):
        config = tiny_config()
        model = build_dualrail_fsim_model(config)
        contour_a, contour_b = pulse_contours(config)
        pulse = DualRailPulse(
            coupler_peak_flux=0.22,
            duration_ns=1.0,
            padding_ns=0.5,
            rise_time_ns=0.0,
            output_samples=9,
        )

        exact = model.simulate(pulse, contour_a, contour_b, method="exact")
        solved = model.simulate(
            pulse,
            contour_a,
            contour_b,
            method="solver",
            solver=SolverConfig(atol=1e-11, rtol=1e-9, nsteps=100_000),
            cpus=2,
        )

        np.testing.assert_allclose(
            solved.projected_unitary,
            exact.projected_unitary,
            atol=1e-6,
        )

    def test_cosine_ramp_uses_solver_and_stays_on_contours(self):
        config = tiny_config()
        model = build_dualrail_fsim_model(config)
        contour_a, contour_b = pulse_contours(config)
        pulse = DualRailPulse(
            coupler_peak_flux=0.22,
            logical_frequency_a_peak_mhz=105.0,
            logical_frequency_b_peak_mhz=95.0,
            duration_ns=4.0,
            rise_time_ns=1.0,
            output_samples=21,
        )

        result = model.simulate(pulse, contour_a, contour_b)

        self.assertEqual(result.solver_method, "solver")
        self.assertAlmostEqual(result.logical_frequency_a_mhz.max(), 105.0)
        self.assertAlmostEqual(result.logical_frequency_b_mhz.min(), 95.0)
        self.assertTrue(np.all(np.isfinite(result.projected_unitary)))
        self.assertTrue(np.all(np.isfinite(result.max_flux_slew_per_ns)))

    def test_cosine_ramp_converges_with_time_grid_resolution(self):
        config = tiny_config()
        model = build_dualrail_fsim_model(config)
        contour_a, contour_b = pulse_contours(config)
        pulse_kwargs = {
            "coupler_peak_flux": 0.22,
            "logical_frequency_a_peak_mhz": 105.0,
            "logical_frequency_b_peak_mhz": 95.0,
            "duration_ns": 2.0,
            "rise_time_ns": 1.0,
        }

        coarse = model.simulate(
            DualRailPulse(output_samples=31, **pulse_kwargs),
            contour_a,
            contour_b,
        )
        refined = model.simulate(
            DualRailPulse(output_samples=61, **pulse_kwargs),
            contour_a,
            contour_b,
        )

        np.testing.assert_allclose(
            coarse.projected_unitary,
            refined.projected_unitary,
            atol=2e-5,
        )

    def test_rejects_out_of_range_frequency_and_excess_slew(self):
        config = tiny_config()
        model = build_dualrail_fsim_model(config)
        contour_a, contour_b = pulse_contours(config)
        with self.assertRaisesRegex(ValueError, "outside contour range"):
            model.simulate(
                DualRailPulse(
                    coupler_peak_flux=0.22,
                    logical_frequency_a_peak_mhz=120.0,
                    duration_ns=4.0,
                ),
                contour_a,
                contour_b,
            )

        constraints = DualRailPulseConstraints(
            max_flux_slew_per_ns=(0.01,) * 5,
        )
        with self.assertRaisesRegex(ValueError, "maximum flux slew"):
            model.simulate(
                DualRailPulse(
                    coupler_peak_flux=0.22,
                    duration_ns=4.0,
                    rise_time_ns=0.0,
                ),
                contour_a,
                contour_b,
                constraints=constraints,
            )


if __name__ == "__main__":
    unittest.main()
