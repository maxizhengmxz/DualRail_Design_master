from dataclasses import replace

import numpy as np
import xarray as xr

from logical_resonance_gate.fsim import (
    CouplerPulse,
    CouplingConfig,
    SolverConfig,
    ThreeTransmonConfig,
    TunableTransmonConfig,
    _extract_fsim_angles,
    build_fsim_model,
    ideal_fsim,
    load_or_compute_fsim_landscape,
    run_fsim_landscape,
    run_swap_oscillation_sweep,
    simulate_fsim,
)


def test_solver_config_rejects_unknown_integrator():
    with np.testing.assert_raises_regex(ValueError, "integrator"):
        SolverConfig(integrator="not-a-solver").validate()


def _small_uncoupled_config():
    return ThreeTransmonConfig(
        qubit1=TunableTransmonConfig(
            EJmax=12.0, EC=0.3, d=0.2, ncut=4, truncated_dim=2
        ),
        qubit2=TunableTransmonConfig(
            EJmax=10.0, EC=0.32, d=0.3, ncut=4, truncated_dim=2
        ),
        coupler=TunableTransmonConfig(
            EJmax=16.0, EC=0.5, d=0.1, ncut=4, truncated_dim=2
        ),
        coupling=CouplingConfig(0.0, 0.0, 0.0),
    )


def test_cosine_pulse_starts_and_ends_at_idle_flux():
    pulse = CouplerPulse(
        peak_flux=0.3, duration_ns=20.0, rise_time_ns=5.0, output_samples=5
    )

    flux = pulse.flux(np.array([0.0, 5.0, 10.0, 15.0, 20.0]), idle_flux=0.1)

    np.testing.assert_allclose(flux, [0.1, 0.3, 0.3, 0.3, 0.1])


def test_ideal_fsim_uses_standard_basis_ordering():
    matrix = ideal_fsim(theta_rad=np.pi / 2, phi_rad=np.pi / 3)

    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(matrix[1, 2], -1j)
    np.testing.assert_allclose(matrix[2, 1], -1j)
    np.testing.assert_allclose(matrix[3, 3], np.exp(-1j * np.pi / 3))

    theta, phi, swap_probability = _extract_fsim_angles(matrix)
    np.testing.assert_allclose(theta, np.pi / 2)
    np.testing.assert_allclose(phi, np.pi / 3)
    np.testing.assert_allclose(swap_probability, 1.0)


def test_uncoupled_device_has_no_swap_conditional_phase_or_leakage():
    pulse = CouplerPulse(
        peak_flux=0.0, duration_ns=1.0, rise_time_ns=0.5, output_samples=3
    )

    result = simulate_fsim(_small_uncoupled_config(), pulse)

    assert result.projected_unitary.shape == (4, 4)
    np.testing.assert_allclose(result.theta_rad, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.phi_rad, 0.0, atol=1e-8)
    np.testing.assert_allclose(result.leakage_by_state, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.computational_assignment_probability, 1.0)
    assert result.parasitic_amplitude < 1e-10


def test_rectangular_three_flux_pulse_uses_exact_propagation():
    pulse = CouplerPulse(
        peak_flux=0.2,
        duration_ns=1.0,
        rise_time_ns=0.0,
        output_samples=3,
        qubit1_peak_flux=0.05,
        qubit2_peak_flux=0.03,
    )

    result = simulate_fsim(_small_uncoupled_config(), pulse)

    np.testing.assert_allclose(result.theta_rad, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.phi_rad, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.leakage_by_state, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.qubit1_flux, 0.05)
    np.testing.assert_allclose(result.qubit2_flux, 0.03)


def test_paper_style_sweeps_return_bounded_maps():
    config = _small_uncoupled_config()
    model = build_fsim_model(config)
    swap = run_swap_oscillation_sweep(
        model,
        coupler_flux_values=np.array([0.0, 0.1]),
        duration_ns=np.linspace(0.0, 2.0, 5),
        qubit1_peak_flux=0.0,
        qubit2_peak_flux=0.0,
    )
    landscape = run_fsim_landscape(
        model,
        qubit1_flux_values=np.array([0.0, 0.05]),
        coupler_flux_values=np.array([0.0, 0.1]),
        qubit2_peak_flux=0.0,
        duration_ns=1.0,
    )

    assert swap.population_01_to_10.shape == (2, 5)
    assert isinstance(landscape, xr.Dataset)
    assert landscape.theta_rad.shape == (2, 2)
    assert landscape.phi_rad.shape == (2, 2)
    assert np.all((swap.population_01_to_10 >= 0) & (swap.population_01_to_10 <= 1))
    assert np.all((landscape.leakage_11 >= 0) & (landscape.leakage_11 <= 1))


def test_parallel_landscape_matches_serial_result():
    model = build_fsim_model(_small_uncoupled_config())
    kwargs = {
        "qubit1_flux_values": np.array([0.0, 0.05]),
        "coupler_flux_values": np.array([0.0, 0.1, 0.2]),
        "qubit2_peak_flux": 0.0,
        "duration_ns": 1.0,
    }

    serial = run_fsim_landscape(model, cpus=1, **kwargs)
    parallel = run_fsim_landscape(model, cpus=2, **kwargs)

    for name in serial.data_vars:
        np.testing.assert_allclose(parallel[name], serial[name], atol=1e-12)
    assert serial.attrs["worker_count"] == 1
    assert parallel.attrs["worker_count"] == 2


def test_parallel_swap_sweep_matches_serial_result():
    model = build_fsim_model(_small_uncoupled_config())
    kwargs = {
        "coupler_flux_values": np.array([0.0, 0.1, 0.2]),
        "duration_ns": np.linspace(0.0, 2.0, 5),
        "qubit1_peak_flux": 0.0,
        "qubit2_peak_flux": 0.0,
    }

    serial = run_swap_oscillation_sweep(model, cpus=1, **kwargs)
    parallel = run_swap_oscillation_sweep(model, cpus=2, **kwargs)

    for name in (
        "population_01_to_10",
        "leakage_from_01",
        "swap_frequency_mhz",
        "effective_coupling_mhz",
    ):
        np.testing.assert_allclose(getattr(parallel, name), getattr(serial, name))


def test_parallel_shaped_pulse_matches_serial_result():
    model = build_fsim_model(_small_uncoupled_config())
    pulse = CouplerPulse(
        peak_flux=0.1,
        duration_ns=1.0,
        rise_time_ns=0.5,
        output_samples=3,
    )

    serial = model.simulate(pulse, cpus=1)
    parallel = model.simulate(pulse, cpus=2)

    np.testing.assert_allclose(
        parallel.projected_unitary,
        serial.projected_unitary,
        atol=1e-12,
    )


def test_fsim_landscape_cache_round_trip_and_validation(tmp_path):
    model = build_fsim_model(_small_uncoupled_config())
    path = tmp_path / "fsim_landscape.nc"
    kwargs = {
        "qubit1_flux_values": np.array([0.0, 0.05]),
        "coupler_flux_values": np.array([0.0, 0.1]),
        "qubit2_peak_flux": 0.0,
        "duration_ns": 1.0,
        "path": path,
    }

    computed, computed_path = load_or_compute_fsim_landscape(
        model, force=True, **kwargs
    )
    loaded, loaded_path = load_or_compute_fsim_landscape(model, **kwargs)

    assert computed_path == loaded_path == path
    xr.testing.assert_allclose(loaded, computed)

    with np.testing.assert_raises_regex(ValueError, "different pulse duration"):
        load_or_compute_fsim_landscape(model, **{**kwargs, "duration_ns": 2.0})

    changed_config = replace(
        _small_uncoupled_config(),
        coupling=CouplingConfig(0.01, 0.0, 0.0),
    )
    changed_model = build_fsim_model(changed_config)
    with np.testing.assert_raises_regex(ValueError, "different device model"):
        load_or_compute_fsim_landscape(changed_model, **kwargs)
