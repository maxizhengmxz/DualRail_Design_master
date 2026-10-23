# Mixed physical/logical gate exploration; see this folder's README.md.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    return


@app.cell
def _():
    from dataclasses import replace

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from logical_resonance_gate import (
        CouplerPulse,
        CouplingConfig,
        DualRailCouplingConfig,
        DualRailPulse,
        FiveTransmonConfig,
        LogicalModeMapping,
        ThreeTransmonConfig,
        TunableTransmonConfig,
        build_dualrail_fsim_model,
        build_fsim_model,
        build_logical_contour,
        calibrate_reduced_pair,
        dualrail_idle_score,
        load_or_compute_fsim_landscape,
        load_or_compute_static_interaction_landscape,
        run_swap_oscillation_sweep,
        select_dualrail_idle_point,
    )
    from core import (
        gen_tdt,
        load_or_compute_operating_point_dispersive_sweep,
        load_or_compute_operating_points,
    )


    return (
        CouplerPulse,
        CouplingConfig,
        DualRailCouplingConfig,
        DualRailPulse,
        FiveTransmonConfig,
        LogicalModeMapping,
        ThreeTransmonConfig,
        TunableTransmonConfig,
        build_dualrail_fsim_model,
        build_fsim_model,
        build_logical_contour,
        calibrate_reduced_pair,
        dualrail_idle_score,
        gen_tdt,
        load_or_compute_fsim_landscape,
        load_or_compute_operating_point_dispersive_sweep,
        load_or_compute_operating_points,
        load_or_compute_static_interaction_landscape,
        mo,
        np,
        plt,
        replace,
        run_swap_oscillation_sweep,
        select_dualrail_idle_point,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Toy gate model: 2 transmon + tunable coupler
    """)
    return


@app.cell
def _(CouplingConfig, ThreeTransmonConfig, TunableTransmonConfig):
    simple_case_config = ThreeTransmonConfig(
        qubit1=TunableTransmonConfig(
            EJmax=18.0,
            EC=0.24,
            d=0.1,
            flux=0.0,
            ng=0.0,
            ncut=30,
            truncated_dim=8,
        ),
        qubit2=TunableTransmonConfig(
            EJmax=15.0,
            EC=0.24,
            d=0.1,
            flux=0.0,
            ng=0.0,
            ncut=30,
            truncated_dim=8,
        ),
        coupler=TunableTransmonConfig(
            EJmax=24.4,
            EC=0.9,
            d=0.1,
            flux=0.24,
            ng=0.0,
            ncut=30,
            truncated_dim=9,
        ),
        coupling=CouplingConfig(
            qubit1_coupler=0.79,
            qubit2_coupler=0.79,
            # Cancels most coupler-mediated exchange at the idle point.
            qubit1_qubit2=0.118,
        ),
        # coupling=CouplingConfig(
        #     qubit1_coupler=0.2,
        #     qubit2_coupler=0.2,
        #     # Cancels most coupler-mediated exchange at the idle point.
        #     qubit1_qubit2=0.048,
        # ),
    )
    return (simple_case_config,)


@app.cell
def _(np):
    np.sqrt(8*24.4*np.cos(0.3*np.pi)*0.9) ## coupler frequency
    return


@app.cell
def _(CouplerPulse):

    simple_case_iswap_pulse = CouplerPulse(
        peak_flux=0.233379106,
        duration_ns=177.829,
        rise_time_ns=0.0,
        output_samples=401,
        qubit1_peak_flux=0.199799520,
        qubit2_peak_flux=0.073374533,
    )
    simple_case_cphase_pulse = CouplerPulse(
        peak_flux=0.269732377,
        duration_ns=232.329087,
        rise_time_ns=0.0,
        output_samples=401,
        qubit1_peak_flux=0.134984373,
        qubit2_peak_flux=0.000003285,
    )
    return simple_case_cphase_pulse, simple_case_iswap_pulse


@app.cell
def _(build_fsim_model, simple_case_config):
    simple_case_model = build_fsim_model(simple_case_config)
    return (simple_case_model,)


@app.cell
def _(np, simple_case_model):
    energies = np.asarray(simple_case_model.hamiltonian_ghz.eigenenergies())
    e00, e01, e10, e11 = energies[simple_case_model.computational_assignment]

    idle_zz_mhz = 1e3 * (e11 - e10 - e01 + e00)
    print(idle_zz_mhz)
    return


@app.cell
def _(simple_case_model):
    simple_case_model.computational_assignment_probability.min()
    return


@app.cell
def _(simple_case_cphase_pulse, simple_case_iswap_pulse, simple_case_model):
    simple_case_iswap_result = simple_case_model.simulate(
        simple_case_iswap_pulse
    )
    simple_case_cphase_result = simple_case_model.simulate(
        simple_case_cphase_pulse
    )
    return simple_case_cphase_result, simple_case_iswap_result


@app.cell(hide_code=True)
def _(mo, simple_case_cphase_result, simple_case_iswap_result):
    mo.md(fr"""
    **Paper-inspired calibrated gates (local Z phases omitted)**

    | Gate | $\theta/\pi$ | $\phi/\pi$ | swap probability | average leakage |
    |---|---:|---:|---:|---:|
    | iSWAP | {simple_case_iswap_result.theta_over_pi:.6f} | {simple_case_iswap_result.phi_over_pi:.6f} | {simple_case_iswap_result.swap_probability:.6f} | {simple_case_iswap_result.average_leakage:.3e} |
    | CPhase | {simple_case_cphase_result.theta_over_pi:.6f} | {simple_case_cphase_result.phi_over_pi:.6f} | {simple_case_cphase_result.swap_probability:.6f} | {simple_case_cphase_result.average_leakage:.3e} |

    Weakest idle bare-to-dressed assignment:
    {simple_case_iswap_result.computational_assignment_probability.min():.6f}.
    """)
    return


@app.cell
def _(plt, simple_case_cphase_result, simple_case_iswap_result):
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 2.7), dpi=180)
    for _ax, _result, _title in zip(
        _axes,
        (simple_case_iswap_result, simple_case_cphase_result),
        ("iSWAP", "CPhase"),
        strict=True,
    ):
        _ax.plot(_result.time_ns, _result.qubit1_flux, label="qubit 1")
        _ax.plot(_result.time_ns, _result.qubit2_flux, label="qubit 2")
        _ax.plot(_result.time_ns, _result.coupler_flux, label="coupler")
        _ax.set(xlabel="Time (ns)", ylabel=r"Flux ($\Phi_0$)", title=_title)
        _ax.grid()
    _axes[0].legend()
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paper-style control maps

    The dense maps below use a reduced subsystem truncation for speed. Figure 1
    places the qubits on resonance and sweeps coupler bias and pulse duration.
    Figure 2 uses a fixed rectangular pulse and sweeps qubit detuning and
    effective coupling, following Figs. 1 and 2 of Foxen *et al.*
    """)
    return


@app.cell
def _(build_fsim_model, replace, simple_case_config):
    figure_sweep_config = replace(
        simple_case_config,
        qubit1=replace(simple_case_config.qubit1, truncated_dim=4),
        qubit2=replace(simple_case_config.qubit2, truncated_dim=4),
        coupler=replace(simple_case_config.coupler, truncated_dim=5),
    )
    figure_sweep_model = build_fsim_model(figure_sweep_config)
    return (figure_sweep_model,)


@app.cell
def _(figure_sweep_model, np, run_swap_oscillation_sweep):
    figure1_sweep = run_swap_oscillation_sweep(
        figure_sweep_model,
        coupler_flux_values=np.linspace(0.2, 0.44, 101),
        duration_ns=np.linspace(0.0, 800.0, 801),
        qubit1_peak_flux=0.187494381,
        qubit2_peak_flux=0.0,
        cpus=8,
    )
    return (figure1_sweep,)


@app.cell
def _(figure1_sweep, np, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(10.5, 3.1), dpi=180)

    # (a) Synchronous rectangular control-pulse schematic.
    _schematic_time = np.array([-20.0, 0.0, 300.0, 320.0])
    for _offset, _height, _label in (
        (2.0, 0.55, "qubit 1"),
        (1.0, 0.35, "qubit 2"),
        (0.0, 0.75, "coupler"),
    ):
        _axes[0].step(
            _schematic_time,
            _offset + _height * np.array([0.0, 1.0, 1.0, 0.0]),
            where="post",
            label=_label,
        )
    _axes[0].set(
        xlabel="Time (ns)",
        yticks=[],
        title="(a) Synchronous flux pulses",
        xlim=(-20, 320),
    )
    _axes[0].legend(frameon=False, fontsize=8)

    # (b) Swap chevrons versus coupler bias and interaction time.
    _mesh = _axes[1].pcolormesh(
        figure1_sweep.coupler_flux,
        figure1_sweep.duration_ns,
        figure1_sweep.population_01_to_10.T,
        shading="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    _axes[1].set(
        xlabel=r"Coupler peak flux ($\Phi_0$)",
        ylabel="Interaction time (ns)",
        title=r"(b) $P_{|01\rangle\to|10\rangle}$",
    )
    _fig.colorbar(_mesh, ax=_axes[1], label="Transferred population")

    # (c) Coupling extracted from the population-oscillation FFT.
    _axes[2].plot(
        figure1_sweep.coupler_flux,
        figure1_sweep.effective_coupling_mhz,
        marker="o",
        markersize=2.5,
    )
    _axes[2].set(
        xlabel=r"Coupler peak flux ($\Phi_0$)",
        ylabel=r"Effective $|g|$ (MHz)",
        title="(c) Extracted coupling",
    )
    _axes[2].grid(alpha=0.3)
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(figure_sweep_model, load_or_compute_fsim_landscape, np):
    figure2_landscape, _figure2_landscape_path = load_or_compute_fsim_landscape(
        figure_sweep_model,
        force=False,
        qubit1_flux_values=np.linspace(0.10, 0.20, 401),
        coupler_flux_values=np.linspace(0.14, 0.43, 801),
        qubit2_peak_flux=0.0,
        duration_ns=15.0,
        cpus=8,
    )
    return (figure2_landscape,)


@app.cell
def _(figure2_landscape, np, plt):
    _fig, _axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.5),
        dpi=300,
        sharex=True,
        constrained_layout=True,
    )
    _x = figure2_landscape.detuning_mhz
    _y = figure2_landscape.effective_coupling_mhz
    _panels = (
        (
            100 * figure2_landscape.leakage_11,
            "Greys",
            0,
            100,
            "(a) Leakage from $|11\\rangle$ (%)",
        ),
        (
            np.degrees(figure2_landscape.phi_rad),
            "RdBu",
            -180,
            180,
            r"(b) Conditional phase $\phi$ (deg)",
        ),
        (
            np.degrees(figure2_landscape.theta_rad),
            "Blues",
            0,
            90,
            r"(c) Swap angle $\theta$ (deg)",
        ),
    )
    for _ax, (_values, _cmap, _vmin, _vmax, _title) in zip(
        _axes, _panels, strict=True
    ):
        _mesh = _ax.pcolormesh(
            _x,
            _y,
            _values,
            shading="auto",
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
        )
        _ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        _ax.axvline(245, color="black", linestyle=":", linewidth=0.8)
        _ax.set(xlabel=r"Qubit detuning $\Delta$ (MHz)", title=_title)
        _ax.title.set_fontsize(10)
        _ax.set_yscale("log")
        _ax.set_ylim(1, 140)
        _fig.colorbar(_mesh, ax=_ax)
    _axes[0].set_ylabel(r"Effective $|g|$ (MHz)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Full double Tunable-Dualrail Transmon
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## parameters defining the TDT system

    the units are a bit funny.

    EJ are in units of GHz, Lr in unit of nH.

    All C in units of fF, except C1 and C2, which are defined in Ec terms and have unit MHz
    """)
    return


@app.cell
def _():
    tdt1_params = {
        "C1": 354,
        "Ej1": 12.2,
        "Cj1": 5,
        "Ej1_2": 7.8,
        "Cj1_2": 5,
        "C2": 434,
        "Ej2": 12.2,
        "Cj2": 5,
        "Ej2_2": 7.8,
        "Cj2_2": 5,
        "C1r": 15,
        "C2r": 15,
        "Cr": 531,
        "Lr": 1.33,
        "C12": 2.3,
    }

    tdt2_params = {
        "C1": 294,
        "Ej1": 12.2,
        "Cj1": 5,
        "Ej1_2": 7.8,
        "Cj1_2": 5,
        "C2": 364,
        "Ej2": 12.2,
        "Cj2": 5,
        "Ej2_2": 7.8,
        "Cj2_2": 5,
        "C1r": 13,
        "C2r": 13,
        "Cr": 531*1.01,
        "Lr": 1.33*1.01,
        "C12": 2.3,
    }

    if tdt1_params["C12"] != tdt2_params["C12"]:
        raise ValueError(
            f"C12 must match: tdt1={tdt1_params['C12']}, "
            f"tdt2={tdt2_params['C12']}"
        )
    return tdt1_params, tdt2_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## convert params into relavent coupling strength and frequencies of a particular dual-rail pair
    """)
    return


@app.cell(hide_code=True)
def _(np, tdt1_params, tdt2_params):
    which_pair = "tdt1"  # Change to "tdt2" for the second pair.

    planck_constant = 6.62607015e-34
    electron_charge = 1.602176634e-19
    hbar = planck_constant/(2*np.pi)
    magnetic_flux_quantum = planck_constant/(2*electron_charge)

    # Node order: rail 1, rail 2, readout resonator. Units: farads.
    # C = [[C1g + C12 + C1r, -C12,            -C1r],
    #      [-C12,            C2g + C12 + C2r, -C2r],
    #      [-C1r,            -C2r,            Cr + C1r + C2r]]
    # Diagonals sum attached capacitances; off-diagonals are negative mutual
    # capacitances. C1g/C2g include both junction capacitances, as in gen_tdt.
    # C1/C2 parameters are EC/h in MHz: C = e**2/(2*h*f). Others are in fF.
    # The inverse enters the charging energy: H_C = Q.T @ inverse(C) @ Q / 2.
    capacitance_matrices = {}
    for _name, _params in (("tdt1", tdt1_params), ("tdt2", tdt2_params)):
        _c1g = electron_charge**2 / (2*planck_constant*_params["C1"]*1e6)
        _c1g += (_params["Cj1"] + _params["Cj1_2"])*1e-15
        _c2g = electron_charge**2 / (2*planck_constant*_params["C2"]*1e6)
        _c2g += (_params["Cj2"] + _params["Cj2_2"])*1e-15
        _c12, _c1r, _c2r, _cr = (
            _params[_key]*1e-15 for _key in ("C12", "C1r", "C2r", "Cr")
        )
        capacitance_matrices[_name] = np.array([
            [_c1g + _c12 + _c1r, -_c12, -_c1r],
            [-_c12, _c2g + _c12 + _c2r, -_c2r],
            [-_c1r, -_c2r, _cr + _c1r + _c2r],
        ])

    # Select the device used by the following approximate coupling estimates.
    readout_params = {"tdt1": tdt1_params, "tdt2": tdt2_params}[which_pair]
    c_matrix = capacitance_matrices[which_pair]

    c_matrix_inv = np.linalg.inv(c_matrix)

    cr_eff = 1/(c_matrix_inv[2, 2])

    # Voltage participation ratios from the full capacitance network.
    beta_1r = c_matrix_inv[0, 2]/c_matrix_inv[2, 2]
    beta_2r = c_matrix_inv[1, 2]/c_matrix_inv[2, 2]

    omega_r = 1/np.sqrt(readout_params["Cr"]*1e-15*readout_params["Lr"]*1e-9)

    # Resonator zero-point voltage (V) and approximate rail-resonator
    # coupling rates g1r/g2r (rad/s), using zero-flux transmon charge elements.
    v_r_zpf = np.sqrt(hbar*omega_r/2/cr_eff)

    _ej1 = readout_params["Ej1"] + readout_params["Ej1_2"]
    _ec1_ghz = (
        electron_charge**2
        * c_matrix_inv[0, 0]
        / (2 * planck_constant)
        / 1e9
    )
    _ej2 = readout_params["Ej2"] + readout_params["Ej2_2"]
    _ec2_ghz = (
        electron_charge**2
        * c_matrix_inv[1, 1]
        / (2 * planck_constant)
        / 1e9
    )

    # Bare resonator charging energy EC = e^2/(2*Cr), reported as EC/h in MHz.
    # Inductive energy EL = (Phi0/(2*pi))^2/Lr, reported as EL/h in GHz.
    # EL is an energy scale, not the resonator oscillation frequency.
    _cr = readout_params["Cr"]*1e-15
    _lr = readout_params["Lr"]*1e-9
    _ec_mhz = electron_charge**2/(2*_cr*planck_constant)/1e6
    _el_ghz = (magnetic_flux_quantum/(2*np.pi))**2/(_lr*planck_constant)/1e9
    # print(f"Resonator EC/h: {_ec_mhz:.6g} MHz")
    # print(f"Resonator EL/h: {_el_ghz:.6g} GHz")

    print(f'==================== {which_pair} ====================')
    print("Resonator:")
    print(f'frequency: {1/np.sqrt(_lr*_cr)/2/np.pi/1e9:.6g} GHz; impedance: {np.sqrt(_lr/_cr):.6g} Ohm')

    g1r = 2*electron_charge*beta_1r/hbar * v_r_zpf * (_ej1 / (32 * _ec1_ghz))**0.25
    g2r = 2*electron_charge*beta_2r/hbar * v_r_zpf * (_ej2 / (32 * _ec2_ghz))**0.25

    # Bare, zero-flux transmon f01 estimate: sqrt(8*EJ*EC)/h - EC/h.
    # EJ/h is the sum of the two junction energies in GHz; EC/h is in MHz.
    # These use the input shunt charging energies, without network dressing.

    transmon_frequency_1 = np.sqrt(8*_ej1*_ec1_ghz) - _ec1_ghz
    transmon_frequency_2 = np.sqrt(8*_ej2*_ec2_ghz) - _ec2_ghz
    print("")
    print("T1:")
    print(f"qubit-resonator coupling: {g1r/1e6/2/np.pi:.6g} MHz")
    print(f"bare zero-flux f01: {transmon_frequency_1:.6g} GHz (capacitively loaded)")

    print("T2:")
    print(f"qubit-resonator coupling: {g2r/1e6/2/np.pi:.6g} MHz")
    print(f"bare zero-flux f01: {transmon_frequency_2:.6g} GHz (capacitively loaded)")

    print("")

    # Weak-coupling estimate of direct rail-to-rail capacitive exchange:
    # J/(2*pi) ~= C12/(2*sqrt(C1*C2))*sqrt(f01_1*f01_2), in Hz.
    # Preserves the original bare-shunt approximation; excludes junction,
    # resonator loading and flux dependence. This is not the dressed splitting.
    _c1 = electron_charge**2/(2*planck_constant*readout_params["C1"]*1e6)
    _c2 = electron_charge**2/(2*planck_constant*readout_params["C2"]*1e6)
    _exchange_hz = readout_params["C12"]*1e-15/(2*np.sqrt(_c1*_c2))*np.sqrt(transmon_frequency_1*transmon_frequency_2)
    print(f"qubit-qubit exchange 2J/(2pi): {2*_exchange_hz*1e3:.6g} MHz (logical frequency)")
    return


@app.cell
def _(gen_tdt, tdt1_params, tdt2_params):
    tdt1 = gen_tdt(**tdt1_params)
    tdt2 = gen_tdt(**tdt2_params)
    return tdt1, tdt2


@app.cell
def _(load_or_compute_operating_points, np, tdt1):
    tdt1_operating_points_ds, tdt1_operating_points_path = (
        load_or_compute_operating_points(
            tdt1,
            phi_1_values=np.linspace(0.1, 0.4, 31),
            phi_2_values=np.linspace(0.1, 0.4, 101),
            path="data/double_tdt_operating_points_tdt1.nc",
            force=True,
            evals_count=6,
            cpus=8,
        )
    )
    return (tdt1_operating_points_ds,)


@app.cell
def _(load_or_compute_operating_points, np, tdt2):
    tdt2_operating_points_ds, tdt2_operating_points_path = (
        load_or_compute_operating_points(
            tdt2,
            phi_1_values=np.linspace(0.1, 0.4, 31),
            phi_2_values=np.linspace(0.1, 0.4, 101),
            path="data/double_tdt_operating_points_tdt2.nc",
            force=True,
            evals_count=6,
            cpus=8,
        )
    )
    return (tdt2_operating_points_ds,)


@app.cell
def _(
    gen_tdt,
    load_or_compute_operating_point_dispersive_sweep,
    tdt1_operating_points_ds,
    tdt1_params,
):
    tdt1_dispersive_ds, tdt1_dispersive_path = (
        load_or_compute_operating_point_dispersive_sweep(
            tdt1_operating_points_ds,
            tdt1_params,
            circuit_factory=gen_tdt,
            path="data/double_tdt_operating_point_dispersive_tdt1.nc",
            force=True,
            evals_count=24,
            cpus=8,
        )
    )
    return (tdt1_dispersive_ds,)


@app.cell
def _(
    gen_tdt,
    load_or_compute_operating_point_dispersive_sweep,
    tdt2_operating_points_ds,
    tdt2_params,
):
    tdt2_dispersive_ds, tdt2_dispersive_path = (
        load_or_compute_operating_point_dispersive_sweep(
            tdt2_operating_points_ds,
            tdt2_params,
            circuit_factory=gen_tdt,
            path="data/double_tdt_operating_point_dispersive_tdt2.nc",
            force=True,
            evals_count=24,
            cpus=8,
        )
    )
    return (tdt2_dispersive_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ec: 2%
    Ej: 5%
    """)
    return


@app.cell
def _(plt, tdt1_dispersive_ds):
    _ds = tdt1_dispersive_ds
    _f = _ds.operating_frequency_ghz.values

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.5))

    _points = _axes[0].scatter(
        _ds.phi_2.values,
        _ds.phi_1.values,
        c=_f,
        s=15,
    )
    _axes[0].set(
        xlabel=r"$\Phi_2/\Phi_0$",
        ylabel=r"$\Phi_1/\Phi_0$",
        title="Operating-point tuning path",
    )
    _fig.colorbar(_points, ax=_axes[0], label="Operating frequency (GHz)")

    _axes[1].scatter(_f, 1e3 * _ds.chi_p.values, s=12, label="chi_p")
    _axes[1].scatter(_f, 1e3 * _ds.chi_m.values, s=12, label="chi_m")
    _axes[1].set(
        xlabel="Operating frequency (GHz)",
        ylabel="Conditional shift (MHz)",
        title="Individual readout shifts",
    )
    _axes[1].legend()

    _axes[2].scatter(_f, 1e3 * _ds.delta_chi.values, s=12)
    _axes[2].axhline(0, color="gray", linestyle="--")
    _axes[2].set(
        xlabel="Operating frequency (GHz)",
        ylabel="chi_p − chi_m (MHz)",
        title="Signed readout mismatch",
    )

    for _ax in _axes:
        _ax.grid(alpha=0.3)

    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(
    plt,
    tdt1_dispersive_ds,
    tdt1_operating_points_ds,
    tdt2_dispersive_ds,
    tdt2_operating_points_ds,
):
    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=300)
    _ax.plot(
        tdt1_operating_points_ds.operating_frequency_ghz[:],
        tdt1_dispersive_ds.abs_delta_chi_over_chi_p[:],
        label="TDT1",
    )
    _ax.plot(
        tdt2_operating_points_ds.operating_frequency_ghz[:],
        tdt2_dispersive_ds.abs_delta_chi_over_chi_p[:],
        label="TDT2",
    )

    _ax.legend()
    _ax.set_yscale("log")
    _ax.grid()
    _ax.set_xlabel("Operating Frequency (GHz)")
    _ax.set_ylabel(r"$|\Delta \chi / \chi_p|$")

    plt.show()
    return


@app.cell
def _(
    plt,
    tdt1_dispersive_ds,
    tdt1_operating_points_ds,
    tdt2_dispersive_ds,
    tdt2_operating_points_ds,
):
    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=300)
    _ax.plot(
        tdt1_operating_points_ds.operating_frequency_ghz[:],
        tdt1_dispersive_ds.splitting_ghz[:]*1e3,
        label="TDT1",
    )
    _ax.plot(
        tdt2_operating_points_ds.operating_frequency_ghz[:],
        tdt2_dispersive_ds.splitting_ghz[:]*1e3,
        label="TDT2",
    )

    _ax.legend()
    # _ax.set_yscale("log")
    _ax.grid()
    _ax.set_xlabel("Operating Frequency (GHz)")
    _ax.set_ylabel(r"Logical Frequency (MHz)")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dual-rail fSim static
    """)
    return


@app.cell
def _(
    build_logical_contour,
    tdt1_operating_points_ds,
    tdt2_operating_points_ds,
):
    # Both measured contours fold at their maximum splitting.  Segment 0 is the
    # increasing-frequency branch; selecting it explicitly prevents interpolation
    # between two flux points with the same logical frequency.
    tdt1_logical_contour = build_logical_contour(
        tdt1_operating_points_ds,
        segment=0,
    )
    tdt2_logical_contour = build_logical_contour(
        tdt2_operating_points_ds,
        segment=0,
    )
    return tdt1_logical_contour, tdt2_logical_contour


@app.cell
def _(np, tdt1_logical_contour, tdt2_logical_contour):
    # Search each TDT on its own validated contour branch. Keep the original
    # 90 MHz lower target, then snap the available upper endpoint down to the
    # 0.5 MHz static-grid resolution. No A/B overlap constraint is imposed.
    _frequency_step_mhz = 0.5

    def _independent_frequency_grid(_contour):
        _available_lower, _available_upper = _contour.frequency_range_mhz
        _lower = _frequency_step_mhz * np.ceil(
            max(90.0, _available_lower) / _frequency_step_mhz
        )
        _upper = _frequency_step_mhz * np.floor(
            _available_upper / _frequency_step_mhz
        )
        if _lower > _upper:
            raise ValueError("logical contour has no searchable range above 90 MHz")
        _count = int(round((_upper - _lower) / _frequency_step_mhz)) + 1
        return np.linspace(_lower, _upper, _count)

    tdt1_static_logical_frequency_mhz = _independent_frequency_grid(
        tdt1_logical_contour
    )
    tdt2_static_logical_frequency_mhz = _independent_frequency_grid(
        tdt2_logical_contour
    )
    return tdt1_static_logical_frequency_mhz, tdt2_static_logical_frequency_mhz


@app.cell
def _(
    mo,
    tdt1_logical_contour,
    tdt1_static_logical_frequency_mhz,
    tdt2_logical_contour,
    tdt2_static_logical_frequency_mhz,
):
    mo.md(f"""
    TDT1 validated branch: "
        f"`{tdt1_logical_contour.frequency_range_mhz[0]:.3f}--"
        f"{tdt1_logical_contour.frequency_range_mhz[1]:.3f} MHz`; "
        f"search grid: `{tdt1_static_logical_frequency_mhz[0]:.1f}--"
        f"{tdt1_static_logical_frequency_mhz[-1]:.1f} MHz`.  "
        f"TDT2 validated branch: "
        f"`{tdt2_logical_contour.frequency_range_mhz[0]:.3f}--"
        f"{tdt2_logical_contour.frequency_range_mhz[1]:.3f} MHz`; "
        f"search grid: `{tdt2_static_logical_frequency_mhz[0]:.1f}--"
        f"{tdt2_static_logical_frequency_mhz[-1]:.1f} MHz`.  "
        "The static landscape evaluates their full Cartesian product, including "
        "the `(105, 90) MHz` idle configuration.
    """)
    return


@app.cell
def _(
    DualRailCouplingConfig,
    FiveTransmonConfig,
    LogicalModeMapping,
    TunableTransmonConfig,
    calibrate_reduced_pair,
    tdt1_logical_contour,
    tdt2_logical_contour,
):
    # Anchor the reduced model to the reference-circuit carrier and splitting at
    # 95 MHz before using it away from that point.  A multi-point fit is still
    # required before interpreting the landscape as a device prediction.
    pair_a_calibration = calibrate_reduced_pair(
        tdt1_logical_contour,
        95.0,
        ejmax_ghz=(20.0, 20.0),
        asymmetry=(0.22, 0.22),
        ncut=15,
        truncated_dim=3,
    )
    pair_b_calibration = calibrate_reduced_pair(
        tdt2_logical_contour,
        95.0,
        ejmax_ghz=(20.0, 20.0),
        asymmetry=(0.22, 0.22),
        ncut=15,
        truncated_dim=3,
    )

    dualrail_preview_config = FiveTransmonConfig(
        rail_a1=pair_a_calibration.rail_1,
        rail_a2=pair_a_calibration.rail_2,
        coupler=TunableTransmonConfig(
            EJmax=24.4,
            EC=0.6,
            d=0.02,
            flux=0.25,
            ng=0.0,
            ncut=15,
            truncated_dim=4,
        ),
        rail_b1=pair_b_calibration.rail_1,
        rail_b2=pair_b_calibration.rail_2,
        coupling=DualRailCouplingConfig(
            a1_a2=pair_a_calibration.coupling_ghz,
            a2_c=0.42,
            c_b1=0.42,
            b1_b2=pair_b_calibration.coupling_ghz,
            a2_b1=0.1,
        ),
        logical_mapping=LogicalModeMapping(
            logical_zero="lower",
            logical_one="upper",
        ),
        parameter_source=(
            "single-anchor reduced-model fit with nominal inter-pair couplings; "
            "not a device prediction"
        ),
        min_assignment_probability=0.8,
        min_leakage_gap_mhz=5.0,
    )
    return (dualrail_preview_config,)


@app.cell
def _(
    dualrail_preview_config,
    load_or_compute_static_interaction_landscape,
    np,
    tdt1_logical_contour,
    tdt1_static_logical_frequency_mhz,
    tdt2_logical_contour,
    tdt2_static_logical_frequency_mhz,
):
    dualrail_static_ds, dualrail_static_path = (
        load_or_compute_static_interaction_landscape(
            dualrail_preview_config,
            tdt1_logical_contour,
            tdt2_logical_contour,
            logical_frequency_a_mhz=tdt1_static_logical_frequency_mhz,
            logical_frequency_b_mhz=tdt2_static_logical_frequency_mhz,
            coupler_flux_values=np.linspace(0.0, 0.4, 81),
            path="data/double_tdt_static_fsim_preview.nc",
            force=True,
            cpus=8,
        )
    )
    return (dualrail_static_ds,)


@app.cell
def _(dualrail_static_ds, plt):
    _on_resonance = dualrail_static_ds.sel(
        logical_frequency_a_mhz=dualrail_static_ds.logical_frequency_b_mhz
    )
    _fig, _axes = plt.subplots(1, 3, figsize=(10.5, 3.1), dpi=220)
    _panels = (
        (_on_resonance.logical_exchange_mhz, "Logical exchange (MHz)", "Blues"),
        (_on_resonance.logical_zz_mhz, r"Logical $\zeta_L$ (MHz)", "RdBu_r"),
        (
            _on_resonance.nearest_leakage_gap_mhz,
            "Nearest leakage gap (MHz)",
            "viridis",
        ),
    )
    for _ax, (_values, _title, _cmap) in zip(_axes, _panels, strict=True):
        _mesh = _ax.pcolormesh(
            _on_resonance.logical_frequency_a_mhz,
            _on_resonance.coupler_flux,
            _values,
            shading="auto",
            cmap=_cmap,
        )
        _ax.set(
            xlabel="Common logical frequency (MHz)",
            ylabel=r"Coupler flux ($\Phi_0$)",
            title=_title,
        )
        _fig.colorbar(_mesh, ax=_ax)
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stage 1: choose a static idle point

    Rank the static control grid using

    $$
    r_\mathrm{idle} = \sqrt{
      |\langle 01_L|H_\mathrm{eff}|10_L\rangle|^2
      + |\zeta_\mathrm{eff}|^2
    }.
    $$

    $H_\mathrm{eff}$ is a Hermitian logical Hamiltonian reconstructed from the
    four assigned dressed eigenstates. It includes virtual interactions even
    when direct `PHP` vanishes, and remains meaningful away from resonance.
    Direct `PHP` and the dressed odd-level gap are retained as diagnostics.
    Only points passing logical-assignment and leakage-gap gates are eligible.

    Static-only progression:

    1. Find a coarse low-residual idle candidate on the full three-control grid.
    2. Refine locally and require a broad minimum under all five flux offsets.
    3. Separately identify high-exchange and high-ZZ interaction regions, and
       compare their static on/off contrast with this idle point.
    4. Begin pulse design only after those static checks pass.
    """)
    return


@app.cell
def _(dualrail_static_ds, mo, np, tdt2_logical_contour):
    _tdt2_logical_frequency_mhz = np.asarray(
        dualrail_static_ds.logical_frequency_b_mhz,
        dtype=float,
    )
    _tdt2_operating_frequency_ghz = tdt2_logical_contour.carrier_frequency(
        _tdt2_logical_frequency_mhz
    )
    tdt2_fixed_operating_frequency = mo.ui.slider(
        steps=np.sort(_tdt2_operating_frequency_ghz).tolist(),
        value=float(_tdt2_operating_frequency_ghz[0]),
        debounce=True,
        show_value=True,
        include_input=True,
        label="Fixed TDT2 operating frequency (GHz)",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                "Choose one of the TDT2 operating frequencies represented in "
                "the static Hamiltonian grid."
            ),
            tdt2_fixed_operating_frequency,
        ]
    )
    return (tdt2_fixed_operating_frequency,)


@app.cell
def _(
    dualrail_static_ds,
    np,
    tdt1_logical_contour,
    tdt2_fixed_operating_frequency,
    tdt2_logical_contour,
):
    _tdt1_logical_frequency_mhz = np.asarray(
        dualrail_static_ds.logical_frequency_a_mhz,
        dtype=float,
    )
    _tdt2_logical_frequency_mhz = np.asarray(
        dualrail_static_ds.logical_frequency_b_mhz,
        dtype=float,
    )
    _tdt1_operating_frequency_ghz = tdt1_logical_contour.carrier_frequency(
        _tdt1_logical_frequency_mhz
    )
    _tdt2_operating_frequency_ghz = tdt2_logical_contour.carrier_frequency(
        _tdt2_logical_frequency_mhz
    )
    _tdt2_index = int(
        np.argmin(
            np.abs(
                _tdt2_operating_frequency_ghz
                - float(tdt2_fixed_operating_frequency.value)
            )
        )
    )
    fixed_tdt2_operating_frequency_ghz = float(
        _tdt2_operating_frequency_ghz[_tdt2_index]
    )
    fixed_tdt2_logical_frequency_mhz = float(
        _tdt2_logical_frequency_mhz[_tdt2_index]
    )
    _operating_detuning_mhz = 1e3 * (
        _tdt1_operating_frequency_ghz
        - fixed_tdt2_operating_frequency_ghz
    )
    _tdt1_order = np.argsort(_operating_detuning_mhz)
    tdt1_operating_detuning_mhz = _operating_detuning_mhz[_tdt1_order]
    fixed_tdt2_static_slice = dualrail_static_ds.isel(
        logical_frequency_b_mhz=_tdt2_index,
        logical_frequency_a_mhz=_tdt1_order,
    )
    return (
        fixed_tdt2_logical_frequency_mhz,
        fixed_tdt2_operating_frequency_ghz,
        fixed_tdt2_static_slice,
        tdt1_operating_detuning_mhz,
    )


@app.cell(hide_code=True)
def _(
    fixed_tdt2_logical_frequency_mhz,
    fixed_tdt2_operating_frequency_ghz,
    mo,
):
    mo.md(f"""
    Selected TDT2 operating frequency: "
        f"`{fixed_tdt2_operating_frequency_ghz:.9f} GHz`; corresponding "
        f"logical frequency: `{fixed_tdt2_logical_frequency_mhz:.3f} MHz`.
    """)
    return


@app.cell
def _(
    fixed_tdt2_operating_frequency_ghz,
    fixed_tdt2_static_slice,
    np,
    plt,
    tdt1_operating_detuning_mhz,
):
    _coupler_flux = np.asarray(fixed_tdt2_static_slice.coupler_flux, dtype=float)
    _effective_a_frequency_ghz = (
        fixed_tdt2_static_slice.effective_hamiltonian_real_ghz.sel(
            logical_output="10",
            logical_input="10",
        )
        - fixed_tdt2_static_slice.effective_hamiltonian_real_ghz.sel(
            logical_output="00",
            logical_input="00",
        )
    )
    _effective_b_frequency_ghz = (
        fixed_tdt2_static_slice.effective_hamiltonian_real_ghz.sel(
            logical_output="01",
            logical_input="01",
        )
        - fixed_tdt2_static_slice.effective_hamiltonian_real_ghz.sel(
            logical_output="00",
            logical_input="00",
        )
    )
    _logical_detuning = 1e3 * np.asarray(
        _effective_a_frequency_ghz - _effective_b_frequency_ghz
    )
    _limit = max(float(np.max(np.abs(_logical_detuning))), 1e-12)
    _fig, _ax = plt.subplots(figsize=(6.2, 4.1), dpi=220)
    _mesh = _ax.pcolormesh(
        tdt1_operating_detuning_mhz,
        _coupler_flux,
        _logical_detuning,
        shading="auto",
        cmap="RdBu_r",
        vmin=-_limit,
        vmax=_limit,
    )
    _ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    _ax.set(
        xlabel=r"Operating-frequency detuning $f_{A}-f_{B}$ (MHz)",
        ylabel=r"Coupler flux ($\Phi_0$)",
        title=(
            "Logical-frequency difference, "
            f"TDT2 at {fixed_tdt2_operating_frequency_ghz:.6f} GHz"
        ),
    )
    _fig.colorbar(
        _mesh,
        ax=_ax,
        label=r"Logical-frequency difference $f_{L,A}-f_{L,B}$ (MHz)",
    )
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(
    fixed_tdt2_operating_frequency_ghz,
    fixed_tdt2_static_slice,
    np,
    plt,
    tdt1_operating_detuning_mhz,
):
    _coupler_flux = np.asarray(fixed_tdt2_static_slice.coupler_flux, dtype=float)
    _exchange_squared = np.square(
        np.abs(np.asarray(fixed_tdt2_static_slice.effective_exchange_mhz))
    )
    _zz_squared = np.square(
        np.abs(np.asarray(fixed_tdt2_static_slice.effective_zz_mhz))
    )
    _fig, _axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.1),
        dpi=220,
        sharex=True,
        sharey=True,
    )
    _panels = (
        (
            _exchange_squared,
            "viridis",
            r"$|\langle01_L|H_\mathrm{eff}|10_L\rangle|^2$",
        ),
        (_zz_squared, "magma", r"$|\zeta_\mathrm{eff}|^2$"),
    )
    for _ax, (_values, _cmap, _title) in zip(_axes, _panels, strict=True):
        _mesh = _ax.pcolormesh(
            tdt1_operating_detuning_mhz,
            _coupler_flux,
            _values,
            shading="auto",
            cmap=_cmap,
        )
        _ax.axvline(0.0, color="white", linestyle="--", linewidth=0.8)
        _ax.set(
            xlabel=r"Operating-frequency detuning $f_{A}-f_{B}$ (MHz)",
            title=_title,
        )
        _fig.colorbar(_mesh, ax=_ax, label=r"MHz$^2$")
    _axes[0].set_ylabel(r"Coupler flux ($\Phi_0$)")
    _fig.suptitle(
        f"Static interactions with TDT2 at "
        f"{fixed_tdt2_operating_frequency_ghz:.6f} GHz"
    )
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(dualrail_idle_score, dualrail_static_ds, np, select_dualrail_idle_point):
    dualrail_idle_score_da = dualrail_idle_score(dualrail_static_ds)
    _qualified = (
        dualrail_static_ds.logical_assignment_valid
        & dualrail_static_ds.spectrally_isolated
    )
    dualrail_idle_point_qualified = bool(np.any(np.asarray(_qualified)))
    # If the coarse grid has no valid point, retain the best finite point only
    # as a visibly unqualified diagnostic rather than silently stopping the notebook.
    dualrail_idle_point = select_dualrail_idle_point(
        dualrail_static_ds,
        require_qualified=dualrail_idle_point_qualified,
    )
    return (
        dualrail_idle_point,
        dualrail_idle_point_qualified,
        dualrail_idle_score_da,
    )


@app.cell
def _(
    build_dualrail_fsim_model,
    dualrail_idle_point,
    dualrail_preview_config,
    replace,
):
    dualrail_idle_config = replace(
        dualrail_preview_config,
        rail_a1=replace(
            dualrail_preview_config.rail_a1,
            flux=float(dualrail_idle_point.phi_a1.item()),
        ),
        rail_a2=replace(
            dualrail_preview_config.rail_a2,
            flux=float(dualrail_idle_point.phi_a2.item()),
        ),
        coupler=replace(
            dualrail_preview_config.coupler,
            flux=float(dualrail_idle_point.coupler_flux.item()),
        ),
        rail_b1=replace(
            dualrail_preview_config.rail_b1,
            flux=float(dualrail_idle_point.phi_b1.item()),
        ),
        rail_b2=replace(
            dualrail_preview_config.rail_b2,
            flux=float(dualrail_idle_point.phi_b2.item()),
        ),
    )
    dualrail_idle_model = build_dualrail_fsim_model(dualrail_idle_config)
    dualrail_idle_static = dualrail_idle_model.static_interaction()
    return dualrail_idle_model, dualrail_idle_static


@app.cell(hide_code=True)
def _(
    dualrail_idle_model,
    dualrail_idle_point,
    dualrail_idle_point_qualified,
    dualrail_idle_static,
    mo,
):
    _status = "qualified" if dualrail_idle_point_qualified else "UNQUALIFIED fallback"
    mo.md(
        fr"""
        **Coarse-grid idle candidate — {_status}**

        | quantity | value |
        |---|---:|
        | pair A logical frequency | {dualrail_idle_point.logical_frequency_a_mhz.item():.6g} MHz |
        | pair B logical frequency | {dualrail_idle_point.logical_frequency_b_mhz.item():.6g} MHz |
        | logical detuning | {dualrail_idle_point.logical_detuning_mhz.item():.6g} MHz |
        | coupler flux | {dualrail_idle_point.coupler_flux.item():.6g} Φ₀ |
        | residual-coupling score | {dualrail_idle_point.idle_residual_coupling_mhz.item():.6g} MHz |
        | effective $|\langle01_L|H_\mathrm{{eff}}|10_L\rangle|$ | {dualrail_idle_static.effective_exchange_mhz:.6g} MHz |
        | effective $\zeta_\mathrm{{eff}}$ | {dualrail_idle_static.effective_zz_mhz:.6g} MHz |
        | direct `PHP` exchange | {dualrail_idle_static.direct_exchange_mhz:.6g} MHz |
        | odd-gap exchange estimate (on resonance only) | {dualrail_idle_static.logical_exchange_mhz:.6g} MHz |
        | dressed $\zeta_L$ | {dualrail_idle_static.logical_zz_mhz:.6g} MHz |
        | nearest non-logical level | {dualrail_idle_static.nearest_leakage_gap_mhz:.6g} MHz |
        | weakest dressed-state assignment | {dualrail_idle_static.logical_assignment_probability.min():.6f} |

        Isolated lower-mode characters are
        `{dualrail_idle_model.pair_a_modes.lower_character}` for pair A and
        `{dualrail_idle_model.pair_b_modes.lower_character}` for pair B.

        This is a coarse static choice, not a gate pulse. A denser local static
        scan and sensitivity to all five flux offsets should still precede final
        calibration; the next section uses it as the operational idle point for
        manual pulse experiments.
        """
    )
    return


@app.cell
def _(
    dualrail_idle_point,
    dualrail_idle_point_qualified,
    dualrail_idle_score_da,
    dualrail_static_ds,
    np,
    plt,
):
    _coupler_flux = float(dualrail_idle_point.coupler_flux.item())
    _frequency_a = float(dualrail_idle_point.logical_frequency_a_mhz.item())
    _frequency_b = float(dualrail_idle_point.logical_frequency_b_mhz.item())

    _score_plane = dualrail_idle_score_da.sel(coupler_flux=_coupler_flux)
    if dualrail_idle_point_qualified:
        _valid_plane = (
            dualrail_static_ds.logical_assignment_valid
            & dualrail_static_ds.spectrally_isolated
        ).sel(coupler_flux=_coupler_flux)
        _score_plane = _score_plane.where(_valid_plane)

    _coupler_cut = dualrail_static_ds.sel(
        logical_frequency_a_mhz=_frequency_a,
        logical_frequency_b_mhz=_frequency_b,
    )
    _effective = np.asarray(
        dualrail_idle_point.effective_hamiltonian_real_ghz
        + 1j * dualrail_idle_point.effective_hamiltonian_imag_ghz
    )
    _diagonal = np.real(np.diag(_effective))
    # Remove identity and independent logical-Z terms. What remains is the
    # interaction part: exchange/parasitic matrix elements and effective ZZ.
    _local_diagonal = np.asarray(
        [
            _diagonal[0],
            _diagonal[1],
            _diagonal[2],
            _diagonal[1] + _diagonal[2] - _diagonal[0],
        ]
    )
    _interaction_mhz = 1e3 * (
        _effective - np.diag(_local_diagonal)
    )

    _fig, _axes = plt.subplots(1, 3, figsize=(12.2, 3.5), dpi=220)
    _mesh = _axes[0].pcolormesh(
        _score_plane.logical_frequency_a_mhz,
        _score_plane.logical_frequency_b_mhz,
        _score_plane.transpose(
            "logical_frequency_b_mhz",
            "logical_frequency_a_mhz",
        ),
        shading="auto",
        cmap="viridis",
    )
    _axes[0].scatter(
        [_frequency_a],
        [_frequency_b],
        marker="x",
        s=55,
        linewidths=1.8,
        color="white",
    )
    _axes[0].set(
        xlabel="Pair A logical frequency (MHz)",
        ylabel="Pair B logical frequency (MHz)",
        title=fr"Residual score at $\Phi_C={_coupler_flux:.4g}$",
    )
    _fig.colorbar(_mesh, ax=_axes[0], label="MHz")

    _axes[1].plot(
        _coupler_cut.coupler_flux,
        np.abs(_coupler_cut.effective_exchange_mhz),
        marker="o",
        label=r"$|\langle01_L|H_\mathrm{eff}|10_L\rangle|$",
    )
    _axes[1].plot(
        _coupler_cut.coupler_flux,
        np.abs(_coupler_cut.effective_zz_mhz),
        marker="s",
        label=r"$|\zeta_L|$",
    )
    _axes[1].axvline(_coupler_flux, color="black", linestyle="--", linewidth=1)
    _axes[1].set(
        xlabel=r"Coupler flux ($\Phi_0$)",
        ylabel="Residual interaction (MHz)",
        title="Static rates at selected A/B frequencies",
    )
    _axes[1].legend(frameon=False, fontsize=8)
    _axes[1].grid(alpha=0.25)

    _matrix_image = _axes[2].imshow(
        np.abs(_interaction_mhz),
        origin="upper",
        cmap="magma",
    )
    for _row in range(4):
        for _column in range(4):
            _axes[2].text(
                _column,
                _row,
                f"{abs(_interaction_mhz[_row, _column]):.2g}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
    _axes[2].set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=("00", "01", "10", "11"),
        yticklabels=("00", "01", "10", "11"),
        xlabel="Logical input",
        ylabel="Logical output",
        title=r"$|H_\mathrm{eff,int}|$ (MHz)",
    )
    _fig.colorbar(_matrix_image, ax=_axes[2], label="MHz")
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stage 2: manual pulse tuning

    The editable cell below sets the initial logical state and pulse parameters.
    `manual_idle_padding_ns = 20.0` adds 20 ns at the operational idle point
    before the active pulse and another 20 ns after it. `manual_pulse_duration_ns`
    continues to mean only the active interaction time.
    """)
    return


@app.cell
def _(dualrail_static_ds, np):
    # Use the strongest sampled exchange point as an editable starting seed.
    _qualified = (
        dualrail_static_ds.logical_assignment_valid
        & dualrail_static_ds.spectrally_isolated
    )
    _exchange = np.abs(np.asarray(dualrail_static_ds.effective_exchange_mhz))
    _score = np.where(np.asarray(_qualified), _exchange, -np.inf)
    if not np.any(np.isfinite(_score)):
        _score = np.nan_to_num(_exchange, nan=-np.inf)
    _seed_index = np.unravel_index(np.argmax(_score), _score.shape)
    dualrail_manual_pulse_seed = dualrail_static_ds.isel(
        coupler_flux=_seed_index[0],
        logical_frequency_a_mhz=_seed_index[1],
        logical_frequency_b_mhz=_seed_index[2],
    )
    return (dualrail_manual_pulse_seed,)


@app.cell
def _(dualrail_manual_pulse_seed, np):
    # Manual tuning controls: edit these values and rerun this cell.
    manual_pulse_initial_state = "10"  # one of: "00", "01", "10", "11"
    manual_pulse_coupler_peak_flux = float(
        dualrail_manual_pulse_seed.coupler_flux.item()
    )
    manual_pulse_frequency_a_mhz = float(
        dualrail_manual_pulse_seed.logical_frequency_a_mhz.item()
    )
    manual_pulse_frequency_b_mhz = float(
        dualrail_manual_pulse_seed.logical_frequency_b_mhz.item()
    )
    _estimated_duration_ns = float(
        dualrail_manual_pulse_seed.estimated_iswap_time_ns.item()
    )
    manual_pulse_duration_ns = (
        _estimated_duration_ns
        if np.isfinite(_estimated_duration_ns) and _estimated_duration_ns > 0
        else 50.0
    )
    manual_pulse_rise_time_ns = 0.0
    manual_idle_padding_ns = 20.0
    manual_pulse_output_samples = 401
    return (
        manual_idle_padding_ns,
        manual_pulse_coupler_peak_flux,
        manual_pulse_duration_ns,
        manual_pulse_frequency_a_mhz,
        manual_pulse_frequency_b_mhz,
        manual_pulse_initial_state,
        manual_pulse_output_samples,
        manual_pulse_rise_time_ns,
    )


@app.cell
def _(
    DualRailPulse,
    manual_idle_padding_ns,
    manual_pulse_coupler_peak_flux,
    manual_pulse_duration_ns,
    manual_pulse_frequency_a_mhz,
    manual_pulse_frequency_b_mhz,
    manual_pulse_output_samples,
    manual_pulse_rise_time_ns,
):
    dualrail_manual_pulse = DualRailPulse(
        coupler_peak_flux=manual_pulse_coupler_peak_flux,
        logical_frequency_a_peak_mhz=manual_pulse_frequency_a_mhz,
        logical_frequency_b_peak_mhz=manual_pulse_frequency_b_mhz,
        duration_ns=manual_pulse_duration_ns,
        padding_ns=manual_idle_padding_ns,
        rise_time_ns=manual_pulse_rise_time_ns,
        output_samples=manual_pulse_output_samples,
    )
    return (dualrail_manual_pulse,)


@app.cell
def _(
    dualrail_idle_model,
    dualrail_manual_pulse,
    manual_pulse_initial_state,
    tdt1_logical_contour,
    tdt2_logical_contour,
):
    dualrail_manual_trajectory = dualrail_idle_model.simulate_trajectory(
        dualrail_manual_pulse,
        tdt1_logical_contour,
        tdt2_logical_contour,
        initial_state=manual_pulse_initial_state,
        method="auto",
    )
    return (dualrail_manual_trajectory,)


@app.cell
def _(
    dualrail_manual_pulse,
    dualrail_manual_trajectory,
    manual_pulse_initial_state,
    plt,
):
    _fig, _axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 3.8),
        dpi=220,
        sharex=True,
        constrained_layout=True,
    )
    for _index, _state in enumerate(("00", "01", "10", "11")):
        _axes[0].plot(
            dualrail_manual_trajectory.time_ns,
            dualrail_manual_trajectory.logical_population[:, _index],
            label=fr"$|{_state}_L\rangle$",
        )
    _axes[0].plot(
        dualrail_manual_trajectory.time_ns,
        dualrail_manual_trajectory.leakage,
        color="black",
        linestyle="--",
        label="leakage",
    )
    _axes[0].set(
        xlabel="Evolution time (ns)",
        ylabel="Population",
        ylim=(-0.02, 1.02),
        title=fr"Population evolution from $|{manual_pulse_initial_state}_L\rangle$",
    )
    _axes[0].legend(frameon=False, fontsize=8, ncol=2)

    for _index, _label in enumerate(("A1", "A2", "C", "B1", "B2")):
        _axes[1].plot(
            dualrail_manual_trajectory.time_ns,
            dualrail_manual_trajectory.subsystem_flux[:, _index],
            label=_label,
        )
    _axes[1].set(
        xlabel="Evolution time (ns)",
        ylabel=r"Flux ($\Phi_0$)",
        title="Five-transmon flux trajectories",
    )
    _axes[1].legend(frameon=False, fontsize=8, ncol=2)

    _active_start = dualrail_manual_pulse.padding_ns
    _active_end = _active_start + dualrail_manual_pulse.duration_ns
    for _ax in _axes:
        _ax.axvspan(0.0, _active_start, color="0.9", zorder=-1)
        _ax.axvspan(
            _active_end,
            dualrail_manual_pulse.total_duration_ns,
            color="0.9",
            zorder=-1,
        )
        _ax.axvline(_active_start, color="0.45", linestyle=":", linewidth=0.9)
        _ax.axvline(_active_end, color="0.45", linestyle=":", linewidth=0.9)
        _ax.grid(alpha=0.25)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
