# Logical-transition iSWAP/CZ workflow; see this folder's README.md for physics.
# Relocation bootstrap only: numerical code and parameter values are unchanged.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from core.runtime import configure_single_threaded_blas

    # Keep setup in the import cell so every numerical cell depends on it.
    configure_single_threaded_blas()
    import os
    from dataclasses import asdict, replace
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    from logical_resonance_gate import (
        DualRailPulse,
        build_dualrail_fsim_model,
        build_logical_contour,
        compare_static_and_dynamic_rates,
        extract_fsim_angles,
        load_or_compute_static_interaction_landscape,
        rank_static_gate_candidates,
        refine_static_gate_candidates,
        run_single_transmon_sweep,
        select_dualrail_idle_point,
        track_dualrail_model_path,
        validate_fixed_basis_along_path,
    )
    from core import (
        load_or_compute_operating_points,
        save_dataset,
    )
    from logical_resonance_gate.logical_fsim_config import (
        load_logical_fsim_workflow_config,
    )
    from logical_resonance_gate.logical_fsim_workflow import (
        load_compatible_pulse_sweep,
        load_or_create_configured_reference,
        pulse_results_to_dataset,
    )

    return (
        DualRailPulse,
        Path,
        asdict,
        build_dualrail_fsim_model,
        build_logical_contour,
        compare_static_and_dynamic_rates,
        extract_fsim_angles,
        load_compatible_pulse_sweep,
        load_logical_fsim_workflow_config,
        load_or_compute_operating_points,
        load_or_compute_static_interaction_landscape,
        load_or_create_configured_reference,
        mo,
        np,
        os,
        plt,
        pulse_results_to_dataset,
        refine_static_gate_candidates,
        replace,
        run_single_transmon_sweep,
        save_dataset,
        select_dualrail_idle_point,
        track_dualrail_model_path,
        validate_fixed_basis_along_path,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Five-transmon dual-rail gate characterization

    This notebook implements the staged workflow in
    `docs/specs/five_transmon_dual_rail_gate_implementation_plan.md` for
    subsystem order $[A_1,A_2,C,B_1,B_2]$. Existing data and device values are
    calibration inputs, not immutable truth.

    Execution gates: validate single-valued dual-rail contours; construct an
    overlap-qualified static atlas; extract the full logical Hamiltonian and
    Pauli terms; choose static iSWAP/CZ mechanisms; validate the fixed pulse
    basis; characterize rectangular and smooth pulses; stop before noise and
    large optimizers if convergence or gate thresholds fail.

    Reporting thresholds are local-$Z$ fidelity $\geq0.99$, average leakage
    $<1\%$, $\sigma_{\min}\geq0.85$, and leakage gap $\geq10$ MHz.
    """)
    return


@app.cell
def _(Path, load_logical_fsim_workflow_config, os):
    CONFIG_PATH = Path(
        os.environ.get(
            "LOGICAL_FSIM_CONFIG",
            str(Path(__file__).resolve().parent / "configs" / "logical_fsim.toml"),
        )
    )
    workflow_config = load_logical_fsim_workflow_config(CONFIG_PATH)
    execution = workflow_config.section("execution")
    thresholds = workflow_config.section("thresholds")
    DATA = Path(
        os.environ.get("LOGICAL_FSIM_DATA", execution["data_directory"])
    )
    DATA.mkdir(parents=True, exist_ok=True)
    CPUS_STATIC = int(execution["cpus_static"])
    CPUS_DYNAMICS = int(execution["cpus_dynamics"])
    RECOMPUTE_REFERENCE = bool(execution["recompute_reference"])
    RECOMPUTE_STATIC = bool(execution["recompute_static"])
    RECOMPUTE_DYNAMICS = bool(execution["recompute_dynamics"])
    FIDELITY_TARGET = float(thresholds["fidelity"])
    LEAKAGE_TARGET = float(thresholds["leakage"])
    SIGMA_MIN_TARGET = float(thresholds["sigma_min"])
    LEAKAGE_GAP_TARGET_MHZ = float(thresholds["leakage_gap_mhz"])
    return (
        CPUS_DYNAMICS,
        CPUS_STATIC,
        DATA,
        FIDELITY_TARGET,
        LEAKAGE_GAP_TARGET_MHZ,
        LEAKAGE_TARGET,
        RECOMPUTE_DYNAMICS,
        RECOMPUTE_REFERENCE,
        RECOMPUTE_STATIC,
        SIGMA_MIN_TARGET,
        workflow_config,
    )


@app.cell
def _(workflow_config):
    device = workflow_config.device_config()
    return (device,)


@app.cell
def _(device):
    pair_a_circuit = device.pair_a.build_circuit()
    pair_b_circuit = device.pair_b.build_circuit()
    return pair_a_circuit, pair_b_circuit


@app.cell
def _(
    CPUS_STATIC,
    DATA,
    RECOMPUTE_REFERENCE,
    load_or_compute_operating_points,
    load_or_create_configured_reference,
    pair_a_circuit,
    pair_b_circuit,
    workflow_config,
):
    reference_grid = {**workflow_config.reference_grid(), "cpus": CPUS_STATIC}
    reference_mode = workflow_config.section("reference", "mode")

    def _points(pair, circuit):
        path = DATA / f"logical_fsim_pair_{pair}_operating_points.nc"
        if reference_mode == "configured":
            fingerprint = workflow_config.fingerprint(
                "reference", "configured_contours", extra={"pair": pair}
            )
            return load_or_create_configured_reference(
                lambda: workflow_config.configured_operating_points(pair),
                path=path,
                fingerprint=fingerprint,
                force=RECOMPUTE_REFERENCE,
            )
        fingerprint = workflow_config.fingerprint(
            f"pair_{pair}", "reference", extra=reference_grid
        )
        return load_or_compute_operating_points(
            circuit,
            path=path,
            force=RECOMPUTE_REFERENCE,
            input_fingerprint=fingerprint,
            **reference_grid,
        )

    pair_a_points, _pair_a_path = _points("a", pair_a_circuit)
    pair_b_points, _pair_b_path = _points("b", pair_b_circuit)
    return pair_a_points, pair_b_points


@app.cell
def _(build_logical_contour, pair_a_points, pair_b_points, workflow_config):
    # Explicit branches prevent interpolation across each folded curve.
    contour_settings = workflow_config.section("contours")
    contour_a = build_logical_contour(
        pair_a_points,
        segment=int(contour_settings["segment_a"]),
        min_hybridization=float(contour_settings["min_hybridization"]),
    )
    contour_b = build_logical_contour(
        pair_b_points,
        segment=int(contour_settings["segment_b"]),
        min_hybridization=float(contour_settings["min_hybridization"]),
    )
    return contour_a, contour_b


@app.cell(hide_code=True)
def _(contour_a, contour_b, mo, pair_a_points, pair_b_points):
    _a_raw = 1e3 * pair_a_points.splitting_ghz
    _b_raw = 1e3 * pair_b_points.splitting_ghz
    _lo = max(contour_a.frequency_range_mhz[0], contour_b.frequency_range_mhz[0])
    _hi = min(contour_a.frequency_range_mhz[1], contour_b.frequency_range_mhz[1])
    mo.md(rf"""
    ## M2 — calibrated logical controls

    | quantity | pair A | pair B |
    |---|---:|---:|
    | accepted points | {pair_a_points.sizes["operating_point"]} | {pair_b_points.sizes["operating_point"]} |
    | raw splitting span (MHz) | {float(_a_raw.min()):.3f}–{float(_a_raw.max()):.3f} | {float(_b_raw.min()):.3f}–{float(_b_raw.max()):.3f} |
    | selected branch (MHz) | {contour_a.frequency_range_mhz[0]:.3f}–{contour_a.frequency_range_mhz[1]:.3f} | {contour_b.frequency_range_mhz[0]:.3f}–{contour_b.frequency_range_mhz[1]:.3f} |
    | minimum hybridization | {contour_a.hybridization.min():.5f} | {contour_b.hybridization.min():.5f} |

    Shared resonant interval: **{_lo:.3f}–{_hi:.3f} MHz**. The guide's
    90–150 MHz assumption is not met, so the atlas uses 125 MHz for exchange
    and opposite validated endpoints for detuned/CZ triage.
    """)
    return


@app.cell
def _(contour_a, contour_b, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=170)
    for _c, _label in ((contour_a, "A"), (contour_b, "B")):
        _axes[0].plot(
            _c.carrier_frequency_ghz, _c.logical_frequency_mhz, "o-", ms=3, label=_label
        )
        _axes[1].plot(
            _c.logical_frequency_mhz, _c.hybridization, "o-", ms=3, label=_label
        )
    _axes[0].axhspan(90, 150, color="0.9", label="guide target")
    _axes[0].set(xlabel="carrier frequency (GHz)", ylabel="logical splitting (MHz)")
    _axes[1].axhline(0.95, color="black", ls="--", lw=0.8)
    _axes[1].set(
        xlabel="logical splitting (MHz)", ylabel="hybridization", ylim=(0.94, 1.0)
    )
    for _ax in _axes:
        _ax.grid(alpha=0.25)
        _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(contour_a, contour_b, device):
    device_config, calibration_a, calibration_b = device.calibrate(contour_a, contour_b)
    return calibration_a, calibration_b, device_config


@app.cell
def _(
    DATA,
    device_config,
    np,
    run_single_transmon_sweep,
    save_dataset,
    workflow_config,
):
    _phase1 = workflow_config.phase1_settings()
    _ncut = list(map(int, _phase1["ncut_values"]))
    _rail_flux = np.linspace(*_phase1["rail_flux_linspace"])
    _coupler_flux = np.linspace(*_phase1["coupler_flux_linspace"])
    transmon_validation = {}
    for _name, _config in zip(
        ("A1", "A2", "C", "B1", "B2"),
        device_config.subsystems(),
        strict=True,
    ):
        _flux = _coupler_flux if _name == "C" else _rail_flux
        _report = run_single_transmon_sweep(
            _config,
            _flux,
            ncut_values=_ncut,
            convergence_tolerance_khz=float(_phase1["ncut_tolerance_khz"]),
        )
        _report.attrs["workflow_fingerprint"] = workflow_config.fingerprint(
            "model", "phase1", extra={"subsystem": _name}
        )
        save_dataset(
            _report,
            DATA / f"logical_fsim_phase1_transmon_{_name.lower()}.nc",
        )
        transmon_validation[_name] = _report
    transmon_convergence_passed = all(
        bool(_report.attrs["ncut_converged"])
        for _report in transmon_validation.values()
    )
    return (transmon_convergence_passed,)


@app.cell
def _(
    DATA,
    calibration_a,
    calibration_b,
    device,
    save_dataset,
    workflow_config,
):
    calibration_reports = {"A": calibration_a.to_dataset(), "B": calibration_b.to_dataset()}
    for _name, _report in calibration_reports.items():
        _report.attrs["workflow_fingerprint"] = workflow_config.fingerprint(
            "model", "phase1", extra={"pair": _name}
        )
        save_dataset(
            _report,
            DATA / f"logical_fsim_phase1_pair_{_name.lower()}_calibration.nc",
        )
    reduced_calibration_passed = not device.requires_direct_reference(
        calibration_a, calibration_b
    )
    _allow_reference = bool(
        workflow_config.section("phase1", "use_direct_reference_on_fit_failure")
    )
    direct_reference_circuit = (
        device.build_reference_circuit()
        if not reduced_calibration_passed and _allow_reference
        else None
    )
    static_truth_backend = (
        "direct-five-transmon"
        if direct_reference_circuit is not None
        else "reduced-five-transmon"
    )
    return (
        direct_reference_circuit,
        reduced_calibration_passed,
        static_truth_backend,
    )


@app.cell(hide_code=True)
def _(
    calibration_a,
    calibration_b,
    mo,
    reduced_calibration_passed,
    static_truth_backend,
    transmon_convergence_passed,
):
    mo.md(
        f"Phase 1 model gates: ncut convergence **{transmon_convergence_passed}**; "
        f"multi-point reduced calibration **{reduced_calibration_passed}**; "
        f"static truth backend **{static_truth_backend}**. "
        f"Maximum spectral residuals are "
        f"{calibration_a.maximum_spectral_residual_mhz:.4f} MHz (A) and "
        f"{calibration_b.maximum_spectral_residual_mhz:.4f} MHz (B)."
    )
    return


@app.cell
def _(workflow_config):
    frequency_a_grid, frequency_b_grid, coupler_flux_grid = (
        workflow_config.static_grids()
    )
    return coupler_flux_grid, frequency_a_grid, frequency_b_grid


@app.cell
def _(
    CPUS_STATIC,
    DATA,
    RECOMPUTE_STATIC,
    contour_a,
    contour_b,
    coupler_flux_grid,
    device_config,
    frequency_a_grid,
    frequency_b_grid,
    load_or_compute_static_interaction_landscape,
    workflow_config,
):
    _phase2 = workflow_config.phase2_settings()
    static_atlas, static_atlas_path = load_or_compute_static_interaction_landscape(
        device_config,
        contour_a,
        contour_b,
        frequency_a_grid,
        frequency_b_grid,
        coupler_flux_grid,
        path=DATA / "logical_fsim_static_atlas.nc",
        force=RECOMPUTE_STATIC,
        cpus=CPUS_STATIC,
        coherent_ceiling_max_time_ns=float(
            _phase2["coherent_ceiling_max_time_ns"]
        ),
        coherent_ceiling_time_samples=int(
            _phase2["coherent_ceiling_time_samples"]
        ),
    )
    return static_atlas, static_atlas_path


@app.cell
def _(
    CPUS_STATIC,
    DATA,
    LEAKAGE_GAP_TARGET_MHZ,
    SIGMA_MIN_TARGET,
    contour_a,
    contour_b,
    device_config,
    np,
    refine_static_gate_candidates,
    save_dataset,
    static_atlas,
    workflow_config,
):
    _variables = [
        "logical_detuning_mhz",
        "model_logical_detuning_mhz",
        "effective_exchange_mhz",
        "effective_zz_mhz",
        "logical_exchange_mhz",
        "logical_zz_mhz",
        "logical_subspace_sigma_min",
        "nearest_leakage_gap_mhz",
        "estimated_iswap_time_ns",
        "estimated_cz_time_ns",
        "logical_assignment_valid",
        "spectrally_isolated",
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
        "nearest_leakage_logical_state",
        "nearest_leakage_branch_label",
        "nearest_leakage_branch_probability",
    ]
    candidate_table = static_atlas[_variables].to_dataframe().reset_index()
    _qualified = (
        candidate_table.logical_assignment_valid
        & candidate_table.spectrally_isolated
        & (candidate_table.logical_subspace_sigma_min >= SIGMA_MIN_TARGET)
        & (candidate_table.nearest_leakage_gap_mhz >= LEAKAGE_GAP_TARGET_MHZ)
    )
    candidate_table["qualified"] = _qualified
    _phase2 = workflow_config.phase2_settings()
    _selection = {
        "minimum_coherent_ceiling": float(
            _phase2["minimum_coherent_ceiling"]
        ),
        "maximum_candidates": int(_phase2["candidate_count"]),
        "minimum_sigma_min": SIGMA_MIN_TARGET,
        "minimum_leakage_gap_mhz": LEAKAGE_GAP_TARGET_MHZ,
        "maximum_cz_offdiagonal_to_zz_ratio": float(
            _phase2["maximum_cz_offdiagonal_to_zz_ratio"]
        ),
    }
    iswap_static_candidate_report = refine_static_gate_candidates(
        device_config,
        contour_a,
        contour_b,
        static_atlas,
        "iswap",
        seed_candidates=int(_phase2["adaptive_seed_candidates"]),
        cpus=CPUS_STATIC,
        **_selection,
    )
    cz_static_candidate_report = refine_static_gate_candidates(
        device_config,
        contour_a,
        contour_b,
        static_atlas,
        "cz",
        seed_candidates=int(_phase2["adaptive_seed_candidates"]),
        cpus=CPUS_STATIC,
        **_selection,
    )
    for _gate, _report in (
        ("iswap", iswap_static_candidate_report),
        ("cz", cz_static_candidate_report),
    ):
        _report.attrs["workflow_fingerprint"] = workflow_config.fingerprint(
            "model", "contours", "static", "phase2", extra={"gate": _gate}
        )
        save_dataset(_report, DATA / f"logical_fsim_{_gate}_static_candidates.nc")

    _display_variables = [
        "pareto_rank",
        "gate_mechanism",
        "static_coherent_ceiling",
        "nearest_leakage_gap_mhz",
        "limiting_leakage_branch",
        "maximum_coupler_participation",
        "model_logical_detuning_mhz",
        "effective_exchange_mhz",
        "effective_zz_mhz",
        "iswap_exchange_component_mhz",
        "iswap_pair_creation_component_mhz",
        "iswap_nuisance_pauli_norm_mhz",
        "cz_offdiagonal_pauli_norm_mhz",
        "cz_offdiagonal_to_zz_ratio",
    ]

    def _display_table(report):
        available = [name for name in _display_variables if name in report]
        return report[available].to_dataframe().reset_index()

    iswap_static_candidates = _display_table(iswap_static_candidate_report)
    cz_static_candidates = _display_table(cz_static_candidate_report)
    _candidate_counts = (
        iswap_static_candidate_report.sizes["candidate"],
        cz_static_candidate_report.sizes["candidate"],
    )
    phase2_execution_gate = bool(
        all(10 <= count <= 30 for count in _candidate_counts)
        and all(
            np.all(report.static_coherent_ceiling >= 0.995)
            and np.all(report.gate_mechanism.astype(bool))
            and np.all(report.limiting_leakage_branch.astype(bool))
            for report in (
                iswap_static_candidate_report,
                cz_static_candidate_report,
            )
        )
    )
    if bool(_phase2["enforce_exit_gate"]) and not phase2_execution_gate:
        raise RuntimeError(
            "Phase 2 exit gate failed; do not advance to pulse optimization"
        )
    return candidate_table, cz_static_candidates, iswap_static_candidates


@app.cell(hide_code=True)
def _(cz_static_candidates, iswap_static_candidates, mo):
    _columns = [
        "coupler_flux",
        "logical_frequency_a_mhz",
        "logical_frequency_b_mhz",
        "model_logical_detuning_mhz",
        "effective_exchange_mhz",
        "effective_zz_mhz",
        "static_coherent_ceiling",
        "iswap_exchange_component_mhz",
        "iswap_pair_creation_component_mhz",
        "iswap_nuisance_pauli_norm_mhz",
        "cz_offdiagonal_to_zz_ratio",
        "maximum_coupler_participation",
        "nearest_leakage_gap_mhz",
        "limiting_leakage_branch",
    ]
    mo.vstack(
        [
            mo.md("## M5 — ranked static candidates\n\n### iSWAP triage"),
            mo.ui.table(iswap_static_candidates[_columns].round(5)),
            mo.md("### CPhase/CZ triage"),
            mo.ui.table(cz_static_candidates[_columns].round(5)),
        ]
    )
    return


@app.cell
def _(select_dualrail_idle_point, static_atlas, workflow_config):
    global_idle_point = select_dualrail_idle_point(static_atlas)
    # The global minimum is too remote for a d=3 fixed local pulse basis.
    _idle = workflow_config.section("idle")
    operational_idle_point = static_atlas.sel(
        logical_frequency_a_mhz=float(_idle["frequency_a_mhz"]),
        logical_frequency_b_mhz=float(_idle["frequency_b_mhz"]),
        coupler_flux=float(_idle["coupler_flux"]),
    )
    return global_idle_point, operational_idle_point


@app.cell
def _(contour_a, contour_b, replace):
    def config_at(config, frequency_a_mhz, frequency_b_mhz, coupler_flux):
        phi_a1, phi_a2 = contour_a.flux(float(frequency_a_mhz))
        phi_b1, phi_b2 = contour_b.flux(float(frequency_b_mhz))
        return replace(
            config,
            rail_a1=replace(config.rail_a1, flux=float(phi_a1)),
            rail_a2=replace(config.rail_a2, flux=float(phi_a2)),
            coupler=replace(config.coupler, flux=float(coupler_flux)),
            rail_b1=replace(config.rail_b1, flux=float(phi_b1)),
            rail_b2=replace(config.rail_b2, flux=float(phi_b2)),
        )

    return (config_at,)


@app.cell
def _(
    build_dualrail_fsim_model,
    config_at,
    device_config,
    operational_idle_point,
    replace,
    workflow_config,
):
    _phase1 = workflow_config.phase1_settings()
    _pulse_config = replace(
        device_config,
        coupler=replace(
            device_config.coupler,
            truncated_dim=int(_phase1["pulse_coupler_truncated_dim"]),
        ),
    )
    operational_idle_config = config_at(
        _pulse_config,
        operational_idle_point.logical_frequency_a_mhz.item(),
        operational_idle_point.logical_frequency_b_mhz.item(),
        operational_idle_point.coupler_flux.item(),
    )
    operational_idle_model = build_dualrail_fsim_model(operational_idle_config)
    operational_idle_static = operational_idle_model.static_interaction()
    return operational_idle_model, operational_idle_static


@app.cell(hide_code=True)
def _(global_idle_point, mo, operational_idle_point, operational_idle_static):
    mo.md(rf"""
    ## Idle selection

    Global static minimum: ({global_idle_point.logical_frequency_a_mhz.item():.1f},
    {global_idle_point.logical_frequency_b_mhz.item():.1f},
    {global_idle_point.coupler_flux.item():.3f}), residual
    {global_idle_point.idle_residual_coupling_mhz.item():.6f} MHz.

    Dynamics use the nearby detuned operational idle
    ({operational_idle_point.logical_frequency_a_mhz.item():.1f},
    {operational_idle_point.logical_frequency_b_mhz.item():.1f},
    {operational_idle_point.coupler_flux.item():.3f}). Its actual detuning is
    {operational_idle_static.logical_detuning_mhz:.3f} MHz,
    $|J_\mathrm{{ex}}|={operational_idle_static.effective_exchange_mhz:.5f}$ MHz,
    $\zeta={operational_idle_static.effective_zz_mhz:.5f}$ MHz, and
    $\sigma={operational_idle_static.logical_subspace_sigma_min:.6f}$.
    This explicit trade keeps the fixed basis local to the gate region.
    """)
    return


@app.cell
def _(candidate_table, np, workflow_config):
    # Select qualified Pareto compromises, not raw score maxima.  The iSWAP
    # point is the calibrated reduced-model resonance with modest ZZ.  The CZ
    # point keeps pair B and the coupler close to the operational idle while
    # retaining the strongest nearby conditional shift.
    iswap_gate_point = {
        key: float(value)
        for key, value in workflow_config.section("gates", "iswap").items()
    }
    cz_gate_point = {
        key: float(value)
        for key, value in workflow_config.section("gates", "cz").items()
    }
    _coordinates = [
        "logical_frequency_a_mhz",
        "logical_frequency_b_mhz",
        "coupler_flux",
    ]
    for _point in (iswap_gate_point, cz_gate_point):
        _match = candidate_table
        for _coordinate, _value in zip(_coordinates, _point.values(), strict=True):
            _match = _match[np.isclose(_match[_coordinate], _value, atol=1e-12)]
        if len(_match) != 1 or not bool(_match.iloc[0].qualified):
            raise ValueError(f"selected gate point is not qualified: {_point}")
    return cz_gate_point, iswap_gate_point


@app.cell
def _(cz_gate_point, iswap_gate_point, plt, static_atlas):
    _fig, _axes = plt.subplots(2, 3, figsize=(13, 7), dpi=170, sharex="col")
    for _row, (_point, _title) in enumerate(
        ((iswap_gate_point, "iSWAP cut"), (cz_gate_point, "CZ cut"))
    ):
        _cut = static_atlas.sel(
            logical_frequency_a_mhz=_point["frequency_a_mhz"],
            logical_frequency_b_mhz=_point["frequency_b_mhz"],
        )
        for _name, _label in (
            ("effective_exchange_mhz", r"$|J_{ex}|$"),
            ("effective_zz_mhz", r"$\zeta$"),
            ("model_logical_detuning_mhz", r"$\Delta_L$"),
        ):
            _axes[_row, 0].plot(_cut.coupler_flux, _cut[_name], label=_label)
        _axes[_row, 0].axvline(_point["coupler_flux"], color="black", ls="--")
        _axes[_row, 0].set(ylabel="MHz", title=_title)
        _axes[_row, 0].legend(frameon=False, fontsize=8)
        _axes[_row, 1].plot(
            _cut.coupler_flux, _cut.logical_subspace_sigma_min, label=r"$\sigma_{min}$"
        )
        _gap = _axes[_row, 1].twinx()
        _gap.plot(_cut.coupler_flux, _cut.nearest_leakage_gap_mhz, color="tab:red")
        _axes[_row, 1].axhline(0.85, color="0.4", ls=":")
        _axes[_row, 1].set(ylabel=r"$\sigma_{min}$", ylim=(0, 1.02))
        _gap.set_ylabel("leakage gap (MHz)", color="tab:red")
        for _term in ("XX", "YY", "ZZ", "ZI", "IZ", "XY", "YX"):
            _axes[_row, 2].plot(
                _cut.coupler_flux,
                _cut.pauli_coefficient_real_mhz.sel(pauli_term=_term),
                label=_term,
            )
        _axes[_row, 2].axvline(_point["coupler_flux"], color="black", ls="--")
        _axes[_row, 2].set(ylabel="Pauli coefficient (MHz)")
        _axes[_row, 2].legend(frameon=False, fontsize=7, ncol=2)
    for _ax in _axes[-1]:
        _ax.set_xlabel(r"coupler flux ($\Phi_0$)")
    for _ax in _axes.flat:
        _ax.grid(alpha=0.2)
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(
    DATA,
    build_dualrail_fsim_model,
    config_at,
    cz_gate_point,
    iswap_gate_point,
    np,
    operational_idle_model,
    operational_idle_point,
    save_dataset,
    track_dualrail_model_path,
    validate_fixed_basis_along_path,
    workflow_config,
):
    _phase1 = workflow_config.phase1_settings()
    _path_coordinate = np.linspace(0.0, 1.0, int(_phase1["path_samples"]))
    _idle = {
        "frequency_a_mhz": operational_idle_point.logical_frequency_a_mhz.item(),
        "frequency_b_mhz": operational_idle_point.logical_frequency_b_mhz.item(),
        "coupler_flux": operational_idle_point.coupler_flux.item(),
    }

    def _path(gate_point):
        return [
            config_at(
                operational_idle_model.config,
                frequency_a_mhz=_idle["frequency_a_mhz"]
                + _fraction
                * (gate_point["frequency_a_mhz"] - _idle["frequency_a_mhz"]),
                frequency_b_mhz=_idle["frequency_b_mhz"]
                + _fraction
                * (gate_point["frequency_b_mhz"] - _idle["frequency_b_mhz"]),
                coupler_flux=_idle["coupler_flux"]
                + _fraction * (gate_point["coupler_flux"] - _idle["coupler_flux"]),
            )
            for _fraction in _path_coordinate
        ]

    iswap_path_configs = _path(iswap_gate_point)
    cz_path_configs = _path(cz_gate_point)
    iswap_target_config = iswap_path_configs[-1]
    cz_target_config = cz_path_configs[-1]
    _iswap_static = build_dualrail_fsim_model(iswap_target_config).static_interaction()
    _cz_static = build_dualrail_fsim_model(cz_target_config).static_interaction()

    _validation_kwargs = dict(
        path_coordinate=_path_coordinate,
        levels=int(_phase1["fixed_basis_levels"]),
        minimum_state_overlap=float(_phase1["minimum_continuity_overlap"]),
        tolerance_fraction=float(_phase1["fixed_basis_tolerance_fraction"]),
    )
    iswap_basis_validation = validate_fixed_basis_along_path(
        operational_idle_model,
        iswap_path_configs,
        relevant_rate_mhz=_iswap_static.effective_exchange_mhz,
        **_validation_kwargs,
    )
    cz_basis_validation = validate_fixed_basis_along_path(
        operational_idle_model,
        cz_path_configs,
        relevant_rate_mhz=abs(_cz_static.effective_zz_mhz),
        **_validation_kwargs,
    )
    iswap_branch_tracking = track_dualrail_model_path(
        operational_idle_model,
        iswap_path_configs,
        path_coordinate=_path_coordinate,
        minimum_continuity_overlap=float(_phase1["minimum_continuity_overlap"]),
    )
    cz_branch_tracking = track_dualrail_model_path(
        operational_idle_model,
        cz_path_configs,
        path_coordinate=_path_coordinate,
        minimum_continuity_overlap=float(_phase1["minimum_continuity_overlap"]),
    )
    for _name, _report in (
        ("iswap_fixed_basis", iswap_basis_validation),
        ("cz_fixed_basis", cz_basis_validation),
        ("iswap_branch_tracking", iswap_branch_tracking),
        ("cz_branch_tracking", cz_branch_tracking),
    ):
        _report.attrs["workflow_fingerprint"] = workflow_config.fingerprint(
            "model", "phase1", "gates", "idle", extra={"report": _name}
        )
        save_dataset(_report, DATA / f"logical_fsim_phase1_{_name}.nc")
    return (
        cz_basis_validation,
        cz_branch_tracking,
        cz_target_config,
        iswap_basis_validation,
        iswap_branch_tracking,
        iswap_target_config,
    )


@app.cell
def _(
    DATA,
    asdict,
    build_dualrail_fsim_model,
    compare_static_and_dynamic_rates,
    cz_target_config,
    extract_fsim_angles,
    iswap_target_config,
    np,
    save_dataset,
    workflow_config,
    xr,
):
    _phase1 = workflow_config.phase1_settings()
    _durations = np.asarray(_phase1["rate_durations_ns"], dtype=float)
    static_dynamic_rate_reports = {}
    for _name, _target_config in (
        ("iswap", iswap_target_config),
        ("cz", cz_target_config),
    ):
        _target_model = build_dualrail_fsim_model(_target_config)
        _static = _target_model.static_interaction()
        _propagators = _target_model.static_logical_propagators(_durations)
        _angles = np.asarray(
            [extract_fsim_angles(_propagator) for _propagator in _propagators]
        )
        _comparison = compare_static_and_dynamic_rates(
            _durations,
            _propagators,
            static_hamiltonian_ghz=_static.effective_hamiltonian_ghz,
            relative_tolerance=float(_phase1["rate_tolerance_fraction"]),
            normalization_rate_mhz=(
                _static.effective_exchange_mhz
                if _name == "iswap"
                else abs(_static.effective_zz_mhz)
            ),
        )
        _report = xr.Dataset(
            data_vars={
                "swap_probability": (
                    "duration_ns",
                    _angles[:, 2],
                ),
                "conditional_phase_rad": (
                    "duration_ns",
                    _angles[:, 1],
                ),
                "projected_unitary_real": (
                    ("duration_ns", "logical_output", "logical_input"),
                    _propagators.real,
                ),
                "projected_unitary_imag": (
                    ("duration_ns", "logical_output", "logical_input"),
                    _propagators.imag,
                ),
            },
            coords={"duration_ns": _durations},
            attrs={
                **{
                    _key: int(_value) if isinstance(_value, bool) else _value
                    for _key, _value in asdict(_comparison).items()
                },
                "workflow_fingerprint": workflow_config.fingerprint(
                    "model", "phase1", "gates", extra={"gate": _name}
                ),
            },
        )
        save_dataset(_report, DATA / f"logical_fsim_phase1_{_name}_rate_check.nc")
        static_dynamic_rate_reports[_name] = _report
    static_dynamic_rates_passed = all(
        bool(_report.attrs["rates_agree"])
        for _report in static_dynamic_rate_reports.values()
    )
    return (static_dynamic_rates_passed,)


@app.cell
def _(
    cz_basis_validation,
    cz_branch_tracking,
    direct_reference_circuit,
    iswap_basis_validation,
    iswap_branch_tracking,
    reduced_calibration_passed,
    static_dynamic_rates_passed,
    transmon_convergence_passed,
    workflow_config,
):
    _checks = {
        "single-transmon ncut convergence": transmon_convergence_passed,
        "reduced fit or direct reference": (
            reduced_calibration_passed or direct_reference_circuit is not None
        ),
        "iSWAP fixed basis": bool(
            iswap_basis_validation.attrs["fixed_basis_valid"]
        ),
        "CZ fixed basis": bool(cz_basis_validation.attrs["fixed_basis_valid"]),
        "iSWAP branch continuity": bool(
            iswap_branch_tracking.attrs["continuity_valid"]
        ),
        "CZ branch continuity": bool(cz_branch_tracking.attrs["continuity_valid"]),
        "static/dynamic rate agreement": static_dynamic_rates_passed,
    }
    phase1_execution_gate = all(_checks.values())
    if (
        bool(workflow_config.section("phase1", "enforce_exit_gate"))
        and not phase1_execution_gate
    ):
        _failed = [name for name, passed in _checks.items() if not passed]
        raise RuntimeError(f"Phase 1 exit gate failed: {_failed}")
    return (phase1_execution_gate,)


@app.cell(hide_code=True)
def _(
    cz_basis_validation,
    cz_branch_tracking,
    iswap_basis_validation,
    iswap_branch_tracking,
    mo,
    phase1_execution_gate,
    static_dynamic_rates_passed,
):
    mo.md(rf"""
    ## Phase 1 — state labels and fixed-basis credibility

    | path | max logical/gap error (MHz) | minimum matched overlap | fixed basis | branch continuity |
    |---|---:|---:|---:|---:|
    | idle → iSWAP | {iswap_basis_validation.attrs["maximum_relevant_error_mhz"]:.4f} | {iswap_basis_validation.attrs["minimum_observed_overlap"]:.5f} | {bool(iswap_basis_validation.attrs["fixed_basis_valid"])} | {bool(iswap_branch_tracking.attrs["continuity_valid"])} |
    | idle → CZ | {cz_basis_validation.attrs["maximum_relevant_error_mhz"]:.4f} | {cz_basis_validation.attrs["minimum_observed_overlap"]:.5f} | {bool(cz_basis_validation.attrs["fixed_basis_valid"])} | {bool(cz_branch_tracking.attrs["continuity_valid"])} |

    Both fixed-seed diabatic labels and phase-aligned neighbor-overlap adiabatic
    branches are persisted. The sampled paths include idle, both endpoints,
    intermediate ramp points, and identify the closest leakage approach.
    Flat-top static/dynamic rates agree within 5%: **{static_dynamic_rates_passed}**.
    Overall Phase 1 execution gate: **{phase1_execution_gate}**.
    """)
    return


@app.cell
def _(iswap_basis_validation, np, plt):
    _v = iswap_basis_validation
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=170)
    _axes[0].plot(_v.path_point, np.abs(_v.logical_energy_error_mhz))
    _axes[0].set(xlabel="path coordinate", ylabel="logical energy error (MHz)")
    _axes[1].plot(_v.path_point, _v.leakage_gap_error_mhz, "o-")
    _axes[1].set(xlabel="path coordinate", ylabel="leakage-gap error (MHz)")
    for _ax in _axes:
        _ax.grid(alpha=0.2)
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(
    DATA,
    DualRailPulse,
    RECOMPUTE_DYNAMICS,
    asdict,
    contour_a,
    contour_b,
    load_compatible_pulse_sweep,
    np,
    operational_idle_model,
    phase1_execution_gate,
    pulse_results_to_dataset,
    save_dataset,
    workflow_config,
):
    _ = phase1_execution_gate
    pulse_settings = workflow_config.section("pulse")
    PULSE_PADDING_NS = float(pulse_settings["padding_ns"])
    PULSE_OUTPUT_SAMPLES = int(pulse_settings["output_samples"])
    dynamic_solver = workflow_config.solver_config()

    def load_or_run_pulse_sweep(
        path_name, gate_type, gate_point, pairs, *, method, cpus
    ):
        path = DATA / path_name
        fingerprint = workflow_config.fingerprint(
            "pulse",
            "model",
            extra=dict(
                gate_type=gate_type,
                gate_point=gate_point,
                pairs=[list(map(float, pair)) for pair in pairs],
                method=method,
                solver=asdict(dynamic_solver),
                config=asdict(operational_idle_model.config),
                contour_a=contour_a.fingerprint(),
                contour_b=contour_b.fingerprint(),
            ),
        )
        if not RECOMPUTE_DYNAMICS:
            dataset = load_compatible_pulse_sweep(path, fingerprint)
            if dataset is not None:
                return dataset

        results = []
        for rise, hold in pairs:
            pulse = DualRailPulse(
                coupler_peak_flux=gate_point["coupler_flux"],
                duration_ns=float(2 * rise + hold),
                rise_time_ns=float(rise),
                logical_frequency_a_peak_mhz=gate_point["frequency_a_mhz"],
                logical_frequency_b_peak_mhz=gate_point["frequency_b_mhz"],
                output_samples=PULSE_OUTPUT_SAMPLES,
                padding_ns=PULSE_PADDING_NS,
            )
            results.append(
                operational_idle_model.simulate(
                    pulse,
                    contour_a,
                    contour_b,
                    solver=dynamic_solver,
                    method=method,
                    cpus=cpus,
                )
            )
        dataset = pulse_results_to_dataset(
            results,
            pairs,
            gate_type=gate_type,
            fingerprint=fingerprint,
            gate_point=gate_point,
        )
        dataset["active_duration_ns"] = dataset.duration_ns
        dataset["total_duration_ns"] = (
            "candidate",
            [result.time_ns[-1] for result in results],
        )
        dataset["padding_ns"] = (
            "candidate",
            np.full(len(results), PULSE_PADDING_NS),
        )
        dataset["maximum_leakage"] = dataset.leakage.max("logical_input")
        dataset.attrs.update(method=method, padding_ns=PULSE_PADDING_NS)
        save_dataset(dataset, path)
        return dataset

    return PULSE_PADDING_NS, dynamic_solver, load_or_run_pulse_sweep


@app.cell
def _(
    CPUS_DYNAMICS,
    cz_gate_point,
    iswap_gate_point,
    load_or_run_pulse_sweep,
    workflow_config,
):
    _sweeps = workflow_config.section("pulse", "sweeps")
    iswap_exact_sweep = load_or_run_pulse_sweep(
        "logical_fsim_iswap_exact_sweep.nc",
        "iswap",
        iswap_gate_point,
        _sweeps["iswap_exact"],
        method="exact",
        cpus=1,
    )
    cz_exact_sweep = load_or_run_pulse_sweep(
        "logical_fsim_cz_exact_sweep.nc",
        "cz",
        cz_gate_point,
        _sweeps["cz_exact"],
        method="exact",
        cpus=1,
    )
    iswap_smooth_sweep = load_or_run_pulse_sweep(
        "logical_fsim_iswap_smooth_sweep.nc",
        "iswap",
        iswap_gate_point,
        _sweeps["iswap_smooth"],
        method="solver",
        cpus=CPUS_DYNAMICS,
    )
    cz_smooth_sweep = load_or_run_pulse_sweep(
        "logical_fsim_cz_smooth_sweep.nc",
        "cz",
        cz_gate_point,
        _sweeps["cz_smooth"],
        method="solver",
        cpus=CPUS_DYNAMICS,
    )
    return (
        cz_exact_sweep,
        cz_smooth_sweep,
        iswap_exact_sweep,
        iswap_smooth_sweep,
    )


@app.cell
def _(cz_smooth_sweep, iswap_smooth_sweep, np):
    best_iswap = iswap_smooth_sweep.isel(
        candidate=int(np.argmax(iswap_smooth_sweep.iswap_fidelity_with_leakage.values))
    )
    best_cz = cz_smooth_sweep.isel(
        candidate=int(np.argmax(cz_smooth_sweep.cphase_fidelity_with_leakage.values))
    )
    return best_cz, best_iswap


@app.cell
def _(
    DATA,
    FIDELITY_TARGET,
    LEAKAGE_TARGET,
    best_cz,
    best_iswap,
    save_dataset,
    workflow_config,
    xr,
):
    gate_comparison = xr.Dataset(
        data_vars=dict(
            gate_time_ns=(
                "gate",
                [best_iswap.total_duration_ns.item(), best_cz.total_duration_ns.item()],
            ),
            active_gate_time_ns=(
                "gate",
                [
                    best_iswap.active_duration_ns.item(),
                    best_cz.active_duration_ns.item(),
                ],
            ),
            padding_ns=(
                "gate",
                [best_iswap.padding_ns.item(), best_cz.padding_ns.item()],
            ),
            rise_time_ns=(
                "gate",
                [best_iswap.rise_time_ns.item(), best_cz.rise_time_ns.item()],
            ),
            hold_time_ns=(
                "gate",
                [best_iswap.hold_time_ns.item(), best_cz.hold_time_ns.item()],
            ),
            local_z_corrected_fidelity=(
                "gate",
                [
                    best_iswap.iswap_fidelity_with_leakage.item(),
                    best_cz.cphase_fidelity_with_leakage.item(),
                ],
            ),
            coherent_fidelity=(
                "gate",
                [
                    best_iswap.iswap_coherent_fidelity.item(),
                    best_cz.cphase_coherent_fidelity.item(),
                ],
            ),
            average_leakage=(
                "gate",
                [best_iswap.average_leakage.item(), best_cz.average_leakage.item()],
            ),
            maximum_leakage=(
                "gate",
                [best_iswap.maximum_leakage.item(), best_cz.maximum_leakage.item()],
            ),
            swap_probability=(
                "gate",
                [best_iswap.swap_probability.item(), best_cz.swap_probability.item()],
            ),
            theta_rad=("gate", [best_iswap.theta_rad.item(), best_cz.theta_rad.item()]),
            phi_rad=("gate", [best_iswap.phi_rad.item(), best_cz.phi_rad.item()]),
            accepted=(
                "gate",
                [
                    bool(
                        best_iswap.iswap_fidelity_with_leakage >= FIDELITY_TARGET
                        and best_iswap.average_leakage < LEAKAGE_TARGET
                    ),
                    bool(
                        best_cz.cphase_fidelity_with_leakage >= FIDELITY_TARGET
                        and best_cz.average_leakage < LEAKAGE_TARGET
                    ),
                ],
            ),
        ),
        coords={"gate": ["iswap", "cz"]},
        attrs=dict(
            description="Decision table for sampled five-transmon logical gates",
            fidelity_target=FIDELITY_TARGET,
            leakage_target=LEAKAGE_TARGET,
            workflow_fingerprint=workflow_config.fingerprint(
                "thresholds",
                "pulse",
                "gates",
                extra={
                    "iswap_sweep": best_iswap.result_fingerprint.item(),
                    "cz_sweep": best_cz.result_fingerprint.item(),
                },
            ),
        ),
    )
    gate_comparison_path = save_dataset(
        gate_comparison, DATA / "logical_fsim_gate_comparison.nc"
    )
    return (gate_comparison_path,)


@app.cell
def _(
    DATA,
    DualRailPulse,
    PULSE_PADDING_NS,
    RECOMPUTE_DYNAMICS,
    asdict,
    best_cz,
    best_iswap,
    contour_a,
    contour_b,
    cz_gate_point,
    dynamic_solver,
    iswap_gate_point,
    operational_idle_model,
    save_dataset,
    workflow_config,
    xr,
):
    trajectory_settings = workflow_config.section("trajectory")

    def load_or_run_trajectory(path_name, gate_point, pulse_row, initial_state):
        path = DATA / path_name
        pulse = DualRailPulse(
            coupler_peak_flux=gate_point["coupler_flux"],
            duration_ns=float(pulse_row.active_duration_ns),
            rise_time_ns=float(pulse_row.rise_time_ns),
            logical_frequency_a_peak_mhz=gate_point["frequency_a_mhz"],
            logical_frequency_b_peak_mhz=gate_point["frequency_b_mhz"],
            output_samples=int(trajectory_settings["output_samples"]),
            padding_ns=PULSE_PADDING_NS,
        )
        payload = dict(
            pulse=asdict(pulse),
            initial_state=initial_state,
            solver=asdict(dynamic_solver),
            config=asdict(operational_idle_model.config),
            contour_a=contour_a.fingerprint(),
            contour_b=contour_b.fingerprint(),
        )
        fingerprint = workflow_config.fingerprint(
            "trajectory", "pulse", "model", extra=payload
        )
        if path.exists() and not RECOMPUTE_DYNAMICS:
            dataset = xr.load_dataset(path)
            if dataset.attrs.get("workflow_fingerprint") == fingerprint:
                return dataset
            dataset.close()
        dataset = operational_idle_model.simulate_trajectory(
            pulse,
            contour_a,
            contour_b,
            initial_state,
            solver=dynamic_solver,
            method="solver",
        ).to_dataset()
        dataset.attrs.update(
            workflow_fingerprint=fingerprint,
            active_duration_ns=pulse.duration_ns,
            padding_ns=pulse.padding_ns,
            total_duration_ns=pulse.total_duration_ns,
        )
        save_dataset(dataset, path)
        return dataset

    initial_states = trajectory_settings["initial_states"]
    iswap_trajectories = {
        state: load_or_run_trajectory(
            f"logical_fsim_iswap_{state}_trajectory.nc",
            iswap_gate_point,
            best_iswap,
            state,
        )
        for state in initial_states
    }
    cz_trajectories = {
        state: load_or_run_trajectory(
            f"logical_fsim_cz_{state}_trajectory.nc",
            cz_gate_point,
            best_cz,
            state,
        )
        for state in initial_states
    }
    cz_11_trajectory = cz_trajectories["11"]
    iswap_01_trajectory = iswap_trajectories["01"]
    iswap_10_trajectory = iswap_trajectories["10"]
    return cz_11_trajectory, iswap_01_trajectory, iswap_10_trajectory


@app.cell
def _(
    PULSE_PADDING_NS,
    cz_11_trajectory,
    cz_exact_sweep,
    cz_smooth_sweep,
    iswap_01_trajectory,
    iswap_10_trajectory,
    iswap_exact_sweep,
    iswap_smooth_sweep,
    np,
    plt,
):
    _fig, _axes = plt.subplots(2, 4, figsize=(17, 7), dpi=170)

    def _mark_idle_padding(_ax, _trajectory):
        _end = float(_trajectory.time_ns.values[-1])
        _ax.axvspan(0.0, PULSE_PADDING_NS, color="0.85", alpha=0.45)
        _ax.axvspan(_end - PULSE_PADDING_NS, _end, color="0.85", alpha=0.45)
        _ax.axvline(PULSE_PADDING_NS, color="0.45", lw=0.7, ls=":")
        _ax.axvline(_end - PULSE_PADDING_NS, color="0.45", lw=0.7, ls=":")

    _axes[0, 0].plot(
        iswap_exact_sweep.duration_ns,
        iswap_exact_sweep.iswap_fidelity_with_leakage,
        label="fidelity",
    )
    _axes[0, 0].plot(
        iswap_exact_sweep.duration_ns,
        iswap_exact_sweep.average_leakage,
        label="leakage",
    )
    _axes[0, 0].set(title="rectangular iSWAP", xlabel="active duration (ns)")
    _axes[0, 0].legend(frameon=False)
    for _rise in np.unique(iswap_smooth_sweep.rise_time_ns):
        _cut = iswap_smooth_sweep.where(
            iswap_smooth_sweep.rise_time_ns == _rise, drop=True
        )
        _axes[0, 1].plot(
            _cut.hold_time_ns,
            _cut.iswap_fidelity_with_leakage,
            "o-",
            label=f"rise {_rise:g} ns",
        )
    _axes[0, 1].set(title="smooth iSWAP", xlabel="hold (ns)", ylabel="fidelity")
    _axes[0, 1].legend(frameon=False, fontsize=8)
    for _trajectory, _target, _label in (
        (iswap_01_trajectory, "10", "01→10"),
        (iswap_10_trajectory, "01", "10→01"),
    ):
        _axes[0, 2].plot(
            _trajectory.time_ns,
            _trajectory.logical_population.sel(logical_state=_target),
            label=_label,
        )
        _axes[0, 2].plot(_trajectory.time_ns, _trajectory.leakage, ls="--", alpha=0.6)
    _axes[0, 2].set(
        title="best iSWAP trajectories", xlabel="time (ns)", ylabel="population"
    )
    _axes[0, 2].legend(frameon=False)
    _mark_idle_padding(_axes[0, 2], iswap_01_trajectory)
    for _subsystem in iswap_01_trajectory.subsystem.values:
        _axes[0, 3].plot(
            iswap_01_trajectory.time_ns,
            iswap_01_trajectory.subsystem_flux.sel(subsystem=_subsystem),
            label=str(_subsystem),
        )
    _axes[0, 3].set(
        title="iSWAP flux trajectory",
        xlabel="time (ns)",
        ylabel=r"flux ($\Phi/\Phi_0$)",
    )
    _axes[0, 3].legend(frameon=False, fontsize=8, ncol=2)
    _mark_idle_padding(_axes[0, 3], iswap_01_trajectory)
    _axes[1, 0].plot(
        cz_exact_sweep.duration_ns,
        cz_exact_sweep.cphase_fidelity_with_leakage,
        label="fidelity",
    )
    _axes[1, 0].plot(
        cz_exact_sweep.duration_ns, cz_exact_sweep.average_leakage, label="leakage"
    )
    _axes[1, 0].set(title="rectangular CZ", xlabel="active duration (ns)")
    _axes[1, 0].legend(frameon=False)
    _axes[1, 1].plot(
        cz_smooth_sweep.rise_time_ns,
        cz_smooth_sweep.cphase_fidelity_with_leakage,
        "o-",
        label="fidelity",
    )
    _axes[1, 1].plot(
        cz_smooth_sweep.rise_time_ns,
        cz_smooth_sweep.average_leakage,
        "s--",
        label="leakage",
    )
    _axes[1, 1].set(title="CZ ramp sensitivity", xlabel="rise time (ns)")
    _axes[1, 1].legend(frameon=False)
    for _state in ("00", "01", "10", "11"):
        _axes[1, 2].plot(
            cz_11_trajectory.time_ns,
            cz_11_trajectory.logical_population.sel(logical_state=_state),
            label=_state,
        )
    _axes[1, 2].plot(
        cz_11_trajectory.time_ns, cz_11_trajectory.leakage, "k--", label="leakage"
    )
    _axes[1, 2].set(
        title="best CZ |11> trajectory", xlabel="time (ns)", ylabel="population"
    )
    _axes[1, 2].legend(frameon=False, fontsize=8, ncol=2)
    _mark_idle_padding(_axes[1, 2], cz_11_trajectory)
    for _subsystem in cz_11_trajectory.subsystem.values:
        _axes[1, 3].plot(
            cz_11_trajectory.time_ns,
            cz_11_trajectory.subsystem_flux.sel(subsystem=_subsystem),
            label=str(_subsystem),
        )
    _axes[1, 3].set(
        title="CZ flux trajectory",
        xlabel="time (ns)",
        ylabel=r"flux ($\Phi/\Phi_0$)",
    )
    _axes[1, 3].legend(frameon=False, fontsize=8, ncol=2)
    _mark_idle_padding(_axes[1, 3], cz_11_trajectory)
    for _ax in _axes.flat:
        _ax.grid(alpha=0.2)
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(best_cz, best_iswap, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=170)
    for _ax, _result, _title in (
        (_axes[0], best_iswap, "best sampled iSWAP"),
        (_axes[1], best_cz, "best sampled CZ"),
    ):
        _matrix = np.asarray(
            _result.projected_unitary_real + 1j * _result.projected_unitary_imag
        )
        _image = _ax.imshow(np.abs(_matrix), vmin=0, vmax=1, cmap="magma")
        for _row in range(4):
            for _column in range(4):
                _ax.text(
                    _column,
                    _row,
                    f"{abs(_matrix[_row, _column]):.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
        _ax.set(
            title=_title,
            xticks=range(4),
            yticks=range(4),
            xticklabels=("00", "01", "10", "11"),
            yticklabels=("00", "01", "10", "11"),
            xlabel="logical input",
            ylabel="logical output",
        )
    _fig.colorbar(_image, ax=_axes, label="projected amplitude", shrink=0.85)
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    FIDELITY_TARGET,
    LEAKAGE_TARGET,
    PULSE_PADDING_NS,
    best_cz,
    best_iswap,
    gate_comparison_path,
    mo,
    static_atlas_path,
):
    _iswap_pass = bool(
        best_iswap.iswap_fidelity_with_leakage >= FIDELITY_TARGET
        and best_iswap.average_leakage < LEAKAGE_TARGET
    )
    _cz_pass = bool(
        best_cz.cphase_fidelity_with_leakage >= FIDELITY_TARGET
        and best_cz.average_leakage < LEAKAGE_TARGET
    )
    _outcome = (
        "Both sampled gates pass the configured reporting thresholds."
        if _iswap_pass and _cz_pass
        else "At least one sampled gate fails the configured reporting thresholds."
    )
    mo.md(rf"""
    ## Decision

    | gate | total / active / rise / hold (ns) | local-Z fidelity | coherent fidelity | average leakage | swap probability | $\theta/\pi$ | $\phi/\pi$ | accepted |
    |---|---:|---:|---:|---:|---:|---:|---:|:---:|
    | iSWAP | {best_iswap.total_duration_ns.item():.1f} / {best_iswap.active_duration_ns.item():.1f} / {best_iswap.rise_time_ns.item():.1f} / {best_iswap.hold_time_ns.item():.1f} | {best_iswap.iswap_fidelity_with_leakage.item():.6f} | {best_iswap.iswap_coherent_fidelity.item():.6f} | {best_iswap.average_leakage.item():.6f} | {best_iswap.swap_probability.item():.6f} | {best_iswap.theta_rad.item() / 3.141592653589793:.4f} | {best_iswap.phi_rad.item() / 3.141592653589793:.4f} | {"yes" if _iswap_pass else "**no**"} |
    | CZ | {best_cz.total_duration_ns.item():.1f} / {best_cz.active_duration_ns.item():.1f} / {best_cz.rise_time_ns.item():.1f} / {best_cz.hold_time_ns.item():.1f} | {best_cz.cphase_fidelity_with_leakage.item():.6f} | {best_cz.cphase_coherent_fidelity.item():.6f} | {best_cz.average_leakage.item():.6f} | {best_cz.swap_probability.item():.6f} | {best_cz.theta_rad.item() / 3.141592653589793:.4f} | {best_cz.phi_rad.item() / 3.141592653589793:.4f} | {"yes" if _cz_pass else "**no**"} |

    Every simulated pulse includes {PULSE_PADDING_NS:g} ns at the operational idle point before
    and after the active waveform; total duration includes both padding windows.

    Neither gate is promoted unless it passes both {FIDELITY_TARGET:.4g} fidelity
    and {100 * LEAKAGE_TARGET:.4g}% average leakage. {_outcome} This workflow
    stops before open-system simulation and large pulse optimization. The next
    meaningful iteration is a multi-point reduced-model fit, improved
    fixed-basis convergence, and broader pair splitting controls.

    Persisted artifacts: `{static_atlas_path}`, `{gate_comparison_path}`, plus
    exact/smooth sweeps and trajectories beside them.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
