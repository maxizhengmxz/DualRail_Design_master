# Archived exploration; shared utilities remain in the repository core.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from core.runtime import configure_single_threaded_blas

    FLUX_PARAM = "Φ1"
    configure_single_threaded_blas()
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from core import (
        gen_dtc,
        gen_dt,
        gen_tdt,
        DEFAULT_PARAM_SIGMA,
        DEFAULT_PARAM_REL_SIGMA,
    )
    from legacy.analysis.monte_carlo import (
        load_or_compute_monte_carlo,
    )
    from legacy.analysis.tunable_resonator import (
        load_or_compute_tunable_resonator_monte_carlo,
        run_tunable_resonator_sample_sweep,
    )
    from pathlib import Path


    return (
        DEFAULT_PARAM_SIGMA,
        Path,
        gen_dt,
        gen_dtc,
        load_or_compute_tunable_resonator_monte_carlo,
        mo,
        np,
        plt,
        run_tunable_resonator_sample_sweep,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Systems
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DTC
    """)
    return


@app.cell
def _(gen_dtc):
    dtc_a_params = {
        "Ej1": 14,
        "Cj1": 5,
        "C1": 194,
        "Ej2": 14,
        "Cj2": 5,
        "C2": 194,
        "Ej3": 14 * 0.0405,
        "Cj3": 0.2,
        "C12": 3,
        "C1r": 15,
        "C2r": 15,
        "Cr": 385,
        "Lr": 1.56,
    }
    dtc_a = gen_dtc(**dtc_a_params)
    dtc_a.Φ1 = 0.5
    return dtc_a, dtc_a_params


@app.cell
def _(dtc_a):
    dtc_a_spectrum = dtc_a.eigenvals(evals_count=20)
    dtc_a_eigenstates = dtc_a.eigensys(evals_count=20)[1]
    return (dtc_a_spectrum,)


@app.cell
def _(dtc_a_spectrum, np):
    dtc_a_f_p0 = dtc_a_spectrum[2] - dtc_a_spectrum[0]
    dtc_a_f_m0 = dtc_a_spectrum[1] - dtc_a_spectrum[0]

    dtc_a_f_p1 = dtc_a_spectrum[8] - dtc_a_spectrum[3]
    dtc_a_f_m1 = dtc_a_spectrum[7] - dtc_a_spectrum[3]

    dtc_a_chi_p = dtc_a_f_p1 - dtc_a_f_p0
    dtc_a_chi_m = dtc_a_f_m1 - dtc_a_f_m0

    dtc_a_dchi_abs = np.abs(dtc_a_chi_p - dtc_a_chi_m)
    return dtc_a_chi_p, dtc_a_dchi_abs


@app.cell
def _(dtc_a_chi_p, dtc_a_dchi_abs, np):
    print(f"DTC parmeter set A 2*delta_chi: {dtc_a_dchi_abs*1e6:.3f} kHz")
    print(f"DTC parmeter set A 2*chi_p: {dtc_a_chi_p*1e3:.3f} MHz")
    print(f"DTC parmeter set A (delta_chi/chi_p): {dtc_a_dchi_abs/np.abs(dtc_a_chi_p):.6f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DT
    """)
    return


@app.cell
def _(gen_dt):
    dt_a_params = {
        "Ej1": 14*0.90,
        "Cj1": 5,
        "C1": 194*1.0444,
        "Ej2": 14,
        "Cj2": 5,
        "C2": 194,
        "C12": 3,
        "C1r": 15*1.0444,
        "C2r": 15,
        "Cr": 385,
        "Lr": 1.56,
    }

    dt_a = gen_dt(**dt_a_params)
    return dt_a, dt_a_params


@app.cell
def _(dt_a):
    dt_a_spectrum = dt_a.eigenvals(evals_count=20)
    dt_a_eigenstates = dt_a.eigensys(evals_count=20)[1]
    return (dt_a_spectrum,)


@app.cell
def _(dt_a_spectrum, np):
    dt_a_f_p0 = dt_a_spectrum[2] - dt_a_spectrum[0]
    dt_a_f_m0 = dt_a_spectrum[1] - dt_a_spectrum[0]

    dt_a_f_p1 = dt_a_spectrum[8] - dt_a_spectrum[3]
    dt_a_f_m1 = dt_a_spectrum[7] - dt_a_spectrum[3]

    dt_a_chi_p = dt_a_f_p1 - dt_a_f_p0
    dt_a_chi_m = dt_a_f_m1 - dt_a_f_m0

    dt_a_dchi_abs = np.abs(dt_a_chi_p - dt_a_chi_m)
    return dt_a_chi_p, dt_a_dchi_abs


@app.cell
def _(dt_a_chi_p, dt_a_dchi_abs, np):
    print(f"DT parmeter set A 2*delta_chi: {dt_a_dchi_abs*1e6:.3f} kHz")
    print(f"DT parmeter set A 2*chi_p: {dt_a_chi_p*1e3:.3f} MHz")
    print(f"DT parmeter set A (delta_chi/chi_p): {dt_a_dchi_abs/np.abs(dt_a_chi_p):.6f}")
    return


@app.cell
def _():
    TUNABLE_MC_RECOMPUTE = False
    TUNABLE_MC_SAMPLES = 200
    TUNABLE_MC_CPUS = 8
    TUNABLE_MC_SEED = 20260701
    TUNABLE_MC_EVALS_COUNT = 12
    LR_BOUNDS = (0.8, 2.5)
    LR_GRID_SIZE = 21
    LR_XATOL = 1e-5
    return (
        LR_BOUNDS,
        LR_GRID_SIZE,
        LR_XATOL,
        TUNABLE_MC_CPUS,
        TUNABLE_MC_EVALS_COUNT,
        TUNABLE_MC_SAMPLES,
        TUNABLE_MC_SEED,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DTC Monte Carlo
    """)
    return


@app.cell
def _(dt_a_params, dtc_a_params):
    dtc_a_poi = [
        {"name": "poi1", "params": dtc_a_params, "flux": 0.5},
    ]

    dt_a_poi = [
        {"name": "poi1", "params": dt_a_params, "flux": 0.0},
    ]
    return (dtc_a_poi,)


@app.cell
def _(DEFAULT_PARAM_SIGMA):
    dtc_a_param_sigma = DEFAULT_PARAM_SIGMA.copy()
    dtc_a_param_rel_sigma = {
        "Ej1": 0.8*0.4,
        "Cj1": 0.8*0.4,
        "C1": 0.4*0.4,
        "Ej2": 0.8*0.4,
        "Cj2": 0.8*0.4,
        "C2": 0.4*0.4,
        "Ej3": 0.8*0.4,
        "Cj3": 0.8*0.4,
        "C12": 0.4*0.4,
        "C1r": 0.4*0.4,
        "C2r": 0.4*0.4,
        "Cr": 0.0,
        "Lr": 0.0,
    }

    dtc_a_param_correlation_groups = [
        {"name": "junction_energy", "params": ["Ej1", "Ej2", "Ej3"], "correlation": 0.8},
        {"name": "junction_capacitance", "params": ["Cj1", "Cj2", "Cj3"], "correlation": 0.8},
        {
            "name": "fabricated_capacitance",
            "params": ["C1", "C2", "C12", "C1r", "C2r", "Cr"],
            "correlation": 0.8,
        },
    ]

    dt_a_param_sigma = DEFAULT_PARAM_SIGMA.copy()
    dt_a_param_rel_sigma = {
        "Ej1": 0.8*0.4,
        "Cj1": 0.8*0.4,
        "C1": 0.4*0.4,
        "Ej2": 0.8*0.4,
        "Cj2": 0.8*0.4,
        "C2": 0.4*0.4,
        "C12": 0.4*0.4,
        "C1r": 0.4*0.4,
        "C2r": 0.4*0.4,
        "Cr": 0.0,
        "Lr": 0.0,
    }
    dt_a_param_correlation_groups = [
        {"name": "junction_energy", "params": ["Ej1", "Ej2"], "correlation": 0.8},
        {"name": "junction_capacitance", "params": ["Cj1", "Cj2"], "correlation": 0.8},
        {
            "name": "fabricated_capacitance",
            "params": ["C1", "C2", "C12", "C1r", "C2r", "Cr"],
            "correlation": 0.8,
        },
    ]
    return (
        dtc_a_param_correlation_groups,
        dtc_a_param_rel_sigma,
        dtc_a_param_sigma,
    )


@app.cell
def _(
    LR_BOUNDS,
    LR_GRID_SIZE,
    LR_XATOL,
    Path,
    TUNABLE_MC_CPUS,
    TUNABLE_MC_EVALS_COUNT,
    TUNABLE_MC_SAMPLES,
    TUNABLE_MC_SEED,
    dtc_a_param_correlation_groups,
    dtc_a_param_rel_sigma,
    dtc_a_param_sigma,
    dtc_a_poi,
    gen_dtc,
    load_or_compute_tunable_resonator_monte_carlo,
):
    dtc_tunable_ds, dtc_tunable_path = load_or_compute_tunable_resonator_monte_carlo(
        dtc_a_poi,
        path=Path("data") / "monte_carlo_dtc_a_tunable_lr_sigma_0p32_updated_tracking.nc",
        force=False,
        samples=TUNABLE_MC_SAMPLES,
        cpus=TUNABLE_MC_CPUS,
        seed=TUNABLE_MC_SEED,
        evals_count=TUNABLE_MC_EVALS_COUNT,
        param_sigma=dtc_a_param_sigma,
        param_rel_sigma=dtc_a_param_rel_sigma,
        param_correlation_groups=dtc_a_param_correlation_groups,
        circuit_factory=gen_dtc,
        flux_param="Φ1",
        lr_bounds=LR_BOUNDS,
        grid_size=LR_GRID_SIZE,
        xatol=LR_XATOL,
    )
    return (dtc_tunable_ds,)


@app.cell
def _():
    # dt_tunable_ds, dt_tunable_path = load_or_compute_tunable_resonator_monte_carlo(
    #     dt_a_poi,
    #     path=Path("data") / "monte_carlo_dt_a_tunable_lr_sigma_0p32.nc",
    #     force=TUNABLE_MC_RECOMPUTE,
    #     samples=TUNABLE_MC_SAMPLES,
    #     cpus=TUNABLE_MC_CPUS,
    #     seed=TUNABLE_MC_SEED,
    #     evals_count=TUNABLE_MC_EVALS_COUNT,
    #     param_sigma=dt_a_param_sigma,
    #     param_rel_sigma=dt_a_param_rel_sigma,
    #     param_correlation_groups=dt_a_param_correlation_groups,
    #     circuit_factory=gen_dt,
    #     flux_param=None,
    #     lr_bounds=LR_BOUNDS,
    #     grid_size=LR_GRID_SIZE,
    #     xatol=LR_XATOL,
    # )
    return


@app.cell
def _(dtc_tunable_ds, plt):
    _dtc_ok = dtc_tunable_ds.where(dtc_tunable_ds.optimizer_success, drop=True)
    # _dt_ok = dt_tunable_ds.where(dt_tunable_ds.optimizer_success, drop=True)


    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=150)
    _ax.hist(_dtc_ok.optimal_lr.values.ravel(), bins=40, alpha=0.5, label="DTC")
    # _ax.hist(_dt_ok.optimal_lr.values.ravel(), bins=40, alpha=0.5, label="DT")
    _ax.set_xlabel(r"Optimized $L_r$ (nH)")
    _ax.set_ylabel("Count")
    _ax.legend()
    plt.show()
    return


@app.cell
def _(dtc_tunable_ds):
    _dtc_ok = dtc_tunable_ds.where(dtc_tunable_ds.optimizer_success, drop=True)
    _dtc_ok
    return


@app.cell
def _(dtc_tunable_ds, np, plt):
    _dtc_ok = dtc_tunable_ds.where(dtc_tunable_ds.optimizer_success, drop=True)
    # _dt_ok = dt_tunable_ds.where(dt_tunable_ds.optimizer_success, drop=True)

    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=150)
    _ax.hist(
        np.abs(_dtc_ok.original_delta_chi.values.ravel()/_dtc_ok.original_chi_p.values.ravel()),
        bins=list(np.linspace(0, 1.0, 40)),
        alpha=0.5,
        label="DTC",
    )
    # _ax.hist(_dt_ok.optimal_lr.values.ravel(), bins=40, alpha=0.5, label="DT")
    # _ax.set_xlabel(r"Optimized $L_r$ (nH)")
    _ax.set_ylabel("Count")
    _ax.legend()
    plt.show()
    return


@app.cell
def _(dtc_tunable_ds, np, plt):
    _dtc_ok = dtc_tunable_ds.where(dtc_tunable_ds.optimizer_success, drop=True)
    # _dt_ok = dt_tunable_ds.where(dt_tunable_ds.optimizer_success, drop=True)

    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=150)
    _ax.hist(
        _dtc_ok.abs_delta_chi_over_chi_p.values.ravel(),
        bins=list(np.linspace(0, 1.0, 40)),
        alpha=0.5,
        label="DTC"
    )
    # _ax.hist(_dt_ok.optimal_lr.values.ravel(), bins=40, alpha=0.5, label="DT")
    # _ax.set_xlabel(r"Optimized $L_r$ (nH)")
    _ax.set_ylabel("Count")
    _ax.legend()
    plt.show()
    return


@app.cell
def _(dtc_tunable_ds, np, plt):
    _dtc_ratio = dtc_tunable_ds.abs_delta_chi_over_chi_p.where(
        dtc_tunable_ds.optimizer_success, drop=True
    ).values.ravel()
    # _dt_ratio = dt_tunable_ds.abs_delta_chi_over_chi_p.where(
    #     dt_tunable_ds.optimizer_success, drop=True
    # ).values.ravel()

    _upper = max(np.nanmax(_dtc_ratio), 0)
    _bins = np.linspace(-0.05, _upper, 101)
    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=150)
    _ax.hist(_dtc_ratio, bins=_bins, cumulative=True, density=True, histtype="step", label="DTC")
    # _ax.hist(_dt_ratio, bins=_bins, cumulative=True, density=True, histtype="step", label="DT")
    _ax.set_xlabel(r"$|\Delta\chi/\chi_p|$")
    _ax.set_ylabel("Cumulative probability")
    _ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Add logic to avoid crossing
    - DASI
    - Case study, energy landscape vs. cavity f
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DTC Case Study
    """)
    return


@app.cell
def _(
    TUNABLE_MC_EVALS_COUNT,
    dtc_a_param_correlation_groups,
    dtc_a_param_sigma,
    dtc_a_params,
    dtc_tunable_ds,
    gen_dtc,
    np,
    run_tunable_resonator_sample_sweep,
):
    single_sample_lr_values = np.linspace(1.35, 2.6, 81)

    case_idx_list = [
        1, 4, 6, 8,
        10, 11, 13, 14, 17,
        18, 19, 21, 25, 26,
        27, 29, 30, 31, 32
    ]
    case_ds_list = [
        run_tunable_resonator_sample_sweep(
            dtc_a_params,
            single_sample_lr_values,
            seed=2313,
            evals_count=TUNABLE_MC_EVALS_COUNT,
            param_sigma=dtc_a_param_sigma,
            param_rel_sigma={
                "Ej1": 0.0,
                "Cj1": 0.0,
                "C1": 0.0,
                "Ej2": 0.0,
                "Cj2": 0.0,
                "C2": 0.0,
                "Ej3": 0.0,
                "Cj3": 0.0,
                "C12": 0.0,
                "C1r": 0.0,
                "C2r": 0.0,
                "Cr": 0.0,
                "Lr": 0.0,
            },
            param_correlation_groups=dtc_a_param_correlation_groups,
            circuit_factory=gen_dtc,
            flux=0.5,
            flux_param="Φ1",
        )
    ]
    for _idx in case_idx_list:
        _case_params = dict(zip(
            dtc_tunable_ds.param.values,
            dtc_tunable_ds.param_value[0, _idx].values,
        ))

        _single_sample_sweep_ds = run_tunable_resonator_sample_sweep(
            _case_params,
            single_sample_lr_values,
            seed=2313,
            evals_count=TUNABLE_MC_EVALS_COUNT,
            param_sigma=dtc_a_param_sigma,
            param_rel_sigma={},
            param_correlation_groups=dtc_a_param_correlation_groups,
            circuit_factory=gen_dtc,
            flux=0.5,
            flux_param="Φ1",
        )

        case_ds_list.append(_single_sample_sweep_ds)
    return (case_ds_list,)


@app.cell
def _(case_ds_list, plt):
    fig_labels = [
        "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O",
        "P", "Q", "R", "S", "T",
    ]
    _fig, _ax = plt.subplots(ncols=5, nrows=4, figsize=(9, 6), dpi=300, sharex=True)

    _fig.subplots_adjust(
        wspace=0.4,
    )
    for _idx, _ds in enumerate(case_ds_list):
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.tracked_energy[:, 1] - _ds.tracked_energy[:, 0],
        )
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.tracked_energy[:, 2] - _ds.tracked_energy[:, 0],
        )

        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.tracked_energy[:, 3] - _ds.tracked_energy[:, 0],
        )
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.tracked_energy[:, 4] - _ds.tracked_energy[:, 0],
        )
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.tracked_energy[:, 5] - _ds.tracked_energy[:, 0],
        )
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.tracked_energy[:, 6] - _ds.tracked_energy[:, 0],
        )
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.tracked_energy[:, 7] - _ds.tracked_energy[:, 0],
        )
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.tracked_energy[:, 8] - _ds.tracked_energy[:, 0],
        )
        if _idx % 5 == 0:
            _ax[_idx // 5, _idx % 5].set_ylabel(r"f (GHz)")

        if _idx > 14:
            _ax[_idx // 5, _idx % 5].set_xlabel(r"$f_r$ (GHz)")
        # _ax[_idx // 5, _idx % 5].set_xlabel("Resonator frequency (GHz)")
        # _ax[_idx // 5, _idx % 5].set_ylabel(r"$|2\Delta\chi/2\chi_p|$")
        _ax[_idx // 5, _idx % 5].annotate(
            fig_labels[_idx],
            xy=(-0.2, 1.05),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
        )
    _fig.align_labels()
    _fig.savefig("figs/dtc_dt_tunable_r_case_spectrum_updated_tracking.png", dpi=300, bbox_inches="tight")
    plt.show()
    return (fig_labels,)


@app.cell
def _(case_ds_list, fig_labels, plt):
    _fig, _ax = plt.subplots(ncols=5, nrows=4, figsize=(10, 6), dpi=300, sharex=True)
    _fig.subplots_adjust(
        wspace=0.4,
    )
    for _idx, _ds in enumerate(case_ds_list):
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.chi_p*1e3,
        )
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.chi_m*1e3,
        )

        if _idx % 5 == 0:
            _ax[_idx // 5, _idx % 5].set_ylabel(r"$2\chi$ (MHz)")

        if _idx > 14:
            _ax[_idx // 5, _idx % 5].set_xlabel(r"$f_r$ (GHz)")
        # _ax[_idx // 5, _idx % 5].set_xlabel("Resonator frequency (GHz)")
        # _ax[_idx // 5, _idx % 5].set_ylabel(r"$|2\Delta\chi/2\chi_p|$")

        _ax[_idx // 5, _idx % 5].annotate(
            fig_labels[_idx],
            xy=(-0.2, 1.05),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
        )
    _fig.align_labels()
    _fig.savefig("figs/dtc_dt_tunable_r_case_chi_updated_tracking.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(case_ds_list, fig_labels, plt):
    _fig, _ax = plt.subplots(ncols=5, nrows=4, figsize=(10, 6), dpi=300, sharex=True)
    _fig.subplots_adjust(
        wspace=0.4,
    )
    for _idx, _ds in enumerate(case_ds_list):
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.delta_chi*1e3
        )
        _ax[_idx // 5, _idx % 5].hlines([0], _ds.resonator_frequency_ghz[0], _ds.resonator_frequency_ghz[-1], colors="r", linestyles="--")
        if _idx % 5 == 0:
            _ax[_idx // 5, _idx % 5].set_ylabel(r"$2\Delta\chi$ (MHz)")

        if _idx > 14:
            _ax[_idx // 5, _idx % 5].set_xlabel(r"$f_r$ (GHz)")
        # _ax[_idx // 5, _idx % 5].set_xlabel("Resonator frequency (GHz)")
        # _ax[_idx // 5, _idx % 5].set_ylabel(r"$|2\Delta\chi/2\chi_p|$")

        _ax[_idx // 5, _idx % 5].annotate(
            fig_labels[_idx],
            xy=(-0.2, 1.05),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
        )
    _fig.align_labels()
    _fig.savefig("figs/dtc_dt_tunable_r_case_delta_chi_updated_tracking.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(case_ds_list, fig_labels, plt):
    _fig, _ax = plt.subplots(ncols=5, nrows=4, figsize=(10, 6), dpi=300, sharex=True)
    _fig.subplots_adjust(
        wspace=0.4,
    )
    for _idx, _ds in enumerate(case_ds_list):
        _ax[_idx // 5, _idx % 5].plot(
            _ds.resonator_frequency_ghz,
            _ds.delta_chi/_ds.chi_p*100,
        )
        _ax[_idx // 5, _idx % 5].hlines(
            [0],
            _ds.resonator_frequency_ghz[0],
            _ds.resonator_frequency_ghz[-1],
            colors="r",
            linestyles="--",
        )
        if _idx % 5 == 0:
            _ax[_idx // 5, _idx % 5].set_ylabel(r"$2\Delta\chi/2\chi_p$ (%)")

        if _idx > 14:
            _ax[_idx // 5, _idx % 5].set_xlabel(r"$f_r$ (GHz)")
        # _ax[_idx // 5, _idx % 5].set_xlabel("Resonator frequency (GHz)")
        # _ax[_idx // 5, _idx % 5].set_ylabel(r"$|2\Delta\chi/2\chi_p|$")

        _ax[_idx // 5, _idx % 5].annotate(
            fig_labels[_idx],
            xy=(-0.2, 1.05),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
        )
    _fig.align_labels()
    _fig.savefig("figs/dtc_dt_tunable_r_case_delta_chi_over_chi_p_updated_tracking.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(case_ds_list, np, plt):
    _baseline_params = case_ds_list[0].sampled_param.values
    _params_sigma = np.array([(_ds.sampled_param.values - _baseline_params)/_baseline_params for _ds in case_ds_list])

    _fig, _ax = plt.subplots(figsize=(8, 5), dpi=300)
    _ax.pcolormesh(
        _params_sigma.T[-3::-1, :],
        vmin=-0.5,
        vmax=0.5,
        cmap="RdBu",
    )
    # print(_params_sigma.shape)
    for _i in range(_params_sigma.shape[0]):
        for _j in range(_params_sigma.shape[1]-2):
            if np.abs(_params_sigma.T[-3-_j, _i]) > 0.25:
                _c = "w"
            else:
                _c = "k"
            _ax.text(
                _i + 0.5,
                _j + 0.5,
                f"{_params_sigma.T[-3-_j, _i]*100:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=_c,
            )

    # _ax.text(0.5, 0.5, f"{_params_sigma.T[-3, 0]*100:.1f}", ha="center", va="center", fontsize=8)
    # _ax.text(0.5, 1.5, f"{_params_sigma.T[-4, 0]*100:.1f}", ha="center", va="center", fontsize=8)
    # _ax.text(1.5, 0.5, f"{_params_sigma.T[-3, 1]*100:.1f}", ha="center", va="center", fontsize=8)
    # _ax.text(1.5, 1.5, f"{_params_sigma.T[-4, 1]*100:.1f}", ha="center", va="center", fontsize=8)
    # _ax.text(1.5, 2.5, f"{_params_sigma.T[-5, 1]*100:.1f}", ha="center", va="center", fontsize=8)
    # _ax.text(x=19.5, y=10.5, s=f"{_params_sigma.T[-13, 19]*100:.1f}", ha="center", va="center", fontsize=8)
    _ax.set_yticks(np.linspace(0.5, 10.5, 11))
    _ax.set_yticklabels(case_ds_list[0].param.values[-3::-1])
    _ax.set_xticks(np.linspace(0.5, 19.5, 20))
    _ax.set_xticklabels(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"])
    # _ax.grid()
    _ax.hlines(np.linspace(0, 11, 12), 0, 20, colors="k", linestyles="-", linewidth=0.5)
    _ax.vlines(np.linspace(0, 20, 21), 0, 11, colors="k", linestyles="-", linewidth=0.5)
    _fig.savefig("figs/dtc_dt_tunable_r_case_param_sigma_updated_tracking.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(dtc_a):
    eigen_states = dtc_a.eigensys(evals_count=20)[1]
    return (eigen_states,)


@app.cell
def _(dtc_a):
    dtc_a.operators_by_name.keys()
    return


@app.cell
def _(dtc_a, eigen_states, np):
    np.abs(eigen_states.conj().T @ dtc_a.Q3_operator() @ eigen_states)
    return


@app.cell
def _(dtc_a, eigen_states, np):
    np.abs(eigen_states.conj().T @ (dtc_a.n1_operator() + dtc_a.n2_operator()) @ eigen_states)
    return


@app.cell
def _(dtc_a, eigen_states, np):
    np.abs(eigen_states.conj().T @ (dtc_a.n2_operator() - dtc_a.n1_operator()) @ eigen_states)
    return


if __name__ == "__main__":
    app.run()
