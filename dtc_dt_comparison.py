import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    from dtc_dualrail.runtime import configure_single_threaded_blas

    FLUX_PARAM = "Φ1"
    configure_single_threaded_blas()
    return (FLUX_PARAM,)


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from dtc_dualrail import (
        gen_dtc,
        gen_dt,
        gen_tdt,
        DEFAULT_PARAM_SIGMA,
        DEFAULT_PARAM_REL_SIGMA,
        load_or_compute_monte_carlo,
    )
    from pathlib import Path


    return (
        DEFAULT_PARAM_SIGMA,
        Path,
        gen_dt,
        gen_dtc,
        load_or_compute_monte_carlo,
        mo,
        np,
        plt,
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
    return dtc_a_eigenstates, dtc_a_spectrum


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


@app.cell
def _(FLUX_PARAM, dtc_a, np):
    sweep = dtc_a.get_spectrum_vs_paramvals(
        param_name=FLUX_PARAM,
        param_vals=np.linspace(0.0, 1.0, 101),
        evals_count=20,
    )
    return (sweep,)


@app.cell
def _(np, plt, sweep):
    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=300)
    _ax.plot(
        np.linspace(0.0, 1.0, 101),
        (sweep.energy_table[:, 1] - sweep.energy_table[:, 0]),
        label="- mode",
        linewidth=4,
    )
    _ax.plot(
        np.linspace(0.0, 1.0, 101),
        (sweep.energy_table[:, 2] - sweep.energy_table[:, 0]),
        label="+ mode",
        linewidth=4,
    )
    _ax.set_xlabel(r"$\Phi_{\text{ext}}/2 \pi$")
    _ax.set_ylabel("f (GHz)")
    _ax.grid()
    _ax.legend(loc="lower left")
    _ax.set_title("Mode Frequency")

    _fig.savefig("dtc_logical_spectrum.png", dpi=300, bbox_inches="tight")

    plt.show()
    return


@app.cell
def _(np, plt, sweep):
    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=300)
    _ax.plot(
        np.linspace(0.0, 1.0, 101),
        (sweep.energy_table[:, 7] - sweep.energy_table[:, 3] - (sweep.energy_table[:, 1] - sweep.energy_table[:, 0]))*1e3,
        label="- mode",
        linewidth=4,
    )
    _ax.plot(
        np.linspace(0.0, 1.0, 101),
        (sweep.energy_table[:, 8] - sweep.energy_table[:, 3] - (sweep.energy_table[:, 2] - sweep.energy_table[:, 0]))*1e3,
        label="+ mode",
        linewidth=4,
    )

    _ax.set_xlabel(r"$\Phi_{\text{ext}}/2 \pi$")
    _ax.set_ylabel(r"$2\chi$ MHz)")
    _ax.grid()

    _ax.legend(loc="upper center")
    _ax.set_title("Mode Dispersive Shift")

    _fig.savefig("dtc_logical_chi.png", dpi=300, bbox_inches="tight")

    plt.show()

    return


@app.cell
def _(dtc_a_spectrum, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    # for i in range(6):
    _ax.hlines(dtc_a_spectrum[:8], 0, 1,linewidth=1, colors=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"])
    _ax.set_axis_off()
    _ax.set_xlim(-1, 2)
    plt.show()
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tunable DT
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Noise Resilience
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Erasure
    """)
    return


@app.cell
def _(dtc_a_eigenstates):
    dtc_a_g0 = dtc_a_eigenstates[:, 0]
    dtc_a_m0 = dtc_a_eigenstates[:, 1]
    dtc_a_p0 = dtc_a_eigenstates[:, 2]

    dtc_a_g1 = dtc_a_eigenstates[:, 3]
    dtc_a_m1 = dtc_a_eigenstates[:, 7]
    dtc_a_p1 = dtc_a_eigenstates[:, 8]
    return dtc_a_m0, dtc_a_m1


@app.cell
def _(dtc_a, dtc_a_m0, dtc_a_m1, np):
    np.abs(dtc_a_m0 @ dtc_a.n1_operator() @dtc_a_m1.conj())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Design Robustness
    """)
    return


@app.cell
def _():
    MONTE_CARLO_RECOMPUTE = False
    MONTE_CARLO_SAMPLES = 10000
    MONTE_CARLO_CPUS = 8
    MONTE_CARLO_SEED = 20260625
    MONTE_CARLO_EVALS_COUNT = 12
    return


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
    return dt_a_poi, dtc_a_poi


@app.cell
def _(DEFAULT_PARAM_SIGMA):
    dtc_a_param_sigma = DEFAULT_PARAM_SIGMA.copy()
    dtc_a_param_rel_sigma = {
        "Ej1": 0.8,
        "Cj1": 0.8,
        "C1": 0.4,
        "Ej2": 0.8,
        "Cj2": 0.8,
        "C2": 0.4,
        "Ej3": 0.8,
        "Cj3": 0.8,
        "C12": 0.4,
        "C1r": 0.4,
        "C2r": 0.4,
        "Cr": 0.01,
        "Lr": 0.01,
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
    return


@app.cell
def _(Path, dtc_a_poi, load_or_compute_monte_carlo):
    mc_dtc_ds_files = [
        "monte_carlo_dtc_a_factor_1",
        "monte_carlo_dtc_a_factor_0p9",
        "monte_carlo_dtc_a_factor_0p8",
        "monte_carlo_dtc_a_factor_0p7",
        "monte_carlo_dtc_a_factor_0p6",
        "monte_carlo_dtc_a_factor_0p5",
        "monte_carlo_dtc_a_factor_0p4",
        "monte_carlo_dtc_a_factor_0p3",
        "monte_carlo_dtc_a_factor_0p2",
        "monte_carlo_dtc_a_factor_0p1",
    ]

    dtc_ds_all = [load_or_compute_monte_carlo(dtc_a_poi, path=Path("data") / file, force=False)[0] for file in mc_dtc_ds_files]
    return (dtc_ds_all,)


@app.cell
def _(Path, dt_a_poi, load_or_compute_monte_carlo):
    mc_dt_ds_files = [
        "monte_carlo_dt_a_factor_1",
        "monte_carlo_dt_a_factor_0p9",
        "monte_carlo_dt_a_factor_0p8",
        "monte_carlo_dt_a_factor_0p7",
        "monte_carlo_dt_a_factor_0p6",
        "monte_carlo_dt_a_factor_0p5",
        "monte_carlo_dt_a_factor_0p4",
        "monte_carlo_dt_a_factor_0p3",
        "monte_carlo_dt_a_factor_0p2",
        "monte_carlo_dt_a_factor_0p1",
    ]

    dt_ds_all = [load_or_compute_monte_carlo(dt_a_poi, path=Path("data") / file, force=False)[0] for file in mc_dt_ds_files]
    return (dt_ds_all,)


@app.cell
def _(dtc_ds_all):
    dtc_chi_p_all = [_ds["chi_p"].sel(poi="poi1").values*1000 for _ds in dtc_ds_all]
    dtc_chi_m_all = [_ds["chi_m"].sel(poi="poi1").values*1000 for _ds in dtc_ds_all]
    dtc_delta_chi_all = [_ds["delta_chi"].sel(poi="poi1").values*1000 for _ds in dtc_ds_all]
    return dtc_chi_m_all, dtc_chi_p_all, dtc_delta_chi_all


@app.cell
def _(dt_ds_all):
    dt_chi_p_all = [_ds["chi_p"].sel(poi="poi1").values*1000 for _ds in dt_ds_all]
    dt_chi_m_all = [_ds["chi_m"].sel(poi="poi1").values*1000 for _ds in dt_ds_all]
    dt_delta_chi_all = [_ds["delta_chi"].sel(poi="poi1").values*1000 for _ds in dt_ds_all]
    return dt_chi_m_all, dt_chi_p_all, dt_delta_chi_all


@app.cell
def _(dt_chi_m_all, dt_chi_p_all, dt_delta_chi_all, np, plt):
    _idx = 9

    _fig, _ax = plt.subplots(ncols=4, figsize=(12, 2))
    _ax[0].hist(dt_chi_p_all[_idx], bins=list(np.linspace(-10, 10, 401)), alpha=0.5, label="chi_p")
    _ax[0].hist(dt_chi_m_all[_idx], bins=list(np.linspace(-10, 10, 401)), alpha=0.5, label="chi_m")

    _ax[1].hist(dt_delta_chi_all[_idx], bins=list(np.linspace(-3, 3, 401)), alpha=0.5, label="delta_chi")

    _ax[2].hist(dt_delta_chi_all[_idx]/dt_chi_p_all[_idx], bins=list(np.linspace(-1, 1, 401)), alpha=0.5, label="|delta_chi|/|chi_p|")

    _ax[3].hist(np.abs(dt_delta_chi_all[_idx]/dt_chi_p_all[_idx]), bins=list(np.linspace(0, 0.5, 401)), alpha=0.5, label="chi_p", cumulative=True, density=False)

    _ax[0].set_xlabel(r"$2\chi$ (MHz)")
    _ax[0].set_ylabel("count")
    _ax[0].legend()
    _ax[1].set_xlabel(r"$2\Delta\chi$ (MHz)")
    _ax[1].set_ylabel("count")
    _ax[1].legend()
    _ax[2].set_xlabel(r"$\Delta\chi/\chi_p$")
    plt.show()
    return


@app.cell
def _(dtc_chi_m_all, dtc_chi_p_all, dtc_delta_chi_all, np, plt):
    _idx = 9

    _fig, _ax = plt.subplots(ncols=4, figsize=(12, 2))
    _ax[0].hist(dtc_chi_p_all[_idx], bins=list(np.linspace(-10, 10, 401)), alpha=0.5, label="chi_p")
    _ax[0].hist(dtc_chi_m_all[_idx], bins=list(np.linspace(-10, 10, 401)), alpha=0.5, label="chi_m")

    _ax[1].hist(dtc_delta_chi_all[_idx], bins=list(np.linspace(-3, 3, 401)), alpha=0.5, label="delta_chi")

    _ax[2].hist(dtc_delta_chi_all[_idx]/dtc_chi_p_all[_idx], bins=list(np.linspace(-1, 1, 401)), alpha=0.5, label="|delta_chi|/|chi_p|")
    _ax[3].hist(np.abs(dtc_delta_chi_all[_idx]/dtc_chi_p_all[_idx]), bins=list(np.linspace(0, 0.5, 401)), alpha=0.5, label="chi_p", cumulative=True, density=False)

    _ax[0].set_xlabel(r"$2\chi$ (MHz)")
    _ax[0].set_ylabel("count")
    _ax[0].legend()
    _ax[1].set_xlabel(r"$2\Delta\chi$ (MHz)")
    _ax[1].set_ylabel("count")
    _ax[1].legend()
    _ax[2].set_xlabel(r"$\Delta\chi/\chi_p$")
    plt.show()
    return


@app.cell
def _():
    jj_sigma_list = [0.4*1, 0.4*0.9, 0.4*0.8, 0.4*0.7, 0.4*0.6, 0.4*0.5, 0.4*0.4, 0.4*0.3, 0.4*0.2, 0.4*0.1]
    c_sigma_list = [0.2*1, 0.2*0.9, 0.2*0.8, 0.2*0.7, 0.2*0.6, 0.2*0.5, 0.2*0.4, 0.2*0.3, 0.2*0.2, 0.2*0.1]
    return c_sigma_list, jj_sigma_list


@app.cell
def _(
    c_sigma_list,
    dt_chi_p_all,
    dt_delta_chi_all,
    dtc_chi_p_all,
    dtc_delta_chi_all,
    jj_sigma_list,
    np,
    plt,
):
    _idx = 7
    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=300)
    _ax.hist(
        np.abs(dtc_delta_chi_all[_idx]/dtc_chi_p_all[_idx]),
        bins=list(np.linspace(0, 0.9, 401)),
        alpha=0.5,
        cumulative=True,
        density=False,
        label="Inductive",
    )
    _ax.hist(
        np.abs(dt_delta_chi_all[_idx]/dt_chi_p_all[_idx]),
        bins=list(np.linspace(0, 0.9, 401)),
        alpha=0.5,
        cumulative=True,
        density=False,
        label="Capacitive",
    )

    _ax.legend(loc="lower right")
    _ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
    _ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _ax.set_xlabel(r"$|\Delta\chi/\chi_p|$")
    _ax.set_ylabel("Cumulative Probability")
    _ax.set_title(rf"Junction parameter: $\pm {jj_sigma_list[_idx]*100:.2f}\%$" + "\n" + f"planar capacitance: $\pm {c_sigma_list[_idx]*100:.2f}\%$")
    plt.show()
    return


@app.cell
def _(
    c_sigma_list,
    dt_chi_p_all,
    dt_delta_chi_all,
    dtc_chi_p_all,
    dtc_delta_chi_all,
    jj_sigma_list,
    np,
    plt,
):
    for _idx in range(10):
        _fig, _ax = plt.subplots(figsize=(4, 3), dpi=300)
        _ax.hist(
            np.abs(dtc_delta_chi_all[_idx]/dtc_chi_p_all[_idx]),
            bins=list(np.linspace(0, 0.9, 401)),
            alpha=0.5,
            cumulative=True,
            density=False,
            label="Inductive",
        )
        _ax.hist(
            np.abs(dt_delta_chi_all[_idx]/dt_chi_p_all[_idx]),
            bins=list(np.linspace(0, 0.9, 401)),
            alpha=0.5,
            cumulative=True,
            density=False,
            label="Capacitive",
        )

        _ax.legend(loc="lower right")
        _ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
        _ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        _ax.set_xlabel(r"$|\Delta\chi/\chi_p|$")
        _ax.set_ylabel("Cumulative Probability")
        _ax.set_title(rf"Junction parameter: $\pm {jj_sigma_list[_idx]*100:.2f}\%$" + "\n" + f"planar capacitance: $\pm {c_sigma_list[_idx]*100:.2f}\%$")
        _fig.savefig(f"figs/dtc_dt_comparison_cdf_{_idx}.png", dpi=300, bbox_inches="tight")
    return


@app.cell
def _(dtc_ds_all, plt):
    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=300)
    _ax.hist(dtc_ds_all[0]["param_value"].sel(poi="poi1")[:, 5], bins=40, alpha=0.5, label="Inductive")
    return


if __name__ == "__main__":
    app.run()
