import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from dtc_dualrail.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from dtc_dualrail import (
        DEFAULT_PARAM_SIGMA,
        DEFAULT_PARAM_REL_SIGMA,
        gen_dtc,
        load_or_compute_monte_carlo,
        gen_tdt,
        track_levels_by_overlap,
        find_operating_points,
        load_or_compute_operating_points,
        load_or_compute_operating_point_dispersive_sweep,
    )


    return (
        gen_tdt,
        load_or_compute_operating_point_dispersive_sweep,
        load_or_compute_operating_points,
        np,
        plt,
    )


@app.cell
def _():
    poi1 = {
        "Ej1": 12.2,
        "Cj1": 5,
        "Ej1_2": 7.8,
        "Cj1_2": 5,
        "C1": 304,
        "Ej2": 12.2,
        "Cj2": 5,
        "Ej2_2": 7.8,
        "Cj2_2": 5,
        "C2": 334,
        "C12": 2.4,
        "C1r": 15,
        "C2r": 15,
        "Cr": 379,
        "Lr": 1.4,
    }

    poi2 = {
        "Ej1": 12.2,
        "Cj1": 5,
        "Ej1_2": 7.8,
        "Cj1_2": 5,
        "C1": 304,
        "Ej2": 12.2,
        "Cj2": 5,
        "Ej2_2": 7.8,
        "Cj2_2": 5,
        "C2": 334,
        "C12": 2.4,
        "C1r": 14,
        "C2r": 15,
        "Cr": 379,
        "Lr": 1.4,
    }
    return (poi1,)


@app.cell
def _(gen_tdt, poi1):
    aws_tdt = gen_tdt(**poi1)
    return (aws_tdt,)


@app.cell
def _(np):
    flux_list = np.linspace(0, 1, 101)
    return


@app.cell
def _(aws_tdt, load_or_compute_operating_points, np):
    operating_points_ds, operating_points_path = (
        load_or_compute_operating_points(
            aws_tdt,
            phi_1_values=np.linspace(0.1, 0.4, 31),
            phi_2_values=np.linspace(0.1, 0.4, 301),
            path="data/aws_tdt_operating_points_4.nc",
            force=False,
            evals_count=6,
            cpus=8,
        )
    )
    return (operating_points_ds,)


@app.cell
def _(operating_points_ds):
    operating_points_ds
    return


@app.cell
def _(aws_tdt, np, operating_points_ds, plt):
    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=300)

    for _phi_1, _phi_2 in zip(operating_points_ds.phi_1.values[::20], operating_points_ds.phi_2.values[::20]):

        aws_tdt.Φ2 = _phi_2
        _q1_flux_sweep = aws_tdt.get_spectrum_vs_paramvals(
            param_name="Φ1",
            param_vals=np.linspace(_phi_1-0.01, _phi_1+0.01, 21),
            evals_count=20,
            num_cpus=8,
            get_eigenstates=True,
        )

        _ax.plot(
            np.linspace(_phi_1-0.01, _phi_1+0.01, 21),
            _q1_flux_sweep.energy_table[:, 1]-_q1_flux_sweep.energy_table[:, 0],
            color="C0",
        )

        _ax.plot(
            np.linspace(_phi_1-0.01, _phi_1+0.01, 21),
            _q1_flux_sweep.energy_table[:, 2]-_q1_flux_sweep.energy_table[:, 0],
            color="C1",
        )
    _ax.set_xlabel(r"$\phi_\text{ext}$")
    _ax.set_ylabel(r"$f$ (GHz)")
    _ax.grid()
    plt.show()
    return


@app.cell
def _(
    gen_tdt,
    load_or_compute_operating_point_dispersive_sweep,
    operating_points_ds,
    poi1,
):
    dispersive_ds, dispersive_path = (
        load_or_compute_operating_point_dispersive_sweep(
            operating_points_ds,
            poi1,
            circuit_factory=gen_tdt,
            path="data/aws_tdt_operating_point_dispersive_4.nc",
            force=True,
            evals_count=24,
            cpus=8,
        )
    )
    return (dispersive_ds,)


@app.cell
def _(dispersive_ds, operating_points_ds, plt):
    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=300)
    _ax.plot(
        operating_points_ds.operating_frequency_ghz,
        dispersive_ds.chi_p*1e3,
        label=r"$\chi_p$",
    )
    _ax.plot(
        operating_points_ds.operating_frequency_ghz,
        dispersive_ds.chi_m*1e3,
        label=r"$\chi_m$",
    )
    _ax.legend()
    _ax.set_xlabel("Operating Frequency (GHz)")
    _ax.set_ylabel(r"$\chi$ (MHz)")
    _ax.grid()
    plt.show()
    return


@app.cell
def _(dispersive_ds, operating_points_ds, plt):
    _fig, _ax = plt.subplots(figsize=(4, 3), dpi=300)
    _ax.plot(
        operating_points_ds.operating_frequency_ghz[:],
        dispersive_ds.abs_delta_chi_over_chi_p[:],
    )
    _ax.vlines(
        [operating_points_ds.operating_frequency_ghz[[96, 170]]],
        0,
        0.01,
        colors="k",
        linestyles="dashed",
    )
    _ax.text(
        operating_points_ds.operating_frequency_ghz[96] + 0.1,
        dispersive_ds.abs_delta_chi_over_chi_p[96] - 0.005,
        f"{operating_points_ds.operating_frequency_ghz[96].values:.3f}",
        horizontalalignment='left',
        verticalalignment='center',
    )

    _ax.text(
        operating_points_ds.operating_frequency_ghz[170] - 0.1,
        dispersive_ds.abs_delta_chi_over_chi_p[170] - 0.005,
        f"{operating_points_ds.operating_frequency_ghz[170].values:.3f}",
        horizontalalignment='right',
        verticalalignment='center',
    )
    _ax.set_yscale("log")
    _ax.grid()
    _ax.set_xlabel("Operating Frequency (GHz)")
    _ax.set_ylabel(r"$|\Delta \chi / \chi_p|$")

    plt.show()
    return


@app.cell
def _(dispersive_ds):
    dispersive_ds.abs_delta_chi_over_chi_p[[96, 170]]
    return


@app.cell
def _(operating_points_ds):
    operating_points_ds.phi_1[86]
    return


@app.cell
def _(operating_points_ds):
    operating_points_ds.phi_2[86]
    return


if __name__ == "__main__":
    app.run()
