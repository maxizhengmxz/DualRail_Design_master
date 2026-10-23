# Archived exploration; shared utilities remain in the repository core.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from core import (
        DEFAULT_PARAM_SIGMA,
        DEFAULT_PARAM_REL_SIGMA,
        gen_dtc,
    )
    from legacy.analysis.monte_carlo import (
        load_or_compute_monte_carlo,
    )


    return DEFAULT_PARAM_SIGMA, gen_dtc, load_or_compute_monte_carlo, plt


@app.cell
def _():
    poi1 = {
        "Ej1": 14,
        "Cj1": 5,
        "C1": 194,
        "Ej2": 14,
        "Cj2": 5,
        "C2": 194,
        "Ej3": 14 * 0.04,
        "Cj3": 0.2,
        "C12": 3,
        "C1r": 15,
        "C2r": 15,
        "Cr": 385,
        "Lr": 1.56,
    }

    phi_ext_poi1 = 0.5

    poi2 = {
        "Ej1": 14,
        "Cj1": 5,
        "C1": 194,
        "Ej2": 14,
        "Cj2": 5,
        "C2": 194 * 0.956,
        "Ej3": 14 * 0.04,
        "Cj3": 0.2,
        "C12": 3,
        "C1r": 15,
        "C2r": 15,
        "Cr": 385,
        "Lr": 1.56,
    }

    phi_ext_poi2 = 0.0

    poi_list = [
        {"name": "poi1", "params": poi1, "flux": phi_ext_poi1},
        {"name": "poi2", "params": poi2, "flux": phi_ext_poi2},
    ]
    return (poi_list,)


@app.cell
def _(DEFAULT_PARAM_SIGMA, gen_dtc):
    MONTE_CARLO_RECOMPUTE = True
    MONTE_CARLO_SAMPLES = 200
    MONTE_CARLO_CPUS = 8
    MONTE_CARLO_SEED = 20260528
    MONTE_CARLO_EVALS_COUNT = 12
    CIRCUIT_FACTORY = gen_dtc
    PARAM_SIGMA = DEFAULT_PARAM_SIGMA.copy()
    PARAM_REL_SIGMA = {
        "Ej1": 0.1,
        "Cj1": 0.05,
        "C1": 0.05,
        "Ej2": 0.1,
        "Cj2": 0.05,
        "C2": 0.05,
        "Ej3": 0.1,
        "Cj3": 0.05,
        "C12": 0.05,
        "C1r": 0.05,
        "C2r": 0.05,
        "Cr": 0.01,
        "Lr": 0.01,
    }
    PARAM_CORRELATION_GROUPS = [
        {"name": "junction_energy", "params": ["Ej1", "Ej2", "Ej3"], "correlation": 0.8},
        {"name": "junction_capacitance", "params": ["Cj1", "Cj2", "Cj3"], "correlation": 0.8},
        {
            "name": "fabricated_capacitance",
            "params": ["C1", "C2", "C12", "C1r", "C2r", "Cr"],
            "correlation": 0.9,
        },
    ]
    return (
        CIRCUIT_FACTORY,
        MONTE_CARLO_CPUS,
        MONTE_CARLO_EVALS_COUNT,
        MONTE_CARLO_RECOMPUTE,
        MONTE_CARLO_SAMPLES,
        MONTE_CARLO_SEED,
        PARAM_CORRELATION_GROUPS,
        PARAM_REL_SIGMA,
        PARAM_SIGMA,
    )


@app.cell
def _(
    CIRCUIT_FACTORY,
    MONTE_CARLO_CPUS,
    MONTE_CARLO_EVALS_COUNT,
    MONTE_CARLO_RECOMPUTE,
    MONTE_CARLO_SAMPLES,
    MONTE_CARLO_SEED,
    PARAM_CORRELATION_GROUPS,
    PARAM_REL_SIGMA,
    PARAM_SIGMA,
    load_or_compute_monte_carlo,
    poi_list,
):
    monte_carlo_ds, monte_carlo_path = load_or_compute_monte_carlo(
        poi_list,
        force=MONTE_CARLO_RECOMPUTE,
        samples=MONTE_CARLO_SAMPLES,
        cpus=MONTE_CARLO_CPUS,
        seed=MONTE_CARLO_SEED,
        evals_count=MONTE_CARLO_EVALS_COUNT,
        param_sigma=PARAM_SIGMA,
        param_rel_sigma=PARAM_REL_SIGMA,
        param_correlation_groups=PARAM_CORRELATION_GROUPS,
        circuit_factory=CIRCUIT_FACTORY,
    )
    return (monte_carlo_ds,)


@app.cell
def _(monte_carlo_ds):
    monte_carlo_ds
    return


@app.cell
def _(monte_carlo_ds, plt):
    for poi_name in monte_carlo_ds["poi"].values:
        values = monte_carlo_ds["delta_chi"].sel(poi=poi_name).values/monte_carlo_ds["chi_p"].sel(poi=poi_name).values
        plt.hist(values, bins=40, alpha=0.5, label=str(poi_name))

    plt.xlabel(r"$|\Delta\chi|$ (MHz)")
    plt.ylabel("count")
    plt.legend()
    plt.show()
    return


@app.cell
def _(monte_carlo_ds, plt):
    for _poi_name in monte_carlo_ds["poi"].values:
        _values = monte_carlo_ds["delta_chi"].sel(poi=_poi_name).values
        plt.hist(_values, bins=40, alpha=0.5, label=str(_poi_name))

    plt.xlabel(r"$|\Delta\chi|$ (MHz)")
    plt.ylabel("count")
    plt.legend()
    plt.show()
    return


@app.cell
def _(monte_carlo_ds, plt):
    for _poi_name in monte_carlo_ds["poi"].values:
        _values = monte_carlo_ds["chi_p"].sel(poi=_poi_name).values
        plt.hist(_values, bins=40, alpha=0.5, label=str(_poi_name))

    plt.xlabel(r"$|\Delta\chi|$ (MHz)")
    plt.ylabel("count")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
