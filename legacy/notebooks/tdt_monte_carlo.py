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
        gen_tdt,
    )
    from legacy.analysis.monte_carlo import (
        load_or_compute_monte_carlo,
    )
    from legacy.analysis.tdt_operating_point_monte_carlo import (
        load_or_compute_tdt_operating_point_monte_carlo,
    )
    from pathlib import Path


    return (
        DEFAULT_PARAM_SIGMA,
        gen_tdt,
        load_or_compute_tdt_operating_point_monte_carlo,
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

    phi_ext_poi1 = 0.5

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

    phi_ext_poi2 = 0.0

    poi_list = [
        {"name": "poi1", "params": poi1, "flux": phi_ext_poi1},
        # {"name": "poi2", "params": poi2, "flux": phi_ext_poi2},
    ]
    return (poi1,)


@app.cell
def _(DEFAULT_PARAM_SIGMA, gen_tdt):
    MONTE_CARLO_RECOMPUTE = False
    MONTE_CARLO_SAMPLES = 40
    MONTE_CARLO_CPUS = 8
    MONTE_CARLO_SEED = 20260528
    MONTE_CARLO_EVALS_COUNT = 40
    CIRCUIT_FACTORY = gen_tdt
    PARAM_SIGMA = DEFAULT_PARAM_SIGMA.copy()
    PARAM_REL_SIGMA = {
        "Ej1": 0.8*0.4,
        "Cj1": 0.8*0.4,
        "Ej1_2": 0.8*0.4,
        "Cj1_2": 0.8*0.4,
        "C1": 0.4*0.4,
        "Ej2": 0.8*0.4,
        "Cj2": 0.8*0.4,
        "Ej2_2": 0.8*0.4,
        "Cj2_2": 0.8*0.4,
        "C2": 0.4*0.4,
        "C12": 0.4*0.4,
        "C1r": 0.4*0.4,
        "C2r": 0.4*0.4,
        "Cr": 0.0,
        "Lr": 0.0,
    }
    PARAM_CORRELATION_GROUPS = [
        {"name": "junction_energy", "params": ["Ej1", "Ej2", "Ej1_2", "Ej2_2"], "correlation": 0.8},
        {"name": "junction_capacitance", "params": ["Cj1", "Cj2", "Cj1_2", "Cj2_2"], "correlation": 0.8},
        {
            "name": "fabricated_capacitance",
            "params": ["C1", "C2", "C12", "C1r", "C2r"],
            "correlation": 0.9,
        },
    ]
    return PARAM_CORRELATION_GROUPS, PARAM_REL_SIGMA


@app.cell
def _(
    PARAM_CORRELATION_GROUPS,
    PARAM_REL_SIGMA,
    load_or_compute_tdt_operating_point_monte_carlo,
    np,
    poi1,
):
    dataset, path = load_or_compute_tdt_operating_point_monte_carlo(
        poi1,
        phi_1_values=np.linspace(0.0, 0.5, 51),
        phi_2_values=np.linspace(0.0, 0.5, 501),
        path="data/tdt_operating_point_monte_carlo.nc",
        force=False,
        samples=40,
        seed=20260528,
        param_rel_sigma=PARAM_REL_SIGMA,
        param_correlation_groups=PARAM_CORRELATION_GROUPS,
        operating_point_evals_count=6,
        dispersive_evals_count=24,
        min_hybridization=0.95,
        cpus=8,
    )
    return (dataset,)


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset, np):
    plot_sample_values = dataset["sample"].values
    if plot_sample_values.size != 40:
        raise ValueError(
            "the 5 x 8 plot grid requires exactly 40 Monte Carlo samples; "
            f"the dataset contains {plot_sample_values.size}"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        abs_delta_chi_over_chi_p = np.abs(
            dataset["delta_chi"] / dataset["chi_p"]
        )
    return abs_delta_chi_over_chi_p, plot_sample_values


@app.cell
def _(abs_delta_chi_over_chi_p, dataset, np, plot_sample_values, plt):
    _fig, _axes = plt.subplots(
        8,
        5,
        figsize=(15, 20),
        dpi=150,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    _sample_index = dataset["sample_index"].values
    _frequency = dataset["operating_frequency_ghz"].values
    _ratio = abs_delta_chi_over_chi_p.values

    for _axis, _sample_value in zip(
        _axes.flat,
        plot_sample_values,
        strict=True,
    ):
        _sample_mask = _sample_index == _sample_value
        _finite = (
            _sample_mask
            & np.isfinite(_frequency)
            & np.isfinite(_ratio)
            & (_ratio > 0)
        )

        if np.any(_finite):
            _order = np.argsort(_frequency[_finite])
            _axis.plot(
                _frequency[_finite][_order],
                _ratio[_finite][_order],
                linewidth=0.7,
                marker=".",
                markersize=1.5,
            )
        else:
            _axis.text(
                0.5,
                0.5,
                "no finite points",
                ha="center",
                va="center",
                transform=_axis.transAxes,
            )   
        _axis.set_xlim(2.0, 6.0)
        _axis.set_yscale("log")
        _axis.set_title(
            f"Sample {int(_sample_value)} "
            f"(n={int(np.count_nonzero(_sample_mask))})",
            fontsize=8,
        )
        _axis.grid(True, which="both", alpha=0.25)
        _axis.tick_params(labelsize=7)

    _fig.suptitle(
        r"TDT Monte Carlo: $|\Delta\chi/\chi_p|$ at dual-rail operating points",
        fontsize=14,
    )
    _fig.supxlabel("Operating frequency (GHz)")
    _fig.supylabel(r"$|\Delta\chi/\chi_p|$")
    plt.show()
    _fig.savefig("figs/tdt_case_study.png", dpi=300, bbox_inches="tight")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
