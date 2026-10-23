# Archived exploration; shared utilities remain in the repository core.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np
    import scqubits as scq
    import xarray as xr

    from scipy.signal import argrelmin
    from scipy.optimize import linear_sum_assignment
    from pathlib import Path
    from legacy.analysis.sweeps import (
        load_or_compute_two_param_sweep,
        run_two_param_sweep,
    )

    return (
        Path,
        colors,
        linear_sum_assignment,
        load_or_compute_two_param_sweep,
        mo,
        np,
        plt,
        run_two_param_sweep,
        scq,
        xr,
    )


@app.cell
def _(np, scq):
    def gen_dtc(
        Ej1:float = 14,
        Cj1:float = 5,
        C1:float = 194,
        Ej2:float = 14,
        Cj2:float = 5,
        C2:float = 194,
        Ej3:float = 2,
        Cj3:float = 0.2,
        C12:float = 3,
        C1r:float = 15,
        C2r:float = 15,
        Cr:float = 385,
        Lr:float = 1.56,
    ):

        dtc_yaml = f"""# dtc-dual-rail
            branches:
            - [JJ, 0,1, Ej1={Ej1}GHz, Cj1={Cj1}fF]
            - [C, 0,1, C1={C1}MHz]
            ### 
            - [JJ, 0,2, Ej2={Ej2}GHz, Cj2={Cj2}fF]
            - [C, 0,2, C2={C2}MHz]
            ###
            - [JJ, 1,2, Ej3={Ej3}nA, Cj3={Cj3}fF]
            - [C, 1,2, C12={C12}fF]
            ###
            - [C, 1,3, {C1r}fF]
            - [C, 2,3, {C2r}fF]
            ###
            - [C, 3,0, {Cr}fF]
            - [L, 3,0, {Lr}nH]
        """

        dtc = scq.Circuit(dtc_yaml, from_file=False, ext_basis="harmonic")

        trans_mat = np.array([
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        dtc.cutoff_n_1 = 20
        dtc.cutoff_n_2 = 20
        dtc.cutoff_ext_3 = 20
        system_hierarchy = [[1,2], [3]]

        dtc.configure(
            transformation_matrix=trans_mat,
            system_hierarchy=system_hierarchy,
            subsystem_trunc_dims=[40, 18],
        )

        return dtc

    return (gen_dtc,)


@app.cell
def _(Path, linear_sum_assignment, np, xr):
    DATA_DIR = Path("data")
    FLUX_PARAM = "Φ1"
    BRANCH_LABELS = np.array(
        ["ground", "plus", "minus", "ground_partner", "plus_partner", "minus_partner"]
    )

    def _state_matrix(state_entry) -> np.ndarray:
        states = np.asarray(state_entry)
        if states.ndim != 2:
            raise ValueError("state entries must be 2D arrays")
        return states

    def track_levels_by_overlap(
        state_table,
        energy_table: np.ndarray | None = None,
        *,
        q3_operator=None,
        reference_order=None,
    ):
        """Select reference branches and their strongest Q3-coupled partners."""
        states = [_state_matrix(state_entry) for state_entry in state_table]
        if len(states) == 0:
            raise ValueError("state_table must contain at least one flux point")
        if q3_operator is None:
            raise ValueError("q3_operator is required for Q3-based tracking")

        q3_matrix = q3_operator
        level_count = states[0].shape[1]
        reference_order = (
            np.arange(min(3, level_count))
            if reference_order is None
            else np.asarray(reference_order, dtype=int)
        )
        if reference_order.ndim != 1 or reference_order.size == 0:
            raise ValueError("reference_order must contain at least one level index")
        if np.unique(reference_order).size != reference_order.size:
            raise ValueError("reference_order entries must be unique")
        if np.any(reference_order < 0) or np.any(reference_order >= level_count):
            raise ValueError("reference_order contains invalid level indices")

        candidate_levels = np.setdiff1d(np.arange(level_count), reference_order)
        if candidate_levels.size < reference_order.size:
            raise ValueError("not enough non-reference levels to select Q3-coupled partners")

        branch_count = 2 * reference_order.size
        raw_level = np.zeros((len(states), branch_count), dtype=int)
        overlap = np.ones((len(states), branch_count))

        for flux_idx, current in enumerate(states):
            if current.shape[1] != level_count:
                raise ValueError("all state entries must contain the same number of eigenstates")

            q3_overlap = np.abs(current.conj().T @ q3_matrix @ current) ** 2
            ref_rows, partner_cols = linear_sum_assignment(
                -q3_overlap[np.ix_(reference_order, candidate_levels)]
            )
            partner_levels = np.empty(reference_order.size, dtype=int)
            partner_strength = np.empty(reference_order.size)
            partner_levels[ref_rows] = candidate_levels[partner_cols]
            partner_strength[ref_rows] = q3_overlap[
                reference_order[ref_rows],
                partner_levels[ref_rows],
            ]

            ordered_reference = reference_order
            if reference_order.size >= 3 and q3_overlap[0, 2] < q3_overlap[0, 1]:
                ordered_reference = reference_order.copy()
                ordered_reference[[1, 2]] = ordered_reference[[2, 1]]
                partner_levels[[1, 2]] = partner_levels[[2, 1]]
                partner_strength[[1, 2]] = partner_strength[[2, 1]]

            raw_level[flux_idx] = np.concatenate((ordered_reference, partner_levels))
            overlap[flux_idx] = np.concatenate(
                (np.ones(reference_order.size), partner_strength)
            )

        if energy_table is None:
            return raw_level, overlap

        energy_table = np.asarray(energy_table)
        tracked_energy = np.take_along_axis(energy_table, raw_level, axis=1)
        return raw_level, overlap, tracked_energy

    def _branch_coord(branch_count: int) -> np.ndarray:
        if branch_count <= BRANCH_LABELS.size:
            return BRANCH_LABELS[:branch_count]
        return np.asarray([f"branch_{idx}" for idx in range(branch_count)])

    def _chi_from_tracked_energy(tracked_energy: xr.DataArray):
        if "branch" not in tracked_energy.dims:
            raise ValueError("tracked_energy must include a branch dimension")
        if tracked_energy.sizes["branch"] < 6:
            raise ValueError("tracked chi calculation requires 6 tracked branches")

        ground = tracked_energy.isel(branch=0)
        plus = tracked_energy.isel(branch=1)
        minus = tracked_energy.isel(branch=2)
        ground_partner = tracked_energy.isel(branch=3)
        plus_partner = tracked_energy.isel(branch=4)
        minus_partner = tracked_energy.isel(branch=5)

        chi_p = plus_partner - ground_partner - plus + ground
        chi_m = minus_partner - ground_partner - minus + ground
        return chi_p, chi_m

    def add_tracked_chi(ds: xr.Dataset) -> xr.Dataset:
        chi_p, chi_m = _chi_from_tracked_energy(ds["tracked_energy"])
        delta_chi = chi_p - chi_m
        out = ds.assign(
            chi_p=chi_p,
            chi_m=chi_m,
            delta_chi=delta_chi,
        )
        out["chi_p"].attrs["units"] = "GHz"
        out["chi_m"].attrs["units"] = "GHz"
        out["delta_chi"].attrs["units"] = "GHz"
        return out

    def spectrum_to_dataset(
        energy,
        flux_values,
        *,
        tracked_energy=None,
        tracked_level=None,
        partner_overlap=None,
        sweep_dim: str | None = None,
        sweep_values=None,
        extra_coords=None,
        flux_param: str = FLUX_PARAM,
        description: str = "DTC spectrum sweep.",
    ) -> xr.Dataset:
        energy = np.asarray(energy)
        prefix_dims = () if sweep_dim is None else (sweep_dim,)
        dims = (*prefix_dims, "flux", "level")
        coords = {
            "flux": np.asarray(flux_values),
            "level": np.arange(energy.shape[-1]),
        }
        if sweep_dim is not None:
            coords[sweep_dim] = np.asarray(sweep_values)
        if extra_coords is not None:
            coords.update(extra_coords)

        data_vars = {
            "energy": (dims, energy),
        }
        if tracked_energy is not None:
            tracked_energy = np.asarray(tracked_energy)
            branch_count = tracked_energy.shape[-1]
            coords["branch"] = _branch_coord(branch_count)
            tracked_dims = (*prefix_dims, "flux", "branch")
            data_vars.update(
                {
                    "tracked_energy": (tracked_dims, tracked_energy),
                    "tracked_level": (tracked_dims, np.asarray(tracked_level)),
                    "partner_overlap": (tracked_dims, np.asarray(partner_overlap)),
                }
            )

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "flux_param": flux_param,
                "energy_units": "GHz",
                "description": description,
            },
        )
        if "tracked_energy" in ds:
            ds = add_tracked_chi(ds)
        return ds

    def flux_sweep_to_dataset(
        spectrum_sweep,
        flux_values,
        *,
        flux_param: str = FLUX_PARAM,
        q3_operator=None,
    ) -> xr.Dataset:
        energy = np.asarray(spectrum_sweep.energy_table)
        tracked_kwargs = {}
        if getattr(spectrum_sweep, "state_table", None) is not None:
            tracked_level, partner_overlap, tracked_energy = track_levels_by_overlap(
                spectrum_sweep.state_table,
                energy,
                q3_operator=q3_operator,
            )
            tracked_kwargs = {
                "tracked_energy": tracked_energy,
                "tracked_level": tracked_level,
                "partner_overlap": partner_overlap,
            }

        return spectrum_to_dataset(
            energy,
            flux_values,
            flux_param=flux_param,
            description="DTC spectrum sweep versus external flux.",
            **tracked_kwargs,
        )



    def save_dataset(ds: xr.Dataset, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)
        return path

    def load_or_compute_dataset(path, compute, *, force: bool = False):
        path = Path(path)
        if path.exists() and not force:
            return xr.load_dataset(path), path

        ds = compute()
        return ds, save_dataset(ds, path)

    return (
        DATA_DIR,
        FLUX_PARAM,
        flux_sweep_to_dataset,
        load_or_compute_dataset,
        spectrum_to_dataset,
        track_levels_by_overlap,
    )


@app.cell
def _(gen_dtc):
    dtc = gen_dtc()
    return (dtc,)


@app.cell
def _(np):
    flux_list = np.linspace(0,0.5,51)
    return (flux_list,)


@app.cell
def _(
    DATA_DIR,
    dtc,
    flux_list,
    flux_sweep_to_dataset,
    load_or_compute_dataset,
):
    RECOMPUTE_FLUX_SWEEP = True
    TRACK_FLUX_SWEEP_LEVELS = True

    def _compute_flux_sweep():
        _spectrum_sweep = dtc.get_spectrum_vs_paramvals(
            "Φ1",
            flux_list,
            evals_count=12,
            get_eigenstates=TRACK_FLUX_SWEEP_LEVELS,
            num_cpus=8,
        )
        return flux_sweep_to_dataset(
            _spectrum_sweep,
            flux_list,
            q3_operator=dtc.Q3_operator(),
        )

    flux_sweep_ds, flux_sweep_path = load_or_compute_dataset(
        DATA_DIR / "flux_sweep_new.nc",
        _compute_flux_sweep,
        force=RECOMPUTE_FLUX_SWEEP,
    )
    return (flux_sweep_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##
    """)
    return


@app.cell
def _(flux_sweep_ds, plt):
    tracked_energy = flux_sweep_ds["tracked_energy"].values

    for _i in range(tracked_energy.shape[1]):
        plt.plot(tracked_energy[:, _i] - tracked_energy[:, 0])
        # plt.plot(spectrum_data.energy_table[:, _i] - spectrum_data.energy_table[:, 0] + 0.1, linestyle="dashed")

    # plt.ylim(3.7, 4.3)
    plt.show()
    return (tracked_energy,)


@app.cell
def _(plt, tracked_energy):
    plt.plot(tracked_energy[:, 4] - tracked_energy[:, 3] - tracked_energy[:, 1] + tracked_energy[:, 0])
    plt.plot(tracked_energy[:, 5] - tracked_energy[:, 3] - tracked_energy[:, 2] + tracked_energy[:, 0])
    # plt.plot(np.abs(tracked_energy[:, 4] - tracked_energy[:, 1] - tracked_energy[:, 5] + tracked_energy[:, 2]), linestyle="dashed")
    # plt.ylim(3.7, 4.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coupling C sweep
    """)
    return


@app.cell
def _(FLUX_PARAM, np, spectrum_to_dataset, track_levels_by_overlap, xr):
    def run_res_coupling_sweep(
        gen_dtc,
        flux_values,
        beta_values,
        *,
        c1r: float = 15,
        evals_count: int = 12,
        flux_param: str = FLUX_PARAM,
        get_eigenstates: bool = False,
    ) -> xr.Dataset:
        energy_tables = []
        tracked_level_tables = []
        partner_overlap_tables = []
        tracked_energy_tables = []

        for beta in beta_values:
            circuit = gen_dtc(C1r=c1r, C2r=c1r * beta)
            sweep = circuit.get_spectrum_vs_paramvals(
                flux_param,
                flux_values,
                evals_count=evals_count,
                get_eigenstates=get_eigenstates,
                num_cpus=8,
            )
            energy_table = np.asarray(sweep.energy_table)
            energy_tables.append(energy_table)

            if get_eigenstates:
                tracked_level, partner_overlap, tracked_energy = track_levels_by_overlap(
                    sweep.state_table,
                    energy_table,
                    q3_operator=circuit.Q3_operator(),
                )
                tracked_level_tables.append(tracked_level)
                partner_overlap_tables.append(partner_overlap)
                tracked_energy_tables.append(tracked_energy)

        tracked_kwargs = {}
        if get_eigenstates:
            tracked_kwargs = {
                "tracked_energy": np.asarray(tracked_energy_tables),
                "tracked_level": np.asarray(tracked_level_tables),
                "partner_overlap": np.asarray(partner_overlap_tables),
            }

        ds = spectrum_to_dataset(
            np.asarray(energy_tables),
            flux_values,
            sweep_dim="beta",
            sweep_values=beta_values,
            extra_coords={"C2r": ("beta", c1r * np.asarray(beta_values))},
            flux_param=flux_param,
            description="DTC resonator-coupling and flux spectrum sweep.",
            **tracked_kwargs,
        )
        ds.attrs.update(
            {
                "sweep_param": "beta = C2r / C1r",
                "C1r_fF": c1r,
                "capacitance_units": "fF",
            }
        )
        return ds

    return (run_res_coupling_sweep,)


@app.cell
def _(DATA_DIR, gen_dtc, load_or_compute_dataset, np, run_res_coupling_sweep):
    RECOMPUTE_RES_COUPLING_SWEEP = False
    beta_list = np.linspace(0.75, 1.25, 51)

    def _compute_res_coupling_sweep():
        return run_res_coupling_sweep(
            gen_dtc,
            np.linspace(0,0.5,201),
            beta_list,
            evals_count=12,
            get_eigenstates=True,
        )

    res_coupling_sweep_ds, res_coupling_sweep_path = load_or_compute_dataset(
        DATA_DIR / "res_coupling_sweep.nc",
        _compute_res_coupling_sweep,
        force=RECOMPUTE_RES_COUPLING_SWEEP,
    )
    return RECOMPUTE_RES_COUPLING_SWEEP, res_coupling_sweep_ds


@app.cell
def _(res_coupling_sweep_ds):
    res_coupling_sweep_ds
    return


@app.cell
def _(np, plt, res_coupling_sweep_ds):
    # for _i in range(6):
    plt.plot(np.abs(res_coupling_sweep_ds["tracked_energy"][30, :, 2] - res_coupling_sweep_ds["tracked_energy"][30, :, 1]))
    # plt.ylim(3.7, 4.3)
    plt.show()
    return


@app.cell
def _(plt, res_coupling_sweep_ds):
    # plt.plot(np.abs(res_coupling_sweep_ds["chi_p"][25, :] - res_coupling_sweep_ds["chi_m"][25, :]))
    plt.plot(res_coupling_sweep_ds["chi_m"][25, :])
    plt.plot(res_coupling_sweep_ds["chi_p"][25, :])
    return


@app.cell
def _(colors, np, plt, res_coupling_sweep_ds):
    delta_chi_mhz = 1e3 * res_coupling_sweep_ds["delta_chi"].values
    sensitivity = np.gradient(np.abs(res_coupling_sweep_ds["tracked_energy"][: , :, 2] - res_coupling_sweep_ds["tracked_energy"][: , :, 1]), axis=1)

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _pc = _ax.pcolormesh(
        res_coupling_sweep_ds["beta"].values,
        res_coupling_sweep_ds["flux"].values,
        np.abs(delta_chi_mhz).T,
        cmap="Reds",
        norm=colors.LogNorm(vmin=1e-4, vmax=1),
    )

    _ax.set_xlabel(r"$\beta = C_{2r}/C_{1r}$")
    _ax.set_ylabel(r"$\phi_\text{ext}$")

    _ax.contour(
        res_coupling_sweep_ds["beta"].values,
        res_coupling_sweep_ds["flux"].values,
        sensitivity.T ,
        levels=[0],
        # norm=colors.LogNorm(vmin=6e-4, vmax=2),
        # alpha=0.7,
        linewidths=1.0,
    )

    _cbar = plt.colorbar(_pc)
    _cbar.set_label(r"$|2\Delta\chi|$ (MHz)")
    # _fig.savefig(Path("figs/res_coupling_sweep.pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(plt, res_coupling_sweep_ds):
    _fig, _ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 4), sharex=True)
    _beta_idx = [0, 15, 20]
    plt.subplots_adjust(wspace=0.5, hspace=0.1)
    for _i, _beta in enumerate(_beta_idx):
        _ax[0, _i].set_title(rf"$\beta = {res_coupling_sweep_ds['beta'].values[_beta]:.2f}$")
        _ax[0, _i].plot(
            res_coupling_sweep_ds["flux"].values,
            res_coupling_sweep_ds["tracked_energy"][_beta, :, 1]-res_coupling_sweep_ds["tracked_energy"][_beta, :, 0],
        )

        _ax[0, _i].plot(
            res_coupling_sweep_ds["flux"].values,
            res_coupling_sweep_ds["tracked_energy"][_beta, :, 2]-res_coupling_sweep_ds["tracked_energy"][_beta, :, 0],
        )

        _ax[1, _i].plot(
            res_coupling_sweep_ds["flux"].values,
            res_coupling_sweep_ds["chi_m"][_beta, :]*1e3,
        )
        _ax[1, _i].plot(
            res_coupling_sweep_ds["flux"].values,
            res_coupling_sweep_ds["chi_p"][_beta, :]*1e3,
        )

        _ax[1, _i].set_xlabel(r"$\phi_\text{ext}$")

    _ax[0, 0].set_ylabel("f (GHz)")
    _ax[1, 0].set_ylabel(r"$2\chi$ (MHz)")
    _fig.align_labels()
    # _ax[0, 0].set_ylabel("Energy (GHz)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transmon Capacitance Sweep
    """)
    return


@app.cell
def _(FLUX_PARAM, np, spectrum_to_dataset, track_levels_by_overlap, xr):
    def run_transmon_c_sweep(
        gen_dtc,
        flux_values,
        beta_values,
        *,
        c1: float = 194,
        evals_count: int = 12,
        flux_param: str = FLUX_PARAM,
        get_eigenstates: bool = False,
    ) -> xr.Dataset:
        energy_tables = []
        tracked_level_tables = []
        partner_overlap_tables = []
        tracked_energy_tables = []

        for beta in beta_values:
            circuit = gen_dtc(C1=c1, C2=c1 * beta)
            sweep = circuit.get_spectrum_vs_paramvals(
                flux_param,
                flux_values,
                evals_count=evals_count,
                get_eigenstates=get_eigenstates,
                num_cpus=8,
            )
            energy_table = np.asarray(sweep.energy_table)
            energy_tables.append(energy_table)

            if get_eigenstates:
                tracked_level, partner_overlap, tracked_energy = track_levels_by_overlap(
                    sweep.state_table,
                    energy_table,
                    q3_operator=circuit.Q3_operator(),
                )
                tracked_level_tables.append(tracked_level)
                partner_overlap_tables.append(partner_overlap)
                tracked_energy_tables.append(tracked_energy)

        tracked_kwargs = {}
        if get_eigenstates:
            tracked_kwargs = {
                "tracked_energy": np.asarray(tracked_energy_tables),
                "tracked_level": np.asarray(tracked_level_tables),
                "partner_overlap": np.asarray(partner_overlap_tables),
            }

        ds = spectrum_to_dataset(
            np.asarray(energy_tables),
            flux_values,
            sweep_dim="beta",
            sweep_values=beta_values,
            extra_coords={"C2": ("beta", c1 * np.asarray(beta_values))},
            flux_param=flux_param,
            description="DTC transmon-capacitance and flux spectrum sweep.",
            **tracked_kwargs,
        )
        ds.attrs.update(
            {
                "sweep_param": "beta = C2 / C1",
                "C1_MHz": c1,
                "capacitance_units": "fF",
            }
        )
        return ds

    return (run_transmon_c_sweep,)


@app.cell
def _(DATA_DIR, gen_dtc, load_or_compute_dataset, np, run_transmon_c_sweep):
    RECOMPUTE_TRANSMON_C_SWEEP = False
    _beta_list = np.linspace(0.9, 1.1, 101)

    def _compute_transmon_c_sweep():
        return run_transmon_c_sweep(
            gen_dtc,
            np.linspace(0,0.5,201),
            _beta_list,
            evals_count=12,
            get_eigenstates=True,
        )

    transmon_c_sweep_ds, transmon_c_sweep_path = load_or_compute_dataset(
        DATA_DIR / "transmon_c_sweep.nc",
        _compute_transmon_c_sweep,
        force=RECOMPUTE_TRANSMON_C_SWEEP,
    )
    return (transmon_c_sweep_ds,)


@app.cell
def _(Path, colors, np, plt, transmon_c_sweep_ds):
    _delta_chi_mhz = 1e3 * transmon_c_sweep_ds["delta_chi"].values
    _sensitivity = np.gradient(np.abs(transmon_c_sweep_ds["tracked_energy"][: , :, 2] - transmon_c_sweep_ds["tracked_energy"][: , :, 1]), axis=1)

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _pc = _ax.pcolormesh(
        transmon_c_sweep_ds["beta"].values,
        transmon_c_sweep_ds["flux"].values,
        np.abs(_delta_chi_mhz).T,
        cmap="Reds",
        norm=colors.LogNorm(vmin=np.abs(_delta_chi_mhz).min(), vmax=np.abs(_delta_chi_mhz).max()),
    )

    _ax.set_xlabel(r"$\beta = C_{2}/C_{1}$")
    _ax.set_ylabel(r"$\phi_\text{ext}$")

    _ax.contour(
        transmon_c_sweep_ds["beta"].values,
        transmon_c_sweep_ds["flux"].values,
        _sensitivity.T ,
        levels=[0],
        # norm=colors.LogNorm(vmin=np.abs(_sensitivity).min(), vmax=np.abs(_sensitivity).max()),
        # alpha=0.7,
        linewidths=1.0,
    )

    _cbar = plt.colorbar(_pc)


    _cbar.set_label(r"$|2\Delta\chi|$ (MHz)")
    _fig.savefig(Path("figs/transmon_c_sweep.pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(plt, transmon_c_sweep_ds):
    _fig, _ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 4), sharex=True)
    _beta_idx = [22, 28, 53]
    plt.subplots_adjust(wspace=0.5, hspace=0.1)
    for _i, _beta in enumerate(_beta_idx):
        _ax[0, _i].set_title(rf"$\beta = {transmon_c_sweep_ds['beta'].values[_beta]:.3f}$")
        _ax[0, _i].plot(
            transmon_c_sweep_ds["flux"].values,
            transmon_c_sweep_ds["tracked_energy"][_beta, :, 1]-transmon_c_sweep_ds["tracked_energy"][_beta, :, 0],
        )

        _ax[0, _i].plot(
            transmon_c_sweep_ds["flux"].values,
            transmon_c_sweep_ds["tracked_energy"][_beta, :, 2]-transmon_c_sweep_ds["tracked_energy"][_beta, :, 0],
        )

        _ax[1, _i].plot(
            transmon_c_sweep_ds["flux"].values,
            transmon_c_sweep_ds["chi_m"][_beta, :]*1e3,
        )
        _ax[1, _i].plot(
            transmon_c_sweep_ds["flux"].values,
            transmon_c_sweep_ds["chi_p"][_beta, :]*1e3,
        )

        _ax[1, _i].set_xlabel(r"$\phi_\text{ext}$")

    _ax[0, 0].set_ylabel("f (GHz)")
    _ax[1, 0].set_ylabel(r"$2\chi$ (MHz)")
    _fig.align_labels()
    # _ax[0, 0].set_ylabel("Energy (GHz)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coupling Ej sweep
    """)
    return


@app.cell
def _(FLUX_PARAM, np, spectrum_to_dataset, track_levels_by_overlap, xr):
    def run_coupling_ej_sweep(
        gen_dtc,
        flux_values,
        alpha_values,
        *,
        ej1: float = 14,
        evals_count: int = 12,
        flux_param: str = FLUX_PARAM,
        get_eigenstates: bool = False,
    ) -> xr.Dataset:
        energy_tables = []
        tracked_level_tables = []
        partner_overlap_tables = []
        tracked_energy_tables = []

        for alpha in alpha_values:
            circuit = gen_dtc(Ej1=ej1, Ej3=ej1 * alpha, C2r=0.90*15)
            sweep = circuit.get_spectrum_vs_paramvals(
                flux_param,
                flux_values,
                evals_count=evals_count,
                get_eigenstates=get_eigenstates,
                num_cpus=8,
            )
            energy_table = np.asarray(sweep.energy_table)
            energy_tables.append(energy_table)

            if get_eigenstates:
                tracked_level, partner_overlap, tracked_energy = track_levels_by_overlap(
                    sweep.state_table,
                    energy_table,
                    q3_operator=circuit.Q3_operator(),
                )
                tracked_level_tables.append(tracked_level)
                partner_overlap_tables.append(partner_overlap)
                tracked_energy_tables.append(tracked_energy)

        tracked_kwargs = {}
        if get_eigenstates:
            tracked_kwargs = {
                "tracked_energy": np.asarray(tracked_energy_tables),
                "tracked_level": np.asarray(tracked_level_tables),
                "partner_overlap": np.asarray(partner_overlap_tables),
            }

        ds = spectrum_to_dataset(
            np.asarray(energy_tables),
            flux_values,
            sweep_dim="alpha",
            sweep_values=alpha_values,
            extra_coords={"Ej3": ("alpha", ej1 * np.asarray(alpha_values))},
            flux_param=flux_param,
            description="DTC coupling-Ej and flux spectrum sweep.",
            **tracked_kwargs,
        )
        ds.attrs.update(
            {
                "sweep_param": "alpha = Ej3 / Ej1",
                "Ej1_GHz": ej1,
                "capacitance_units": "fF",
            }
        )
        return ds

    return (run_coupling_ej_sweep,)


@app.cell
def _(
    DATA_DIR,
    flux_list,
    gen_dtc,
    load_or_compute_dataset,
    np,
    run_coupling_ej_sweep,
):
    RECOMPUTE_COUPLING_EJ_SWEEP = True
    _alpha_list = np.linspace(0.01, 0.51, 51)

    def _compute_coupling_ej_sweep():
        return run_coupling_ej_sweep(
            gen_dtc,
            flux_list,
            _alpha_list,
            evals_count=12,
            get_eigenstates=True,
        )

    coupling_ej_sweep_ds, coupling_ej_sweep_path = load_or_compute_dataset(
        DATA_DIR / "coupling_ej_sweep_c2r_over_c1r_0.90.nc",
        _compute_coupling_ej_sweep,
        force=RECOMPUTE_COUPLING_EJ_SWEEP,
    )
    return (coupling_ej_sweep_ds,)


@app.cell
def _(coupling_ej_sweep_ds, plt):
    _i = 20
    plt.plot(coupling_ej_sweep_ds["tracked_energy"][_i, :, 1] - coupling_ej_sweep_ds["tracked_energy"][_i, :, 2])
    # plt.plot(coupling_ej_sweep_ds["tracked_energy"][_i, :, 2] - coupling_ej_sweep_ds["tracked_energy"][_i, :, 0])
    return


@app.cell
def _(coupling_ej_sweep_ds, plt):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _beta_idx = 3

    _ax.plot(
        coupling_ej_sweep_ds["flux"].values,
        coupling_ej_sweep_ds["tracked_energy"][_beta_idx, :, 1] - coupling_ej_sweep_ds["tracked_energy"][_beta_idx, :, 0],
        label="minus branch",
    )
    _ax.plot(
        coupling_ej_sweep_ds["flux"].values,
        coupling_ej_sweep_ds["tracked_energy"][_beta_idx, :, 2] - coupling_ej_sweep_ds["tracked_energy"][_beta_idx, :, 0],
        label="plus branch",
    )
    return


@app.cell
def _(coupling_ej_sweep_ds, plt):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _beta_idx = 3
    _ax.plot(
        coupling_ej_sweep_ds["flux"].values,
        coupling_ej_sweep_ds["chi_m"][_beta_idx, :],
        label="minus branch",
    )
    _ax.plot(
        coupling_ej_sweep_ds["flux"].values,
        coupling_ej_sweep_ds["chi_p"][_beta_idx, :],
        label="plus branch",
    )
    plt.show()
    return


@app.cell
def _(Path, colors, coupling_ej_sweep_ds, np, plt):
    _delta_chi_mhz = 1e3 * coupling_ej_sweep_ds["delta_chi"].values
    _sensitivity = np.gradient(np.abs(coupling_ej_sweep_ds["tracked_energy"][: , :, 2] - coupling_ej_sweep_ds["tracked_energy"][: , :, 1]), axis=1)

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _pc = _ax.pcolormesh(
        coupling_ej_sweep_ds["alpha"].values,
        coupling_ej_sweep_ds["flux"].values,
        np.abs(_delta_chi_mhz).T,
        cmap="Reds",
        norm=colors.LogNorm(vmin=np.abs(_delta_chi_mhz).min(), vmax=np.abs(_delta_chi_mhz).max()),
    )

    _ax.set_xlabel(r"$\alpha = E_{j3}/E_{j1}$")
    _ax.set_ylabel(r"$\phi_\text{ext}$")

    _ax.contour(
        coupling_ej_sweep_ds["alpha"].values,
        coupling_ej_sweep_ds["flux"].values,
        _sensitivity.T ,
        levels=[0],
        # norm=colors.LogNorm(vmin=np.abs(_sensitivity).min(), vmax=np.abs(_sensitivity).max()),
        # alpha=0.7,
        linewidths=1.0,
    )

    _cbar = plt.colorbar(_pc)


    _cbar.set_label(r"$|2\Delta\chi|$ (MHz)")
    _fig.savefig(Path("figs/coupling_ej_sweep_c2r_over_c1r_0.90.pdf"), dpi=300, bbox_inches="tight")
    # _fig.savefig(Path("figs/coupling_ej_sweep.pdf"), dpi=300, bbox_inches="tight")

    plt.show()
    return


@app.cell
def _(coupling_ej_sweep_ds, plt):
    _fig, _ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 4), sharex=True)
    _alpha_idx = [13, 20, 40]
    plt.subplots_adjust(wspace=0.5, hspace=0.1)
    for _i, _alpha in enumerate(_alpha_idx):
        _ax[0, _i].set_title(rf"$\alpha = {coupling_ej_sweep_ds['alpha'].values[_alpha]:.3f}$")
        _ax[0, _i].plot(
            coupling_ej_sweep_ds["flux"].values,
            coupling_ej_sweep_ds["tracked_energy"][_alpha, :, 1]-coupling_ej_sweep_ds["tracked_energy"][_alpha, :, 0],
        )

        _ax[0, _i].plot(
            coupling_ej_sweep_ds["flux"].values,
            coupling_ej_sweep_ds["tracked_energy"][_alpha, :, 2]-coupling_ej_sweep_ds["tracked_energy"][_alpha, :, 0],
        )

        _ax[1, _i].plot(
            coupling_ej_sweep_ds["flux"].values,
            coupling_ej_sweep_ds["chi_m"][_alpha, :]*1e3,
        )
        _ax[1, _i].plot(
            coupling_ej_sweep_ds["flux"].values,
            coupling_ej_sweep_ds["chi_p"][_alpha, :]*1e3,
        )

        _ax[1, _i].set_xlabel(r"$\phi_\text{ext}$")

    _ax[0, 0].set_ylabel("f (GHz)")
    _ax[1, 0].set_ylabel(r"$2\chi$ (MHz)")
    _fig.align_labels()
    # _ax[0, 0].set_ylabel("Energy (GHz)")
    plt.show()
    return


@app.cell
def _(coupling_ej_sweep_ds):
    coupling_ej_sweep_ds
    return


@app.cell
def _(coupling_ej_sweep_ds, plt):
    plt.plot(coupling_ej_sweep_ds["tracked_energy"][10 , :, 2] - coupling_ej_sweep_ds["tracked_energy"][10 , :, 1])
    return


@app.cell
def _(coupling_ej_sweep_ds, plt):
    plt.plot(coupling_ej_sweep_ds["tracked_energy"][10, :, 1] -coupling_ej_sweep_ds["tracked_energy"][10, :, 2])
    # plt.plot(coupling_ej_sweep_ds["tracked_energy"][10, :, 2])
    return


@app.cell
def _(coupling_ej_sweep_ds, np, plt):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _sensitivity = np.gradient(np.abs(coupling_ej_sweep_ds["tracked_energy"][: , :, 2] - coupling_ej_sweep_ds["tracked_energy"][: , :, 1]), axis=1)

    # _ax.plot(coupling_ej_sweep_ds["tracked_energy"][40, :, 1] - coupling_ej_sweep_ds["tracked_energy"][40, :, 2])
    _ax.pcolormesh(
        coupling_ej_sweep_ds["alpha"].values,
        coupling_ej_sweep_ds["flux"].values,
        np.abs(coupling_ej_sweep_ds["tracked_energy"][: , :, 2] - coupling_ej_sweep_ds["tracked_energy"][: , :, 1]).T ,
        # levels=[0],
        # norm=colors.LogNorm(vmin=np.abs(_sensitivity).min(), vmax=np.abs(_sensitivity).max()),
        # alpha=0.7,
        # linewidths=1.0,
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coupling C ratio sweep
    """)
    return


@app.cell
def _(FLUX_PARAM, np, spectrum_to_dataset, track_levels_by_overlap, xr):
    def run_res_coupling_ratio_sweep(
        gen_dtc,
        flux_values,
        beta_values,
        *,
        c1r: float = 15,
        c2r: float = 15,
        c1: float = 194,
        c2: float = 194,
        evals_count: int = 12,
        flux_param: str = FLUX_PARAM,
        get_eigenstates: bool = False,
    ) -> xr.Dataset:
        energy_tables = []
        tracked_level_tables = []
        partner_overlap_tables = []
        tracked_energy_tables = []

        for beta in beta_values:
            circuit = gen_dtc(C1r = c1 * beta, C2r = c2 * beta, C1 = c1, C2 = c2)
            sweep = circuit.get_spectrum_vs_paramvals(
                flux_param,
                flux_values,
                evals_count=evals_count,
                get_eigenstates=get_eigenstates,
                num_cpus=8,
            )
            energy_table = np.asarray(sweep.energy_table)
            energy_tables.append(energy_table)

            if get_eigenstates:
                tracked_level, partner_overlap, tracked_energy = track_levels_by_overlap(
                    sweep.state_table,
                    energy_table,
                    q3_operator=circuit.Q3_operator(),
                )
                tracked_level_tables.append(tracked_level)
                partner_overlap_tables.append(partner_overlap)
                tracked_energy_tables.append(tracked_energy)

        tracked_kwargs = {}
        if get_eigenstates:
            tracked_kwargs = {
                "tracked_energy": np.asarray(tracked_energy_tables),
                "tracked_level": np.asarray(tracked_level_tables),
                "partner_overlap": np.asarray(partner_overlap_tables),
            }

        ds = spectrum_to_dataset(
            np.asarray(energy_tables),
            flux_values,
            sweep_dim="beta",
            sweep_values=beta_values,
            extra_coords={"C2r": ("beta", c1r * np.asarray(beta_values))},
            flux_param=flux_param,
            description="DTC resonator-coupling and flux spectrum sweep.",
            **tracked_kwargs,
        )
        ds.attrs.update(
            {
                "sweep_param": "beta = C2r / C1r",
                "C1r_fF": c1r,
                "capacitance_units": "fF",
            }
        )
        return ds

    return (run_res_coupling_ratio_sweep,)


@app.cell
def _(
    DATA_DIR,
    RECOMPUTE_RES_COUPLING_SWEEP,
    gen_dtc,
    load_or_compute_dataset,
    np,
    run_res_coupling_ratio_sweep,
):
    RECOMPUTE_RES_COUPLING_RATIO_SWEEP = False
    _beta_list = np.linspace(0.01, 0.21, 41)

    def _compute_res_coupling_ratio_sweep():
        return run_res_coupling_ratio_sweep(
            gen_dtc,
            np.linspace(0,0.5,201),
            _beta_list,
            evals_count=12,
            get_eigenstates=True,
        )

    res_coupling_ratio_sweep_ds, res_coupling_ratio_sweep_path = load_or_compute_dataset(
        DATA_DIR / "res_coupling_ratio_sweep.nc",
        _compute_res_coupling_ratio_sweep,
        force=RECOMPUTE_RES_COUPLING_SWEEP,
    )
    return (res_coupling_ratio_sweep_ds,)


@app.cell
def _(res_coupling_ratio_sweep_ds):
    res_coupling_ratio_sweep_ds
    return


@app.cell
def _(plt, res_coupling_ratio_sweep_ds):
    _fig, _ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 4), sharex=True)
    _beta_idx = [0, 20, 40]
    plt.subplots_adjust(wspace=0.5, hspace=0.1)
    for _i, _beta in enumerate(_beta_idx):
        _ax[0, _i].set_title(rf"$\beta = {res_coupling_ratio_sweep_ds['beta'].values[_beta]:.2f}$")
        _ax[0, _i].plot(
            res_coupling_ratio_sweep_ds["flux"].values,
            res_coupling_ratio_sweep_ds["tracked_energy"][_beta, :, 1]-res_coupling_ratio_sweep_ds["tracked_energy"][_beta, :, 0],
        )

        _ax[0, _i].plot(
            res_coupling_ratio_sweep_ds["flux"].values,
            res_coupling_ratio_sweep_ds["tracked_energy"][_beta, :, 2]-res_coupling_ratio_sweep_ds["tracked_energy"][_beta, :, 0],
        )

        _ax[1, _i].plot(
            res_coupling_ratio_sweep_ds["flux"].values,
            res_coupling_ratio_sweep_ds["chi_m"][_beta, :]*1e3,
        )
        _ax[1, _i].plot(
            res_coupling_ratio_sweep_ds["flux"].values,
            res_coupling_ratio_sweep_ds["chi_p"][_beta, :]*1e3,
        )

        _ax[1, _i].set_xlabel(r"$\phi_\text{ext}$")

    _ax[0, 0].set_ylabel("f (GHz)")
    _ax[1, 0].set_ylabel(r"$2\chi$ (MHz)")
    _fig.align_labels()
    # _ax[0, 0].set_ylabel("Energy (GHz)")
    plt.show()
    return


@app.cell
def _(plt, res_coupling_ratio_sweep_ds):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _beta_idx = 10

    _ax.plot(res_coupling_ratio_sweep_ds["flux"].values, res_coupling_ratio_sweep_ds["chi_p"][_beta_idx, :], label=r"$\chi_+$")
    _ax.plot(res_coupling_ratio_sweep_ds["flux"].values, res_coupling_ratio_sweep_ds["chi_m"][_beta_idx, :], label=r"$\chi_-$")
    _ax.set_xlabel(r"$\phi_\text{ext}$")
    _ax.set_ylabel(r"$\chi$ (GHz)")
    _ax.legend()
    plt.show()
    return


@app.cell
def _(Path, colors, np, plt, res_coupling_ratio_sweep_ds):
    _delta_chi_mhz = 1e3 * res_coupling_ratio_sweep_ds["delta_chi"].values
    _sensitivity = np.gradient(np.abs(res_coupling_ratio_sweep_ds["tracked_energy"][: , :, 2] - res_coupling_ratio_sweep_ds["tracked_energy"][: , :, 1]), axis=1)

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _pc = _ax.pcolormesh(
        res_coupling_ratio_sweep_ds["beta"].values,
        res_coupling_ratio_sweep_ds["flux"].values,
        np.abs(_delta_chi_mhz).T,
        cmap="Reds",
        norm=colors.LogNorm(vmin=1e-4, vmax=1),
    )

    _ax.set_xlabel(r"$\beta = C_{ir}/C_{i}$")
    _ax.set_ylabel(r"$\phi_\text{ext}$")

    # _ax.contour(
    #     res_coupling_ratio_sweep_ds["beta"].values,
    #     res_coupling_ratio_sweep_ds["flux"].values,
    #     _sensitivity.T ,
    #     levels=[0],
    #     # norm=colors.LogNorm(vmin=6e-4, vmax=2),
    #     # alpha=0.7,
    #     linewidths=1.0,
    # )

    _cbar = plt.colorbar(_pc)
    _cbar.set_label(r"$|2\Delta\chi|$ (MHz)")
    _fig.savefig(Path("figs/res_coupling_ratio_sweep.pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transmon C sweep with C/Cr ratio fixed
    """)
    return


@app.cell
def _(FLUX_PARAM, np, spectrum_to_dataset, track_levels_by_overlap, xr):
    def run_transmon_c_sweep_ratio_fixed(
        gen_dtc,
        flux_values,
        beta_values,
        *,
        c1: float = 194,
        c1r: float = 15,
        evals_count: int = 12,
        flux_param: str = FLUX_PARAM,
        get_eigenstates: bool = False,
    ) -> xr.Dataset:
        energy_tables = []
        tracked_level_tables = []
        partner_overlap_tables = []
        tracked_energy_tables = []

        for beta in beta_values:
            circuit = gen_dtc(C1=c1, C2=c1 * beta, C1r=c1r, C2r=c1r * beta)
            sweep = circuit.get_spectrum_vs_paramvals(
                flux_param,
                flux_values,
                evals_count=evals_count,
                get_eigenstates=get_eigenstates,
                num_cpus=8,
            )
            energy_table = np.asarray(sweep.energy_table)
            energy_tables.append(energy_table)

            if get_eigenstates:
                tracked_level, partner_overlap, tracked_energy = track_levels_by_overlap(
                    sweep.state_table,
                    energy_table,
                    q3_operator=circuit.Q3_operator(),
                )
                tracked_level_tables.append(tracked_level)
                partner_overlap_tables.append(partner_overlap)
                tracked_energy_tables.append(tracked_energy)

        tracked_kwargs = {}
        if get_eigenstates:
            tracked_kwargs = {
                "tracked_energy": np.asarray(tracked_energy_tables),
                "tracked_level": np.asarray(tracked_level_tables),
                "partner_overlap": np.asarray(partner_overlap_tables),
            }

        ds = spectrum_to_dataset(
            np.asarray(energy_tables),
            flux_values,
            sweep_dim="beta",
            sweep_values=beta_values,
            extra_coords={"C2": ("beta", c1 * np.asarray(beta_values))},
            flux_param=flux_param,
            description="DTC transmon-capacitance and flux spectrum sweep.",
            **tracked_kwargs,
        )
        ds.attrs.update(
            {
                "sweep_param": "beta = C2 / C1",
                "C1_MHz": c1,
                "capacitance_units": "fF",
            }
        )
        return ds

    return


@app.cell
def _(DATA_DIR, gen_dtc, load_or_compute_dataset, np, run_transmon_c_sweep):
    RECOMPUTE_TRANSMON_C_FIXED_RATIO_SWEEP = True
    _beta_list = np.linspace(0.9, 1.1, 101)

    def _compute_transmon_c_sweep():
        return run_transmon_c_sweep(
            gen_dtc,
            np.linspace(0,0.5,201),
            _beta_list,
            evals_count=12,
            get_eigenstates=True,
        )

    transmon_c_fixed_ratio_sweep_ds, transmon_c_fixed_ratio_sweep_path = load_or_compute_dataset(
        DATA_DIR / "transmon_c_fixed_ratio_sweep.nc",
        _compute_transmon_c_sweep,
        force=RECOMPUTE_TRANSMON_C_FIXED_RATIO_SWEEP,
    )
    return (transmon_c_fixed_ratio_sweep_ds,)


@app.cell
def _(plt, transmon_c_fixed_ratio_sweep_ds):
    _fig, _ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 4), sharex=True)
    _beta_idx = [22, 28, 53]
    plt.subplots_adjust(wspace=0.5, hspace=0.1)
    for _i, _beta in enumerate(_beta_idx):
        _ax[0, _i].set_title(rf"$\beta = {transmon_c_fixed_ratio_sweep_ds['beta'].values[_beta]:.3f}$")
        _ax[0, _i].plot(
            transmon_c_fixed_ratio_sweep_ds["flux"].values,
            transmon_c_fixed_ratio_sweep_ds["tracked_energy"][_beta, :, 1]-transmon_c_fixed_ratio_sweep_ds["tracked_energy"][_beta, :, 0],
        )

        _ax[0, _i].plot(
            transmon_c_fixed_ratio_sweep_ds["flux"].values,
            transmon_c_fixed_ratio_sweep_ds["tracked_energy"][_beta, :, 2]-transmon_c_fixed_ratio_sweep_ds["tracked_energy"][_beta, :, 0],
        )

        _ax[1, _i].plot(
            transmon_c_fixed_ratio_sweep_ds["flux"].values,
            transmon_c_fixed_ratio_sweep_ds["chi_m"][_beta, :]*1e3,
        )
        _ax[1, _i].plot(
            transmon_c_fixed_ratio_sweep_ds["flux"].values,
            transmon_c_fixed_ratio_sweep_ds["chi_p"][_beta, :]*1e3,
        )

        _ax[1, _i].set_xlabel(r"$\phi_\text{ext}$")

    _ax[0, 0].set_ylabel("f (GHz)")
    _ax[1, 0].set_ylabel(r"$2\chi$ (MHz)")
    _fig.align_labels()
    # _ax[0, 0].set_ylabel("Energy (GHz)")
    plt.show()
    return


@app.cell
def _(plt, res_coupling_ratio_sweep_ds):
    _fig, _ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 4), sharex=True)
    _beta_idx = [0, 20, 40]
    plt.subplots_adjust(wspace=0.5, hspace=0.1)
    for _i, _beta in enumerate(_beta_idx):
        _ax[0, _i].set_title(rf"$\beta = {res_coupling_ratio_sweep_ds['beta'].values[_beta]:.2f}$")
        _ax[0, _i].plot(
            res_coupling_ratio_sweep_ds["flux"].values,
            res_coupling_ratio_sweep_ds["tracked_energy"][_beta, :, 1]-res_coupling_ratio_sweep_ds["tracked_energy"][_beta, :, 0],
        )

        _ax[0, _i].plot(
            res_coupling_ratio_sweep_ds["flux"].values,
            res_coupling_ratio_sweep_ds["tracked_energy"][_beta, :, 2]-res_coupling_ratio_sweep_ds["tracked_energy"][_beta, :, 0],
        )

        _ax[1, _i].plot(
            res_coupling_ratio_sweep_ds["flux"].values,
            res_coupling_ratio_sweep_ds["chi_m"][_beta, :]*1e3,
        )
        _ax[1, _i].plot(
            res_coupling_ratio_sweep_ds["flux"].values,
            res_coupling_ratio_sweep_ds["chi_p"][_beta, :]*1e3,
        )

        _ax[1, _i].set_xlabel(r"$\phi_\text{ext}$")

    _ax[0, 0].set_ylabel("f (GHz)")
    _ax[1, 0].set_ylabel(r"$2\chi$ (MHz)")
    _fig.align_labels()
    # _ax[0, 0].set_ylabel("Energy (GHz)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coupling C & Coupling Ej Sweep
    """)
    return


@app.cell
def _(np, run_two_param_sweep):
    ds = run_two_param_sweep(
        param1_name="C1r",
        param1_targets={"C1r": 1.0, "C1": 12.93},
        param1_values=np.linspace(15*0.9, 15*1.1, 41),
        param2_name="Ej3",
        param2_values=np.linspace(14*0.01, 14*0.51, 51),
        fixed_flux=0.5,
        evals_count=12,
        get_eigenstates=True,
        base_params={"Ej1": 14*1.2},
        cpus=8,
    )
    return (ds,)


@app.cell
def _(colors, ds, np, plt):
    _fig, _ax = plt.subplots(figsize=(4, 3))
    _pc = _ax.pcolormesh(
        ds["C1r"].values,
        ds["Ej3"].values,
        np.abs(ds["delta_chi"].values/ds["chi_p"].values).T,
        cmap="Reds",
        norm=colors.LogNorm(vmin=1e-4, vmax=1),
    )
    _ax.set_xlabel(r"$C_{1r}$ (fF)")
    _ax.set_ylabel(r"$E_{j3}$ (GHz)")
    _cb = plt.colorbar(_pc)
    _cb.set_label(r"$|2\Delta\chi/2\chi_p|$")
    plt.show()
    return


@app.cell
def _(DATA_DIR, load_or_compute_two_param_sweep):
    ds0p8, _ = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_0p8ej2.nc",
        force=False,
    )

    ds0p9, _ = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_0p9ej2.nc",
        force=False,
    )

    ds0p95, _ = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_0p95ej2.nc",
        force=False,
    )

    ds1p0, _ = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio.nc",
        force=False,
    )

    ds1p05, _ = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_1p05ej2.nc",
        force=False,
    )

    ds1p1, _ = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_1p1ej2.nc",
        force=False,
    )

    ds1p2, _ = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_1p2ej2.nc",
        force=False,
    )
    return ds0p8, ds0p9, ds0p95, ds1p0, ds1p05, ds1p1, ds1p2


@app.cell
def _(
    Path,
    colors,
    ds0p8,
    ds0p9,
    ds0p95,
    ds1p0,
    ds1p05,
    ds1p1,
    ds1p2,
    np,
    plt,
):
    _fig, _ax = plt.subplots(ncols=7, figsize=(19, 2), sharey=True)

    for _i, (_ds, _r) in enumerate(zip([ds0p8, ds0p9, ds0p95, ds1p0, ds1p05, ds1p1, ds1p2], [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])):
        _pc = _ax[_i].pcolormesh(
            _ds["C1r"].values/15,
            _ds["Ej3"].values/14,
            np.abs(_ds["delta_chi"].values/_ds["chi_p"].values).T,
            cmap="Reds",
            norm=colors.LogNorm(vmin=1e-4, vmax=1),
        )
        _ax[_i].set_xlabel(r"$C_{1r}/C_{2r} = C_1/C_2$", fontsize=12)
        _ax[_i].set_title(rf"$E_{{j1}}/E_{{j2}} = {_r:.2f}$", fontsize=14)
        # _cb = plt.colorbar(_pc)
        # _cb.set_label(r"$|2\Delta\chi/2\chi_p|$")

    _ax[0].set_ylabel(r"$E_{j3}/E_{j2}$", fontsize=12)
    _cbar_ax = _fig.add_axes(rect=(0.92, 0.15, 0.01, 0.7))
    _cb = _fig.colorbar(_pc, cax=_cbar_ax)
    _cb.set_label(r"$|2\Delta\chi/2\chi_p|$", fontsize=14)
    _fig.savefig(Path(f"figs/ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_sweep.pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(DATA_DIR, load_or_compute_two_param_sweep, np):
    ds_1p0_phi_0 = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_1p0ej2_phi_0p0.nc",
        force=False,
        param1_name="C1r",
        param1_targets={"C1r": 1.0, "C1": 12.93},
        param1_values=np.linspace(15*0.9, 15*1.1, 41),
        param2_name="Ej3",
        param2_values=np.linspace(14*0.01, 14*0.51, 51),
        fixed_flux=0.0,
        evals_count=12,
        get_eigenstates=True,
        base_params={"Ej1": 14*1.0},
        cpus=8,
    )

    ds_0p95_phi_0 = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_0p95ej2_phi_0p0.nc",
        force=False,
        param1_name="C1r",
        param1_targets={"C1r": 1.0, "C1": 12.93},
        param1_values=np.linspace(15*0.9, 15*1.1, 41),
        param2_name="Ej3",
        param2_values=np.linspace(14*0.01, 14*0.51, 51),
        fixed_flux=0.0,
        evals_count=12,
        get_eigenstates=True,
        base_params={"Ej1": 14*0.95},
        cpus=8,
    )

    ds_1p05_phi_0 = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_1p05ej2_phi_0p0.nc",
        force=False,
        param1_name="C1r",
        param1_targets={"C1r": 1.0, "C1": 12.93},
        param1_values=np.linspace(15*0.9, 15*1.1, 41),
        param2_name="Ej3",
        param2_values=np.linspace(14*0.01, 14*0.51, 51),
        fixed_flux=0.0,
        evals_count=12,
        get_eigenstates=True,
        base_params={"Ej1": 14*1.05},
        cpus=8,
    )
    return ds_0p95_phi_0, ds_1p05_phi_0, ds_1p0_phi_0


@app.cell
def _(DATA_DIR, load_or_compute_two_param_sweep, np):
    ds_0p8_phi_0 = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_0p8ej2_phi_0p0.nc",
        force=False,
        param1_name="C1r",
        param1_targets={"C1r": 1.0, "C1": 12.93},
        param1_values=np.linspace(15*0.9, 15*1.1, 41),
        param2_name="Ej3",
        param2_values=np.linspace(14*0.01, 14*0.51, 51),
        fixed_flux=0.0,
        evals_count=12,
        get_eigenstates=True,
        base_params={"Ej1": 14*0.8},
        cpus=8,
    )

    ds_0p9_phi_0 = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_0p9ej2_phi_0p0.nc",
        force=False,
        param1_name="C1r",
        param1_targets={"C1r": 1.0, "C1": 12.93},
        param1_values=np.linspace(15*0.9, 15*1.1, 41),
        param2_name="Ej3",
        param2_values=np.linspace(14*0.01, 14*0.51, 51),
        fixed_flux=0.0,
        evals_count=12,
        get_eigenstates=True,
        base_params={"Ej1": 14*0.9},
        cpus=8,
    )

    ds_1p1_phi_0 = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_1p1ej2_phi_0p0.nc",
        force=False,
        param1_name="C1r",
        param1_targets={"C1r": 1.0, "C1": 12.93},
        param1_values=np.linspace(15*0.9, 15*1.1, 41),
        param2_name="Ej3",
        param2_values=np.linspace(14*0.01, 14*0.51, 51),
        fixed_flux=0.0,
        evals_count=12,
        get_eigenstates=True,
        base_params={"Ej1": 14*1.1},
        cpus=8,
    )

    ds_1p2_phi_0 = load_or_compute_two_param_sweep(
        path=DATA_DIR / "ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_1p2ej2_phi_0p0.nc",
        force=False,
        param1_name="C1r",
        param1_targets={"C1r": 1.0, "C1": 12.93},
        param1_values=np.linspace(15*0.9, 15*1.1, 41),
        param2_name="Ej3",
        param2_values=np.linspace(14*0.01, 14*0.51, 51),
        fixed_flux=0.0,
        evals_count=12,
        get_eigenstates=True,
        base_params={"Ej1": 14*1.2},
        cpus=8,
    )
    return ds_0p8_phi_0, ds_0p9_phi_0, ds_1p1_phi_0, ds_1p2_phi_0


@app.cell
def _(
    Path,
    colors,
    ds_0p8_phi_0,
    ds_0p95_phi_0,
    ds_0p9_phi_0,
    ds_1p05_phi_0,
    ds_1p0_phi_0,
    ds_1p1_phi_0,
    ds_1p2_phi_0,
    np,
    plt,
):
    _fig, _ax = plt.subplots(ncols=7, figsize=(19, 2), sharey=True)

    for _i, (_ds, _r) in enumerate(zip(
        [ds_0p8_phi_0[0], ds_0p9_phi_0[0], ds_0p95_phi_0[0], ds_1p0_phi_0[0], ds_1p05_phi_0[0], ds_1p1_phi_0[0], ds_1p2_phi_0[0]],
        [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2],
    )):
        _pc = _ax[_i].pcolormesh(
            _ds["C1r"].values/15,
            _ds["Ej3"].values/14,
            np.abs(_ds["delta_chi"].values/_ds["chi_p"].values).T,
            cmap="Reds",
            norm=colors.LogNorm(vmin=1e-4, vmax=1),
        )
        _ax[_i].set_xlabel(r"$C_{1r}/C_{2r} = C_1/C_2$", fontsize=12)
        _ax[_i].set_title(rf"$E_{{j1}}/E_{{j2}} = {_r:.2f}$", fontsize=14)
        # _cb = plt.colorbar(_pc)
        # _cb.set_label(r"$|2\Delta\chi/2\chi_p|$")

    _ax[0].set_ylabel(r"$E_{j3}/E_{j2}$", fontsize=12)
    _cbar_ax = _fig.add_axes(rect=(0.92, 0.15, 0.01, 0.7))
    _cb = _fig.colorbar(_pc, cax=_cbar_ax)
    _cb.set_label(r"$|2\Delta\chi/2\chi_p|$", fontsize=14)
    _fig.savefig(Path(f"figs/ej3_c1r_sweep_fixed_c1r_c1_ratio_ej1_sweep_phi_0p0.pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
