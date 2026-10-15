import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    from model import Basis, Device, ReadoutModel, find_readout_points, save_results

    return (
        Basis,
        Device,
        Path,
        ReadoutModel,
        mo,
        np,
        plt,
        save_results,
        threadpool_limits,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Single-junction dual-rail readout

    One pair: **transmon 1 + transmon 2 + a shared resonator**.
    Each transmon has one ideal Josephson junction. There are no flux parameters
    and no junction parasitic capacitances. Tune the two junction energies directly.

    **Workflow:** define components → inspect an EJ2 sweep → scan EJ2 and refine
    avoided crossings in EJ1 → compare the conditional resonator shifts.
    The search minimizes rail splitting, then filters for strong hybridization.
    It does not minimize the readout mismatch itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from reference_devices import original_preset

    _presets = {name: original_preset(name) for name in ("tdt1", "tdt2")}
    _fields = [
        ("C1_shunt_fF", mo.md(r"$C_1$"), "fF"),
        ("C2_shunt_fF", mo.md(r"$C_2$"), "fF"),
        ("C12_fF", mo.md(r"$C_{12}$"), "fF"),
        ("C1r_fF", mo.md(r"$C_{1r}$"), "fF"),
        ("C2r_fF", mo.md(r"$C_{2r}$"), "fF"),
        ("resonator_frequency_ghz", mo.md(r"$f_{r}$"), "GHz"),
        ("resonator_mode_impedance_ohm", mo.md(r"$Z_{12}$"), "ohm"),
        ("ej1_4p4_ghz", mo.md(r"$E_{J1}$"), "GHz"),
        ("ej2_4p4_ghz", mo.md(r"$E_{J2}$"), "GHz"),
    ]
    _rows = [{"Parameter": label, "Unit": unit,
              **{name: f"{float(preset[key]):.2f}" for name, preset in _presets.items()}}
             for key, label, unit in _fields]

    mo.vstack([
        mo.md("""
        ## Sample values from Tongyu
        The EJ values corresponds to a 4.4 GHz operation frequency for these devices.
        """),
        mo.ui.table(_rows, selection=None, page_size=20),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Device parameters

    Edit only the physical values in `device_parameters` below.
    Cr and Lr are calculated in a separate cell from frequency and impedance.
    EJ values initialize the inspection controls and search range; they are not
    automatically recalibrated when capacitances change. Numerical basis settings
    and the impedance convention have their own cells.
    """)
    return


@app.cell
def _():
    # All manual physical inputs. EJ means EJ/h in GHz; capacitances are in fF.
    device_parameters = {
        "EJ1_GHz": 11.5,
        "EJ2_GHz": 10.3,
        "C1_shunt_fF": 64,
        "C2_shunt_fF": 54,
        "C12_fF": 2.3,
        "C1r_fF": 6.0,
        "C2r_fF": 6.0,
        "resonator_frequency_ghz": 5.99,
        "resonator_impedance_ohm": 50,
    }
    return (device_parameters,)


@app.cell(hide_code=True)
def _(Basis, Device, ReadoutModel, device_parameters, mo, np):
    # Numerical settings, separate from the physical device parameters.
    # Increase these to check convergence before interpreting small mismatches.
    basis = Basis(ncut=15, transmon_levels=6, resonator_levels=5, eigenlevels=20)

    # Impedance convention and conversions belong here, not in the input list.
    # "mode": equivalent LC impedance; CPW options use characteristic line Z0.
    # CPW conversion assumes a uniform fundamental mode at a voltage antinode.
    resonator_impedance_kind = "mode"  # or "cpw_lambda4", "cpw_lambda2"
    _mode_factors = {"mode": 1.0, "cpw_lambda4": 4/np.pi, "cpw_lambda2": 2/np.pi}
    if resonator_impedance_kind not in _mode_factors:
        raise ValueError("Choose mode, cpw_lambda4, or cpw_lambda2")
    _frequency = device_parameters["resonator_frequency_ghz"]
    _impedance = device_parameters["resonator_impedance_ohm"]
    for _key in ("EJ1_GHz", "EJ2_GHz", "resonator_frequency_ghz", "resonator_impedance_ohm"):
        if not np.isfinite(device_parameters[_key]) or device_parameters[_key] <= 0:
            raise ValueError(f"{_key} must be positive and finite")
    _omega_r = 2*np.pi*_frequency*1e9
    _mode_impedance = _impedance*_mode_factors[resonator_impedance_kind]
    # omega = 1/sqrt(LC), Zmode = sqrt(L/C): C=1/(omega*Z), L=Z/omega.
    device = Device(
        **{_key: device_parameters[_key] for _key in (
            "C1_shunt_fF", "C2_shunt_fF", "C12_fF", "C1r_fF", "C2r_fF"
        )},
        Cr_fF=1/(_omega_r*_mode_impedance)*1e15,
        Lr_nH=_mode_impedance/_omega_r*1e9,
    )

    model = ReadoutModel(device, basis)
    _bare_fr_ghz = 1/(2*np.pi*np.sqrt(device.Lr_nH*1e-9*device.Cr_fF*1e-15))/1e9
    mo.md(f"""
    Local charging energies from C⁻¹: **{model.ec_ghz[0]*1e3:.3f} MHz** (T1),
    **{model.ec_ghz[1]*1e3:.3f} MHz** (T2).
    Local resonator frequency, with capacitive loading: **{model.fr_ghz:.4f} GHz**.
    Bare resonator frequency, using only Cr and Lr: **{_bare_fr_ghz:.4f} GHz**.
    Derived resonator components: **Cr = {device.Cr_fF:.3f} fF**,
    **Lr = {device.Lr_nH:.4f} nH**; equivalent-mode impedance:
    **{np.sqrt(device.Lr_nH*1e-9/(device.Cr_fF*1e-15)):.3f} Ω**.
    """)
    return device, model


@app.cell(disabled=True, hide_code=True)
def _(device, mo, model):
    # Convert F to fF and F^-1 to fF^-1. Both tables use node order T1,T2,r.
    _nodes = ("T1", "T2", "r")
    def _rows(_matrix):
        return [{"Node": _nodes[_i], **{
            _nodes[_j]: float(_matrix[_i, _j]) for _j in range(3)
        }} for _i in range(3)]
    mo.vstack([
        mo.md(r"**Capacitance matrix $C$ (fF)** — node order: T1, T2, resonator."),
        mo.ui.table(_rows(device.capacitance_matrix()*1e15), selection=None),
        mo.md(r"**Inverse capacitance matrix $C^{-1}$ (fF$^{-1}$)**"),
        mo.ui.table(_rows(model.cinv*1e-15), selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect hybridization: sweep EJ2 at fixed EJ1

    Fix EJ1 and give explicit **lower and upper EJ2 limits**.
    Both energies mean EJ/h, in GHz. The sweep contains 61 equally spaced points.
    Dashed curves are individual, capacitively loaded transmon frequencies with
    interactions omitted. Solid curves are dressed rail-like transitions of the
    full circuit, including the resonator. The dotted line is the local resonator
    frequency before interactions.

    The second panel shows the conditional resonator shifts chi_minus and
    chi_plus, in MHz, relative to the dressed ground-state resonator transition.
    Low assignment overlap indicates
    ambiguous rail-like labels, particularly near the resonator.
    """)
    return


@app.cell
def _(device_parameters, mo):
    hybridization_controls = mo.ui.dictionary({
        "EJ1_GHz": mo.ui.number(start=1, stop=50, step=0.001, value=device_parameters["EJ1_GHz"]),
        "EJ2_min_GHz": mo.ui.number(start=0.1, stop=100, step=0.001, value=0.7*device_parameters["EJ2_GHz"]),
        "EJ2_max_GHz": mo.ui.number(start=0.1, stop=100, step=0.001, value=1.3*device_parameters["EJ2_GHz"]),
    })
    hybridization_controls
    return (hybridization_controls,)


@app.cell(hide_code=True)
def _(hybridization_controls, mo, model, np, plt, threadpool_limits):
    _ej1 = hybridization_controls.value["EJ1_GHz"]
    _ej2_min = hybridization_controls.value["EJ2_min_GHz"]
    _ej2_max = hybridization_controls.value["EJ2_max_GHz"]
    mo.stop(_ej2_min >= _ej2_max, mo.md("**EJ2 lower limit must be smaller than its upper limit.**"))
    hybridization_ej2_ghz = np.linspace(_ej2_min, _ej2_max, 61)
    with threadpool_limits(limits=1):
        hybridization_scan = [model.evaluate(_ej1, _ej2) for _ej2 in hybridization_ej2_ghz]

    _get = lambda key: np.array([p[key] for p in hybridization_scan])
    _f1, _f2 = _get("bare_f1_ghz"), _get("bare_f2_ghz")
    _chi_minus = _get("chi_m_ghz")
    _chi_plus = _get("chi_p_ghz")
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _axes[0].plot(hybridization_ej2_ghz, _f1, "--", label="T1 loaded, uncoupled")
    _axes[0].plot(hybridization_ej2_ghz, _f2, "--", label="T2 loaded, uncoupled")
    _axes[0].plot(hybridization_ej2_ghz, _get("dressed_m_ghz"), label="Dressed m transition")
    _axes[0].plot(hybridization_ej2_ghz, _get("dressed_p_ghz"), label="Dressed p transition")
    _axes[0].axhline(model.fr_ghz, color="gray", linestyle=":", label="Local resonator")
    _axes[0].set(xlabel="EJ2/h (GHz)", ylabel="Transition frequency (GHz)", title="Loaded and hybridized transmons")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(hybridization_ej2_ghz, 1e3*_chi_minus, label=r"$\chi_-$")
    _axes[1].plot(hybridization_ej2_ghz, 1e3*_chi_plus, label=r"$\chi_+$")
    _axes[1].axhline(0, color="gray", linestyle="--")
    _axes[1].set(xlabel="EJ2/h (GHz)", ylabel=r"$\chi_-,\;\chi_+$ (MHz)", title="Conditional resonator shifts")
    _axes[1].legend()
    for _ax in _axes:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"Minimum six-branch assignment overlap across this sweep: **{_get('min_assignment_overlap').min():.3f}**. "
              "Branches are assigned separately at each point; label changes can cause jumps near strong mixing."),
    ])
    return (hybridization_scan,)


@app.cell(disabled=True, hide_code=True)
def _(hybridization_scan, mo):
    mo.vstack([
        mo.md(r"""
        **Local transmon–resonator coupling strengths**

        $$\frac{|g_{ir}|}{2\pi}=
        \frac{2e}{h}|(C^{-1})_{ir}|\,Q_{r,\mathrm{zpf}}
        |\langle0_i|\hat n_i|1_i\rangle|.$$

        Values below are in **MHz**, using numerical local transmon charge
        matrix elements and the loaded resonator's zero-point charge.
        They are local coupling magnitudes, not dressed frequency shifts.
        T1 coupling is fixed at the selected EJ1; T2 coupling varies with EJ2.
        """),
        mo.ui.table([{
            "EJ2/h (GHz)": _point["ej2_ghz"],
            "|g1r| / (2pi) (MHz)": _point["g1r_over_2pi_mhz"],
            "|g2r| / (2pi) (MHz)": _point["g2r_over_2pi_mhz"],
        } for _point in hybridization_scan], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Find readout candidates: resonance-guided search

    Edit `sweep_parameter` in the next cell. EJ1 is always the main sweep.
    Choose EJ2, either shunt capacitance, or either transmon-resonator capacitance.
    C12 stays fixed. When sweeping a capacitance, `fixed_ej2_ghz` sets EJ2.

    For each EJ1, predict equality of the two **loaded local** transmon frequencies,
    then search a neighborhood of that prediction for full-circuit splitting minima.
    Rebuild the inverse capacitance matrix at every capacitance trial.
    Predictions are not assumed to be the true dressed avoided crossings.

    `fractional`: prediction +/- fraction ? prediction, with a sample count.
    `absolute`: prediction +/- half-width, with a step in GHz or fF.
    Physical bounds clip the neighborhood. Missing/ambiguous predictions, edge
    minima and rejected candidates appear in the diagnostics table.
    """)
    return


@app.cell
def _():
    from guided_search import PARAMETERS, guided_search

    return PARAMETERS, guided_search


@app.cell
def _(device_parameters, np):
    # Change this name in situ; all second-parameter units follow the name.
    # Choices: EJ2_GHz, C1_shunt_fF, C2_shunt_fF, C1r_fF, C2r_fF.
    sweep_parameter = "EJ2_GHz"
    ej1_grid_ghz = np.linspace(0.5*device_parameters["EJ1_GHz"], 1.5*device_parameters["EJ1_GHz"], 61)
    fixed_ej2_ghz = device_parameters["EJ2_GHz"]  # Used only when the second sweep is a capacitance.

    prediction_bounds = {
        "EJ2_GHz": (1.0, 50.0),
        "C1_shunt_fF": (10.0, 100.0),
        "C2_shunt_fF": (10.0, 100.0),
        "C1r_fF": (.1, 20.0),
        "C2r_fF": (.1, 20.0),
    }[sweep_parameter]
    neighborhood_mode = "fractional"  # "fractional" or "absolute"
    fractional_half_width = 0.15
    neighborhood_samples = 15
    absolute_half_width = 2.0  # Selected parameter's units (GHz or fF).
    neighborhood_step = 0.25  # Selected parameter's units (GHz or fF).
    min_hybridization = 0.95
    prediction_samples = 65
    minimizer_xtol = 1e-4  # Selected parameter's units.
    search_settings = {
        "fixed_ej2_ghz": fixed_ej2_ghz,
        "mode": neighborhood_mode,
        "fractional_half_width": fractional_half_width,
        "samples": neighborhood_samples,
        "absolute_half_width": absolute_half_width,
        "step": neighborhood_step,
        "min_hybridization": min_hybridization,
        "prediction_samples": prediction_samples,
        "xtol": minimizer_xtol,
    }
    return ej1_grid_ghz, prediction_bounds, search_settings, sweep_parameter


@app.cell
def _(
    ej1_grid_ghz,
    mo,
    model,
    prediction_bounds,
    search_settings,
    sweep_parameter,
):
    # Reset button whenever any model or sweep input changes.
    _ = model, ej1_grid_ghz, sweep_parameter, prediction_bounds, search_settings
    run_sweep = mo.ui.run_button(label="Run resonance-guided search")
    run_sweep
    return (run_sweep,)


@app.cell(hide_code=True)
def _(
    Path,
    ej1_grid_ghz,
    guided_search,
    mo,
    model,
    prediction_bounds,
    run_sweep,
    save_results,
    search_settings,
    sweep_parameter,
):
    mo.stop(not run_sweep.value, mo.md("Click **Run resonance-guided search** above."))
    with mo.status.spinner(title="Predicting resonance and searching nearby dressed crossings"):
        operating_points, coarse_scan, search_diagnostics = guided_search(
            model, ej1_grid_ghz, sweep_parameter, prediction_bounds, **search_settings,
        )
    result_path = save_results(
        Path(__file__).resolve().parent / "data" / "readout_results.npz",
        model, operating_points, coarse_scan,
        {"ej1_grid_ghz": ej1_grid_ghz.tolist(), "sweep_parameter": sweep_parameter,
         "prediction_bounds": prediction_bounds, **search_settings},
        diagnostics=search_diagnostics,
    )
    mo.md(f"Found **{len(operating_points)}** candidates from **{len(coarse_scan)}** coarse evaluations. Saved `{result_path}`.")
    return coarse_scan, operating_points, search_diagnostics


@app.cell(hide_code=True)
def _(
    PARAMETERS,
    coarse_scan,
    ej1_grid_ghz,
    mo,
    np,
    operating_points,
    plt,
    search_diagnostics,
    sweep_parameter,
):
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4.5))
    if coarse_scan:
        from matplotlib.colors import Normalize
        _norm = Normalize(vmin=min(p["splitting_ghz"] for p in coarse_scan)*1e3,
                          vmax=max(p["splitting_ghz"] for p in coarse_scan)*1e3)
        # Each slice has its own local y grid. Draw separate strips so no
        # interpolation or coloring bridges unevaluated neighborhoods.
        _x = np.asarray(ej1_grid_ghz)
        _xedges = (np.r_[_x[0]-0.25, _x[0]+0.25] if len(_x) == 1 else
                   np.r_[_x[0]-(_x[1]-_x[0])/2, (_x[:-1]+_x[1:])/2, _x[-1]+(_x[-1]-_x[-2])/2])
        _groups = sorted(set((p["slice_index"], p["root_index"]) for p in coarse_scan))
        for _slice, _root in _groups:
            _rows = sorted([p for p in coarse_scan if p["slice_index"] == _slice and p["root_index"] == _root], key=lambda p: p["sweep_value"])
            _y = np.array([p["sweep_value"] for p in _rows])
            # End cells stop at the evaluated neighborhood endpoints.
            _yedges = np.r_[_y[0], (_y[:-1]+_y[1:])/2, _y[-1]]
            _mesh = _axes[0].pcolormesh(_xedges[_slice:_slice+2], _yedges,
                                      np.array([p["splitting_ghz"]*1e3 for p in _rows])[:, None],
                                      norm=_norm, shading="flat")
        _fig.colorbar(_mesh, ax=_axes[0], label="Dressed splitting (MHz)")
    _predicted = [d for d in search_diagnostics if d["prediction"] is not None]
    if _predicted:
        _axes[0].scatter([d["EJ1_GHz"] for d in _predicted], [d["prediction"] for d in _predicted],
                         marker="x", color="red", s=18, label="Local resonance prediction")
    if operating_points:
        _get = lambda key: np.array([p[key] for p in operating_points])
        _f = _get("operating_frequency_ghz")
        _axes[0].scatter(_get("ej1_ghz"), _get("sweep_value"), s=18, color="white", edgecolors="black", label="Accepted crossing")
        _axes[1].scatter(_f, 1e3*_get("splitting_ghz"), s=18)
        for _chi, _label in (("chi_p_ghz", r"$\chi_i=\chi_+$"), ("chi_m_ghz", r"$\chi_i=\chi_-$")):
            _den = np.abs(_get(_chi))
            _ratio = np.divide(np.abs(_get("delta_chi_ghz")), _den,
                               out=np.full_like(_den, np.nan), where=_den > 0)
            _axes[2].scatter(_f, _ratio, s=18, label=_label)
        _axes[2].legend()
    if _predicted or operating_points:
        _axes[0].legend(fontsize=7)
    _axes[0].set(xlabel="EJ1/h (GHz)", ylabel=PARAMETERS[sweep_parameter], title="1. Search neighborhoods")
    _axes[1].set(xlabel="Operating frequency (GHz)", ylabel="Logical frequency (MHz)", title="2. Logical frequency")
    _axes[2].set(xlabel="Operating frequency (GHz)", ylabel=r"$|\Delta\chi/\chi_i|$", title="3. Relative mismatch")
    _axes[2].set_yscale("symlog", linthresh=1e-8)
    _axes[2].set_ylim(0.001,1)
    for _ax in _axes[1:]:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.vstack([mo.md("No accepted candidates. Inspect the diagnostics below.") if not operating_points else mo.md(""), _fig])
    return


@app.cell(disabled=True)
def _(mo, search_diagnostics):
    mo.vstack([mo.md("**Prediction and neighborhood diagnostics**"), mo.ui.table(search_diagnostics)])
    return


@app.cell
def _(mo, operating_points):
    mo.ui.table(sorted(operating_points, key=lambda p: p["abs_delta_chi_over_chi_p"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    Energies are H/h in GHz. Branch labels g, m, p, r, mr, pr are assigned by
    overlap with eigenstates of the interacting isolated pair × resonator Fock states.
    A small overlap signals ambiguous branch identity; inspect it before interpreting chi.

    $$\chi_p=(E_{pr}-E_p)-(E_r-E_g),\quad
      \chi_m=(E_{mr}-E_m)-(E_r-E_g),\quad
      R=| (\chi_p-\chi_m)/\chi_p |.$$

    The operating frequency is the mean of the two rail transitions; the logical
    frequency is their separation. Near a simple avoided crossing, half the minimum
    separation estimates the effective exchange coupling.

    Small R measures matching; inspect the actual shift magnitudes as well.
    This static spectrum calculation does not simulate measurement pulses,
    resonator decay, assignment fidelity, or measurement-induced transitions.
    """)
    return


if __name__ == "__main__":
    app.run()
