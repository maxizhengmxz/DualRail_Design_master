## DTC Dual-Rail Analysis

This repository contains exploratory Marimo notebooks plus reusable helpers for
DTC dual-rail circuit simulations, Monte Carlo robustness studies, and resonator
sweeps.

For the single-junction readout capacitance-map and robustness study, start with
[`Readout_ParamSpace_Robustness.py`](Readout_ParamSpace_Robustness.py).
The calculation-only modules are described in the
[readout guide](readout_workflow/README.md).
Running the entry point without arguments shows saved status; it starts no search.
Historical reports, report generators and superseded readout-study scripts
have been removed.

The reusable API now lives under `dtc_dualrail/`:

- `circuits.py`: circuit factories for DTC, DT, tunable-DT, and the direct
  five-transmon reference model.
- `device.py`: unified device definition, single-transmon validation, and
  multi-point reduced-pair calibration.
- `tracking.py`: eigenstate and branch tracking helpers.
- `sampling.py`: parameter perturbation and correlation utilities.
- `monte_carlo.py`: Monte Carlo dataset generation and cache loading.
- `tunable_resonator.py`: dispersive-shift evaluation and resonator tuning.
- `sweeps.py`: reusable parameter sweep helpers.
- `io.py`: dataset persistence helpers.
- `runtime.py`: shared numerical-runtime setup, including BLAS thread limiting.

Legacy helper modules have been removed; import reusable functionality from
`dtc_dualrail`.

Generated artifacts are intentionally outside version control:

- `data/`: NetCDF datasets and cached simulation outputs.
- `figs/`: publication or diagnostic figures.
- root-level `*.png` and `*.pdf`: ad hoc exported plots.

Prefer writing reusable calculations to `dtc_dualrail/` and keeping Marimo files
focused on experiment parameters, cache paths, and plotting.

`logical_fsim.py` consumes the versioned
`configs/logical_fsim.toml` configuration. It defines the physical pairs,
coupler, reduced-model basis, reference and static grids, selected operating
points, pulse sweeps, solver, execution controls, and reporting thresholds.
Set `LOGICAL_FSIM_CONFIG` to run another configuration and
`LOGICAL_FSIM_DATA` to override only the artifact directory. Once the reference
contours are available, `device.calibrate(contour_a, contour_b)` returns the
derived `FiveTransmonConfig` and both pair-calibration reports.

This replaces the notebook's separate `gen_tdt(**parameters)` dictionaries and
manually assembled reduced-model inputs. Rail `EJmax` is the junction-energy sum;
asymmetry is the absolute junction-energy difference divided by that sum. Reduced
rail charging energies and intra-pair couplings are fitted across every point of
the selected contour; carrier, logical splitting, two-excitation levels, and
charge matrix elements are included whenever the reference data provides them.
The coupler and its reduced-model links use GHz, while the direct-reference
links use explicitly configured capacitances in fF.
The lower-level factories and simulation configs remain available to other users.

Phase 1 produces fingerprinted NetCDF reports for the five single-transmon
`ncut = 15, 20, 25, 30` sweeps, pair-fit residuals, diabatic and adiabatic branch
tracking, overlap-matched fixed-basis validation, and flat-top static/dynamic
rate agreement. The production config enforces these gates before pulse
characterization; the smoke config records them without enforcement so the full
DAG remains testable with deliberately tiny numerical inputs.

`TDTPairConfig` preserves the circuit factory's units: Josephson energies in GHz,
`C1`/`C2` in **MHz of charging energy**, other capacitances in fF, and `Lr` in nH.
Reference, static, pulse, and trajectory caches validate fingerprints of all
numerically relevant inputs. An incompatible reference or static cache reports
that its corresponding recompute setting must be enabled; pulse and trajectory
caches are regenerated automatically. `configs/logical_fsim_smoke.toml` is the
small deterministic configuration used by the headless notebook-DAG test.

Before numerical work in a notebook, call
`dtc_dualrail.runtime.configure_single_threaded_blas()` in the cell that imports
the simulation helpers. This limits both already-loaded BLAS libraries and
future imports, so process-parallel sweeps do not each launch a full BLAS thread
pool. Setting environment variables after NumPy/SciPy import is insufficient.
After changing runtime setup, restart the notebook kernel and its workers.
Inspect active limits with `threadpoolctl.threadpool_info()`; BLAS entries should
report `num_threads: 1`.

Run the unit tests with:

```powershell
uv run python -m unittest discover -s tests
```
