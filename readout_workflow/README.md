# Single-junction dual-rail readout

## Reproduce the latest capacitance-map and robustness study

Start with **[`reproduce_readout.py`](reproduce_readout.py)**. This is one annotated
Python file containing the circuit model, state diagnostics, nominal feasibility
search, EJ-only recalibration, sampling, checkpointing and plotting. It does not
import any other project Python files. The earlier scripts and Marimo notebook
remain available as history and for interactive exploration.

From the repository root, rebuild the recorded figures and tables:

```powershell
.\.venv\Scripts\python.exe readout_workflow/reproduce_readout.py
```

The default run reads archived results; it does **not** rerun the overnight
optimization. Outputs go to `output/readout_reproduction/`: six figures, CSV
tables, a summary and provenance. The existing narrative PDF is preserved.
The state-mixing figure uses a linear H axis from 80% to 100%.

Add `--mode verify` to audit all saved witness constraints and independently
recalculate the three selected designs, their label-exchanged mirrors, and their
larger-basis convergence. Use `--mode map` or `--mode robustness --workers 4`
for fresh, resumable numerical searches; `--mode all` runs both. These can take
hours. Pilot with `--mode map --pair 87 73`, or
`--mode robustness --limit 1` (one perturbation per design).

Editable settings and full-precision selected inputs are near the top of the
file. Changing the source or numerical settings requires a new `--output-dir`
to avoid mixing incompatible checkpoints. Nominal loaded EC is 150-400 MHz
(upper bound strict), logical frequency is 90-110 MHz, and H>=95%. The robustness
rescue order is EC floor, flexible operation frequency, logical frequency, H,
then EJ/EC. Operation frequency always means the mean dressed transitions.

Exact archived figures/counts are reproduced by the default mode. A fresh bounded
optimization may discover different witnesses or counts; no-witness results are
not global infeasibility proofs. Run `--help` for all options and definitions.

## Interactive Marimo workflow

A standalone Marimo project for one dual-rail pair and its shared resonator.
The original notebooks are unchanged. Only NumPy/SciPy, Matplotlib, Marimo and
threadpoolctl are required; no imports from the parent project.

From this directory:

```powershell
uv run marimo edit readout.py
```

Or use the already installed environment, from the parent project directory:

```powershell
.\.venv\Scripts\python.exe -m marimo edit readout_workflow/readout.py
```

1. In the device-definition cell, set `resonator_frequency_ghz` and
   `resonator_impedance_ohm`. Cr and Lr are derived using C=1/(omega*Zmode)
   and L=Zmode/omega. `resonator_impedance_kind="mode"` means equivalent LC
   impedance; `"cpw_lambda4"` or `"cpw_lambda2"` means characteristic line Z0,
   converted to fundamental-mode impedance (4*Z0/pi or 2*Z0/pi respectively).
   CPW conversions assume an ideal uniform unloaded line, referenced to voltage
   at an antinode. Frequency is bare, before coupling-capacitor loading.
   All editable transmon and coupling capacitances remain in **fF**.
   A reference table at the beginning lists both original-device presets.
   The **Device parameters** cell contains one editable `device_parameters`
   dictionary: both EJs, five capacitances, bare resonator frequency and impedance.
   It preserves the current manual values; copy from the reference table when
   desired. Numerical basis settings and impedance conversion/device construction
   are in separate cells. The table does not automatically populate the device.
   EJs initialize the inspection controls and search neighborhood and are not
   recalibrated automatically after device edits.
2. Inspect the capacitance matrix, then choose EJ1 and explicit lower/upper EJ2
   limits (default 10–30 GHz). The 61-point EJ2 sweep plots loaded uncoupled and
   dressed transmon frequencies and the conditional resonator shifts chi_minus
   and chi_plus in MHz.
3. In the search configuration cell, edit `sweep_parameter` and `ej1_grid_ghz`.
   Choose `EJ2_GHz`, `C1_shunt_fF`, `C2_shunt_fF`, `C1r_fF` or `C2r_fF`.
   C12 stays fixed. `fixed_ej2_ghz` sets EJ2 during capacitance sweeps.
4. Set `prediction_bounds` and select `neighborhood_mode`: `fractional` uses
   `fractional_half_width` and `neighborhood_samples`; `absolute` uses
   `absolute_half_width` and `neighborhood_step` in the selected parameter's units.
5. Click **Run resonance-guided search**. Edits reset the button.
6. Inspect (1) the search colormap with predictions and candidates, (2) logical
   frequency, (3) both absolute relative mismatches, divided by chi_plus/chi_minus.
   Inspect prediction diagnostics and candidate assignment overlaps in the tables.

Each transmon has exactly one junction, with no separate junction capacitance.
For comparison with the original device, the two original 5 fF junction capacitors
are absorbed into each explicit shunt. This preserves the full capacitance matrix
while replacing the SQUID potential with its effective single-junction energy.
The original shunt EC/h inputs are 354/434 MHz (TDT1) or 294/364 MHz (TDT2);
these are not the loaded EC values derived from the inverse capacitance matrix.
No SQUID asymmetry, flux, second junction or Cj parameter remains in this project.

`compare_original.py` reads both original NetCDF dispersive caches using xarray
(available in the parent environment), converts the cached flux points using
EJeff = sqrt(EJa^2+EJb^2+2*EJa*EJb*cos(2*pi*flux)), and compares spectra/shifts.
It solves for a 4.4 GHz dressed carrier along a full-circuit splitting minimum,
checks a larger basis, and writes `data/original_comparison.json`.
The exact cancellation is narrow: the cached sub-1e-3 points occur around
4.256 GHz (TDT1) and 4.467 GHz (TDT2), rather than throughout 4.2–4.5 GHz.

`model.py` constructs C and its inverse. Local EC values are e²(C⁻¹)ii/2; off-diagonal
entries generate all three charge-charge interactions. Transmons are diagonalized
in a charge basis and projected onto retained local levels. The resonator uses a
Fock basis with effective capacitance 1/(C⁻¹)rr. The resulting coupled Hamiltonian
is diagonalized to extract dressed transitions and conditional shifts.

For each EJ1, `guided_search.py` predicts the second parameter by solving equality
of the two loaded local transmon frequencies, recalculating the inverse capacitance
matrix for every capacitance trial. It searches only neighborhoods of these roots,
then refines interior full-circuit splitting minima versus the second parameter.
It retains hybridization >= 0.95 and evaluates readout mismatch afterward.
Unselected device parameters stay fixed. Both shunt and coupling capacitances
are physical capacitances in fF; there are no charging-energy inputs under C names.

The predictor detects sign-changing roots on a configurable `prediction_samples`
grid. Multiple roots get separate neighborhoods. A tangent root without a sign
change can be missed; inspect bounds or refine the prediction grid if necessary.
Missing or nonunique roots, clipped neighborhoods, edge minima and hybridization
rejections are reported explicitly. The colormap leaves uncomputed regions blank;
it draws separate local strips, not a rectangular grid interpolated across gaps.
The positive prediction bounds must be adapted to the desired physical range.
As before, splitting minima only approximate maximum mixing if coupling varies
strongly with the selected parameter; inspect the retained hybridization scores.

The notebook saves the latest run as `data/readout_results.npz`, including device,
basis, selected parameter, bounds, neighborhood settings, prediction diagnostics,
candidate rows and the coarse scan. It does not load old
flux caches. Every explicit new run replaces this result file.

The reported loaded resonator baseline is not the fully dressed resonator
transition. `dressed_r_ghz` contains the latter, defined as (Er-Eg)/h.
`python check_resonator.py` compares the bare, loaded and dressed values and
checks a larger basis at three EJ2 settings. With the current defaults the
couplings are not weak relative to detuning throughout the sweep; some branch
assignments become ambiguous. Stable eigenenergies alone do not validate chi
labels in that regime.

For a distributed CPW, Cr and Lr must represent the selected mode with a defined
voltage normalization at the coupling location, not arbitrary total line values.
The current values are inherited lumped defaults, not a calibrated CPW model.
For an unloaded uniform line, using voltage at an antinode as the mode coordinate,
Cr = c_per_length * length / 2 for the fundamental lambda/4 or lambda/2 mode;
Lr = 1/(omega_r**2 * Cr). Coupling at other positions introduces mode-shape factors.
Include coupling-capacitor loading only once when using extracted mode parameters.

Validate convergence by increasing ncut, transmon_levels, resonator_levels and
eigenlevels, then refining the EJ grid. Overlap scores diagnose label ambiguity;
they do not establish basis convergence. No dissipation or readout pulses are modeled.

Run checks with `python -m unittest discover -s tests` from this directory.
