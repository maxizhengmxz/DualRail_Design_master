# Single-junction dual-rail readout

Start with [Readout_ParamSpace_Robustness.py](../Readout_ParamSpace_Robustness.py)
in the main folder. It is a small entry point; opening it or running it without
arguments only shows saved status. Calculations are performed on request.

## Reading the calculation

| File | Purpose |
| --- | --- |
| [model.py](model.py) | Single pair/resonator circuit implementation: capacitance matrix, Hamiltonian, dressed-state assignment, chi and hybridization; notebook search/export helpers |
| [calculation.py](calculation.py) | Batch device defaults, acceptance conditions, multilevel diagnostics and initial feasibility search; reuses model.py's circuit evaluator |
| [optimization.py](optimization.py) | Maximize readout strength at fixed C1/C2, then select nearby solutions along each row |
| [diagnostics.py](diagnostics.py) | Explain unsuccessful pixels using the nearest found chi-matched candidate |
| [validation.py](validation.py) | Larger-basis validation, chi quantities and inter-transmon coupling |
| [storage.py](storage.py) | Append-only SQLite results, resume identity and Windows file-lock retries |
| [workflow.py](workflow.py) | Schedule map stages, checkpoint every result and export numerical data |

The batch calculation has no report/PDF dependencies. There are no dated runners,
source-generation scripts or automatic robustness jobs.

## Physical model and units

Shared local-transmon diagonalization and Basis now live in `core/transmon.py`;
loaded EC, charge coefficient J and J01 extraction live in
`core/electrostatics.py`. This workflow owns its pair/resonator circuit,
observables, conditions and searches. Coupler-off imports shared physics
directly, and uses this workflow only for readout diagnostics. See
[the dependency map](../core/README.md). There is one pair/resonator circuit
implementation, in `model.py`. `calculation.py` inherits that evaluator and
supplies its original batch Device defaults. The notebook's original defaults
and its guided/minimum-splitting searches remain distinct from the batch
feasibility/strength optimization. These search methods were not duplicates.
Checkpoint identity includes model.py as well as the shared core physics.

Charge coupling is defined by `H_int/h = J_ij n_i n_j`, with Cooper-pair
number operators and `J_ij = 4 e^2 (C^-1)_ij/h` in GHz. Loaded EC and J
come from the same full capacitance matrix. `J12_ghz` reports this coefficient;
`J01_12_mhz` reports `1000*|<10|H_int/h|01>|`, extracted using the numerical
local charge matrix elements. J01 is a cyclic frequency, not J, an angular
rate, or the avoided-crossing gap (2*J01 at resonance in a two-state model).
The older audit key `g12_over_2pi_mhz` remains an alias for J01 in MHz.
Transmon-resonator `g*r_over_2pi_mhz` keys likewise remain cyclic matrix
elements using the resonator zero-point charge, not bare charge coefficients.

Two single-junction transmons, specified by Ci and EJi, couple through C12 and
to a shared resonator through Cir. Junction parasitic capacitors are omitted.
The inverse of the full three-node capacitance matrix determines the loaded
charging energies and interactions.

- Physical capacitances: fF; inductance: nH.
- EJs and ECs: E/h in GHz; the Hamiltonian is H/h.
- The nominal resonator is 6 GHz with 50 ohm equivalent-mode impedance.
  Cr and Lr follow from frequency and impedance.
- Operation frequency means the mean **dressed** logical transitions.
- Logical frequency means the separation of those transitions.
- Optimizer coordinates use the two local bare frequencies to determine EJs;
  they are not the reported operation frequency.

The editable physical thresholds are together in the Conditions dataclass near
the beginning of calculation.py.

## Accepted nominal points

Let mean_chi = (chi_plus + chi_minus)/2 and delta_chi = chi_plus - chi_minus.
The objective is **maximize abs(mean_chi)**, equivalently abs(chi_sum)/2.

| Condition | Threshold |
| --- | --- |
| Matching | abs(delta_chi / mean_chi) <= 1% |
| Readout strength | abs(mean_chi) >= 1 MHz |
| Multilevel dispersive ratio | <= 0.1 |
| Logical frequency | 90-110 MHz |
| Hybridization H | >= 95% |
| State-assignment overlaps | >= 95% |
| Loaded EC/h, each transmon | >= 150 MHz and strictly < 400 MHz |
| EJ/h, each transmon | strictly < 25 GHz |
| EJ/EC, each transmon | strictly > 50 |
| Operation frequency | 4-8 GHz, strictly below the dressed resonator |
| Resonance sensitivity | Both chi signs survive independent +/-1% EJ probes |

The below-resonator condition is intended to suppress MIST effects. The sign
screen uses a 5-by-5 EJ grid to reject points near qubit-resonator resonances,
requiring reliable assignments. Neither check models finite-drive MIST.

H is normalized charge-transition balance, not a full-model state overlap.
In the ideal one-excitation two-state limit, populations are H/2 and 1-H/2;
H = 95% corresponds to 47.5% / 52.5% mixing.

## Map stages

1. Search C1 and C2 from 36 to 128 fF at 1 fF spacing, retaining only C1 > C2:
   4,278 independent pixels. Retry unsuccessful positions more deeply.
2. Diagnose every unsuccessful pixel at the same spacing. Minimize normalized
   constraint deficits subject to 1% chi matching; retain all failed conditions.
3. At viable pairs optimize both EJs, C12 and the common C1r = C2r to maximize
   abs(mean_chi). C12 spans 0.005-15 fF and Cir spans 0.1-25 fF.
4. Continue along increasing C1 within each C2 row, retaining at least 99.5%
   of the best validated strength found at each pixel.
5. Audit accepted points at the larger basis and save numerical JSON/CSV.

Search misses do not prove infeasibility, and best-found maxima do not prove
global optimality. Neighboring pixels can have different optimized components.

**Robustness is outside the current scope.** The agreed future policy is
first-feasible EJ recalibration, with extra starts only for failed cases.
There is no robustness command in this runner.

## Fresh runs and safe resumption

No saved map, historical seed archive, perturbation samples or approved-device
result is shipped or required. A fresh run starts from algorithmic seeds;
newly calculated neighboring points supply additional starts. The workflow
creates a new ledger in the selected output directory and can resume that
ledger later. Changing equations or settings requires a fresh output directory.

Readout optimization is separate from coupler-off analysis. Run it only when
explicitly requested; otherwise supply an externally optimized pair directly
to the coupler workflow.

The default command is read-only:

~~~powershell
.\.venv\Scripts\python.exe Readout_ParamSpace_Robustness.py status
~~~

When a calculation is requested, the same entry point resumes the map:

~~~powershell
.\.venv\Scripts\python.exe Readout_ParamSpace_Robustness.py run --workers 4 --background
~~~

The hidden background supervisor retries interrupted processes. Results commit
one event at a time to SQLite; status-file locks do not stop the calculation.
Changed equations, settings or seeds require a new output directory.

## Interactive notebook

[readout.py](readout.py) is the manually edited Marimo notebook. Its original
lightweight helpers are model.py, guided_search.py and reference_devices.py.
They are retained separately so the interactive workflow and manual inputs
remain unchanged. Use calculation.py for the batch map's acceptance rules.

## Verification

From the main folder:

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s readout_workflow/tests
~~~

Tests cover the circuit, transmon-label symmetry, strict energy
bounds, notebook helpers, checkpoint identity and real Windows file locks.
