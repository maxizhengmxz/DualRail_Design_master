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
The legacy strength-optimization workflow maximizes **abs(mean_chi)**, equivalently
abs(chi_sum)/2. The retained feasibility scans stop at the first validated
solution and do not optimize readout strength.

| Condition | Threshold |
| --- | --- |
| Matching | abs(delta_chi / mean_chi) <= 1% |
| Readout strength | abs(mean_chi) >= 1 MHz |
| Multilevel dispersive ratio | <= 0.1 |
| Logical frequency | 90-110 MHz model default; 140-160 MHz in the retained scans |
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

## Frequency-keyed fine C1-C2 campaigns

The fine-map driver accepts an explicit logical-frequency interval:

~~~bash
python -m readout_workflow.fine_readout_feasibility --run \
  --logical-frequency-mhz 140 160 --output PATH_FOR_RAIL_A --workers 4
python -m readout_workflow.fine_readout_feasibility --run \
  --logical-frequency-mhz 80 100 --output PATH_FOR_RAIL_B --workers 4
~~~

Each output directory contains its own append-only `samples.sqlite` database.
The interval is recorded in both `settings.json` and the database calculation
signature. A campaign can be resumed with the same interval and source version;
attempting to reuse the directory with a different interval is rejected. Do not
mix rail-A and rail-B events in one database.

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

## Retained results and feasibility workflows

Only two scientific sweep datasets are retained:

| Dataset | Contents | Driver |
| --- | --- | --- |
| `data/feasibility_150mhz_step1_diag2/` | 4,278 capacitance pairs at 1 fF spacing; 339 viable designs; all 998 requested 2 fF diagnostic targets complete, plus 341 earlier off-grid diagnoses | [fine_readout_feasibility.py](fine_readout_feasibility.py) |
| `data/direct_ec_step5/` | 48 directly searched loaded-EC pairs at 5 MHz spacing, all validated; all 36 tested holes in the old occupancy plot filled | [direct_ec_feasibility.py](direct_ec_feasibility.py) |

The two reusable calculation drivers/templates live inside this package;
this README is their guide. Opening/importing them starts no calculation.
From the project root, module entry points are
`python -m readout_workflow.direct_ec_feasibility` and
`python -m readout_workflow.fine_readout_feasibility`. Without `--run` these
only show usage or saved status. Calculations are performed only when requested.
The grid and window in each file reproduce its retained campaign; inspect/edit
those settings before preparing a different campaign.

### How the direct loaded-EC campaign uses the fine capacitance map

The fine map scans the physical shunt-capacitance coordinates
`C1_shunt_fF,C2_shunt_fF` on the strict triangular grid
`36 <= C2 < C1 <= 128` fF. At each physical grid point it searches the mean
bare frequency, differential detuning, `C12_fF` and the equal readout
capacitors `C1r_fF=C2r_fF`. A successful pixel retains the first fully audited
witness found under the bounded search policy, not every feasible solution and
not a global optimum. Its loaded EC values are outputs of the inverse of that
witness's complete three-node capacitance matrix.

The direct-EC driver reads only old records with status `validated_pass` or
`recovered_viable` and projects each retained physical witness to

```text
(point["loaded_ec1_mhz"], point["loaded_ec2_mhz"]).
```

Thus target selection occurs in the already converted loaded-EC plane. It does
not construct a hull in the original physical `(C1,C2)` plane. Differences in
the witnesses' `C12`, readout capacitors and EJs have already affected their
loaded-EC coordinates through the circuit calculation, but those additional
coordinates are discarded by this two-dimensional projection.

For a 5 MHz step, the candidate axes are

```text
EC1 = first multiple of 5 MHz >= min(observed EC1), ..., max(observed EC1)
EC2 = first multiple of 5 MHz >= min(observed EC2), ..., max(observed EC2).
```

The driver supports two target-domain policies:

- `--domain hull` is the default. It expands the convex hull of the projected
  witnesses by a factor of 1.5 about the mean of the hull vertices, then
  retains lattice points inside that scaled hull. SciPy triangulates the
  expanded polygon and tests whether each lattice point belongs to it.
- `--domain rectangle` retains every Cartesian pair from the two candidate
  axes, including the corners excluded by the hull. It is a diagnostic for
  feasible points that the one-witness-per-physical-pixel projection may have
  omitted from the historical hull.

The hull is a campaign-selection heuristic, not a physical feasibility
boundary. Because only one witness is retained at each successful physical
pixel, other feasible choices of `C12`, readout capacitance and EJ can project
outside the observed hull. Conversely, a convex hull fills straight-line gaps
between witnesses and can contain infeasible targets. Therefore neither

```text
inside observed hull  => feasible
outside observed hull => infeasible
```

is valid. The rectangle mode reduces this particular geometric restriction,
but it is still bounded by the coordinate-wise extrema of the old witnesses;
it does not explore EC values beyond the observed min/max rectangle. The
scaled-hull mode does explore beyond those extrema while retaining the observed
hull shape as the campaign-selection guide.

Use `--source` to provide the `results.json` exported by the fine C1-C2
campaign for the same rail and logical-frequency interval. The source path,
content hash, interval, and hull scale are recorded in the direct-EC settings.
The scaled lattice is clipped to the configured 150-400 MHz loaded-EC range;
the upper bound remains strict.

The default `--orientation-domain full` uses isolated-readout label symmetry
without forming one convex hull across both orientations. It constructs and
scales the observed hull and its `(EC1,EC2) -> (EC2,EC1)` mirror separately,
then takes their lattice union. Mirrored fine-map witnesses provide warm starts
with the differential detuning sign reversed, but every ordered direct-EC
target is rebuilt and audited normally. Downstream gate tasks must retain the
ordered labels and must not copy fidelity from the mirrored point because the
physical A2-B1 link breaks this symmetry.

### What is recalculated at each direct-EC target

The direct driver does not introduce a second Hamiltonian expressed only in EC
values. For each fixed target `(EC1,EC2)` and each trial `(C12,Cg)`, it uses the
Schur complement of the physical three-node capacitance matrix to solve
analytically for physical shunts `(C1,C2)` whose inverse-matrix diagonals give
the requested loaded ECs. It then calls the same physical-capacitance
`point_from_x` evaluator as the fine map. That evaluator rebuilds and inverts
the complete capacitance matrix, solves EJ from the trial local frequencies,
diagonalizes the same multilevel Hamiltonian and calculates the same readout
observables. The audit rejects a target error above `1e-6` MHz and shunts
outside 36-128 fF.

For each target, the three old witnesses nearest in Euclidean loaded-EC space
provide initial guesses for mean frequency, detuning, `C12` and `Cg`. Their old
shunt capacitances are not reused: new shunts are reconstructed at every trial.
The bounded local search uses at most three starts, 35 least-squares evaluations
per start and a 25-second soft budget. Only after a screening candidate passes
does adaptive validation recompute that fixed candidate at stronger bases and
perform the EJ sign probes. Accordingly, `no_validated_solution_found` means
that these warm starts and this budget found no audited witness; it is not proof
of physical infeasibility. Outside-hull targets can be harder partly because all
available warm starts lie on one side of the target, as well as because circuit
constraints may genuinely become tighter there.

Use a new output directory for each domain because the target-domain choice,
target list, source-witness count and source hashes are part of the saved policy
identity:

```sh
python -m readout_workflow.direct_ec_feasibility --run --domain rectangle \
  --output readout_workflow/data/direct_ec_step5_rectangle
```

The default logical-frequency acceptance interval remains 140--160 MHz. A
different independent campaign must use a new output directory and specify the
interval explicitly, for example:

```sh
python -m readout_workflow.direct_ec_feasibility --run \
  --logical-frequency-mhz 80 100 \
  --output readout_workflow/data/direct_ec_80_100_step5
```

Both datasets target logical frequency 140-160 MHz and use first-feasible
selection, without readout-strength optimization or robustness. The direct-EC
driver reads the fine map as a contour and seed library. It solves C1 and C2
analytically at every trial so the loaded EC targets remain exact as C12 and
equal readout coupling capacitances vary. The physical C1/C2 range is 36-128 fF.
Neither a bounded-search miss nor an untested plotting cell proves infeasibility.

The fine-map driver is self-contained: it no longer depends on the deleted
coarse pilot, old quick-search driver, or diagnosis-resume script. It supports
fresh 1 fF feasibility maps followed by 2 fF diagnosis. The default command only
shows saved status. Running revised code requires a new output directory;
existing datasets retain their original source/policy signatures and are not
rewritten or relabeled. Historical provenance may name removed pilot files.

The latest retained reports are in `output/pdf/`:

- `readout_140_160mhz_fine_feasibility_diagnosis_ec_binary.pdf` (the original
  report snapshot; final data in SQLite/results.json contain all 998 diagnoses).
- `direct_loaded_ec_5mhz_feasibility.pdf`.

Saved data and reports are ignored by Git; copy these folders when moving
the project. Shared physics, adaptive validation, its tests, and compact
speed-benchmark evidence remain available. No scientific scan is started by
this cleanup.

### Adaptive validation and equivalent real Hamiltonian

Routine feasibility validation and EJ sign probes now use 6 transmon levels
per rail and 5 resonator levels (180 states). Near numerical acceptance
boundaries, nominal checks escalate to 8 levels/6 resonator levels, then the
original 12 levels/10 resonator levels if needed. A deterministic 5% sample
receives full nominal and probe checks. Guard widths are engineering settings,
not rigorous global error bounds. `audit_point(..., force_full=True)` retains
the original large-basis checks for final selected designs. Physical
acceptance thresholds are unchanged. Nominal failures skip the probe grid;
probe checks can stop at their first failure. Certificates record nominal
and probe bases independently and invalidate when either changes.

The same Hamiltonian is represented in a real Fock-state phase convention:
U = I_pair tensor diag(i**n), H_real = U^dagger H_complex U. This is an exact
basis change, not a rotating-wave or two-level approximation. Prior saved
results remain unchanged. All 48 saved direct-EC designs pass the revised
routine validation; compared with their old full-basis records, the largest
nominal chi error is 0.00445 kHz and largest EJ-probe chi error is 0.11461 kHz.
The evidence is in `data/adaptive_validation_regression.json`. These checks
cover the existing 140-160 MHz examples, not the entire proposed 90-110 MHz
search domain. New convergence checks are still needed when exploring it.

`python -m readout_workflow.benchmark_validation` performs the bounded saved-design
benchmark without launching a large parameter sweep. The measured four-worker
comparison is retained in `data/validation_speed_test/`: 12 complex/full audits
161.34 s; the same 12 real/full audits 58.32 s; adaptive audits of all 48 designs
13.21 s including three representative full checks. Repeating the complete
48-target feasibility search with the new setup took 15.04 s; all 48 passed.
At identical saved inputs, maximum probe chi difference was 0.11461 kHz.
Historical old search runtime was about 10.6 minutes; it was measured at a
different time/system load. No broad 90-110 MHz sweep was started by this test.
