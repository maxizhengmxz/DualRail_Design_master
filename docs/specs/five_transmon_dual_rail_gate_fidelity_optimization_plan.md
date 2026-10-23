# Five-Transmon Dual-Rail Gate Fidelity Optimization Plan

Status: Phases 0--2 implemented; Phases 3--4 not started
Last assessed: 2026-09-15  
Primary workflow: `logical-resonance gate/logical_fsim.py`
Simulation backend: `logical-resonance gate/logical_resonance_gate/dualrail_fsim.py`

## 1. Purpose and source documents

This document is the execution handoff for improving the logical iSWAP and CZ
fidelities produced by `logical-resonance gate/logical_fsim.py`. It does not replace the broader design
and implementation documents:

- `docs/specs/five_transmon_dual_rail_gate_implementation_plan.md`
- `docs/specs/dual_rail_fsim_gate_plan.md`

Use those documents for the physical model, required diagnostics, API design,
and overall milestone definitions. This document records the current numerical
baseline, the gaps found during review, and the order in which fidelity work
should proceed.

## 2. Current conclusion

### Initial gate-search scope (agreed 2026-10-22)

Use a transmon-only model for the initial approximate search of gate operating
points. Omit explicit readout resonators and their dynamical dressing at this
stage; their omission is not an integration blocker for the colleague's CZ
search. Use the colleague's reduced parameterization for this stage: four
transmons specified by EC and EJ, with the coupler-mediated interaction
approximated by an effective direct charge coupling J*n_A2*n_B1. An explicit
coupler transmon is restored for full-model validation when the device has one.

Isolated-pair logical references are acceptable for this initial search.
Defer resonator-inclusive dressing, readout compatibility, and preparation/
tuning-path validation to validation of shortlisted candidates. This staging
does not change the resonator-inclusive readout or coupler-off workflows.

### EC convention and transfer to the full model (agreed 2026-10-22)

For the approximate search, use the colleague's definition: EC_GHz is the
coefficient in the local Hamiltonian H/h = 4*EC_GHz*(n-ng)^2 - EJ_GHz*cos(phi).
Treat EC and J as effective reduced-model parameters, with omitted-circuit
loading already absorbed into EC. Do not add a second loading correction to
that Hamiltonian coefficient during the toy search.

If using the colleague's capacitor helper to parameterize J, its precise
convention is Cnom_i = e^2/(2*h*ECtoy_i*1e9) in farads, followed by inversion
of [[Cnom_1, -Cm], [-Cm, Cnom_2]]. This is a reduced-model prescription for J;
its nominal capacitances are not physical full-model shunts. Inverting that
matrix also implies diagonal charging energies different from ECtoy. Do not
silently substitute those implied energies for the actual toy Hamiltonian's
EC inputs or apply them as another loading correction.

On transfer to the full model, target the actual EC coefficients used by the
successful toy Hamiltonian:

    e^2 * (Cfull^-1)_ii / (2*h*1e9) = ECtoy_i_GHz.

Choose physical shunts within the full capacitance budget to meet these loaded
EC targets after all physical links and resonator capacitors are included.
Do not set the shunt-only charging energies to ECtoy and then add capacitors.
For fixed existing hardware, use its actual loaded ECs instead and refine the
candidate; exact matching is not guaranteed to be physically feasible.

Loaded EC is an electrostatic Hamiltonian parameter, not a dressed transition
frequency. Match/refine relevant dressed transitions and the effective
interaction separately by tuning EJs and the coupler operating point. The toy
effective inter-pair J includes the approximated mediated interaction; it is
not automatically the full circuit's direct-capacitor coefficient. Restore the
physical coupler and calibrate the combined effective interaction without
double counting its mediation as an additional direct term.

Add our charge-transition-balance H-score to the colleague's search diagnostics
during integration. Near-zero bare rail detuning remains a useful initial-search
proxy, not an exact H=1 guarantee. These search conventions do not change the
full inverse-capacitance definitions used by readout and coupler-off workflows.

The reusable simulation core contains much of the minimum viable machinery:
fixed-basis dynamics, four-state propagation, logical projection, leakage
metrics, polar-unitary separation, local-Z optimization, static logical
Hamiltonian extraction, and NetCDF persistence. The workflow is nevertheless
incomplete relative to the implementation plan and is not yet a trustworthy
foundation for large pulse optimization.

The current best saved results in
`data/logical_fsim_gate_comparison.nc` are:

| gate | fidelity including leakage | coherent fidelity | average leakage | maximum final leakage |
|---|---:|---:|---:|---:|
| iSWAP | 0.822858 | 0.851927 | 0.033940 | 0.068203 |
| CZ | 0.797187 | 0.824649 | 0.031556 | 0.088208 |

Both fail the workflow thresholds of fidelity at least 0.99 and average leakage
below 0.01. These are not merely pulse-duration misses: the selected static
Hamiltonians contain large non-fSim terms, the reduced pair model is only fitted
at one anchor, and convergence/state-tracking gates remain unfinished.

## 3. Evidence and diagnosed blockers

### 3.1 Reduced-model calibration is not valid across the control range

`LogicalFSimDeviceConfig.calibrate()` performs a single-anchor fit at 125 MHz.
The implementation itself notes that it is not a fit over the entire logical
frequency range. A concrete symptom is the selected CZ point: its coordinate
labels imply 7.5 MHz pair detuning, while the reduced model reports 14.43 MHz.

The current maximal-hybridization contour ranges are:

| pair | selected contour range (MHz) |
|---|---:|
| A | 122.7401--131.6396 |
| B | 112.7099--126.4326 |
| shared resonant interval | 122.7401--126.4326 |

This is much narrower than the 90--150 MHz design target and limits the ability
to suppress unwanted interactions through detuning or cancellation.

### 3.2 Static ranking does not reject non-fSim Hamiltonians

The notebook candidate scores use exchange, ZZ, logical singular value, and
leakage gap, but do not penalize the other Pauli coefficients already stored in
the atlas.

At the selected iSWAP point `(125, 125, 0.415)`, important real Pauli
coefficients in MHz include:

```text
XI =  11.366
IX =  -3.427
XX =  -1.309
YY =   0.364
ZX =  -0.612
XZ =  -0.170
ZZ =  -0.049
```

The desired `XX` and `YY` terms are neither balanced nor dominant. Local-Z
corrections cannot remove `XI`, `IX`, `XZ`, or `ZX`.

At the selected CZ point `(124.5, 117, 0.415)`, the static model reports about
3.58 MHz effective exchange and -11.93 MHz effective ZZ, with substantial
`XI`, `ZX`, `XX`, `XZ`, and `YY` terms. Maximum dressed logical-state coupler
participation is about 0.346. This is not a clean dispersive CZ mechanism.

### 3.3 Static logical-state tracking is incomplete

`assign_dressed_logical_states()` performs a one-point Hungarian assignment.
Every atlas point is built independently. The workflow does not yet store both:

- diabatic labels from fixed idle logical seeds; and
- adiabatic branches tracked by neighbor-to-neighbor overlap and phase alignment.

Consequently, static metrics near avoided crossings can jump branches without a
continuity failure being reported.

### 3.4 Fixed-basis and truncation validation are incomplete

The notebook validation compares eigenvalues by energy-order index rather than
overlap-matched states and does not validate logical overlaps or avoided-crossing
gaps along the full trajectory. The d=4 calculation is commented out, while the
following display cell still consumes `d4_iswap_validation`. There is no d=5
check and no executable acceptance gate before dynamics.

### 3.5 The dynamical search is only a small characterization sweep

Although the notebook ranks ten static candidates per gate, it manually chooses
one iSWAP and one CZ point. It then evaluates only six smooth iSWAP timing pairs
and three smooth CZ timing pairs. It does not optimize the required five baseline
parameters `(u_A, u_B, Phi_C, rise_time, hold_time)`.

The saved CZ trajectory follows `|11>` even though the worst final leakage for
the chosen pulse is `|10>` at about 0.0882. The `|11>` trajectory reaches about
0.233 transient leakage. All four input trajectories and transient maxima are
required before accepting a candidate.

## 4. Ordered optimization program

Do not skip directly to a large numerical pulse optimizer. Complete each phase's
exit gate before advancing.

### Phase 0 -- Repair reproducibility and execution gates

Tasks:

1. Restore a real d=4 validation result or remove the stale dependency until it
   exists.
2. Connect `RECOMPUTE_STATIC` to the static-atlas load call instead of using
   unconditional `force=True`.
3. Move physical device parameters and search settings from notebook constants
   into a versioned configuration consumed by every cache fingerprint.
4. Persist the complete `DualRailPulseResult`, including per-input leakage,
   parasitic amplitude, coupler excitation, pair-invalid probabilities,
   virtual-Z angles, solver status, and transient maxima.
5. Add a headless smoke test that runs the notebook DAG from an empty temporary
   cache with a tiny deterministic configuration.

Exit gate:

- Clean notebook execution from an empty cache.
- Identical inputs load fingerprint-compatible artifacts.
- Changed model, contour, grid, pulse, or solver inputs invalidate the cache.
- Existing automated tests continue to pass.

### Phase 1 -- Establish model and state-label credibility

Implementation note (2026-09-15): reusable Phase 1 reports now live in
`core/device.py` and `core/dualrail_fsim.py`. The workflow saves
single-transmon/ncut sweeps, full-contour reduced-fit residuals, fixed-seed and
overlap-continuous branch labels, overlap-matched fixed-basis path checks, and
flat-top static/dynamic rate checks. A dedicated direct five-transmon circuit
factory is selected when either reduced-pair fit misses its configured residual
tolerance. Numerical pass/fail values remain device-configuration dependent;
implementation status does not imply that the present physical parameters pass
every Phase 1 exit gate.

Tasks:

1. Add single-transmon sweeps for `f01`, `f12`, anharmonicity, `n01`, and `n12`.
2. Run charge-basis convergence at least over `ncut = 15, 20, 25, 30`.
3. Replace the single-anchor reduced-pair calibration with a multi-point fit over
   every pulse-relevant contour segment. Match carrier frequency, logical
   splitting, relevant two-excitation levels, and charge matrix elements.
4. If the reduced model cannot match those quantities within tolerance, build
   and use the direct five-transmon reference circuit specified by
   `docs/specs/dual_rail_fsim_gate_plan.md` for static truth and selected pulse
   validation.
5. Implement fixed-seed diabatic assignment and overlap-continuous adiabatic
   branch tracking. Store both labels, logical weights, phases, and dominant
   leakage-branch compositions.
6. Validate the fixed pulse basis at idle, both endpoints, intermediate ramp
   points, and the closest leakage approach. Compare overlap-matched logical
   energies and avoided-crossing gaps, not only energy-order indices.
7. Compare flat-top dynamic exchange/phase rates with static `J_ex` and `zeta`.

Exit gate:

- Increasing `ncut` changes relevant transitions by less than 10--50 kHz.
- Reduced/reference model residuals are below 10% of the smallest relevant
  interaction or leakage-gap scale; record stricter absolute tolerances where
  achievable.
- No unexplained branch discontinuity exists along either gate path.
- Fixed-basis errors are below 10% of the smallest relevant coupling, phase
  rate, or avoided-crossing gap.
- Static and flat-top dynamic rates agree within 5%.

### Phase 2 -- Build a Pauli-aware static gate atlas

Implementation note (2026-09-15): the reusable atlas now evaluates bounded
local-Z-corrected iSWAP and CZ coherent ceilings from the complete numerical
logical Hamiltonian at every point. It stores the desired exchange,
pair-creation imbalance, explicit iSWAP nuisance norm, complete CZ
off-diagonal norm and ZZ ratio, maximum coupler participation, and a named
nearest leakage eigenbranch with dominant bare-state composition. Candidate
selection uses hard qualification and coherent-ceiling gates followed by
non-dominated sorting; it persists every component rather than an aggregate
score. Half-grid rescans around coarse Pareto seeds provide adaptive refinement,
and a separately calibrated outer-architecture API supports pair coupling,
coupler EJ/EC, rail-coupler coupling, and intentional-asymmetry studies while
keeping parasitic paths disabled by default. As in Phase 1, implementation
status does not imply that the present physical parameters pass the numerical
exit gate.

For every atlas point, compute an inexpensive static coherent ceiling by
evolving the full numerical logical Hamiltonian over a bounded time interval and
maximizing local-Z-corrected target fidelity. Use the existing heuristic score
only for initial triage.

For iSWAP, rank or Pareto-filter on:

- large desired exchange component `(XX + YY) / 2`;
- small imbalance/pair-creation component `(XX - YY) / 2`;
- small `XI`, `IX`, `XZ`, `ZX`, `XY`, and `YX`;
- small logical detuning and manageable ZZ;
- large logical singular value and leakage gap;
- low coupler participation; and
- high static coherent ceiling to iSWAP.

For dispersive CZ, rank or Pareto-filter on:

- large usable ZZ;
- exchange and all other off-diagonal Pauli terms small relative to ZZ;
- a resolved, identified leakage branch rather than only a scalar nearest gap;
- low coupler participation; and
- high static coherent ceiling to CZ.

Search coupler flux and logical pair controls adaptively around Pareto regions.
If the current contour is too restrictive, introduce an outer architecture loop
over internal pair coupling, coupler `EJ/EC`, `g_Ac`, `g_Bc`, and intentional
coupling asymmetry. Keep parasitic couplings disabled until the baseline
mechanism is understood.

Current-atlas smoke-test candidates, not validated recommendations:

| gate | A MHz | B MHz | coupler flux | reason to test |
|---|---:|---:|---:|---|
| iSWAP | 123.0 | 125.5 | 0.410 | top current heuristic score; higher sigma and lower ZZ than the selected point |
| dispersive CZ | 128.0 | 126.0 | 0.405 | about 0.124 MHz effective exchange versus -3.67 MHz effective ZZ |

Exit gate:

- Retain 10--30 Pareto candidates per gate.
- Reject any candidate whose static coherent ceiling is below 0.995.
- Every retained candidate has a named gate mechanism and identified limiting
  leakage branch.

### Phase 3 -- Optimize smooth pulses progressively

For each retained candidate:

1. Optimize hold time only using the static rate as the initial estimate.
2. Sweep a small set of cosine rise times and retain candidates with improving
   coherent and leakage behavior.
3. Optimize the five baseline variables
   `(u_A_gate, u_B_gate, Phi_C_gate, rise_time, hold_time)` with bounds confined
   to the validated atlas and contour domains.
4. Use a coarse space-filling or differential-evolution search, followed by a
   bounded local method. Cache every evaluation and preserve the Pareto front.
5. If one shared envelope is insufficient, add independent rail/coupler ramp
   times and relative delays.
6. Only then consider a two-segment, net-zero, or echoed pulse. For avoided-
   crossing CZ, explicitly optimize the return of transient leakage.

The primary cost must include separate terms for:

```text
local-Z-corrected target infidelity
average final leakage
maximum final leakage across inputs
maximum transient leakage across inputs
parasitic logical amplitude
target-angle error
weak gate-duration penalty
```

Do not fold the stored results into one score. Preserve every component and the
optimizer termination status.

Progressive acceptance levels:

1. Mechanism validation: coherent fidelity at least 0.95 and average leakage
   below 0.05.
2. Working gate: fidelity including leakage at least 0.99, average leakage below
   0.01, and maximum final leakage below 0.02.
3. High-fidelity candidate: pursue 0.999 only after numerical convergence and
   robustness checks pass.

### Phase 4 -- Convergence, robustness, and noise

Tasks:

1. Re-evaluate final candidates at d=4 and selectively at d=5. Increase the
   coupler and boundary-rail dimensions first.
2. Compare `J_ex`, `zeta`, leakage gap, gate time, fidelity, final leakage, and
   transient leakage between truncations.
3. Sweep ODE tolerances, output/interpolation resolution, and any piecewise-
   constant time step.
4. Save full trajectories for all four logical inputs and report the worst input.
5. Produce local two-dimensional robustness maps for control amplitudes and
   duration/ramp errors. Compute first- and second-order sensitivities.
6. Prefer a broad high-fidelity plateau over an isolated maximum.
7. Only after unitary convergence, add relaxation, dephasing, and coupler decay
   with projected physical operators.

Final exit gate:

- Performance is stable against the next larger truncation and stricter solver
  settings.
- The candidate meets the configured fidelity/leakage targets throughout a
  declared control-error neighborhood.
- The dominant unitary and open-system error channels are identified.
- The result is reproducible from a saved configuration and fingerprinted data.

## 5. Recommended implementation boundaries

Keep the notebook as orchestration and reporting. Put reusable work into the
library:

```text
logical-resonance gate/logical_resonance_gate/device.py
    multi-point reduced-model calibration and fit report

logical-resonance gate/logical_resonance_gate/dualrail_fsim.py
    branch tracking, static coherent ceilings, pulse objectives,
    optimization results, and complete persistence

logical-resonance gate/logical_fsim.py
    configuration selection, staged execution gates, plots, and decisions

tests/test_device.py
    multi-point fit and ncut/truncation convergence tests

tests/test_dualrail_fsim.py
    branch-continuity, static/dynamic consistency, objective, and cache tests
```

Do not bury optimization logic or acceptance decisions in anonymous notebook
cells; future sessions must be able to test them without running the expensive
workflow.

## 6. First next-session checklist

1. Run the production Phase 1 and Phase 2 exit gates with fresh fingerprinted
   artifacts.
2. Inspect the persisted iSWAP and CZ Pareto reports, including each limiting
   leakage branch and coherent-ceiling hold time.
3. If fewer than 10 candidates survive for either gate, use the outer
   architecture loop before beginning pulse optimization.
4. For each retained candidate, begin Phase 3 with hold-time-only optimization
   around the static coherent-ceiling time.
5. Decide from those results whether the next bottleneck is architecture range
   or pulse shape.

## 7. Verification already performed

At the time of this assessment:

- `python -m py_compile logical_fsim.py` succeeded in the project environment.
- `python -m unittest discover -s tests -q` ran 117 tests successfully.
- Ruff passed for the package, workflow, and Phase 2 regression tests.
- The headless smoke workflow ran twice from an empty cache and reused the
  fingerprint-compatible Phase 2 atlas and candidate artifacts.

## 8. Suggested skills for future sessions

Before implementing the next phase, call the Skill tool for:

- `diagnosing-bugs` when resolving the static/dynamic rate mismatch, state-label
  discontinuities, or fixed-basis convergence failures.
- `tdd` when adding multi-point calibration, branch tracking, optimization
  objectives, and notebook regression coverage.
- `codebase-design` if the optimization and tracking interfaces need to be split
  into deeper reusable modules rather than extending `dualrail_fsim.py` further.
