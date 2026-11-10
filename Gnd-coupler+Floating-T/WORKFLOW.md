# Tunable-coupler design workflow

## Authority and temporary OFF policy

The neutral shared module `../core/cz_channel_policy.py` owns executable logical
definitions, S1/S2 and ON selection. The written specification remains in
`../2Q-gate_XM/WORKFLOW.md`; the active toy search is `../2Q-gate_XM/`.
Logical 0/1 are the lower/upper isolated-pair one-excitation modes; pair ground
is separate. Logical bit order is A,B. Physical rail order is A1,A2,B1,B2.
Physical auxiliary 0020 means B20; it never determines the logical source.

**Temporary OFF policy, confirmed 2026-10-28:** both five- and seven-mode
optimization null physical `0100 <-> 0010` exchange. Three modes null physical
A2/B1 `01 <-> 10`. Whether logical `01L <-> 10L` is a better isolation objective
remains OPEN; see [OFF_POLICY.md](OFF_POLICY.md) for implementation and reversion.
Logical exchange, CZ-auxiliary exchange and secular ZZ remain separate residuals.
A physical null does not certify a logical null or isolation of every channel.

## 1. Confirm request and constraints

Obtain immutable **loaded** EC tuples for all four rails; idle/gate bare
frequencies; isolated-pair logical splittings and their reference positions;
moved rail and tuning path; target gate speed; desired CZ auxiliary; coupler
and loading constraints; permitted tuning controls; and supplied resonator data.
Confirm defaults for each new case:

- Ground C2, active C1, floating rail transmons. Actual capacitors A22-C1,
  C1-B11, A22-B11 and both intra-rail links. No unspecified parasitics.
  The fourteen explicitly named parasitic links in PARASITICS.md default to zero;
  use actual port-to-port fF values when supplied. Keep them out of circuit
  diagrams unless specifically requested. Defining these inputs does not launch
  a search, retune EJs or silently compensate fixed physical hardware.
- Loaded coupler EC 150-600 MHz. ON >7 GHz, or approximately 1 GHz above a
  supplied relevant resonator frequency. ON < OFF <10 GHz.
- Loading <10% per coupler per rail; prefer <7%, minimize combined loading
  toward 5%. Report the separately reconstructed four-coupler aggregate.
- Coupler own self-capacitance contributes >=55% of the total effective
  capacitance budget. Define all percentages and removal counterfactuals.
- Select the feasible minimum-S1 ON resonance at common intended g and in a
  common score basis. Exclude its intended edge; default unintended |g/delta|
  <0.2; keep S2 and crossing-order diagnostics separate.

The current exploration explicitly permits EC down to **100 MHz**. On
2026-10-29 the user restored the hard **OFF <10 GHz** requirement; the earlier
12 GHz relaxation is superseded and must not seed a claimed accepted OFF bound.
The EC lower limit is a case override, not a replacement for general defaults.
Every geometry change must compensate pad-to-pad/ground capacitances to preserve
loaded rail ECs. Never change rail EJ to conceal EC drift. Idle and nominal gate
rail frequencies stay fixed during design refinement. Do not launch readout now.

## 2. Three-body A2-C-B1 design

Search physical capacitor ratios and the signed physical `01 <-> 10` OFF branch
before increasing coupling scale. Use roots, not extrapolation or same-sign
endpoints alone; inspect interiors and extrema. Verify overlaps and convergence.
Use the active **2Q-gate_XM** toy workflow for the dressed ON strength
and minimum-S1 channel. A reduced physical matrix element is only a strength seed,
not a logical channel definition or gate-fidelity certificate.

Refine capacitors to achieve ON strength, loading and frequency requirements
while retaining the verified OFF branch. Print a successful design in chat and
continue. If none is found, report that and stop; do not silently relax constraints.

## 3. Five-body refinement, no resonators

Restore A1/B2, preserve loaded ECs through self-capacitance compensation, and
fit/retain isolated-pair splitting targets at the supplied reference positions.
Start from the three-body solution. Refine coupler capacitors/geometry within
confirmed limits; keep rail idle and nominal gate positions fixed.

**Temporary OFF objective:** the `0100 <-> 0010` entry of the canonical effective
**four-physical-single-excitation** Hamiltonian (1000,0100,0010,0001), with C in
its ground reference. Project directly from full dressed eigenstates. Retain
within-rail hybridization and virtual paths; do not eliminate the other two
physical rail references or rotate this observable to logical energy axes.
For a frozen geometry, tune coupler EJ only; freeze all rail EJs and capacitors.

Report all physical capacitances and loading, OFF frequency/EJ, physical-root
residual and overlap, both logical exchanges `01L <-> 10L` and `00L <-> 11L`,
auxiliary exchanges with connected-state detunings and |g/delta|, secular logical
ZZ and basis convergence. Logical residuals use full-eigenstate canonical
four-logical-state projection followed by local single-rail energy-axis alignment.
Logical ZZ is `(E11-E10-E01+E00)/h` for tracked logical eigenbranches.

Plot both logical residual exchanges versus coupler frequency over a sufficiently
wide diagnostic interval to reveal nonmonotonic behavior. Mark the physical null,
allowed OFF interval and all excursions outside it as diagnostic only. An
out-of-range minimum is not an accepted OFF design. Report failure without
calling a finite minimum a zero.

## 4. Five-body ON gate calibration

Retain the toy S1-selected source/auxiliary. With the stationary rail fixed,
optimize coupler EJ, moved-pair gate frequency and duration; start within 200 MHz
of the preceding solution without inventing another hard tuning box. Hardware
EC/J stay fixed. Unlike the toy's numerical strength fitting, do not refit
physical capacitors during final gate calibration.

Enforce nonlocal conditional phase pi, remove permitted local phases, then
improve return. Use consistently coupler-dressed static ON logical references;
do not introduce a sudden idle/ON jump. Report duration, raw fidelity,
postselected score, mean/worst-input survival and worst nonlogical population
at the SAME operating point. Report actual-bias S1/S2 separately from common-g
toy selection; do not silently rerank or overwrite selection provenance.

**Refinement precision (2026-10-29):** local improvement stops when raw fidelity,
postselected score, survival and per-input leakage all vary by <=1e-4 absolute
probability over three distinct feasible refinement iterates. A plateau is only
an optimizer stop, not a basis-convergence or gate-quality certificate. Retain the
best phase-pi feasible endpoint, record the stop reason and evaluation count, and
use finite budgets (default 12 local iterations / 48 control evaluations).
Local scalar polishing uses 0.02 ns time and 1e-5 GHz coupler-frequency tolerances,
plus the same metric stop, rather than chasing micro-nanosecond or Hz-scale gate
controls. Do not round Hamiltonian inputs or weaken the conditional phase check.

Do NOT restart a frequency-plus-time optimizer in the larger verification basis.
Freeze the selected duration and coupler bias; perform only a bounded local
moved-rail frequency correction to phase pi (<=6 evaluations), then fixed-control
basis checks at the agreed 1e-4 precision. Failure to reach phase pi or basis
agreement must remain explicit; it does not justify unlimited fine optimization.
The same rule applies to seven-mode gate refinement.

Seven-mode ON search starts from the actual five-body solution's coupler bias,
moved-rail frequency AND duration; it must not reset to 100 ns and a broad
three-coupler-frequency grid. Try the exact bias first, transfer its phase locally
if needed, and only then expand initial B-frequency starts within the authorized
200 MHz neighborhood. Exhaustion is an explicit finite-search failure.

At each trial bias, construct one gate spectrum and reuse it for the complete
duration grid and time-root calculations. An offset frequency need not admit a
phase-pi time at all: use its best cached return as a proposal, then correct the
moved-rail frequency and report metrics from that corrected Hamiltonian. Do not
declare the cached proposal phase calibrated or freeze a stale spectrum after
changing frequency. Retain the established S1 channel and unintended-edge screen.
The outer local search now proposes frequency biases; duration is handled by
cached spectral evaluation. Default seven-mode refinement is limited to six
iterations / twelve bias proposals, with the existing metric plateau stop.
Cache actual-bias scores once per spectrum; reuse same-cell/basis gate evaluators
for fixed-control verification. Record spectrum-build counts, cache statistics
and elapsed time. No spectral-input rounding or physics approximation is used.

The user's requested future threshold stop refers to **postselected fidelity
>=99.9%**, not raw fidelity. Implementing that target/full-map search mode is
separate from this precision fix: preserve raw fidelity and survival reporting,
and generate a frequency-duration image whenever an explicit full search runs.

## 5. Seven-mode validation

First design each rail-pair plus shared resonator independently under the
readout workflow, excluding C and the opposite pair. The standard standalone
optimization uses the prescribed-resonator-range goal from
`../readout_workflow/RESONATOR_ROBUST_DESIGN.md` by default. **When an actual
design is started, prompt for all four bare resonator-frequency bounds in GHz:
A_low, A_high, B_low, B_high.** Leave these inputs unset when updating the
workflow itself. They are independent free parameters: different A/B centers
are expected, and equal interval widths may be convenient but are not enforced.
Do not infer any bound from a nominal frequency, a previous design or an example
range. A common width may be expanded into the four bounds only when explicitly
supplied by the user. Record both intervals before optimization starts.
The objective is **maximum worst-case absolute signed-mean chi**,
`min_over_range(abs((chi_plus + chi_minus)/2))`, at **H > 0.96 throughout the
range**. A nominal-frequency chi maximum alone does not complete this step.
Use one fixed capacitor design across all resonator scenarios; EJs can
recalibrate within the supplied allowed bias bounds. At each scenario change
only the resonator C/L consistently with its impedance and the EJs; do not
re-fit transmon self capacitors to hide changes in loading. Specified loaded
ECs refer to the nominal reference design, while scenario EC changes and all
scenario constraints must be reported. Confirm the nominal reference and
range with the user before numerical optimization. The independent readout
runner's present 6 GHz reference assumption must not silently replace an
explicit-pad case's supplied resonator reference. Use the exact equivalent
circuit and preserve physical-versus-effective capacitor conventions.

`standalone_pad_readout.py` now runs nominal feasibility and then the explicit-pad
prescribed-range adapter in `floating_coupler/readout_range.py`. It uses the
readout workflow's shared acceptance, physical scenario reconstruction, frequency
grid and quantum audit; no effective-capacitor substitution is made. Nominal
feasibility alone is an initialization, not a range-optimized result.
The standalone CLI accepts the four bounds under
`standalone_readout_optimization.bare_resonator_frequency_range_ghz` in the input
request, or each `pairs.A/B.bare_resonator_frequency_range_ghz`. Both rails must
be supplied before either design starts. The Python per-rail `design` interface
requires that rail's range. Conflicting input locations are rejected.
The alternative exploration
goal (maximize resonator tolerance with chi/H as constraints only) is available
only if explicitly selected, not the coupler-design default.
The handoff now rejects new nominal-frequency-only optimization records,
including earlier records identified by their nominal-chi objective. Existing
historical designs without an optimization record remain readable and labelled
legacy; they are not certified by the prescribed-range policy.

Return the optimized **nominal physical pad model**, nominal coordinates and
audited nominal point in the existing readout-result format, plus a
`readout_strength_optimization.range_validation` record. This contains the
requested interval, fitted physical LC reference, sampled per-frequency EJ/EC
witnesses, worst chi/H, selected H floor, fallback flag and numerical basis.
Seven-mode assembly receives this nominal model, not a scenario model or
effective-capacitor design. Default optimization spacing is 0.05 GHz; validate
at half steps, adjacent optimized-frequency midpoints and the nominal reference.
Failed interior samples trigger EJ-only repair; unresolved failures become
explicit optimization scenarios before the H fallback is attempted. Defaults
are 60 optimizer iterations/start, at most four starts, two candidate audits and
two refinement rounds per H tier. Per-case `readout_range_options` can set
`frequency_step_ghz`, `maxiter`, `max_audits`, and `refinement_rounds`.
Scenario bare-frequency coordinates default to the readout operation search
box with 0.2 GHz numerical padding; actual operation/EJ constraints still apply.
Supply `readout_recalibration_bare_bounds_ghz` to restrict either transmon's
allowed EJ-recalibration frequency range explicitly.

Keep supplied loaded ECs, isolated-pair splitting, idle/readout calibration
targets, physical parameter bounds and every existing readout constraint. Added
loading is compensated by the existing self-capacitance reconstruction. The
minimum-readout-link fit is only an initialization, not the final objective.

If the bounded search finds no audited design covering the requested interval
at H > 0.96, rerun worst-case chi maximization over the **same interval** at
the standard **H > 0.95**. Do not silently shrink the interval. Print a warning and retain
`fallback_used`, the selected H floor, per-tier attempts/audits and the warning
in the saved readout result and integrated-cell metadata. Report the fallback
explicitly in chat and the design report. This is a best-found local maximum,
not a global optimum or proof that H > 0.96 is impossible. If neither tier
passes, retain the diagnostic result as rejected and stop integration; do not
silently accept the original seed. Audit selected candidates, including chi
sign stability, and test the strict H floor at the audited basis. The 96% goal
is standalone design margin; the seven-mode physical H acceptance threshold
remains 95%, and the integrated result must still be checked separately.
No default readout-map, robustness-rescue or global readout threshold changes.

Integrate both resonators
and C, compensating self-capacitances to retain loaded rail ECs.

**Standard standalone readout contrast maps (2026-11-03):** after fitting each
independent readout triple, freeze its capacitors and resonator components and
scan the two rail EJs independently over the supplied transmon tuning ranges.
From the repository root call `python -m readout_workflow.readout_shift_map`;
record a 61x61 EJ grid and display the corresponding
full-cosine bare f01 axes, with nominal idle stars. Plot the three absolute
cavity-frequency contrasts 01-10, 01-00 and 10-00 with 0.5/1/2 MHz contours,
orange shading when ALL three exceed 0.5 MHz and pink when ALL exceed 1 MHz.
Show the <10 kHz logical matching band and flag reference overlap below95%.
Here 00 is pair ground, 10/01 are lower/upper one-excitation branches, not
physical product occupations. Keep all values; these contrast thresholds are
diagnostics, not new readout acceptance constraints or measurement fidelities.
Include the maps, fixed hardware, ranges, conventions and basis checks in the
coupler-design PDF. This is a pair+resonator THREE-mode characterization, distinct
from A2-C-B1 design. Seven-mode versions of these 2D maps are optional and must
not run automatically. Existing integrated idle checks and +/-100 MHz constraint
sweeps below remain part of validation. See ../readout_workflow/READOUT_MAPS.md for commands.

**First retune seven-mode OFF**, varying only coupler EJ. Apply the SAME temporary
physical `0100 <-> 0010` target as five modes, with C ground and both resonators
vacuum reference states. Include their virtual paths and report logical residuals.

**Mandatory dressed-idle calibration (2026-10-29):** adding the coupler generally
shifts the inner and outer transmons differently even at exchange OFF. Retune
the individual rail EJs, with all loaded ECs, capacitors and resonator components
fixed, to restore near-resonant dressed rails and satisfy the existing readout
H-score, chi-matching and other constraints. Equal bare frequencies are not a
certificate of dressed resonance. Distinguish bare EJ-derived settings, effective
diagonal frequencies and actual logical eigenfrequencies in reports. Use the
supplied idle positions as operating-frequency guides; report any displacement.
Do not require exact equality of convention-dependent effective diagonal terms
or invent another hybridization threshold beyond the readout policy.

After both rail pairs are calibrated, re-null the physical OFF channel by varying
only coupler EJ. Recheck both readouts and repeat this idle/OFF calibration if
needed; never redesign capacitors during this calibration. Retain one joint
four-rail idle setting and one common OFF bias for final acceptance. Keep original
uncompensated controls as a diagnostic, not as the final integrated readout result.

Then plot each readout constraint (H-score/hybridization, chi ratio, worst
dispersive ratio, logical frequency, EJ/EC and applicable limits) across the
case-specific idle center +/-100 MHz, with standalone pair+resonator and integrated
seven-mode traces in each panel, using each case's own calibrated idle center.
Keep the differential frequency setting fixed during each common-shift sweep;
do not recalibrate at every point. Freeze geometry and other controls; do not
invent frequency acceptance bounds or refit links at each point.

Run the A/B readout sweeps concurrently by default in two spawned processes,
one rail sweep per process, with one BLAS thread in each. Use
`readout_common_sweep.py --cell ... --readout-A ... --readout-B ... --output ...`
or `final_seven_validation.py --stage readout` with those explicit inputs.
The integrated sweep starts at local6/pair12/C3, with resonators |0>,|1>,|2>.
Audit local6/pair12/C5 at centers, endpoints and near constraint boundaries.
These guards do not change acceptance thresholds or add sign certification. Save separate
atomic progress/results per rail, source/input-checked checkpoints, per-job times
and parent elapsed time. Summed worker durations are not elapsed wall time. Do
not overlap these two memory-heavy jobs with another seven-mode gate optimizer.

Finally recalibrate ON with coupler EJ, moved-pair frequency and duration,
conditional phase pi; report raw/postselected fidelity, survival and convergence.

## Execution and records

This is a staged workflow, not a fully automatic end-to-end pipeline. The master
request template is a schema, not an executable stage config. Source hashes and
OFF version distinguish runs; never relabel superseded logical/CZ/ZZ nulls.
Generated outputs/caches remain ignored under output/. No upload ZIPs. Preserve
unrelated readout/toy edits. Run run_tests.py after numerical source changes.
All eight acceleration proposals are approved as of 2026-10-29; see SPEED_PROPOSALS.md.
Residual numerical accuracy is 1 Hz absolute; gate-error accuracy is 1e-4
absolute probability (0.01 percentage point). These are convergence targets, not
fidelity acceptance limits. Compare fixed controls across modest bases and stop
on first agreement; do not force exhaustive large-level/photon audits. If the
configured budget is exhausted, report unconverged. Preserve phase-pi, S1 and
readout/loading constraints; escalate near physical acceptance boundaries when
needed. Do not silently change a threshold or relabel a finite minimum as zero.

Readout H-score escalation guard: 0.0003 absolute around H=0.95, covering
the observed 0.000206 C3-to-C5 shift with margin. The physical acceptance
threshold stays H>=0.95; this changes only when the basis is increased.

## Report format

All subsequent full design reports must follow REPORT_TEMPLATE.md and the
preserved detailed PDF in templates/report-references/. Keep its form,
section sequence and level of detail; the short benchmark report is supplementary.
Use current results and explicit validation status, not the reference case
numbers. Render and inspect every page. No optimization runs during reporting.

## Optional post-design studies

Parasitic compensation uses the minimum-movement policy in PARASITICS.md.
This is an optional post-design study, enabled only on explicit request.
Its staged commands and retained results are in OPTIONAL_STUDIES.md. Normal
design never launches its capacitor search or quantum validation automatically.
For comparable matrix matching, prefer the lower physical-capacitor changes.
The default engineering target is 1 ppt relative error for each intended
nonzero coupling and the matched-matrix norm, with exact diagonal constraints.
Minimize the maximum fractional capacitor movement first, then combined movement.
This target is configurable and independent of EM/fabrication uncertainty.
Preserve the group-specific matching mask and rerun quantum validation for any
newly selected geometry. Never inherit fidelity or OFF certification from another
capacitor allocation, or promote a nonconverged geometry as a required correction.

An explicitly requested five-transmon pulse branch is defined separately in
[PULSED_CZ_WORKFLOW.md](PULSED_CZ_WORKFLOW.md). It propagates idle logical states
through simultaneous EJ-tanh coupler/B controls and symmetric return, calibrates
phase pi, and reports idle-space survival/fidelity. It never replaces the static
ON benchmark above. Use pulse_cz.py; no resonators or noise are added implicitly.

Its optional finite-bandwidth comparison uses pulse_shape_comparison.py to compare
tanh and half-cosine EJ edges at matched 10-90% rise times, with an explicitly
specified line response. Freeze the saved five-body hardware, stationary A EJs,
coupler ON endpoint and idle logical input/output basis; calibrate moved-B EJs and
hold duration to phase pi. The retained case assumes a 1 GHz ideal zero-phase
cutoff applied to EJ, not a nonlinear flux waveform. Include the complete pulse,
filter ringing and explicit tail-window checks. When requested, include its
definitions, scope, results, figure, numerical status and timings as an optional
section of the design report. It is OFF by default and never runs during report
assembly. See PULSED_CZ_WORKFLOW.md and OPTIONAL_STUDIES.md for commands. Proposed
additional speedups must be distinguished from already implemented limits/caches;
do not silently change bases, filter physics or precision to speed up a new case.

Robustness and decoherence are separate, opt-in branches after a saved nominal
solution. Neither runs by default or on import. Use study.py with explicit inputs
and output folders; see OPTIONAL_STUDIES.md. Current approved report references
are in templates/report-references/ and governed by STUDY_REPORT_TEMPLATES.md.
Normal three -> five -> ON -> seven design/validation is unchanged.
