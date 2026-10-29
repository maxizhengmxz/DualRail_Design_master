# Tunable-coupler design workflow

## Authority and temporary OFF policy

The toy workflow (`../transmon-based CZ/WORKFLOW.md` and its
`transmon_cz/channel_policy.py`) owns logical states, S1/S2 and ON selection.
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
Use the established **transmon-based CZ** toy workflow for the dressed ON strength
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
readout workflow, excluding C and the opposite pair. Integrate both resonators
and C, compensating self-capacitances to retain loaded rail ECs.

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
preserved detailed 13-page PDF in output/report-reference/. Keep its form,
section sequence and level of detail; the short benchmark report is supplementary.
Use current results and explicit validation status, not the reference case
numbers. Render and inspect every page. No optimization runs during reporting.
