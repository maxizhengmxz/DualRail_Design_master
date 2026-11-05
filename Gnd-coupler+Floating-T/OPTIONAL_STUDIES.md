# Optional studies

The normal design workflow ends with nominal seven-mode validation. None of the
studies below runs automatically. Each requires an explicit nominal input and an
output folder. Importing the modules performs no calculation.

## Finite-rise pulse gate and control bandwidth

The optional five-transmon branch in PULSED_CZ_WORKFLOW.md starts/ends in the same
dressed idle logical space. It retains nominal hardware and the S1-selected CZ
channel; A is fixed, C and B move simultaneously and return symmetrically. The
shape comparison uses tanh/half-cosine EJ controls, supplied 10-90% widths and an
explicit line response. The saved case uses widths10/5/2/1/0.5ns and an ideal
zero-phase1GHz sharp cutoff on EJ. Do not infer a nonlinear flux-line result.

Run one shape per process, BLAS=1, with explicit input/output paths:

```powershell
.venv/Scripts/python.exe Gnd-coupler+Floating-T/pulse_shape_comparison.py --input <nominal-input.json> --output <new-folder> --shape tanh
.venv/Scripts/python.exe Gnd-coupler+Floating-T/pulse_shape_comparison.py --input <nominal-input.json> --output <new-folder> --shape cosine
```

Calibrate B endpoint and hold at phase pi, optimizing raw fidelity as in the
original pulse study. Report raw/postselected scores and survival at that SAME
point; this does not change the robustness branch's postselected objective.
Include rise+hold+fall and explicit pre/post-ringing propagation. Show programmed
duration separately from numerical tail windows; an ideal brick wall has no exact
finite settling time. Report basis/time-step/tail checks without substituting
regression-test success for a per-case numerical certificate. No resonators,
T1 or flux-noise channels are added implicitly. This branch does not replace
static ON benchmarks or certify OFF isolation.

`plot_pulse_shape_comparison.py` plots saved results. `pulse_comparison_report.py`
inserts the OPTIONAL section into an existing design PDF using saved data only.
Include it only when requested; preserve all earlier validation status. The
retained PDF also describes measured costs, implemented search limits and further
speed proposals. Extra approximations/basis changes require separate agreement.

## Parasitic compensation and validation

Run this branch only when requested, after a saved nominal seven-mode design.
It performs no readout optimization, pulse propagation or decoherence calculation.
Defining the fourteen zero-default parasitic inputs does not enable the study.

The standard batch adds each of eight group-1 and four group-2 parasitics
independently at 0.1 and 0.2 fF. The two remaining registered links A21-C1 and
B12-C1 are outside these study groups. See PARASITICS.md for the port list.
The report template expects this standard 24-case batch.

### Compensation and controls

Use K = P^T C_node^-1 P with floating common charges fixed to zero and C2
grounded. Preserve all seven K diagonals, hence loaded rail/coupler ECs and
bare resonator frequencies. Keep equal grounds on each floating rail and
positive physical capacitors. The added parasitic and resonator inductances
stay fixed. Twenty intentional-capacitor variables may compensate the geometry.

The default engineering matching tolerance is 0.001, or 0.1%. Require it for
each of nine intended nonzero coupling coefficients and the matched Frobenius
norm. Group 1 matches all matrix entries. Group 2 matches seven diagonals and
nine intended pairs, retaining twelve other offdiagonals as residuals. First
minimize maximum fractional physical-capacitor movement, then its squared sum
at the optimal ceiling. The +/-95% bound is a numerical guard. Local optimizer
success and explicit matching checks are required. EM/fabrication uncertainty
is a separate quantity. This compensated-design study differs from robustness,
which perturbs fixed hardware without restoring loaded ECs.

Quantum validation freezes each fitted geometry and all idle rail EJs. Retune
coupler EJ alone to null physical 0100-0010, with all four physical references,
between the saved ON and 10 GHz. Keep every failed case and its best evaluated
allowed bias. Logical 01L-10L, 00L-11L and secular ZZ remain diagnostics. Static
ON uses the saved coupler bias and duration, correcting moved-B frequency to
conditional phase pi. Report raw/selected fidelity at the same controls,
logical survival, worst-input nonlogical population and B-frequency correction.
This stage does not repeat the full gate-duration/coupler optimization.

Measure idle physical components directly from normalized dressed eigenstates.
For a B logical state, show its absolute A1/A2 weights in 1000/0100 with C ground
and resonators vacuum in the references. Do not renormalize four reference
weights or replace them with a perturbative coupling/detuning estimate.

### Explicit stages

Commands below are run from the repository root. Substitute the supplied
nominal paths for a new design and use a new output folder after source changes.
The named nominal design is a complete parameters-only input. The prior
parasitic calculations were deleted on request. Its report is retained as a
visual template. Commands below calculate a fresh study when explicitly requested.

```powershell
.venv/Scripts/python.exe Gnd-coupler+Floating-T/studies/parasitic_study.py fit --input Gnd-coupler+Floating-T/configs/Design_V1_20261105.json --output Gnd-coupler+Floating-T/output/new-parasitics/candidates --amounts .1 .2 --match-rtol .001
.venv/Scripts/python.exe Gnd-coupler+Floating-T/studies/parasitic_study.py reference --input Gnd-coupler+Floating-T/configs/Design_V1_20261105.json --gate Gnd-coupler+Floating-T/configs/Design_V1_20261105.json --output Gnd-coupler+Floating-T/output/new-parasitics/nominal-reference.json
.venv/Scripts/python.exe Gnd-coupler+Floating-T/studies/minimum_change_validation.py --candidates Gnd-coupler+Floating-T/output/new-parasitics/candidates/study.json --gate Gnd-coupler+Floating-T/configs/Design_V1_20261105.json --reference Gnd-coupler+Floating-T/output/new-parasitics/nominal-reference.json --output Gnd-coupler+Floating-T/output/new-parasitics/validation --workers 2
.venv/Scripts/python.exe Gnd-coupler+Floating-T/studies/minimum_change_postcheck.py --root Gnd-coupler+Floating-T/output/new-parasitics/validation --workers 2
```

The reference stage computes this nominal circuit's audit-basis OFF, slope,
static gate baseline and physical participation once. Validation checks the
reference charge matrix against its supplied nominal circuit. Two bounded
workers and one BLAS thread per worker are the default. Checkpoints are
source/input hashed. A numerical source change requires fresh output.

After a new study completes, generate its report from saved data only. Supply
that study's circuit drawing with the requested green/red parasitic overlay.
The usual circuit omits parasitics unless requested. The drawing below is an
explicit input that must be generated for the new study.

```powershell
.venv/Scripts/python.exe Gnd-coupler+Floating-T/studies/validated_minimum_change_report.py --campaign Gnd-coupler+Floating-T/output/new-parasitics/validation --candidates Gnd-coupler+Floating-T/output/new-parasitics/candidates --circuit <new-study-circuit.pdf> --output Gnd-coupler+Floating-T/output/new-parasitics/parasitic-compensation-report.pdf --temporary tmp/pdfs/parasitics
```

New result reports embed nominal inputs, complete fits, quantum records and
solver source. The preserved reference PDF has those attachments removed and
serves as a visual template. Deleted calculations cannot be resumed or rebuilt
from that template. New campaigns record their own input/source identities.

### Numerical qualifications and retained files

Production uses local6/pair12/C5/ncut35 and audit uses local7/pair15/C6/ncut40.
Both resonators retain states 0,1,2. Compare controls identically across bases.
Physical OFF passes below 1 Hz within the stated basis. The completed 24-case
study has up to 32.52 Hz physical and 10.74 Hz logical cross-basis differences,
so absolute 1-Hz certification remains unresolved. Secular-ZZ differences are
at most 0.37 Hz. Raw/selected gate and survival differences meet the agreed
1e-4 absolute-probability target. Retain these limitations with future reports.

The user requested deletion of all saved parasitic results and archives on
2026-11-05. Keep configs/Design_V1_20261105.json as the complete nominal parameter
record and templates/report-references/parasitic-compensation-reference.pdf as
the report template. The nominal design, decoherence, robustness and pulse
studies outside this parasitic branch retain their existing preservation policy.
The completed 24-case validation cost about 66 minutes with two workers.

## Default policy and scope

The current approved rescue ladder is the DEFAULT for an explicitly launched
robustness study. It is NOT the default acceptance policy for nominal design or
standalone readout optimization. A sample starts at baseline; cumulative rescue
tiers are tried only after failure, stopping at the first feasible tier. Always
report the tier used, rejection flags, and whether a gate is accepted or merely
the best available diagnostic. Do not label a rescued result a baseline pass.

| Quantity | Robustness baseline | Last cumulative rescue tier |
|---|---|---|
| Loaded readout EC floor | 150 MHz | 120 MHz |
| Idle frequency | nominal target | may retune |
| Dressed splitting window | supplied case window (current A140–160/B80–100 MHz) | widened by 10 MHz at each end |
| H-score | >=0.95 | >=0.90 |
| EJ/EC | >45 | >40 |
| Chi mismatch | <=1% | <=5%, final tier only |
| OFF upper bound | <10 GHz | <11 GHz, then basis escalation |

Mean |chi|>=1 MHz, EJ/h<25 GHz, dispersivity, overlap and sign checks remain
hard. The current robustness gate duration interval is 70–130 ns; maximize
postselected fidelity with phase pi and tracking validity, with no fidelity or
survival threshold. These study policies do not replace supplied design inputs.

CLI defaults: independent sampling, +/-5% capacitor errors, 200 samples, two
workers, 48-hour budget, seed 20261103. Maximum-deviation bin sampling must be
selected explicitly; its defaults are 0.25% bins and 200 finite-gate records per
bin. A CLI budget/count is not a guarantee of completed samples.

## Known nominal seven-mode OFF accuracy limitation

The retained nominal OFF is a numerical cancellation in the selected basis,
NOT a certified 1 Hz physical null. In the saved fixed-control audit
(`output/nominal-design/final/off-convergence.json`), physical exchange changes
from -275.981441 Hz to approximately zero on enlarging the basis (difference
275.981442 Hz); logical exchange and double-flip change by about 154 Hz, and
secular ZZ by 2.487 Hz. The audit explicitly records `converged: false`.
These values describe that saved audit, not universal model errors.

Report the basis and this unresolved convergence status alongside nominal OFF
residuals. A sub-1 Hz residual in one basis is acceptance at that basis, not
cross-basis certification. Never claim certified hardware isolation, reinterpret
the logical residuals as zero, or remove this warning because a root was found.
New designs must retain their own audit status; this historical limitation is
not a reason to launch an unrequested calculation.

## Robustness

```powershell
.venv/Scripts/python.exe Gnd-coupler+Floating-T/study.py robustness --input Gnd-coupler+Floating-T/output/nominal-design/fabrication-input.json --output Gnd-coupler+Floating-T/output/new-robustness --sampling independent --samples 200 --deviation-percent 5 --workers 2 --hours 12
```

For maximum-deviation strata, use `--sampling max-bins --bin-width-percent .25
--target-per-bin 200`. `--reuse <robustness-data.tar.xz>` retains previous records
and credits their finite gate results toward each bin. A dry run validates the
input without launching numerical work: add `--dry-run`.

Twenty-four actual positive capacitors vary; no fabrication-time EC compensation
is made. The same signed fractional error is shared across the independently
synthesized five-body/readout representations. Readout-derived individual bare
frequency shifts transfer relative to each subsystem's nominal calibration.
Only EJs and gate duration recover performance. Resonator L remains fixed.

Independent sampling draws every deviation uniformly in the requested interval.
Max-stratified sampling draws the observed maximum uniformly within a bin, chooses
a random signed limiting component, and draws the other 23 errors uniformly
inside that maximum. These distributions differ. Structured/targeted/stratified
records are excluded from independent-random correlation analyses.

Nominal readout uses A140-160/B80-100 MHz dressed windows, H>=.95, mismatch<=1%,
absolute mean chi>=1 MHz, worst dispersive ratio<=.1, overlap>=.95, EJ/EC>45,
EJ/h<25 GHz and the established sign grid. Recovery is cumulative:
EC floor120 MHz; free idle frequency; splitting windows widened10 MHz per side;
H>=.90; EJ/EC>40; mismatch<=5%. Mean chi>=1 MHz and EJ<25 GHz remain hard.
Rejected readouts continue from their best diagnostic point with rejection flags.

OFF searches below10 GHz, then below11 GHz in the same basis, then larger bases.
Physical |g0100,0010|<1 Hz is acceptance; logical exchanges and ZZ are residuals.
Report 10 GHz reference residuals when the accepted OFF is above10 GHz.

Gate recovery maximizes postselected fidelity with phase pi, the S1-selected
channel and validity screens retained; 70-130 ns. No raw, return, survival or
fidelity-magnitude cutoff defines failure. Extra bounded starts and closest-phase
finite diagnostics are retained even when acceptance fails. No result is silently
skipped. Loaded EC, basis checks and flags are saved per point.

Runs use 1-4 processes with BLAS=1, exact checkpoints/source signatures and bounded
replacement attempts. `STOP` requests graceful shutdown. Resume with unchanged
input/source/output. A deadline includes a final-report reserve; incomplete
coverage is reported as partial, never completed. Do not edit numerical source
while a source-hashed run is active.

The archive format is a single `.tar.xz` containing `records.jsonl`, self-contained
input, source/run provenance, summary and per-member SHA256 manifest. Reports
can be rebuilt without unpacking it:

```powershell
.venv/Scripts/python.exe Gnd-coupler+Floating-T/study.py verify-archive Gnd-coupler+Floating-T/output/robustness/robustness-data.tar.xz
.venv/Scripts/python.exe Gnd-coupler+Floating-T/study.py report --kind robustness --study Gnd-coupler+Floating-T/output/robustness/robustness-data.tar.xz --output Gnd-coupler+Floating-T/output/robustness-report-check
```

## Decoherence

The report starts with the central isolated-rail figure on page 1. The circuit
and SQUID hardware tables belong on page 1 of the coupler design report only.
Use `study.py report --kind decoherence --study STUDY_FOLDER --design-report DESIGN.pdf`
to refresh that design cover while generating the separate decoherence PDF.
Omitting `--design-report` writes a separate hardware page under the sibling
`design-hardware/` directory and never inserts it in the decoherence PDF.
`--a-rail-tuning 4 5` specifies the A1/A2
asymmetric-SQUID hardware range; `--rail-tuning 4.4 5.8` remains the B1/B2 range.
The central gate estimate includes intrinsic A noise at idle and B noise at gate.
All four loaded ECs remain fixed when inverting f01 to EJ. The B-only baseline
is retained in data for provenance but is not an extra curve in the report.

```powershell
.venv/Scripts/python.exe Gnd-coupler+Floating-T/study.py decoherence --input Gnd-coupler+Floating-T/output/nominal-design/fabrication-input.json --output Gnd-coupler+Floating-T/output/new-decoherence --rail-tuning 4.4 5.8 --coupler-tuning 6.5 11 --amplitudes 1 100 --gate-time 100
```

Endpoints are GHz; amplitudes are microPhi0/sqrt(Hz) at1 Hz per loop; gate duration
is ns. `--low-cutoff` sets the Ramsey infrared cutoff in Hz. Defaults are explicit
examples, not automatically inferred user inputs. Confirm them for a new study.

The rail calculation uses isolated B, resonant at each supplied idle/
gate mean, with independent B1/B2 flux noise. Include logical B, B20-B0 and an
isolated B1 0-1 reference. Also calculate isolated logical A at idle, using the
specified A SQUID range. The report omits idle B20-B0 and idle isolated B1 curves.
Every ideal gate curve includes B20, raw/selected ERROR and echo-/Ramsey-based
exponential rates. Independent A-idle and B-gate generators add; preserve A's
transition to pair ground in the auxiliary. No coupler noise is added here.

Separately compute coupler-induced dephasing in independent A1-A2-C and C-B1-B2
subsystems, with noiseless rail EJs. Calibrate differential settings once at idle
with C OFF and freeze them during the common gate-frequency shift. This differs
from independently resonating the isolated B pair. Do not attribute the resulting
static dressing/calibration change to an added noise rate. Do not automatically
add combined-noise sections. Quote actual rate ratios; the nominal auxiliary
comparison is about47x, while logical B is about5x, not uniformly10x.

Use first-order Gaussian1/f reference times and explicitly label exponential
substitution. Isolated Ramsey times are solved at each amplitude; the separate
coupler plot scales its amplitude100 reference inversely with amplitude, holding
the Ramsey logarithm fixed. No T1, quadratic/transverse noise, noisy matrix
elements, ramps or total-device fidelity claim. Static coupler participation and
ground-decay matrix elements are different quantities.

Report-only rendering takes `--kind decoherence --study <saved-folder>` and does
not rerun a Hamiltonian. Retained nominal study data and exact approved report
are under `output/decoherence/`.

## Reports and accuracy

Use [STUDY_REPORT_TEMPLATES.md](STUDY_REPORT_TEMPLATES.md). Preserve the latest
approved references under `templates/report-references/`. Residual accuracy is
1 Hz; gate probability accuracy is1e-4 absolute (0.01 percentage point). These are
precision targets, not fidelity acceptance cuts. No exhaustive basis search is
required once modest fixed-control checks agree.
The five-mode physical-basis participation data are retained separately in
output/decoherence/participation.json. To calculate them for a new nominal case,
add `--participation` to the explicit decoherence command. This optional diagnostic
uses the full five-mode OFF model; it is not silently substituted for the
independent-rail dephasing calculation.
