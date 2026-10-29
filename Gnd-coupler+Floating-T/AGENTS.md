# Floating coupler workflow

Read WORKFLOW.md and README.md before a new design or model extension. WORKFLOW.md
is the end-to-end specification; configs/workflow-request-template.json is its
request schema, not a directly executable stage config. Keep new floating-coupler source,
templates and tests in this folder. Preserve other workflows and existing edits.

Use the toy workflow's WORKFLOW.md and transmon_cz/channel_policy.py as the sole
authority for logical state and CZ channel definitions. The local channel_policy
module imports that authority. Select one feasible minimum-S1 resonance at common
intended g, with common score basis and comparison conditions. Exclude the
intended edge, keep S2 separate and apply the stated unintended |g/delta| screen.
Crossing count/order and minimum gap do not select the resonance. Do not promote
rescaled-score estimates or first-crossing records to accepted designs.
Bare 0020 never specifies a logical source.
The selected source/auxiliary owns ON calibration. OFF is separately versioned.
TEMPORARY-OFF: both five and seven modes null physical 0100-0010, in physical
rail order A1,A2,B1,B2, after full-state canonical projection onto ALL FOUR
physical single-excitation references. C is ground and resonators vacuum in the
references; virtual excitations and intra-rail mixing remain. Three-body OFF
nulls physical A2/B1 01-10. Logical exchanges, CZ auxiliaries and ZZ are residuals,
not acceptance targets. Plot both logical residuals over a wide diagnostic
coupler-frequency range to reveal nonmonotonicity and mark allowed bounds.
This physical-versus-logical choice remains OPEN. Read OFF_POLICY.md for the
restoration map; do not silently change it or relabel historical results.
Use the agreed three -> five -> ON calibration -> seven stages in WORKFLOW.md.
Do not call a bare two-reference elimination a four-reference physical OFF.

During capacitor design refinement compensate self capacitances to preserve all
supplied loaded rail ECs. The stationary rail remains fixed; the moved rail may
be frequency-trimmed during final gate calibration as authorized by the user.
For frozen-hardware off-point validation freeze rail EJs and vary coupler EJ only; EC and charge
coefficients J stay fixed. For gate calibration, vary only the explicitly
authorized rail/coupler EJs and duration, keeping the geometry and EC/J fixed.
Design changes to link strengths/capacitors require explicit ranges and consistent
electrostatic reconstruction. Ask for the specific CZ channel and missing pair
parameters before numerical refinement. A canceled channel is not cancellation
of every residual interaction. Report detunings, overlaps and basis convergence.

Static gate benchmarks use consistently coupler-dressed ON logical references,
retaining the CZ auxiliaries. First calibrate the nonlocal conditional phase to
pi, then improve return/survival while maintaining that constraint. Do not mix
these results with an instantaneous idle/on/idle quench or a ramped pulse.
Report raw fidelity, survival-normalized score and logical survival together.

Gate refinement must stop at the agreed 1e-4 probability precision, with finite
budgets and phase-pi feasibility retained. Large-basis work is phase-only frequency
correction at fixed duration/coupler, then fixed-control convergence checks; do
not restart a high-basis frequency/time optimizer. See WORKFLOW.md stage 4.
The requested 99.9% threshold concerns POSTSELECTED fidelity, never raw fidelity.
Start seven-mode calibration from the supplied five-body bias and duration;
transfer phase locally before trying wider starts. Reuse one captured gate
spectrum for all duration proposals/time roots, then recompute at any corrected
frequency before accepting phase pi. Reuse exact same-cell/basis audit caches and
record spectrum-build counts. Never freeze a spectrum across frequency changes.

Mandatory integrated readout idle calibration: with loaded ECs and all geometry
fixed, retune individual rail EJs to compensate differential coupler/resonator
dressing and recover near-resonant dressed rails under the existing readout
constraints. Equal bare frequencies do not imply dressed resonance. Re-null OFF
with coupler EJ only after rail calibration, then recheck both pairs at one common
OFF bias. Freeze the calibrated differential settings for common-frequency maps.
Do not compare a retuned historical design with unretuned current controls as an
intrinsic hardware comparison. See WORKFLOW.md stage 5.
Run A/B integrated readout sweeps concurrently with two spawned processes and
one BLAS thread per worker; use the explicit-input readout_common_sweep runner.
Keep the calibrated centers and differential settings fixed. Record parent wall
time separately from summed worker times; do not run another heavy gate search
alongside the two sweeps. Concurrency alone does not change numerical bases.

For new cases obtain loaded EC tuples, idle/gate rail positions, logical splitting
reference/convention, CZ channel, target duration/scope, design constraints and
allowed tuning controls. Do not inherit any prior numerical case. Fit rail
splittings at the agreed reference and freeze the resulting capacitances; do
not silently refit intra-pair J at every frequency. Translate the speed target
to the dressed CZ channel, not an identically numbered reduced port exchange.

Keep generated results in output/ (ignored); do not create upload ZIPs. Do not
launch readout optimization before a successful coupler design; it belongs to
the explicitly requested seven-mode validation stage. Run run_tests.py for
numerical source changes.

Approved numerical accuracy (2026-10-29): residuals 1 Hz absolute; raw/postselected
gate error and survival 1e-4 absolute probability (0.01 percentage point). Stop
modest fixed-control basis audits on agreement; no compulsory huge audit. Record
unconverged on budget exhaustion. All eight SPEED_PROPOSALS.md items are approved;
use exact caches, odd-only OFF work, spectral time reuse, early feasibility,
bounded workers/checkpoints and precision-aware convergence. Physics/phase/S1/
readout thresholds remain unchanged. Readout already uses adaptive validation;
do not launch its scans or modify its constraints as part of this implementation.

Integrated common readout sweeps use local6/pair12/C3 screening and C5 audits
at centers, endpoints and near constraint boundaries. Both resonators retain
exactly |0>,|1>,|2>. The proposed local4/pair8 truncation failed empirical
chi checks and is not the default. Record each selected basis and audit changes;
this readout policy never certifies residual couplings to 1 Hz.

## Report format

All subsequent full design reports must follow REPORT_TEMPLATE.md and the
preserved detailed 13-page PDF in output/report-reference/. Keep its form,
section sequence and level of detail; the short benchmark report is supplementary.
Use current results and explicit validation status, not the reference case
numbers. Render and inspect every page. No optimization runs during reporting.
