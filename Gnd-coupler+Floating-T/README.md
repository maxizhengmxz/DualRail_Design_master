# Grounded-C2 tunable coupler with floating rails

Start with [WORKFLOW.md](WORKFLOW.md). The currently confirmed five/seven-mode
OFF target is **temporarily physical 0100-0010 exchange**, not encoded logical
exchange. [OFF_POLICY.md](OFF_POLICY.md) records the open question and restoration
map. Logical exchanges and secular ZZ are mandatory residual diagnostics.

## Source map

- `floating_coupler/`: reusable electrostatics, spectra, OFF observables,
  capacitor design and static gate kernels.
- `staged_three_body.py`, `grounded_c2_search.py`: three-body design kernels.
- `grounded_five.py`, `refine.py`: five-body
  construction and physical OFF refinement.
- `search_correct_channel_caps.py`: existing case-specific capacitor optimizer;
  its ON equations require an explicit matching S1-selected channel. It is not
  a general channel selector. Supply new inputs/output for each numerical case.
- `off_sweep.py`: frozen five-body OFF search and wide logical-residual plots;
  accepts an explicit circuit record and frequency intervals.
- `calibrate_on.py`, `floating_coupler/on_gate.py`: static dressed ON calibration.
- `calibrate_idle.py`: mandatory individual-EJ dressed-idle compensation,
  alternating with the integrated physical OFF root at frozen geometry/EC.
- `seven_readout.py`, `final_seven_validation.py`: integrated seven-mode kernels;
  readout is invoked only in the final stage, never automatically.
- `readout_common_sweep.py --cell ... --readout-A ... --readout-B ... --output ...`:
  concurrent calibrated A/B comparison sweeps, default two BLAS=1 processes.
  Generates one JSON/PNG per pair, source/input-checked resume checkpoints and
  a timing summary with parent elapsed and summed worker times. `--workers 1`
  is an explicit serial diagnostic. Integrated sweeps use local6/pair12/C3 screening with guarded C5 audits.
  Local4/pair8 was benchmarked and rejected for inaccurate chi matching.
- `configs/*template.json`: input templates. Historical campaign inputs are
  archived under ignored `output/retired-2026-10-28/legacy-inputs/`; only the two
  required regression inputs remain under `tests/fixtures/`.
- `tests/`: six consolidated suites (circuit, OFF, CZ, readout, ZZ, runtime),
  retaining physics coverage and refinement/cache/readout guards (101 checks).
  `independent_readout.py` is the separate
  mathematical reference used by readout tests.
- `run_tests.py`: the single regression entry point. Use
  `.venv/Scripts/python.exe Gnd-coupler+Floating-T/run_tests.py` from the project root.

The complete staged workflow is not one automated command. Use explicit stage
inputs and preserve immutable loaded rail ECs and S1 channel provenance.
All frequencies/couplings are cyclic GHz internally; capacitances are fF.

## Frozen sweep example

```
.venv/Scripts/python.exe Gnd-coupler+Floating-T/off_sweep.py --circuit Gnd-coupler+Floating-T/output/current-three-to-five-comparison/frozen-circuit.json --off-range 7 12 --plot-range 7 20 --output Gnd-coupler+Floating-T/output/new-physical-off-sweep
```

The extended plot interval is diagnostic, not an expansion of allowed OFF bounds.
No capacitor or rail-EJ refitting occurs in this command.

## Cleanup and performance

See [CLEANUP.md](CLEANUP.md) for retired artifacts and retained reference records.
[SPEED_PROPOSALS.md](SPEED_PROPOSALS.md) records the subsequently approved
accelerations and implementation map. Numerical stopping targets are 1 Hz for
residuals and 1e-4 absolute gate error. Audits stop on fixed-control agreement,
with unconverged results explicitly marked. Independent capacitor starts use
bounded resumable workers (`--workers`, default 4); source changes require a new
checkpoint directory. No production design campaign runs automatically.

Standalone exploratory audit scripts, alternate case optimizers and old report
campaigns are archived under ignored output/retired-2026-10-29/standalone-tools/.
They are not part of the active workflow. Reusable numerical verification is in
floating_coupler/precision.py; automated physics checks are in the six suites.

Local gate refinement now stops at a 1e-4 plateau across raw/postselected
fidelity, survival and per-input leakage, with finite iteration/evaluation
budgets. Large-basis work corrects phase at fixed duration and coupler bias;
it does not restart a frequency/time optimizer. Phase-pi and fixed-control
basis validation remain mandatory. A future 99.9% target stop means the
postselected score; it is separate from this precision-limited refinement.

Seven-mode calibration starts at the supplied five-body solution, with local
phase transfer before wider starts. Duration grids/time roots reuse one captured
spectrum per bias. Frequency changes always construct/use the corresponding new
spectrum; fixed-control audits reuse existing same-cell/basis caches. Search
results record spectrum builds and elapsed time. No broad production search is
launched merely by importing these tools.

## Reports and retained results

[REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) defines the required detailed report
format, content and visual style. The preserved reference is
[the detailed design report](output/report-reference/coupler-design-report.pdf).
Future full reports retain this level of detail; timing-only summaries are
supplementary. Reference numbers are historical examples, not current defaults.

Current numerical results are in `output/full-benchmark-20261029/`:
`final/validated-gate.json`, `idle-retuned/final/calibrated-cell.json`,
`final/off-convergence.json`, `B-center-larger-basis.json`, and `readout-sweeps/`.
The final gate precision and B-center readout check pass; seven-mode OFF residuals
remain uncertified to 1 Hz. `output/RESULTS.json` indexes retained evidence.

One-off scripts, rendering previews and superseded campaigns are archived under
`output/retired-2026-10-29/final-cleanup/`; its manifest records original paths.
Old hardcoded readout CLIs have been retired. Use the explicit-input concurrent
readout runner; the supported numerical kernels and all 101 tests are retained.
