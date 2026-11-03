# Grounded coupler with floating rail transmons

The default workflow remains [WORKFLOW.md](WORKFLOW.md): confirm inputs; solve
A2-C-B1; refine all five transmons and calibrate the S1-selected CZ; validate with
both resonators. Loaded rail ECs are immutable during design. Compensation of
self capacitances preserves them. S1/logical definitions come from the transmon
CZ workflow. The temporary physical OFF target is documented in OFF_POLICY.md.

## Optional studies

Neither is part of a normal design run. Explicitly invoke `study.py robustness`
or `study.py decoherence`; both take a saved nominal input and output folder.
`study.py report` renders saved data only. See [OPTIONAL_STUDIES.md](OPTIONAL_STUDIES.md)
for sampling, noise assumptions, constraints, rescue policy, resume and examples.
Its default-policy table distinguishes robustness-only rescue from nominal
constraints and documents the unresolved nominal seven-mode OFF basis accuracy.
The old fabrication_study.py/fabrication_report.py paths retain compatibility
imports; study.py is the supported runner.

## Source and validation

- `floating_coupler/`: circuit, physical/logical OFF, CZ and precision kernels.
- Existing stage CLIs: three-body design, five-body refinement, idle/gate
  calibration and seven-mode validation. No automatic full workflow on import.
- `studies/`: optional study runner, consolidated physics/reporting, archive I/O.
- `templates/`: circuit template and exact latest design/decoherence/robustness
  report references. Use REPORT_TEMPLATE.md and STUDY_REPORT_TEMPLATES.md.
- `tests/`: focused regression suites. Run `run_tests.py` after numerical changes.

All frequencies/couplings are cyclic GHz internally; physical capacitances fF.
Residual precision target1 Hz; gate probability target1e-4 absolute. Acceptance
and convergence are separate. Preserve failed diagnostics and explicit bounds.

## Retained results

- `output/nominal-design/`: authoritative nominal inputs, capacitances, controls,
  validation/readout/gate results, timing and reports. The seven-mode physical
  OFF residual is not retrospectively certified to1 Hz.
- `output/decoherence/`: approved nominal decoherence data, figures and final PDF.
- `output/robustness/`: final summary PDF plus ONE verified robustness-data.tar.xz
  containing all9,456 retained evaluation records and reproduction inputs.
- `output/RESULTS.json`: current index and checksums.

Superseded campaigns, previews, duplicate PDFs, patch/launch scripts and caches
are removed after archive/preservation verification. Other workflows and unrelated
Git changes are untouched. See CLEANUP.md for the cleanup record.
