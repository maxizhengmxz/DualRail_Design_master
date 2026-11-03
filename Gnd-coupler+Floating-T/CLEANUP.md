# Cleanup record: 2026-11-03

Approved scope: keep the authoritative nominal design and its decoherence study;
compress full retained robustness data into one file; consolidate both studies
as opt-in workflows; preserve the latest reports as future report templates.

## Retained

- output/nominal-design/: self-contained fabrication input, three/five/seven
  nominal results, calibrated controls, readout/gate validation, timing and
  reports. Detailed hardware report and later benchmark validation have distinct
  provenance. The final seven-mode OFF result is not recertified to1 Hz.
- output/decoherence/: approved six-page PDF, normalized results.json, validation
  and participation records, circuit assets and final figures. No superseded
  combined-noise report section is retained as an active result.
- output/robustness/: approved twelve-page summary PDF and ONE64,032,856-byte
  robustness-data.tar.xz. It holds9,456 evaluation records including failed,
  approximate and unavailable outcomes, reproduction input and source/run
  provenance. Repeated physical settings remain separate evaluation records.
- templates/report-references/: exact latest decoherence and robustness PDFs,
  plus the existing detailed design reference. manifest.json records checksums.
  STUDY_REPORT_TEMPLATES.md governs their future use; example values are not
  future defaults.
- output/RESULTS.json and output/cleanup-manifest.json index retained files,
  archive checksum, validation and deletion results.

## Source consolidation

study.py is the explicit entry point. studies/ owns archive I/O, recovery runner,
shared policy, decoherence calculation and report rendering. No imports from
output folders remain. Previous fabrication entry modules retain compatibility
imports. The shared readout rescue implementation and default coupler-design
physics remain intact. Source-hashed historical data are readonly; do not claim
that old runs were resumed or re-certified under the refactored source hash.

The default workflow remains nominal three -> five -> ON -> seven validation.
Robustness and decoherence run only when explicitly requested. They have
independent controls/configuration and generated output folders. Report-only
commands do not run Hamiltonian optimization.

## Removed

28 obsolete result directories/subtrees containing11,822 files and944,833,364
bytes were removed only after preservation and archive checks. These included
superseded topology/capacitor searches, dated campaign copies, intermediate PDFs,
per-point loose files, previews, one-off launch/patch/audit scripts and downloaded
paper copies. Temporary cleanup/check outputs and Python caches are also removed.
No unrelated project workflow or existing user edit was deleted; no commit/push
or upload ZIP was created.

## Verification

- Per-archive-member SHA256 and size checks; record count and outcome statistics
  matched the completed final study (9,456 records,9,436 finite gate results).
- Both optional report commands rendered from preserved data/archive alone.
- The consolidated nominal decoherence calculation exactly reproduced the saved
  echo/Ramsey postselected fidelity at noise amplitude10.
- Archive reuse credited every filled bin and ran zero new samples.
- Required regression suite:123 tests passed. Formatting and undefined/unused
  symbol checks passed. A final check is repeated after file removal.
# Deferred cleanup outside this workflow

2026-11-03 user decision: KEEP `../transmon-based CZ/output/ec_gate_campaign_4to5GHz/`
as is for now (approximately 1.23 GiB; 82,008 retained parameter points).
It is a candidate for later cleanup, not authorized for deletion or compression.
At the next project-wide results cleanup, ask the user whether to clean up this
campaign before taking action. Preserve its final report and provenance unless
the user explicitly directs otherwise. No scheduled reminder date was specified.
