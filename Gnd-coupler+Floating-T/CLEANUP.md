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


## Parasitic study cleanup, 2026-11-05

The optional study is documented in PARASITICS.md and OPTIONAL_STUDIES.md.
Normal design does not launch it. The explicit preparation runner owns fit and
nominal-reference stages. Quantum validation takes explicit gate/reference
inputs, replacing hardcoded dependencies on exploratory result directories.
Report generation uses saved data only. Static controls, physical/logical OFF
definitions, matching masks, local optimizer status, physical probabilities and
the unresolved residual-basis precision are annotated in source and workflow.
The gate basis check now explicitly includes logical_survival. The old records
already contain both survival values, and their report checks all three metrics.

Retained output/parasitics holds the final 17-page PDF, exact current 24 fits,
all quantum case records and basis checks, nominal inputs/reference, approved
circuit assets and a checksum manifest. Historical exploratory JSON and exact
pre-cleanup source are preserved in one historical-provenance.tar.xz. All 71
historical JSON members were checked byte-for-byte before removal. Original
campaign identities are preserved and are not resumable under changed source.

Removed 22 superseded loose result directories containing 163 files and
16,965,023 bytes after preservation checks. The consolidated retained study
occupies about 6.6 MB. Superseded comparison/report builders, their unused
calculation entry points, duplicate PDFs and parasitic-only preview folders
were removed. Shared physics/projection/plot helpers remain where the active
workflow calls them. Other workflows and the deferred legacy toy data remain.

Verification included 134 passing workflow tests, import/CLI compilation,
one minimum-change compensation smoke check, and rendering/inspection of all
17 report pages. The smoke fit preserved loaded ECs below 1 Hz. The report
was regenerated from the retained data without quantum recalculation. No Git
staging, commit or push was performed.

## Parameters-only retention, 2026-11-05

The user's subsequent instruction supersedes the parasitic archive retention
above. Preserve configs/Design_V1_20261105.json as the complete seven-mode
nominal design with all fourteen parasitics zero. It contains intentional
capacitors, loaded ECs, resonator L/C/frequencies, idle/gate frequencies and EJs,
gate duration, channel definitions, asymmetric-SQUID hardware and tuning ranges.
The known nominal OFF basis limitation remains annotated.

Keep templates/report-references/parasitic-compensation-reference.pdf as the
approved 17-page visual template. Remove all embedded calculation attachments.
Page render checksums match the original report exactly after attachment removal.
The template manifest records its new file checksum.

Delete output/parasitics including all case records, candidates, nominal
baseline calculations, diagrams, duplicate parameter copies and historical
archives. The optional workflow remains executable from the named parameter
record and a fresh output directory. Other study outputs follow their prior
retention instructions. No additional simulation was launched for this cleanup.
