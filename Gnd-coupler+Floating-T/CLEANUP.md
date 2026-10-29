# Cleanup record: 2026-10-28

The current source, templates and instructions consistently implement the
confirmed temporary physical OFF objective. S1 selection and ON physics were
preserved. The complete workflow and future reversion are documented separately.

- Retired 12 one-off case/report scripts from active source paths. A generic
  `off_sweep.py` replaces hardcoded fixed-case physical/logical sweep scripts.
- Relocated 51 historical case inputs to `configs/legacy-inputs/`.
  Explicit source/test references were updated. These are reproduction fixtures,
  not current accepted design choices. Readout-only selected inputs were preserved.
- Retired 488 old files/artifacts from active paths,
  including superseded logical/CZ/ZZ-null capacitor searches and loose scripts.
  They reside in ignored `output/retired-2026-10-28/` for reversible recovery;
  they are not imported, resumed or presented as current results.
- Retained the original physical three-body seed, compensated five-body circuit
  and verified five-body physical-null numerical reference, with original provenance.
- Preserved unrelated toy/readout implementation edits and readout-only outputs.
- A machine-readable inventory is `output/cleanup/manifest.json`; no upload ZIPs.
- No production optimization campaign or readout run was launched by cleanup.
  No additional performance proposal was enabled. Tests and small smoke checks
  validate the refactor.

The original broad floating topology and older same-source/CZ/ZZ studies are
historical only; the old HISTORICAL_WORKFLOW.md is now in the ignored retirement archive.

## Approved follow-up: 2026-10-29

The eight performance proposals were subsequently approved and implemented.
Residual checks now target 1 Hz and gate-error checks 1e-4 absolute probability.
No new production campaign was started; small numerical smoke/regression checks
validate exact accelerations, adaptive stopping and process checkpointing.

## Working-tree follow-up: 2026-10-29

The first cleanup still left 51 historical JSON inputs plus their README in the
Git-visible tree. These are now in the ignored retirement archive; only two
required regression inputs were retained under tests/fixtures. This reduces the
working-tree file count by 50 without deleting unrelated changes or useful code.
The complete coupler package remains untracked because it has not been committed;
Git therefore counts all its source, tests and templates as additions.

## Source consolidation: 2026-10-29

- Consolidated 29 regression files into six suites, preserving all 89 checks.
  Original source sections are labeled inside each suite. The independent
  readout reference helper and two JSON fixtures remain test-only material.
- Retired 34 standalone audit, plotting, alternate optimizer and legacy CLI files
  from active source paths. Kept the supported stage entry points and extracted
  reusable ZZ diagnostics into the numerical library.
- Archived six saved readout-case inputs and the obsolete workflow narrative.
- Removed superseded A90/B150 case defaults from active code; the generic readout
  regression uses explicit current inputs instead.
- Archives are ignored output artifacts, not importable active workflow modules.
  Unrelated existing changes in other workflows were preserved.

## Final benchmark/report cleanup: 2026-10-29

- Archived two obsolete active readout scripts (hardcoded calibration and plotting),
  and removed the historical-case CLI/function from readout_common_sweep.py.
  Its supported explicit-input concurrent runner and numerical kernels are intact.
- Archived 244 files across 123 moves: superseded campaigns, loose performance
  experiments, one-off case runners/logs and PDF rendering previews. Original and
  destination paths/sizes are in output/retired-2026-10-29/final-cleanup/manifest.json.
  Final numerical records, source hashes, convergence data, figures and PDFs stay
  at their existing case paths. No unrelated workflow files were changed.
- Preserved the detailed 13-page report and added REPORT_TEMPLATE.md to specify
  subsequent reports' content, layout and validation requirements. The missing
  original detailed PDF was regenerated from its retained authoring source/data;
  numerical provenance files were protected from the historical renderer's writes.
- Moved the readout suite's unittest entry point after all test classes so direct
  execution includes the new basis guards. All existing tests remain.
- Archives are ignored and reversible, not active source. No capacitor/gate/readout
  optimization was rerun, no acceptance threshold was changed, and no new numerical
  approximation was introduced by this cleanup.
