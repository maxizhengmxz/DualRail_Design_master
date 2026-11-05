# Legacy CZ retirement, 2026-11-05

The user authorized cleanup of the old CZ data and retirement of this module.
This supersedes the 2026-11-03 temporary keep request.

Source, five example configurations and optional tests moved from the root
transmon-based CZ directory into legacy/transmon-based CZ. The runner's
repository-root path and root optional-test discovery were updated. Numerical
algorithms remain unchanged. Active XM, readout and explicit-coupler calculations
import no archived CZ kernels. Policy remains in core. Active policy-location
tests remain in the default root coverage.

The campaign contained 82,328 files totaling 1,321,165,196 bytes. Cleanup removed
82,008 per-point JSON checkpoints totaling 1,275,317,915 bytes. The 320 remaining
files were preserved in one verified 25,613,460-byte historical-summary.tar.xz
under ignored legacy/output/transmon-cz. These include all available PDF/PNG
maps, compact CSV tables, manifests, original configurations and the campaign's
calculation-source snapshot. The adjacent index contains member checksums and
table row counts. Every archived member was verified before deleting loose data.
The seven point_table CSVs retain compact numerical campaign outcomes.

The documented reports/dual_rail_CZ_gate_workflow_report.pdf was already missing
before cleanup. No matching report was found elsewhere in the checkout. The
summary archive cannot resume the deleted campaign. Historical absolute paths
remain provenance. New calculations use 2Q-gate_XM and fresh output folders.

Generated archives/data remain ignored by Git. Source, examples, tests and
documentation remain reviewable as ordinary repository changes.

Validation passed all 287 active tests and all 335 tests when the two archived
suites were included through root run_tests.py --legacy. The active import
guard rejects transmon_cz and still imports XM/coupler successfully. The moved
runner's --help works and the XM smoke input validates without a calculation.
All 18 relocated code/config/test files other than runner/documentation retain
their Git content hashes. Archive and index are ignored by Git.

The current Ruff check additionally reports pre-existing import ordering in
2Q-gate_XM/tests/test_physics.py and formatting in xm_cz/channel_policy.py and
xm_cz/runner.py. Those three files were untouched by the source relocation.
The subsequent final consistency pass corrected those style-only issues and
confirmed all nine XM tests and its lint/format checks pass. The expanded active
import guard covers static, seven-mode, pulse, robustness/parasitic and separate
logical-resonance gate entry points with the archived CZ package blocked.
