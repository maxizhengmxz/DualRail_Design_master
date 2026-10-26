# Scope and maintenance

Keep this workflow independent of `transmon-based CZ/`; that folder is the
preserved original version. Shared `core.runtime` is permitted. Do not import
original exploratory scripts or use dynamic source rewriting/monkeypatches.

Read README.md for physics conventions and public input ports. Keep physical
occupation order A1,A2,B1,B2 explicit: 0200 is A02-like, not A20-like. Preserve
the single-minimum-S1 selection and phase-pi calibration contract. Keep S2 separate.

Generated calculations, caches, report artifacts and machine-specific process
IDs belong under ignored output/. Never migrate or overwrite old checkpoints:
source hashes deliberately separate old and new implementations. Configuration
and source changes require a new output directory. Do not launch a production
grid during cleanup; validate/dry-run and small smoke calculations suffice.

Run the tests and Ruff commands listed in README.md before preparing a commit.
The user handles Git commit/push; do not create an upload ZIP. Preserve unrelated
working-tree edits, particularly readout and coupler-off workflows.
