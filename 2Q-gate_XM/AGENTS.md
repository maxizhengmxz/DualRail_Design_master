# Scope and maintenance

Keep the numerical model independent of the original exploratory simulation
and search kernels. Shared `core.runtime` and the stable definitions/S1 policy
in `transmon-based CZ/transmon_cz/channel_policy.py` are permitted; both the
coupler and this active toy search import that single authority. Do not import
original exploratory scripts or use dynamic source rewriting/monkeypatches.

Read README.md for physics conventions and public input ports. Keep physical
occupation order A1,A2,B1,B2 explicit: 0200 is A02-like, not A20-like. Preserve
the single-minimum-S1 selection at common intended g and phase-pi calibration
contract. Compare all allowed logical sources; physical 0020/0200 do not fix
a logical bit. Keep S2 and crossing-order diagnostics separate from ranking.

Generated calculations, caches, report artifacts and machine-specific process
IDs belong under ignored output/. Never migrate or overwrite old checkpoints:
source hashes deliberately separate old and new implementations. Configuration
and source changes require a new output directory. Do not launch a production
grid during cleanup; validate/dry-run and small smoke calculations suffice.

Run the tests and Ruff commands listed in README.md before preparing a commit.
The user handles Git commit/push; do not create an upload ZIP. Preserve unrelated
working-tree edits, particularly readout and coupler-off workflows.
