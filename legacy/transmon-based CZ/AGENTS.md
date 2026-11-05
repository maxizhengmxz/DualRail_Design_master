# Archived transmon CZ source

This package is historical. New toy CZ and coupler toy-stage calculations use
../../2Q-gate_XM/. The active written specification is
../../2Q-gate_XM/WORKFLOW.md. Executable policy belongs to
../../core/cz_channel_policy.py. The local channel_policy file is only a
compatibility re-export. Do not introduce active callers of archived kernels.

Read README.md and CLEANUP.md before changing archived behavior. Preserve the
historical numerical algorithms, configurations and optional tests. Imports
launch nothing. Run archived tests through root run_tests.py --legacy.
The active policy/import-isolation tests stay in the root tests directory.

Per-point campaign checkpoints were removed on 2026-11-05. Compact maps,
CSV summaries, original manifests and calculation-source provenance are under
ignored ../output/transmon-cz/. The archive cannot resume deleted checkpoints.
Generated results and caches stay ignored. No production grids during cleanup.
