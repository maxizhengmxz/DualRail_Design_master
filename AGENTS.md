# Project guidance for coding agents

For a static coupler-off task, first read
[coupler turn-off/AGENTS.md](<coupler turn-off/AGENTS.md>) and
[coupler turn-off/README.md](<coupler turn-off/README.md>).
They define the required input questions, physics conventions and staged workflow.

All coupler-off source, tests and templates live in `coupler turn-off/`.
Readout optimization is a separate workflow in `readout_workflow/` and must
not be launched automatically during coupler-off analysis. Preserve unrelated
project functionality and existing user edits. Do not create upload ZIPs;
the user will commit/push source to Git. Exclude generated results and caches.

For combined dual-rail readout/XM-CZ work, especially the fixed-rail-A
physical-frequency-anchor workflow, first read
[weekend_simulation/DEVELOPMENT_MEMORY.md](weekend_simulation/DEVELOPMENT_MEMORY.md)
and [2Q-gate_XM/AGENTS.md](2Q-gate_XM/AGENTS.md). The development-memory file
records required independent-map semantics, reconciliation rules, known wrong
approaches, current entry points and data-retention contracts. Update it when a
workflow invariant changes.
