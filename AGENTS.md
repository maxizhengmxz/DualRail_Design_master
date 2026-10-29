# Project guidance for coding agents

Dual-rail logical definitions, S1/S2 scoring and selected-resonance policy are
owned by `transmon-based CZ/WORKFLOW.md` and
`transmon-based CZ/transmon_cz/channel_policy.py`. The active S1 toy search
(`2Q-gate_XM/`) and explicit coupler workflow import that stable authority.
Select one feasible minimum-S1 resonance at common intended coupling and in
the same score basis; keep S2 and crossing-order diagnostics separate. Never
infer a logical source from a physical auxiliary or reuse first-crossing-policy
metadata as S1 certification. The selected CZ channel owns ON calibration.
OFF in the explicit coupler workflow temporarily nulls physical 0100-0010
exchange in five/seven modes, with all four physical single-excitation
references retained. Three modes null physical A2/B1 01-10. Logical exchanges,
CZ-auxiliary couplings and secular ZZ are residual diagnostics. This temporary
physical-versus-logical choice is an open question; see Gnd-coupler+Floating-T/OFF_POLICY.md.


For floating-transmon tunable-coupler design, first read
[Gnd-coupler+Floating-T/AGENTS.md](Gnd-coupler+Floating-T/AGENTS.md) and
[Gnd-coupler+Floating-T/WORKFLOW.md](Gnd-coupler+Floating-T/WORKFLOW.md).
That independent workflow owns its source, tests and templates under
`Gnd-coupler+Floating-T/`; do not launch the grounded or readout workflows automatically.

For a grounded static coupler-off task, first read
[coupler turn-off/AGENTS.md](<coupler turn-off/AGENTS.md>) and
[coupler turn-off/README.md](<coupler turn-off/README.md>).
They define the required input questions, physics conventions and staged workflow.

All grounded coupler-off source, tests and templates live in `coupler turn-off/`.
Readout optimization is a separate workflow in `readout_workflow/` and must
not be launched automatically during coupler-off analysis. Preserve unrelated
project functionality and existing user edits. Do not create upload ZIPs;
the user will commit/push source to Git. Exclude generated results and caches.
