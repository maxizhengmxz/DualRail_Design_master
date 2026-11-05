# DTC Dual-Rail Analysis

The workflows are organized separately:

- [Coupler turn-off](<coupler turn-off/README.md>): static physical/logical
  cancellation, fixed-capacitance EJ optimization and optional frequency maps.
- [Readout optimization](readout_workflow/README.md): optimize a dual-rail pair
  independently, then supply its selected parameters to the coupler workflow.
- [Logical-resonance gates](<logical-resonance gate/README.md>): five-transmon
  logical-transition iSWAP and controlled-phase/CZ searches and pulse studies.
- [XM CZ](2Q-gate_XM/README.md): active four-transmon toy CZ search and maps;
  shared logical/S1 specification in [its workflow](2Q-gate_XM/WORKFLOW.md).
- [Archived transmon CZ](<legacy/transmon-based CZ/README.md>): historical
  simulator, search, inputs and optional compatibility tests. New toy CZ work
  uses `2Q-gate_XM/`. Compact historical maps/summaries are under ignored
  `legacy/output/transmon-cz/`; per-point checkpoints have been removed.

Coding agents should start with [AGENTS.md](AGENTS.md).

## Gate simulation ownership

| Calculation | Numerical implementation |
| --- | --- |
| Four-transmon toy CZ search and maps, including the coupler's toy stage | `2Q-gate_XM/xm_cz/` |
| Five-transmon explicit-coupler static CZ | `Gnd-coupler+Floating-T/floating_coupler/on_gate.py` and `pairs.py` |
| Seven-mode CZ including readout resonators | `Gnd-coupler+Floating-T/final_seven_validation.py` and `seven_readout.py` |
| Five-transmon tuning-pulse CZ | `Gnd-coupler+Floating-T/floating_coupler/pulsed_gate.py` and `filtered_pulse.py` |
| Existing logical-resonance iSWAP/CZ studies | `logical-resonance gate/logical_resonance_gate/` |

XM is the active toy backend. The explicit-circuit and pulse models build and
propagate their own Hamiltonians. XM and the explicit coupler share logical
definitions, S1/S2 and selected-channel policy through `core.cz_channel_policy`.
The separate logical-resonance workflow owns its existing gate objectives and
backend. Active simulations never import the archived `transmon_cz` package.
The root import-isolation test covers these gate entry points with that package
explicitly blocked. Archived tests are enabled only through `--legacy`.

Shared physics and the dependency map are documented in [core/README.md](core/README.md).
The Python package formerly named `dtc_dualrail` is now `core`. Readout and
coupler-off share its transmon and electrostatic primitives; their searches
and physical circuit definitions remain in their own workflow folders.

Older studies unused by these workflows are archived in [legacy/](legacy/README.md),
including their analysis helpers and dedicated tests.

```sh
uv sync
uv run python "coupler turn-off/run.py" --help
uv run python "coupler turn-off/run_tests.py"
```

For a new coupler case, supply the externally readout-optimized pair, direct
bare A2-B1 coupling (rate or capacitance), hardware/readout constraints and
candidate coupler geometries. The minimization objective must be chosen
explicitly for each new parameter set. Start from
[the blank template](<coupler turn-off/configs/template.json>).

All coupler-related code, tests, documentation and templates are in
`coupler turn-off/`. Generated results go in its ignored `output/` folder.
No saved case results or ZIP packaging are required; the source is ready
for normal Git staging and review. Existing general-purpose project modules
remain available separately in `core/`.

Run all active workflow regression tests with `uv run python run_tests.py`.
Add `--legacy` to include archived tests. Gate code imports its workflow-local
`logical_resonance_gate` package directly; core contains no gate or legacy
forwarding modules. No additional package installation is required for these
checkout entry points.
