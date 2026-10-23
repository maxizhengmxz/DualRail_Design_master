# DTC Dual-Rail Analysis

The workflows are organized separately:

- [Coupler turn-off](<coupler turn-off/README.md>): static physical/logical
  cancellation, fixed-capacitance EJ optimization and optional frequency maps.
- [Readout optimization](readout_workflow/README.md): optimize a dual-rail pair
  independently, then supply its selected parameters to the coupler workflow.
- [Logical-resonance gates](<logical-resonance gate/README.md>): five-transmon
  logical-transition iSWAP and controlled-phase/CZ searches and pulse studies.

Coding agents should start with [AGENTS.md](AGENTS.md).

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
