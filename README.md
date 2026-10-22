# DTC Dual-Rail Analysis

The two workflows are organized separately:

- [Coupler turn-off](<coupler turn-off/README.md>): static physical/logical
  cancellation, fixed-capacitance EJ optimization and optional frequency maps.
- [Readout optimization](readout_workflow/README.md): optimize a dual-rail pair
  independently, then supply its selected parameters to the coupler workflow.

Coding agents should start with [AGENTS.md](AGENTS.md).

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
remain available separately in `dtc_dualrail/`.
