# XM two-qubit gate workflow

Independent, cleaned implementation of the weekend dual-rail CZ search.
The original `transmon-based CZ/` source, configurations and saved results are
preserved. Numerical kernels use shared `core.runtime`; this package does not import
the original exploratory CZ kernels or launch readout/coupler-off optimization.
Stable logical definitions, S1/S2 scoring and common-g selection are imported
from `../transmon-based CZ/transmon_cz/channel_policy.py`, the same authority
used by the coupler workflow.

Run from the repository root using its existing `uv` environment:

```powershell
uv run python "2Q-gate_XM/run.py" validate "2Q-gate_XM/configs/example_grid.json"
uv run python "2Q-gate_XM/run.py" run "2Q-gate_XM/configs/example_grid.json" --dry-run
uv run python "2Q-gate_XM/run.py" run "2Q-gate_XM/configs/smoke.json"
uv run python "2Q-gate_XM/run.py" run "2Q-gate_XM/configs/example_grid.json"
uv run python "2Q-gate_XM/run.py" plot "2Q-gate_XM/configs/example_grid.json"
uv run python -m unittest discover -s "2Q-gate_XM/tests" -v
uv run python -m ruff check "2Q-gate_XM"
uv run python -m ruff format --check "2Q-gate_XM"
```

No arguments show help. `run` explicitly launches calculations; `validate`
and `--dry-run` do not. The example full grid contains 6,727 points, not a quick
smoke test. Grid runs plot automatically after successful completion. Config
output paths resolve relative to the config file; outputs and caches are ignored.

## Input ports

All energies and matrix elements are cyclic GHz; time is ns. The grid uses MHz.
Physical order: A1,A2,B1,B2. See `configs/example_grid.json` for an executable
template. `task_defaults` applies to every point; individual task fields override it.

| Field | Meaning | Default |
|---|---|---|
| `logical_splittings_GHz` | Isolated A and B rail eigenmode splittings | `[0.150,0.090]` |
| `bare_f01_bounds_GHz` | Separate bounds for A1,A2,B1,B2 | `[4,5]` each |
| `frequency_GHz` | Common A frequency scan: start, stop, step | `[4,5,0.05]` |
| `resonance_channel` | Physical auxiliary occupation, `0020` or `0200` | `0020` |
| `logical_sources` | Logical source candidates in A,B bit order | `["00","01","10","11"]` |
| `g_GHz` | Intended dressed matrix-element magnitude | `0.005` |
| `duration_ns` | Optional explicit calibration time window | `[0.7,1.3]/(2*g)` |

The same rail's two bare f01 frequencies remain equal. Its allowed common
frequency is the intersection of its two input bounds; empty intersections are
invalid. These bounds also apply to B's final phase-calibration adjustment.

## Physics and selection

1. Full-cosine transmons, ncut=30, four local levels each. Diagonalize rails
   separately; logical 0/1 are the lower/upper single-excitation eigenmodes.
2. Solve rail charge couplings for the requested splittings. Scan A frequency
   and solve B frequency for each allowed logical-source collision with the
   requested physical auxiliary.
3. Set the inter-pair charge coefficient J so the intended dressed matrix element
   is g. Transform the full n_A2 tensor n_B1 interaction; no extra propagated links.
4. Require all unintended logical-origin |g/delta|<0.2. Rank by S1, the mean
   logical-origin sum of squared ratios. Record auxiliary-origin S2 separately.
5. Select ONE minimum-S1 source/frequency position at common intended g. Refine
   interior minima; retain and flag boundary minima without extending bounds.
   S2, crossing count and first-crossing order do not rank candidates. Do not
   compare unequal-g scores or promote uniformly rescaled estimates to designs.
6. Keep A frequency fixed, vary B frequency by +/-12 MHz and stopping time to
   obtain conditional phase pi, then maximize raw fidelity. Dependent B rail
   and inter-pair couplings are re-solved to preserve splitting targets and g.
7. Report both fidelities at the SAME selected gate, with local Z corrections.
   Acceptance is Tr(M-dagger M)/4; postselected fidelity is raw/acceptance.

Fixed retained limits: EC<=300 MHz, EJ<30 GHz (numerical cap 29.999), EJ/EC>=25,
inter-pair charge coefficient J<=20 MHz. No old 100 MHz spacing filter or S2 cutoff.

## Channel labels and caveats

- `0020`: B20-like auxiliary. Its logical source is selected by minimum S1.
- `0200`: A02-like auxiliary. Its logical source is selected by minimum S1.

Logical 0/1 remain the lower/upper isolated-pair one-excitation modes. Pair
ground is separate. Low-level `layout` and `collision_residual` require the
source explicitly. Set `logical_sources` to an explicit subset only when the
user restricts the allowed channels. Preserve the selected source/auxiliary
through phase-pi calibration and downstream coupler ON refinement. OFF in the explicit coupler workflow temporarily nulls physical 0100-0010
exchange in five/seven modes, with all four physical single-excitation
references retained. Three modes null physical A2/B1 01-10. Logical exchanges,
CZ-auxiliary couplings and secular ZZ are residual diagnostics. This temporary
physical-versus-logical choice is an open question; see Gnd-coupler+Floating-T/OFF_POLICY.md.
The selection record stores S1, common g, reference basis and definition version;
old fixed-source/first-crossing outputs are not silently relabeled.

Auxiliaries are selected by overlap with those explicit bare occupations, not
energy ordering. The two channels are equivalent only under full device/node
relabeling, not for a fixed asymmetric device. The calibration tracks the B
auxiliary with physical local-basis overlaps for 0020; the A auxiliary is fixed
during 0200 calibration. The initial scan still uses pointwise bare-overlap
assignment and same-index root bracketing, not a complete adiabatic branch
tracker. Ambiguous avoided crossings and missed roots remain model limitations.

Static coherent model: no pulse ramps, noise, resonators or explicit coupler.
Calibration adjusts dependent couplings numerically, not a fixed-coupling hardware
waveform. Maxima are finite-search results, not proofs of global optimality.
High-fidelity candidates require truncation convergence checks before certification.

## Source and checkpoint organization

- `xm_cz/model.py`: spectra, isolated pairs, interaction and S1/S2.
- `xm_cz/evaluator.py`: one-point frequency selection and phase calibration.
- `xm_cz/config.py`: public input validation.
- `xm_cz/runner.py`: spawned workers, BLAS limits, checkpoints and resume.
- `xm_cz/plotting.py`: fidelity maps, frequency panels and independent count traces.
- `tests/`: numerical reference, channel symmetry, bounds and process-resume checks.

Each task is atomically checkpointed. Source/input hashes guard resume; changed
source requires a new output directory. The new package intentionally cannot
resume old-package checkpoints. Repeating an unchanged command resumes its own
results; `--retry-failed` retries execution errors. Do not run duplicate writers
against the same output directory. No upload ZIP or automatic Git push is needed.

For a Git change limited to this workflow, review and add `2Q-gate_XM/` only.
Existing changes elsewhere in the checkout belong to other work. The repository's
existing `output/` and `__pycache__/` ignore rules also apply within this folder.

Provenance: numerical search/calibration and runner extracted from the weekend
implementation; independent port validation, channel mapping and namespace added
here. Machine-specific watchdogs, one-off campaigns, unrestricted exploratory
sweeps and generated reports are intentionally not copied into this package.

## Downstream coupler numerical precision (2026-10-29)

The user-approved engineering targets for coupler validation are 1 Hz absolute
for residual interactions and 1e-4 absolute gate infidelity (0.01 percentage
point), comparing the same controls before retuning. Modest adaptive checks stop
on agreement; exhaustive large-basis gate audits are not mandatory. This does
not change the toy's S1/common-g policy, phase-pi requirement, physical acceptance
limits or saved results. The readout workflow retains its own adaptive checks
and physical thresholds. See ../Gnd-coupler+Floating-T/SPEED_PROPOSALS.md.
