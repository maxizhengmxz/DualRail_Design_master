# Codex handoff: fixed-rail-A readout/XM-CZ workflow

This is the entry point for another Codex agent operating this branch. Read
this file and `weekend_simulation/DEVELOPMENT_MEMORY.md` before changing or
running the workflow. The development-memory file is the authoritative record
of physics and implementation decisions; do not reconstruct the workflow from
older scripts or generated results.

## 1. Scope and ownership

The current production workflow fixes rail A at one applied bias between idle
readout and the CZ operating point. It samples complete rail-A readout hardware
witnesses from the largest connected physical-frequency interval, generates an
independent gate map on every ordered rail-B EC target, overlays that map with
the already-generated independent rail-B idle-readout map, and reconciles
fixed hardware separately at every overlay point.

Do not automatically run the resonator-inclusive full-network validation. That
stage is owned by the collaborator and is downstream of this repository's
fixed-hardware reconciliation.

Use the user's Conda environment:

```bash
conda run -n work python --version
```

If `conda run` is inconvenient, the existing environment interpreter on the
original machine is `/opt/anaconda3/envs/work/bin/python`. Do not hard-code
that machine-specific path into tracked source.

## 2. Non-negotiable workflow rules

1. Readout and gate maps are independent. Gate every ordered rail-B EC target;
   never restrict gate tasks to rail-B readout-valid points.
2. Readout is an idle-point calculation. Do not test readout using gate-point
   EJs.
3. Ordered EC pairs are distinct downstream of isolated-readout domain
   construction. Do not restore an `EC1 < EC2` symmetry reduction.
4. Rail A uses one complete readout anchor. Its ECs, EJs and intra-rail
   coupling remain fixed at the gate point.
5. Rail-B EJs and bare frequencies may move between idle and gate. Reconcile
   its fixed fabrication hardware at every overlay point.
6. Evaluate all four logical sources (`00`, `01`, `10`, `11`) for the
   `0020` collision. Do not hard-code a source.
7. Retain ranked fidelity fallback across all five rail-A anchors. Across the
   12-run validation campaign it rescued 196 gate-valid points, 114 of which
   remained valid after readout overlay and fixed-hardware reconciliation.
8. Current strict gate thresholds are raw CZ fidelity `> 0.95` and
   post-selected CZ fidelity `> 0.999`.
9. Produce required plots and a verified compact SQLite database before
   pruning per-point checkpoint JSON.
10. Final design ranking uses a 5 MHz anchor-consistent neighborhood, median
    raw and post-selected fidelity, geometric margin, and maximum absolute
    rail-B bare-frequency shift. EJ shift is diagnostic, not the ranking
    metric.

## 3. Authoritative source files

- `readout_workflow/physical_frequency_bounds.py`
- `2Q-gate_XM/xm_cz/model.py`
- `2Q-gate_XM/xm_cz/evaluator.py`
- `2Q-gate_XM/xm_cz/anchor_evaluator.py`
- `weekend_simulation/prepare_fixed_a_anchor_campaign.py`
- `weekend_simulation/postprocess_anchor_campaign.py`
- `weekend_simulation/reconcile_anchor_campaign.py`
- `weekend_simulation/plot_anchor_reconciled_metrics.py`
- `weekend_simulation/compact_anchor_campaign.py`
- `weekend_simulation/run_anchor_post_pipeline.py`
- `weekend_simulation/run_sampled_anchor_repetition.py`

`combined_feasibility_test/` and the older variable-bias campaigns are
historical. Do not use them as the production algorithm.

## 4. Required external data

Generated readout datasets are intentionally ignored by Git. Before running a
campaign, obtain the files listed in `weekend_simulation/READOUT_DATA_MANIFEST.md`
from the branch owner and preserve their repository-relative paths.

The two direct-EC result files are the minimum runtime inputs:

- rail A, 2 MHz: selects all readout-feasible ordered rail-A repetitions and
  supplies warm-start witnesses for each physical-frequency scan;
- rail B, 2 MHz: supplies all 4,955 ordered gate targets and the independent
  idle-readout map used after gate calculation.

The fine C1-C2 result files are reusable provenance/rebuild inputs. They are
not read by an ordinary fixed-A repetition once the direct-EC maps exist.

Do not commit the readout datasets, gate checkpoints, SQLite result databases,
plots, PDFs, process IDs, or caches. Transfer readout data out of band (shared
storage, `rsync`, or equivalent) and verify SHA-256 checksums.

## 5. Run one ordered rail-A EC repetition

Choose an EC pair that is `validated_pass` in the 2 MHz rail-A direct-EC map.
Use a fresh output directory whenever source or configuration changes.

```bash
conda run -n work python weekend_simulation/run_sampled_anchor_repetition.py \
  --selection-id A_example \
  --ec-a-mhz 170 250 \
  --output weekend_simulation/sampled_runs/example_A_170_250 \
  --workers 4
```

This performs, in order:

1. rail-A physical-frequency/readout scan from 4 to 8 GHz at 25 MHz spacing;
2. largest-connected-component detection and fractional anchor sampling at
   `[0.1, 0.3, 0.5, 0.7, 0.9]`;
3. independent gate evaluation on all 4,955 ordered rail-B EC targets with
   rail-B bare-frequency bounds 4.0-5.5 GHz;
4. independent readout/gate overlay;
5. per-overlay-point fixed-hardware reconciliation;
6. plot generation, compact SQLite verification, then checkpoint pruning.

The script is restartable: completed artifacts are detected and reused. Do not
delete a manifest or compact database merely to force a rerun; use a new output
directory after a source/configuration change.

Important: `run_tonight_first12.py` is a historical 12-selection launcher. It
is not an all-2-MHz-map launcher. For a new multi-repetition campaign, build a
selection list from `status == "validated_pass"` rows in the rail-A 2 MHz map
and launch each ordered pair through `run_sampled_anchor_repetition.py`.

The proven laptop scheduling policy is four simultaneous rail-A repetitions,
each with four gate/reconciliation workers (16 worker processes total). Do not
silently multiply both concurrency levels beyond the available CPU count.

## 6. Outputs and completion checks

For each repetition require:

- `readout_frequency_scan/results.json`
- `anchor_selection.json`
- `rail_A_anchors.json`
- `config.json`
- `gate_output/manifest.json` with `status == "complete"` and 4,955 completed
  tasks
- `gate_output/anchor_campaign_summary.json`
- `gate_output/compact_anchor_results.sqlite`
- `fixed_hardware_reconciliation/summary.json`
- `fixed_hardware_reconciliation/fixed_hardware_results.sqlite`
- `fixed_hardware_reconciliation/metric_plot_summary.json`
- the gate, overlay, reconciliation, fidelity, selected-anchor, and frequency-
  shift plots generated before pruning

Treat a repetition as failed or incomplete if the gate manifest is short, the
SQLite row count does not match the completed task count, reconciliation was
not run, or plot generation failed.

## 7. Validation before handing off source changes

Run at least:

```bash
PYTHONPATH=2Q-gate_XM conda run -n work python -m unittest discover -s 2Q-gate_XM/tests -v
conda run -n work python -m unittest discover -s readout_workflow/tests -v
conda run -n work python -m ruff check 2Q-gate_XM readout_workflow weekend_simulation
conda run -n work python -m ruff format --check 2Q-gate_XM readout_workflow weekend_simulation
git diff --check
```

Do not launch a production grid as a source-validation step. Use unit tests and
a small smoke calculation in a new ignored output directory.

## 8. Git discipline

- Preserve unrelated user changes in the dirty worktree.
- Keep generated data ignored and out of commits.
- Do not create upload ZIP files.
- Check `git status`, staged versus unstaged changes, and `git diff --check`
  before asking the user to commit or push.
- The user owns the final Git commit and push.
