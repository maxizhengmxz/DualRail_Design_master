# Combined readout/gate workflow: developer memory

This file is the authoritative development handoff for the combined dual-rail
readout/XM-CZ work. Read it before modifying `Ec_tuple search workflow/`,
`readout_workflow/physical_frequency_bounds.py`, or the fixed-A paths in
`2Q-gate_XM/`.

It exists because several superficially reasonable shortcuts are **not** the
workflow requested by the user. Do not reconstruct the workflow from memory or
from an older script. Check this document and the current code first.

## 1. Authoritative workflows and obsolete predecessors

### Current fixed-rail-A workflow

The current workflow is the physical-frequency-anchor campaign implemented by:

- `readout_workflow/physical_frequency_bounds.py`
- `2Q-gate_XM/xm_cz/model.py`
- `2Q-gate_XM/xm_cz/evaluator.py`
- `2Q-gate_XM/xm_cz/anchor_evaluator.py`
- `Ec_tuple search workflow/prepare_fixed_a_anchor_campaign.py`
- `Ec_tuple search workflow/postprocess_anchor_campaign.py`
- `Ec_tuple search workflow/reconcile_anchor_campaign.py`
- `Ec_tuple search workflow/plot_anchor_reconciled_metrics.py`
- `Ec_tuple search workflow/run_anchor_post_pipeline.py`
- `Ec_tuple search workflow/compact_anchor_campaign.py`
- `Ec_tuple search workflow/run_sampled_anchor_repetition.py`

The 12-run example campaign is under:

`Ec_tuple search workflow/sampled_runs/tonight_first12_B4_5p5/`

### Earlier variable-bias workflow

The earlier production campaign allowed both rail-A and rail-B EJs to change
between idle and gate points. It is useful historical code, but it is not the
default for new fixed-rail-A studies.

### Test workflow

`combined_feasibility_test/` is an earlier test run. Do not copy its strategy
into production code. It contains useful plots and historical experiments but
is not authoritative.

## 2. Non-negotiable physics/workflow invariants

### 2.1 Readout and gate maps are independent

When regenerating the independent readout inputs, the EC-tuple search default is the
union of original and mirrored readout search domains. Use direct_ec_feasibility
with --domain hull --orientation-domain full --hull-scale 1.5. This constructs
targets and swapped warm starts only. Recalculate and validate every target.
The ordinary readout driver's reduced domain and explicit EC rectangles remain
available for other studies. Never reduce the downstream gate grid by symmetry.

The independent rail-B gate map is evaluated on **every ordered rail-B EC
target** in:

`Ec_tuple search workflow/readout/direct_ec/rail_b_step2/results.json`

There are currently 4,955 ordered targets. Their readout status must not be used
to restrict gate calculation.

Why: idle readout and gate operation may use different EJs. The final Boolean
overlay is meaningful only when both maps were constructed independently.

The option `prepare_gate.py --readout-feasible-only` is diagnostic-only. Never
use it for a production feasibility map.

### 2.2 Readout is checked at idle, not at the gate point

Do not ask whether a gate-valid point also satisfies readout using its gate EJs.
The readout witness is an idle-point witness. The gate result is a separate
operating point. Hardware consistency is checked afterward by reconciliation.

### 2.3 Ordered EC points remain distinct in the gate workflow

Isolated readout permits exact label swapping and may use mirrored treatment.
The gate model does not assume this symmetry. `(EC1, EC2)` and `(EC2, EC1)` are
separate repetitions/points downstream of readout-domain construction.

### 2.4 Reconciliation is per overlay point

Do not search for one compatible realization and treat it as certification of
the region. Every fidelity-valid/readout-valid EC-B point must be reconciled.

The output is a final per-point feasibility map. A single preferred design is
chosen only in a later ranking stage.

### 2.5 Fabrication hardware versus operating bias

General workflow:

- EC and physical capacitance/coupling implementation are fabrication-fixed.
- Readout and gate EJs may differ.
- Readout resonator hardware belongs to the idle witness.

Current fixed-rail-A workflow:

- Rail-A EC, EJ and intra-rail coupling are fixed from one complete readout
  anchor and are passed unchanged to the gate model.
- Rail-A applied bias must not change between idle and gate operation.
- Rail-B EC and intra-rail coupling implementation are reconciled, while rail-B
  EJ is allowed to move between idle and gate.
- Compare bare transmon frequencies when reporting applied-bias motion.
  Dressed frequencies can move when the coupled model changes even if EJ is
  fixed.

### 2.6 Full-network validation is not automatic

The resonator-inclusive two-rail full-network check is owned by the user's
colleague. Do not silently launch it after reconciliation. The collaborator's
stage should include both resonators, update S1, recheck dispersiveness and
recompute the gate result.

## 3. Current fixed-rail-A anchor algorithm

For one ordered `(EC_A1, EC_A2)` repetition:

1. Run `readout_workflow.physical_frequency_bounds` at fixed EC-A.
2. Search for an existential readout witness at each requested rail-A physical
   operating frequency while retaining:
   - logical splitting in 140-160 MHz;
   - established sensible C12 and Cg ranges;
   - the established readout/dispersive/hybridization/assignment policy.
3. Split the validated samples into connected components.
4. Select the largest connected component.
5. Default anchor sampling uses user-configurable fractional positions. The
   current automatic default is `[0.1, 0.3, 0.5, 0.7, 0.9]`.
   Actual sampled anchors must stay inside the inner 10-90% of that largest
   sampled component. Nearest-grid rounding and duplicate handling may not add
   endpoint anchors. If fewer than five distinct validated interior samples
   exist, refine the frequency scan. Save requested and actual fractions.
6. Each selected frequency must already have a validated complete readout
   witness. Store the complete witness once in `rail_A_anchors.json`.
7. Put only the lightweight fields used by the gate evaluator in `config.json`.
   Never duplicate `complete_readout_witness` into 4,955 task checkpoints.
8. For every ordered EC-B target, screen every anchor.
9. At each anchor, solve collision candidates for all four logical sources.
10. Rank anchors using the screening policy, then run exact-pi calibration with
    ranked fallback. If all candidates fail configured S-score requirements,
    skip fidelity work and mark the point failed.
    A remains fixed at its interior readout anchor throughout gate calibration.
    Cached anchor configurations that differ from the interior selection require
    a fresh output directory, rather than silently reusing edge anchors.
11. Build the independent gate map.
12. Overlay it with the already-generated independent rail-B idle readout map.
13. Reconcile fixed hardware independently at every overlay point.
14. Generate plots before compacting/pruning detailed point JSON.

Tonight's orchestration scripts use a 25 MHz coarse physical-frequency scan.
That is appropriate for exploration, not a final edge-precision claim. A
fabrication candidate can receive a finer local scan later.

## 4. Logical-source and S-score contract

Updated by the user on 2026-11-05. Choose the cleanest gate across all four
logical inputs. Allow the target input's intentional auxiliary excursion,
while suppressing unwanted transfer from it and the other logical inputs.
The other inputs can accumulate phases. Optimizing just the target's unwanted
coupling can hide spectator mixing. Aggregate S1 is a screening proxy for this
whole-gate objective. Phase-pi calibration and fidelity check the actual return.
The shared authority is ../2Q-gate_XM/WORKFLOW.md and core.cz_channel_policy.

Never hard-code the logical state coupled to the auxiliary.

For each physical-frequency/EC candidate, check all four logical inputs:

`00, 01, 10, 11`

For a proposed source, the intended source-to-auxiliary edge is excluded from
its unwanted-coupling score. For every remaining coupled destination `k`, use
the squared perturbative ratio:

`|J * <source|n_A2 n_B1|k> / (E_source - E_k)|^2`

The source-specific score is the sum of these unwanted terms. `S1` is the mean
of the four logical-origin scores. `S2` is the auxiliary-origin score and stays
separate.

Selection sequence in `evaluator.py`:

1. Find valid collision candidates per logical source.
2. Keep the lowest aggregate S1 candidate for each allowed source.
3. Choose across sources using shared minimum-aggregate-S1 selection at common
   intended g and in the same score basis. Record the selected channel and its
   comparison metadata. Source-specific scores and S2 remain diagnostics.
4. The anchor evaluator screens every rail-A anchor and ranks by aggregate S1.
5. Full fidelity calibration uses ranked fallback; it does not assume that the
   smallest S-score guarantees the best fidelity.

The optional maximum_source_score remains an explicit extra screening ceiling
when supplied. It does not rank resonances or break S1 ties. The default ceiling
is inactive. Source restrictions supplied through logical_sources remain
explicit and are respected. The selected source is preserved during phase/time
calibration at each anchor. Do not silently switch it to improve return.

Earlier campaign numbers and artifacts in this document used the superseded
source-specific resonance policy. They remain historical observations. A fresh
output directory and new calculation are required before claiming results for
the updated policy. Never relabel old selections as aggregate-S1 certification.

Physical occupation conventions:

- `0020` means the B20-like auxiliary.
- `0200` means the A02-like auxiliary, not A20.

Keep explicit physical order `A1, A2, B1, B2`.

## 5. Gate calibration contract

### Fixed B logical-splitting approximation

The EC-tuple gate search fixes the isolated B-rail logical splitting at
0.080 GHz. This is a deliberate reduction of the search dimension: the
readout-map study identified 0.080 GHz as the most likely feasible splitting,
and the user reports that independent checks found gate fidelity was not
strongly dependent on the logical splittings over the tested cases. Keeping
this representative value avoids an additional logical-splitting sweep and
reduces calculation cost. It is not an optimized output of each gate task or
a claim of splitting independence outside the tested cases. The independent
B idle-readout acceptance window remains 0.080--0.100 GHz.

Rationale confirmed by the user on 2026-11-05. The implementation default is
in `prepare_fixed_a_anchor_campaign.py`; no numerical setting changed when
this explanation was recorded.

- Energies and matrix elements are cyclic GHz; time is ns.
- Rail A is fixed when a `fixed_A` witness is supplied.
- The gate search adjusts rail-B common bare frequency to solve the selected
  logical-to-auxiliary collision.
- Calibration varies rail-B frequency locally and interaction duration to
  obtain total conditional phase pi.
- The configured intended dressed matrix element is `g_GHz`, normally 0.005.
- The charge-coupling coefficient implementing it is a separate recorded value.
- Raw and post-selected fidelities must refer to the same calibrated gate.
- Current feasibility thresholds are strict:
  - raw CZ fidelity > 0.95;
  - post-selected CZ fidelity > 0.999.

The model is a static coherent hold. It has no ramps, decoherence, resonators or
explicit tunable coupler. Do not describe a static time-evolution plot as a
complete pulse-level verification.

## 6. Reconciliation meaning

The gate map and readout map are first overlaid as Boolean maps. Reconciliation
then asks whether the gate-required rail coupling and the idle readout
constraints admit one consistent fabricated rail-B realization at that exact
EC-B point.

For the fixed-A workflow:

- Rail A is audited against the chosen anchor. EJ and J must match the anchor;
  it is not reoptimized.
- Rail B readout optimization is performed at idle subject to the gate-required
  intra-rail coupling target and fixed EC-B.
- Rail-B EJ may differ from the gate EJ.
- The reconciled record retains the gate result, selected rail-A anchor and
  complete rail-B idle readout witness.

Do not reopen C12 and Cg as unconstrained unrelated variables and then claim
hardware consistency. The reconciler must use the gate coupling requirement
and the established readout hardware ranges.

## 7. Final design ranking

Ranking is a postprocessing decision, not part of gate-map construction.

Current agreed policy:

1. Keep only reconciled points passing both fidelity thresholds.
2. At each candidate center, require a circular 5 MHz neighborhood on the
   ordered 2 MHz EC-B lattice. A fully sampled interior disk has 21 points.
3. Define the **anchor-consistent robustness margin** as the Euclidean distance
   to the nearest lattice point that either:
   - is outside the fidelity-passing reconciled region; or
   - selects a different rail-A physical-frequency anchor.
   The sampled-domain exterior is also outside the admissible region.
4. Prefer candidates with anchor-consistent margin >=5 MHz. This guarantees
   that the 5 MHz neighborhood is feasible and uses one rail-A anchor.
5. Calculate neighborhood mean, median, minimum and standard deviation for raw
   and post-selected fidelity. Plot the neighborhood medians, not the center
   fidelities.
6. Inside the robust pool, minimize the worst tuning burden in the full
   neighborhood:

   `max over neighborhood(max(|delta bare_f_B1|, |delta bare_f_B2|))`

7. Tie-break with higher neighborhood median raw fidelity, then higher
   neighborhood median post-selected fidelity, then larger anchor-consistent
   margin. Mean, minimum and standard deviation remain diagnostics only; use
   median consistently for both fidelity types in selection and summary maps.
8. If no point reaches 5 MHz, report the best available fallback explicitly as
   low-margin. Do not label it equally robust or use its undersized neighborhood
   statistics as though they represented a complete 5 MHz disk.

Do not use EJ shift as the primary metric. It may remain a diagnostic.

## 8. Data contracts and retention

Per anchor repetition:

- `readout_frequency_scan/results.json`: frequency scan records.
- `anchor_selection.json`: connected component and selected frequencies.
- `rail_A_anchors.json`: complete readout witnesses, stored once.
- `config.json`: lightweight gate task defaults and 4,955 EC-B tasks.
- `gate_output/manifest.json`: runner state.
- `gate_output/anchor_campaign_summary.json`: gate/overlay counts.
- `gate_output/compact_anchor_results.sqlite`: compact full gate results.
- `fixed_hardware_reconciliation/fixed_hardware_results.sqlite`: per-point
  reconciliation records.
- `fixed_hardware_reconciliation/metric_plot_summary.json`: derived plot data.

Retention order:

1. Complete gate tasks and verify the manifest count.
2. Generate gate, overlay, reconciliation and metric plots.
3. Write and verify the compact SQLite database.
4. Only then prune bulky per-point JSON.

Do not discard intermediate data before required plots are made. Do not put the
complete readout witness into every task payload.

## 9. Known wrong approaches: do not regress

1. **Wrong:** restrict the gate grid to rail-B readout-valid points.
   **Correct:** gate all 4,955 ordered EC-B targets, then overlay.
2. **Wrong:** test readout at gate EJs.
   **Correct:** readout is an idle witness; reconcile afterward.
3. **Wrong:** reconcile one example and infer the whole region.
   **Correct:** reconcile every overlay-valid point.
4. **Wrong:** hard-code the logical source for `0020` or `0200`.
   **Correct:** evaluate all four sources dynamically.
5. **Wrong:** select only the lowest-S anchor and stop even if fidelity fails.
   **Correct:** ranked fidelity fallback; optionally pre-reject all anchors if
   they fail explicit S-score requirements.
6. **Wrong:** when expanding a search range, discard candidates found in the
   narrower range.
   **Correct:** preserve/pool all passing candidates before reconciliation and
   ranking.
7. **Wrong:** assume readout label symmetry makes the ordered gate map symmetric.
   **Correct:** keep both EC orders in the gate campaign.
8. **Wrong:** interpret the readout dressed operating frequency as exactly the
   bare transmon frequency.
   **Correct:** store/report bare `f01` when available; the operating frequency
   is only a representative anchor label.
9. **Wrong:** require the dressed rail-A frequency to be numerically identical
   after coupling is enabled.
   **Correct:** fixed EJ means fixed applied bias/bare frequency; dressed values
   may shift.
10. **Wrong:** call the reconciled static model the final full-network result.
    **Correct:** the resonator-inclusive validator is a separate collaborator
    stage.

## 10. Testing and execution environment

Use the user's conda environment:

`/opt/anaconda3/envs/work/bin/python`

Gate tests:

```text
PYTHONPATH=2Q-gate_XM /opt/anaconda3/envs/work/bin/python \
  -m unittest discover -s 2Q-gate_XM/tests -v
```

Readout tests:

```text
/opt/anaconda3/envs/work/bin/python \
  -m unittest discover -s readout_workflow/tests -v
```

Important runner rules:

- Source hashes guard resume.
- Source changes require a new output directory.
- Never run two writers against the same output directory.
- A completed/compacted campaign should be analyzed from SQLite rather than
  regenerated unless the physics/source changed.

## 11. Current reference artifacts

- Independent readout inputs:
  `Ec_tuple search workflow/readout/`
- Current 12-run campaign:
  `Ec_tuple search workflow/sampled_runs/tonight_first12_B4_5p5/`
- Final selections:
  `.../best_configuration_selection/best_configurations.json`
- Detailed A4 validation:
  `.../best_configuration_selection/A4_detailed_validation/`
- Report:
  `output/pdf/fixed_rail_a_anchor_workflow_report.pdf`

The A4 analysis script reconstructs the exact 256-state static Hamiltonian and
checks the archived fidelity to numerical precision. Its current detailed
artifact refers to the earlier center-value/10 MHz selection at EC-B=(166,258)
MHz. The anchor-consistent 5 MHz neighborhood policy later selected
EC-B=(266,162) MHz for A4. Regenerate the detailed A4 artifact and report before
presenting them as the current selected design.

## 12. Before changing this workflow

1. Read this file and `2Q-gate_XM/AGENTS.md`.
2. Inspect the current code; do not infer behavior from an old chat summary.
3. State whether the requested change affects readout, gate construction,
   overlay, reconciliation, ranking or full-network validation.
4. Preserve independent-map semantics unless the user explicitly changes them.
5. Add or update a focused test for any physics/data-contract change.
6. Use a new output directory for changed source/configuration.
7. Update this memory file when the user changes a workflow invariant.
