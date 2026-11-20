# Dual-rail readout/XM-CZ campaign evolution

> [!summary] Current position - 2026-11-13
> The current design workflow fixes rail A at a validated idle-readout hardware witness, moves rail B to the CZ interaction point, and uses independent idle-readout maps for both rails. A strict saved-data 4D neighborhood check leaves two rail-A projections. The preferred exploratory tuple, `(154, 208, 216, 252)` MHz, was rerun with the corrected cavity provenance (`f_r,A=6.05 GHz`, `f_r,B=5.95 GHz`) and passed gate fidelity plus fixed-hardware reconciliation at its center point.

This note records why the workflow changed, which results remain valid, and where the corresponding data and figures live. It complements [[DEVELOPMENT_MEMORY]], which is the coding-agent authority for implementation invariants.

## Non-negotiable conventions

- Rail A is the lower-physical-frequency rail but uses the higher logical-frequency band, 140-160 MHz.
- Rail B uses the 80-100 MHz logical-frequency band.
- Readout is an idle-point requirement. Gate-operation EJs do not need to satisfy readout.
- Independent readout and gate maps are built separately and overlaid afterward. Readout validity must not prefilter the gate calculation.
- Ordered EC pairs remain distinct in the gate workflow.
- In the fixed-A workflow, rail-A EC, EJ and intra-rail coupling are retained from one complete readout anchor. Rail-B EJ may change between idle and gate operation.
- Gate acceptance requires raw CZ fidelity `>95%` and post-selected fidelity `>99.9%`.
- Aggregate S1 screens unwanted transfer from all four logical inputs. The selected logical source is not hard-coded. Phase-pi calibration and fidelity remain required.
- Full resonator-inclusive validation belongs to the collaborator workflow and is not launched automatically.

## Version history

### V0 - independent readout preparation

Separate fine C1-C2 searches were created for the two logical bands. Mirrored readout treatment was retained because isolated dual rails permit an exact label exchange. Direct EC maps use the union of original and mirrored domains, a 1.5-scaled hull, full orientation, and a 2 MHz EC lattice in the current datasets.

Reasons:

1. Rail A and rail B occupy different logical-frequency bands and must not reuse the same fine C1-C2 map.
2. The scaled hull explores beyond the EC range directly implied by the capacitance map.
3. Symmetry is acceptable for constructing readout targets and warm starts, but not for reducing the downstream ordered gate grid.

Historical preparatory figures are retained in `output/pdf/combined_feasibility_report_figures/`:

![[../output/pdf/combined_feasibility_report_figures/fine_c1_c2_independent_maps.png]]

![[../output/pdf/combined_feasibility_report_figures/rail_a_independent_ec_readout_map.png]]

![[../output/pdf/combined_feasibility_report_figures/rail_b_independent_ec_readout_map.png]]

### V1 - independent gate map, overlay, and per-point reconciliation

The first combined campaign established the central separation of concerns:

1. Build rail-A and rail-B idle-readout maps independently.
2. Build the gate map independently over every ordered rail-B EC target.
3. Overlay the gate and rail-B readout Boolean maps.
4. Reconcile fixed fabrication hardware at every overlay-valid point.

The important correction was that gate tasks must not be restricted to rail-B points already marked readout-valid. Doing so gives the same Boolean intersection for one immutable map, but destroys the independent gate map and prevents inspection outside the current readout region.

The older campaign report is [[../output/pdf/combined_feasibility_campaign_report]]. It describes the variable-bias workflow and is historical rather than the current fixed-A procedure.

### V2 - fixed-rail-A physical-frequency anchors

The design goal changed from retuning both rails to keeping rail A at one applied bias from idle through the gate. At each ordered rail-A EC pair:

1. Scan physical operating frequency with `readout_workflow.physical_frequency_bounds`.
2. Enforce rail-A logical splitting in 140-160 MHz and the established C12/Cg/readout constraints.
3. Find the largest connected validated frequency component.
4. Sample five complete readout witnesses at configurable fractional positions. The automatic production default is `[0.1, 0.3, 0.5, 0.7, 0.9]` within the connected component's interior band.
5. Screen every anchor and all four logical sources at every ordered rail-B EC point.
6. Rank by aggregate S1 and use calibrated-fidelity fallback when a better-S1 anchor fails calibration or fidelity.
7. Keep rail-A EJ fixed; allow rail-B EJ to move.

Why the full witness is an anchor: operating frequency alone is not hardware. Each anchor retains EJ-A1/A2, C12, Cg, logical frequency, bare frequencies, and readout provenance.

Why fallback remains: lower S1 is a useful collision-screening proxy but is not a monotonic guarantee of phase calibration or fidelity. Earlier campaign statistics showed that fallback recovered valid points at modest extra runtime.

The archived report [[../output/pdf/fixed_rail_a_anchor_workflow_report]] describes the early 12-repetition implementation. Its source-specific S-score results predate the restored aggregate-S1 policy and must not be relabeled as current certification.

### V3 - 2 MHz rail-A map and full campaign partition

Rail-A readout was regenerated at 2 MHz resolution to match rail B. The full readout-feasible rail-A domain was partitioned between two workers by ordered orientation. Campaign checkpoints, compact SQLite gate records, and independent readout JSON data were retained so later analyses could reuse results without repeating physics calculations.

Operational lessons:

- Lid closure and power loss interrupt local workers; checkpoint/resume is required.
- Subprocess errors are operational failures and are distinguishable from physical failures.
- A coarse 25 MHz readout-frequency scan can under-sample short connected components. It remains an exploratory default, not an edge-precision certificate.

### V4 - split cavity frequencies and provenance correction

The cavity design was varied to `f_r,A=6.05 GHz` and `f_r,B=5.95 GHz`. New fine and direct-EC readout maps were generated.

![[cavity_split_campaign_2026-11-09/readout/comparison/split_cavity_ec_readout_map_comparison.png]]

An audit later found that the first split-cavity downstream campaign silently reverted to 6.00 GHz in two places:

- Rail-A physical-frequency anchor generation.
- Rail-B fixed-hardware reconciliation.

The independent 6.05/5.95 GHz readout maps themselves were valid. The downstream gate anchors and reconciliation products were not corrected-cavity certification. The implementation was changed to propagate and assert cavity provenance in every applicable stage. New output directories are mandatory after a provenance or physics change.

### V5 - signed idle clearance and campaign-level EC selection

The selection objective was corrected: a small absolute rail-A/rail-B detuning is not intrinsically desirable. The rail-B idle point should sit above all retained CZ interaction frequencies, allowing downward flux tuning into the gate region.

The hard clearance is

\[
\min(f^{idle}_{B1},f^{idle}_{B2})-
\max_{\text{retained logical sources}} f^{CZ}_B > 200\ \text{MHz}.
\]

Ranking then uses fabrication robustness and the rail-B idle-to-gate bare-frequency tuning burden. EJ shifts remain diagnostics; the maximum absolute bare-frequency shift is the preferred tuning metric.

![[combined_overlay_selection_2026-11-12/idle_above_all_interactions_200MHz/reselected_with_signed_clearance.png]]

### V6 - worst-case anchor diagnostic

The first stress-test implementation randomly sampled five validated rail-A anchors from the largest connected component and deliberately selected the anchor with the largest aggregate S1 for fidelity evaluation. That campaign was stopped after 104/556 repetitions when the sampling policy was reconsidered. Its checkpoints remain historical random-anchor diagnostic data and must not be mixed with the replacement campaign.

The replacement diagnostic uses the same fractional positions as the production anchor construction, `[0.1,0.3,0.5,0.7,0.9]`, and then deliberately selects the largest aggregate-S1 anchor. This keeps the sampled physical-frequency positions comparable across rail-A EC candidates while retaining the intended worst-S1 stress test. It uses corrected 6.05/5.95 GHz cavity provenance, skips rail-A EC points that failed the earlier production campaign, performs one fidelity attempt with no fallback, and omits fixed-hardware reconciliation.

This remains a diagnostic only. Production uses the same fractional interior sampling but selects minimum aggregate S1 and retains calibrated-fidelity fallback.

#### Retrospective screening-versus-fidelity audit

Before deciding whether the full worst-case campaign was useful, the retained production databases were audited to answer two questions.

**Does `r_max <= 0.2` guarantee high fidelity? No.**

Across the two retained production halves, 2,626,790 EC-B points had at least one anchor passing the collision screen, including the maximum logical-mixing ratio requirement. Of those, 2,501 points still had no anchor that completed calibration and passed both fidelity thresholds. The conditional failure rate was approximately 0.095%.

The failures include both absence of exact phase-pi calibration and calibrated gates below the raw/post-selected thresholds. For example:

- `EC-A=(196,150) MHz`, `EC-B=(194,248) MHz`;
- all five anchors passed the screening requirements;
- the best calibrated attempt had raw fidelity 85.01% and post-selected fidelity 98.20%;
- the remaining four anchors could not obtain exact phase-pi calibration.

Thus `r_max <= 0.2` is a perturbative local safety screen, not a dynamical fidelity guarantee. Aggregate S1 is a whole-gate screening proxy, while calibrated fidelity remains the final acceptance test.

**Do successful EC-B points always have all five anchors satisfying `r_max <= 0.2`? No.**

Among 2,624,289 fidelity-passing EC-B points:

| Eligible anchors | Successful EC-B points | Fraction |
|---:|---:|---:|
| 5/5 | 2,116,803 | 80.66% |
| 4/5 | 257,688 | 9.82% |
| 3/5 | 153,802 | 5.86% |
| 2/5 | 86,115 | 3.28% |
| 1/5 | 9,881 | 0.38% |

Therefore, 507,486 successful points, or 19.34%, had fewer than five eligible anchors. Every one of the 278 completed rail-A repetitions in the original campaign contained at least one successful EC-B point with fewer than five eligible anchors. The historical split-cavity half has a cavity-provenance caveat, but the original half alone establishes the same conclusion.

#### Interpretation of the continuing diagnostic

These findings mean that the worst-case campaign is not required to establish either screening limitation. The production workflow already performs the necessary fidelity calculation and does not require all five anchors to work. Experimentally, only one fabrication-consistent anchor is needed throughout the selected neighborhood.

The campaign is nevertheless being allowed to continue during otherwise idle weekend compute time as a deliberately conservative sensitivity diagnostic:

- success means even the largest-S1 eligible fractional anchor passes fidelity;
- failure does not invalidate the design when a lower-S1 eligible anchor passes;
- the result must not replace or mask the production minimum-S1 feasibility map;
- the most protocol-relevant future test remains a corrected-cavity rerun at one common anchor inside the neighborhood-shared operating-frequency interval.

### V7 - strict saved-data 4D EC-neighborhood test

The latest analysis constructs tuples

\[
(E_{C,A1},E_{C,A2},E_{C,B1},E_{C,B2})
\]

from retained readout and gate results. No readout or gate physics was rerun. A center is accepted only if every sampled point in its 5 MHz-radius 4D neighborhood satisfies:

- independent idle-readout validity for both rails;
- calibrated gate solution away from frequency/calibration boundaries;
- raw fidelity `>95%` and post-selected fidelity `>99.9%`;
- signed rail-B idle clearance `>200 MHz`;
- a nonempty intersection of rail-A connected operating-frequency intervals;
- every already selected rail-A anchor lies inside that shared interval;
- the selected logical source is constant throughout the neighborhood;
- numerical records are complete rather than missing or operationally failed.

The adjacent gate-frequency-jump restriction was lifted. Gate-frequency span remains a diagnostic because rail B is intentionally tunable.

On the 2 MHz lattice, the 5 MHz 4D ball contains 233 offsets. Its largest sampled norm is 4.899 MHz, while the largest displacement along one axis is 4 MHz.

The strict analysis contains 1,323 valid 4D centers but projects to only two rail-A candidates:

- Original 6.00/6.00 GHz campaign: `(208,154)` MHz, 616 rail-B centers.
- Historical split-cavity campaign: `(154,208)` MHz, 707 rail-B centers.

![[four_dimensional_feasibility_joint_anchor_range_5MHz_no_jump_limit_2026-11-12/four_dimensional_feasibility_projections.png]]

![[four_dimensional_feasibility_joint_anchor_range_5MHz_no_jump_limit_2026-11-12/rail_a_two_surviving_candidates.png]]

Both survivors have an individual connected rail-A range 4.025-4.650 GHz. Across the required neighborhood, the shared interval is 4.025-4.225 GHz, and the stored selected anchor is 4.100 GHz.

The shared width is not unique. Randomly inspected rejected rail-A candidates had shared widths from 125 to 400 MHz, and many rejected candidates had at least 200 MHz. They fail because their independently selected stored anchors are not all inside the shared interval. With existing results, the strict selected-anchor check is the only defensible certification. A nonempty shared interval alone identifies a rerun candidate, not a certified common-frequency gate.

Relaxing the radius to 4 MHz reduces the offset set from 233 to 89 points and increases valid 4D centers from 1,323 to 3,516. This is a real robustness relaxation, so the 5 MHz result remains the current strict result.

## Corrected-cavity center-tuple validation

The preferred exploratory strict projection was `(154,208)` MHz because it had more rail-B centers, equal maximum discrete robustness, better minimum clearance, and slightly lower worst-neighborhood tuning than `(208,154)` MHz. Its preferred historical rail-B coordinate was `(216,252)` MHz.

The exact tuple was rerun in a fresh directory with corrected provenance:

\[
\boxed{(E_{C,A1},E_{C,A2},E_{C,B1},E_{C,B2})=(154,208,216,252)\ \text{MHz}}.
\]

Configuration:

- Rail-A designed bare cavity: 6.05 GHz.
- Rail-B designed bare cavity: 5.95 GHz.
- Corrected rail-A connected readout range: 4.025-4.725 GHz.
- Fractional anchors: 4.100, 4.225, 4.375, 4.525, 4.650 GHz.
- Selected anchor: 4.100 GHz.
- Selected logical source: `10`, coupled to the B20-like `|~0020>` auxiliary.

![[corrected_cavity_strict_tuple_A154_208_B216_252_2026-11-13/readout_frequency_scan/physical_frequency_feasibility.png]]

![[corrected_cavity_strict_tuple_A154_208_B216_252_2026-11-13/gate_output/plots/anchor_screening_s1.png]]

Typical corrected center result:

| Quantity | Value |
|---|---:|
| Raw CZ fidelity | 99.2784% |
| Post-selected CZ fidelity | 99.9899% |
| Acceptance | 99.2885% |
| Gate duration | 98.231 ns |
| Aggregate S1 | 0.002676 |
| Rail-A operating frequency | 4.1000 GHz |
| Rail-B gate frequency | 4.3930 GHz |
| Rail-B idle operating frequency | 4.8651 GHz |
| Rail-B idle bare frequencies | 4.8709, 4.8674 GHz |
| Maximum rail-B bare-frequency shift | 477.91 MHz |
| Fixed-hardware reconciliation | Pass |

![[corrected_cavity_strict_tuple_A154_208_B216_252_2026-11-13/gate_output/plots/selected_anchor_gate_maps.png]]

![[corrected_cavity_strict_tuple_A154_208_B216_252_2026-11-13/fixed_hardware_reconciliation/plots/rail_b_bare_frequency_shifts.png]]

At the gate point, rail B is approximately 293 MHz above the rail-A operating frequency. The closest bare single-transmon separation is approximately 285 MHz, compared with the 5.56 MHz inter-rail coupling. The direct single-excitation detuning is therefore large relative to the coupling in the toy model, while the intended higher-excitation CZ channel remains resonant.

The corrected rerun certifies this center tuple only. It does not replace the historical split-cavity results at every neighboring 4D point, so the complete 5 MHz neighborhood remains exploratory until corrected-cavity neighborhood points are rerun.

## Data and result locations

| Artifact | Location / format |
|---|---|
| Original independent rail-A map | `readout/direct_ec/rail_a_step2/results.json` |
| Original independent rail-B map | `readout/direct_ec/rail_b_step2/results.json` |
| Split-cavity rail-A map | `cavity_split_campaign_2026-11-09/readout/direct_ec/rail_a_step2_fr6p05/results.json` |
| Split-cavity rail-B map | `cavity_split_campaign_2026-11-09/readout/direct_ec/rail_b_step2_fr5p95/results.json` |
| Original full campaign | `sampled_runs/rail_a_gt_partition_2026-11-06/` |
| Split-cavity historical campaign | `cavity_split_campaign_2026-11-09/sampled_runs/rail_a_lt_partition/` |
| Strict 4D summary | `four_dimensional_feasibility_joint_anchor_range_5MHz_no_jump_limit_2026-11-12/summary.json` |
| Strict 4D centers | `.../four_dimensional_feasible_centers.csv` |
| Corrected center rerun | `corrected_cavity_strict_tuple_A154_208_B216_252_2026-11-13/` |
| Corrected compact gate record | `.../gate_output/compact_anchor_results.sqlite` |
| Corrected reconciliation record | `.../fixed_hardware_reconciliation/fixed_hardware_results.sqlite` |
| Corrected compact summary | `.../corrected_tuple_summary.json` |

JSON files can be viewed in any text editor or with `jq`. SQLite files can be inspected using `sqlite3`, DB Browser for SQLite, or Python's `sqlite3` module. PNG/PDF figures are directly viewable; campaign plotting scripts regenerate figures from compact SQLite and JSON records.

## Current interpretation and next steps

1. The exact corrected tuple `(154,208,216,252)` MHz passes the present toy-model gate and fixed-hardware checks.
2. The strict 5 MHz neighborhood conclusion still depends partly on historical split-cavity gate results with the known 6.00 GHz rail-A anchor leak.
3. If neighborhood certification is required, rerun only the 233-offset neighborhood needed around the selected tuple with corrected 6.05/5.95 GHz provenance rather than repeating the entire campaign.
4. Pass a promoted tuple to the collaborator's full-network validator with both resonators, updated S1, dispersiveness checks, and recomputed fidelity.
5. Pulse ramps, decoherence, and flux-line dynamics remain outside this campaign and are covered by the collaborator's separate work.

## Promoted low-coupling tuple and corrected 5 MHz neighborhood

The later calibration-adaptive ranking removed the strict requirements that a
neighborhood use one stored anchor or one CZ source merely because fabrication
parameters change.  It retained independent readout, gate thresholds, signed
clearance, numerical completeness and a complete 233-point 5 MHz 4D ball.  The
first relaxed ranking prioritized radius before absolute required coupling and
therefore proposed `(234,160,240,170)` MHz.  That tuple has a comparatively
large neighborhood-median `J_req/h` of 5.876 MHz.

Three complete tuples were then compared with orthogonal cross-sections: fix
rail A and scan rail B, and separately fix rail B and scan rail A.  The selected
CZ partner and `J_req` were plotted for every stored best-case point, with the
selected coordinate and its 5 MHz rail-plane circle marked.  This avoids the
ambiguity of a 2D projection in which each displayed coordinate can silently
choose a different hidden rail pair.

The comparison promoted

\[
\boxed{(E_{C,A1},E_{C,A2},E_{C,B1},E_{C,B2})
=(158,240,158,226)\ \mathrm{MHz}}.
\]

Reasons:

- corrected center `J_req/h = 4.3668 MHz`, versus 5.556 MHz for the previous
  reference and 5.875 MHz for the radius-first candidate;
- at common fixed `J_hardware`, gate duration scales approximately with
  `J_req`, so the new center needs about 21% less interaction time than the old
  reference at the same hardware coupling;
- its historical 5 MHz neighborhood already showed only about 4.4% relative
  `J_req` spread and one `10 <-> B20` channel;
- the fixed-A and fixed-B cross-sections show a locally smooth region rather
  than the artificial discontinuity suggested by the earlier collapsed
  projection.

The tuple and its complete 5 MHz-radius, 2 MHz-lattice neighborhood were rerun
from scratch using corrected cavity provenance on 2026-11-17:

- rail-A bare cavity: 6.05 GHz;
- rail-B bare cavity: 5.95 GHz;
- 233/233 numerical records present;
- 233/233 pass raw `>95%` and post-selected `>99.9%` fidelity;
- 233/233 pass fixed-hardware reconciliation;
- all 233 select logical source `10` and the `B20`-like `|~0020>` auxiliary;
- `J_req/h = 4.2607-4.4535 MHz`, median 4.3549 MHz, full relative spread 4.425%;
- calibrated gate time `98.154-100.062 ns`, median 99.080 ns;
- raw fidelity `99.2718-99.6680%`;
- post-selected fidelity `99.9940-99.9958%`;
- aggregate S1 `0.003020-0.003424`;
- maximum logical ratio `0.0558-0.0601`, well below 0.2.

The corrected center selects the 4.600 GHz rail-A anchor.  Its raw and
post-selected fidelities are 99.4153% and 99.9954%, and the calibrated hold is
98.792 ns.  Rail B moves from an idle operating frequency of 5.2862 GHz to a
gate operating frequency of 4.8207 GHz.  The two rail-B bare frequencies move
by -469.13 and -465.55 MHz.  Rail A remains at the same applied EJ hardware.

Artifacts:

| Artifact | Location |
|---|---|
| Corrected neighborhood | `corrected_cavity_full_neighborhood_A158_240_B158_226_2026-11-17/` |
| Complete center witness | `.../candidate_analysis/complete_corrected_configuration.json` |
| Neighborhood metrics | `.../candidate_analysis/corrected_neighborhood_metrics.json` |
| Metric distributions | `.../candidate_analysis/corrected_neighborhood_metric_distributions.png` |
| Three-tuple cross-sections | `relaxed_adaptive_4d_best_case_2026-11-17/three_tuple_orthogonal_cross_sections/` |

This is now the preferred toy-model tuple.  The collaborator-owned
resonator-inclusive full-network validation, updated S1/dispersiveness checks,
finite ramps and decoherence remain outside this campaign.

## How the current tuple was selected

This section records the selection sequence that was previously only visible in
the analysis figures and chat discussion.  The current tuple was not selected by
one scalar optimization over a collapsed 2D projection.

### 1. Radius-first candidate was rejected as unnecessarily slow

After removing the requirements for one stored anchor and one CZ source across
the neighborhood, the calibration-adaptive 4D ranking first selected

\[
(234,160,240,170)\ \mathrm{MHz}.
\]

It has a 12 MHz saved-data robustness radius, but its neighborhood-median
`J_req/h` is 5.876 MHz.  Radius was being prioritized ahead of interaction
strength, so this result was treated as a useful robustness reference rather
than the final device choice.

### 2. Three candidates were compared in both rail planes

Three complete tuples were then compared with orthogonal cross-sections.  For
each tuple, one panel fixes rail A and displays the complete rail-B grid; another
fixes rail B and displays the complete rail-A grid.  This makes the selected CZ
source and `J_req` continuity visible without allowing hidden EC coordinates to
change from pixel to pixel.

| Candidate | EC tuple (MHz) | Center CZ source | Center `J_req/h` |
|---|---|---|---:|
| Earlier strict reference | `(154,208,216,252)` | `10 -> B20` | 5.556 MHz |
| Radius-first optimum | `(234,160,240,170)` | `11 -> B20` | 5.875 MHz |
| Upper-left low-J candidate | `(158,240,158,226)` | `10 -> B20` | 4.367 MHz |

![[relaxed_adaptive_4d_best_case_2026-11-17/three_tuple_orthogonal_cross_sections/three_tuple_best_case_A_and_B_cross_sections.png]]

The upper-left search required at least a 10 MHz saved-data robustness radius,
relative neighborhood `J_req` spread no larger than 15%, `EC_A1 <= 170 MHz`,
and `EC_A2 >= 235 MHz`.  Among the 36 centers surviving those conditions,
`(158,240,158,226)` had the lowest neighborhood-median `J_req/h`:

- minimum / median / maximum `J_req/h = 4.261 / 4.365 / 4.454 MHz`;
- relative spread 4.42%;
- minimum signed idle clearance 394.2 MHz;
- minimum raw / post-selected fidelity 99.277% / 99.9942%;
- one `10 -> B20` channel in the retained historical neighborhood.

The choice deliberately accepts a smaller saved-data radius than the
radius-first point in exchange for materially lower required hardware coupling
and therefore a faster gate at common `J_hardware`.  It was promoted only after
the fresh corrected-cavity 233-point neighborhood rerun reproduced the smooth
low-J region and passed every stored gate and reconciliation check.

Data supporting this decision:

| Artifact | Location |
|---|---|
| Relaxed 4D ranking | `relaxed_adaptive_4d_best_case_2026-11-17/summary.json` |
| Upper-left low-J shortlist | `relaxed_adaptive_4d_best_case_2026-11-17/upper_left_low_jreq_candidates.json` |
| Three-candidate values | `relaxed_adaptive_4d_best_case_2026-11-17/three_tuple_orthogonal_cross_sections/summary.json` |
| Corrected promoted neighborhood | `corrected_cavity_full_neighborhood_A158_240_B158_226_2026-11-17/` |

## Matched target-coupling study: g/h = 5 versus 10 MHz

The fixed rail-A pair `(158,240)` MHz was rerun over the same 4,498 ordered
rail-B EC points with corrected cavity provenance (rail A 6.05 GHz, rail B
5.95 GHz).  The only intended gate-setting change was `g/h`, from 5 to
10 MHz.  Independent readout data and the five rail-A anchors were unchanged.

![[g5_vs_g10_A158_240_comparison_2026-11-17/feasibility_and_cz_source_comparison.png]]

Gate-feasibility counts:

- 2,669 points pass at both couplings;
- 1,817 pass only at 5 MHz;
- no point passes only at 10 MHz;
- 12 pass at neither coupling.

After the current fixed-hardware reconciliation:

- 978 pass at both couplings;
- 415 pass only at 5 MHz;
- 2 pass only at 10 MHz;
- 3,103 pass at neither coupling.

On the 2,669 common gate-passing points, the median changes are:

| Quantity | g/h = 5 MHz | g/h = 10 MHz | Change |
|---|---:|---:|---:|
| Calibrated gate time | 99.319 ns | 51.156 ns | -48.259 ns |
| Required `J_req/h` | 4.997 MHz | 9.990 MHz | +4.991 MHz |
| Raw CZ fidelity | 99.535% | 98.263% | -1.250 percentage points |
| Post-selected fidelity | 99.9950% | 99.9339% | -0.0610 percentage points |
| Aggregate S1 | 0.00283 | 0.01131 | approximately 4x |
| Maximum logical `|g/Delta|` | 0.05975 | 0.11966 | approximately 2x |
| Rail-B gate frequency | 4.8763 GHz | 4.8786 GHz | +0.652 MHz median |

![[g5_vs_g10_A158_240_comparison_2026-11-17/pointwise_metric_comparison.png]]

Thus doubling the intended coupling approximately halves the gate time, as
expected, but it does not leave the remaining physics unchanged.  It requires
approximately twice the physical inter-rail coupling, increases spectator
mixing, lowers both fidelity measures, removes a large fraction of the 5 MHz
gate-feasible/reconciled region, and collapses most selected channels onto
`10 -> B20`.  The rail-B collision frequency itself usually changes only
slightly; large outliers coincide with channel or anchor changes.  This study
therefore supports treating 10 MHz as a speed-versus-mixing tradeoff rather than
a drop-in rescaling of the 5 MHz solution.

Artifacts:

| Artifact | Location |
|---|---|
| Matched comparison summary | `g5_vs_g10_A158_240_comparison_2026-11-17/summary.json` |
| Feasibility/source figure | `g5_vs_g10_A158_240_comparison_2026-11-17/feasibility_and_cz_source_comparison.png` |
| Pointwise metric figure | `g5_vs_g10_A158_240_comparison_2026-11-17/pointwise_metric_comparison.png` |
| Corrected 5 MHz campaign | `g5_corrected_cavity_A158_240_2026-11-17/` |
| Corrected 10 MHz campaign | `g10_corrected_cavity_A158_240_2026-11-17/` |

## Rail-B gate-detuning sensitivity of the promoted center

A diagnostic sweep kept the promoted tuple, the selected 4.600 GHz rail-A
anchor and fixed rail-B coupling hardware, then introduced differential bare
detuning `Delta_B = f01_B1 - f01_B2`.  The collision frequency and phase-pi
calibration were recomputed at each detuning.  All tested points from -10 to
+10 MHz passed the two fidelity thresholds.  However, the symmetric center is
at the exact 80 MHz lower logical-frequency limit: +1 and +2 MHz detunings gave
79.9918 and 79.9961 MHz and therefore fail that strict band check despite raw
fidelities above 99.5%.  The tuple is gate-tolerant but has essentially no
margin to the rail-B logical-frequency lower bound.  A future robustness-aware
design should place the nominal rail-B splitting inside the 80-100 MHz band
rather than exactly at 80 MHz.
