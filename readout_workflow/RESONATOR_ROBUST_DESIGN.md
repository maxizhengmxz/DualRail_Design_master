# Resonator-frequency readout design

This is the optional capacitor-design/range optimization workflow. For the
distinct fixed-design tolerance study and its entry points, see
[RESONATOR_FREQUENCY_STUDIES.md](RESONATOR_FREQUENCY_STUDIES.md).

The standard opt-in entry point is `resonator_robust_design.py`. Opening or
importing the module never launches work. Existing numerical datasets and
reports are historical records; changing this policy does not relabel them.

## Exploration: no prescribed resonator range

`--mode tolerance` maps maximum symmetric resonator-frequency tolerance around
6 GHz versus the nominal loaded EC tuple. Optimize C12 and common C1r=C2r;
calculate C1/C2 to preserve that tuple at 6 GHz / 50 Ohm. A candidate shares
those five capacitors across the whole range. Cr/Lr vary at 50 Ohm; both EJs
and operation frequency can recalibrate. Loaded ECs are not compensated as the
resonator changes.

Chi and H are acceptance constraints, NOT secondary objectives: require the
source chi strength floor (currently 1 MHz), mismatch <=1%, and source H floor
(currently 95%), together with all other source conditions. Do not maximize
chi after finding the range. Record the achieved worst chi and H as diagnostics.
Do not use the incidental chi of a starting point as an extra validation floor.

Use geometric width bracketing, first-feasible starts, warm starts, and final
full-grid validation. Extra starts rescue failures only. The compatibility
flag --quick-tolerance is accepted, but this is now the default tolerance
algorithm. The width ceiling is explicitly censored if reached. Only a
validated sampled interval is reported; endpoints alone do not certify it.

`resonator_design_campaign.py` schedules one strict half-plane (default EC2>EC1) with
four workers and resumable per-point checkpoints. --ec-step-mhz accepts
multiples of 2 MHz. An 8 MHz lattice anchored at 150 MHz has 100 source-feasible
points. Start with the coarse lattice; add finer original-grid points later.

## Prescribed resonator range

`--mode prescribed --resonator-range-ghz LOW HIGH` fixes a required interval
containing 6 GHz. Optimize the WORST absolute signed-mean chi across that
interval, with independent EJ recalibration and ONE shared capacitor design.
`--h-targets` specifies minimum H at every frequency. Solve for multiple H
targets to expose the chi/H tradeoff rather than using arbitrary weights.
The default targets are the acceptance floor, 97% and 98% (excluding targets
below the supplied acceptance floor). No claim of globally maximal H is made.

Final validation preserves the optimized chi floor within 1 kHz numerical
allowance, in addition to the original 1 MHz floor and each requested H target.

### Explicit-pad coupler integration status

The coupler-design workflow selects prescribed-range validation by default,
preferring H>96% throughout the range, with an explicitly reported H>95%
fallback over the same unchanged interval. Its four A/B interval bounds remain
independent inputs to request when an actual design starts; no example range,
equal-center or equal-width constraint is imposed.

The explicit-pad coupler adapter defaults to **minimum physical intra-pair
capacitance**, on BOTH rails, subject to the full readout constraints and
prescribed chi floor (normally 1 MHz). It reports each readout-link capacitor
because the two capacitor sizes trade off. Explicit alternatives are
`minimize_readout_link` and `maximize_chi`. This is a coupler synthesis policy;
the independent three-node CLI's prescribed-mode chi objective and tolerance-
mode range objective above remain separate. Historical records retain their
original objective, including the selected A pair-minimization precursor and
its B chi-maximization precursor; they are not relabelled as unified designs.

This CLI implements the three-node 6 GHz-reference formulation above; its
seed/reference checks remain intentional and active. The coupler's separate
`Gnd-coupler+Floating-T/floating_coupler/readout_range.py` adapter now applies the
prescribed-range constraints and the selected synthesis objective to actual pads, using `device_at_frequency`
from this workflow to reconstruct each scenario with fixed physical capacitors.
It supports each rail's own nominal LC reference and range and returns the
optimized nominal pad model plus audited range witnesses to seven-mode assembly.
Only EJs recalibrate within a candidate's interval; no scenario self-capacitance
compensation occurs. New nominal-only optimization records remain rejected as
range certification. Do not replace physical pad capacitors with effective
modal capacitors, silently assume 6 GHz, or relabel historical results.

The explicit-pad adapter also supports an opt-in
`dressed_nominal_with_range` splitting policy. It pins the dressed logical
splitting at a supplied bare-resonator reference and enforces the supplied
dressed-splitting interval at every range scenario. The default remains the
exact isolated-pair fit; this option does not change the independent three-node
CLI or any readout acceptance threshold.

## Common constraints and validation

Supply the conditions JSON belonging to the rail. Do not infer another
rail's defaults. Current B conditions: logical 80-100 MHz, chi ratio <=1%,
mean absolute chi >=1 MHz, H>=95%, multilevel dispersive ratio <=0.1,
assignment >=95%, EJ/h<25 GHz, EJ/EC>50, loaded EC/h>=150 and <400 MHz,
operation 4-8 GHz below dressed resonator, independent +/-1% EJ sign and
assignment check on the existing 5x5 grid. Nominal C1/C2 synthesis retains
36-128 fF and original C12/Cg bounds. The EC lower bound is inclusive; numerical
interior guards must not reject nominal EC=150 MHz.

Audit endpoint and nominal points, half-step samples and adjacent optimized-
frequency midpoints using the original adaptive quantum bases and sign checks.
Failed interior points trigger repair or range reduction. This is finite-grid
coverage, not a continuum proof or global optimization certificate. No physical
constraint or final validation is relaxed to achieve the speedup.

Save settings, input/source hashes, source snapshots, optimizer attempts,
nominal candidate, frequency witnesses, achieved chi/H and completion status.
Resume only identical source/settings. --diagnose-tolerance optionally rechecks
full fixed/flexible-operation intervals under original acceptance conditions.
It is not launched automatically. This workflow is not capacitor-error Monte
Carlo and does not launch gate or CZ calculations.

## Command templates (calculations only on request)

```powershell
python -m readout_workflow.resonator_robust_design --run --mode tolerance --ec 200 238 --input NOMINAL.json --conditions SETTINGS.json --output NEW_OUTPUT
python -m readout_workflow.resonator_robust_design --run --mode prescribed --ec 200 238 --resonator-range-ghz 5.9 6.1 --h-targets 0.95 0.97 0.98 --input NOMINAL.json --conditions SETTINGS.json --output NEW_OUTPUT
```

The completed 100-point run included secondary chi maximization and took
66 minutes. Its measured phase split suggests 25-40 minutes for a comparable
exploration-only pass; that full-run projection has not yet been benchmarked.
It remains unchanged under B_EC_robust_design_step8_quick_20261110. The older
1,488-point campaign remains stopped with all checkpoints retained.

## Push / transfer checklist

Commit the new calculation modules, their tests, this guide and the README
link. Generated data, PDFs, caches and source snapshots remain ignored and
are not included in a code push. The campaign accepts --source and --conditions
for transferred datasets and uses the current Python interpreter on all platforms;
the local historical dataset is only its default. Single-design runs require
only a nominal input JSON and its original conditions JSON.
The separate nominal-only `ec_design_improvement.py` helper also accepts
`--source DIRECTORY` for transferred settings/results; its historical data
directory is merely the backward-compatible default, not a required location.

Resume checks include the conditions-file hash and shared physics source hashes.
Formatting or changing source therefore requires a fresh output directory, or
the exact saved source snapshot for a historical resume. Completed numerical
results retain their original settings and have not been relabeled as the newer
exploration-only policy. No script starts calculations without --run.

The individual workflow defaults are three starts (extra starts only after
failure in exploration), 50 MHz scenario spacing and 25 MHz interior validation
spacing. Its boundary resolution defaults to 6.25 MHz. The campaign defaults
to the requested coarse budget: four workers, an 8 MHz EC lattice, 25 MHz
boundary resolution and 50 optimizer iterations. Use --ec-step-mhz and
--boundary-resolution-ghz to refine explicitly. H is inherited from the
conditions file, not hardcoded by the campaign.
The default range-search halfwidth ceiling is 1 GHz and is explicitly censored
if reached. These numerical budgets are configurable; the physical acceptance
limits come from the supplied conditions file.

## EC-tuple-anchored cavity tolerance

`anchored_resonator_tolerance.py` is the separate operating-point-preserving
study. It must not be confused with the existential `--mode tolerance` search
above. It requires each rail's own nominal bare cavity frequency, exact nominal
bare transmon frequencies, loaded EC tuple and selected fixed readout witness.
At the nominal cavity it reuses those transmon frequencies exactly. Away from
nominal it walks outward on one connected branch, constraining each bare
frequency to its independently centered tuning budget. It reports the frozen
witness and capacitor redesign separately, checks the requested asymmetric
interval, and brackets the maximum symmetric finite-grid interval. A result
that reaches the configured cavity ceiling is censored, not an exact maximum.
Widening a bare-frequency budget may reuse a complete ceiling-censored
certificate from a smaller budget because the identical witness remains
feasible. Existing unanchored tolerance campaigns and their records are not
relabelled.

## Coarse-to-fine refinement

Use --refine-from COARSE_OUTPUT with a different --output directory and a
smaller --boundary-resolution-ghz. This applies to tolerance mode only.
The original run must be complete. Nominal EC tuple, physical acceptance,
50 Ohm/6 GHz reference, frequency-validation grid and range ceiling must
match. Changes to recorded physical equations are rejected; formatting-only
changes and optimizer-code updates do not invalidate an otherwise compatible
certificate. Historical runs that did not snapshot some extra, inactive core
modules record that provenance limitation explicitly.

The implementation reads the saved passing halfwidth, failed-search bracket,
validated capacitor design and calibrated EJs. It performs only midpoint
trials inside that bracket, warm-started from the saved design. Any newly
accepted range receives the normal full sampled validation. If trials fail,
the old certified range is retained. Interrupted refinement trials are also
checkpointed. If the bracket already meets the requested precision, no new
optimization or quantum validation is run. A ceiling-censored source remains
censored; extending the search ceiling is a new search, not refinement.

Source coarse files are read-only. Refined outputs record their hashes,
physics compatibility checks, reused witnesses, and new/cached trial counts.
25 to 6.25 MHz normally requires only two extra bisection levels, plus
validation of any newly accepted design. It does not repeat the full width
expansion. Search non-discovery remains a numerical bound, not a proof.

```powershell
python -m readout_workflow.resonator_robust_design --run --mode tolerance --ec 238 198 --input NOMINAL.json --conditions SETTINGS.json --refine-from COARSE_POINT_OUTPUT --boundary-resolution-ghz 0.00625 --output NEW_REFINED_POINT_OUTPUT
python -m readout_workflow.resonator_design_campaign --run --workers 4 --ec-step-mhz 2 --source SOURCE_RESULTS.json --conditions SETTINGS.json --refine-from COARSE_CAMPAIGN_OUTPUT --boundary-resolution-ghz 0.00625 --output NEW_REFINED_CAMPAIGN_OUTPUT
```

For a campaign, only matching completed coarse tuples are refined. Missing
EC tuples receive a fresh normal search. Thus upgrading both the EC grid and
the boundary precision does require full searches at newly added EC points;
only the already-computed tuples avoid the coarse-search cost. The source
campaign fingerprints are included in resume identity.

## Sequential full B / A run

`resonator_pair_sequence.py` runs B at 6.25 MHz boundary resolution, then A
at 25 MHz, then refines those A results to 6.25 MHz. All stages use the 2 MHz
EC lattice, EC2>EC1, four workers and exploration-only objectives. Rail-specific
conditions are snapshotted independently: B logical 80-100 MHz, A 140-160 MHz.
Current source data provide 1,488 unique B pairs and 385 unique A pairs.
A label-swapped starting seed fills one missing orientation per source; all
final results and certificates are recalculated, never copied by symmetry.

The sequence saves immutable input copies, source snapshots, stage commands,
source/input fingerprints, a duplicate-scheduler lock, live progress, stage
exit records and all per-tuple checkpoints. It will not advance past errors
or changed source. A completed stage is skipped safely on resume.

Creating PAUSE_REQUESTED in the sequence output stops further scheduling;
active tuples finish their checkpoints before the pipeline pauses. On explicit
resume, remove that file and restart the same sequence command. After a forced
restart, stale locks are checked against live process IDs before reuse. Source
changes require the saved exact version or a new output, never silent mixing.
