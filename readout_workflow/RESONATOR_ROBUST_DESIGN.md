# Resonator-frequency readout design

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

`resonator_design_campaign.py` schedules the strict EC1>EC2 B half-plane with
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

The coupler-design workflow selects prescribed-range optimization by default,
preferring H>96% throughout the range, with an explicitly reported H>95%
fallback over the same unchanged interval. Its four A/B interval bounds remain
independent inputs to request when an actual design starts; no example range,
equal-center or equal-width constraint is imposed.

This CLI implements the three-node 6 GHz-reference formulation above; its
seed/reference checks remain intentional and active. The coupler's separate
`Gnd-coupler+Floating-T/floating_coupler/readout_range.py` adapter now applies the
same prescribed-range objective to actual pads, using `device_at_frequency`
from this workflow to reconstruct each scenario with fixed physical capacitors.
It supports each rail's own nominal LC reference and range and returns the
optimized nominal pad model plus audited range witnesses to seven-mode assembly.
Only EJs recalibrate within a candidate's interval; no scenario self-capacitance
compensation occurs. New nominal-only optimization records remain rejected as
range certification. Do not replace physical pad capacitors with effective
modal capacitors, silently assume 6 GHz, or relabel historical results.

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
