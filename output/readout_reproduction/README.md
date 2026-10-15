# Dual-rail readout reproduction

Source: **Archived study, 14-15 October 2026**.

## Conditions

- Bare resonator: 6 GHz; equivalent-mode impedance 50 ohm.
- Nominal conditions: `{"mismatch_max": 0.01, "dispersive_max": 0.1, "hybridization_min": 0.95, "reference_overlap_min": 0.95, "operation_frequency_ghz": [4.0, 8.0], "logical_frequency_ghz": [0.09, 0.11], "loaded_ec_mhz": [150.0, 400.0], "ej_upper_ghz": 25.0, "ej_over_ec_min": 50.0, "c12_bounds_ff": [0.005, 15.0], "readout_cap_bounds_ff": [0.1, 25.0], "fixed_frequency_tolerance_ghz": 1e-07}`.
- Only C1>C2; uniform 1 fF pixel centers from 36 to 128 fF.
- Each nominal pixel can choose both EJs, C12 and common C1r=C2r.
- 90% participation is an initial estimate, not an enforced equality.
- Operation frequency means the mean dressed logical transitions; logical frequency means their separation.
- H measures charge-transition balance. Ideal bare-state populations are H/2 and 1-H/2.
- H=95% means 47.5%/52.5% populations only in the ideal two-state limit; H is not Bell fidelity.
- Cr and C12 enter the inverse C matrix: changing them can spoil a chosen matched witness; EJ recalibration can sometimes restore it.

## Map

4278 computed/saved centers; status counts: `{"no_witness_found": 2666, "validated_pass": 1608, "validation_failed": 4}`.
Gray means no witness found by bounded search; orange means unresolved; white is excluded/uncomputed.
A passing pixel is an existence result for its full component set, not a guarantee for arbitrary C12 or Cr.

## Robustness

900 saved fabricated circuits across three designs.
Each of 6 parameters can vary independently by +/-2%: C1_shunt_fF, C2_shunt_fF, C12_fF, C1r_fF, C2r_fF, bare_resonator_frequency.
Cr and Lr scale inversely with bare resonator frequency, preserving mode impedance.
Only EJs recalibrate. Fixed operation frequency uses the nominal dressed mean within 100 Hz numerical tolerance.
Flexible operation frequency permits 4-8 GHz. Logical frequency and H follow the cumulative ladder.
224 interior samples, 64 corners and 12 isolated endpoints/design; combined fractions are not fabrication yields.

| Design | Total | Baseline | EC120 | Flexible frequency | Logical frequency 80-120 | H90 | EJ/EC40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 30 | 30 | 151 | 178 | 300 | 300 |
| 2 | 300 | 235 | 235 | 266 | 266 | 300 | 300 |
| 3 | 300 | 211 | 300 | 300 | 300 | 300 | 300 |

The H CDF is conditional on all other baseline screens, with EC>=120 MHz and fixed operation frequency.
No score is invented for ineligible samples. Counts and CDFs are best-found bounded-search results.

## Files

- `figures/`: six PNG figures, including the marked map, loaded-EC CDFs and latest rescue comparison.
- `capacitance_pairs.csv`: all map statuses and witness metrics.
- `robustness_counts.csv`: separate interior, corner, endpoint and combined counts.
- `selected_designs.json`, `hybridization_cdf.json`, `robustness_analysis.json`: numerical inputs/results.
- `provenance.json`: source hashes, configuration, dependencies and run mode.

Cached mode redraws archived results. Numerical reruns may rediscover different witnesses/counts because the finite search is not exhaustive.
The original study data and narrative PDF are preserved.
