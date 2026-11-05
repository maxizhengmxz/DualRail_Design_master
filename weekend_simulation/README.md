# Weekend simulation campaign

This top-level directory contains the reusable inputs and generated results for
the new full-orientation, logical-frequency-grid feasibility campaign. It is
kept outside the earlier `combined_feasibility_test/` test run.

## Readout layout

- `readout/fine_c1_c2/rail_a_140_160/`: authoritative 140-160 MHz fine C1-C2
  SQLite campaign and JSON exports.
- `readout/fine_c1_c2/rail_b_80_100/`: authoritative 80-100 MHz fine C1-C2
  SQLite campaign and JSON exports.
- `readout/direct_ec/rail_a_step5/`: full-orientation, 1.5-scaled EC map on a
  5 MHz lattice.
- `readout/direct_ec/rail_b_step2/`: full-orientation, 1.5-scaled EC map on a
  2 MHz lattice.
- `readout/plots/`: PNG and PDF readout figures.
- `readout/summaries/`: compact JSON summaries.

The fine maps exploit exact isolated-readout label swapping only when creating
the direct-EC exploration domain. Ordered EC points remain distinct in all
downstream gate calculations.

The production gate domain contains every ordered target in the rail-B
direct-EC campaign, regardless of readout status. Readout and gate witnesses
use separate EJs and are calculated independently. Their Boolean maps are
overlaid only after both calculations finish. The
`prepare_gate.py --readout-feasible-only` option is diagnostic-only and must
not be used for production feasibility maps.

## Gate-data retention policy

Gate runs use detailed per-task JSON only as temporary, interruption-safe
checkpoints. After a campaign reaches `complete`, run
`archive_gate_campaign.py CAMPAIGN --prune`. The archiver first creates and
verifies:

- joint minimum-fidelity and calibration-coverage maps;
- logical-grid sensitivity maps;
- per-logical-pair calibration, threshold-pass and runtime maps;
- `compact_results.sqlite`, containing final fidelities, operating points,
  EJs, couplings, S1/S2, leakage, failure categories and runtimes;
- `archive_summary.json`, including original and compact byte counts.

Only after the SQLite row count and plot set are verified does `--prune`
remove the detailed `points/` JSON directory. Intermediate screening and
calibration traces are intentionally not retained. Re-running the original
configuration can regenerate them if deeper diagnostics are needed.
