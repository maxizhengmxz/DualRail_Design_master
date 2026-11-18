# Optional resonator-frequency readout studies

Use these only on explicit request. They do not run as part of ordinary
readout feasibility or the coupler-design sequence. All frequencies below are
in GHz. Keep generated calculations/reports outside Git in ignored output.

## 1. Tolerance of a saved physical design

Question: how far can this particular design's resonator frequency move while
EJ recalibration still recovers acceptable readout?

Keep the pad-ground, across-pad, intra-pair and resonator-link capacitors
fixed. Change only Cr/Lr at fixed 50 ohm impedance and recalibrate the two EJs.
Rebuild the capacitance matrix and loaded ECs without compensation. Use each
saved rail's constraints; historical EJ/EC exceptions are case-specific.
Calculate the pair plus its resonator alone, without the other rail or coupler.

For an explicit-pad standalone readout record:

```powershell
python -m readout_workflow.resonator_frequency_tolerance --run --input PAIR_READOUT.json --output NEW_TOLERANCE --range-ghz LOW HIGH
python -m readout_workflow.resonator_operation_map --run --input PAIR_READOUT.json --output NEW_MAP --resonator-range-ghz LOW HIGH --operation-range-ghz OP_LOW OP_HIGH
python -m readout_workflow.resonator_operation_bounds --run --input PAIR_READOUT.json --map MAP_RESULTS.json --prior TOLERANCE_RESULTS.json --output NEW_BOUNDS
```

The first command finds a connected, grid-resolved interval with flexible
operation frequency. The second maps resonator versus dressed operation
frequency, maximizing mean chi through EJ calibration at each pixel. The third
refines operation-frequency extrema using saved passing seeds. It does not
redesign capacitors. Choose the supplied ranges for the actual case; examples
or historical scan boxes are not new hardware constraints.

For saved three-node EC-design records, `ec_resonator_tolerance.py` provides
fixed-operation and flexible-operation interval checks. The optional
`asymmetric_design_bounds.py` campaign extends saved optimized designs to
separate lower/upper limits using their original certificates; it expects the
saved pair-sequence dataset, not an arbitrary nominal JSON.

Report the accepted interval/region, calibrated operation frequencies, chi,
chi mismatch, H, dispersive ratios and failed constraints. Failed finite
searches are not proofs of infeasibility. Disconnected branches must not be
joined into one certified band. This is calibration-recoverable tolerance,
not passive drift immunity or capacitor-error Monte Carlo.

## 2. Maximum range for a specified nominal loaded EC tuple

Question: which capacitor design supports the widest resonator-frequency
interval for this tuple under the supplied readout constraints?

Use `resonator_robust_design.py --mode tolerance`; the full protocol is in
[RESONATOR_ROBUST_DESIGN.md](RESONATOR_ROBUST_DESIGN.md). Optimize C12 and the
common resonator link, compensating the two nominal self capacitors to match
the input EC tuple at 6 GHz / 50 ohm. ONE candidate capacitor network must
work throughout its symmetric interval around 6 GHz. Only EJs recalibrate
within that interval; loaded EC changes away from the reference are not
compensated. Maximize validated interval width, with chi/H and all other
readout limits as constraints, not secondary strength objectives.

```powershell
python -m readout_workflow.resonator_robust_design --run --mode tolerance --ec EC1_MHZ EC2_MHZ --input SEED.json --conditions CONDITIONS.json --output NEW_DESIGN
```

Use the existing adaptive quantum/sign validation and sampled interior checks.
Report certified sampled endpoints, numerical boundary resolution, achieved
worst chi/H, and whether the configured width ceiling was reached. This is a
best-found finite-budget interval, not proof of a global maximum or continuous
coverage. A prescribed interval with chi maximization is a separate mode;
do not confuse it with maximum-width exploration or the coupler adapter's
minimum-intra-pair-capacitance objective.

Optional helpers:

- `resonator_refinement.py`, via `--refine-from`: reuse a completed coarse
  design and refine its saved boundary bracket into a separate directory.
- `resonator_design_campaign.py`: repeat the same optimization over supplied
  EC tuples, with checkpointing and one selected strict half-plane. Mirroring
  supplies seeds only; every reported solution is recalculated.
- `resonator_pair_sequence.py`: explicitly launch the saved B-fine,
  A-coarse, A-refined campaign plan with separate rail conditions and input
  snapshots. It is a dataset-specific scheduler, not a normal design stage.

Keep source/input identity checks, rejected cases, source snapshots and
read-only coarse inputs. A fresh search or refinement needs a separate output
when required by the existing provenance checks. No study launches on import
or without `--run`.
