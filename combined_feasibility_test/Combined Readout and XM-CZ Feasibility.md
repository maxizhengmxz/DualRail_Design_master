---
title: Combined Readout and XM-CZ Feasibility
aliases:
  - Dual-rail readout–gate feasibility workflow
tags:
  - dual-rail
  - readout
  - cz-gate
  - feasibility
  - simulation
status: in-progress
created: 2026-10-30
updated: 2026-10-30
---

# Combined Readout and XM-CZ Feasibility

## Purpose

This study combines the independent single-rail readout workflow in
`readout_workflow/` with the reduced four-transmon XM-CZ workflow in
`2Q-gate_XM/`.

The device contains two dual rails:

- **Rail A:** logical frequency in 140–160 MHz.
- **Rail B:** logical frequency in 80–100 MHz.
- Rail A may have the lower physical operating frequency even though its
  logical splitting is higher.

Readout is required at the **idle operating point**. Gate fidelity is evaluated
at a separate **gate operating point**. Josephson energies may change between
these points, while fabricated capacitances and other fixed hardware must remain
consistent.

The present work is a staged feasibility test. It is not yet the final
resonator-inclusive certification of a complete fabricated device.

## Feasibility statement

For fixed fabricated hardware \(x\), idle controls \(u_I\), and gate controls
\(u_G\), the desired condition is

$$
\exists x,u_I,u_G:\quad R_A(x,u_I)\land R_B(x,u_I)\land G(x,u_G).
$$

The two operating points may use different EJs and physical operating
frequencies. They may not silently use different fixed capacitances or fixed
charge-coupling coefficients.

## Agreed staged procedure

1. Create an **independent rail-A readout map** for 140–160 MHz.
2. Create an **independent rail-B readout map** for 80–100 MHz on the same
   loaded-EC target domain.
3. Select one readout-feasible rail-A loaded-EC pair.
4. At the central logical frequencies \((150,90)\) MHz, calculate the XM-CZ
   gate map over the rail-B loaded-EC targets.
5. Retain gate points satisfying both strict conditions:
   - raw CZ fidelity \(>95\%\);
   - post-selected CZ fidelity \(>99.9\%\).
6. Overlay the independently generated rail-B readout and gate maps.
7. Reconcile fixed reduced-model hardware on every overlay point by matching
   the intra-rail charge coefficients required by the gate model while retaining
   readout feasibility at idle.
8. On shortlisted points, build the complete capacitance network with both
   rails and resonators, re-solve operating points, update S1/channel checks,
   and validate dispersiveness and convergence.
9. After fixed-hardware validation, sweep logical frequencies over the required
   ranges and evaluate sensitivity. The present gate calculation uses only the
   central pair \((150,90)\) MHz.

## Direct loaded-EC target selection

The direct-EC domain is not an arbitrary rectangle.

1. `readout_workflow.fine_readout_feasibility` scans the physical shunt plane
   \(36\le C_2<C_1\le128\) fF.
2. The regenerated 140–160 MHz run produced 342 validated or recovered
   physical witnesses.
3. Each witness was projected to its full-circuit loaded coordinates
   \((E_{C1},E_{C2})\).
4. `direct_ec_feasibility.target_grid(..., "hull", step_mhz=5)` constructed the
   convex hull of those projected points and retained 5 MHz lattice points
   inside it.

This produced 48 targets with axis extents

$$
E_{C1}=155\text{–}175\ \mathrm{MHz},\qquad
E_{C2}=205\text{–}265\ \mathrm{MHz}.
$$

The hull is a campaign-selection heuristic, not a physical feasibility proof.
Inside the hull does not imply feasibility, and outside it does not imply
infeasibility.

## Readout-map implementation

### Physical model

For every fixed loaded-EC target and every trial \((C_{12},C_g)\), the direct-EC
driver uses the Schur complement of the full two-transmon-plus-resonator
capacitance matrix to reconstruct physical shunts \((C_1,C_2)\). The complete
matrix is then inverted and passed to the standard readout evaluator.

The interaction convention is

$$
H_\mathrm{int}/h=J_{ij}n_i n_j,
\qquad J_{ij}=\frac{4e^2}{h}(C^{-1})_{ij}.
$$

Loaded EC and charge coefficients therefore come from the same capacitance
matrix.

Each target uses up to three nearby source witnesses, at most 35 least-squares
evaluations per start, and the standard adaptive audit and EJ-sign-stability
checks. A search miss is not a proof of physical infeasibility.

### Parameterized logical-frequency interval

`readout_workflow/direct_ec_feasibility.py` now accepts

```text
--logical-frequency-mhz LOW HIGH
```

The default remains 140–160 MHz. This allowed both rails to use the same solver,
budget, target list, reconstruction, and validation policy:

- Rail A: `--logical-frequency-mhz 140 160`
- Rail B: `--logical-frequency-mhz 80 100`

### Independent readout results

| Map | Logical interval | Validated targets |
|---|---:|---:|
| Rail A | 140–160 MHz | 48/48 |
| Rail B | 80–100 MHz | 48/48 |

The first provisional rail-B implementation found 47/48 points. Re-running the
official parameterized driver recovered \((175,245)\) MHz. The provisional and
official searches can select different valid witnesses because the policy is
first-feasible, not a unique-solution or global-optimization policy.

Audited logical-frequency ranges of the selected witnesses were:

- Rail A: 140.03–141.74 MHz.
- Rail B: 80.70–99.99 MHz.

Rail A lies near the lower acceptance boundary because the driver stops at the
first fully validated solution rather than optimizing toward 150 MHz.

## Gate-map implementation

The test selected the interior rail-A target

$$
(E_{C,A1},E_{C,A2})=(165,240)\ \mathrm{MHz}.
$$

`2Q-gate_XM` evaluated the same 48 rail-B targets at central logical splittings
\((150,90)\) MHz. The reduced gate model:

1. solves isolated-rail charge coefficients for the requested logical
   splittings;
2. scans the common rail-A physical frequency;
3. solves the rail-B frequency for the selected auxiliary collision;
4. sets the effective A2–B1 coefficient to realize the requested dressed
   coupling \(g=5\) MHz;
5. selects one minimum-S1 operating position;
6. calibrates the rail-B frequency offset and gate duration to conditional
   phase \(\pi\);
7. reports raw and post-selected CZ fidelities at the same calibrated gate.

The gate screen retained points only when

$$
F_\mathrm{raw}>0.95,
\qquad F_\mathrm{post}>0.999.
$$

The independent readout/gate overlay contained 28 points.

## Reduced fixed-hardware reconciliation

The independent gate model normally re-solves the A1–A2 and B1–B2 charge
coefficients. Those coefficients are fixed by fabricated capacitances in the
readout witnesses, so a Boolean map intersection alone is insufficient.

For each of the 28 overlay points, `reconcile_and_plot.py`:

1. held the loaded EC target fixed;
2. reconstructed physical shunts at every trial;
3. re-solved the idle readout witness;
4. imposed equality to the gate-required intra-rail charge coefficient;
5. ran the normal readout audit and sign-stability checks;
6. allowed idle and gate EJs to differ;
7. recorded the gate-required effective A2–B1 coefficient as the inter-rail
   target for later full-model realization.

Sixteen of the 28 overlay points passed this reduced reconciliation.

This stage matches loaded EC and intra-rail charge coefficients. It does **not**
yet prove that a single complete capacitance matrix including the inter-rail
link and both resonators preserves those quantities. That is the next major
physics-validation stage.

## Current combined results

On the 16 reduced fixed-hardware feasible points:

| Quantity | Range |
|---|---:|
| Raw CZ fidelity | 95.601–98.491% |
| Post-selected CZ fidelity | 99.908–99.978% |

The operating-point EJ shift is defined as

$$
\Delta E_J=E_{J,\mathrm{gate}}-E_{J,\mathrm{idle}}.
$$

| Junction | EJ-shift range |
|---|---:|
| A1 | −3.356 to +0.013 GHz |
| A2 | −2.313 to +0.040 GHz |
| B1 | −5.653 to +1.113 GHz |
| B2 | −3.443 to +0.868 GHz |

The full idle EJ values, gate EJ values, signed shifts, and fidelities are saved
per point in `data/combined_operating_point_ej_shifts.json`.

## Resonator-inclusive full-network endpoint validation

The 16 reduced reconciled candidates were promoted to a candidate-specific
six-node physical network with node order

```text
A1, A2, rA, B1, B2, rB
```

The XM-CZ effective A2–B1 coefficient was interpreted as a fixed direct
A2–B1 capacitor for this validation. This assumption must be replaced if the
eventual hardware uses an explicit tunable coupler.

### Physical hardware solve

For each candidate, seven physical variables were solved simultaneously:

- four transmon-to-ground shunts;
- direct A2–B1 capacitance;
- rail-A intra-pair capacitance;
- rail-B intra-pair capacitance.

The seven target equations were:

- four loaded transmon EC coefficients from the complete inverse-capacitance
  matrix;
- gate-required A1–A2 charge coefficient;
- gate-required B1–B2 charge coefficient;
- gate-required A2–B1 charge coefficient.

All 16 hardware solves converged with maximum relative target errors below
\(10^{-15}\). The same physical matrix was then frozen at idle and gate; only
the four EJs changed.

### Hamiltonian and basis

Each rail plus its resonator was diagonalized locally, retaining dressed
pair/resonator states. All inverse-capacitance-induced cross-rail interactions
were then retained in the two-block Hamiltonian. Two truncations were evaluated:

| Validation | Local transmon levels | Resonator levels | Retained states per rail |
|---|---:|---:|---:|
| Base | 5 | 3 | 12 |
| Larger | 6 | 4 | 16 |

The maximum base-to-larger changes were:

- raw fidelity: \(2.02\times10^{-4}\);
- post-selected fidelity: \(7.03\times10^{-5}\);
- updated S1: \(6.19\times10^{-6}\).

### Checks performed

- Conditional idle readout for each rail with the other rail in `g`, lower
  logical, and upper logical reference states.
- Readout strength, chi matching, logical-frequency interval, operation below
  the resonator, hybridization, and assignment overlap.
- Five-transmon-style multilevel dispersive ratios generalized to the
  four-transmon matter network and each resonator coupling vector.
- Updated S1 channel ratios using resonator-dressed local states and every
  inverse-capacitance-induced cross-rail interaction.
- Full-network time evolution at the inherited gate EJs and duration, followed
  by logical projection, leakage accounting, and local-Z optimization.
- A separate duration-only recalibration test over 60–140 ns.

### Endpoint result

| Category | Larger-basis passes |
|---|---:|
| Physical hardware realization | 16/16 |
| Resonator dispersiveness | 16/16 |
| Updated S1 screen | 16/16 |
| Full conditional idle readout | 0/16 |
| Full-network gate thresholds | 0/16 |
| All requirements | 0/16 |

Larger-basis gate ranges at the inherited endpoints were:

- raw CZ fidelity: 81.09–97.07%;
- post-selected CZ fidelity: 92.55–99.28%.

The readout failures were primarily conditional chi mismatch and spectator
dependence after assembling both rails into one network. Gate degradation came
from the extra cross couplings retained by the complete inverse-capacitance
matrix. The dispersive ratios remained below 0.1 and updated maximum S1 ratios
remained below 0.2 for all candidates.

Duration-only recalibration did not restore 99.9% post-selected fidelity.

This is an **endpoint validation failure**, not a global infeasibility result.
The inherited idle and gate EJs were not jointly re-optimized in the complete
network. A subsequent repair search must retune idle EJs for conditional
readout and gate EJs/operating frequency/duration for the full-network gate
while keeping the solved capacitance matrix frozen.

## Implementation map

| File | Responsibility |
|---|---|
| `readout_workflow/direct_ec_feasibility.py` | Shared parameterized direct-EC driver |
| `rail_b_readout_map.py` | Provisional rail-B implementation retained for comparison only |
| `prepare_gate.py` | Select rail A and create the 48-point central-frequency gate config |
| `reconcile_and_plot.py` | Independent-map overlay and reduced fixed-hardware reconciliation |
| `compare_rail_b_maps.py` | Compare provisional and official rail-B maps |
| `plot_readout_maps.py` | Binary independent readout maps |
| `plot_readout_logical_frequency.py` | Audited logical-frequency maps |
| `plot_combined_metrics.py` | Gate-fidelity and idle-to-gate EJ-shift maps |
| `full_network_validation.py` | Six-node hardware solve and two-basis endpoint validation |
| `plot_full_network_validation.py` | Category, fidelity, readout, dispersive, and S1 summary maps |

Generated data are under `combined_feasibility_test/data/` and are ignored by Git.

## Important generated artifacts

- `data/fine_readout_140_160/`: regenerated physical-capacitance contour.
- `data/rail_a_readout_map/`: official rail-A direct-EC map.
- `data/rail_b_readout_map_official/`: official rail-B direct-EC map.
- `data/gate_central_output/`: central-frequency XM-CZ checkpoints.
- `data/reconciliation/`: per-point reduced hardware reconciliation.
- `data/combined_operating_point_ej_shifts.json`: idle/gate EJ record.
- `data/full_network_validation/`: complete-network per-candidate validation.
- `data/plots/independent_readout_maps.png`
- `data/plots/readout_logical_frequency_maps.png`
- `data/plots/agreed_test_overlay.png`
- `data/plots/combined_gate_fidelity.png`
- `data/plots/rail_a_ej_shifts.png`
- `data/plots/rail_b_ej_shifts.png`
- `data/plots/full_network_validation.png`

## Stored-data locations and formats

### Base location and persistence

All paths below are relative to

```text
/Users/chenlu/work/量旋/Dual-rail/Simulation/code/DualRail_Design_master/
  combined_feasibility_test/
```

Generated numerical data live under `combined_feasibility_test/data/`. The root
`.gitignore` excludes `data/` directories, so these results are local working
artifacts and are **not** included automatically in a Git commit. Copy or back
up the data directory separately when moving the study to another machine.

### Fine-map source contour

Directory: `data/fine_readout_140_160/`

| File | Format and contents |
|---|---|
| `settings.json` | Complete search policy, constraints, grid, source hashes, and validation policy |
| `progress.json` | Current/completed phase and event counts |
| `samples.sqlite` | Append-only event ledger; tables `events(sample_id,phase,pair_key,payload)` and `metadata(key,value)` |
| `feasibility_results.json` | JSON array of all 4,278 first-pass physical-shunt results |
| `diagnostic_results.json` | JSON array of 997 diagnostic results |
| `diagnostic_batch.json` | Diagnostic target list and policy identity |
| `results.json` | Final merged result per physical-shunt pixel; source of the 342 projected witnesses |

Each final result contains the physical pair, status, attempts, failures, timing,
and—when available—a `point` object containing the audited device and
observables.

### Independent direct-EC readout maps

Official directories:

- Rail A: `data/rail_a_readout_map/`
- Rail B: `data/rail_b_readout_map_official/`

The older `data/rail_b_readout_map/` is the provisional comparison run and
must not be used as the current rail-B map.

| File pattern | Format and contents |
|---|---|
| `settings.json` | Exact interval, target list, search budget, source hashes, and audit policy |
| `progress.json` | Completion and pass counts |
| `capacitance_reconstruction_check.json` | Maximum reconstruction error against source witnesses |
| `ec_<EC1>_<EC2>.json` | One checkpoint/result for a loaded-EC target, for example `ec_165_240.json` |
| `results.json` | Sorted JSON array containing all 48 target results |

The direct-EC result object has the principal fields:

```json
{
  "ec1_target_mhz": 165,
  "ec2_target_mhz": 240,
  "status": "validated_pass",
  "point": {"...": "audited readout witness"},
  "closest_screening_point": {"...": "best screened candidate"},
  "attempts": [],
  "seconds": 0.0,
  "completed_utc": "...",
  "readout_optimized": false
}
```

Important fields inside `point` include:

- `device`: physical shunts, `C12_fF`, resonator coupling capacitances,
  `Cr_fF`, and `Lr_nH`;
- `loaded_ec1_mhz`, `loaded_ec2_mhz`;
- `ej1_ghz`, `ej2_ghz`, `ej1_over_ec1`, `ej2_over_ec2`;
- `splitting_ghz`, `operating_frequency_ghz`;
- `J12_ghz`, `J01_12_mhz`;
- `chi_m_ghz`, `chi_p_ghz`, `ratio`;
- `lambda_max`, `hybridization`, `min_assignment_overlap`;
- `chi_sign_stability`, `basis`, and `validation`.

### Gate configuration and checkpoints

| Location | Format and contents |
|---|---|
| `data/gate_central.json` | Input configuration and 48 explicit loaded-EC tasks |
| `data/gate_central_output/manifest.json` | Run/source hashes, worker count, completion, and failure counts |
| `data/gate_central_output/points/<SHA256>.json` | One atomic checkpoint per gate task |

The SHA-256 filename is derived from the canonical task payload. Read the
checkpoint's `payload.ec_GHz` to identify its A1, A2, B1, and B2 coordinates.
Each successful record has this shape:

```json
{
  "task_id": "...",
  "payload": {"ec_GHz": [], "logical_splittings_GHz": []},
  "status": "ok",
  "result": {
    "status": "calibrated",
    "S1_position": {},
    "final": {
      "raw_fidelity": 0.0,
      "postselected_fidelity": 0.0,
      "acceptance": 0.0,
      "EJ_GHz": [],
      "couplings_GHz": {}
    }
  }
}
```

`final.EJ_GHz` is ordered `[A1,A2,B1,B2]`. The coupling dictionary contains
`A1_A2`, `B1_B2`, and `A2_B1` in cyclic GHz.

### Overlay and reconciliation

Directory: `data/reconciliation/`

| File | Format and contents |
|---|---|
| `ec_<B1>_<B2>.json` | Per-overlay-point A/B reconciliation attempts and witnesses |
| `results.json` | JSON array of all 28 readout/gate overlay candidates |
| `summary.json` | Counts at each stage and the reconciliation scope statement |

A reconciliation result records raw/post-selected fidelity, the fixed
inter-rail target, complete reconciled rail-A and rail-B idle witnesses, failure
reasons, and `reduced_fixed_hardware_pass`.

### Operating-point EJ table

File: `data/combined_operating_point_ej_shifts.json`

This is a JSON array over the 16 reduced fixed-hardware-pass points. Every row
contains:

```json
{
  "ec_b_mhz": [155, 205],
  "raw_fidelity": 0.0,
  "postselected_fidelity": 0.0,
  "idle_ej_ghz": {"A1": 0.0, "A2": 0.0, "B1": 0.0, "B2": 0.0},
  "gate_ej_ghz": {"A1": 0.0, "A2": 0.0, "B1": 0.0, "B2": 0.0},
  "delta_ej_ghz": {"A1": 0.0, "A2": 0.0, "B1": 0.0, "B2": 0.0}
}
```

The zeros above illustrate the schema only. Stored rows contain the calculated
floating-point values.

### Plot files

Directory: `data/plots/`. Every plot is stored as both PNG and PDF.

| Plot basename | Content |
|---|---|
| `independent_readout_maps` | Binary rail-A and rail-B readout feasibility |
| `readout_logical_frequency_maps` | Audited logical frequency on each readout map |
| `agreed_test_overlay` | Readout, gate, and reconciliation status |
| `combined_gate_fidelity` | Raw and post-selected CZ fidelity on the 16-point region |
| `rail_a_ej_shifts` | Signed idle-to-gate shifts for A1 and A2 |
| `rail_b_ej_shifts` | Signed idle-to-gate shifts for B1 and B2 |
| `full_network_validation` | Full-network gate, readout, dispersive, S1, and category results |

### Full-network validation files

Directory: `data/full_network_validation/`

| File | Format and contents |
|---|---|
| `ec_<B1>_<B2>.json` | One complete physical-network result with base/larger basis checks |
| `results.json` | JSON array containing all 16 candidate results |
| `summary.json` | Top-level pass counts and physical-realization convention |
| `analysis_summary.json` | Category counts, convergence maxima, and fidelity ranges |

Each candidate file contains the solved physical capacitance matrix, inverse
matrix, target/achieved EC and J values, shunts and mutual capacitances, idle
and gate EJs, conditional readout reports, dispersive reports, updated S1,
projected logical gate matrices, leakage, fidelities, and convergence changes.

In Obsidian, a plot can be embedded directly in this note with, for example:

```markdown
![[data/plots/combined_gate_fidelity.png]]
```

## Viewing and editing results

### View in Obsidian

- Open this repository, or `combined_feasibility_test/`, as an Obsidian vault.
- Use the file explorer to open JSON as plain text and PNG/PDF files as
  attachments.
- Embed plots with `![[relative/path.png]]` and link data with
  `[[data/rail_a_readout_map/results.json]]`.
- Obsidian is suitable for notes and quick JSON inspection, but not for
  querying the 45 MB SQLite ledger.

### View JSON from the terminal

Pretty-print a complete file:

```sh
python -m json.tool combined_feasibility_test/data/reconciliation/summary.json
```

Inspect selected fields with `jq` when installed:

```sh
jq '.[] | {ec1_target_mhz,ec2_target_mhz,status}' \
  combined_feasibility_test/data/rail_b_readout_map_official/results.json
```

Use Python for portable filtering:

```sh
/opt/anaconda3/bin/conda run -n work python -c \
  "import json; p=json.load(open('combined_feasibility_test/data/combined_operating_point_ej_shifts.json')); print(p[0])"
```

### View the SQLite ledger

With the `sqlite3` command-line client:

```sh
sqlite3 -readonly combined_feasibility_test/data/fine_readout_140_160/samples.sqlite
```

Useful read-only commands inside SQLite are:

```sql
.tables
.schema events
SELECT phase, COUNT(*) FROM events GROUP BY phase;
SELECT key, value FROM metadata;
```

### Regenerate plots after code changes

Run plotting scripts from the repository root:

```sh
MPLCONFIGDIR=/tmp/dualrail-mpl \
  /opt/anaconda3/bin/conda run -n work python \
  combined_feasibility_test/plot_readout_maps.py

MPLCONFIGDIR=/tmp/dualrail-mpl \
  /opt/anaconda3/bin/conda run -n work python \
  combined_feasibility_test/plot_readout_logical_frequency.py

MPLCONFIGDIR=/tmp/dualrail-mpl \
  /opt/anaconda3/bin/conda run -n work python \
  combined_feasibility_test/plot_combined_metrics.py
```

### Editing policy

- Treat `settings.json`, `manifest.json`, per-point checkpoints, `results.json`,
  and `samples.sqlite` as generated, append-only or immutable evidence.
- Do **not** manually edit a checkpoint and then resume the same output
  directory; hashes and aggregate results may no longer describe its contents.
- Change physics/search settings in source scripts or input configuration, then
  use a **new output directory**.
- Plot style and derived-field changes should be made in the plotting scripts;
  rerun them to replace the derived PNG/PDF files.
- For ad hoc analysis, read the JSON into a separate script or notebook and
  save new derived outputs under a clearly named new subdirectory.
- Back up `combined_feasibility_test/data/` before deleting or rerunning campaigns;
  Git cannot restore these ignored files.

## Reproduction environment

The calculations use the Conda environment `work`:

```sh
/opt/anaconda3/bin/conda run -n work python <script>
```

Representative plotting and comparison commands are listed in
[[README|the combined-feasibility README]].

## Known limitations and next work

1. **Central frequencies only:** the gate map currently uses \((150,90)\) MHz.
   Logical-frequency sensitivity over 140–160 MHz and 80–100 MHz has not been
   run for the reconciled devices.
2. **Reduced inter-rail interaction:** XM-CZ uses an effective A2–B1 charge
   coefficient rather than the complete physical coupler/full-capacitance
   realization.
3. **No full resonator-inclusive combined diagonalization:** the two readout
   witnesses have not yet been assembled into one complete circuit containing
   both rails, both resonators, and the inter-rail hardware.
4. **Loading must be recomputed:** adding the inter-rail connection can alter
   loaded EC and charge coefficients. Physical shunts may need to be solved
   again within fabrication bounds.
5. **S1 and new channels must be updated:** the full model must recompute S1,
   logical/resonator channels, state assignments, detunings, dispersive ratios,
   and basis convergence.
6. **Search misses are bounded-search outcomes:** neither the fine map, the
   direct-EC map, nor reconciliation provides a global infeasibility proof.

## Recommended next stage

For each of the 16 reduced reconciled candidates:

1. assemble one physical capacitance network containing A1, A2, B1, B2, both
   readout resonators, and the inter-rail coupling implementation;
2. solve physical shunts so the complete inverse-capacitance matrix targets the
   selected loaded ECs;
3. freeze all fabricated capacitances;
4. independently solve idle and gate EJs;
5. validate both idle readout requirements;
6. rerun the gate calculation with fixed physical coefficients;
7. update S1 and all resonator-related dispersive/channel diagnostics;
8. perform basis-convergence checks;
9. finally sweep logical frequencies and quantify worst-case fidelity over the
   required ranges.
