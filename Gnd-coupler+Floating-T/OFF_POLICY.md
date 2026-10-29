# Temporary OFF policy and restoration record

Confirmed 2026-10-28. **OPEN QUESTION:** should isolation ultimately target
physical port exchange or encoded logical exchange? The current physical
criterion is temporary; do not infer that residual logical exchange vanishes.

## Current rule

- Three bodies: physical A2/B1 01-10 with C ground reference.
- Five/seven bodies: physical 0100-0010, rail order A1,A2,B1,B2.
- Retain all four physical references 1000,0100,0010,0001. Build their canonical
  Hamiltonian from full dressed eigenstates, not a raw matrix entry or a
  two-state elimination. Include virtual excitations and within-rail mixing.
- Seven-mode references additionally have both resonators in vacuum.
- Freeze all geometry/rail EJs during OFF retuning; vary C EJ only.
- Logical 01L-10L and 00L-11L, auxiliary exchanges and secular ZZ are residuals.
  Logical residual extraction includes virtual paths and local energy axes.
- Version `physical-single-excitation-off-v1` marks active results. Five/seven
  records explicitly say `temporary_open_question`. Old `logical-exchange-off-v1`
  and CZ/ZZ-null records must fail current acceptance validation.

## Exact change map

Line numbers were refreshed 2026-10-29 after the performance update; stable function/anchor names identify the
same locations after later edits. The logical extractor was retained intact in
physics, with diagnostic-only metadata, to make revisiting the choice explicit.

| File | Line | Function / anchor | Responsibility |
|---|---:|---|---|
| [../transmon-based CZ/transmon_cz/channel_policy.py](../transmon-based%20CZ/transmon_cz/channel_policy.py#L23) | 23 | `def off_target_definition` | Versioned temporary target and physical/logical status |
| [../transmon-based CZ/transmon_cz/channel_policy.py](../transmon-based%20CZ/transmon_cz/channel_policy.py#L55) | 55 | `def physical_exchange_from_eigenpairs` | Four-physical-reference canonical extraction |
| [../transmon-based CZ/transmon_cz/channel_policy.py](../transmon-based%20CZ/transmon_cz/channel_policy.py#L106) | 106 | `def logical_exchange_from_eigenpairs` | Preserved logical residual calculation |
| [floating_coupler/off_observables.py](floating_coupler/off_observables.py#L10) | 10 | `def physical_single_excitation_references` | Physical reference vectors in hybridized-pair basis |
| [floating_coupler/pairs.py](floating_coupler/pairs.py#L196) | 196 | `physical_exchange=physical_exchange_from_eigenpairs` | Five-mode odd-parity extraction |
| [seven_readout.py](seven_readout.py#L85) | 85 | `def physical_off_detail` | Seven-mode odd-parity extraction with vacuum photons |
| [floating_coupler/paired_design.py](floating_coupler/paired_design.py#L19) | 19 | `def off_channel` | Single acceptance/optimization router |
| [floating_coupler/refine_capacitors.py](floating_coupler/refine_capacitors.py#L124) | 124 | `g_off_mhz=off_channel` | Capacitor optimizer objective metric |
| [floating_coupler/pairs.py](floating_coupler/pairs.py#L253) | 253 | `def refine_pair_off` | Frozen-hardware root objective |
| [final_seven_validation.py](final_seven_validation.py#L20) | 20 | `def off_exchange` | Seven-mode optimization route |
| [configs/workflow-request-template.json](configs/workflow-request-template.json#L186) | 186 | `"off_target"` | Request schema and explicit open question |

## How to revisit or restore logical-exchange cancellation

Only act after user confirmation. Do not change ON logical definitions or the
minimum-S1 source/auxiliary selection.

1. Choose and document a new OFF version (never reuse an old certificate version).
   Change the five/seven metadata together in `off_target_definition`.
2. Route `off_channel` to the full-state logical residual. Explicitly promote a
   copied result into the new OFF definition; its current diagnostic record is
   not an OFF certificate. Do not use the raw retained logical block.
3. Demote physical extractor metadata to an independent diagnostic definition;
   do not allow its fixed physical matrix element to inherit a logical OFF label.
4. Update `refine_pair_off`'s accepted objective name, capacitor metrics, stage
   templates, docs and regression expectations. Both five/seven root paths must
   use the same router; preserve three-body physical seeding.
5. Keep BOTH physical and logical calculations available for comparison. Re-run
   fixed-hardware convergence checks and fresh OFF searches before any acceptance.
   Do not reinterpret previous roots, geometry searches or gate calibrations.

## Retained numerical reference, not a newly optimized final design

Ignored `output/physical-exchange-three-body-v1/` retains the original physical
three-body seed. `output/current-three-to-five-comparison/frozen-circuit.json`
retains the compensated A150/B90 five-body embedding. The prior independent
`output/five-body-physical-exchange-null/results.json` found a physical zero near
9.994663 GHz with logical residuals approximately 1052.06 and 1035.70 Hz.
These are comparison records under their original provenance, not automatic
certificates from the newly unified production kernels or a completed seven-mode
gate/readout validation. Do not mistake the 100 MHz logical-minimum campaign for
the selected physical-null reference.
