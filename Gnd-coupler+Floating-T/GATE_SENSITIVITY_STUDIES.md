# Optional static-gate sensitivity studies

These are potential post-design studies, invoked only on request. They do not
extend or replace the prescribed three -> five -> ON -> seven design sequence.
No study runs during normal design, import, reporting or a toy-gate search.
Source lives in `studies/`; calculations use the existing explicit-circuit
gate evaluator and the shared logical/channel policy. The four-transmon toy
under `../2Q-gate_XM/` has no explicit coupler or resonators and cannot reproduce
coupler-path diagnostics by itself. Do not move these kernels into that toy.

Use a supplied saved circuit and a fresh output folder. Keep loaded ECs fixed.
Retain the selected source/auxiliary, consistently dressed static references
and local-Z correction. This is static ON sensitivity, not pulse propagation,
idle-referenced ramp fidelity, unknown-noise immunity or decoherence. Report raw
and postselected fidelity together with survival and conditional-phase error.
Fixed-control basis differences precede any phase-only audit correction.
Target probability accuracy is 1e-4; residual precision target is 1 Hz, with
unresolved drift reported rather than hidden by a numerical root.

Commands below run from `Gnd-coupler+Floating-T/` with the project Python.
No numerical values in a historical example become requirements for a new case.

Self/ground-capacitance allocation is documented independently in
[CAPACITANCE_ALLOCATION.md](CAPACITANCE_ALLOCATION.md).

## B differential-frequency sensitivity

`python studies/b_gate_detuning.py --input CIRCUIT.json --calibration GATE.json
--detunings-mhz -10 -2 0 2 10 --output FIXED_DIR`

Use the saved physical circuit without charge-coefficient overrides. A saved
static operating point can supply controls if `--calibration` is omitted.

Define delta_B = f01(B1)-f01(B2), in MHz at the CLI. Apply bare frequencies
mean+delta_B/2 and mean-delta_B/2; convert to EJ using each loaded EC. Keep
mean, duration, coupler ON, A EJs and every charge coefficient fixed. Do not
correct phase during this first sweep. Keep failed diagnostics and identify
the connected central passing interval rather than merging distant lobes.

Then optionally compare calibrated sensitivity:
`python studies/b_detuning_phase_recalibration.py --fixed-sweep
FIXED_DIR/result.json --output RETUNED_DIR --mean-bounds-ghz LOW HIGH`

Keep the same detuning grid/Hamiltonian and frozen duration/coupler. Only B
mean is corrected to recover phase pi. Continue each side independently and
retain phase failures, basis drift and the required mean-frequency correction.
The numerical mean-search box defaults to +/-200 MHz around the reference;
it is not an assumed physical SQUID range. Supply tighter physical limits when
needed. A calibration file, if supplied, must match the frozen sweep controls.
This measures active recalibration performance, not passive noise tolerance.

## Records and reporting

Preserve original nominal inputs, physical-circuit
provenance, controls, matrices, failures and numerical qualifications. A
`selected_static_point.json` sensitivity snapshot nests the circuit provenance
and stores the unchanged charge matrix explicitly; it is not an ordinary capacitor
config and must not be passed to an unmodified circuit loader.
The saved allocation and circuit-target PDFs remain report references. Their
renderers read historical records only; no report generation runs these studies.
Never transfer gate fidelity or OFF certification between distinct matrices.
