# Authoritative dual-rail CZ definitions

This workflow owns the state definitions used by both the four-transmon toy
model and the explicit coupler circuit. Executable definitions live in
`transmon_cz/channel_policy.py`; the coupler imports that module rather than
maintaining its own convention.

## Logical states and auxiliaries

For each isolated rail pair, logical **0** is the lower-energy and logical **1**
the upper-energy one-excitation-like eigenmode. Pair ground is `g`, not logical
0. Two-bit logical labels are ordered **A,B**; the computational order is
`00,01,10,11`. Continue these physical branches by overlap tracking when
dressing or tuning; do not independently relabel them at each avoided crossing.

Physical occupations are ordered A1,A2,B1,B2. Thus `0020` identifies the B1
two-excitation auxiliary `B20`, with A in its pair ground. It does **not** specify
whether the logical source is `01` or `10`. Every resolved CZ channel must give
both its logical source and its auxiliary.

## Selecting the intended resonance: minimum S1 at common intended coupling

Only the specified rail pair moves between idle and gate operation; the other
pair stays fixed. Identify feasible resonances for each allowed logical source
and the requested physical auxiliary. Do not infer a source from `0020`.

Evaluate every candidate at the **same intended matrix-element magnitude g**,
with the same EC inputs, logical targets, reference-basis convention, numerical
truncation and allowed design/tuning controls. Compute
`S1 = mean_over_four_logical_inputs(sum_unintended |g_ik/delta_ik|^2)`.
Exclude the intended resonant source-to-auxiliary edge. Score all remaining
logical-origin edges, including other sources into that auxiliary. S2 is the
separate auxiliary-origin score and does not rank candidates. Apply the toy's
logical-origin screen `max |g/delta| < 0.2` before ranking, unless the user has
explicitly supplied a different limit. Select **one minimum-S1 feasible position**.
Refine interior frequency minima within the allowed ranges; label boundary
minima rather than silently extending the bounds.

Crossing order, crossing count and minimum avoided gap are **not** selection
criteria or silent tie-breaks. Keep path crossings as diagnostics for later
pulse validation. Resolve indistinguishable S1 ties explicitly. Never hardcode
the logical source as 01, 10 or 11.

The toy can adjust its effective J to enforce common g before scoring. Explicit
hardware must achieve that target through permitted circuit/coupler refinement,
preserving all loaded rail ECs. A rescaled estimate `S1*(g_target/g_actual)^2`
is a screening diagnostic only; it cannot certify an S1-selected physical design.
Never compare full-toy and truncated dressed-circuit scores as if their bases
were identical. The selector ranks supplied verified S1 candidates; it does not
perform a spectral or capacitor search itself.

Export the resulting `ChannelSelection`, including source, auxiliary, moved
pair, resonance position, S1, common target g, score basis, comparison conditions
and definition version. Carry that same
record into ON calibration and preserve it as design provenance during OFF
analysis. It does **not** define what is nulled at OFF.

## Explicit-coupler OFF policy: temporary physical exchange

ON definitions and minimum-S1 selection above are unchanged. OFF is a separate
hardware criterion; the toy itself contains no coupler and cannot certify it.
The confirmed temporary five/seven-mode target is physical `0100 <-> 0010`
(rail order A1,A2,B1,B2), using a canonical effective Hamiltonian built directly
from full dressed eigenstates and ALL FOUR physical single-excitation references
1000,0100,0010,0001. Retain intra-rail hybridization. C and readout resonators
are ground/vacuum references; their virtual paths remain. No local logical-axis
rotation is applied to this physical matrix element. Three-body OFF remains
physical A2/B1 single-excitation `01 <-> 10`.

`off_target_definition` and `physical_exchange_from_eigenpairs` implement this
versioned policy. Old logical-exchange OFF records are rejected, not relabeled.
`logical_exchange_from_eigenpairs` remains the full-dressed logical residual
extractor, including virtual auxiliaries and local energy-axis alignment. Its
record is a diagnostic definition, not an OFF certificate. Report both logical
exchange and double flip, auxiliary residuals and secular ZZ at physical OFF.
Check roots, extrema, projection validity and numerical convergence.

**OPEN QUESTION:** whether encoded logical exchange is the preferable isolation
objective. The physical target is temporary and uniform in five/seven modes.
See `../Gnd-coupler+Floating-T/OFF_POLICY.md` for exact implementation/restoration
locations. Do not change S1/CZ-source selection when revisiting OFF.

The active S1 toy search is `../2Q-gate_XM/`; it imports the stable policy and
score kernel here. The original simulation/sweep kernels in this folder remain
available as exploratory tools and do not themselves certify minimum-S1 selection.

## Toy gate calibration and circuit handoff

The toy consists of four full-cosine transmons and an effective direct A2鈥揃1
charge interaction; it has no explicit coupler or resonators. Keep the supplied
loaded rail ECs fixed. The stationary rail stays fixed; refine the moved rail
around its gate seed and gate duration within the user's permitted controls.
For the current unrestricted-frequency calibration permission, start within
200 MHz of the preceding solution. Calibrate conditional phase
`phi11-phi10-phi01+phi00 = pi mod 2pi`, then improve logical return while
maintaining that phase. Remove permitted single-logical-qubit phases consistently.

Use consistently dressed static ON references retaining the CZ auxiliary.
Report raw fidelity, survival-normalized/postselected score, mean survival and
worst-input nonlogical population. A static ON benchmark is separate from an
idle-to-ON quench or a pulse with ramps. Postselected fidelity must always be
reported with survival.

The toy's effective interaction is a design target. Once explicit hardware is
constructed, never independently rescale it: changes in capacitor geometry
require circuit reconstruction and self-capacitance compensation to preserve
every supplied loaded rail EC. Carry the selected channel into the coupler
workflow, which owns electrostatics and the three-, five- and seven-mode stages.

## Historical calculations

Saved outputs and older fixed-source spectroscopy/sweep diagnostics retain
their original labels. A physical-collision seed alone does not resolve the
logical CZ source. Historical CZ-auxiliary-null and ZZ-null capacitor results
retain their values and remain geometry seeds, but are not certified logical
exchange OFF designs. Historical `01`/`10` diagnostics, the superseded
first-crossing rule, and example seeds are not a new S1-selection result. Do not silently relabel saved data
or promote results lacking a verified channel record to current certification.

## Downstream coupler numerical precision (2026-10-29)

The user-approved engineering targets for coupler validation are 1 Hz absolute
for residual interactions and 1e-4 absolute gate infidelity (0.01 percentage
point), comparing the same controls before retuning. Modest adaptive checks stop
on agreement; exhaustive large-basis gate audits are not mandatory. This does
not change the toy's S1/common-g policy, phase-pi requirement, physical acceptance
limits or saved results. The readout workflow retains its own adaptive checks
and physical thresholds. See ../Gnd-coupler+Floating-T/SPEED_PROPOSALS.md.
