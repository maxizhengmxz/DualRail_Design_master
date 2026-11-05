# Physical parasitic capacitors

All values are actual port-to-port capacitances in fF, default zero. They are
independent inputs, not mediated or effective modal capacitances. Reflection is
A11/A12/A21/A22 ↔ B22/B21/B12/B11, r1 ↔ r2, with C1 unchanged.

The compensated-study group 1 includes the four extra resonator links,
A21-B11/B12-A22 and the cross-pad links A11-A22/B22-B11. The latter are
Cp11/Cp12 in dark green, preserving existing Cp1-Cp6 and red Cp7-Cp10 labels.
Group 2 contains A11-C1, A12-C1, B21-C1 and B22-C1. B11-B22 uses the canonical
input spelling B22-B11 for the same physical capacitor.

New compensation searches minimize the largest fractional change of every
nonzero physical capacitor, then reduce combined movement while retaining that
minimax value. Matrix error is constrained at a configurable engineering
matching tolerance, default 0.001 (one part per thousand). The tolerance applies
individually to the nine intended nonzero mode-coupling coefficients and to the
relative Frobenius error over the matched matrix entries. All seven diagonal
entries stay fixed, preserving loaded transmon ECs and resonator bare frequencies.
Equal floating-rail grounds and positive capacitors remain required. The numeric
search guard is +/-95%, with actual movement minimized. This fit tolerance does
not certify fabricated or EM-extracted capacitances to 0.1% accuracy.

Group 1 includes every K entry in the matrix norm. The intentional-channel
group-two convention includes seven diagonals and nine intended pairs. Its
twelve remaining entries, including newly introduced coupling paths, are
residual diagnostics. Near-zero and excluded entries require absolute coupling
and full-Hamiltonian checks. Existing OFF and gate-precision requirements remain.

The implementation is studies/bounded_parasitic_compensation.py. The opt-in
preparation runner studies/parasitic_study.py exposes --match-rtol, default .001.
See OPTIONAL_STUDIES.md for separate fit, nominal-reference, quantum validation,
fixed-control basis checks and saved-data reporting stages. New electrostatic
candidates require their own gate/OFF validation before inheriting any fidelity
or residual claims. SLSQP establishes feasible local fits, with explicit
constraint checks. It does not certify a globally minimal allocation.

The current retained study uses 24 individually compensated cases, eight
group-1 and four group-2 ports at 0.1 and 0.2 fF. These are single-parasitic tests,
not simultaneous combinations or a fabrication probability distribution.
All 24 selected fits were quantum validated. On 2026-11-05 the user requested
removal of the calculation records and archives. The approved report remains a
visual template under templates/report-references, with embedded calculation
attachments removed. The zero-parasitic nominal circuit and operating parameters
are preserved in configs/Design_V1_20261105.json. A new study requires fresh
calculations. The earlier unconverged +455.8% diagnostic is superseded.

| A-side input | Mirrored B-side input | Default (each, fF) |
|---|---|---|
| A11-r1 | B22-r2 | 0 |
| A22-r1 | B11-r2 | 0 |
| A11-A22 | B22-B11 | 0 |
| A21-C1 | B12-C1 | 0 |
| A21-B11 | B12-A22 | 0 |
| A12-C1 | B21-C1 | 0 |
| A11-C1 | B22-C1 | 0 |

Use `parasitic_capacitances_ff` with these exact keys. Missing keys are zero;
negative/nonfinite values and unknown names are rejected. Values need not be
equal on the two sides. Expanded physical links carry `parasitic: true` and
`name: Cpar_<ports>`. Intended capacitors remain distinct and are never replaced.
Do not draw parasitics unless explicitly requested, even when they are nonzero.

Report compensation changes both in fF and as 100*(C_new-C_nominal)/C_nominal,
so they can be compared with fabrication-robustness budgets. A zero-nominal
parasitic has no defined percentage change and is reported by its absolute fF.

The full request template contains all fourteen zero defaults. The integrated builder
reads this field from its request; standalone pair/readout synthesis reads it
from its case. For saved hardware studies, put it in `five_physical` or each
readout's `physical` record. Explicit-pad model inputs accept it in
`electrostatics`. The helper `floating_coupler.parasitics.with_parasitics`
also expands it into a normal physical-link list for other circuit builders.

Only links whose two ports exist in a subsystem are retained: the five-transmon
model excludes resonator links; standalone readout triples exclude coupler and
opposite-pair links. Removed ports are not grounded or virtually eliminated.
The full integrated circuit retains all fourteen links.

Adding parasitics to fixed physical hardware changes its loaded ECs; expansion
does not silently compensate self capacitances or change EJs. Existing design
synthesis still fits self capacitances to preserve the supplied loaded ECs.
Choose the appropriate fixed-hardware or compensated-design workflow before
evaluating nonzero parasitics. No search or recalibration is launched by defining
these inputs, and the retained nominal design remains unchanged.
