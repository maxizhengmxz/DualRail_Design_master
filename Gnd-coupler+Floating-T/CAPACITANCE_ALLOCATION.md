# Optional self/ground-capacitance allocation study

This post-design study runs only on explicit request. It does not extend or
replace the standard three -> five -> ON -> seven design sequence. No sweep
runs during import, normal design, reporting or toy-gate search. Source lives
in `studies/` and uses the existing circuit reconstruction and OFF evaluator.
Use a supplied saved physical circuit and a fresh output directory. Generated
results remain ignored.

Commands below run from `Gnd-coupler+Floating-T/` with the project Python.

## Matching constraints and residual minimization

`python studies/capacitance_allocation.py --input CIRCUIT.json --output NEW_DIR
--scope A --self-fractions 0.1 0.3 0.5 0.7 0.9`

Available scopes: `A`, `B`, `A2`, `all`. Use the same physical definition
`p = Cs/(Cs + Cg_each/2)` everywhere, with equal grounds per floating transmon.
Apply the prescribed p to selected rails; retain the nominal physical fractions
of nonselected rails. Compensation may change their absolute capacitances.
The grounded coupler is excluded from this ratio constraint.

Hard-match all five loaded ECs, both resonator charging diagonals at fixed
inductances, and nine intended charge coefficients:

- A1-A2 and B1-B2;
- A2-C, C-B1 and direct A2-B1;
- A1-r1, A2-r1, B1-r2 and B2-r2.

Require positive physical capacitors and the supplied loading bounds. Among
feasible matches minimize absolute RMS of the twelve unintended local exchange
coefficients at the same reference controls; within 1 Hz in residual RMS, prefer
smaller maximum fractional capacitor movement, then its squared sum. Include
pad-ground capacitors in that movement report. Report absolute residuals and
changes from nominal, the largest residual, compensation values, full matrices and Jacobian rank. A rank-16 match
with sixteen variables has no remaining continuous freedom at fixed p.
Finite search failure is not proof that a physical design is impossible.

Optional OFF check:
`python studies/allocation_off.py --source NEW_DIR/result.json --nominal
CIRCUIT.json --output NEW_OFF_DIR --self-fractions 0.1 0.5`

Use only accepted matched circuits. This is a five-transmon check: freeze rail
EJs, compare the original bias and a common nominal five-body OFF bias, then
retune coupler EJ inside the supplied allowed interval. Report physical
0100-0010, both logical exchange residuals and secular ZZ, including failed
roots/best minima. It does not automatically run gate/readout optimization.
Smaller local residuals do not certify a better gate. Recalibrate the gate and
validate integrated readout separately before adopting a redistributed circuit.

Historical A2-only records used a loaded-budget ground coordinate eta and
allowed other rail fractions to float. Preserve those records and their
converted-p report; they are not equivalent to the new constrained pair/A2
scope. Superseded EC-only, eta-based and one-edge-only optimization runners
were removed. The asymmetric-ground experiment is retained as historical data,
not automatically introduced as another current study path.

## Records and report helpers

Preserve the original nominal circuit, compensation values, controls, matrices,
failed matches, loading metrics and numerical qualifications. Loaded EC and
intended K coefficients are matching constraints; the remaining couplings are
residual diagnostics. Never transfer gate fidelity or OFF certification between
distinct matrices. Residual precision target is 1 Hz; gate probability target
is 1e-4 if separate gate validation is requested.

`studies/allocation_report.py` and `studies/target_circuit_report.py` are retained
renderers for the historical allocation and circuit-target report layouts.
They require the saved datasets named in their source and use Windows Arial
fonts. They do not launch physics searches and are not generic fresh-run report
entry points. Numerical allocation/OFF studies do not depend on PDF packages
or historical results. Generated reports and data are excluded from Git.
