# Coupler design report template

Use this structure and level of detail for every subsequent full coupler-design
report. The detailed reference is
[the 13-page design report](output/report-reference/coupler-design-report.pdf).
The visual PDF is a local reference under ignored output/, excluded from Git.
This checked-in specification is the portable template for a fresh checkout.
The six-page timing benchmark is supplementary; it does not replace this format.
Reference case: loaded EC/h = (165,250,160,260) MHz, 29 October 2026.
Its numerical values and historical completion status are examples, not defaults.

## Visual format

- A4 portrait, margins 42 pt left/right and 38 pt top/bottom.
- Helvetica body 9 pt with 12 pt leading; main headings 17 pt, subheadings 12 pt.
- Pale blue table headers (#e7eff5), top-aligned cells, repeated headers,
  light horizontal rules, 6 pt horizontal and 5 pt vertical cell padding.
- Numbered, readable sections with deliberate page breaks. Footer includes
  topology, case ID, applicable OFF bound, report date and page number.
- Scientific figures with units, legends, actual constraint lines and marked
  ON/OFF points. Readout uses eight panels per pair, two traces per panel.
- Approximately 13 pages at this case's complexity. Preserve the content and
  readability rather than forcing an exact page count or shrinking text.

## Required report sequence and detail

1. **Summary and all confirmed inputs.** Loaded EC tuple; idle and gate positions;
   splitting convention and reference; moved pair; selected CZ auxiliary;
   duration scope; topology; all frequency, EC, loading and tuning constraints;
   resonator targets/impedance; numerical tolerances. Distinguish supplied values,
   defaults, case overrides and achieved results. State incomplete validation.
2. **Toy channel selection and three-body design.** Common intended coupling,
   minimum-S1 source/auxiliary, S1, S2, unintended ratio, score basis/provenance,
   toy gate result. Explain ratio/EC search, verified OFF branches, intermediate
   three-body capacitors, loading, ON strength seed and verified OFF root.
3. **Five-body refinement and physical capacitances.** Actual capacitors and
   compensated self capacitors, each pad-ground value, loaded EC reconstruction,
   five/seven-mode side-by-side component table. Explain material refinement and
   rejected candidates without presenting them as accepted solutions.
4. **Capacitance budgets and OFF results.** Define own C, effective C and loading
   counterfactuals. Report A2/B1/coupler budgets and separately reconstructed four-
   coupler loading. Tabulate three/five/seven-mode OFF frequency and EJ, physical
   null, both logical exchanges, secular ZZ and connected-state detunings.
5. **Residual channels and wide diagnostic sweep.** Eight logical-to-B20/B02
   exchanges, connected detunings and |g/detuning|. Plot both logical residuals
   and ZZ over a range showing non-monotonicity. Mark allowed bounds; excursions
   are diagnostics, not accepted OFF solutions.
6. **Calibrated CZ gate.** Five/seven-mode ON frequency/EJ, moved-rail frequency,
   duration, phase error, raw and postselected fidelity, average logical survival,
   worst-input nonlogical population, per-input leakage and actual-bias channel
   scores. All values must refer to the same point and stated basis. State
   static dressed-reference scope and excluded pulse/decoherence physics.
7. **Standalone readout design and coupler/resonator diagnostics.** Each actual
   equal resonator link, intra-pair link, resonator self C/L, bare LC and dressed
   readout frequency; every readout constraint. Coupler/resonator transition
   couplings, detunings and dispersive ratios at the final ON point.
8. **Integrated dressed-idle calibration.** Original and calibrated individual
   bare frequencies/EJs, dressed operating means and logical splittings; common
   OFF before/after. Explicitly state fixed geometry and loaded EC. Explain
   differential dressing; do not call unequal bare settings unequal dressed rails.
9. **Readout at calibrated centers.** Standalone/integrated metrics for each
   pair: dressed mean, logical gap, worst dispersive ratio, chi mismatch, signed
   mean chi, H score, overlap, all failures and independent EJ sign-grid status.
10. **Rail A +/-100 MHz sweep.** Eight constraint panels, standalone and full
    system in each panel; each trace uses its own calibrated center and resulting
    dressed-frequency x axis. Freeze differential settings/hardware/other controls.
11. **Rail B +/-100 MHz sweep.** Same conventions. Do not invent extra frequency
    bounds or retune each sweep point. Distinguish nominal traces from sign grids.
12. **Numerical validation.** Exact local/pair/coupler/photon bases; fixed-control
    differences for physical/logical exchanges, ZZ, fidelities and per-input
    leakage. Give actual changes, tolerances and pass/unconverged status. Include
    adaptive readout audit counts and near-boundary larger-basis checks.
13. **Timing and reproducibility.** Per-stage elapsed times, concurrent-parent
    wall time vs summed worker time, retries and follow-up audits. Separate
    measured numerical time, coordination and reporting. Identify final records,
    source hashes and excluded/failed attempts; never fabricate a complete timer.

## Rules for subsequent reports

- Use current WORKFLOW.md and the toy channel authority. Do not copy the
  reference's historical provisional labels, obsolete controls or numbers.
- Report the final calibrated gate, rather than reusing the reference's
  explicitly historical gate result. Include the final B-center basis audit.
- Required data that are unavailable must be explicitly marked **not evaluated**
  or **unconverged**; never silently omit a section or imply validation passed.
- The 1 Hz residual criterion is a basis-agreement tolerance, not a requirement
  that all physical/logical residuals have magnitude below 1 Hz.
- Keep fidelity precision and fidelity quality targets distinct. State whether
  optimization ended on metric plateau, target attainment or finite budget.
- Preserve raw numerical records separately from prose and figures. Authoring
  must not rerun optimization, overwrite provenance or change stored physics.
- Use the PDF skill, render and visually inspect every page before delivery.
  Keep final PDFs and figure inputs; move rendering previews and one-off scripts
  into ignored archives when cleaning up.
