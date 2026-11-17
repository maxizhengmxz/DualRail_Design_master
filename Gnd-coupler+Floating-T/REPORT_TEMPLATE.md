# Coupler design report template

Use this structure and level of detail for every subsequent full coupler-design
report. The detailed reference is
[the design report](templates/report-references/design-reference.pdf).
The approved reference PDF and its checksum manifest are intentionally checked
in under templates/report-references/. Generated case reports and figures stay
under ignored output/. This specification is the portable report policy.
The six-page timing benchmark is supplementary; it does not replace this format.
Reference case: loaded EC/h = (165,250,160,260) MHz, 29 October 2026.
Its numerical values and historical completion status are examples, not defaults.

## Visual format

Circuit schematics follow [CIRCUIT_TEMPLATE.md](CIRCUIT_TEMPLATE.md), including
the approved component orientations, terminal extensions, C1 lead, mathematical
typesetting and two-decimal circuit labels.

- EVERY page must have the same A4 portrait size (595.28 x 841.89 pt), with
  margins 42 pt left/right and 38 pt top/bottom. Do not insert landscape or
  larger figure sheets. Preserve grouped panel layouts and scale whole figures to fit portrait pages.
- Helvetica body 9 pt with 12 pt leading; main headings 17 pt, subheadings 12 pt.
- Pale blue table headers (#e7eff5), top-aligned cells, repeated headers,
  light horizontal rules, 6 pt horizontal and 5 pt vertical cell padding.
- Numbered, readable sections with deliberate page breaks. Footer includes
  topology, case ID, applicable OFF bound, report date and page number.
- Scientific figures with units, legends, actual constraint lines and marked
  ON/OFF points. Readout uses eight panels per pair, two traces per panel.
- Approximately 13-15 core pages at this case's complexity, including standalone
  contrast-map characterization; an explicitly requested pulse section may add
  approximately six pages including frequency trajectories. Preserve the content and
  readability rather than forcing an exact page count or shrinking text.

## Required report sequence and detail

The FIRST page is the approved full circuit diagram and its associated hardware
tables: all five loaded ECs, frequency tunabilities, asymmetric SQUID junction
EJs, asymmetries and effective idle/OFF and gate/ON EJs. Preserve the approved
diagram layout; keep capacitances on the drawing and EC/EJ values in the tables.
Save an exact standalone hardware page beside the design PDF. This page belongs
only to the design report, not the optional decoherence report. The sequence
below follows this hardware cover; do not discard the confirmed-input section.

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
   readout frequency; every readout constraint.
   Report both prescribed bare resonator intervals in GHz (four independently
   supplied bounds; do not impose equal centers or widths), the nominal
   reference, shared hardware and per-frequency EJ recourse. Maximize worst
   absolute mean chi across the range, not nominal/peak chi. Report preferred
   H>96% everywhere versus fallback H>95% over the same unchanged interval,
   selected H floor, achieved H/chi, audited status and optimization time.
   Explicitly flag any 95% fallback and absence of the requested extra margin;
   distinguish a bounded search miss from proven infeasibility. Preserve the
   fallback flag when presenting seven-mode validation. Historical designs with
   no optimization record must not be described as optimized by this policy.
   Coupler/resonator transition couplings, detunings and dispersive ratios at
   the final ON point.
   Include the standard standalone A1-r1-A2 and B1-r2-B2 2D contrast maps over
   supplied tuning ranges: bare-f01 axes, nominal stars, absolute 01-10/01-00/
   10-00 cavity contrasts with 0.5/1/2 MHz contours, shared orange >0.5 MHz and
   pink >1 MHz shading, blue <10 kHz logical band and flagged overlap <95%.
   Define 00 as pair ground and 10/01 as lower/upper hybridized branches; give
   grid, fixed hardware, basis checks and nominal contrasts. Contrast shading
   is not a readout-feasibility certificate. Seven-mode 2D versions are opt-in.
   Keep the six-panel readout layout grouped and scale it to fit one portrait page.
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


Optional robustness/decoherence reports use STUDY_REPORT_TEMPLATES.md and the exact latest PDFs under templates/report-references/. They are not part of the default design workflow.

## Optional finite-rise / bandwidth section

Include only after explicit pulse-study request; place after static CZ results.
The preserved reference contains this optional example, not a requirement to run
it for every new design. Use pulse_comparison_report.py for existing PDFs; it
only reads completed records and figures. Preserve historical input/validation
scope in the rest of the report.

- Define10-90% rise as the normalized EJ envelope's transition from0.1 to0.9,
  excluding its remaining tails. Give tanh K=4 and compact half-cosine formulas,
  full-edge support and EJ interpolation; do not call these flux pulses.
- Specify five transmons, fixed hardware/EC/J and A EJs, saved five-mode OFF,
  fixed coupler ON endpoint, simultaneously moved B controls and symmetric return.
  Use the same dressed idle logical input/output states through the entire pulse.
- State the ideal1GHz cutoff's coordinate and phase convention. Filter the whole
  pulse without clipping ringing; report programmed rise+hold+fall separately
  from finite simulation tail windows. No finite settling time is implied.
- Explain phase-pi calibration and raw-fidelity objective; raw, postselected,
  survival, hold and total durations must refer to the same pulse. Explain the
  survival normalization, local-Z correction and excluded noise/resonator physics.
- Include all ten retained rows, the four-panel figure, representative numerical
  checks and limitations. Do not claim all rows have independent1e-4 certificates.
- Give measured filtered-trial and larger-basis timings, missing timers and
  concurrent-vs-summed scope. Separate implemented stopping/caching improvements
  from proposed speedups; do not promise an unmeasured speed factor.
- Add frequency-versus-time diagrams for the saved tanh/cosine examples. Show
  stationary A, moved B and C; mark ramp, hold and return,10-90% EJ width and
  full-edge/total durations. Convert the filtered EJ through the loaded-EC
  full-cosine spectrum; label these bare local frequencies, not logical energies.
  Keep every page A4 portrait; keep the four-panel comparison together and scale it to fit.
