# Fixed-hardware cavity-shift maps

These explicitly requested diagnostics do not optimize capacitors or retune OFF
at each pixel. All frequencies and charge couplings are cyclic; plots use MHz
for cavity differences and GHz for bare single-transmon frequency axes.

These maps are an OPTIONAL readout diagnostic, enabled only on explicit request.
They do not run with the default readout search. The coupler-design workflow
delegates its standard post-design contrast-map report to these readout-owned
entry points, after independent links are fitted. The seven-mode 2D
extension below is opt-in; it is not required for a normal design report.
Retain the usual integrated idle and common-frequency constraint sweeps.

`readout_shift_map.py` uses separate pair/resonator triples and saved nominal
readout hardware. Its initial grids are 61x61, with local6/photon3 and selected
local8 checks. Three-state plots compare |chi01-chi10|, |chi01-chi00| and
|chi10-chi00| at levels 0.5,1,2 MHz. Orange means every gap exceeds 0.5 MHz;
pink means every gap exceeds 1 MHz. Gray denotes minimum reference assignment
overlap below 0.95. Full data are retained, including gray regions. These gap
thresholds are requested spectral-contrast diagnostics, not measurement fidelity.

Use explicit new-case bounds (no inherited A/B frequency defaults):

```powershell
python -m readout_workflow.readout_shift_map --input <nominal-input.json> --output <maps-folder> --pair A --points 61 --frequency-bounds-ghz 4 5
python -m readout_workflow.readout_shift_map --input <nominal-input.json> --output <maps-folder> --pair B --points 61 --frequency-bounds-ghz 4.4 5.8
python -m readout_workflow.readout_shift_map --output <maps-folder> --three-state-contours --frequency-axes --dual-threshold
```

The ranges above reproduce the retained case only; use each new case's supplied
tuning ranges. Keep source data and basis audits. A/B calculations can run as two
processes with one BLAS thread each. Replotting uses saved results only.

`readout_maps_report.py --report <design.pdf> --maps <maps-folder> --output
<updated-design.pdf> --date YYYY-MM-DD` inserts a methods/nominal
summary page and one A4 portrait map page with the original six-panel layout after the calibrated-idle readout
section, including when an optional pulse section has shifted page numbers. Use
`--after-page N` only to override that anchor explicitly. This
report-only helper imports no physics solver and replaces its own insert on a
repeated edit. Existing design/gate/readout validation status remains unchanged.
For coupler design reports use ../Gnd-coupler+Floating-T/REPORT_TEMPLATE.md and place this material in the
standalone readout section; the PDF insertion helper supports existing reports.

Seven-mode extension: `seven_readout_shift_map.py --cell <calibrated-cell.json>
--output <new-folder> --pair A|B --points 31`. Use the retained seven-mode idle
calibration, including its differential rail trims and final OFF EJ. Freeze
hardware, coupler EJ and the unswept rail EJs. Scan the selected pair over A4-5GHz
or B4.4-5.8GHz independently. BOTH other-pair logical states are evaluated at each
point; no spectator-ground approximation or implicit choice of one logical bit.
All five transmons and both readout resonators remain in the Hamiltonian.

Screening basis is local6/pair12/C3, photons0,1,2 in EACH resonator, ncut35.
The sparse solver uses the identical Hamiltonian and dressed-matter references
as SevenReadout. Its low-energy window grows45->70->100 when a selected branch
approaches the edge or a target-reference overlap is below0.95. A missing target
can otherwise falsely match an unrelated low-energy state. Benchmark against
dense diagonalization before production.
Check nine grid locations and nominal idle using C5 at frozen controls. This
does not certify 1 Hz OFF convergence or that the nominal OFF bias stays an
exchange null away from nominal rail frequencies. Report the saved OFF limitation.

`plot_seven_readout_maps.py --output <completed-folder>` only processes saved
grids. It produces individual spectator0L/1L figures and a conservative BOTH
figure. In the latter, each contour is the minimum gap over spectators, shading
requires every gap to pass for both states, and assignment validity also applies
to both. Blue matching bands require logical gaps below10kHz in both states.
Narrow seven-mode blue bands use grid interpolation: they are screening estimates,
not the locally refined narrow boundaries of the standalone maps. Record the
31x31 resolution explicitly. Bare frequency axes use the SAME loaded ECs and
full-cosine inversion; do not use dressed logical-transition frequencies.

The three pair-state plot labels are 00=pair ground, 10=lower one-excitation-like
branch, 01=upper branch. These are not unmixed physical occupations. Seven-mode
references are the existing full-matter-dressed branches with logical spectator;
record matter-product overlap separately from photon/matter assignment overlap.
Avoided crossings can invalidate a simple spectator/product interpretation.
For the new seven-mode display, gray applies when EITHER assignment stage falls
below the existing 0.95 overlap reference screen. This conservatively flags
cross-rail or auxiliary hybridization even when the photon/matter assignment is
good. It does not discard the saved raw data or change the existing readout
optimizer's acceptance definitions.

## Ownership and integration

All map runners, plotting, map-report insertion and map tests belong here.
The three-body Hamiltonian remains ReadoutModel. Floating-pad input reduction
uses the existing coupler circuit adapter, and the explicitly requested seven-mode
map reuses the coupler Hamiltonian; these imports do not run design searches.
Saved numerical results remain in their existing ignored output folders. Moving
source changes checkpoint hashes; do not relabel an old checkpoint as resumable.
Use module commands from the repository root; direct script paths also work.
