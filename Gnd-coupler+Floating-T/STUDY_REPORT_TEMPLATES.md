# Optional study report templates

These are project report references, not auto-executing workflows or case defaults.

- `templates/report-references/decoherence-reference.pdf`: latest approved six-page
  report, preserved after the coupler-dephasing plot was restored.
- `templates/report-references/robustness-reference.pdf`: latest approved final
  twelve-page summary. No per-record appendix and no Spearman heatmap.
- `templates/report-references/design-reference.pdf`: detailed nominal-design
  reference; governed separately by REPORT_TEMPLATE.md.

Use the references' section order, typography, tables, circuit style and level
of explanation. Populate them from the new study's actual inputs/results. Never
copy a reference result or historical constraint as a new default.

Every new report must distinguish baseline passes, rescued passes and failed
diagnostics. List the active rescue tier and numerical convergence separately
from threshold acceptance. The retained nominal seven-mode OFF has an unresolved
1 Hz basis-convergence limitation; preserve that annotation when citing its
results. See OPTIONAL_STUDIES.md for the saved audit and default-policy table.

## Decoherence

Lead with the isolated B-rail two-panel figure. First panel: logical B, B20-B0,
and isolated B1 0-1; idle/gate; echo/Ramsey. Second: ideal B20-inclusive CZ,
echo/Ramsey reference rates, raw/selected. x=1-100 microPhi0/sqrt(Hz), log scale;
times in us. Then circuit/inputs, SQUID parameters, separate coupler frequency
response, coupler-induced dephasing-time plots, and a qualified comparison.
Do not restore the superseded combined-noise section by default.

## Robustness

Present all records together. State sampling, failures, diagnostics and missing
values; do not split by calendar/run. Keep full numerical records in the single
archive, not a PDF appendix. Rejected readout positions and finite diagnostic
(gate not accepted) results are included and flagged.

Conditional error-density panels: x=max absolute capacitor deviation (%), .25%
bins; y=1-F from1e-6 to0.1, uniformly logarithmic bins. Fold all lower probability
into the bottom bin and all higher probability into the top bin. Every supported
vertical column integrates to one in the displayed log coordinate. Show counts
below. Gray means no observed density; counts distinguish unsupported columns.
A dashed line at1e-4 denotes numerical accuracy, not a physical floor/pass cut.
Use a distinct hue per density decade and shades inside each decade.

Include the same conditional-density views for |g01L,10L| and |g00L,11L| at
physical OFF; H/h in Hz, magnitude displayed, signed values retained. OFF is
physical0100-0010, not a logical null.

Retain conditional deviation profiles (median and interquartile range, not
mean/std) and supported two-capacitor failure/fidelity maps with sample counts.
Use only independent all-capacitor draws for these associations. Do not add a
Spearman heatmap; optional statistical summaries must explain signed rank
association and not call it failure probability. Ignore sub-precision color
variation as evidence of resolved fidelity changes.

Render every PDF page and inspect changed figures before delivery. Reports must
not launch optimization. Reusable renderers live in studies/; generated results
remain under ignored output/.
