# Optional study report templates

These are project report references, not auto-executing workflows or case defaults.

- `templates/report-references/decoherence-reference.pdf`: latest approved core
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

Page 1: circuit context and all five asymmetric SQUID designs (four rails plus
coupler), listing loaded EC, tunability, large/small junction EJs, asymmetry and
effective idle/OFF and gate/ON EJs. Remove these EC/EJ parameter labels from the
schematic; retain component names, resonator frequencies and capacitor labels.
Current hardware ranges are
A1/A2 4-5 GHz, B1/B2 4.4-5.8 GHz; populate from each case's explicit assumptions.
Page 2: isolated-rail two-panel figure. Panel A: logical A at idle, logical B at
idle/gate, B20-B0 and isolated B1 0-1 only at gate; echo/Ramsey for each. Panel B:
ideal B20-inclusive CZ ERROR (1-fidelity), echo/Ramsey raw/selected on a log y-axis,
including independent A-idle and B-gate exponential generators. Keep A in pair
ground for the auxiliary; do not model it as logical A0. x=1-100
microPhi0/sqrt(Hz) per independent physical loop, log scale;
times in us. Then B details, separate coupler frequency
response, coupler-induced dephasing-time plots, and a qualified comparison.
Do not restore the superseded combined-noise section by default.
Save an exact standalone copy of page 1 as circuit-and-squid-design.pdf while
retaining the page in the main report. Coupler-induced noise remains separate
from the central A+B intrinsic-noise gate estimate.
When tuning-comparison.json is supplied, append a clearly labelled alternative
comparison page. Preserve the nominal 4-5 GHz A range and core curves; the
500 MHz 4.36-4.86 GHz range remains an alternative, not an adopted design.

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
