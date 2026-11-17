# Grounded coupler with floating rail transmons

The default workflow remains [WORKFLOW.md](WORKFLOW.md): confirm inputs; solve
A2-C-B1; refine all five transmons and calibrate the S1-selected CZ; validate with
both resonators. Loaded rail ECs are immutable during design. Compensation of
self capacitances preserves them. S1/logical definitions come from
`../2Q-gate_XM/WORKFLOW.md` and `core.cz_channel_policy`.
The temporary physical OFF target is documented in OFF_POLICY.md.

The standard standalone readout goal is prescribed-resonator-range optimization:
when an actual design starts, ask for four independent GHz inputs (A minimum,
A maximum, B minimum, B maximum), then maximize the worst absolute
mean chi across it, preferring H>96% everywhere. If no audited solution is found,
retry H>95% over the same range and explicitly report the reduced margin. If both fail,
integration rejects the result. Loaded ECs and existing calibration/physical
constraints remain fixed. See WORKFLOW.md stage 5. This runs during design,
not during report generation or ordinary frequency sweeps.
The standalone designer now calls the physical-pad prescribed-range adapter,
returning the winning nominal pad model and its range-validation witnesses.
It supports independent resonator references without a 6 GHz assumption.
Nominal feasibility is only an initialization. Workflow updates require no EC
tuple or numerical range. Request bounds only when a concrete design starts;
different A/B centers and equal widths are allowed but never imposed.
Seven-mode handoff rejects newly generated nominal-only optimization records;
historical files remain readable without being relabelled as range optimized.

## Optional studies

The opt-in parasitic study fits one actual port capacitor at a time against a
saved nominal circuit, then evaluates static CZ, physical OFF, logical exchanges,
ZZ and idle participation. See [PARASITICS.md](PARASITICS.md) and
[OPTIONAL_STUDIES.md](OPTIONAL_STUDIES.md) for masks, minimum-change objectives,
commands and numerical qualifications. It never launches during normal design.
The complete zero-parasitic nominal parameters are saved as
`configs/Design_V1_20261105.json`. The approved report layout is preserved in
`templates/report-references/parasitic-compensation-reference.pdf`. At the user's
request, the parasitic calculation records and archives were removed. Future
studies use a fresh output folder. Report generation reads that study's saved data.

The explicitly invoked five-transmon tuning-pulse study is documented in
[PULSED_CZ_WORKFLOW.md](PULSED_CZ_WORKFLOW.md). It starts/ends in dressed idle
logical states and is separate from static ON gate benchmarks. Use pulse_cz.py.
The opt-in pulse_shape_comparison.py branch compares tanh/cosine EJ edges with
the stated control bandwidth. Its saved-data report section is optional; see
OPTIONAL_STUDIES.md and PULSED_CZ_WORKFLOW.md. Report assembly never runs a gate search.

Neither is part of a normal design run. Explicitly invoke `study.py robustness`
or `study.py decoherence`; both take a saved nominal input and output folder.
`study.py report` renders saved data only. See [OPTIONAL_STUDIES.md](OPTIONAL_STUDIES.md)
for sampling, noise assumptions, constraints, rescue policy, resume and examples.
Its default-policy table distinguishes robustness-only rescue from nominal
constraints and documents the unresolved nominal seven-mode OFF basis accuracy.
The old fabrication_study.py/fabrication_report.py paths retain compatibility
imports; study.py is the supported runner.

## Source and validation

- `floating_coupler/`: circuit, physical/logical OFF, CZ and precision kernels.
- Existing stage CLIs: three-body design, five-body refinement, idle/gate
  calibration and seven-mode validation. No automatic full workflow on import.
- `../readout_workflow/readout_shift_map.py`: standard standalone readout-triple contrast maps in
  the design report; see ../readout_workflow/READOUT_MAPS.md. Seven-mode 2D maps are optional.
- `studies/`: optional study runner, consolidated physics/reporting, archive I/O.
- `templates/`: circuit template and exact latest design/decoherence/robustness
  report references. Use REPORT_TEMPLATE.md and STUDY_REPORT_TEMPLATES.md.
- `tests/`: focused regression suites. Run `run_tests.py` after numerical changes.

All frequencies/couplings are cyclic GHz internally; physical capacitances fF.
Residual precision target1 Hz; gate probability target1e-4 absolute. Acceptance
and convergence are separate. Preserve failed diagnostics and explicit bounds.

PDF report helpers additionally require the optional `reportlab` and `pymupdf`
packages. In a fresh project environment, install them with
`uv pip install reportlab pymupdf`. Rendering/visual review uses Poppler's
`pdftoppm`; it is not needed to run the physics kernels. Report-only commands
read existing results and do not launch optimization.

## Retained results

- `output/nominal-design/`: authoritative nominal inputs, capacitances, controls,
  validation/readout/gate results, timing and reports. The seven-mode physical
  OFF residual is not retrospectively certified to1 Hz.
- `output/decoherence/`: approved nominal decoherence data, figures and final PDF.
- `output/robustness/`: final summary PDF plus ONE verified robustness-data.tar.xz
  containing all9,456 retained evaluation records and reproduction inputs.
- `output/RESULTS.json`: current index and checksums.

Superseded campaigns, previews, duplicate PDFs, patch/launch scripts and caches
are removed after archive/preservation verification. Other workflows and unrelated
Git changes are untouched. See CLEANUP.md for the cleanup record.
