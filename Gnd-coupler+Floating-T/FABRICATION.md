# Fabrication robustness (optional)

The supported workflow, nominal constraints, cumulative recovery policy,
independent/max-stratified sampling and report rules are consolidated in
[OPTIONAL_STUDIES.md](OPTIONAL_STUDIES.md).

Use `study.py robustness --help`. The five-transmon/readout numerical kernel
remains `floating_coupler/fabrication.py`; the standard readout rescue ladder is
`readout_workflow/recovery.py`. No fabrication-time EC compensation is performed.
No study runs automatically during nominal design or import.

Historical output folders are replaced by one verified data archive and the
latest final report under output/robustness/. Use `study.py verify-archive` or
`study.py report --kind robustness --study <archive> --output <folder>` to inspect
or reproduce reports without rerunning device calculations.


## Resonator frequencies in new robustness runs

The existing workflow now samples bare A/B resonator frequencies independently
within user-supplied GHz intervals at fixed 50 Ohm. Supply
`resonator_frequency_study` with `ranges_ghz.A`, `ranges_ghz.B`, and
`impedance_ohm: 50` in the study input; missing bounds fail before scheduling.
This replaces resonator-self-capacitor errors. Cr and Lr are reconstructed
together; all other capacitor perturbations and readout -> five-mode OFF ->
gate recovery stages remain unchanged. No self-capacitance compensation is
added. Frequency values are separate from capacitor-deviation statistics.
Historical fixed-L capacitor-only archives retain their original interpretation.
