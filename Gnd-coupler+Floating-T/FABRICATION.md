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
