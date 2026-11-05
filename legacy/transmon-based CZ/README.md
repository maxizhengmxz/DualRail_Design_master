# Archived transmon CZ research

This package preserves the original four-transmon physical-collision simulator
and search. New toy CZ and explicit-coupler toy-stage calculations use
[../../2Q-gate_XM/](../../2Q-gate_XM/README.md). Shared logical definitions,
S1/S2 and selected-channel policy belong to core.cz_channel_policy, with the
written specification in ../../2Q-gate_XM/WORKFLOW.md. The local policy module
is a compatibility re-export. Active calculations do not import this package.

## Historical physics

Four full-cosine transmons A1,A2,B1,B2 encode two dual-rail qubits. Coupler
mediation is approximated by direct A2-B1 charge coupling. No explicit coupler
or resonators are present. EC is the loaded local Hamiltonian coefficient.
Energies are cyclic GHz, time is ns, and couplings are J in H_int/h=J*n_i*n_j.
Logical 0/1 are the lower/upper isolated-pair one-excitation modes, with pair
ground separate. Logical source and auxiliary are explicit channel inputs.

Archived numerical algorithms and example inputs retain their previous behavior.
Example configurations are historical seeds. They do not establish a validated
physical design or modern channel certification. The old campaign's calculation
source differed from this tracked simulator. Its snapshot is preserved in the
summary archive alongside the original configurations.

## Explicit historical commands

Run from the repository root. No arguments show help. Imports launch nothing.

```powershell
.venv/Scripts/python.exe "legacy/transmon-based CZ/run.py" --help
.venv/Scripts/python.exe "legacy/transmon-based CZ/run.py" simulate
.venv/Scripts/python.exe "legacy/transmon-based CZ/run.py" search
.venv/Scripts/python.exe "legacy/transmon-based CZ/run.py" sweep detuning
.venv/Scripts/python.exe run_tests.py --legacy
```

Numerical commands can launch expensive calculations. Relative output paths
still resolve against the archived input configuration directory. Generated
outputs belong under ignored output/ directories. Default root test discovery
excludes this package. The root --legacy option adds its tests and other
archived tests. Active policy/import-isolation tests stay in ../../tests/.

## Files and retained data

- transmon_cz/simulation.py holds the archived full-cosine simulator.
- transmon_cz/gate_search.py and experiments/ hold historical searches and plots.
- configs/ contains original example inputs.
- tests/ checks archived physics and shared-policy compatibility.
- CLEANUP.md records the source move and result preservation.

The campaign's 82,008 per-point JSON checkpoints were removed on 2026-11-05.
Available maps, compact point-table CSVs, configuration copies, manifests and
the calculation-source snapshot remain in
../output/transmon-cz/historical-summary.tar.xz. The adjacent index.json records
member checksums, archive hash and table sizes. This summary cannot resume the
deleted campaign. The old README's claimed 59-page final report was already
missing before cleanup. Available PDF maps are preserved.
