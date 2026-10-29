# Transmon-based CZ: approximate physical-collision search

Four full-cosine transmons A1,A2,B1,B2 encode two dual-rail qubits. The
coupler-mediated interaction is approximated by an effective direct A2-B1
charge coupling. There are no explicit resonators or coupler dynamics here.
Physical collisions provide seeds for full four-transmon logical dynamics.
The intended logical source is selected by minimum S1 at common intended g, not
inferred from a physical collision label. This is separate from the
logical-transition iSWAP/CZ workflow.

[WORKFLOW.md](WORKFLOW.md) is the authoritative definition shared with the
coupler workflow. `transmon_cz/channel_policy.py` implements the logical ordering
and minimum-S1/common-g selection rule. Logical source and physical auxiliary must both
be recorded for ON. Explicit-coupler OFF temporarily nulls physical 0100-0010
exchange in five/seven modes using all four physical single-excitation references;
its three-body seed nulls physical single-excitation 01-10 exchange.
Logical exchanges, CZ-auxiliary couplings and secular ZZ remain residual
diagnostics; see WORKFLOW.md for the open physical-versus-logical choice. Existing
fixed-source sweeps are historical diagnostics, not minimum-S1 certification.

## Temporary design policy

All four `transmons.<label>.EC_GHz` inputs are independent in the simulator
and constrained candidate search. `configs/gate_search.json` defaults to the
colleague's symmetric choice EC_A1=EC_B1=0.200 GHz and EC_A2=EC_B2=0.350 GHz.
These are **temporary exploratory seeds**, not a validated device design.
Similar A/B logical frequencies may complicate idle operation, but are not
alone proof that cancellation is impossible. This toy model cannot certify
an idle/off position, a physical tuning path, or readout feasibility.

Later, choose EC/design inputs from the readout feasibility map and validate
gate/idle compatibility in the full circuit. No readout or coupler-off
optimization is invoked automatically by this workflow.

## Definitions and dependency direction

`core/transmon.py` owns full-cosine projected charge, squared-charge and cosine
operators, including optional offset charge ng. The reference-basis convention
is retained from the colleague's implementation. The existing zero-ng
`transmon_spectrum` used by readout/coupler-off is unchanged.

`core/electrostatics.py` owns EC/capacitance conversions and charge-coupling
helpers. Energies are cyclic GHz, time is ns, and QuTiP evolution uses 2*pi
times H/h in rad/ns. `couplings_GHz` means J in H_int/h=J*n_i*n_j, not J01.
J01 is extracted using local charge matrix elements.

For this toy search EC is the effective local Hamiltonian coefficient, already
representing omitted loading. Do not add another loading correction to it.
The imported two-node capacitor helper is a nominal prescription for J,
not a reconstruction of full-circuit physical shunts. Its EC-derived nominal
diagonals do not imply replacing the ECs in the toy Hamiltonian.

When restoring the full circuit, target its loaded EC_i/h =
e^2*(Cfull^-1)_ii/(2*h) to the EC actually used in this toy Hamiltonian.
Match dressed transitions and the combined effective interaction separately.
Do not double-count coupler mediation as an additional physical direct link.

Dependencies point from `transmon_cz` to `core`, never the reverse. It does not
depend on `logical_resonance_gate`, `legacy`, or the optional colleague idle
search. Existing workflows remain independent.

## What is searched and reported

`gate_search.find_gate_operating_points` scans EJ_A1 and solves EJ_A2, EJ_B1
and EJ_B2 for the two signed physical rail detunings and the selected bare
CZ-collision detuning. All roots lie in the supplied EJ range. Optional filters
apply EJ/EC, other-collision separation, and first-odd-pi logical gate criteria.
The legacy explicit `ec_a1_b1_GHz` keyword intentionally requests symmetric
EC shorthand; omit it to use the four independent EC fields.

Logical 0/1 are the lower/upper isolated-pair one-excitation eigenmodes.
Products are ordered 00,01,10,11. The static four-transmon interaction acts
during evolution, without ramp modeling. CZ phase is phi11-phi10-phi01+phi00.
Reports distinguish unconditional fidelity, erasure-conditioned fidelity,
acceptance, wrong-logical population, and detectable/undetectable leakage.
Always interpret conditional fidelity together with acceptance.

H_A/H_B use the same normalized charge-transition-balance definition as our
readout workflow, but here for isolated pairs without resonator dressing.
Both logical frequencies and their signed A-minus-B detuning are diagnostics;
there is no new hard logical-frequency separation filter.

## Explicit commands

From the repository root:

```powershell
uv run python "transmon-based CZ/run.py" --help
uv run python "transmon-based CZ/run.py" select-channel s1-candidates.json --output selected-channel.json
uv run python "transmon-based CZ/run.py" simulate
uv run python "transmon-based CZ/run.py" search
uv run python "transmon-based CZ/run.py" sweep detuning
uv run python run_tests.py --legacy
```

`select-channel` ranks supplied verified S1 candidates evaluated at common g; it performs no
spectral search. The blank `configs/channel-selection-template.json` documents
the input. Missing state verification, unequal intended strengths, old first-crossing records or unresolved S1 ties fail explicitly.

No arguments show help. Simulate/search/sweep commands deliberately launch
calculations and write output; the configured production grids can be costly.
Importing the package or sweep helpers launches nothing. Relative simulation
output paths resolve against the input configuration directory. Search output
defaults to this folder's ignored `output/` directory.

The retained grid sweeps are explicit design studies: their EC axes change A's
low/high EC and preserve the configured B-minus-A EC offsets. Symmetric inputs
reproduce the original symmetric grids; these are not a general four-EC
optimizer. Use the candidate search for independently specified ECs. Select
one sweep at a time; the old script's automatic sequence was removed.

The detailed runner reports coherent CZ metrics only and rejects non-null
noise. Low-level evolve_state still supports population evolution with noise;
four basis inputs alone do not provide noisy process fidelity. The physical
CZ diagnostic remains a coherent calculation.

## File map and provenance

- `transmon_cz/simulation.py`: four-transmon construction, preparation, dynamics,
  gate/leakage metrics and pair diagnostics.
- `transmon_cz/gate_search.py`: gate-only constrained search and persistence.
- `transmon_cz/experiments/`: detailed single-point runner, explicit sweep
  functions, and spectroscopy tools.
- `configs/`: illustrative inputs, not certified fabrication parameters.
- `tests/`: shared-physics, population-closure, independent-EC search and
  import-safety regression checks.

Adapted from the colleague's `cz_gate/core` and `cz_gate/experiments` package.
The original folder is untouched. Saved results, bytecode, macOS metadata,
optional TDT/readout bridge experiments and missing idle-search APIs were not
imported. No existing requirements/lockfile was replaced.
