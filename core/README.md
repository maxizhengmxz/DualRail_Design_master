# Shared physics and retained analysis utilities

This package replaces `dtc_dualrail/`. Project imports now use `core`.
The distribution/environment name in `pyproject.toml` remains `dtc-dualrail`;
that metadata name is independent of the Python package name. The colleague's
unintegrated source has not been edited: adapt its old package imports when
integrating it.

## Shared physics

| Module | Definitions | Used by |
| --- | --- | --- |
| `transmon.py` | `Basis`, `transmon_spectrum`: full-cosine local energies and charge matrices | Readout and coupler-off |
| `electrostatics.py` | Loaded EC, charge coefficient J, local exchange J01 | Readout and coupler-off |

The approximate `transmon-based CZ` workflow also uses
`projected_transmon_operators` and the explicitly named reduced two-node
capacitance conversions. They preserve its effective-EC/reference-basis
conventions; they do not reconstruct physical shunts or redefine loaded EC.
The older local solver and existing workflow defaults are unchanged.

The local Hamiltonian is `H/h = 4 EC n^2 - EJ cos(phi)` at zero offset charge.
Energies are cyclic GHz, and n counts Cooper pairs. `Basis` preserves existing
truncation fields; the local solver uses only ncut and transmon_levels.
For an inverse capacitance matrix in 1/F:

    EC_i [GHz] = e^2 (C^-1)_ii / (2 h 1e9)
    J_ij [GHz] = 4 e^2 (C^-1)_ij / (h 1e9)
    H_int/h = J_ij n_i n_j
    J01_ij [GHz] = |<10|H_int/h|01>|

Only off-diagonal J entries are interactions; the diagonal is 8*EC, not another
self-interaction. J01 includes numerical local charge matrix elements. It is
not J, an angular rate, or the resonant two-state gap (2*J01). Physical loaded
EC and J come from the same inverse matrix. The effective toy-gate convention
remains documented in the [gate plan](../docs/specs/five_transmon_dual_rail_gate_fidelity_optimization_plan.md).

## Dependency direction

```text
readout_workflow -> core.transmon + core.electrostatics
coupler-off      -> core.transmon + core.electrostatics
coupler-off      -> readout_workflow.calculation (readout diagnostics only)
logical-resonance gate -> logical_resonance_gate (workflow-local package)
logical-resonance gate -> core circuit/operating-point/storage utilities
```

Shared physics imports neither workflow. Coupler-off uses readout's Device,
ReadoutModel, annotate and failures for readout checks, without launching its
optimizer. Circuit topologies, capacitance budgets, conditions and searches
remain workflow-owned. The old gate backend retains its existing SQUID model;
this extraction does not silently replace its physics.

Public `core` exports are lazy: importing shared physics does not load the
gate backend or Monte Carlo modules. Readout's historical Basis/_transmon and
coupling imports remain compatibility aliases; current internals use core
directly. Readout checkpoint identity includes both shared physics sources.

## Remaining modules and usage

| Modules | Used by the three workflows? | Purpose |
| --- | --- | --- |
| `circuits.py`, `operating_points.py` | Logical-resonance gates | Reference circuits and pair contours |
| `operating_point_dispersive.py`, `tracking.py` | Gate exploration in double_tdt and its dependencies | Dispersive shifts and spectral assignment |
| `constants.py`, `runtime.py`, `io.py` | Logical-resonance gates and supporting utilities | Defaults, thread limits and persistence |
| Former Monte Carlo, sampling, generic sweep and tunable-resonator helpers | **Not used by the three main workflows; moved out of core** | Now in `../legacy/analysis/`, with callers in `../legacy/notebooks/` and dedicated tests in `../legacy/tests/` |

See [legacy/README.md](../legacy/README.md) for the exact caller map.
Gate and legacy forwarding exports have been removed. Dependencies point
from workflows to core, never from core back into a workflow or archive.
