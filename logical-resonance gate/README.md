# Logical-resonance gates: iSWAP and controlled phase / CZ

This folder owns the existing five-transmon logical-transition gate studies.
Subsystem order is **A1, A2, C, B1, B2**: two encoded dual-rail qubits and an
explicit tunable transmon coupler. It is separate from
[readout optimization](../readout_workflow/README.md) and
[coupler-off analysis](<../coupler turn-off/README.md>).

This folder contains the three gate notebooks, their TOML configurations, and
the implementation modules in `logical_resonance_gate/`.
Their Hamiltonians, numerical algorithms, search grids, pulse settings,
acceptance thresholds, and iSWAP/CZ calculations are unchanged. General-purpose
circuit/readout utilities and regression tests remain in their existing locations.

The implementation is a normal Python package named `logical_resonance_gate`,
located beside these notebooks. Notebooks set their source path at entry;
the repository test runner does the same. No gate imports pass through core.
For an interactive shell, add this workflow directory to PYTHONPATH.

## What resonance means here

Each isolated pair has two one-excitation eigenmodes. Logical 0 is the lower
mode, approximately (|01>-|10>)/sqrt(2); logical 1 is the upper mode,
approximately (|01>+|10>)/sqrt(2). Neither is the pair ground state.

The logical transition frequency is the **splitting between these modes**:

    f_L,A = (E_1L,A - E_0L,A)/h
    f_L,B = (E_1L,B - E_0L,B)/h
    delta_L = f_L,A - f_L,B

The search coordinates are the two logical splittings and the coupler flux.
Physical pair fluxes are obtained from calibrated hybridization contours.
Matching logical transitions means f_L,A approximately equals f_L,B. It does
not mean matching the individual A2 and B1 transmons' 01 transitions. The
pair carrier frequency (mean ground-to-mode transition) is a separate quantity.

The full five-transmon Hamiltonian is diagonalized; encoded states are assigned
using isolated-pair products with the coupler in its ground state as references.
The logical basis order is 00, 01, 10, 11. The resulting effective logical
interaction, state overlaps, and nearby leakage levels determine whether a
nominal logical-resonance point is useful.

## iSWAP and CZ are different objectives

For **iSWAP**, near-resonant logical transitions permit exchange between
|01>_L and |10>_L. The desired interaction is logical XX+YY exchange, with a
duration producing a full logical swap up to the iSWAP phase convention and
correctable local Z phases. Detuning, residual ZZ, pair-creation terms and
other non-fSim components are competing errors. Resonance alone does not
guarantee a high-fidelity iSWAP.

For **controlled phase / CZ**, the same logical-frequency/coupler search
examines the state-dependent energy combination

    zeta = (E_11 - E_10 - E_01 + E_00)/h
    phi_CZ = phi_11 - phi_10 - phi_01 + phi_00.

A diagonal static interaction gives phi_CZ = -2*pi*zeta*t; CZ requires pi
modulo 2*pi after local-phase correction. The objective is useful conditional
phase with suppressed unwanted logical exchange and leakage. **CZ does not
require exact equality of the logical transitions.** Near resonance is a search
region, not the acceptance rule; the primary backend also screens off-diagonal
logical terms relative to ZZ and evaluates a bounded static coherent ceiling.

This differs from the colleague's approximate four-transmon CZ study, which
explicitly seeds a physical collision such as f_12,A2 approximately f_01,B1
(|0110> approximately |0200>). This workflow does not impose that collision
as its defining search condition, although physical avoided crossings can
influence the full logical interaction.

## Annotated file map

| File | Role |
| --- | --- |
| [logical_fsim.py](logical_fsim.py) | Primary configured marimo workflow: contours/calibration, static atlas, iSWAP/CZ candidate ranking/refinement, basis and branch checks, rectangular/smooth pulse sweeps, and trajectories. |
| [logical_fsim_re.py](logical_fsim_re.py) | Alternative, largely self-contained notebook with its own model construction, coarse gate scoring and slice-propagated shaped pulses. Preserved as an alternative implementation, not made identical to the primary workflow. |
| [configs/logical_fsim.toml](configs/logical_fsim.toml) | Existing production parameters, numerical grids and thresholds. |
| [configs/logical_fsim_smoke.toml](configs/logical_fsim_smoke.toml) | Existing small test configuration; not a validated device design. |
| [double_tdt.py](double_tdt.py) | Earlier mixed exploration of physical three-transmon fSim and five-transmon logical gates; retained alongside the configured notebooks. |
| [logical_resonance_gate/dualrail_fsim.py](logical_resonance_gate/dualrail_fsim.py) | Five-transmon backend. `build_logical_contour`: logical-frequency coordinates; `build_dualrail_fsim_model`: model and logical assignment; `evaluate_static_interaction`: logical detuning/exchange/ZZ; `rank_static_gate_candidates` and `refine_static_gate_candidates`: separate iSWAP/CZ selection; `simulate_dualrail_pulse` and `simulate_dualrail_population_trajectory`: dynamics. |
| [logical_resonance_gate/device.py](logical_resonance_gate/device.py) | Reference circuits, spectral checks and reduced-model calibration. |
| [logical_resonance_gate/logical_fsim_config.py](logical_resonance_gate/logical_fsim_config.py) | TOML validation and configuration fingerprints. |
| [logical_resonance_gate/logical_fsim_workflow.py](logical_resonance_gate/logical_fsim_workflow.py) | Cache compatibility and complete pulse-result persistence. |
| [logical_resonance_gate/fsim.py](logical_resonance_gate/fsim.py) | Transmon/solver definitions and physical three-transmon fSim tools used by these gate studies. |
| [../tests/test_logical_fsim_phase0.py](../tests/test_logical_fsim_phase0.py) | Config/cache checks and repeated headless notebook smoke execution for both gates. |
| [../tests/test_dualrail_fsim.py](../tests/test_dualrail_fsim.py), [../tests/test_device.py](../tests/test_device.py) | Shared backend and device regression coverage. |
| [Optimization plan](../docs/specs/five_transmon_dual_rail_gate_fidelity_optimization_plan.md) | Existing implementation assessment and agreed later integration/validation decisions. |

Circuit factories, runtime setup, tracking and operating-point utilities
remain shared under `core/`. Generic sweeps, Monte Carlo helpers and their
older notebooks are archived in `legacy/`, outside the active gate workflow.

## Running the existing workflow

From the repository root, using its existing environment:

```powershell
uv run marimo edit "logical-resonance gate/logical_fsim.py"
uv run marimo edit "logical-resonance gate/logical_fsim_re.py"
uv run marimo edit "logical-resonance gate/double_tdt.py"
```

The primary notebook defaults to the config beside it in `configs/`.
`LOGICAL_FSIM_CONFIG` still overrides that path; a relative override is resolved
against the working directory. `LOGICAL_FSIM_DATA` still overrides the data
directory. Configured relative output paths retain their working-directory
semantics: run from the repository root to retain existing `data/` and `tmp/`
locations and caches. Opening/running a notebook may perform calculations and
write outputs according to its existing execution gates.

```powershell
uv run python run_tests.py
```

## Units and scope

The shared package has been renamed from `dtc_dualrail` to `core`; project
callers now use `core.*`. [core/README.md](../core/README.md) documents the
dependency map and retained older utilities. Readout and coupler-off now
share `core.transmon` and `core.electrostatics`. This gate backend retains its
existing model and calibration; its physics has not been replaced.

Hamiltonian coefficients represent H/h in GHz, logical splittings and many
reported interaction rates use MHz, and evolution times use ns. Local charge
coupling J multiplies n_i*n_j; physical J01 is its extracted local exchange
matrix element, while logical exchange is a different encoded matrix element.

The moved workflow retains its existing flux-tunable SQUID models and circuit
calibration conventions. Relocation does not convert it to the newer
single-junction readout/coupler-off model or implement the proposed approximate
CZ integration. No new fidelity, convergence, or hardware qualification is
claimed by placing these files in this folder.
