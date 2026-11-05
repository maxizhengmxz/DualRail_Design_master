---
title: Using readout_workflow for One Dual-Rail Pair
aliases: [Dual-rail readout operating point, Single-pair readout study]
tags: [dual-rail, erasure-readout, operating-point, readout-workflow]
status: working-guide
---

# Using `readout_workflow` for one dual-rail pair

This note introduces the current single-pair erasure-readout model. The companion [[../../examples/readout_workflow_single_pair_example.py|interactive example]] evaluates a fixed fabricated circuit, searches its tunable junction energies, and optionally runs the design optimizer.

> [!important] Two different questions
> **Operational search:** hold all capacitances fixed and vary only tunable $E_{J1},E_{J2}$.
>
> **Design search:** before fabrication, also optimize $C_{12}$ and the nominal common readout capacitance $C_{1r}=C_{2r}$.

Do not interpret the full capacitance map as controls available after fabrication.

## Dependency structure

`readout_workflow` does not use `core.gen_tdt` or the flux-contour routines. It owns its three-node circuit and search logic while importing shared low-level physics:

```text
core/transmon.py          core/electrostatics.py
         \                       /
          readout_workflow/model.py
                       |
        calculation.py, optimization.py,
        diagnostics.py, validation.py, workflow.py
```

| File              | Responsibility                                                                           |     |     |
| ----------------- | ---------------------------------------------------------------------------------------- | --- | --- |
| `model.py`        | Physical capacitance matrix, Hamiltonian, dressed-state assignment, $H$, and $\chi_\pm$. |     |     |
| `calculation.py`  | Current thresholds, diagnostics, fixed-pixel feasibility search, and basis validation.   |     |     |
| `optimization.py` | Maximize $\bar\chi$ at fixed $(C_1,C_2)$ and continue nearby solutions.                  |     |     |
| `validation.py`   | Recompute at the audit basis and report couplings/metrics.                               |     |     |
| `workflow.py`     | Schedule and checkpoint the full capacitance map.                                        |     |     |


## Physical model

The device contains two single-junction transmons and one shared resonator. Its physical capacitances are

- transmon shunts $C_1,C_2$;
- fixed inter-transmon coupling $C_{12}$;
- fixed resonator couplings $C_{1r},C_{2r}$;
- resonator mode capacitance $C_r$ and inductance $L_r$.

The full inverse capacitance matrix determines both loaded charging energies and charge-charge interactions. Junction parasitic capacitances are omitted.

The literal model uses direct single-junction energies. If the hardware is SQUID-tunable, treat `ej1_ghz` and `ej2_ghz` as effective $E_J(\Phi)$ values and separately enforce the accessible flux range, asymmetry, and junction-capacitance model needed for your device.

## Main observables

`ReadoutModel.evaluate(EJ1, EJ2)` returns:

- dressed logical transitions and their mean operation frequency;
- logical splitting;
- hybridization score $H$;
- conditional shifts $\chi_-$ and $\chi_+$;
- state-assignment overlap;
- local bare transitions and coupling matrix elements.

`calculation.annotate` adds loaded $E_C$, $E_J/E_C$, the multilevel dispersive diagnostic, physical inputs, and the current matching ratio

$$
\bar\chi=\frac{\chi_++\chi_-}{2},\qquad
R=\left|\frac{\chi_+-\chi_-}{\bar\chi}\right|.
$$

## Operational search for a fabricated circuit

For fixed `Device(...)` capacitances:

1. create `ReadoutModel(device, basis)`;
2. scan $E_{J1}$ and $E_{J2}$ using `model.find_readout_points`;
3. annotate candidates with `calculation.annotate`;
4. inspect `calculation.failures`;
5. run the independent chi-sign probe;
6. repeat at larger bases and denser grids.

`find_readout_points` locates local minima of the dressed logical splitting on each $E_{J2}$ slice and retains sufficiently hybridized candidates. It is useful for studying one fabricated circuit, but it is not the same optimizer used to generate the full design map.

## Design search for a new circuit

For a chosen shunt pair `(C1, C2)`, `calculation.search_and_validate` searches four coordinates:

1. mean local bare frequency;
2. differential local detuning;
3. $C_{12}$;
4. common $C_{1r}=C_{2r}$.

The first two coordinates are converted numerically into $E_{J1},E_{J2}$. The returned point is then recomputed at a larger basis. Use `level="fast"` only for exploration; `normal` and `deep` increase search effort but still do not prove global optimality.

## Current nominal conditions

- $R\leq1\%$;
- $|\bar\chi|\geq1$ MHz;
- multilevel dispersive ratio at most 0.1;
- logical frequency 90-110 MHz;
- $H\geq95\%$;
- state-assignment overlap at least 95%;
- loaded $E_C/h$ from 150 MHz to strictly below 400 MHz;
- $E_J/h<25$ GHz and $E_J/E_C>50$;
- operation frequency 4-8 GHz and below the dressed resonator;
- both chi signs stable on an independent $5\times5$ grid spanning $\pm1\%$ in each $E_J$.

These conditions are centralized in `calculation.Conditions`. Record any change as a new study policy and use a new result directory.

## Interpretation cautions

> [!warning] $H$ is not state fidelity
> It is normalized balance of the two charge-transition branches. In an ideal two-state picture, $H=95\%$ corresponds to approximately 47.5/52.5 population balance.

> [!warning] A search miss is not infeasibility
> All searches are bounded numerical searches. Inspect diagnostics and try justified additional seeds before drawing a physical conclusion.

> [!warning] Nominal is not robust
> A nominally accepted point has not passed fabrication-error sampling. Perturb capacitances as fixed fabricated quantities and recalibrate only accessible $E_J$ controls.

Also distinguish the charge coefficient `J12_ghz`, its local 01 matrix element `J01_12_mhz`, and the resonant avoided-crossing gap $2J_{01}$.

## Recommended workflow for your own study

1. Reproduce the companion example at the reported design.
2. Replace the capacitances with your fabricated or proposed values.
3. Define physically accessible effective-$E_J$ ranges.
4. Search the fixed circuit and inspect every candidate and failed condition.
5. Increase basis and grid sizes until observables converge.
6. Add SQUID/flux constraints if the hardware is tunable.
7. Perturb fabricated components and recalibrate only $E_J$.
8. Use the full map workflow only when exploring alternative fabricated designs.

## Related material

- [[../../readout_workflow/README.md|readout_workflow reference]]
- [[../../core/README.md|shared physics]]
- [[../../examples/readout_workflow_single_pair_example.py|interactive example]]
