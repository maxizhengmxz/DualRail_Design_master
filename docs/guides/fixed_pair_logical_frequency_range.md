---
title: Fixed-pair readout logical-frequency range study
aliases: [Readout logical-frequency extrema, Fixed C1-C2 readout study]
tags: [dual-rail, readout, optimization, logical-frequency]
status: working-guide
---

# Fixed-pair readout logical-frequency range study

## Study question

At fixed shunt capacitances

$$
C_1=100\ \mathrm{fF},\qquad C_2=75\ \mathrm{fF},
$$

find the smallest and largest logical frequency for which at least one nominal
configuration satisfies every current readout requirement.

The required result is a best-found interval, not a proof of global extrema or
a fabrication-yield statement.

## What the original C1-C2 map does

The outer map is discrete: `C1` and `C2` are evaluated at 1 fF spacing with
`36 <= C2 < C1 <= 128 fF`. Inside each pixel, however, the design variables are
continuous. The optimizer coordinate is

$$
x=(f_{\mathrm{mean}},d,C_{12},C_g),
$$

where

$$
f_1=f_{\mathrm{mean}}+\frac{0.01d}{2},\qquad
f_2=f_{\mathrm{mean}}-\frac{0.01d}{2},
$$

frequencies are in GHz, and `Cg = C1r = C2r`. The local frequencies are
numerically inverted to obtain `EJ1` and `EJ2`. Therefore the physical design
variables are effectively `EJ1`, `EJ2`, `C12`, and the common readout coupling
capacitance.

A seed is an initial guess for `x`, not a known valid configuration or a grid
sample. From each seed, a local optimizer evaluates a trajectory
`x0, x1, ..., xfinal`. Multiple seeds are used because distinct seeds can enter
different attraction basins or disconnected feasible branches.

The initial feasibility search minimizes normalized constraint violations. It
stops after the first candidate that survives validation and the independent
chi-sign check. A later phase maximizes absolute mean chi and retains one
audited point per `(C1,C2)` pixel. The original workflow therefore establishes
best-found existence and selects one design; it does not enumerate the
feasible subset at a pixel.

## Feasible set

For this study, define

$$
\mathcal F=\{x:\text{all current nominal readout requirements pass}\}.
$$

The requirements are:

- mean-chi mismatch at most 1%;
- absolute mean chi at least 1 MHz;
- multilevel dispersive ratio at most 0.1;
- hybridization at least 95%;
- reference-state assignment overlap at least 95%;
- logical frequency from 90 to 110 MHz;
- operation frequency from 4 to 8 GHz and below the dressed resonator;
- each loaded EC at least 150 MHz and strictly below 400 MHz;
- each EJ strictly below 25 GHz;
- each EJ/EC strictly greater than 50;
- independent chi-sign stability on the 5-by-5, +/-1% EJ probe grid.

The sign certificate is discontinuous and expensive. It is applied to
converged candidates rather than used inside every optimizer iteration.

## Endpoint optimization

The lower endpoint solves

$$
\min_{x\in\mathcal F} f_\mathrm{logical}(x),
$$

and the upper endpoint solves

$$
\max_{x\in\mathcal F} f_\mathrm{logical}(x).
$$

SciPy minimizes objectives, so the maximum run minimizes
`-logical_frequency`. Both runs use identical bounds and hard constraints.
Logical frequency is never allowed to compensate for a failed readout check.

Each seed is used once for the lower objective and once for the upper
objective. Every distinct endpoint is recomputed at the validation basis,
then at the audit basis with unchanged physical inputs. It is retained only if
`failures(point, require_sign_check=True)` is empty.

## Why use more than one seed

One seed normally explores one local basin. More seeds do not enumerate the
feasible set, but diverse seeds improve the chance of discovering distinct
basins and more extreme valid endpoints. Useful diversity includes:

- the workflow's physics-informed seeds;
- low, middle, and high mean operation frequencies;
- negative, zero, and positive detuning;
- small and large `C12` and `Cg`;
- deterministic space-filling Sobol starts;
- any previously validated `(100,75)` design.

Increase seed batches until the audited extrema stop improving within a stated
tolerance. Reproducibility requires recording the Sobol seed, all initial
guesses, rejected endpoints, and failure reasons.

## What the endpoint study does not show

Two valid endpoints do not by themselves demonstrate that every intermediate
logical frequency is viable. They also do not reveal:

- gaps in the projected feasible set;
- disconnected solution families;
- multiple designs at the same frequency;
- parameter continuity between endpoints;
- fabrication-error robustness;
- mathematical convergence with Hilbert-space size;
- global optimality.

After finding stable endpoints, a sparse continuation check across the
interval is recommended. This is much cheaper than cataloging all feasible
configurations and can test whether one continuous validated branch connects
the endpoints.

## Numerical validation versus robustness

The workflow searches at a smaller basis, re-polishes at a validation basis,
and recomputes selected candidates at the audit basis. This is larger-basis
validation, not a formal convergence proof. A formal convergence study would
compare observable changes across several increasing bases against declared
tolerances.

The +/-1% EJ sign grid is a resonance-sensitivity diagnostic, not fabrication
robustness. A robustness study would perturb fabricated capacitances and the
resonator, recalibrate only experimentally tunable controls, and reapply every
acceptance condition over the perturbation ensemble.

## Interactive implementation

Use `examples/readout_fixed_pair_logical_range_study.py` in VS Code's Python
Interactive window. Run cells in order. Start with a small seed batch for a
smoke test, then increase the number of deterministic Sobol starts and compare
endpoint stability across batches. Its numerical helpers are in
`examples/readout_fixed_pair_scan.py`.
