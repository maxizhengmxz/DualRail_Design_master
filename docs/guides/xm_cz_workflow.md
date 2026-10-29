---
title: Using 2Q-gate_XM for a Dual-Rail CZ Study
aliases: [XM CZ operating point, Dual-rail dressed-collision CZ]
tags: [dual-rail, cz-gate, operating-point, dressed-state, 2Q-gate-XM]
status: working-guide
---

# Using `2Q-gate_XM` for a dual-rail CZ study

This note introduces the cleaned `2Q-gate_XM` workflow and its state-label
conventions. The companion [[../../examples/xm_cz_single_point_example.py|interactive example]]
shows the common low-level objects, inspects their data structures, reconstructs
the intended dressed collision, and optionally runs the complete one-point
selection and phase calibration.

> [!important] What this workflow searches
> The selected collision is between an **isolated-pair dressed logical product**
> and an **isolated-pair dressed auxiliary product**. It is not merely equality
> of two named bare-product energies.

## Physical and numerical hierarchy

The physical order is always `A1,A2,B1,B2`. Each dual rail contains two
transmons: pair A is `A1,A2`, and pair B is `B1,B2`. The retained Hilbert spaces
are built in stages:

```text
one transmon: 61 charge states (ncut=30)
       -> four local energy eigenstates
one dual rail: 4 x 4 = 16 product states
       -> diagonalize A and B separately
two dual rails: 16 x 16 = 256 products of pair eigenstates
       -> add the transformed A2-B1 charge interaction
```

`local_model(ec, frequency)` solves the Josephson energy needed to obtain the
requested bare transmon transition. Its tuple is

```python
(EJ_GHz, energies_GHz, projected_charge, charge_basis_vectors)
```

with shapes `(scalar, (4,), (4,4), (61,4))`. The energies are measured from the
local ground state. `projected_charge` is the charge operator in the four-level
local energy basis.

## The isolated-pair Hamiltonian

For a pair with local spectra $E_i^L,E_j^R$, the code constructs

```python
np.diag((left[1][:, None] + right[1][None, :]).ravel())
```

The broadcast sum is the $4\times4$ table $E_i^L+E_j^R$. Flattening uses
row-major order, so bare pair state $|ij\rangle$ has index

$$
k=4i+j.
$$

Important indices are therefore

| Index | Bare pair state |
|---:|---|
| 0 | $|00\rangle$ |
| 1 | $|01\rangle$ |
| 2 | $|02\rangle$ |
| 4 | $|10\rangle$ |
| 8 | $|20\rangle$ |

The complete isolated-pair Hamiltonian is

$$
H_{\rm pair}=H_L\otimes I+I\otimes H_R+J_{LR}n_L\otimes n_R.
$$

`np.linalg.eigh` returns `energies` and `vectors`. Column `vectors[:, k]` is
dressed pair eigenstate $|\psi_k\rangle$ in the 16-state bare product basis;
`vectors[m, k]` is $\langle m|\psi_k\rangle$. These are pair eigenvectors, not
eigenvectors of the complete four-transmon Hamiltonian.

## Logical-mode and auxiliary assignment

For every dressed pair eigenstate, the logical-subspace weight is

$$
w_k^L=|\langle01|\psi_k\rangle|^2+|\langle10|\psi_k\rangle|^2.
$$

The code selects the two states with the largest $w_k^L$ and then sorts those
two by energy:

```python
logical_weights = abs(vectors[1])**2 + abs(vectors[4])**2
candidates = np.argsort(logical_weights)[-2:]
modes = sorted(candidates, key=lambda k: energies[k])
```

Thus `modes[0]` and `modes[1]` are the lower and upper isolated-pair logical
modes. Near equal bare frequencies they are generally symmetric/antisymmetric
combinations of $|01\rangle$ and $|10\rangle$, rather than states with one
unique bare identity.

An auxiliary is assigned differently: for one specified bare state $|m\rangle$,
select the dressed eigenstate with the largest individual overlap,

```python
target = np.argmax(abs(vectors[target_bare_index])**2)
```

Index 8 requests the most-$|20\rangle$-like branch; index 2 requests the
most-$|02\rangle$-like branch. One sums squared overlaps only when identifying
a subspace such as `span{|01>,|10>}`, not when matching one product state.

`pair_model(...)` returns a dictionary:

| Key | Meaning |
|---|---|
| `energies` | 16 dressed pair energies |
| `vectors` | $16\times16$ eigenvector matrix; eigenvectors are columns |
| `modes` | indices of lower/upper logical pair modes |
| `target` | index of the selected dressed auxiliary branch |
| `target_weight` | squared overlap with the requested bare auxiliary |
| `J` | solved intra-pair charge coefficient |
| `EJ` | solved Josephson energies of the two transmons |
| `left_charge`, `right_charge` | charge operators transformed to the pair eigenbasis |
| `local_vectors` | local transmon eigenvectors in their charge bases |

## Forming the two-pair target

A and B are diagonalized separately. For the `0020` channel, the target is
formed as

$$
|\widetilde{00}_A\rangle\otimes|\widetilde{20}_B\rangle.
$$

In flattened two-pair indexing, $K=16i_A+i_B$. Because the A-pair ground-state
index is 0, the code can write simply

```python
target = b["target"]
```

which is equivalent to `0 * 16 + b["target"]`. For `0200`,

```python
target = a["target"] * 16
```

means $|\widetilde{02}_A\rangle\otimes|\widetilde{00}_B\rangle$.

This construction is a direct product of separately identified pair states.
It is not yet an eigenstate of the fully interacting four-transmon Hamiltonian.

## Logical products and implemented collision channels

The logical indices are generated in this order:

```python
logical = [i * 16 + j for i in a["modes"] for j in b["modes"]]
```

|   List entry | Logical product |                  |
| -----------: | --------------- | ---------------- |
| `logical[0]` | $               | 0_A0_B\rangle_L$ |
| `logical[1]` | $               | 0_A1_B\rangle_L$ |
| `logical[2]` | $               | 1_A0_B\rangle_L$ |
| `logical[3]` | $               | 1_A1_B\rangle_L$ |

The current workflow implements

$$
\begin{aligned}
\texttt{0020}:&\quad |0_A1_B\rangle_L
\leftrightarrow
|\widetilde{00}_A\rangle|\widetilde{20}_B\rangle,\\
\texttt{0200}:&\quad |1_A0_B\rangle_L
\leftrightarrow
|\widetilde{02}_A\rangle|\widetilde{00}_B\rangle.
\end{aligned}
$$

These pairings are workflow choices, not unique selection rules. For example,
$|1_A0_B\rangle_L$ may also have a nonzero matrix element to the `0020`
auxiliary. It is normally detuned when the implemented `01` collision is exact:

$$
E_{10}-E_{01}=\Delta_A-\Delta_B.
$$

With default logical splittings 150 and 90 MHz, that separation is 60 MHz.
Supporting `10`-to-`0020` would require another channel definition, collision
residual, branch policy, scoring exclusion, tests, and calibration validation.

The bare shorthand $|0110\rangle\leftrightarrow|0020\rangle$ describes an
important microscopic exchange component under $n_{A2}n_{B1}$. The actual
source used by the workflow is the complete dressed logical product, not the
bare state $|0110\rangle$ alone.

## Which quantities are inputs and which are solved?

For one EC point:

| Quantity | Treatment in `2Q-gate_XM` |
|---|---|
| $E_{C,A1},E_{C,A2},E_{C,B1},E_{C,B2}$ | supplied as `ec_GHz`; grid/design variables |
| common bare $f_{01}$ within A | scanned over `frequency_GHz` |
| common bare $f_{01}$ within B | root-solved to satisfy the selected dressed collision |
| four $E_J$ values | solved inside `local_model` from each $E_C$ and requested bare $f_{01}$ |
| $J_{A1,A2}$ and $J_{B1,B2}$ | solved to reproduce requested logical splittings |
| $J_{A2,B1}$ | solved so the intended dressed matrix element has magnitude `g_GHz` |
| collision channel | explicit discrete input: `0020` or `0200` |
| final B-frequency offset | calibrated within $\pm12$ MHz |
| gate duration | calibrated in `duration_ns` to obtain conditional phase $\pi$ and maximize raw fidelity |

The numerical calibration re-solves dependent $E_J$ and coupling values. It is
a design-point calculation, not by itself a fixed-coupling hardware pulse model.

## Top-level differences from the older `cz_gate`

Both codes use full-cosine transmons, retain four local levels, use charge-charge
couplings, and define logical pair modes through maximal weight in
`span{|01>,|10>}`. The top-level design questions differ:

| Aspect | Older `cz_gate` | `2Q-gate_XM` |
|---|---|---|
| Main parameters | explicit $E_C,E_J$ and coupling coefficients | $E_C$, frequency bounds, target logical splittings and target dressed $g$ |
| Junction energies | supplied or scanned directly | solved from $E_C$ and requested bare $f_{01}$ |
| Intra-pair couplings | supplied | solved from target logical splittings |
| Collision equation | named local-product/bare collision such as `0110_to_0020` | dressed logical product to dressed auxiliary product |
| Four-transmon treatment | complete $256\times256$ QuTiP Hamiltonian and time evolution | isolated-pair spectra plus transformed 256-state interaction, followed by focused calibration |
| Noise and general dynamics | optional noise and general state evolution | static coherent gate-search model |
| Search objective | operating points constrained by bare collision, then full dynamics | minimum unwanted-transition score `S1`, then exact-$\pi$ calibration and fidelity maximization |

The newer collision definition avoids assuming that a bare collision is also an
exact dressed logical collision. Conversely, its specialized parameter-solving
contract is less general than the old full-system simulator. A serious candidate
should ultimately be checked with truncation convergence, fixed hardware
parameters, pulse ramps, noise, and a full-system simulation.

## Common usage

For inspection and development, call the layers directly:

1. `local_model` to inspect one transmon;
2. `pair_model` to inspect one isolated dual rail;
3. `collision_residual` and `layout` to examine one proposed collision;
4. `evaluate_ec` to run the complete one-EC-point selection and calibration;
5. `run.py` only for checkpointed multi-point grids.

The following table maps the complete operating-point sequence to code. Names
in backticks under “production” are currently nested inside `evaluate_ec`; the
companion example provides importable-looking top-level equivalents so each
step can be executed independently.

| Step | Production code | Companion-example helper |
|---|---|---|
| Solve the dressed collision | `screening` + `collision_residual` | `solve_common_b_frequency` |
| Select the minimum-`S1` position | outer frequency loop + optional `objective` refinement | inspect `layout(...)[-1]` over candidate roots |
| Hold A and rebuild detuned B | `build(offset)` | `build_calibration_point(offset_mhz)` |
| Construct the projected logical map | `at(offset, duration)` | `logical_map_at(offset_mhz, duration_ns)` |
| Root-solve exact conditional phase $\pi$ | `calibrate_time(duration)` | `exact_pi_solutions(duration_ns)` |
| Trace the contour and maximize fidelity | duration loop + final refinement | `trace_exact_pi_contour(durations_ns)` then `max(..., key=raw_fidelity)` |

The runners are not part of these physics steps; they distribute EC points,
checkpoint results, enforce timeouts, and resume interrupted grids.

### Solving the common B frequency

The runner is not needed to understand the collision root. For fixed pair A,
define pair B as a function of its common requested bare frequency and root-solve

```python
def b_pair(f_b_ghz):
    return pair_model(ec_b1, ec_b2, f_b_ghz, split_b)

def equation(f_b_ghz):
    return collision_residual(a, b_pair(f_b_ghz), channel)

f_b_root = brentq(equation, bracket_low, bracket_high)
b_at_collision = b_pair(f_b_root)
```

For `0020`, keep the selected B auxiliary index fixed inside a root bracket; a
change of `b["target"]` means maximum-overlap assignment switched branches and
the apparent sign change must not be treated as a continuous root. The companion
example includes a small scanning helper that finds a same-branch bracket before
calling `brentq`.

### Frequency offset is not collision detuning

During phase calibration the code changes the common B control coordinate,

```python
b_at_offset = pair_model(
    ec_b1,
    ec_b2,
    f_b_root + 0.001 * offset_mhz,
    split_b,
)
```

and the actual dressed detuning must be recomputed:

```python
collision_detuning_mhz = 1e3 * collision_residual(
    a, b_at_offset, channel
)
```

In general `collision_detuning_mhz != offset_mhz`. The offset is the modeled
control; collision detuning is a derived spectral quantity. The example plots
their nonlinear mapping and records the selected auxiliary indices so branch
changes remain visible.

`layout` returns

```python
(energies, unit, logical, target, coupling, scores)
```

where `energies` has length 256, `unit` is the transformed $n_{A2}n_{B1}$
matrix, `logical` contains four flattened indices, `coupling` is the required
physical charge coefficient, and `scores` contains `S1`, `S2`, per-input
contributions, and maximum $|g/\delta|$ diagnostics.

`evaluate_ec(payload)` returns JSON-compatible dictionaries. Always inspect
`result["status"]` first. A calibrated result contains the selected `S1_position`,
screening records, solved frequencies, and `final` calibration metrics. Normal
search failures are reported as statuses rather than raised exceptions.

## Phase calibration and relation to the older code

Both implementations use the same gauge-invariant conditional phase of the
projected logical map $M$:

$$
\phi_{\rm cond}=\arg\!\left(M_{11,11}M_{00,00}
M_{10,10}^*M_{01,01}^*\right).
$$

They differ in how a gate point is selected:

- The older `cz_gate` builds one fixed full four-transmon Hamiltonian, evolves
  all four logical inputs on a sampled time grid, and reports the first sampled
  odd-$\pi$ encounter within a tolerance. Its operating-point search is anchored
  to a named local-product collision such as `0110_to_0020`.
- `2Q-gate_XM` first solves an isolated-pair dressed logical-to-auxiliary
  collision and chooses one minimum-`S1` frequency position. It then scans gate
  duration and root-solves a common B-frequency offset for exact conditional
  phase $\pi$ at each duration. It finally maximizes raw fidelity along those
  exact-$\pi$ solutions.

Thus the definition of conditional phase is essentially unchanged, while the
calibration coordinates, collision definition, root accuracy, and selection
policy are different. The newer procedure also re-solves B-pair and inter-pair
couplings to preserve target splitting and dressed `g`; it should be understood
as a design-point calibration rather than a fixed-coupling experimental pulse.

## Why exact phase does not imply high fidelity

The exact-$\pi$ equation constrains only one scalar combination of four diagonal
logical amplitudes. Fidelity additionally depends on their magnitudes, logical
off-diagonal transfer, and population outside the logical space. Even in an
ideal logical-source/auxiliary two-level model with detuning $\Delta$, coupling
$g$, and generalized Rabi frequency $\Omega=\sqrt{\Delta^2+4g^2}$, the source
return amplitude contains

$$
A(t)=e^{-i\bar E t}\left[
\cos\left(\frac{\Omega t}{2}\right)
+i\frac{\Delta}{\Omega}\sin\left(\frac{\Omega t}{2}\right)
\right].
$$

Its phase can satisfy the CZ condition while $|A(t)|<1$, leaving auxiliary
population. With the full 256-state Hamiltonian, other logical and nonlogical
channels contribute additional oscillation frequencies; their returns need not
refocus when the conditional phase reaches $\pi$.

The current workflow exposes partial diagnostics:

- `return_by_input` and `leakage_by_input` show which input fails to return;
- `S1_by_input` estimates unwanted coupling from each logical input;
- `score_parts` separates unwanted logical and nonlogical channels;
- `maximum_logical_ratio` and `maximum_auxiliary_ratio` flag a particularly
  strong off-resonant edge;
- acceptance distinguishes raw from postselected fidelity.

It does not save time-resolved populations of individual auxiliary eigenstates,
so it cannot by itself identify which leakage branch causes a dark point. The
older `cz_gate` analysis is more diagnostic here: it records wrong-logical
transfer, total/detectable/undetectable leakage, level-2 populations, and
leakage-destination histograms versus time. Those tools are suitable for testing
whether phase accumulation, intended-auxiliary return, logical exchange, and
other leakage channels refocus at compatible times.

## Interpretation cautions

> [!warning] A tilde label is ancestry, not purity
> `target_weight` reports the overlap with the requested bare auxiliary. Near
> avoided crossings, the most-overlapping eigenstate can still be strongly mixed.

> [!warning] Pointwise assignment is not complete branch tracking
> Initial auxiliary assignment uses maximum bare overlap at each point. The
> `0020` calibration adds an overlap check, but the initial scan can still miss
> or switch branches near ambiguous avoided crossings.

> [!warning] Search failure is not a proof of infeasibility
> Frequency scans, root brackets, time windows, and refinement budgets are finite.

## Related material

- [[../../2Q-gate_XM/README.md|2Q-gate_XM reference]]
- [[../../examples/xm_cz_single_point_example.py|interactive example]]

The older comparison code lives outside this repository, under the previous
`dtc-dualrail/cz_gate/core/` folder (`dualrail_sim.py` and
`gate_operating_points.py`).
