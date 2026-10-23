# Static coupler-off workflow

This is the source of truth for humans and coding agents. The workflow can be
reproduced on new inputs; no particular case-study result is required or shipped.
See [AGENTS.md](AGENTS.md) for the required user conversation.

## 1. Obtain the readout-optimized pair (separate workflow)

Shared physics now comes directly from `../core/transmon.py` (Basis and the
local full-cosine transmon) and `../core/electrostatics.py` (loaded EC, J, J01).
Only readout-specific Device/ReadoutModel/annotate/failures diagnostics depend
on `../readout_workflow/calculation.py`. Circuit topology, capacitance budget
and cancellation searches remain here. See [the dependency map](../core/README.md).

**Request this input. Do not automatically optimize readout.** The separate
[readout workflow](../readout_workflow/README.md) can produce a pair design if
the user explicitly requests that task. Copy the chosen pair's parameters and
baseline readout into `pair` in a new configuration.

The model uses one identical nominal capacitor design for A1-A2 and B1-B2.
Their EJs may subsequently tune independently. `pair.device` contains physical
shunt, intra-pair and resonator capacitors plus the nominal readout inductance.
Loaded `EC` values must come from that same circuit, not just `e²/(2*Cshunt)`.
The original `splitting_ghz`, `chi_m_ghz`, `chi_p_ghz` record the supplied baseline.
The two `resonator_inductances_nh` specify the actual A/B readout setup and may
differ. No previous pair design is silently loaded.

## 2. Obtain direct coupling and set the capacitance budget

Throughout the circuit model, `H_int/h = J_ij n_i n_j` and
`J_ij = 4 e^2 (C^-1)_ij/h` in GHz, with Cooper-pair number operators.
The physical local exchange is `J01_ij = |<10|H_int/h|01>|`, evaluated
from J and numerical local charge matrix elements. J01 is a cyclic frequency;
the isolated resonant two-state gap is 2*J01. Loaded EC and J use the same
inverse capacitance matrix. After hardware is frozen, J is fixed while J01
can vary with EJ. This notation does not rename effective Jport or logical J1.

Preferred direct inputs are exactly one of `capacitance_ff`,
`charge_coupling_ghz` (J), or `j01_mhz` (local exchange magnitude), under
`direct_a2_b1`. J and J01 inputs are converted to a fixed capacitor in the
disconnected-mediator reference circuit described below. The old
`bare_exchange_mhz` input remains a J01 alias; never reinterpret it as J.
Outputs expose `charge_coupling_J_ghz` with its node order (off-diagonals
are J; diagonals are 8*EC, not added self-interactions), `J_a2_b1_ghz`,
`J_a2_c_ghz`, `J_c_b1_ghz` and the corresponding `J01_*_mhz` values.
Old `g_a2_c_mhz` and `g_c_b1_mhz` outputs remain J01 aliases.

**Ask for direct bare A2-B1 coupling** using the preferred inputs above.
The original supported inputs remain valid:

- `direct_a2_b1.capacitance_ff`: physical direct mutual capacitor, fF.
- `direct_a2_b1.bare_exchange_mhz`: nonnegative cyclic matrix-element magnitude
  `|<10|H/h|01>|`, MHz, at nominal rail EJs with both coupler links disconnected.

A rate is not an avoided-crossing splitting (`2|g|`) or angular frequency.
The rate-to-capacitance conversion solves the inverse-capacitance problem
while preserving loaded rail EC. It is NOT a requested effective `Jport` zero.
The converted capacitor remains fixed after conversion; the bare rate can then
change when EJ changes. Mediated and direct contributions are not double counted.

For each `coupler_candidates` entry specify a loaded `ec_ghz` and the two
`links_ff` in **[A2-C, C-B1]** order. Provide several candidates to perform a
geometry/EC feasibility search. Equal capacitors do not generally mean equal
exchange matrix elements. The kernel compensates ground shunts to preserve
the supplied rail EC, checks positive physical shunts and positive definiteness,
and records the full matrix and remaining shunts.

## 3. Find a nominal physical-port off-position

Keep A1,A2,B1,B2 at their supplied nominal EJs. For each candidate geometry,
scan the explicit coupler EJ bracket and solve sign-changing roots of Jport.
The reduced dynamical model is A2-C-B1; its coefficients come from the full
seven-node inverse capacitance matrix. A1/B2/readout dynamics are not included
in this diagnostic. No full logical cancellation is claimed at this stage.

Inspect `nominal.json`, including rejected candidates. A sign scan can miss a
tangential or extremely narrow zero; no root in a sampled bracket is not proof
that the entire design space is infeasible. Choose geometry and root, then
freeze every capacitor. If none is feasible, change explicit candidate inputs
or the agreed search bracket; do not quietly relax hardware constraints.

## 4. Ask for the objective, then optimize EJs

**For each new set of parameters, ask the user to choose**:

| CLI objective | Minimized magnitude | Meaning |
| --- | --- | --- |
| `physical` | `|Jport|` | Reduced A2-B1 physical-port exchange |
| `logical` | `|J1|` | Full-circuit logical 01 ↔ 10 matrix element |
| `combined` | `sqrt(|J1|² + zeta²)` | Logical exchange and residual ZZ, equal quadratic weight |

Do not automatically compare all three unless the user asks. Each requested
optimization starts from the nominal physical-port root, changes five EJs only,
and screens readout/overlaps/logical frequency under the supplied bounds.
There is no chi-sum reward and no additional percentage trim constraint.
SLSQP is a local optimizer; an unsuccessful/infeasible result remains labeled
as such. `optimizer_success` and `nominal_constraints_pass` are separate.

The config has explicit hardware constraints in `screens` and numerical search
intervals in `ej_search_bounds_ghz`, ordered A1,A2,B1,B2,C. A null
`screens.coupler_ej_max_ghz` removes that hardware ceiling; the solver still
needs a finite user-chosen numerical bracket. Extend it if needed. Do not
describe the numerical bracket as a fabrication constraint.

## 5. Validate and report

The full grounded five-transmon circuit includes both readout resonators and
all inverse-capacitance interactions. Its Hamiltonian **is diagonalized**.
Isolated-pair eigenstates identify the encoded states; local energy rotations
define the logical matrix entries. This differs from diagonalizing only three
transmons and declaring the resulting physical zero a logical zero.

Use `audit` to recompute the endpoint in both configured bases. Check numerical
changes before printing near-zero values to high precision. The four basis
dimensions are `[rail transmon levels, resonator Fock levels, retained pair
states, coupler levels]`. `spectral_ceiling_ghz: null` retains the entire chosen
product basis. A finite spectral ceiling speeds low-spectrum solves, but must
include the logical AND one-photon readout manifolds; raise it in convergence
checks. No product-energy pruning is used by this workflow.

Report EJs, bare and operating frequencies, logical frequency, chi-sum,
chi-ratio, hybridization, dispersive diagnostic and overlap checks. The
optimizer writes conditional readout comparisons for both spectator logical
states, including chi-sum magnitude degradation from the supplied baseline:

`chi_sum = chi_0 + chi_1`,
`chi_ratio = |chi_1-chi_0| / |(chi_0+chi_1)/2|`.

The dispersive check uses isolated pair blocks with full electrostatic loading;
it is not a full driven-readout or leakage simulation. Full-circuit conditional
chi and hybridization checks are reported separately. No automatic EJ sign-grid
robustness qualification is claimed.

Residual definitions (all are matrix elements of H/h, cyclic frequencies):

- Intra-pair exchange hybridizes A1-A2 or B1-B2; it is desired for encoding.
- Direct bare A2-B1 coupling is an electrostatic input, not Jport.
- `J1 = <0L,1L|H_eff/h|1L,0L>` and
  `zeta = F11,11 - F10,10 - F01,01 + F00,00` in vacuum.
- `J2L(nA,nB) = <0L,0L;nA,nB|H_eff/h|1L,1L;nA,nB>` is a logical double flip.
- `R_ij,kl = <iL,jL;1,0|H_eff/h|kL,lL;0,1>` transfers a nearby resonator photon.
- Single-excitation transfer connects `|gA,jL_B>` with `|iL_A,gB>`; g denotes
  a pair ground state, whereas 0L and 1L are one-excitation encoded states.

Vacuum logical order is 00,01,10,11. Photon sectors are (0,0),(1,0),(0,1).
Audit JSON contains signed matrices, units, and connected-state diagonal
detunings. Matrix elements should be compared with their state detunings;
they are not by themselves observed oscillation rates or leakage probabilities.
Photon-expanded and vacuum-only effective matrices can differ because their
retained reference spaces differ. Only the specified photon sectors are covered.

## 6. Optional bare-frequency characterization

Run only when requested. Fix B1/B2 at the selected optimized endpoint and all
capacitors at the selected geometry. Sweep bare A2 and coupler frequencies;
invert the local transmon spectrum to set their EJs, and tune A1 near equal
bare rail frequencies to maximize dressed charge-transition balance. This
bounded local search does not prove globally maximal or perfect hybridization.
It does not reoptimize or certify readout at every pixel.

Supply `sweep` as:

```json
{"fa_bare_ghz": [4.3, 4.7, 21], "fc_bare_ghz": [7.0, 8.3, 31],
 "a1_ej_search_half_width_ghz": 0.15}
```

These numbers illustrate syntax, not recommended physical inputs. Search EJ
brackets must cover the requested frequency ranges. Invalid pixels are saved
with an error, never silently replaced. Checkpoint identity includes the config,
operating point and axes; matching interrupted sweeps resume point by point.
At default serial execution, halving both steps roughly quadruples point count
and runtime. Measure a small grid before estimating the cost of a large one.

`plot` creates shared-log-scale maps (1 kHz-1 MHz), marks sampled P/L/Q minima
and bare A2-B1 resonance, and plots EJ-frequency conversions. Logical maps mask
points with inadequate overlap; the physical-port map remains a distinct
diagnostic. Failed points are also gray, with reasons retained in JSON. Minima
receive larger-basis checks but are not continuously optimized/re-ranked in
that basis. Interpret near-zero results at their demonstrated convergence.

## Commands and files

From the project root, install dependencies with `uv sync`. Copy
`coupler turn-off/configs/template.json` to a new config and fill the null physical
inputs with values supplied by the user. `null` means missing, not zero.
The template intentionally cannot launch a physical run without those inputs.

```sh
uv run python "coupler turn-off/run.py"
uv run python "coupler turn-off/run.py" check --config "coupler turn-off/configs/my-case.json"
uv run python "coupler turn-off/run.py" nominal --config "coupler turn-off/configs/my-case.json"
# Ask for the minimization objective; substitute the user's answer below.
uv run python "coupler turn-off/run.py" optimize --config "coupler turn-off/configs/my-case.json" --candidate geometry-1 --root 0 --objective logical
uv run python "coupler turn-off/run.py" audit --config "coupler turn-off/configs/my-case.json" --candidate geometry-1 --root 0 --objective logical
# Optional, only after a requested sweep is configured:
uv run python "coupler turn-off/run.py" sweep --config "coupler turn-off/configs/my-case.json" --candidate geometry-1 --root 0 --objective logical
uv run python "coupler turn-off/run.py" plot --config "coupler turn-off/configs/my-case.json" --candidate geometry-1 --root 0 --objective logical
```

No objective is selected by default. Output is under
`coupler turn-off/output/<case_id>/<config-hash>/<geometry>/root-<index>/<objective>/`.
The full input snapshot is saved before calculation. A completed optimization
is not overwritten; use a new output location for another attempt. Optimize,
audit and sweep are separate operations. Saved files are generated outputs,
not prerequisites for sharing the repository.

| Source | Responsibility |
| --- | --- |
| `config.py` | Input validation and cache identity |
| `engine.py` | Budget, fixed-rail roots, EJ-only minimization, readout comparisons |
| `channels.py` | Logical/photon residual matrix elements and detunings |
| `sweep.py` | Optional bare-frequency scan and plots |
| `circuit.py` | Reusable capacitance matrix and full/reduced Hamiltonians; no case runner |
| `logical.py` | Logical energy-convention exchange/ZZ extraction |
| `readout.py` | Loaded isolated-pair readout diagnostics |
| `dispersive.py` | Optional full matter-network dispersive diagnostic API |

The paths in the table are relative to `coupler turn-off/coupler_turn_off/`. Other existing reusable
gate/Monte Carlo modules remain separate from this static workflow.

## Folder layout and tests

```text
coupler turn-off/
  README.md                 Workflow, physics definitions and commands
  AGENTS.md                 Required questions and rules for coding agents
  run.py                    Command-line entry point (help without arguments)
  run_tests.py              Focused regression-test runner
  configs/template.json     Blank physical inputs; no inherited case values
  coupler_turn_off/          Circuit, diagnostics, optimization and sweep modules
  tests/                    Self-contained numerical and workflow regression tests
  output/                   Generated per-case output (gitignored)
```

From the repository root:

```sh
uv run python "coupler turn-off/run_tests.py"
```

Commit this folder together with shared `core/`, the `readout_workflow/` package and
root `pyproject.toml` / `uv.lock`. Coupler calculations import readout circuit
primitives and diagnostics, but do not launch readout optimization. The script
entry points work from any current directory; command examples above assume
the repository root so relative config paths resolve as shown.

No ZIP, saved results, virtual environment or cache belongs in Git. New case
configs may be committed deliberately; generated results remain ignored.
