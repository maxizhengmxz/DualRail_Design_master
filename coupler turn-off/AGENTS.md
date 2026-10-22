# Start here: static coupler-off analysis

Read [README.md](README.md) before a new
coupler-off task. It is the workflow specification. The executable entry point
is `python "coupler turn-off/run.py"`; without arguments it only shows help.

## Required conversation for EVERY new parameter set

1. Request the **externally readout-optimized pair** and its readout baseline.
   Readout optimization is a separate workflow in the shared `../readout_workflow/`. Do not
   run it as part of coupler-off analysis unless separately instructed.
2. Request the **direct bare A2-B1 coupling** as either a cyclic exchange rate
   or a capacitor. Ask for units. A rate means `|<10|H/h|01>|`, not an angular
   rate, avoided-crossing splitting, or effective exchange after the coupler.
   If the user's convention differs, clarify/convert before using it.
3. Ask the user to choose **physical**, **logical**, or **combined** minimization.
   Ask again for each new parameter set. Do not inherit an old objective or
   automatically run all three. Ask for constraints that were not supplied.
4. Explain the staged calculation briefly. Do not call a physical-port zero a
   logical off-position. After feasibility, show the candidates and select the
   fixed geometry/root with the user when the choice is ambiguous.
5. Offer the optional bare-frequency sweep; run it only when requested.

## Physics and implementation rules

- Circuit: grounded A1-A2-C-B1-B2, with one nearby readout resonator per pair.
  The two nominal pairs share one supplied pair design. No independently added
  A1-C or C-B2 capacitor. Inverse-capacitance-induced terms are retained.
- Include direct/link capacitors in the complete EC budget. After selecting
  hardware, freeze **all** capacitances and change EJ only.
- Follow explicit user EJ/EC/readout bounds. No inherited 2% EJ trim, 30 GHz
  coupler ceiling, 285 MHz coupler EC, or 0.1 fF direct capacitor.
- Do not add chi-sum to the objective. Report its magnitude degradation versus
  the supplied uncoupled readout baseline, and report chi-ratio explicitly.
- H/h frequencies: GHz internally; result keys specify MHz/Hz. No implicit
  factor of 2 or 2*pi. Do not equate matrix elements with transfer probabilities.
- Logical 0L is approximately (01-10)/sqrt(2), logical 1L (01+10)/sqrt(2).
  Use `g` for the pair ground state; 0L is NOT its ground state. Use i,j for
  logical labels and nA,nB for nearby resonator photon occupations.
- Full-circuit eigenstates are diagonalized, then identified with isolated-pair
  reference states. Explain this plainly; avoid unexplained "projected frame".
- Check convergence and logical-space overlaps. Gray logical-map points mean
  an unreliable four-state description or an explicitly recorded failed point.
  Keep physical-port validity separate. Report connected-state detunings.
- Optional maps use **bare** frequency axes, B fixed and A1 rebalanced, mark P/L/Q
  sampled minima and bare A2-B1 resonance, and use a shared 1 kHz-1 MHz color
  range. A sweep is not readout reoptimization or certification.

## Repository hygiene

There are intentionally no saved case results or report generators. Do not
reintroduce case-specific imports, file paths, constants, or chains of repair
scripts. Put new numerical work in reusable modules and new inputs in a
separate config. Generated output belongs under `coupler turn-off/output/` (gitignored).
Never start a long calculation just to inspect the project. Run targeted tests
for code changes. Prefer concise progress updates and economical tool output.

Keep unrelated project functionality and user edits intact. All coupler-off
source, tests and input templates belong in this folder. Keep the shared
readout calculation primitives in `../readout_workflow/`; do not duplicate them.
Run `python "coupler turn-off/run_tests.py"` from the repository root after
changes. Do not create ZIP archives. The user will push source to Git; exclude
saved outputs, environments, caches and credentials.
