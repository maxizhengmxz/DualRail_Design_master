# Approved performance policy: 2026-10-29

All eight proposals below are now approved and implemented in the explicit
coupler workflow. User precision: residual couplings **1 Hz absolute**; gate
infidelity **1e-4 absolute** (0.01 percentage point, not relative 0.01%).
Use modest routine checks and stop at first fixed-control convergence. No
production design/readout campaign was launched by the implementation.

Implementation map:

- 1: `PairModel.physical_off`, `SevenReadout(off_only=True)` and root dispatch.
- 2-3: immutable exact-key local/pair caches in `model.py`, `pairs.py` and cached
  frequency inversion in `design.py`; the shared readout primitive is unchanged.
- 4: batched spectral gate metrics and reuse of the same auxiliary eigenvectors
  in `on_gate.py`/`calibrate_on.py`.
- 5: geometry/loading rejection in `DesignEvaluator.evaluate` before spectra.
- 6: `floating_coupler/batch.py`, bounded spawned workers, BLAS=1, atomic
  checkpoints and source/input guards; capacitor CLI supports `--workers`.
- 7-8: `precision.py`, adaptive five/seven audits, compact default templates,
  Hz-aware root acceptance and feasible-plateau optimizer stopping. Basis
  exhaustion is recorded as unconverged, not a license for unlimited escalation.

## What the existing toy actually does

- `../2Q-gate_XM/xm_cz/model.py:local_model`: ncut=30, four full-cosine local
  levels, selected **tridiagonal** charge-space eigensolve; LRU cache size 1024.
- `pair_model`: separate isolated rails, cached pair solutions (size 256),
  explicit bare-overlap auxiliary assignment and full transformed charge operators.
- `../2Q-gate_XM/xm_cz/evaluator.py:evaluate_ec` (nested `build`): cache the spectrum
  at a frequency offset; vary duration using spectral phases, not a new eigensolve.
  Calibrate conditional phase pi, then compare raw fidelity at that same gate.
- `../transmon-based CZ/transmon_cz/simulation.py:spectral_decomposition/unitary_trajectories`: cached
  static spectrum and batched propagation of inputs/times.
- `../2Q-gate_XM/xm_cz/runner.py`: process workers with one BLAS thread, atomic
  checkpoints, source/input hashes and resume without duplicate writers.
- The explicit coupler already uses parity sectors and selected low-energy
  eigenpairs, a pair-eigenbasis and some frequency/evaluation caches. These are
  existing features, not new speed-ups delivered by cleanup.

The toy's four-local-level truncation is a numerical approximation, NOT an RWA
or justification for discarding virtual auxiliary/coupler paths. Common-g S1
selection, the unintended-ratio screen, physical auxiliary tracking and final
phase calibration must survive any optimization. The toy refits its artificial
J during frequency calibration; fixed hardware must not imitate that refitting.

## Approved exact-computation improvements

1. **Separate OFF, ON and reporting evaluation.** For an OFF-only C-EJ root,
   compute only the odd-parity physical channel. Evaluate even-parity logical
   residuals at roots and requested plot points; defer ON calibration/readout
   until a valid design exists. Capacitor fitting still evaluates every required
   ON constraint; do not skip it or use a stale ON result.
2. **Cache invariant spectra and operators.** Reuse rail spectra/charge matrices,
   frequency-to-EJ solves and Kronecker operators where EC/EJ/J/basis are unchanged.
   Cache keys must include all physics parameters and truncations, without
   rounding optimizer inputs or reusing data across changed source versions.
3. **Use the toy's tridiagonal single-transmon eigensolver.** This preserves the
   full-cosine charge Hamiltonian. Verify energies, operator phases/parity and
   coupled observables against the current dense implementation before enabling.
4. **Reuse one diagonalization per gate bias.** Batch all four logical inputs
   and all tested durations using spectral phases. Some kernels already do this;
   extend only missing paths and verify exactly identical phase/fidelity metrics.
5. **Cheap feasibility rejection first.** Reject nonpositive compensation,
   wrong loaded ECs, excessive capacitance loading or invalid frequency ordering
   before constructing/dense-diagonalizing the quantum Hamiltonian. Quantum
   overlap and unintended-channel checks still require actual evaluation.
6. **Parallel independent work with resumable records.** Use bounded process
   workers and BLAS=1 for independent starts/biases, using the toy's source-hashed
   checkpoints. Do not parallelize sequential continuation dependencies or permit
   concurrent writers. Benchmark overhead before choosing a worker count.

## Approved search/verification approximations

7. **Coarse-to-fine truncation and continuation.** Use a smaller basis only to
   locate brackets/capacitor seeds, then refine and certify in the full basis.
   Carry a root bracket between nearby geometries, with periodic full-range
   scans and interior-extremum checks so nonmonotonic branches are not missed.
   Always perform a fixed-control larger-basis audit: do not hide basis error
   by recalibrating first. The previous near-10-GHz boundary shift is an example
   of why this distinction matters.
8. **Convergence-aware optimizer stopping.** Stop when improvements fall below
   independently measured numerical noise and required physical accuracy, rather
   than continuing identical line searches. Agree tolerances first; never relax
   a physical acceptance threshold or call a finite minimum a root.

## Not proposed as an unqualified shortcut

No new Duffing/harmonic/RWA/total-excitation truncation for acceptance, no
uniformly rescaled S1 certificate, no dropping counter-rotating or virtual paths
at a cancellation, and no unverified sparse-eigensolver substitution. These
would need a separate controlled comparison and explicit approval.

Routine gate checks compare raw fidelity, postselected score, mean survival and
per-input leakage when available, at identical controls. Residual checks compare
physical exchange, both logical exchanges and secular logical ZZ. The five-body
ladder starts small and escalates only if necessary. Seven-mode defaults remain
modest (SEARCH -> AUDIT); extra matter/photon tiers require an explicit configured
budget if those checks do not reach the target. There is no compulsory 8/9-level
or four-photon final audit. Report a failure to converge rather than overstating
accuracy. Physical loading, readout thresholds, phase-pi condition and the S1
selection policy remain unchanged. No speed-up factor is claimed without a
representative timing comparison.

## Readout timing investigation (2026-10-29)

A/B comparison sweeps now have a default two-process runner with BLAS=1,
independent atomic progress, source/input-checked resumption and separate parent
wall/worker timing. A one-point concurrent check confirmed two PIDs and one BLAS
thread per process. Historical A/B sweeps were already launched concurrently by
hand; this makes concurrency reproducible, not a new claim of twofold speedup.

At one saved idle point, standalone routine 6x6x5 evaluation took 0.007 s;
standalone sweep 8x8x6 took 0.017 s. Integrated local6/pair12/C5/photon3x3 has
6480 states: construction 0.261 s, solve 5.756 s, metrics 0.027 s (6.045 s total).
About 95% was dense eigensolving, not observable extraction or file output.
The independent readout workflow uses smaller routine bases and adaptive audits;
the integrated sweep still used the audit basis at every frequency. Transferring
that adaptive policy is a proposed next change, not enabled by the concurrency
patch. It must retain all seven modes and validate thresholds/representative
points before claiming a smaller truncation sufficient. No numerical thresholds,
Hamiltonian terms or default sweep bases changed in this timing investigation.

### Measured reduced readout basis (2026-10-29)
Tested both calibrated rails at -100,-50,0,+50,+100 MHz, fixed hardware/EJs.
Median point times: local4/pair8/C3 0.155 s; local6/pair10/C4
0.786 s; local6/pair12/C5 5.767 s; local6/pair12/C3 1.180 s.
The smallest basis falsely rejects center chi matching (A 2.044% vs
0.0767%; B 6.603% vs 0.6975%), so it is not the production default.
Retaining pair12 and local6 but C3 changes signed chi by at most 113 Hz
on these ten points. This is an empirical readout check, not a universal
error bound or a 1 Hz residual certificate. Resonators retain |0>,|1>,|2>.
Default common sweeps now use that C3 basis, escalating to C5 at centers,
endpoints and near acceptance boundaries; each row records selected basis
and audit differences. Explicit caller-supplied basis remains fixed.
Further solver changes (e.g. sparse targeted diagonalization) remain proposals.
Full 21-point A/B concurrent sweep with this policy: 78.67 s compute,
80.06 s including plots; A 71.31 s / B 77.50 s (overlapping workers).
Audits: A 3 points, B 4 points. Earlier full audit sweeps took ~225.7 s
elapsed: observed end-to-end reduction ~2.9x. Regression tests overlapped
part of this timing, so this is a practical measurement, not an isolated
hardware benchmark. All 99 regression tests passed. No circuit redesign.

Readout H-score escalation guard: 0.0003 absolute around H=0.95, covering
the observed 0.000206 C3-to-C5 shift with margin. The physical acceptance
threshold stays H>=0.95; this changes only when the basis is increased.
Readout B-center follow-up (fixed calibrated hardware/EJs, 2026-10-29):
local6/pair12/C5 -> local7/pair15/C6, both resonators |0>,|1>,|2>.
Worst H: 0.95011458 -> 0.95009455; worst chi mismatch 0.65278355% ->
0.65242440%. Nominal readout remains feasible. Check cost 38.81 s.
Signed chi changes <=147 Hz. This does not certify seven-mode OFF residuals.
The measured numerical workflow plus this follow-up costs ~581 s (~9m41s),
excluding coordination/report preparation and extra audits from the wider guard.
Use about 10 minutes per EC tuple as the practical budget; not a guarantee.
