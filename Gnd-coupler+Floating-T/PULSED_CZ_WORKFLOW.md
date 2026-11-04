# Optional five-transmon pulse CZ

This explicitly requested post-design study uses saved nominal hardware, not a
new capacitor search. It does not run with design, robustness or decoherence.
The first nominal run is `output/pulsed-cz-10ns/`; no resonators or decoherence
are included. Logical/channel definitions remain those of `2Q-gate_XM/WORKFLOW.md`
and `core/cz_channel_policy.py`; the saved S1 channel is retained, not reranked.

## Controls and pulse convention

Start at the saved FIVE-MODE OFF bias and five-mode bare idle rail controls.
Do not substitute seven-mode readout-retuned controls. Freeze all capacitances,
loaded ECs and charge coefficients J. A1/A2 EJs stay fixed. Simultaneously vary
C, B1 and B2 EJs from idle to their calibrated gate endpoints and back. B1/B2
endpoints correspond to the same bare gate f01; their EJ excursions differ.

For each rising edge use
`s(t) = [tanh((t-T/2)/tau)+tanh(K)]/[2*tanh(K)]`, `K=4`,
`tau = rise_10_90/[2*atanh(0.8*tanh(K))]`, `T=2*K*tau`.
Then `EJ_i(t)=EJ_idle_i+s(t)*(EJ_gate_i-EJ_idle_i)`.
The hold has s=1; falling edge exactly reverses the rising schedule. Edges are
finite and exactly endpoint normalized. A 10 ns 10–90% transition has T=36.46 ns:
total pulse duration is hold+72.92 ns. Always report both hold and total duration.
The frequency trajectories are extracted from the full-cosine local spectrum at
each EJ and fixed loaded EC. This is an EJ waveform, not a tanh frequency or flux
waveform. A flux implementation needs separate SQUID calibration and line response.

## Propagation and references

Use one FIXED local basis spanning both endpoints. Build full-cosine local
Hamiltonians there: changing EJ changes off-diagonal cos(phi) terms as well as
energies. Instantaneous spectra without a moving-basis connection are not a
valid pulse propagator. All capacitive charge terms, including counter-rotating
terms and all spectator links, remain present.

Total even parity is exact for ng=0 and these four logical inputs. The additional
sum-of-local-level-indices cutoff is only a numerical truncation. It does NOT
make excitation number conserved. Record local and coupler levels, charge ncut,
reference EJs, cutoff and dimension; verify by increasing them at fixed controls.

Assign isolated-pair logical products to four full-system dressed IDLE
eigenstates using unique overlap assignments. Record overlap and product-space
projection norms. These same four idle states are inputs and final projections.
They differ from the static ON reference preparation. Propagate all four inputs
through the complete rise/hold/fall, including their coherent phase/leakage.

The real symmetric Hamiltonian and symmetric return give `U_down=U_up.T`,
not its Hermitian adjoint. Cache the forward-edge amplitudes and ON spectrum;
reuse them for all hold durations and phase roots. Fourth-order unitary Magnus
steps are checked by step halving. No phenomenological leakage or jump is added.

## Calibration and checks

First fix the coupler gate endpoint to the saved nominal five-mode ON value.
Start the B endpoint search around the saved gate frequency (initial +/-12 MHz)
and hold interval 40–170 ns. These are configurable finite SEARCH intervals,
not new hardware acceptance constraints. Report if no phase-pi solution is found.
Keep the selected source/auxiliary; optimize raw fidelity on the phase-pi
constraint, reporting postselected fidelity and survival at that SAME pulse.
The current bounded local search is not a certified global optimum.

Increase basis at fixed controls. If phase moves, allow bounded local B-frequency
phase correction at FIXED hold and coupler (at most five edge builds), not a new
large-basis fidelity optimizer. Mark phase transfer separately from fixed-control
convergence. Check the corrected controls against the next cutoff and halved
time step; probabilities target 1e-4 absolute, phase uses shared GATE_PHASE_TOL.
Residual OFF certification is a separate audit; do not infer it from pulse fidelity.

Report raw fidelity, survival-normalized/postselected score, average and
per-input survival/nonlogical population, phase error, endpoints, both times and
audit changes. Use the established local-Z corrections in gate_metrics. This is
a coherent pulse benchmark, not a total noise-limited hardware fidelity.
The saved static ON benchmark is a reference with different preparation, not
an identical-input control experiment; do not attribute every difference to edges.

## Explicit commands

```powershell
.venv/Scripts/python.exe Gnd-coupler+Floating-T/pulse_cz.py --input Gnd-coupler+Floating-T/output/nominal-design/fabrication-input.json --output Gnd-coupler+Floating-T/output/new-pulse
```

For a fixed-control check, add `--calibration <preceding-result.json>` and new
`--levels`, `--coupler-levels`, `--cutoff`, `--step-ns`; always use a new output
folder. `--phase-correct` permits only the bounded phase transfer above. Report
rendering is separate and never starts a search. Source: pulsed_gate.py kernel,
pulse_cz.py explicit CLI, plot_pulse_cz.py saved-result figure/summary renderer.

## Optional line-bandwidth diagnostic

`pulse_bandwidth.py --output <new-folder>` compares the current K=4 tanh and a
compact half-cosine, at EQUAL 10–90% widths, through an ideal 1 GHz brick wall.
It filters the whole rise/hold/fall pulse with long zero padding, includes
pre/post-ringing in maximum full-excursion error, and checks sampling/padding.
Zero phase is assumed, or equivalently a linear delay is removed. This ideal
filter has infinite sinc tails and is a waveform diagnostic, not a causal line
model or a settling-time guarantee. The default hold is the nominal pulse's
85.284369 ns; no quantum Hamiltonian, capacitors or gate controls are recalibrated.

The half-cosine is `s=(1-cos(pi*t/T))/2`, clamped outside [0,T], with
`T=rise_10_90/(1-2*acos(0.8)/pi)=1.693955*rise_10_90`.
The current tanh support is 3.645897 times its 10–90% width. Compare both support
and distortion, not merely nominal rise labels. Faster edges or shorter tails
require a separate pulse-gate recalibration before claiming a new CZ duration.

The bandwidth calculation applies to the line's normalized control variable.
When the physical line drives flux, its output must be converted through the
SQUID response before calling an EJ-pulse Hamiltonian. Waveform error percentages
are not gate infidelities. Stored thresholds (1%, 0.1%, 0.01%) are comparison
levels, not newly imposed design constraints or globally optimal pulse bounds.

## Calibrated shape/bandwidth comparison

`pulse_shape_comparison.py --input <nominal-input.json> --output <new-folder>
--shape tanh|cosine` explicitly runs five-transmon quantum pulse calibrations at
10/5/2/1/0.5 ns 10-90% widths. Both shapes use the definitions above. Freeze the
saved five-mode hardware, A EJs and coupler ON endpoint, and retain the S1 channel
and dressed idle input/output basis. The ideal zero-phase 1 GHz brick wall acts
on the affine EJ envelope (not nonlinear SQUID flux). A real flux-line prediction
would instead require the specified flux-to-EJ calibration before propagation.

First obtain an unfiltered local phase-pi frequency/hold seed using cached edge
spectra. Then filter the ENTIRE pulse, including both edges, at every changed
hold time. Ringing makes the delivered plateau nonconstant; it is invalid to
reuse an ON spectrum for these filtered hold proposals. Evolve from the idle
reference through pre-ringing to the symmetry midpoint, and use U_half.T U_half
for the complete real, time-symmetric pulse. Include post-ringing by symmetry.
No overshoot clipping or ad hoc endpoint renormalization is applied.

The initial filtered refinement uses scaled local SLSQP with at most five
iterations AND a hard12-trial budget (line searches can exceed an iteration
budget). After eight trials, stop on four near-phase-pi proposals whose raw/
postselected fidelity, survival and per-input leakage vary by <=1e-4. Then
perform bounded phase correction: a plateau is not phase acceptance. The search
optimizes raw fidelity subject to phase pi, as in the first pulse
study. It is a bounded local calibration, not a certified global optimum. Report
postselected fidelity, survival and per-input leakage at the SAME accepted pulse.
Transfer phase to local8/C7/cutoff8 at fixed hold/coupler, then perform fixed-control
time-step, tail-window and cutoff checks at representative extremes and wherever
agreement is doubtful. Stop probability audits on 1e-4 agreement; retain failures.

Reuse a measured phase slope and high-basis secant updates during phase transfer,
rather than building a new finite-difference probe on every iteration. This
changes the search routine only; every accepted bias uses its actual filtered
Hamiltonian. The first campaign exposed excess phase-polishing work; its exact
as-run driver is retained with the results, and the current driver is bounded.

The FFT span is4096ns with a1/256ns sample, independently of the Hamiltonian time
step. The default step is0.0375ns for rises<=1ns and0.075ns otherwise: the first
0.5ns audit found a0.015 percentage-point raw-fidelity shift on step halving.
Initial propagation adds16ns of tail on each side; audit longer windows.
Report BOTH programmed rise+hold+fall span and numerical propagation span.
A perfect brick wall has infinite sinc tails, so padding is a convergence window,
not a newly imposed hardware guard or an exact finite settling time. Holding only
the waveform error below a threshold does not establish a gate-error bound.
The two shapes may run in separate processes with one BLAS thread each. Exact
source/input hashes guard completed checkpoints; do not modify numerical source
while these workers run. Figures/reporting use the saved results only.

## Optional design-report section

After an explicitly requested pulse study, include its definitions, calibrated
rows, figure, numerical scope and measured timings in the design PDF. This is
optional and does not replace its static ON section or require pulses for every
design. Explain commanded10-90% EJ rise, full-edge support, programmed span,
filtered-tail simulation window, idle input/output states, phase-pi objective and
raw/survival-normalized scoring. Keep noise/resonator omissions explicit.
Use identical A4 portrait pages throughout. Include representative bare-frequency
trajectories derived from filtered EJ and loaded EC, and scale grouped comparison panels to fit
instead of splitting them or inserting landscape sheets. `plot_pulse_shape_comparison.py --output
<saved-folder> --tuning-only` builds that schematic without coupled propagation.

```powershell
.venv/Scripts/python.exe Gnd-coupler+Floating-T/pulse_comparison_report.py --report <design.pdf> --study <completed-comparison-folder> --output <updated-design.pdf> --date YYYY-MM-DD
```

The helper reads existing figures and records only. It inserts after the static
gate benchmark, replaces an existing optional insert on repeated edits and keeps
earlier validation status. Speed proposals in the report are suggestions, not
implemented approximations; obtain agreement before further speedup changes.
