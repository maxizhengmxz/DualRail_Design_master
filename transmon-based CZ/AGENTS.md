# Toy-model workflow guidance

Read README.md and WORKFLOW.md before changing this workflow. WORKFLOW.md and
transmon_cz/channel_policy.py own dual-rail logical definitions, S1/S2 scoring
and minimum-S1 selection at common intended g for toy and circuit workflows.
Coupler code must import these definitions, never duplicate them.

Logical 0/1 mean lower/upper isolated-pair one-excitation modes; pair ground is
separate. Physical auxiliary occupation alone never fixes the logical source.
Choose one feasible minimum-S1 resonance after evaluating candidates at common
intended g and in the same score basis. Apply the stated logical-origin ratio
screen. S2 and crossing order do not rank candidates. Uniformly rescaled scores
are diagnostics, not certified designs. Preserve the selected ON CZ channel
through gate calibration. OFF in the explicit coupler workflow temporarily nulls physical 0100-0010
exchange in five/seven modes, with all four physical single-excitation
references retained. Three modes null physical A2/B1 01-10. Logical exchanges,
CZ-auxiliary couplings and secular ZZ are residual diagnostics. This temporary
physical-versus-logical choice is an open question; see Gnd-coupler+Floating-T/OFF_POLICY.md.

Missing or ambiguous state data
must not become an implicit 01, 10 or 11 default. Old first-crossing records must
be rejected rather than silently migrated. Active S1 toy search is in 2Q-gate_XM;
it may import the stable policy here without importing exploratory kernels.

Preserve historical results and clearly label fixed-source diagnostics. A
definition update does not authorize design searches or readout optimization.
Run relevant regression tests after source changes.
