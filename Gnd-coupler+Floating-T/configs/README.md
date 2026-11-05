# Configuration layout

`*template.json` files are reusable input ports. The master workflow request is
schema v5, not itself a directly executable stage config. The active OFF goal is
temporarily physical 0100-0010 in both five/seven modes; logical exchanges are
residuals. Confirm constraints per case; latest case exceptions are not defaults.

Historical requests are archived under ignored `output/retired-2026-10-28/legacy-inputs/`.
Regression fixtures live under `tests/fixtures/`, outside the active config ports.
Historical selected readout cases are in ignored output/retired-2026-10-29/case-inputs/.
Fresh readout design is a final workflow stage and is not launched automatically.

The master request includes fourteen independent zero-default physical parasitic
capacitors in `parasitic_capacitances_ff`; see ../PARASITICS.md for endpoints,
subsystem input locations and diagram visibility policy.

`Design_V1_20261105.json` is the frozen nominal seven-mode design with all
fourteen parasitics explicitly zero. It includes physical capacitances,
resonator parameters, loaded ECs, selected CZ/OFF conventions, idle/gate bare
frequencies and EJs, duration, and all five asymmetric-SQUID junction designs.
The A tuning range remains 4-5 GHz. Use this record explicitly as a saved input,
not as implicit defaults for another design. Its nominal OFF retains the known
basis-convergence qualification. Both `--input` and `--gate` in the optional
parasitic preparation workflow may point to this same parameters-only record.
