# Approved circuit drawing template

User-approved layout, 30 October 2026. Use this layout for this workflow's
grounded-coupler / floating-transmon circuit figures. It is a presentation
template, not a source of numerical design defaults.

- Draw A1, A2, B1 and B2 with left/right floating island terminals. Their
  Josephson junction and self/shunt capacitor are horizontal parallel branches.
  Retain a separate pad-to-ground capacitor on each island. Keep the bodies
  wide and shallow, with short ground legs.
- Draw only the grounded coupler vertically: an asymmetric two-junction SQUID
  and shunt capacitor in parallel, with C2 grounded.
- Both rail links terminate on C1. Include a short vertical lead from the
  coupling bus at C1 down to the coupler body.
- Extend A22 to the right and B11 to the left with short horizontal leads.
  Connect the coupler links and actual direct A22-B11 capacitor at these
  extended terminals. Route the direct capacitor just above the lower coupling
  branch, with compact spacing and no ambiguous wire crossings.
- Label every physical capacitance, loaded EC/h and relevant idle/gate or
  ON/OFF EJ/h. Use exactly two decimal places in circuit labels while retaining
  full precision in numerical inputs. Preserve the actual pad topology.
- Typeset symbols and units mathematically, including subscripts and
  mu Phi_0 / sqrt(Hz). Use vector PDF/SVG output with STIX mathematical fonts.
- Use a regular module pitch, equal rail-body widths and equal coupling-link
  lengths. All ground symbols share a single horizontal baseline; ground
  capacitors also share a common height. Align component labels with their
  symbol centers and retain clear gaps from both plates and connecting wires.
- Place component values below their corresponding transmons to avoid crowded
  wire labels. Render the final report and inspect node connections and text.

## Reusable source

`templates/circuit-schematic.py.in` contains the approved geometry. Substitute
`{{PARAMETERS_JSON}}` and `{{OUTPUT_DIRECTORY}}`, materialize the result as a
script under ignored `output/`, then execute it using the project Python.
The `.py.in` file is a presentation template, not an active numerical module.

The input JSON must supply:

- `physical`: `pad_pairs`, `grounds_ff`, `shunts_ff`, and `links` with `from`,
  `to`, and `capacitance_ff` fields, from the actual circuit reconstruction.
- `rail_labels`: A1/A2/B1/B2 records with `ec_mhz`, `ej_idle_ghz`, and optional
  `ej_gate_ghz` values.
- `coupler_labels`: `ec_mhz`, `ej_off_ghz`, `ej_on_ghz`, `ej1_ghz`, `ej2_ghz`.

The equal-ground caption reflects the present symmetric-ground geometry. If
future input grounds differ, label both values separately instead of claiming
equality. Other topologies require explicit adaptation; do not reuse these
connections merely because the figure is a template.

The approved rendered example and its full-precision parameter JSON are under
`output/coupler-induced-decoherence-report-20261030/`. They remain generated
results excluded from Git; the template and these rules are portable source.

## Optional readout resonators

When requested, draw one shared parallel-LC resonator above A1/A2 and one
above B1/B2, connected through A12-r1-A21 and B12-r2-B21 capacitors. Align
the two resonator grounds with each other on an upper baseline. Keep all
transmon grounds on their common lower baseline. Optional `readouts` input
contains A/B records with `links` (two physical readout links), `capacitance_ff`,
`inductance_nh`, and `target_frequency_ghz`. Keep labels beside the corresponding
branches and put numerical coupling-capacitor values in a separate lower legend.
Do not display resonator Cr or Lr values in the schematic.
A contextual overlay of standalone readout branches is not a newly validated
integrated circuit: state whether capacitances were recompiled and whether
resonator dynamics are included in the reported calculation.

Latest routing refinement: add short outward leads at A12/A21/B12/B21, and
branch the shared resonator links from those exposed terminals. Draw the LC
signal port below and its ground terminal above, with both upper grounds
aligned. Vertical capacitor rendering must handle either endpoint direction:
wire strokes end at the capacitor plates and must never bridge the plate gap.

Final annotation conventions: use the same unhighlighted descriptive header
style for r1/r2 as for A1/A2/B1/B2. Show only fr beneath the resonator name in
the top row. Do not add dashed outlines or colored highlight boxes. Keep only
capacitor identifiers beside the symbols; put every numeric capacitance value
at the bottom, beneath its corresponding module or link, with two decimal places.
Omit repeated fF units and add one shared note: all capacitances are in fF.

Keep equal x/y geometric scaling: junction crosses remain square and capacitor
plates retain the same aspect in either orientation. Preserve the larger symbol
dimension when adjusting the overall layout. Keep the grounded coupler taller
than the shallow rail bodies; do not flatten the SQUID/shunt into a squat box.
Place each label clear of plate ends and wires. Ground-capacitor tags may sit
directly below their own aligned ground symbols to avoid collisions.

Spacing refinement: use equal horizontal pitch for all seven named elements,
A1-r1-A2-C-B1-r2-B2, including their header alignment. Keep the lower ground
baseline slightly below the coupler body to provide a visible C1 vertical lead.
Move the body and ground baseline together when lengthening this lead so the
SQUID proportions do not collapse. Keep A22/B11 labels clear of the bypass
branch wires, and report individual SQUID junction EJs in the coupler header.
# Parasitic visibility

The physical model supports explicitly tagged parasitic capacitors. Omit these
from the circuit diagram, including zero-valued placeholders, unless the user
specifically requests their display. See PARASITICS.md for the port registry.

The explicitly requested group-one overlay uses `parasitics_to_draw` and
`parasitic_capacitances_ff` in the drawing JSON. Render wires, symbols and labels
in dark green (#145A32), with connection dots and overpass bridges. Use the
table-based parameter layout (`show_transmon_parameters=false`). Keep numerical
values in the lower legend, with zero defaults and two decimal places. The
overlay routing lives in studies/circuit_parasitics.py.

When explicitly requested, `group_two_parasitics_to_draw` adds A11-C1, A12-C1,
B21-C1 and B22-C1 in red (#B22222), labelled Cp7-Cp10 above the complete green
group-one layer. Route their common lead to the actual C1 conductor and bridge
all unrelated crossings. Keep their zero-default values in the lower red legend.

The group-one overlay also supports cross-pad A11-A22 and B22-B11, drawn in
dark green as Cp11 and Cp12. Retain Cp1-Cp6 and the red Cp7-Cp10 numbering.
Route the new horizontal branches above the resonator headers, with explicit
overpass bridges at unrelated green/red vertical conductors. Eight green labels
use two rows of four in the lower legend.
