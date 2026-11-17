# Codex task: install transferred independent-readout data

This file is written for the Codex agent working in the recipient's clone of
`DualRail_Design_master`. Follow it before launching the fixed-rail-A EC-tuple
readout/gate workflow.

The active workflow source is now under `Ec_tuple search workflow/`. Do not
install these inputs under the historical `weekend_simulation/` directory.

## 1. Files supplied by the user

Ask the user where the four transferred files were downloaded. The recommended
transfer names are:

```text
rail_a_step2_results.json
rail_a_step2_settings.json
rail_b_step2_results.json
rail_b_step2_settings.json
```

Both original result files were named `results.json`, and both settings files
were named `settings.json`. Do not infer rail identity from a generic filename
or file size. If the sender did not rename them, ask which original directory
each file came from.

## 2. Required installation layout

From the repository root, create missing directories and install the files at
these exact repository-relative paths:

```text
Ec_tuple search workflow/
└── readout/
    └── direct_ec/
        ├── rail_a_step2/
        │   ├── results.json
        │   └── settings.json
        └── rail_b_step2/
            ├── results.json
            └── settings.json
```

Destination mapping:

| Transferred file | Destination |
|---|---|
| `rail_a_step2_results.json` | `Ec_tuple search workflow/readout/direct_ec/rail_a_step2/results.json` |
| `rail_a_step2_settings.json` | `Ec_tuple search workflow/readout/direct_ec/rail_a_step2/settings.json` |
| `rail_b_step2_results.json` | `Ec_tuple search workflow/readout/direct_ec/rail_b_step2/results.json` |
| `rail_b_step2_settings.json` | `Ec_tuple search workflow/readout/direct_ec/rail_b_step2/settings.json` |

The directory name contains spaces. Quote every shell path. Prefer copying to
moving so the downloaded originals remain recoverable until verification is
complete.

## 3. Safe installation procedure

1. Locate the repository root and the four transferred files.
2. Create only the two missing destination directories.
3. Before replacing an existing destination, calculate its SHA-256 checksum.
4. If an existing result file already has the expected checksum, keep it.
5. If an existing destination differs, preserve it and ask the user whether to
   use a new directory or replace it. Never silently overwrite simulation data.
6. Copy each transferred file to its mapped destination.
7. Do not add the installed data to Git. The complete
   `Ec_tuple search workflow/readout/` directory is intentionally ignored.
8. Do not regenerate an independent readout map when valid transferred data
   are available.

Example directory creation from the repository root:

```bash
mkdir -p \
  "Ec_tuple search workflow/readout/direct_ec/rail_a_step2" \
  "Ec_tuple search workflow/readout/direct_ec/rail_b_step2"
```

Use the actual download paths in the copy commands. Do not blindly copy the
example names if they do not exist.

## 4. Verify result-file identity

Run from the repository root:

```bash
shasum -a 256 \
  "Ec_tuple search workflow/readout/direct_ec/rail_a_step2/results.json" \
  "Ec_tuple search workflow/readout/direct_ec/rail_b_step2/results.json"
```

Expected output:

```text
6d1e59be6beec3d0aa9a1745d10e947df3534315a826ef4d9a8a6c58f30157ce  Ec_tuple search workflow/readout/direct_ec/rail_a_step2/results.json
49d288cefd1e804abb31420b461defe6d6d88505bbf40c524678acb0c20fe7bb  Ec_tuple search workflow/readout/direct_ec/rail_b_step2/results.json
```

Then validate the JSON contents:

```bash
conda run -n work python -c "import json; from pathlib import Path; root=Path('Ec_tuple search workflow/readout/direct_ec'); a=json.loads((root/'rail_a_step2/results.json').read_text()); b=json.loads((root/'rail_b_step2/results.json').read_text()); ap=sum(x.get('status')=='validated_pass' for x in a); bp=sum(x.get('status')=='validated_pass' for x in b); assert (len(a),ap)==(1336,769), (len(a),ap); assert (len(b),bp)==(4955,2975), (len(b),bp); print({'rail_a_targets':len(a),'rail_a_pass':ap,'rail_b_targets':len(b),'rail_b_pass':bp})"
```

Expected counts:

- rail A: 1,336 targets, 769 `validated_pass`;
- rail B: 4,955 targets, 2,975 `validated_pass`.

## 5. Verify settings and Git exclusion

Inspect both installed `settings.json` files and require:

- `ec_step_mhz`: 2.0;
- `target_domain`: `hull`;
- `orientation_domain`: `full`;
- `hull_scale`: 1.5;
- rail-A `logical_frequency_mhz`: `[140.0, 160.0]`;
- rail-B `logical_frequency_mhz`: `[80.0, 100.0]`.

Confirm that Git ignores the installed files:

```bash
git check-ignore -v \
  "Ec_tuple search workflow/readout/direct_ec/rail_a_step2/results.json" \
  "Ec_tuple search workflow/readout/direct_ec/rail_b_step2/results.json"
```

Both paths should be reported as ignored by the root `.gitignore` rule for
`Ec_tuple search workflow/readout/`.

## 6. Confirm the active code can find the data

The current fixed-A workflow reads these paths relative to its own directory:

- `run_sampled_anchor_repetition.py` uses the rail-A 2 MHz map for warm starts;
- `prepare_fixed_a_anchor_campaign.py`, `postprocess_anchor_campaign.py`,
  `reconcile_anchor_campaign.py`, and the plotting scripts use the rail-B 2 MHz
  map for the full ordered gate target set and later independent readout
  overlay/reconciliation.

Do not redirect these scripts to the old `weekend_simulation/readout/` path.

After installation, read:

```text
Ec_tuple search workflow/CODEX_HANDOFF.md
Ec_tuple search workflow/DEVELOPMENT_MEMORY.md
Ec_tuple search workflow/READOUT_DATA_MANIFEST.md
2Q-gate_XM/AGENTS.md
```

before configuring or launching a campaign.

## 7. Completion report

Report to the user:

- the four installed absolute paths;
- whether each destination directory was created or already existed;
- the two SHA-256 checksums;
- the target and validated-pass counts;
- the six settings checks above;
- confirmation that Git ignores both installed result files;
- any mismatch or pre-existing file that was preserved.
