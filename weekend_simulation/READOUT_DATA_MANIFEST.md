# Reusable independent-readout data manifest

These generated datasets are required to reproduce the fixed-rail-A campaign
without rerunning the independent readout studies. They are ignored by Git and
must be transferred separately while preserving the paths below.

| Role | Repository-relative path | Bytes | SHA-256 |
|---|---|---:|---|
| Rail-A fine C1-C2 map, logical 140-160 MHz | `weekend_simulation/readout/fine_c1_c2/rail_a_140_160/results.json` | 23,842,441 | `aea33921718c4065887cb12b626d9d65bd5711db3d6e748b04e2cd396c31f891` |
| Rail-B fine C1-C2 map, logical 80-100 MHz | `weekend_simulation/readout/fine_c1_c2/rail_b_80_100/results.json` | 47,595,194 | `50ab1970c744a72bdc554b8bf069bb0ccb5d92547f5b19dd9435e44c00dc1510` |
| Rail-A ordered direct-EC map, 2 MHz | `weekend_simulation/readout/direct_ec/rail_a_step2/results.json` | 21,182,143 | `6d1e59be6beec3d0aa9a1745d10e947df3534315a826ef4d9a8a6c58f30157ce` |
| Rail-B ordered direct-EC map, 2 MHz | `weekend_simulation/readout/direct_ec/rail_b_step2/results.json` | 79,686,504 | `49d288cefd1e804abb31420b461defe6d6d88505bbf40c524678acb0c20fe7bb` |

The direct-EC files (approximately 101 MB together) are the minimum practical
runtime bundle. All four files (approximately 172 MB) preserve the reusable
fine-map provenance as well.

After transfer, verify from the repository root:

```bash
shasum -a 256 \
  weekend_simulation/readout/fine_c1_c2/rail_a_140_160/results.json \
  weekend_simulation/readout/fine_c1_c2/rail_b_80_100/results.json \
  weekend_simulation/readout/direct_ec/rail_a_step2/results.json \
  weekend_simulation/readout/direct_ec/rail_b_step2/results.json
```

The rail-A direct-EC map contains 1,336 targets and 769 `validated_pass`
ordered pairs. The rail-B direct-EC map contains 4,955 ordered targets. Gate
evaluation must use all 4,955 rail-B targets; the rail-B readout status is used
only for the later independent overlay and reconciliation.

## Regenerate instead of transferring

The branch contains all source needed to regenerate the independent maps. From
the repository root, using the `work` Conda environment:

```bash
conda run -n work python -m readout_workflow.fine_readout_feasibility --run \
  --logical-frequency-mhz 140 160 \
  --output weekend_simulation/readout/fine_c1_c2/rail_a_140_160 \
  --workers 4

conda run -n work python -m readout_workflow.fine_readout_feasibility --run \
  --logical-frequency-mhz 80 100 \
  --output weekend_simulation/readout/fine_c1_c2/rail_b_80_100 \
  --workers 4

conda run -n work python -m readout_workflow.direct_ec_feasibility --run \
  --source weekend_simulation/readout/fine_c1_c2/rail_a_140_160/results.json \
  --output weekend_simulation/readout/direct_ec/rail_a_step2 \
  --domain hull --orientation-domain full --hull-scale 1.5 \
  --ec-step-mhz 2 --logical-frequency-mhz 140 160 --workers 4

conda run -n work python -m readout_workflow.direct_ec_feasibility --run \
  --source weekend_simulation/readout/fine_c1_c2/rail_b_80_100/results.json \
  --output weekend_simulation/readout/direct_ec/rail_b_step2 \
  --domain hull --orientation-domain full --hull-scale 1.5 \
  --ec-step-mhz 2 --logical-frequency-mhz 80 100 --workers 4
```

Run the fine maps first because each direct-EC map derives its scaled hull and
warm-start witnesses from the corresponding fine C1-C2 result. All four
commands are resumable when invoked with the same settings and source version.
Use new output directories after changing code or parameters.

Fresh optimization may not be byte-for-byte identical across machines or
software versions, and first-feasible witnesses can differ even when the
feasible region is equivalent. Transfer the archived files when the coworker
must use exactly the same EC target sets and readout witnesses; regenerate
when an independent reproduction is desired.
