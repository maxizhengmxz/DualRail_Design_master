# Archived studies outside the three active workflows

These files are unused by the current readout, coupler-off, and
logical-resonance gate workflows. They are retained research code, not all
unreferenced/dead functions. Numerical algorithms and input values are
unchanged; only locations, imports and notebook path setup were adjusted.

## Where the archived helpers are used

| Helper in `analysis/` | Existing callers |
| --- | --- |
| `monte_carlo.py` | `notebooks/dtc_monte_carlo.py`, `notebooks/dtc_dt_comparison.py`; also imported by other archived notebook setup cells |
| `sampling.py` | Archived Monte Carlo and tunable-resonator helpers; sampling tests |
| `sweeps.py` | `notebooks/dtc_tracking.py` |
| `tdt_operating_point_monte_carlo.py` | `notebooks/tdt_monte_carlo.py` |
| `tunable_resonator.py` | `notebooks/dtc_dt_tunable_r.py` |

`notebooks/aws_tdt.py` is an older tunable-pair/readout exploration; it uses
shared operating-point/dispersive tools from `core`, but is not the recent
readout workflow. `main.py` is the old greeting-only placeholder.

## Dependencies and compatibility

Archived code uses `legacy.analysis.*` for these helpers and `core.*` for
shared circuit construction, constants, tracking, runtime and storage. It
does not import or launch the recent readout/coupler-off workflows.

Archived callers import legacy.analysis directly. Core no longer forwards
legacy functions, and active workflows do not depend on this archive.

Run archived notebooks from the repository root to preserve working-directory
based data/cache paths, for example:

```powershell
uv run marimo edit legacy/notebooks/dtc_monte_carlo.py
uv run python -m unittest discover -s legacy/tests -t .
```

Dedicated sampling, Monte Carlo and sweep tests moved to `legacy/tests/`.
The circuit-factory tests remain in `tests/` because they verify shared core
constructors still used by active gate studies. No saved outputs were moved.
