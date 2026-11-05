"""VS Code interactive study of the readout range at C1=100 fF, C2=75 fF.

Run this file cell-by-cell using VS Code's ``Run Cell`` commands. The numerical
helpers live in ``readout_fixed_pair_scan.py`` so this file stays focused on
inspection, plots and deliberate changes to the study policy.
"""

# %% Locate the repository when cells are launched from another working folder.
from pathlib import Path
import json
import sys


def _find_project_root() -> Path:
    starts = [Path.cwd().resolve()]
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists() and (
                candidate / "readout_workflow"
            ).is_dir():
                return candidate
    raise RuntimeError(
        "Start inside DualRail_Design_master or open this file directly."
    )


PROJECT_ROOT = _find_project_root()
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
for search_path in (PROJECT_ROOT, EXAMPLES_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))


# %% Configure numerical libraries before importing NumPy/SciPy users.
from core.runtime import configure_single_threaded_blas


configure_single_threaded_blas()

import matplotlib.pyplot as plt
import numpy as np

import readout_fixed_pair_scan as range_scan
from readout_workflow import calculation as study


# %% Study policies -- edit the fixed values before a production run.
C1_FF = range_scan.C1_FF
C2_FF = range_scan.C2_FF
SOBOL_STARTS = 8  # Use 0-4 for a smoke test; increase to 16, 32, ...
MAX_STARTS = 16  # Total starts allowed per endpoint objective.
SOBOL_SEED = 10075

# None means that coordinate remains a continuous optimizer variable. Add or
# remove policies freely; every entry is run independently in the cells below.
RUN_POLICIES = {
    "variable_capacitances": {
        "fixed_c12_ff": None,
        "fixed_readout_cap_ff": None,
    },
    "fixed_readout_capacitances": {
        "fixed_c12_ff": None,  # C12 in fF
        "fixed_readout_cap_ff": 10.0,  # common C1r=C2r in fF
    },
}

OUTPUT_ROOT = PROJECT_ROOT / "readout_workflow" / "data" / "fixed_pair_100_75"
print(json.dumps(RUN_POLICIES, indent=2))


# %% Inspect the effective continuous ranges for every policy.
coordinate_names = (
    ("mean local frequency", "GHz"),
    ("detuning coordinate", "10 MHz"),
    ("C12", "fF"),
    ("C1r=C2r", "fF"),
)
for policy_name, policy in RUN_POLICIES.items():
    lower, upper = range_scan.search_bounds(**policy)
    print(f"\n{policy_name}")
    for index, (coordinate, unit) in enumerate(coordinate_names):
        status = "fixed" if lower[index] == upper[index] else "variable"
        print(
            {
                "coordinate": coordinate,
                "lower": lower[index],
                "upper": upper[index],
                "unit": unit,
                "status": status,
            }
        )


# %% Generate policy-specific Sobol starts.
policy_seeds = {
    name: range_scan.sobol_seeds(SOBOL_STARTS, SOBOL_SEED, **policy)
    for name, policy in RUN_POLICIES.items()
}
for name, seeds in policy_seeds.items():
    seed_array = np.asarray(seeds)
    print(name, "Sobol starts:", len(seed_array))
    if len(seed_array):
        print("  C12 range:", seed_array[:, 2].min(), seed_array[:, 2].max())
        print("  C1r=C2r range:", seed_array[:, 3].min(), seed_array[:, 3].max())


# %% Run both endpoint objectives for every policy. This is the expensive cell.
endpoint_runs = {}
for name, policy in RUN_POLICIES.items():
    print(f"Starting {name}")
    lower_point, lower_attempts = range_scan.find_boundary(
        "min", policy_seeds[name], MAX_STARTS, **policy
    )
    upper_point, upper_attempts = range_scan.find_boundary(
        "max", policy_seeds[name], MAX_STARTS, **policy
    )
    endpoint_runs[name] = {
        "lower_point": lower_point,
        "upper_point": upper_point,
        "lower_attempts": lower_attempts,
        "upper_attempts": upper_attempts,
    }
    print(
        name,
        "completed attempts:",
        len(lower_attempts),
        "+",
        len(upper_attempts),
    )


# %% Compare audited endpoint results.
def endpoint_row(policy_name, endpoint, point):
    if point is None:
        return {
            "policy": policy_name,
            "endpoint": endpoint,
            "accepted": False,
            "failures": ["no_candidate"],
        }
    failures = study.failures(point, require_sign_check=True)
    return {
        "policy": policy_name,
        "endpoint": endpoint,
        "accepted": not failures,
        "logical_frequency_MHz": 1e3 * point["splitting_ghz"],
        "C12_fF": point["device"]["C12_fF"],
        "C1r_C2r_fF": point["device"]["C1r_fF"],
        "EJ1_GHz": point["ej1_ghz"],
        "EJ2_GHz": point["ej2_ghz"],
        "absolute_mean_chi_MHz": study.absolute_mean_chi_mhz(point),
        "chi_mismatch": point["ratio"],
        "hybridization": point["hybridization"],
        "dispersive_ratio": point["lambda_max"],
        "assignment_overlap": point["min_assignment_overlap"],
        "failures": failures,
    }


endpoint_table = []
for name, run in endpoint_runs.items():
    endpoint_table.extend(
        [
            endpoint_row(name, "minimum", run["lower_point"]),
            endpoint_row(name, "maximum", run["upper_point"]),
        ]
    )
endpoint_table


# %% Summarize and plot the best-found ranges.
range_summary = {}
for name, run in endpoint_runs.items():
    lower_point = run["lower_point"]
    upper_point = run["upper_point"]
    if lower_point is None or upper_point is None:
        range_summary[name] = {
            "lower_found": lower_point is not None,
            "upper_found": upper_point is not None,
        }
        continue
    low_mhz = 1e3 * lower_point["splitting_ghz"]
    high_mhz = 1e3 * upper_point["splitting_ghz"]
    range_summary[name] = {
        "minimum_logical_frequency_MHz": low_mhz,
        "maximum_logical_frequency_MHz": high_mhz,
        "best_found_span_MHz": high_mhz - low_mhz,
    }

figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
for index, (name, summary) in enumerate(range_summary.items()):
    if "minimum_logical_frequency_MHz" not in summary:
        continue
    low = summary["minimum_logical_frequency_MHz"]
    high = summary["maximum_logical_frequency_MHz"]
    axis.plot([low, high], [index, index], marker="o", linewidth=3)
axis.set(
    yticks=range(len(RUN_POLICIES)),
    yticklabels=list(RUN_POLICIES),
    xlabel="Logical frequency (MHz)",
    title="Best-found audited range by capacitance policy",
)
axis.grid(axis="x", alpha=0.25)
plt.show()
range_summary


# %% Optional full continuation scans for the same policies.
# Set this True only after the endpoint comparison is satisfactory.
RUN_FULL_SCANS = False
full_scan_results = {}
if RUN_FULL_SCANS:
    for name, policy in RUN_POLICIES.items():
        full_scan_results[name] = range_scan.run_study(
            mode="all",
            output=OUTPUT_ROOT / name,
            low_mhz=90.0,
            high_mhz=110.0,
            step_mhz=0.5,
            target_tolerance_mhz=0.02,
            starts=MAX_STARTS,
            sobol_starts_count=SOBOL_STARTS,
            seed=SOBOL_SEED,
            **policy,
        )
else:
    print("Full scans disabled; set RUN_FULL_SCANS=True to run both policies.")


# %% Optional endpoint export after reviewing the comparison.
EXPORT_RESULTS = False
if EXPORT_RESULTS:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "fixed-pair-logical-extrema-interactive-v2",
        "pair_fF": [C1_FF, C2_FF],
        "run_policies": RUN_POLICIES,
        "range_summary": range_summary,
        "endpoints": endpoint_table,
    }
    output_path = OUTPUT_ROOT / "interactive_policy_comparison.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved", output_path)
else:
    print("Export disabled; set EXPORT_RESULTS=True after reviewing the run.")
