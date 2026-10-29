"""Interactive single-pair example using the current readout_workflow model.

The default cells hold fabricated capacitances fixed and search only EJ1/EJ2.
An optional final cell demonstrates the pre-fabrication design optimizer.
"""

# %% Locate the repository.
from pathlib import Path
import sys


def _find_project_root() -> Path:
    starts = [Path.cwd().resolve()]
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "readout_workflow").is_dir():
                return candidate
    raise RuntimeError("Start inside DualRail_Design_master or install the project.")


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# %% Configure numerical libraries before importing NumPy/SciPy users.
from core.runtime import configure_single_threaded_blas


configure_single_threaded_blas()

import matplotlib.pyplot as plt
import numpy as np

from core.transmon import Basis
from readout_workflow import calculation as study
from readout_workflow.model import find_readout_points
from readout_workflow.validation import audit_point, metrics


# %% Fixed fabricated circuit.
# Values reproduce the report's first-round GDS design before rounding caveats.
# All capacitances are physical fF; Lr is nH.
DEVICE = study.Device(
    C1_shunt_fF=90.0,
    C2_shunt_fF=75.0,
    C12_fF=2.05,
    C1r_fF=7.74,
    C2r_fF=7.74,
    Cr_fF=530.516477,
    Lr_nH=1.326291192,
)

SCREENING_BASIS = Basis()
model = study.ReadoutModel(DEVICE, SCREENING_BASIS)

print("Capacitance matrix (fF):")
print(model.device.capacitance_matrix() * 1e15)
print("Loaded EC/h (MHz):", 1e3 * model.ec_ghz[:2])
print("Loaded resonator frequency (GHz):", model.fr_ghz)


# %% Evaluate the reported nominal EJ values directly.
REFERENCE_EJ1_GHZ = 13.269374
REFERENCE_EJ2_GHZ = 11.445213

reference = study.annotate(model, REFERENCE_EJ1_GHZ, REFERENCE_EJ2_GHZ)
reference_summary = {
    "operation_frequency_ghz": reference["operating_frequency_ghz"],
    "logical_frequency_mhz": 1e3 * reference["splitting_ghz"],
    "hybridization": reference["hybridization"],
    "chi_minus_mhz": 1e3 * reference["chi_m_ghz"],
    "chi_plus_mhz": 1e3 * reference["chi_p_ghz"],
    "absolute_mean_chi_mhz": study.absolute_mean_chi_mhz(reference),
    "matching_ratio": reference["ratio"],
    "multilevel_dispersive_ratio": reference["lambda_max"],
    "minimum_assignment_overlap": reference["min_assignment_overlap"],
    "nominal_failures_before_sign_probe": study.failures(reference),
}
reference_summary


# %% Search only tunable EJ1/EJ2 for this fixed fabricated circuit.
# Start coarse, inspect the result, then widen or refine the ranges deliberately.
ej1_values = np.linspace(11.0, 16.0, 31)
ej2_values = np.linspace(9.0, 14.0, 31)

candidates, coarse_scan = find_readout_points(
    model,
    ej1_values,
    ej2_values,
    min_hybridization=study.CONDITIONS.hybridization_min,
    xtol=1e-5,
)
annotated = [study.annotate(model, p["ej1_ghz"], p["ej2_ghz"]) for p in candidates]

print(f"Coarse evaluations: {len(coarse_scan)}")
print(f"Hybridized splitting minima: {len(annotated)}")


# %% Build a readable candidate table without adding a dataframe dependency.
def candidate_row(point):
    return {
        "EJ1_GHz": point["ej1_ghz"],
        "EJ2_GHz": point["ej2_ghz"],
        "operation_GHz": point["operating_frequency_ghz"],
        "logical_MHz": 1e3 * point["splitting_ghz"],
        "H": point["hybridization"],
        "chi_minus_MHz": 1e3 * point["chi_m_ghz"],
        "chi_plus_MHz": 1e3 * point["chi_p_ghz"],
        "abs_mean_chi_MHz": study.absolute_mean_chi_mhz(point),
        "matching_ratio": point["ratio"],
        "dispersive_ratio": point["lambda_max"],
        "assignment_overlap": point["min_assignment_overlap"],
        "failures": study.failures(point),
    }


candidate_table = sorted(
    (candidate_row(point) for point in annotated),
    key=lambda row: (len(row["failures"]), -row["abs_mean_chi_MHz"]),
)
candidate_table


# %% Visualize the search and the retained minima.
figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

if coarse_scan:
    axes[0].scatter(
        [p["ej1_ghz"] for p in coarse_scan],
        [p["ej2_ghz"] for p in coarse_scan],
        c=1e3 * np.array([p["splitting_ghz"] for p in coarse_scan]),
        s=10,
        cmap="viridis",
    )
if annotated:
    axes[0].scatter(
        [p["ej1_ghz"] for p in annotated],
        [p["ej2_ghz"] for p in annotated],
        marker="x",
        color="red",
        label="refined minima",
    )
axes[0].set(xlabel="EJ1/h (GHz)", ylabel="EJ2/h (GHz)", title="Fixed-circuit search")
axes[0].legend(frameon=False)

if annotated:
    frequency = np.array([p["operating_frequency_ghz"] for p in annotated])
    axes[1].scatter(frequency, 1e3 * np.array([p["splitting_ghz"] for p in annotated]))
    axes[1].axhspan(90, 110, color="C2", alpha=0.15)
    axes[2].semilogy(frequency, [p["ratio"] for p in annotated], ".")
axes[1].set(xlabel="Operation frequency (GHz)", ylabel="Logical frequency (MHz)")
axes[2].axhline(study.CONDITIONS.mismatch_max, color="black", linestyle="--")
axes[2].set(xlabel="Operation frequency (GHz)", ylabel="Mean-chi mismatch")
for axis in axes:
    axis.grid(alpha=0.25)
figure


# %% Run the independent sign-stability certificate on nominally viable points.
nominally_viable = [p for p in annotated if not study.failures(p)]
certified = [study.check_chi_sign_stability(p, model) for p in nominally_viable]
certified = [p for p in certified if not study.failures(p, require_sign_check=True)]
print(f"Nominally viable before sign probes: {len(nominally_viable)}")
print(f"Certified after sign probes: {len(certified)}")


# %% Audit the strongest certified point at the larger production basis.
if certified:
    strongest = max(certified, key=study.absolute_mean_chi_mhz)
    audited = audit_point(strongest, sign=True)
    audit_summary = {
        **metrics(audited),
        "EJ1_GHz": audited["ej1_ghz"],
        "EJ2_GHz": audited["ej2_ghz"],
        "logical_frequency_MHz": 1e3 * audited["splitting_ghz"],
        "operation_frequency_GHz": audited["operating_frequency_ghz"],
        "H": audited["hybridization"],
        "J12_GHz": audited["J12_ghz"],
        "J01_12_MHz": audited["J01_12_mhz"],
        "failures": study.failures(audited, require_sign_check=True),
    }
else:
    audit_summary = {
        "status": "No candidate passed. Inspect failures before changing bounds or policy."
    }
audit_summary


# %% Optional pre-fabrication design search.
# This is a different question: C12 and the common C1r=C2r are allowed to vary.
# Keep False unless you intend to explore alternative fabricated designs.
RUN_DESIGN_SEARCH = False
if RUN_DESIGN_SEARCH:
    design_result = study.search_and_validate(
        {"pair": (DEVICE.C1_shunt_fF, DEVICE.C2_shunt_fF), "level": "fast"}
    )
else:
    design_result = {
        "status": "disabled",
        "reason": "Operational example keeps fabricated capacitances fixed.",
    }
design_result
