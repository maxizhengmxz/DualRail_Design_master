"""Safe coarse-to-fine refinement of a completed resonator-tolerance search.

Reuse a certified lower bound and failed-search upper bracket, not the entire
coarse search. Old data are read-only. Physics/acceptance compatibility is
checked separately from optimizer source changes. Every enlarged design still
receives full sampled validation. Results remain finite-budget witnesses.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from . import calculation as study
from .storage import atomic_json, utc

ROOT = Path(__file__).resolve().parents[1]
PHYSICS_MODULES = (
    "readout_workflow/calculation.py",
    "readout_workflow/model.py",
    "readout_workflow/validation.py",
    "core/transmon.py",
    "core/electrostatics.py",
    "core/operating_point_dispersive.py",
)
EXTRA_PHYSICS = (
    "core/circuits.py",
    "core/constants.py",
    "core/runtime.py",
    "core/tracking.py",
)
PHYSICS_FUNCTIONS = (
    ("readout_workflow/direct_ec_feasibility.py", "capacitances"),
    ("readout_workflow/ec_design_improvement.py", "slacks"),
    ("readout_workflow/ec_resonator_tolerance.py", "at_frequency"),
    ("readout_workflow/ec_resonator_tolerance.py", "set_policy"),
)


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _semantic_tree(text, function=None):
    """Ignore formatting, positions and docstrings, never numerical statements."""
    tree = ast.parse(text)
    if function:
        tree = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == function
        )

    class WithoutDocs(ast.NodeTransformer):
        def generic_visit(self, node):
            node = super().generic_visit(node)
            body = getattr(node, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]
            return node

    return ast.dump(WithoutDocs().visit(tree), include_attributes=False)


def _check_source(source, settings, relative, function=None, required=True):
    records = settings.get("source_hashes", {})
    key = next((k for k in records if k.replace("\\", "/").endswith(relative)), None)
    if key is None:
        if required:
            raise ValueError(f"Missing coarse physics provenance: {relative}")
        return dict(file=relative, comparison="not_recorded_by_older_run")
    snapshot = source / "source_snapshot" / relative
    if (
        not snapshot.exists()
        or sha256(snapshot.read_bytes()).hexdigest() != records[key]
    ):
        raise ValueError(f"Coarse source snapshot missing or changed: {relative}")
    current = ROOT / relative
    if function is None and sha256(current.read_bytes()).hexdigest() == records[key]:
        return dict(file=relative, comparison="identical_bytes")
    if _semantic_tree(
        snapshot.read_text(encoding="utf-8-sig"), function
    ) != _semantic_tree(current.read_text(encoding="utf-8-sig"), function):
        raise ValueError(
            f"Physics implementation changed: {relative}"
            + (f":{function}" if function else "")
        )
    return dict(file=relative, function=function, comparison="same_semantics")


def load_refinement(source, ec_mhz, step, ceiling, resolution):
    """Verify coarse coverage and certificates before reusing its bracket.

    Historical optimizer policies may differ: a feasible design remains a valid
    lower bound if physics, acceptance, nominal ECs, frequency coverage and the
    fabrication convention are unchanged. Optimizer hashes need not match.
    """
    source = Path(source).resolve()
    required = ("settings.json", "completion.json", "results.json")
    for name in required:
        if not (source / name).exists():
            raise ValueError(
                f"Refinement requires completed coarse output: {source / name}"
            )
    settings = json.loads((source / "settings.json").read_text(encoding="utf-8-sig"))
    complete = json.loads((source / "completion.json").read_text(encoding="utf-8-sig"))
    args = settings["arguments"]
    if args.get("mode") != "tolerance" or complete.get("phase") != "complete":
        raise ValueError("Only a completed tolerance-mode result can be refined")
    if _canonical(settings["conditions"]) != _canonical(asdict(study.CONDITIONS)):
        raise ValueError("Coarse acceptance conditions differ; start a new search")
    if list(map(float, args["ec"])) != list(map(float, ec_mhz)):
        raise ValueError("Coarse EC tuple differs")
    if abs(args["frequency_step_ghz"] - step) > 1e-10:
        raise ValueError(
            "Frequency-grid changes require separate validation, not boundary-only refinement"
        )
    if abs(args["max_halfwidth_ghz"] - ceiling) > 1e-10:
        raise ValueError(
            "Range ceiling differs; boundary refinement does not extend a censored search"
        )
    if resolution > args["boundary_resolution_ghz"] + 1e-10:
        raise ValueError("Refinement resolution must be no coarser than the source")
    checks = [_check_source(source, settings, f) for f in PHYSICS_MODULES]
    checks += [
        _check_source(source, settings, f, required=False) for f in EXTRA_PHYSICS
    ]
    checks += [
        _check_source(source, settings, f, function=fn) for f, fn in PHYSICS_FUNCTIONS
    ]
    results = json.loads((source / "results.json").read_text(encoding="utf-8-sig"))
    candidates = [
        row["result"]
        for row in results
        if row["result"].get("status") == "validated_sampled_range"
    ]
    if not candidates:
        raise ValueError("No validated coarse design to preserve")
    # The final chi-optimized witness of an older run is also a valid robust
    # starting design. Prefer it over reoptimizing the coarse interval.
    certified = max(candidates, key=lambda r: r["worst_mean_chi_mhz"])
    boundary = complete["boundary"]
    good = float(boundary["validated_halfwidth_ghz"])
    bad = boundary.get("failed_search_halfwidth_ghz")
    bad = float(bad) if bad is not None else None
    if good < 0 or (bad is not None and bad <= good) or good > ceiling + 1e-9:
        raise ValueError("Invalid coarse boundary bracket")
    if any(
        abs(a - b) > 1e-8
        for a, b in zip(certified["range_ghz"], [6 - good, 6 + good], strict=True)
    ):
        raise ValueError("Saved design does not cover the reported coarse interval")
    if bad is None and not boundary.get("censored_by_search_ceiling"):
        raise ValueError(
            "Coarse search has neither a failed bracket nor a censored ceiling"
        )
    validation = certified["validation"]
    if not validation.get("passed") or validation.get("failed_frequencies_ghz"):
        raise ValueError("Coarse interval lacks passing validation")
    samples = validation["samples"]
    frequencies = [v["frequency_ghz"] for v in samples]
    for target in (6 - good, 6.0, 6 + good):
        if not any(abs(f - target) < 1e-8 for f in frequencies):
            raise ValueError(
                "Coarse validation is missing an endpoint or nominal frequency"
            )
    if validation.get("validation_step_ghz", float("inf")) > step / 2 + 1e-10:
        raise ValueError("Coarse validation frequency grid is too sparse")
    ordered = sorted(set(frequencies))
    if any(
        b - a > step / 2 + 1e-8 for a, b in zip(ordered[:-1], ordered[1:], strict=True)
    ):
        raise ValueError("Coarse validation has a frequency gap larger than allowed")
    reference = samples[0]["point"]["device"]
    for sample in samples:
        point = sample.get("point")
        if (
            point is None
            or sample.get("failures")
            or study.failures(point, require_sign_check=True)
        ):
            raise ValueError("Coarse witness fails current certificate checks")
        if any(abs(point["device"][k] - reference[k]) > 1e-8 for k in study.CAPS):
            raise ValueError(
                "Coarse interval changes fabricated capacitors between frequencies"
            )
    nominal = next(v["point"] for v in samples if abs(v["frequency_ghz"] - 6) < 1e-8)
    if any(abs(nominal[f"loaded_ec{i}_mhz"] - ec_mhz[i - 1]) > 1e-5 for i in (1, 2)):
        raise ValueError("Coarse nominal loaded EC does not match target")
    records = {
        name: sha256((source / name).read_bytes()).hexdigest() for name in required
    }
    state = dict(
        source=str(source),
        good=good,
        bad=bad,
        certified=certified,
        censored=bool(boundary.get("censored_by_search_ceiling")),
        provenance=dict(
            coarse_files_sha256=records,
            physics_checks=checks,
            source_resolution_ghz=args["boundary_resolution_ghz"],
            target_resolution_ghz=resolution,
            reused_witnesses=len(samples),
        ),
    )
    return state


def _warm(certified):
    samples = certified["validation"]["samples"]
    return dict(
        frequencies_ghz=[v["frequency_ghz"] for v in samples],
        points=[v["point"] for v in samples],
        worst_mean_chi_mhz=certified["worst_mean_chi_mhz"],
    )


def refine_tolerance(state, ec, nominal, folder, starts, maxiter, step, resolution):
    """Only bisect the saved bracket; never redo expansion or lose a coarse pass."""
    from .resonator_robust_design import frequency_grid, optimize_scenarios, run_range

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    good, bad = state["good"], state["bad"]
    best = state["certified"]
    warm = _warm(best)
    iteration = 0
    new_trials = 0
    reused_trials = 0
    atomic_json(folder / "provenance.json", state["provenance"])
    while bad is not None and bad - good > resolution + 1e-10:
        width = (good + bad) / 2
        trial = folder / f"halfwidth_{width:.10f}"
        decision = trial / "decision.json"
        if decision.exists():
            reused_trials += 1
            saved = json.loads(decision.read_text())
            result = (
                json.loads((trial / "validated" / "result.json").read_text())
                if saved["passed"]
                else None
            )
        else:
            new_trials += 1
            fit = optimize_scenarios(
                ec,
                nominal,
                frequency_grid(6 - width, 6 + width, max(step, 2 * width)),
                study.CONDITIONS.hybridization_min,
                trial / "screen",
                starts=starts,
                maxiter=maxiter,
                warm=warm,
                feasibility_only=True,
            )
            result = None
            if fit["best"]:
                result = run_range(
                    ec,
                    nominal,
                    6 - width,
                    6 + width,
                    study.CONDITIONS.hybridization_min,
                    trial / "validated",
                    starts,
                    maxiter,
                    step,
                    warm_seed=fit["best"],
                    feasibility_only=True,
                )
            saved = dict(
                halfwidth_ghz=width,
                passed=result is not None
                and result["status"] == "validated_sampled_range",
            )
            atomic_json(decision, saved)
        if saved["passed"]:
            good = width
            best = result
            warm = _warm(best)
        else:
            bad = width
        iteration += 1
        atomic_json(
            folder / "progress.json",
            dict(
                validated_halfwidth_ghz=good,
                failed_search_halfwidth_ghz=bad,
                refinement_trials=iteration,
                updated_utc=utc(),
            ),
        )
    return dict(
        status="validated_sampled_range",
        range_ghz=[6 - good, 6 + good],
        boundary=dict(
            validated_halfwidth_ghz=good,
            failed_search_halfwidth_ghz=bad,
            censored_by_search_ceiling=state["censored"],
        ),
        validated_result=best,
        refinement=dict(
            source=state["source"],
            trials=iteration,
            new_width_trials=new_trials,
            cached_width_trials=reused_trials,
            coarse_lower_bound_preserved=good >= state["good"],
            provenance=state["provenance"],
        ),
    )
