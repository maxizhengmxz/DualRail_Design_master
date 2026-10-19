"""Resonance predictions and narrow second-parameter searches at fixed EJ1."""
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from threadpoolctl import threadpool_limits

if __package__:
    from .model import ReadoutModel, _transmon
else:
    # Marimo opens readout.py as a standalone notebook in this directory.
    from model import ReadoutModel, _transmon


PARAMETERS = {
    "EJ2_GHz": "EJ2/h (GHz)",
    "C1_shunt_fF": "T1 shunt capacitance (fF)",
    "C2_shunt_fF": "T2 shunt capacitance (fF)",
    "C1r_fF": "T1–resonator capacitance (fF)",
    "C2r_fF": "T2–resonator capacitance (fF)",
}


def trial_model(base, parameter, value, fixed_ej2_ghz):
    if parameter not in PARAMETERS:
        raise ValueError(f"Choose sweep_parameter from {tuple(PARAMETERS)}")
    if parameter == "EJ2_GHz":
        return base, float(value)
    return ReadoutModel(replace(base.device, **{parameter: float(value)}), base.basis), fixed_ej2_ghz


def predict_resonances(base, ej1, parameter, bounds, fixed_ej2_ghz=20.0, samples=65):
    """Return all bracketed roots of local f1-f2, or an explicit diagnostic.

    Prediction uses only local transmon diagonalizations, not the full circuit.
    Roots tangent to zero without a sign change may require a finer prediction grid.
    """
    if parameter not in PARAMETERS:
        raise ValueError(f"Unsupported sweep parameter: {parameter}")
    lo, hi = map(float, bounds)
    if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
        raise ValueError("Prediction bounds must be positive and increasing")
    if not isinstance(samples, int) or samples < 3:
        raise ValueError("Use at least three prediction samples")
    if not np.isfinite([ej1, fixed_ej2_ghz]).all() or min(ej1, fixed_ej2_ghz) <= 0:
        raise ValueError("Josephson energies must be positive")

    def detuning(value):
        model, ej2 = trial_model(base, parameter, value, fixed_ej2_ghz)
        e1, _ = _transmon(ej1, model.ec_ghz[0], model.basis)
        e2, _ = _transmon(ej2, model.ec_ghz[1], model.basis)
        return e1[1]-e2[1]

    grid = np.linspace(lo, hi, samples)
    differences = np.array([detuning(v) for v in grid])
    if np.max(np.abs(differences)) < 1e-9:
        return [], "nonunique resonance: loaded frequencies match throughout bounds"
    roots = []
    for i, value in enumerate(grid):
        if abs(differences[i]) < 1e-10:
            roots.append(float(value))
        if i and differences[i-1]*differences[i] < 0:
            roots.append(brentq(detuning, grid[i-1], value, xtol=1e-8))
    unique = []
    for root in sorted(roots):
        if not unique or abs(root-unique[-1]) > 1e-6:
            unique.append(root)
    return unique, "predicted" if unique else "no bracketed resonance within bounds"


def guided_search(base, ej1_values, parameter, bounds, *, fixed_ej2_ghz=20.0,
                  mode="fractional", fractional_half_width=0.15, samples=15,
                  absolute_half_width=2.0, step=0.25, min_hybridization=0.95,
                  xtol=1e-4, prediction_samples=65):
    """Predict local resonance; refine full-circuit splitting versus parameter.

    Only evaluate neighborhoods of predicted roots. Diagnostics preserve missing
    roots, clipped ranges, edge minima and rejected hybridization for each slice.
    """
    ej1_values = np.asarray(ej1_values, dtype=float)
    if ej1_values.ndim != 1 or not len(ej1_values) or not np.isfinite(ej1_values).all():
        raise ValueError("Supply a finite one-dimensional EJ1 grid")
    if np.any(ej1_values <= 0) or np.any(np.diff(ej1_values) <= 0):
        raise ValueError("EJ1 grid must be positive and strictly increasing")
    if parameter not in PARAMETERS or mode not in ("fractional", "absolute"):
        raise ValueError("Unsupported parameter or neighborhood mode")
    if not 0 <= min_hybridization <= 1 or not np.isfinite(xtol) or xtol <= 0:
        raise ValueError("Invalid threshold or minimizer tolerance")
    if mode == "fractional":
        if not np.isfinite(fractional_half_width) or fractional_half_width <= 0 or not isinstance(samples, int) or samples < 3:
            raise ValueError("Fractional mode needs positive width and at least 3 samples")
    elif not np.isfinite([absolute_half_width, step]).all() or min(absolute_half_width, step) <= 0:
        raise ValueError("Absolute mode needs positive finite half-width and step")
    points, scan, diagnostics = [], [], []
    with threadpool_limits(limits=1):
        for slice_index, ej1 in enumerate(ej1_values):
            roots, status = predict_resonances(base, ej1, parameter, bounds, fixed_ej2_ghz, prediction_samples)
            if not roots:
                diagnostics.append({"EJ1_GHz": float(ej1), "prediction": None, "status": status})
                continue
            cache, accepted = {}, []
            def evaluate(value):
                value = float(value)
                if value not in cache:
                    model, ej2 = trial_model(base, parameter, value, fixed_ej2_ghz)
                    cache[value] = {**model.evaluate(ej1, ej2), "sweep_value": value,
                                    "slice_index": slice_index}
                return cache[value]
            for root_index, root in enumerate(roots):
                width = root*fractional_half_width if mode == "fractional" else absolute_half_width
                left, right = max(bounds[0], root-width), min(bounds[1], root+width)
                if mode == "fractional":
                    grid = np.linspace(left, right, samples)
                else:
                    # Uniform requested step anchored on the prediction, plus
                    # the clipped endpoints. Never silently replace the step.
                    offsets = np.arange(np.ceil((left-root)/step), np.floor((right-root)/step)+1)*step
                    grid = np.unique(np.r_[left, root+offsets, right])
                diagnostic = {"EJ1_GHz": float(ej1), "prediction": root,
                              "search_lower": left, "search_upper": right,
                              "clipped": left > root-width or right < root+width,
                              "status": "", "accepted": 0}
                if len(grid) < 3:
                    diagnostic["status"] = "fewer than 3 samples: reduce step"
                    diagnostics.append(diagnostic)
                    continue
                coarse = [{**evaluate(v), "prediction": root, "root_index": root_index} for v in grid]
                scan.extend(coarse)
                splitting = np.array([p["splitting_ghz"] for p in coarse])
                minima = [i for i in range(1, len(grid)-1)
                          if splitting[i] <= splitting[i-1] and splitting[i] <= splitting[i+1]
                          and (splitting[i] < splitting[i-1] or splitting[i] < splitting[i+1])]
                for i in minima:
                    result = minimize_scalar(lambda v: evaluate(v)["splitting_ghz"],
                                             bounds=(grid[i-1], grid[i+1]), method="bounded",
                                             options={"xatol": xtol})
                    if not result.success:
                        raise RuntimeError(f"Refinement failed at EJ1={ej1}")
                    p = evaluate(result.x)
                    if p["hybridization"] >= min_hybridization and all(abs(result.x-v) > 10*xtol for v in accepted):
                        points.append({**p, "prediction": root, "root_index": root_index})
                        accepted.append(result.x)
                        diagnostic["accepted"] += 1
                diagnostic["status"] = (
                    "accepted" if diagnostic["accepted"] else
                    "minima rejected by hybridization threshold" if minima else
                    "minimum at neighborhood edge: widen search" if np.argmin(splitting) in (0, len(grid)-1) else
                    "no interior splitting minimum"
                )
                diagnostics.append(diagnostic)
    return points, scan, diagnostics
