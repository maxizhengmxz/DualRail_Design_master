"""Shared constants for DTC dual-rail analyses."""

from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
FLUX_PARAM = "Φ1"
BRANCH_LABELS = np.array(["g", "p", "m", "r", "pp", "pm", "mm", "pr", "mr"])
BRANCH_INDEX = {label: idx for idx, label in enumerate(BRANCH_LABELS)}
TRACKING_SCHEMA = "two-excitation-no-rr-hybridization-fallback-v3"
DEFAULT_PARAM_REL_SIGMA = {
    "Ej1": 0.01,
    "Cj1": 0.01,
    "C1": 0.01,
    "Ej2": 0.01,
    "Cj2": 0.01,
    "C2": 0.01,
    "Ej3": 0.01,
    "Cj3": 0.01,
    "C12": 0.01,
    "C1r": 0.01,
    "C2r": 0.01,
    "Cr": 0.01,
    "Lr": 0.01,
}
DEFAULT_PARAM_SIGMA = {}

__all__ = [
    "BRANCH_LABELS",
    "BRANCH_INDEX",
    "DATA_DIR",
    "DEFAULT_PARAM_REL_SIGMA",
    "DEFAULT_PARAM_SIGMA",
    "FLUX_PARAM",
    "TRACKING_SCHEMA",
]
