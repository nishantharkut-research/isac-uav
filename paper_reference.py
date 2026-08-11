"""Strict loader for digitized paper observations kept separate from results."""

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def _load_payload(source):
    if isinstance(source, Mapping):
        return dict(source)
    with Path(source).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_paper_references(source) -> dict:
    payload = _load_payload(source)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported paper-reference schema.")
    figures = payload.get("figures")
    if not isinstance(figures, dict) or not figures:
        raise ValueError("Paper references require a non-empty figures mapping.")

    for key, figure in figures.items():
        if figure.get("kind") != "digitized_reference":
            raise ValueError(
                f"Figure {key} must be labeled kind='digitized_reference'."
            )
        if int(figure.get("source_page", 0)) <= 0:
            raise ValueError(f"Figure {key} requires a positive source_page.")
        for axis_name in ("x_axis", "y_axis"):
            axis = figure.get(axis_name, {})
            if not axis.get("label") or not axis.get("units"):
                raise ValueError(f"Figure {key} requires labeled axis units.")
        if not figure.get("digitization_method"):
            raise ValueError(f"Figure {key} requires a digitization method.")
        uncertainty = float(figure.get("uncertainty", {}).get("absolute_y", 0.0))
        if not np.isfinite(uncertainty) or uncertainty <= 0.0:
            raise ValueError(f"Figure {key} requires positive y uncertainty.")
        x = np.asarray(figure.get("x", []), dtype=float)
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError(f"Figure {key} has an invalid x array.")
        curves = figure.get("curves", {})
        if not curves:
            raise ValueError(f"Figure {key} requires at least one curve.")
        for label, values in curves.items():
            y = np.asarray(values, dtype=float)
            if y.shape != x.shape or not np.all(np.isfinite(y)):
                raise ValueError(f"Figure {key} curve {label!r} is invalid.")

    calibration = {int(value) for value in payload.get("calibration_figures", [])}
    holdout = {int(value) for value in payload.get("holdout_figures", [])}
    known = {int(value) for value in figures}
    if calibration & holdout:
        raise ValueError("Calibration and holdout figures must be disjoint.")
    if not calibration <= known or not holdout <= known:
        raise ValueError("Calibration/holdout lists reference unknown figures.")
    return payload


def paper_reference_digest(source) -> str:
    """Return the canonical digest of a schema-validated reference payload."""
    payload = load_paper_references(source)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
