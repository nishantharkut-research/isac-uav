"""Pure lower-confidence selector for baseline-anchored residual actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateDecision:
    scale: float
    selected_index: int
    baseline_value: float
    selected_lcb: float
    uncertainty: float
    fallback_reason: str | None


def select_residual_scale(
    scales,
    ensemble_values,
    uncertainty_coefficient: float,
    margin: float,
) -> GateDecision:
    """Select a residual scale only after a strict conservative improvement."""
    scale_array = np.asarray(scales, dtype=np.float64)
    if scale_array.ndim != 1 or len(scale_array) < 2:
        raise ValueError("scales must be a one-dimensional candidate grid.")
    if not np.all(np.isfinite(scale_array)) or np.any(
        (scale_array < 0.0) | (scale_array > 1.0)
    ):
        raise ValueError("scales must be finite and lie in [0, 1].")
    if len(np.unique(scale_array)) != len(scale_array):
        raise ValueError("scales must be unique.")
    zero_indices = np.flatnonzero(np.isclose(scale_array, 0.0, atol=0.0, rtol=0.0))
    if len(zero_indices) != 1:
        raise ValueError("The candidate grid must contain exactly one zero scale.")
    baseline_index = int(zero_indices[0])

    values = np.asarray(ensemble_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != len(scale_array):
        raise ValueError("ensemble_values must have shape (ensemble, scales).")
    coefficient = float(uncertainty_coefficient)
    gate_margin = float(margin)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("uncertainty_coefficient must be finite and nonnegative.")
    if not np.isfinite(gate_margin) or gate_margin < 0.0:
        raise ValueError("margin must be finite and nonnegative.")

    if not np.all(np.isfinite(values)):
        baseline_column = values[:, baseline_index]
        baseline_value = (
            float(np.mean(baseline_column))
            if np.all(np.isfinite(baseline_column))
            else float("nan")
        )
        return GateDecision(
            scale=0.0,
            selected_index=baseline_index,
            baseline_value=baseline_value,
            selected_lcb=baseline_value,
            uncertainty=float("inf"),
            fallback_reason="nonfinite_ensemble_values",
        )

    means = values.mean(axis=0)
    uncertainties = values.std(axis=0, ddof=0)
    lower_bounds = means - coefficient * uncertainties
    baseline_value = float(means[baseline_index])

    candidate_indices = np.flatnonzero(scale_array > 0.0)
    best_candidate = int(
        candidate_indices[np.argmax(lower_bounds[candidate_indices])]
    )
    selected_lcb = float(lower_bounds[best_candidate])
    if selected_lcb <= baseline_value + gate_margin:
        return GateDecision(
            scale=0.0,
            selected_index=baseline_index,
            baseline_value=baseline_value,
            selected_lcb=selected_lcb,
            uncertainty=float(uncertainties[best_candidate]),
            fallback_reason="insufficient_conservative_advantage",
        )
    return GateDecision(
        scale=float(scale_array[best_candidate]),
        selected_index=best_candidate,
        baseline_value=baseline_value,
        selected_lcb=selected_lcb,
        uncertainty=float(uncertainties[best_candidate]),
        fallback_reason=None,
    )
