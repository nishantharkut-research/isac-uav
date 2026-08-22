"""Objective, Pareto-front, and paired-statistics contracts for BA-PCDTL."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

import numpy as np
from scipy import stats


OBJECTIVE_DIM = 2
PNG_PREFERENCE = np.array([0.5, 0.5], dtype=np.float64)


def discounted_return_scale(gamma: float, horizon: int) -> float:
    """Map a stationary per-step rate to its finite-horizon discounted return."""
    gamma = float(gamma)
    horizon = int(horizon)
    if not 0.0 <= gamma < 1.0 or horizon <= 0:
        raise ValueError("Discount scale requires gamma in [0, 1) and horizon > 0.")
    if gamma == 0.0:
        return 1.0
    return float((1.0 - gamma ** horizon) / (1.0 - gamma))


def _objective_array(values, name: str = "values") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != OBJECTIVE_DIM:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}.")
    if array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite.")
    return array


def validate_preferences(preferences) -> np.ndarray:
    """Return a validated ``(N, 2)`` simplex array."""
    array = _objective_array(preferences, "preferences")
    if np.any(array < 0.0):
        raise ValueError("Preferences must be nonnegative.")
    if not np.allclose(array.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Each preference must sum to one.")
    return array.copy()


class ParetoAnchors:
    """Training-derived ideal/nadir anchors that become immutable when frozen."""

    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self.ideal: np.ndarray | None = None
        self.nadir: np.ndarray | None = None
        self.frozen = False
        self.provenance: dict = {}

    def update_training(self, objective_values, lower_quantile=0.0,
                        upper_quantile=1.0) -> None:
        if self.frozen:
            raise RuntimeError("Pareto anchors are frozen and cannot be updated.")
        values = _objective_array(objective_values, "training objective values")
        lower_quantile = float(lower_quantile)
        upper_quantile = float(upper_quantile)
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError(
                "Pareto anchor quantiles must satisfy 0 <= low < high <= 1."
            )
        batch_ideal = np.quantile(values, upper_quantile, axis=0)
        batch_nadir = np.quantile(values, lower_quantile, axis=0)
        self.ideal = (
            batch_ideal.copy()
            if self.ideal is None
            else np.maximum(self.ideal, batch_ideal)
        )
        self.nadir = (
            batch_nadir.copy()
            if self.nadir is None
            else np.minimum(self.nadir, batch_nadir)
        )

    def freeze(self, provenance: Mapping | None = None) -> None:
        if self.ideal is None or self.nadir is None:
            raise ValueError("Cannot freeze anchors before observing training values.")
        span = self.ideal - self.nadir
        if not np.all(np.isfinite(span)) or np.any(span <= 0.0):
            raise ValueError("Every Pareto anchor span must be finite and positive.")
        self.provenance = dict(provenance or {})
        self.frozen = True

    def normalize(self, objective_values) -> np.ndarray:
        if not self.frozen:
            raise RuntimeError("Pareto anchors must be frozen before normalization.")
        values = _objective_array(objective_values, "objective values")
        return (values - self.nadir) / (self.ideal - self.nadir)

    def state_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "ideal": None if self.ideal is None else self.ideal.tolist(),
            "nadir": None if self.nadir is None else self.nadir.tolist(),
            "frozen": bool(self.frozen),
            "provenance": dict(self.provenance),
        }

    def digest(self) -> str:
        """Return a canonical identity for checkpoint and run provenance."""
        encoded = json.dumps(
            self.state_dict(), sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "ParetoAnchors":
        if int(state.get("schema_version", -1)) != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported Pareto-anchor checkpoint schema.")
        result = cls()
        ideal = state.get("ideal")
        nadir = state.get("nadir")
        if (ideal is None) != (nadir is None):
            raise ValueError("Ideal and nadir anchors must be present together.")
        if ideal is not None:
            result.ideal = _objective_array([ideal], "ideal anchor")[0]
            result.nadir = _objective_array([nadir], "nadir anchor")[0]
            if np.any(result.ideal < result.nadir):
                raise ValueError("Ideal anchor cannot be below the nadir anchor.")
        result.provenance = dict(state.get("provenance", {}))
        result.frozen = bool(state.get("frozen", False))
        if result.frozen:
            span = result.ideal - result.nadir if result.ideal is not None else None
            if span is None or np.any(span <= 0.0):
                raise ValueError("Frozen Pareto anchors require positive spans.")
        return result


def normalized_physical_zero_reference(
    anchors: ParetoAnchors, *, margin: float = 0.05
) -> np.ndarray:
    """Return a frozen reference strictly below every nonnegative rate pair."""
    margin = float(margin)
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("Hypervolume reference margin must be positive and finite.")
    zero = anchors.normalize([[0.0, 0.0]])[0]
    return zero - margin


def scalarize_returns(
    q_values,
    preference,
    *,
    mode: str,
    anchors: ParetoAnchors | None = None,
    augmentation: float = 0.05,
) -> np.ndarray:
    """Scalarize two maximization objectives without changing PNG units."""
    values = _objective_array(q_values, "q_values")
    weights = validate_preferences([preference])[0]
    if mode == "png_joint_sum":
        if not np.allclose(weights, PNG_PREFERENCE, rtol=0.0, atol=1e-8):
            raise ValueError("PNG joint-sum scalarization requires balanced preference.")
        return values.sum(axis=1)
    if mode != "chebyshev":
        raise ValueError(f"Unknown scalarization mode: {mode!r}.")
    if anchors is None:
        raise ValueError("Chebyshev scalarization requires frozen Pareto anchors.")
    if not np.isfinite(augmentation) or augmentation < 0.0:
        raise ValueError("Chebyshev augmentation must be finite and nonnegative.")

    normalized = anchors.normalize(values)
    active = weights > 0.0
    ratios = normalized[:, active] / weights[active]
    achievement = ratios.min(axis=1)
    return achievement + float(augmentation) * (normalized * weights).sum(axis=1)


def nondominated_mask(points) -> np.ndarray:
    """Return maximization nondominance for each finite two-objective point."""
    values = _objective_array(points, "points")
    mask = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        mask[index] = not np.any(dominates)
    return mask


def hypervolume_2d(points, reference: Iterable[float]) -> float:
    """Exact dominated hypervolume for finite 2-D maximization points."""
    values = _objective_array(points, "points")
    reference_array = np.asarray(reference, dtype=np.float64)
    if reference_array.shape != (2,) or not np.all(np.isfinite(reference_array)):
        raise ValueError("Hypervolume reference must be a finite two-vector.")
    if np.any(values < reference_array):
        raise ValueError("Hypervolume points cannot be below the reference point.")

    front = values[nondominated_mask(values)]
    order = np.argsort(front[:, 0], kind="stable")
    front = front[order]
    area = 0.0
    previous_x = float(reference_array[0])
    for x_value, y_value in front:
        width = float(x_value - previous_x)
        if width < 0.0:
            raise ValueError("Invalid Pareto-front ordering.")
        area += width * float(y_value - reference_array[1])
        previous_x = max(previous_x, float(x_value))
    return float(area)


@dataclass(frozen=True)
class PairedInterval:
    n: int
    mean_difference: float
    lower: float
    upper: float
    standard_deviation: float
    standard_error: float
    p_value: float
    cohen_dz: float


def paired_confidence_interval(
    candidate, baseline, *, confidence: float = 0.95
) -> PairedInterval:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if candidate_array.ndim != 1 or baseline_array.ndim != 1:
        raise ValueError("Paired observations must be one-dimensional.")
    if candidate_array.shape != baseline_array.shape:
        raise ValueError("Candidate and baseline observations must be paired.")
    if len(candidate_array) < 2:
        raise ValueError("At least two paired observations are required.")
    if not np.all(np.isfinite(candidate_array)) or not np.all(np.isfinite(baseline_array)):
        raise ValueError("Paired observations must be finite.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")

    differences = candidate_array - baseline_array
    count = len(differences)
    mean = float(np.mean(differences))
    standard_deviation = float(np.std(differences, ddof=1))
    standard_error = standard_deviation / np.sqrt(count)
    if standard_deviation == 0.0:
        lower = upper = mean
        p_value = 1.0 if mean == 0.0 else 0.0
        cohen_dz = 0.0 if mean == 0.0 else float(np.copysign(np.inf, mean))
    else:
        critical = float(stats.t.ppf((1.0 + confidence) / 2.0, count - 1))
        lower = mean - critical * standard_error
        upper = mean + critical * standard_error
        p_value = float(stats.ttest_1samp(differences, popmean=0.0).pvalue)
        cohen_dz = mean / standard_deviation
    return PairedInterval(
        n=count,
        mean_difference=mean,
        lower=float(lower),
        upper=float(upper),
        standard_deviation=standard_deviation,
        standard_error=float(standard_error),
        p_value=p_value,
        cohen_dz=float(cohen_dz),
    )


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down family-wise-error adjustment in original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and lie in [0, 1].")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.empty_like(sorted_values)
    running = 0.0
    total = len(values)
    for index, value in enumerate(sorted_values):
        running = max(running, (total - index) * float(value))
        adjusted_sorted[index] = min(1.0, running)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted
