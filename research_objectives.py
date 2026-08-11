"""Explicit optimization contracts for baseline and sensing-extension runs."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np


class ObjectiveMode(str, Enum):
    PAPER_BASELINE = "paper_baseline"
    PNG_JOINT_SUM = "png_joint_sum"
    COMM_RETENTION_SENSING = "comm_retention_sensing"
    NORMALIZED_PARETO = "normalized_pareto"


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive(name: str, value: Optional[float]) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class ObjectiveSpec:
    mode: ObjectiveMode
    target_rate_min: float = 0.0
    user_rate_min: float = 0.0
    communication_reference: Optional[float] = None
    sensing_reference: Optional[float] = None
    communication_retention_fraction: float = 1.0
    pareto_weight: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "mode", ObjectiveMode(self.mode))
        object.__setattr__(
            self, "target_rate_min",
            _finite_nonnegative("target_rate_min", self.target_rate_min),
        )
        object.__setattr__(
            self, "user_rate_min",
            _finite_nonnegative("user_rate_min", self.user_rate_min),
        )
        retention = float(self.communication_retention_fraction)
        if not np.isfinite(retention) or not 0.0 < retention <= 1.0:
            raise ValueError("communication_retention must be in (0, 1]")
        object.__setattr__(self, "communication_retention_fraction", retention)
        weight = float(self.pareto_weight)
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("pareto_weight must be in [0, 1]")
        object.__setattr__(self, "pareto_weight", weight)

        if self.mode in {
            ObjectiveMode.COMM_RETENTION_SENSING,
            ObjectiveMode.NORMALIZED_PARETO,
        }:
            object.__setattr__(
                self, "communication_reference",
                _positive("communication_reference", self.communication_reference),
            )
        elif self.communication_reference is not None:
            object.__setattr__(
                self, "communication_reference",
                _positive("communication_reference", self.communication_reference),
            )

        if self.mode == ObjectiveMode.NORMALIZED_PARETO:
            object.__setattr__(
                self, "sensing_reference",
                _positive("sensing_reference", self.sensing_reference),
            )
        elif self.sensing_reference is not None:
            object.__setattr__(
                self, "sensing_reference",
                _positive("sensing_reference", self.sensing_reference),
            )

    @classmethod
    def paper(cls, target_rate_min: float = 0.0):
        return cls(
            mode=ObjectiveMode.PAPER_BASELINE,
            target_rate_min=target_rate_min,
        )

    @classmethod
    def png_joint(cls, user_rate_min: float = 0.0,
                  target_rate_min: float = 0.0):
        return cls(
            mode=ObjectiveMode.PNG_JOINT_SUM,
            user_rate_min=user_rate_min,
            target_rate_min=target_rate_min,
        )

    @classmethod
    def communication_retention(
        cls,
        communication_reference: float,
        communication_retention: float = 0.95,
        user_rate_min: float = 0.0,
        target_rate_min: float = 0.0,
    ):
        return cls(
            mode=ObjectiveMode.COMM_RETENTION_SENSING,
            communication_reference=communication_reference,
            communication_retention_fraction=communication_retention,
            user_rate_min=user_rate_min,
            target_rate_min=target_rate_min,
        )

    @classmethod
    def normalized_pareto(
        cls,
        communication_reference: float,
        sensing_reference: float,
        pareto_weight: float,
        user_rate_min: float = 0.0,
        target_rate_min: float = 0.0,
    ):
        return cls(
            mode=ObjectiveMode.NORMALIZED_PARETO,
            communication_reference=communication_reference,
            sensing_reference=sensing_reference,
            pareto_weight=pareto_weight,
            user_rate_min=user_rate_min,
            target_rate_min=target_rate_min,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "mode": self.mode.value,
            "target_rate_min": self.target_rate_min,
            "user_rate_min": self.user_rate_min,
        }
        if self.mode == ObjectiveMode.COMM_RETENTION_SENSING:
            payload.update({
                "communication_reference": self.communication_reference,
                "communication_retention_fraction":
                    self.communication_retention_fraction,
            })
        elif self.mode == ObjectiveMode.NORMALIZED_PARETO:
            payload.update({
                "communication_reference": self.communication_reference,
                "sensing_reference": self.sensing_reference,
                "pareto_weight": self.pareto_weight,
            })
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]):
        return cls(**dict(payload))


@dataclass(frozen=True)
class ObjectiveEvaluation:
    mode: ObjectiveMode
    value: float
    communication_sum_rate: float
    sensing_sum_rate: float
    communication_required: float
    communication_deficit: float
    user_rate_deficits: np.ndarray
    target_rate_deficits: np.ndarray
    rate_constraints_feasible: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "value": self.value,
            "communication_sum_rate": self.communication_sum_rate,
            "sensing_sum_rate": self.sensing_sum_rate,
            "communication_required": self.communication_required,
            "communication_deficit": self.communication_deficit,
            "user_rate_deficits": self.user_rate_deficits.tolist(),
            "target_rate_deficits": self.target_rate_deficits.tolist(),
            "rate_constraints_feasible": self.rate_constraints_feasible,
        }


def _rates(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} values must be finite and nonnegative")
    return array


def evaluate_objective(spec: ObjectiveSpec, rate_u, rate_k) -> ObjectiveEvaluation:
    """Evaluate a raw objective and its rate-only constraints."""
    if not isinstance(spec, ObjectiveSpec):
        spec = ObjectiveSpec.from_mapping(spec)
    rate_u = _rates("rate_u", rate_u)
    rate_k = _rates("rate_k", rate_k)
    communication = float(np.sum(rate_u))
    sensing = float(np.sum(rate_k))

    if spec.mode == ObjectiveMode.PAPER_BASELINE:
        value = communication
    elif spec.mode == ObjectiveMode.PNG_JOINT_SUM:
        value = communication + sensing
    elif spec.mode == ObjectiveMode.COMM_RETENTION_SENSING:
        value = sensing
    elif spec.mode == ObjectiveMode.NORMALIZED_PARETO:
        value = (
            (1.0 - spec.pareto_weight)
            * communication / float(spec.communication_reference)
            + spec.pareto_weight
            * sensing / float(spec.sensing_reference)
        )
    else:  # pragma: no cover - Enum construction prevents this branch.
        raise ValueError(f"unsupported objective mode: {spec.mode}")

    communication_required = 0.0
    if spec.mode == ObjectiveMode.COMM_RETENTION_SENSING:
        communication_required = (
            spec.communication_retention_fraction
            * float(spec.communication_reference)
        )
    communication_deficit = max(
        0.0, communication_required - communication
    )
    user_deficits = np.maximum(0.0, spec.user_rate_min - rate_u)
    target_deficits = np.maximum(0.0, spec.target_rate_min - rate_k)
    feasible = bool(
        communication_deficit <= 1e-12
        and np.all(user_deficits <= 1e-12)
        and np.all(target_deficits <= 1e-12)
    )
    return ObjectiveEvaluation(
        mode=spec.mode,
        value=float(value),
        communication_sum_rate=communication,
        sensing_sum_rate=sensing,
        communication_required=float(communication_required),
        communication_deficit=float(communication_deficit),
        user_rate_deficits=user_deficits,
        target_rate_deficits=target_deficits,
        rate_constraints_feasible=feasible,
    )
