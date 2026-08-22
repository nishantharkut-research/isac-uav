"""Objective-aware stream-power allocation with explicit feasibility."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

from research_objectives import ObjectiveSpec, evaluate_objective


@dataclass(frozen=True)
class AllocationConstraints:
    total_power: float
    target_rate_min: float = 0.0
    user_rate_min: float = 0.0
    sensing_only_sinr_min: float = 0.0
    active_power_max: float = np.inf

    def __post_init__(self):
        for name in (
            "total_power", "target_rate_min", "user_rate_min",
            "sensing_only_sinr_min", "active_power_max",
        ):
            value = float(getattr(self, name))
            if np.isnan(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.total_power) or self.total_power <= 0.0:
            raise ValueError("total_power must be finite and positive.")


@dataclass(frozen=True)
class AllocationResult:
    powers: np.ndarray
    rate_u: np.ndarray
    rate_k: np.ndarray
    sensing_only_sinr: np.ndarray
    active_power: float
    objective_value: float
    feasible: bool
    total_normalized_deficit: float
    method: str
    solver_success: bool


def _as_vector(name, values, expected_size):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (expected_size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {expected_size} finite values.")
    return vector


def _assess(powers, user_count, target_count, evaluate, constraints, objective,
            method, solver_success=True):
    powers = _as_vector("powers", powers, user_count + target_count)
    metrics = evaluate(powers)
    rate_u = _as_vector("rate_u", metrics["rate_u"], user_count)
    rate_k = _as_vector("rate_k", metrics["rate_k"], target_count)
    sensing_sinr = _as_vector(
        "sensing_only_sinr", metrics["sensing_only_sinr"], target_count
    )
    active_power = float(metrics["active_power"])
    if not np.isfinite(active_power) or active_power < 0.0:
        raise ValueError("active_power must be finite and nonnegative.")

    objective_evaluation = evaluate_objective(objective, rate_u, rate_k)
    power_deficit = max(0.0, float(np.sum(powers)) - constraints.total_power)
    negative_deficit = float(np.maximum(-powers, 0.0).sum())
    user_deficit = float(
        np.maximum(constraints.user_rate_min - rate_u, 0.0).sum()
    )
    target_deficit = float(
        np.maximum(constraints.target_rate_min - rate_k, 0.0).sum()
    )
    sensing_deficit = float(
        np.maximum(constraints.sensing_only_sinr_min - sensing_sinr, 0.0).sum()
    )
    active_deficit = max(0.0, active_power - constraints.active_power_max)
    total_deficit = (
        power_deficit / constraints.total_power
        + negative_deficit / constraints.total_power
        + user_deficit / max(constraints.user_rate_min * user_count, 1.0)
        + target_deficit / max(constraints.target_rate_min * target_count, 1.0)
        + sensing_deficit
          / max(constraints.sensing_only_sinr_min * target_count, 1.0)
        + active_deficit / max(constraints.active_power_max, 1.0)
        + objective_evaluation.communication_deficit
          / max(objective_evaluation.communication_required, 1.0)
    )
    feasible = bool(
        total_deficit <= 1e-8
        and objective_evaluation.rate_constraints_feasible
    )
    return AllocationResult(
        powers=powers.copy(),
        rate_u=rate_u,
        rate_k=rate_k,
        sensing_only_sinr=sensing_sinr,
        active_power=active_power,
        objective_value=float(objective_evaluation.value),
        feasible=feasible,
        total_normalized_deficit=float(total_deficit),
        method=method,
        solver_success=bool(solver_success),
    )


def _is_better(candidate, incumbent):
    if incumbent is None:
        return True
    if candidate.feasible != incumbent.feasible:
        return candidate.feasible
    if candidate.feasible:
        return candidate.objective_value > incumbent.objective_value + 1e-12
    return (
        candidate.total_normalized_deficit,
        -candidate.objective_value,
    ) < (
        incumbent.total_normalized_deficit,
        -incumbent.objective_value,
    )


def _group_power_vector(user_count, target_count, total_power, comm_fraction):
    communication = total_power * float(comm_fraction)
    sensing = total_power - communication
    return np.concatenate([
        np.full(user_count, communication / user_count),
        np.full(target_count, sensing / target_count),
    ])


def _required_sinr(rate_floor):
    return max(float(np.exp2(float(rate_floor)) - 1.0), 0.0)


def _rebalance_within_groups(result, user_count, constraints):
    """Redistribute fixed group budgets toward streams below hard floors."""
    powers = result.powers.copy()
    target_count = powers.size - user_count

    def rebalance(group, observed_sinr, required_sinr):
        budget = float(np.sum(group))
        if budget <= 0.0 or required_sinr <= 0.0:
            return group
        ratios = required_sinr / np.maximum(observed_sinr, 1e-12)
        weights = group * np.clip(ratios, 0.02, 50.0)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            return group
        return budget * weights / total_weight

    user_sinr = np.maximum(np.exp2(result.rate_u) - 1.0, 0.0)
    powers[:user_count] = rebalance(
        powers[:user_count], user_sinr,
        _required_sinr(constraints.user_rate_min),
    )

    target_sinr = np.maximum(np.exp2(result.rate_k) - 1.0, 0.0)
    required_target = np.full(
        target_count, _required_sinr(constraints.target_rate_min), dtype=float
    )
    if constraints.sensing_only_sinr_min > 0.0:
        sensing_ratio = (
            constraints.sensing_only_sinr_min
            / np.maximum(result.sensing_only_sinr, 1e-12)
        )
        rate_ratio = required_target / np.maximum(target_sinr, 1e-12)
        combined_ratio = np.maximum(rate_ratio, sensing_ratio)
        budget = float(np.sum(powers[user_count:]))
        weights = powers[user_count:] * np.clip(combined_ratio, 0.02, 50.0)
        if budget > 0.0 and float(np.sum(weights)) > 0.0:
            powers[user_count:] = budget * weights / float(np.sum(weights))
    else:
        powers[user_count:] = rebalance(
            powers[user_count:], target_sinr,
            _required_sinr(constraints.target_rate_min),
        )
    return powers


def select_fast_power_allocation(
    user_count: int,
    target_count: int,
    evaluate: Callable[[np.ndarray], dict],
    constraints: AllocationConstraints,
    objective_spec: ObjectiveSpec,
    previous_powers: Optional[np.ndarray] = None,
) -> AllocationResult:
    """Rank a deterministic bounded candidate set for use inside training."""
    user_count, target_count = int(user_count), int(target_count)
    if user_count <= 0 or target_count <= 0:
        raise ValueError("At least one user and one target are required.")
    previous = None
    if previous_powers is not None:
        previous = _as_vector(
            "previous_powers", previous_powers, user_count + target_count
        )
        if np.any(previous < 0.0) or np.sum(previous) > constraints.total_power + 1e-9:
            previous = None
    fractions = np.concatenate([
        np.array([0.01, 0.025, 0.05, 0.95, 0.975, 0.99]),
        np.linspace(0.1, 0.9, 17),
    ])
    if previous is not None:
        previous_fraction = float(
            np.sum(previous[:user_count]) / constraints.total_power
        )
        local_fractions = np.clip(
            previous_fraction
            + np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]),
            0.01,
            0.99,
        )
        fractions = np.concatenate([fractions, local_fractions])
    # A previous allocation is a warm start only. Restricting the grid around
    # it can falsely report infeasibility after the channel changes.
    fractions = np.unique(fractions)
    candidates = [
        _group_power_vector(
            user_count, target_count, constraints.total_power, fraction
        )
        for fraction in fractions
    ]
    if previous is not None:
        candidates.append(previous)

    best = None
    for powers in candidates:
        result = _assess(
            powers, user_count, target_count, evaluate, constraints,
            objective_spec, method="fast_candidates",
        )
        if _is_better(result, best):
            best = result

    # Equal group splits can falsely declare an unequal-channel scene
    # infeasible.  A few deterministic, fixed-budget refinements retain the
    # fast training path while allowing individual hard floors to be met.
    for _ in range(4):
        powers = _rebalance_within_groups(best, user_count, constraints)
        result = _assess(
            powers, user_count, target_count, evaluate, constraints,
            objective_spec, method="fast_floor_rebalance",
        )
        if np.allclose(result.powers, best.powers, rtol=1e-10, atol=1e-12):
            break
        if _is_better(result, best):
            best = result
        else:
            break
    return best


def optimize_power_allocation(
    user_count: int,
    target_count: int,
    evaluate: Callable[[np.ndarray], dict],
    constraints: AllocationConstraints,
    objective_spec: ObjectiveSpec,
    initial_powers: Optional[np.ndarray] = None,
    max_iterations: int = 100,
) -> AllocationResult:
    """Refine stream powers with SciPy SLSQP for evaluation/calibration."""
    fast = select_fast_power_allocation(
        user_count, target_count, evaluate, constraints, objective_spec,
        previous_powers=initial_powers,
    )
    x0 = fast.powers if initial_powers is None else _as_vector(
        "initial_powers", initial_powers, user_count + target_count
    )
    x0 = np.maximum(x0, 0.0)
    if np.sum(x0) > constraints.total_power:
        x0 *= constraints.total_power / np.sum(x0)

    cached_x = None
    cached_assessment = None

    def assessed(x):
        nonlocal cached_x, cached_assessment
        candidate = np.asarray(x, dtype=float)
        if cached_x is not None and np.array_equal(candidate, cached_x):
            return cached_assessment
        cached_x = candidate.copy()
        cached_assessment = _assess(
            x, user_count, target_count, evaluate, constraints,
            objective_spec, method="scipy_slsqp",
        )
        return cached_assessment

    def objective_function(x):
        return -assessed(x).objective_value

    def constraint_values(x):
        result = assessed(x)
        values = [constraints.total_power - float(np.sum(x))]
        values.extend(result.rate_u - constraints.user_rate_min)
        values.extend(result.rate_k - constraints.target_rate_min)
        values.extend(result.sensing_only_sinr - constraints.sensing_only_sinr_min)
        values.append(constraints.active_power_max - result.active_power)
        evaluation = evaluate_objective(
            objective_spec, result.rate_u, result.rate_k
        )
        if evaluation.communication_required > 0.0:
            values.append(
                evaluation.communication_sum_rate
                - evaluation.communication_required
            )
        return np.asarray(values, dtype=float)

    solution = minimize(
        objective_function,
        x0,
        method="SLSQP",
        bounds=[(0.0, constraints.total_power)] * (user_count + target_count),
        constraints=[{"type": "ineq", "fun": constraint_values}],
        options={"maxiter": int(max_iterations), "ftol": 1e-9, "disp": False},
    )
    refined = assessed(solution.x)
    refined = AllocationResult(
        **{**refined.__dict__, "solver_success": bool(solution.success)}
    )
    return refined if _is_better(refined, fast) else fast
