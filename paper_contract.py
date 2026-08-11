"""Shared paper objective and constraint helpers.

The channel equations stay in ``channel_model.py`` and ``sinr_rate.py``. This
module only centralizes the optimization contract from Eq. (16) so the
DRL environment and AO baseline cannot silently optimize different objectives.
"""
import numpy as np


FIGURE_6_SYSTEM_CONFIGURATION = {
    "M": 4,
    "N": 2,
    "U": 3,
    "K": 3,
    "L": 20,
    "La": 10,
    "Lp": 10,
}

CALIBRATION_SOLVER_SCOPE = "scipy_ao_surrogate_not_author_algorithm3"


def communication_sum_rate(rate_u) -> float:
    """Return the communication objective in paper Eq. (16)."""
    return float(np.sum(rate_u))


def joint_sum_rate(rate_u, rate_k) -> float:
    """Return the paper objective value from Eq. (16): sum_u R_u.

    The sensing target rates ``rate_k`` are intentionally not added to the
    objective; they enter through the sensing-rate floor constraint.
    The historical function name is kept to avoid breaking older scripts.
    """
    return communication_sum_rate(rate_u)


def sensing_rate_floor_penalty(rate_k, r_k_min: float) -> float:
    """Return the violation of the published sensing floor, Eq. (16e)."""
    rate_k = np.asarray(rate_k, dtype=float)
    return float(np.sum(np.maximum(0.0, r_k_min - rate_k)))


def rate_floor_penalty(rate_u, rate_k, r_u_min=None, r_k_min: float = 0.0):
    """Compatibility wrapper with no communication-rate floor.

    The separately supplied extension PNG adds a UE-rate constraint, but the
    published Eq. (16) does not. ``r_u_min`` is therefore ignored deliberately.
    """
    sensing_penalty = sensing_rate_floor_penalty(rate_k, r_k_min)
    return sensing_penalty, {
        "comm_penalty": 0.0,
        "sensing_penalty": sensing_penalty,
    }


def active_power_penalty(p_ai, p_a_max: float):
    """Penalty for aggregate active-HRIS power, Eqs. (18) and (16d)."""
    power = float(p_ai)
    penalty = max(0.0, power - float(p_a_max))
    return penalty, bool(penalty == 0.0)


def normalized_constraint_violation(excess: float, limit: float) -> float:
    """Scale a nonnegative physical excess by its constraint magnitude.

    This makes power-domain violations dimensionless before a diagnostic or
    numerical surrogate combines them with a rate-domain objective.
    """
    excess = max(0.0, float(excess))
    scale = max(abs(float(limit)), np.finfo(float).tiny)
    return excess / scale


def cascade_power_16c(ch, theta) -> float:
    """Worst-user coherent passive-cascade power in paper Eq. (16c)."""
    theta = np.asarray(theta, dtype=complex)
    return max(
        float(np.linalg.norm(sum(
            (
                np.outer(
                    ch.H_ru_hat[u][:, j] * theta[j],
                    ch.G_brj_hat[j, :],
                )
                for j in range(theta.size)
            ),
            start=np.zeros(
                (ch.H_ru_hat[u].shape[0], ch.G_brj_hat.shape[1]),
                dtype=complex,
            ),
        ), "fro") ** 2)
        for u in range(len(ch.H_ru_hat))
    )


def cascade_penalty_16c(ch, theta, psi_max: float):
    """Return (penalty, ok, power) for passive cascade constraint (16c)."""
    power = cascade_power_16c(ch, theta)
    penalty = max(0.0, power - psi_max)
    return penalty, bool(penalty == 0.0), power


def _limit_norm(vec, max_norm: float):
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm <= 1e-12:
        return vec
    return vec * (max_norm / norm)


def project_uav_step(
    current_xy,
    proposed_delta_xy,
    start_xy,
    slot_index: int,
    n_slots: int,
    d_max: float,
    bounds,
):
    """Project one UAV step onto movement, area, and closure feasibility.

    The environment calls this at every step. If all previous steps also used
    this projection, the final step can satisfy both ``omega[1] = omega[N]`` and
    ``||omega[n+1] - omega[n]|| <= D``.
    """
    current_xy = np.asarray(current_xy, dtype=float)
    proposed_delta_xy = np.asarray(proposed_delta_xy, dtype=float)
    start_xy = np.asarray(start_xy, dtype=float)
    low, high = (np.asarray(bounds[0], dtype=float),
                 np.asarray(bounds[1], dtype=float))

    if n_slots <= 1 or slot_index >= n_slots - 1:
        return np.clip(start_xy.copy(), low, high)

    delta = _limit_norm(proposed_delta_xy, d_max)
    candidate = np.clip(current_xy + delta, low, high)

    remaining_steps = max(0, n_slots - slot_index - 1)
    max_distance_from_start = remaining_steps * d_max
    if np.linalg.norm(candidate - start_xy) <= max_distance_from_start + 1e-12:
        return candidate

    direction_to_start = start_xy - current_xy
    current_distance = float(np.linalg.norm(direction_to_start))
    if current_distance <= 1e-12:
        return np.clip(start_xy.copy(), low, high)

    required_reduction = max(0.0, current_distance - max_distance_from_start)
    step_len = min(d_max, required_reduction)
    candidate = current_xy + (step_len / current_distance) * direction_to_start
    return np.clip(candidate, low, high)
