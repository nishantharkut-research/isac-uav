"""
ao_optimizer.py — Alternating Optimisation (AO) baseline.

Section V of Saikia et al., IEEE Trans. Commun., 2025.

Numerical approximation of the three blocks in Algorithms 2 and 3. The paper
does not publish its CVX implementation or all auxiliary matrices, so this file
uses SciPy with the published objective and constraints rather than claiming to
be the unavailable author solver.

SCA convergence is declared when the fractional increase of the objective
falls below δ = cfg.ao_tol.
"""
from config import cfg
import numpy as np

from channel_model import ChannelSet
from paper_contract import (
    active_power_penalty,
    cascade_penalty_16c,
    joint_sum_rate,
)
from sinr_rate import (
    active_tx_power,
    compute_rates,
    select_feasible_beamformers,
)
from research_objectives import ObjectiveSpec


def _minimize(*args, **kwargs):
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required only for AO optimization. Install requirements.txt "
            "before running AO figure generation."
        ) from exc
    return minimize(*args, **kwargs)


def _score_beamformers(ch: ChannelSet,
                       theta: np.ndarray,
                       phi: np.ndarray,
                       C: np.ndarray,
                       T: np.ndarray,
                       sensing_rate_min: float = None) -> float:
    rate_u, rate_k, _, _ = compute_rates(ch, theta, phi, C, T)
    return joint_sum_rate(rate_u, rate_k)


def _constraint_diagnostics(ch: ChannelSet,
                            theta: np.ndarray,
                            phi: np.ndarray,
                            C: np.ndarray,
                            T: np.ndarray,
                            rate_k: np.ndarray,
                            sensing_rate_min: float = None) -> tuple:
    """Return literal Eq. (16) feasibility metadata for one AO candidate."""
    r_k_min = cfg.R_min if sensing_rate_min is None else float(sensing_rate_min)
    target_ok = bool(np.all(np.asarray(rate_k) >= r_k_min - 1e-9))

    _, cascade_ok, cascade_power = cascade_penalty_16c(
        ch, theta, cfg.Psi_max)
    p_ai = active_tx_power(ch, T, phi)
    _, active_ok = active_power_penalty(p_ai, cfg.P_a_max)

    total_power = float(
        np.linalg.norm(C, "fro") ** 2 + np.linalg.norm(T, "fro") ** 2
    )
    total_power_ok = bool(total_power <= cfg.PT + 1e-12)
    diagnostics = {
        "target_rate": {
            "ok": target_ok,
            "minimum_observed": float(np.min(rate_k)),
            "threshold": r_k_min,
        },
        "passive_cascade": {
            "ok": bool(cascade_ok),
            "observed": float(cascade_power),
            "maximum": float(cfg.Psi_max),
        },
        "active_power": {
            "ok": bool(active_ok),
            "observed": float(p_ai),
            "maximum": float(cfg.P_a_max),
        },
        "total_transmit_power": {
            "ok": total_power_ok,
            "observed": total_power,
            "maximum": float(cfg.PT),
        },
    }
    feasible = all(item["ok"] for item in diagnostics.values())
    return bool(feasible), diagnostics


def optimize_power_split(ch: ChannelSet,
                         theta: np.ndarray,
                         phi: np.ndarray,
                         use_capon: bool = True,
                         sensing_rate_min: float = None) -> tuple:
    """Optimize all U+K stream powers for the paper objective with SLSQP."""
    threshold = cfg.R_min if sensing_rate_min is None else sensing_rate_min
    objective = ObjectiveSpec.paper(target_rate_min=threshold)
    C, T, info = select_feasible_beamformers(
        ch,
        theta,
        phi,
        sensing_rate_min=threshold,
        use_capon=use_capon,
        objective_spec=objective,
        allocation_mode="scipy",
    )
    return (
        float(info["comm_power_frac"]),
        C,
        T,
        float(info["beamformer_score"]),
    )




# ─── Sub-problem P2a1: Transmit Beamformer ────────────────────────────────────

def solve_beamformer(ch: ChannelSet,
                     theta: np.ndarray,
                     phi:   np.ndarray,
                     sensing_rate_min: float = None,
                     return_info: bool = False,
                     use_capon: bool = True) -> tuple:
    """
    Numerical transmit-beamformer block initialized by ZF and Capon.

    Returns: C (M,U), T (M,K)
    """
    frac, C, T, score = optimize_power_split(
        ch, theta, phi,
        use_capon=use_capon,
        sensing_rate_min=sensing_rate_min,
    )

    # Power scaling to satisfy Eq. 16f: ‖C‖_F² + ‖T‖_F² ≤ P_T
    total_pwr = (np.linalg.norm(C, 'fro')**2 +
                 np.linalg.norm(T, 'fro')**2)
    if total_pwr > cfg.PT:
        scale  = np.sqrt(cfg.PT / (total_pwr + 1e-12))
        C     *= scale
        T     *= scale

    if return_info:
        return C, T, dict(comm_power_frac=frac, beamformer_score=score)
    return C, T


# ─── Sub-problem P2b1: HRIS Phase Shifts (SCA) ────────────────────────────────

def _scee_phase_objective(beta_vec: np.ndarray,
                          ch:    ChannelSet,
                          C:     np.ndarray,
                          T:     np.ndarray,
                          is_passive: bool) -> float:
    """Return the negative raw communication objective for SciPy."""
    if is_passive:
        theta = np.exp(1j * beta_vec)
        phi   = _current_phi
        rate_u, rate_k, _, _ = compute_rates(ch, theta, phi, C, T)
        return -joint_sum_rate(rate_u, rate_k)
    else:
        La    = cfg.La
        phi_p = beta_vec[:La]
        alpha = np.exp(beta_vec[La:]) * cfg.alpha_max   # reparameterise α > 0
        alpha = np.clip(alpha, 0, cfg.alpha_max)
        phi   = alpha * np.exp(1j * phi_p)
        theta = _current_theta
        rate_u, rate_k, _, _ = compute_rates(ch, theta, phi, C, T)
        return -joint_sum_rate(rate_u, rate_k)


def _phase_constraint_values(beta_vec, ch, C, T, is_passive):
    """Return Eq. (16) margins; nonnegative means physically feasible."""
    if is_passive:
        theta = np.exp(1j * beta_vec)
        phi = _current_phi
    else:
        active_count = cfg.La
        theta = _current_theta
        alpha = np.clip(
            np.exp(beta_vec[active_count:]) * cfg.alpha_max,
            0.0,
            cfg.alpha_max,
        )
        phi = alpha * np.exp(1j * beta_vec[:active_count])
    _, rate_k, _, _ = compute_rates(ch, theta, phi, C, T)
    threshold = globals().get("_current_sensing_rate_min", cfg.R_min)
    _, _, cascade_power = cascade_penalty_16c(ch, theta, cfg.Psi_max)
    active_power = active_tx_power(ch, T, phi)
    return np.concatenate([
        np.asarray(rate_k, dtype=float) - float(threshold),
        np.array([
            float(cfg.Psi_max) - float(cascade_power),
            float(cfg.P_a_max) - float(active_power),
        ]),
    ])


def _phase_gradient(beta_vec: np.ndarray, ch, C, T, is_passive: bool,
                    eps: float = 1e-4) -> np.ndarray:
    """Numerical gradient via central differences (SCA surrogate gradient)."""
    grad = np.zeros_like(beta_vec)
    f0   = _scee_phase_objective(beta_vec, ch, C, T, is_passive)
    for i in range(len(beta_vec)):
        dv      = beta_vec.copy()
        dv[i]  += eps
        grad[i] = (_scee_phase_objective(dv, ch, C, T, is_passive) - f0) / eps
    return grad


def solve_passive_phases(ch:    ChannelSet,
                         theta: np.ndarray,
                         C:     np.ndarray,
                         T:     np.ndarray,
                         phi:   np.ndarray,
                         sensing_rate_min: float = None) -> np.ndarray:
    """
    P2b1 for passive elements: SCA (Taylor expansion around θ^r, Eq. 43-44).

    Updates: θ^{r+1}_j = θ^r_j − ∇h(θ^r_j) / τ_r
    Returns updated theta_phase (Lp,).
    """
    global _current_phi, _current_sensing_rate_min
    _current_phi = phi.copy()
    _current_sensing_rate_min = cfg.R_min if sensing_rate_min is None else sensing_rate_min

    beta0  = np.angle(theta)           # initial passive phases
    result = _minimize(
        _scee_phase_objective,
        beta0,
        args=(ch, C, T, True),
        method='SLSQP',
        constraints=[{
            "type": "ineq",
            "fun": lambda beta: _phase_constraint_values(
                beta, ch, C, T, True
            ),
        }],
        options=dict(maxiter=cfg.sca_max_iter, ftol=cfg.ao_tol)
    )
    if result.success and np.min(_phase_constraint_values(
        result.x, ch, C, T, True
    )) >= -1e-8:
        return result.x
    return beta0


def solve_active_phases(ch:    ChannelSet,
                        phi:   np.ndarray,
                        C:     np.ndarray,
                        T:     np.ndarray,
                        theta: np.ndarray,
                        sensing_rate_min: float = None) -> np.ndarray:
    """
    P2b1 for active elements: joint optimisation of phase + amplification.

    Algorithm 3 objective: maximize R_u subject to the Eq. (16) constraints.
    Sensing performance R_k comes from the Capon beamformer T;
    φ is jointly steered so amplification helps sensing without driving
    active noise (σ²_qa·Σ|φ_i|²) too high at UEs.
    Returns complex phi (La,) = α · exp(jβ^{ac}).
    """
    global _current_theta, _current_sensing_rate_min
    _current_theta = theta.copy()
    _current_sensing_rate_min = cfg.R_min if sensing_rate_min is None else sensing_rate_min

    La    = cfg.La
    alpha = np.abs(phi)
    # Reparameterise: u = [β^{ac}_i | log(α_i / α_max)]
    beta0 = np.angle(phi)
    log_a = np.log(alpha / (cfg.alpha_max + 1e-9) + 1e-9)
    u0    = np.concatenate([beta0, log_a])

    # Finite log-amplitude bounds keep SLSQP numerically well-defined.
    bounds = ([(None, None)] * La +
              [(-20.0, 0.0)] * La)

    result = _minimize(
        _scee_phase_objective,
        u0,
        args=(ch, C, T, False),
        method='SLSQP',
        bounds=bounds,
        constraints=[{
            "type": "ineq",
            "fun": lambda beta: _phase_constraint_values(
                beta, ch, C, T, False
            ),
        }],
        options=dict(maxiter=cfg.sca_max_iter, ftol=cfg.ao_tol)
    )
    selected = result.x if (
        result.success
        and np.min(_phase_constraint_values(result.x, ch, C, T, False)) >= -1e-8
    ) else u0
    phi_p = selected[:La]
    alpha_opt = np.exp(selected[La:]) * cfg.alpha_max
    alpha_opt = np.clip(alpha_opt, 0, cfg.alpha_max)
    return alpha_opt * np.exp(1j * phi_p)


# ─── Sub-problem P2c1: UAV Trajectory (SCA) ──────────────────────────────────

def _traj_objective(pos_xy: np.ndarray,
                    pos_ues: list, pos_tes: list, theta, phi, C, T,
                    sigma2: float,
                    channel_seed: int = None,
                    sensing_rate_min: float = None) -> float:
    """
    Negative sum-rate at horizontal UAV position pos_xy = [x, y].
    Uses the same seeded small-scale realization at every candidate position.
    """
    pos_uav = np.array([pos_xy[0], pos_xy[1], cfg.z_UAV])
    ch_joint = ChannelSet(pos_uav, pos_ues, pos_tes, sigma2,
                          seed=channel_seed)
    return -_score_beamformers(
        ch_joint, theta, phi, C, T,
        sensing_rate_min=sensing_rate_min,
    )


def _trajectory_constraint_values(pos_xy, initial_xy, pos_ues, pos_tes,
                                  theta, phi, C, T, sigma2, channel_seed,
                                  sensing_rate_min):
    """Return target, HRIS-power, BS-power, and displacement margins."""
    position = np.array([pos_xy[0], pos_xy[1], cfg.z_UAV])
    channel = ChannelSet(
        position, pos_ues, pos_tes, sigma2, seed=channel_seed
    )
    _, rate_k, _, _ = compute_rates(channel, theta, phi, C, T)
    threshold = cfg.R_min if sensing_rate_min is None else sensing_rate_min
    _, _, cascade_power = cascade_penalty_16c(
        channel, theta, cfg.Psi_max
    )
    active_power = active_tx_power(channel, T, phi)
    transmit_power = float(
        np.linalg.norm(C, "fro") ** 2 + np.linalg.norm(T, "fro") ** 2
    )
    displacement_margin = cfg.D_max ** 2 - float(
        np.sum((np.asarray(pos_xy) - np.asarray(initial_xy)) ** 2)
    )
    return np.concatenate([
        np.asarray(rate_k, dtype=float) - float(threshold),
        np.array([
            float(cfg.Psi_max) - float(cascade_power),
            float(cfg.P_a_max) - float(active_power),
            float(cfg.PT) - transmit_power,
            displacement_margin,
        ]),
    ])


def solve_uav_trajectory(pos_uav:  np.ndarray,
                         pos_ues:  list,
                         pos_tes:  list,
                         theta:    np.ndarray,
                         phi:      np.ndarray,
                         C:        np.ndarray,
                         T:        np.ndarray,
                         sigma2:   float = None,
                         channel_seed: int = None,
                         sensing_rate_min: float = None) -> np.ndarray:
    """
    P2c1: SCA-based UAV trajectory optimisation (Eq. 50-51, Algorithm 2).

    Returns updated UAV [x, y] position for the next slot.
    The displacement is bounded by D_max = v_max · Δ_T  (Eq. 1-2).
    """
    x0     = pos_uav[:2].copy()
    D_max  = cfg.D_max

    # Box constraint: stay in service area AND within one step from current pos
    bounds = [
        (max(cfg.x_min, x0[0] - D_max), min(cfg.x_max, x0[0] + D_max)),
        (max(cfg.y_min, x0[1] - D_max), min(cfg.y_max, x0[1] + D_max)),
    ]

    result = _minimize(
        _traj_objective,
        x0,
        args=(pos_ues, pos_tes, theta, phi, C, T, sigma2, channel_seed,
              sensing_rate_min),
        method='SLSQP',
        bounds=bounds,
        constraints=[{
            "type": "ineq",
            "fun": lambda xy: _trajectory_constraint_values(
                xy, x0, pos_ues, pos_tes, theta, phi, C, T, sigma2,
                channel_seed, sensing_rate_min
            ),
        }],
        options=dict(maxiter=cfg.sca_max_iter, ftol=cfg.ao_tol)
    )
    selected = result.x if (
        result.success
        and np.min(_trajectory_constraint_values(
            result.x, x0, pos_ues, pos_tes, theta, phi, C, T, sigma2,
            channel_seed, sensing_rate_min
        )) >= -1e-8
    ) else x0
    new_xy = np.clip(selected,
                     [cfg.x_min, cfg.y_min],
                     [cfg.x_max, cfg.y_max])
    displacement = new_xy - x0
    distance = float(np.linalg.norm(displacement))
    if distance > D_max:
        new_xy = x0 + displacement * (D_max / distance)
        new_xy = np.clip(new_xy,
                         [cfg.x_min, cfg.y_min],
                         [cfg.x_max, cfg.y_max])
    new_pos = np.array([new_xy[0], new_xy[1], cfg.z_UAV])
    return new_pos


# ─── Algorithm 3: Full AO ─────────────────────────────────────────────────────

def run_ao(pos_uav:  np.ndarray,
           pos_ues:  list,
           pos_tes:  list,
           sigma2:   float = None,
           n_iter:   int   = cfg.ao_max_iter,
           fix_uav:  bool  = False,
           sensing_rate_min: float = None,
           stop_on_convergence: bool = True,
           seed: int = None) -> dict:
    """
    Algorithm 3: Alternating Optimisation.

    Iterates:
      1. Beamformer design  C, T           (P2a1)
      2. HRIS phase shifts  θ_j, ϕ_i      (P2b1)
      3. UAV trajectory     ι             (P2c1 — one step)
    until convergence  (fractional Δ < δ).

    Returns dict with final sum-rate, beamformers, and phase shifts.
    """
    # ── Initialisation ────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    theta = np.exp(1j * rng.uniform(0, 2 * np.pi, cfg.Lp))
    phi   = (cfg.alpha_max / 2) * np.exp(1j * rng.uniform(0, 2 * np.pi, cfg.La))
    pos   = pos_uav.copy()

    # Fix: sample ONE channel realisation and hold it fixed for the entire AO
    # run.  Re-sampling each iteration makes the objective non-stationary so
    # the algorithm can never converge — this was the root cause of oscillation.
    channel_seed = int(rng.integers(0, 2**32 - 1))
    ch = ChannelSet(pos, pos_ues, pos_tes, sigma2, seed=channel_seed)

    history_objective = []
    history_sum_rate = []
    history_sensing_rate = []
    history_feasible = []
    # Best-solution tracking uses the paper Eq. (16) communication objective,
    # while histories keep communication and sensing rates separate for labels.
    best_objective = -np.inf
    best_state = None
    best_power_split = None
    best_diagnostics = None
    best_feasible_objective = -np.inf
    best_feasible_state = None
    best_feasible_power_split = None
    best_feasible_diagnostics = None

    # Iteration 0 is the initialized feasible beamformer/phase state. Keeping it
    # explicit makes the convergence trace match the paper's 0..20 x-axis.
    C, T, bf_info = solve_beamformer(
        ch, theta, phi,
        sensing_rate_min=sensing_rate_min,
        return_info=True,
    )
    rate_u, rate_k, _, _ = compute_rates(ch, theta, phi, C, T)
    cur_sum_rate = float(np.sum(rate_u))
    cur_sensing_rate = float(np.sum(rate_k))
    cur_objective = joint_sum_rate(rate_u, rate_k)
    cur_feasible, cur_diagnostics = _constraint_diagnostics(
        ch, theta, phi, C, T, rate_k,
        sensing_rate_min=sensing_rate_min,
    )
    best_objective = cur_objective
    best_power_split = float(bf_info["comm_power_frac"])
    best_diagnostics = cur_diagnostics
    best_state = (C.copy(), T.copy(), theta.copy(), phi.copy(), pos.copy(),
                  rate_u.copy(), rate_k.copy())
    if cur_feasible:
        best_feasible_objective = cur_objective
        best_feasible_power_split = best_power_split
        best_feasible_diagnostics = cur_diagnostics
        best_feasible_state = best_state
    history_objective.append(cur_objective)
    history_sum_rate.append(cur_sum_rate)
    history_sensing_rate.append(cur_sensing_rate)
    history_feasible.append(cur_feasible)
    prev_objective = cur_objective

    for r in range(1, n_iter + 1):
        # ── P2a1: Beamformer ──────────────────────────────────────────────────
        C, T, bf_info = solve_beamformer(
            ch, theta, phi,
            sensing_rate_min=sensing_rate_min,
            return_info=True,
        )

        # ── P2b1a: Passive phase shifts ───────────────────────────────────────
        beta_pas = solve_passive_phases(
            ch, theta, C, T, phi,
            sensing_rate_min=sensing_rate_min)
        theta    = np.exp(1j * beta_pas)

        # ── P2b1b: Active phase shifts + amplification ────────────────────────
        if cfg.La > 0:
            phi = solve_active_phases(
                ch, phi, C, T, theta,
                sensing_rate_min=sensing_rate_min)

        # ── P2c1: UAV trajectory — skip when fix_uav=True (convergence plots) ──
        if not fix_uav:
            pos = solve_uav_trajectory(
                pos, pos_ues, pos_tes, theta, phi, C, T, sigma2,
                channel_seed=channel_seed,
                sensing_rate_min=sensing_rate_min)
            ch = ChannelSet(pos, pos_ues, pos_tes, sigma2,
                            seed=channel_seed)

        # ── Evaluate current objective ────────────────────────────────────────
        rate_u, rate_k, _, _ = compute_rates(ch, theta, phi, C, T)
        cur_sum_rate = float(np.sum(rate_u))
        cur_sensing_rate = float(np.sum(rate_k))
        cur_objective = joint_sum_rate(rate_u, rate_k)
        cur_feasible, cur_diagnostics = _constraint_diagnostics(
            ch, theta, phi, C, T, rate_k,
            sensing_rate_min=sensing_rate_min,
        )

        # Update best-seen solution
        if cur_objective > best_objective:
            best_objective = cur_objective
            best_power_split = float(bf_info["comm_power_frac"])
            best_diagnostics = cur_diagnostics
            best_state = (C.copy(), T.copy(), theta.copy(), phi.copy(),
                          pos.copy(), rate_u.copy(), rate_k.copy())

        if cur_feasible and cur_objective > best_feasible_objective:
            best_feasible_objective = cur_objective
            best_feasible_power_split = float(bf_info["comm_power_frac"])
            best_feasible_diagnostics = cur_diagnostics
            best_feasible_state = (
                C.copy(), T.copy(), theta.copy(), phi.copy(), pos.copy(),
                rate_u.copy(), rate_k.copy(),
            )

        # Publication histories are raw numerical iterates. Best-feasible
        # tracking is used only for the final returned solution and must not
        # repair or force a monotone convergence curve.
        history_objective.append(cur_objective)
        history_sum_rate.append(cur_sum_rate)
        history_sensing_rate.append(cur_sensing_rate)
        history_feasible.append(cur_feasible)

        frac_inc = abs(cur_objective - prev_objective) / (abs(prev_objective) + 1e-12)
        if stop_on_convergence and frac_inc < cfg.ao_tol:
            break
        prev_objective = cur_objective

    if best_feasible_state is not None:
        selected_state = best_feasible_state
        selected_objective = best_feasible_objective
        selected_power_split = best_feasible_power_split
        selected_diagnostics = best_feasible_diagnostics
        selected_is_feasible = True
    else:
        selected_state = best_state
        selected_objective = best_objective
        selected_power_split = best_power_split
        selected_diagnostics = best_diagnostics
        selected_is_feasible = False

    C, T, theta, phi, pos, rate_u, rate_k = selected_state
    communication_rate = float(np.sum(rate_u))
    sensing_rate = float(np.sum(rate_k))
    return dict(objective=selected_objective,
                joint_rate=communication_rate + sensing_rate,
                sum_rate=communication_rate,
                sensing_rate=sensing_rate,
                rate_u=rate_u, rate_k=rate_k,
                C=C, T=T, theta=theta, phi=phi,
                pos_uav=pos,
                comm_power_frac=selected_power_split,
                is_feasible=selected_is_feasible,
                constraint_diagnostics=selected_diagnostics,
                history_objective=history_objective,
                history_sum_rate=history_sum_rate,
                history_sensing_rate=history_sensing_rate,
                history_feasible=history_feasible)
