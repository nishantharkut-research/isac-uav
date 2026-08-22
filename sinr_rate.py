"""
sinr_rate.py — SINR / rate computation with SCEE.

Implements:
  Eq. (12)-(15) : SINR and achievable rates
  Eq. (21),(24),(25): matrix SCEE covariance and log-determinant rates
  Eq. (17)      : Capon beamformer
  ZF precoder   : standard closed-form

Reference:
  Saikia et al., IEEE Trans. Commun., 2025.
"""
import numpy as np
from config import cfg
from beam_allocation import (
    AllocationConstraints,
    optimize_power_allocation,
    select_fast_power_allocation,
)
from research_objectives import ObjectiveSpec


SENSING_BEAMFORMER_MODES = frozenset({
    "capon", "matched_filter", "channel_independent_dft"
})


def resolve_sensing_beamformer_mode(sensing_beamformer_mode=None,
                                    use_capon=None):
    """Resolve the explicit mode while retaining the legacy boolean API."""
    if sensing_beamformer_mode is None:
        mode = "capon" if use_capon is not False else "matched_filter"
    else:
        mode = str(sensing_beamformer_mode)
        if use_capon is not None:
            legacy = "capon" if bool(use_capon) else "matched_filter"
            if mode != legacy:
                raise ValueError(
                    "sensing_beamformer_mode conflicts with use_capon"
                )
    if mode not in SENSING_BEAMFORMER_MODES:
        raise ValueError(f"Unknown sensing beamformer mode: {mode}")
    return mode


# ─── Beamformers ─────────────────────────────────────────────────────────────

def zf_precoder(H_stack: np.ndarray, P_budget: float) -> np.ndarray:
    """
    Zero-Forcing precoder: F = H^H (H H^H)^{-1}, power-normalised.
    H_stack : (N*U, M) stacked estimated channels
    Returns : (M, U)  – one column per user (Du = 1 stream each)
    """
    H   = H_stack
    HHH = H @ H.conj().T                                      # (N*U, N*U)
    reg = max(float(np.mean(np.abs(HHH.diagonal()))), 1e-30) * 1e-6
    F   = H.conj().T @ np.linalg.inv(HHH + reg * np.eye(HHH.shape[0]))
    # Take first U columns (one per user)
    F   = F[:, :cfg.U]
    # Normalise to power budget
    F  /= (np.linalg.norm(F, 'fro') + 1e-12)
    F  *= np.sqrt(P_budget)
    return F  # (M, U)


def capon_beamformer(Ry: np.ndarray, a_steer: np.ndarray) -> np.ndarray:
    """
    Capon (MVDR) beamformer  t* = R_y^{-1} a / (a^H R_y^{-1} a)  (Eq. 17).

    Ry      : (M, M) sample covariance  (Hermitian positive-definite)
    a_steer : (M,)   steering vector toward target
    Returns : (M,) beamforming vector
    """
    reg    = max(float(np.mean(np.abs(Ry.diagonal()))), 1e-30) * 1e-6
    Ry_inv = np.linalg.inv(Ry + reg * np.eye(Ry.shape[0]))
    denom  = (a_steer.conj() @ Ry_inv @ a_steer).real
    return Ry_inv @ a_steer / (denom + 1e-12)


def _eff_channel(H: np.ndarray) -> np.ndarray:
    """
    Max-SVD effective channel row for an (N, M) composite channel H.

    Returns the dominant right-singular-vector scaled by its singular value:
        h_eff = σ₁ · V[:, 0]   shape (M,)

    This choice provides maximum receive gain and is the correct effective
    channel direction for ZF precoder design with single-stream per user
    when N > 1 receive antennas are present.
    """
    _, s, Vt = np.linalg.svd(H, full_matrices=False)
    return s[0] * Vt[0, :]   # (M,) complex


def build_beam_directions(ch, theta: np.ndarray, phi: np.ndarray,
                          use_capon=None,
                          sensing_beamformer_mode=None):
    """Construct unit-norm communication and sensing beam directions.

    WFB mode uses matched-filter sensing beams. Sensing is not projected into
    the communication null space because Eqs. (12), (14), and Fig. 13 retain
    cross-function interference.
    """
    mode = resolve_sensing_beamformer_mode(
        sensing_beamformer_mode, use_capon
    )
    h_ue = ch.composite_ue(theta)
    h_stack = np.vstack([_eff_channel(H)[None, :] for H in h_ue])
    C = zf_precoder(h_stack, 1.0)
    C /= np.maximum(np.linalg.norm(C, axis=0, keepdims=True), 1e-12)

    h_target = ch.composite_te(phi)
    target_dirs = []
    for H in h_target:
        direction = _eff_channel(H).conj()
        target_dirs.append(direction / (np.linalg.norm(direction) + 1e-12))

    # The Q snapshots used for R_y in Eq. (17) are not published. The
    # estimated spatial covariance is the explicit deterministic assumption.
    comm_cov = h_stack.conj().T @ h_stack
    cov_scale = max(float(np.trace(comm_cov).real) / max(cfg.M, 1), 1e-30)
    T = np.zeros((cfg.M, cfg.K), dtype=complex)
    for k, steering in enumerate(target_dirs):
        if mode == "capon":
            R_y = comm_cov.copy()
            for other, direction in enumerate(target_dirs):
                if other != k:
                    R_y += cov_scale * np.outer(direction, direction.conj())
            R_y += max(cfg.sigma2_u, 1e-12 * cov_scale) * np.eye(cfg.M)
            T[:, k] = capon_beamformer(R_y, steering)
        elif mode == "matched_filter":
            T[:, k] = steering
        else:
            element = np.arange(cfg.M, dtype=float)
            T[:, k] = np.exp(
                2j * np.pi * element * (k % cfg.M) / cfg.M
            ) / np.sqrt(cfg.M)

    T /= np.maximum(np.linalg.norm(T, axis=0, keepdims=True), 1e-12)
    return C, T


def beamformers_from_powers(C_directions, T_directions, powers):
    """Scale unit directions by square-root watts for exact stream powers."""
    powers = np.asarray(powers, dtype=float)
    if powers.shape != (cfg.U + cfg.K,) or np.any(powers < 0.0):
        raise ValueError("powers must be a nonnegative U+K vector.")
    C = C_directions * np.sqrt(powers[:cfg.U])[None, :]
    T = T_directions * np.sqrt(powers[cfg.U:])[None, :]
    return C, T


def build_beamformers(ch, theta: np.ndarray, phi: np.ndarray,
                      use_capon=None,
                      sensing_beamformer_mode=None,
                      comm_power_frac: float = None):
    """Construct equal-per-group streams under Eq. (16f)."""
    if comm_power_frac is None:
        comm_power_frac = getattr(cfg, "comm_power_frac")
    fraction = float(np.clip(comm_power_frac, 1e-3, 1.0 - 1e-3))
    directions = build_beam_directions(
        ch, theta, phi, use_capon=use_capon,
        sensing_beamformer_mode=sensing_beamformer_mode,
    )
    powers = np.concatenate([
        np.full(cfg.U, cfg.PT * fraction / cfg.U),
        np.full(cfg.K, cfg.PT * (1.0 - fraction) / cfg.K),
    ])
    return beamformers_from_powers(*directions, powers)


def select_feasible_beamformers(ch, theta: np.ndarray, phi: np.ndarray,
                                sensing_rate_min: float = None,
                                sensing_only_sinr_min: float = 0.0,
                                use_capon=None,
                                sensing_beamformer_mode=None,
                                objective_spec: ObjectiveSpec = None,
                                allocation_mode: str = "fast",
                                previous_powers=None):
    """Allocate stream powers for the active research objective."""
    threshold = cfg.R_min if sensing_rate_min is None else sensing_rate_min
    if objective_spec is None:
        objective_spec = ObjectiveSpec.paper(target_rate_min=threshold)
    elif not isinstance(objective_spec, ObjectiveSpec):
        objective_spec = ObjectiveSpec.from_mapping(objective_spec)
    if not np.isclose(float(threshold), objective_spec.target_rate_min):
        raise ValueError("sensing_rate_min conflicts with objective_spec.")

    C_directions, T_directions = build_beam_directions(
        ch, theta, phi, use_capon=use_capon,
        sensing_beamformer_mode=sensing_beamformer_mode,
    )

    def evaluate(powers):
        C, T = beamformers_from_powers(C_directions, T_directions, powers)
        rate_u, rate_k, _, _ = compute_rates(ch, theta, phi, C, T)
        return {
            "rate_u": rate_u,
            "rate_k": rate_k,
            "sensing_only_sinr": sensing_only_sinr(ch, phi, T),
            "active_power": active_tx_power(ch, C, T, phi),
        }

    constraints = AllocationConstraints(
        total_power=cfg.PT,
        target_rate_min=objective_spec.target_rate_min,
        user_rate_min=objective_spec.user_rate_min,
        sensing_only_sinr_min=sensing_only_sinr_min,
        active_power_max=cfg.P_a_max,
    )
    if allocation_mode == "fast":
        best = select_fast_power_allocation(
            cfg.U, cfg.K, evaluate, constraints, objective_spec,
            previous_powers=previous_powers,
        )
    elif allocation_mode == "scipy":
        best = optimize_power_allocation(
            cfg.U, cfg.K, evaluate, constraints, objective_spec,
            initial_powers=previous_powers,
        )
    else:
        raise ValueError("allocation_mode must be 'fast' or 'scipy'.")
    C, T = beamformers_from_powers(C_directions, T_directions, best.powers)
    rate_deficit = float(
        np.maximum(objective_spec.target_rate_min - best.rate_k, 0.0).sum()
    )
    sinr_deficit = float(np.maximum(
        sensing_only_sinr_min - best.sensing_only_sinr, 0.0
    ).sum())
    active_deficit = max(0.0, best.active_power - float(cfg.P_a_max))
    return C, T, dict(
        comm_power_frac=float(np.sum(best.powers[:cfg.U]) / cfg.PT),
        stream_powers=best.powers.copy(),
        beamformer_score=best.objective_value,
        allocation_method=best.method,
        allocation_solver_success=best.solver_success,
        allocation_feasible=best.feasible,
        allocation_normalized_deficit=best.total_normalized_deficit,
        sensing_constraint_feasible=best.feasible,
        sensing_rate_deficit=rate_deficit,
        sensing_only_sinr_deficit=sinr_deficit,
        active_power=best.active_power,
        active_power_feasible=active_deficit <= 1e-12,
        active_power_deficit=active_deficit,
    )


def _ju_scee_ue(ch, C_bar: np.ndarray, u: int) -> float:
    """
    Scalar approximation of Tr(Ju) for the u-th UE  (Eq. 24).

    Uses ch.sv_* — per-channel effective variances σ²_eff = ρ·E[|h|²] —
    instead of raw cfg.sigma2_xx which are dimensionless ratios (not watts).

    Four SCEE interference terms:
      ① σ²_{b,u}  · Tr(C̄) · N
      ② σ²_{br,j} · Tr(C̄) · Σ_j ‖ĥ_{ru,j}‖²
      ③ σ²_{ru,j} · Σ_j  ĝ_{br,j} C̄ ĝ^H_{br,j}
      ④ σ²_{br,j} · σ²_{ru,j} · Lp · Tr(C̄) · N   (2nd-order cross term)
    """
    tr_C    = float(np.real(np.trace(C_bar)))
    s2_bu   = ch.sv_bu[u]     # σ²_eff for BS→UE_u
    s2_brj  = ch.sv_brj       # σ²_eff for BS→passive-HRIS
    s2_ruj  = ch.sv_ruj[u]    # σ²_eff for passive-HRIS→UE_u

    term1 = s2_bu  * tr_C * cfg.N

    term2 = s2_brj * tr_C * sum(
        np.linalg.norm(ch.H_ru_hat[u][:, j])**2
        for j in range(cfg.Lp))

    term3 = s2_ruj * sum(
        float(np.real(ch.G_brj_hat[j].conj() @ C_bar @ ch.G_brj_hat[j]))
        for j in range(cfg.Lp))

    term4 = s2_brj * s2_ruj * cfg.Lp * tr_C * cfg.N

    return term1 + term2 + term3 + term4


def _jk_scee_te(ch, T_bar: np.ndarray, k: int) -> float:
    """Scalar approximation of Tr(Jk) for the k-th TE  (Eq. 24 analogue)."""
    tr_T    = float(np.real(np.trace(T_bar)))
    s2_bk   = ch.sv_bk[k]
    s2_bri  = ch.sv_bri
    s2_rti  = ch.sv_rti[k]

    term1 = s2_bk  * tr_T * cfg.N

    term2 = s2_bri * tr_T * sum(
        np.linalg.norm(ch.H_rt_hat[k][:, i])**2
        for i in range(cfg.La))

    term3 = s2_rti * sum(
        float(np.real(ch.G_bri_hat[i].conj() @ T_bar @ ch.G_bri_hat[i]))
        for i in range(cfg.La))

    term4 = s2_bri * s2_rti * cfg.La * tr_T * cfg.N

    return term1 + term2 + term3 + term4


def _outer(vector: np.ndarray) -> np.ndarray:
    return np.outer(vector, vector.conj())


def _cascade_error_covariance(H_r: np.ndarray, G_br: np.ndarray,
                              weights: np.ndarray, Q: np.ndarray,
                              sigma2_g: float,
                              sigma2_h: float) -> np.ndarray:
    """Gaussian-Kronecker cascade-error covariance from Eq. (24)."""
    n_receive = H_r.shape[0]
    identity = np.eye(n_receive, dtype=complex)
    trace_q = max(float(np.trace(Q).real), 0.0)
    weight_power = np.abs(np.asarray(weights, dtype=complex)) ** 2
    if weight_power.size == 0:
        return np.zeros((n_receive, n_receive), dtype=complex)
    weighted_h = H_r * weight_power[np.newaxis, :]
    receive_covariance = weighted_h @ H_r.conj().T
    g_q_g = np.real(np.einsum(
        "lm,mn,ln->l", G_br, Q, G_br.conj(), optimize=True
    ))
    isotropic_power = (
        sigma2_h * float(np.sum(weight_power * np.maximum(g_q_g, 0.0)))
        + sigma2_g * sigma2_h * trace_q * float(np.sum(weight_power))
    )
    return (
        sigma2_g * trace_q * receive_covariance
        + isotropic_power * identity
    )


def _logdet_rate(signal_covariance: np.ndarray,
                 interference_covariance: np.ndarray) -> float:
    """Compute log2|I + S J^-1| through a Hermitian whitening transform."""
    J = (interference_covariance + interference_covariance.conj().T) / 2.0
    scale = max(float(np.trace(J).real) / max(J.shape[0], 1), 1e-30)
    eigenvalues, eigenvectors = np.linalg.eigh(
        J + max(1e-15, 1e-12 * scale) * np.eye(J.shape[0])
    )
    inverse_sqrt = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-30)))
        @ eigenvectors.conj().T
    )
    whitened = inverse_sqrt @ signal_covariance @ inverse_sqrt
    whitened = (whitened + whitened.conj().T) / 2.0
    modes = np.maximum(np.linalg.eigvalsh(whitened).real, 0.0)
    return float(np.sum(np.log2(1.0 + modes)))


def _compute_scee_rates(ch, theta, phi, C, T, H_bar_u, H_bar_k):
    """Matrix rates for Eqs. (21), (24), and (25)."""
    from ambiguity_hypotheses import get_cross_function_interference_hypothesis

    cross_interference = get_cross_function_interference_hypothesis(
        cfg.cross_function_interference_hypothesis_id
    )
    identity = np.eye(cfg.N, dtype=complex)
    C_total = C @ C.conj().T
    T_total = T @ T.conj().T
    trace_c = max(float(np.trace(C_total).real), 0.0)
    trace_t = max(float(np.trace(T_total).real), 0.0)

    def amplifier_noise_covariance(target_index=None):
        indices = range(cfg.K) if target_index is None else [target_index]
        covariances = []
        for k in indices:
            covariance = np.zeros((cfg.N, cfg.N), dtype=complex)
            for i in range(cfg.La):
                h = ch.H_rt_hat[k][:, i]
                covariance += cfg.sigma2_a * abs(phi[i]) ** 2 * (
                    _outer(h) + ch.sv_rti[k] * identity
                )
            covariances.append(covariance)
        return sum(covariances) / max(len(covariances), 1)

    rate_u = np.zeros(cfg.U)
    for u in range(cfg.U):
        Hu = H_bar_u[u]
        desired = Hu @ _outer(C[:, u]) @ Hu.conj().T
        C_interference = C_total - _outer(C[:, u])
        J = Hu @ C_interference @ Hu.conj().T
        if cross_interference.receiver_consistent:
            J += Hu @ T_total @ Hu.conj().T
            J += ch.sv_bu[u] * trace_t * identity
            J += _cascade_error_covariance(
                ch.H_ru_hat[u], ch.G_brj_hat, theta, T_total,
                ch.sv_brj, ch.sv_ruj[u],
            )
        else:
            for k in range(cfg.K):
                Tk = _outer(T[:, k])
                Hk = H_bar_k[k]
                J += Hk @ Tk @ Hk.conj().T
                J += ch.sv_bk[k] * max(float(np.trace(Tk).real), 0.0) * identity
                J += _cascade_error_covariance(
                    ch.H_rt_hat[k], ch.G_bri_hat, phi, Tk,
                    ch.sv_bri, ch.sv_rti[k],
                )
        J += ch.sv_bu[u] * trace_c * identity
        J += _cascade_error_covariance(
            ch.H_ru_hat[u], ch.G_brj_hat, theta, C_total,
            ch.sv_brj, ch.sv_ruj[u],
        )
        J += amplifier_noise_covariance() + cfg.sigma2_u * identity
        rate_u[u] = _logdet_rate(desired, J)

    rate_k = np.zeros(cfg.K)
    for k in range(cfg.K):
        Hk = H_bar_k[k]
        desired = Hk @ _outer(T[:, k]) @ Hk.conj().T
        T_interference = T_total - _outer(T[:, k])
        J = Hk @ T_interference @ Hk.conj().T
        if cross_interference.receiver_consistent:
            J += Hk @ C_total @ Hk.conj().T
            J += ch.sv_bk[k] * trace_c * identity
            J += _cascade_error_covariance(
                ch.H_rt_hat[k], ch.G_bri_hat, phi, C_total,
                ch.sv_bri, ch.sv_rti[k],
            )
        else:
            for u in range(cfg.U):
                Cu = _outer(C[:, u])
                Hu = H_bar_u[u]
                J += Hu @ Cu @ Hu.conj().T
                J += ch.sv_bu[u] * max(float(np.trace(Cu).real), 0.0) * identity
                J += _cascade_error_covariance(
                    ch.H_ru_hat[u], ch.G_brj_hat, theta, Cu,
                    ch.sv_brj, ch.sv_ruj[u],
                )
        J += ch.sv_bk[k] * trace_t * identity
        J += _cascade_error_covariance(
            ch.H_rt_hat[k], ch.G_bri_hat, phi, T_total,
            ch.sv_bri, ch.sv_rti[k],
        )
        J += amplifier_noise_covariance(k) + cfg.sigma2_k * identity
        rate_k[k] = _logdet_rate(desired, J)

    return rate_u, rate_k, np.exp2(rate_u) - 1.0, np.exp2(rate_k) - 1.0


# ─── Main SINR / Rate Computation ────────────────────────────────────────────

def compute_rates(ch,
                  theta: np.ndarray,
                  phi:   np.ndarray,
                  C:     np.ndarray,
                  T:     np.ndarray,
                  use_scee_model: bool = True):
    """
    Compute per-user and per-target SINR and achievable rates
    under SCEE  (Eq. 12–15, 24).

    Parameters
    ----------
    ch    : ChannelSet
    theta : (Lp,)  complex, |θ_j| = 1  (passive phase shifts)
    phi   : (La,)  complex  (active: amplitude × phase, |ϕ_i| ≤ α_max)
    C     : (M, U) communication precoder
    T     : (M, K) sensing beamformer

    Returns
    -------
    rate_u : (U,)  per-user  achievable rate  [bits/s/Hz]
    rate_k : (K,)  per-target achievable rate [bits/s/Hz]
    sinr_u : (U,)  per-user SINR  (linear)
    sinr_k : (K,)  per-target SINR  (linear)
    """
    H_bar_u = ch.composite_ue(theta)
    H_bar_k = ch.composite_te(phi)
    from ambiguity_hypotheses import get_cross_function_interference_hypothesis
    cross_interference = get_cross_function_interference_hypothesis(
        cfg.cross_function_interference_hypothesis_id
    )
    # The paper does not reconcile scalar pCSI Eqs. (12)-(15) with matrix SCEE
    # Eqs. (21)/(25) at zero NMSE. Keep that choice explicit and fingerprinted.
    from ambiguity_hypotheses import use_matrix_scee_rate
    if use_scee_model and use_matrix_scee_rate(
        cfg, getattr(ch, "scee_nmse", 0.0)
    ):
        return _compute_scee_rates(
            ch, theta, phi, C, T, H_bar_u, H_bar_k
        )
    C_bar = C @ C.conj().T
    T_bar = T @ T.conj().T

    def active_amplifier_noise(target_index=None):
        """Compute sigma_a^2 sum_i ||h_rt,i phi_i||^2 from Eqs. (12),(14)."""
        if cfg.La == 0:
            return 0.0
        indices = range(cfg.K) if target_index is None else [target_index]
        values = []
        for idx in indices:
            h_rt = ch.H_rt_hat[idx]
            values.append(sum(
                abs(phi[i]) ** 2 * float(np.linalg.norm(h_rt[:, i]) ** 2)
                for i in range(cfg.La)
            ))
        return cfg.sigma2_a * float(np.mean(values))

    # The primary branch follows the published cross-channel terms exactly.
    # The sensitivity branch applies every waveform to the current receiver.
    target_beam_powers = None
    comm_beam_powers = None
    if not cross_interference.receiver_consistent:
        target_beam_powers = [
            float(np.linalg.norm(H_bar_k[k] @ T[:, k]) ** 2)
            for k in range(cfg.K)
        ]
        comm_beam_powers = [
            float(np.linalg.norm(H_bar_u[u] @ C[:, u]) ** 2)
            for u in range(cfg.U)
        ]

    rate_u, sinr_u = np.zeros(cfg.U), np.zeros(cfg.U)
    for u in range(cfg.U):
        Hu = H_bar_u[u]
        sig = float(np.linalg.norm(Hu @ C[:, u]) ** 2) + 1e-30
        intf_uu = sum(float(np.linalg.norm(Hu @ C[:, i]) ** 2)
                      for i in range(cfg.U) if i != u)
        intf_ku = (
            float(np.sum(np.abs(Hu @ T) ** 2))
            if cross_interference.receiver_consistent
            else float(np.sum(target_beam_powers))
        )
        scee_u = _ju_scee_ue(ch, C_bar, u)
        denom = (intf_uu + intf_ku + active_amplifier_noise()
                 + scee_u + cfg.sigma2_u)
        sinr_u[u] = sig / (denom + 1e-15)
        rate_u[u] = np.log2(1.0 + sinr_u[u])

    rate_k, sinr_k = np.zeros(cfg.K), np.zeros(cfg.K)
    for k in range(cfg.K):
        Hk = H_bar_k[k]
        sig = float(np.linalg.norm(Hk @ T[:, k]) ** 2) + 1e-30
        intf_kk = sum(float(np.linalg.norm(Hk @ T[:, i]) ** 2)
                      for i in range(cfg.K) if i != k)
        intf_uk = (
            float(np.sum(np.abs(Hk @ C) ** 2))
            if cross_interference.receiver_consistent
            else float(np.sum(comm_beam_powers))
        )
        scee_k = _jk_scee_te(ch, T_bar, k)
        denom = (intf_kk + intf_uk + active_amplifier_noise(k)
                 + scee_k + cfg.sigma2_k)
        sinr_k[k] = sig / (denom + 1e-15)
        rate_k[k] = np.log2(1.0 + sinr_k[k])

    return rate_u, rate_k, sinr_u, sinr_k


def sensing_only_sinr(ch, phi: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Return Remark 1's sensing SINR, excluding communication interference."""
    H_bar_k = ch.composite_te(phi)
    values = np.zeros(cfg.K, dtype=float)
    for k, Hk in enumerate(H_bar_k):
        beam_powers = np.sum(np.abs(Hk @ T) ** 2, axis=0).real
        signal = float(beam_powers[k])
        interference = float(np.sum(beam_powers) - signal)
        active_noise = cfg.sigma2_a * float(np.sum(
            np.abs(phi) ** 2
            * np.sum(np.abs(ch.H_rt_hat[k]) ** 2, axis=0).real
        ))
        values[k] = signal / (interference + active_noise + cfg.sigma2_k + 1e-15)
    return values


def sum_rate(ch, theta, phi, C, T) -> float:
    """Convenience wrapper: returns scalar Σ_u R_u."""
    r_u, _, _, _ = compute_rates(ch, theta, phi, C, T)
    return float(np.sum(r_u))


# ─── Active Element Transmit Power  (Eq. 18) ─────────────────────────────────

def active_tx_power_components(ch, C: np.ndarray, T: np.ndarray,
                               phi: np.ndarray) -> np.ndarray:
    """Return Eq. (18) signal powers for the complete BS waveform.

    The columns of ``C`` and ``T`` already carry square-root watts under
    Eq. (16f). Multiplying by ``cfg.PT`` here would count BS power twice.
    """
    transmit_covariance = C @ C.conj().T + T @ T.conj().T
    powers = np.zeros(cfg.La, dtype=float)
    for i in range(cfg.La):
        g_bri = ch.G_bri_hat[i, :]
        incident_power = float(np.real(
            g_bri @ transmit_covariance @ g_bri.conj()
        ))
        powers[i] = abs(phi[i])**2 * max(incident_power, 0.0)
    return powers


def active_tx_power(ch, C: np.ndarray, T: np.ndarray,
                    phi: np.ndarray) -> float:
    """Return aggregate P_ai with amplifier noise added once, per Eq. (18)."""
    if cfg.La == 0 or np.asarray(phi).size == 0:
        return 0.0
    return float(np.sum(active_tx_power_components(ch, C, T, phi))
                 + cfg.sigma2_a)


def active_tx_power_envelope(ch) -> float:
    """Upper-bound Eq. (18) for every allowed ``phi`` and BS beamformer.

    For ``||T||_F^2 <= P_T`` and ``|phi_i| <= alpha_max``, the signal term is
    bounded by ``alpha_max^2 P_T lambda_max(G^H G)``. This is used to screen
    an omitted active-power limit before policy actions exist.
    """
    if cfg.La == 0:
        return 0.0
    G = np.asarray(ch.G_bri_hat, dtype=complex)
    gram = G.conj().T @ G
    largest_eigenvalue = max(float(np.max(np.linalg.eigvalsh(gram)).real), 0.0)
    return float(
        cfg.alpha_max ** 2 * cfg.PT * largest_eigenvalue + cfg.sigma2_a
    )


# ─── Coherence-Interval Corrected Rate  (Eq. 19) ─────────────────────────────

def coherence_corrected_rate(rate_u: np.ndarray, Omega: int) -> np.ndarray:
    """
    R^{(n)}_u = (1 − (2U + L − 1)/Ω) R_u   (Eq. 19)
    Omega : channel training period (number of time slots).
    """
    overhead_slots = 2 * cfg.U + cfg.L - 1
    if isinstance(Omega, bool) or int(Omega) != Omega or Omega < overhead_slots:
        raise ValueError(
            f"Omega must be an integer at least {overhead_slots} slots."
        )
    overhead_frac = overhead_slots / int(Omega)
    return (1 - overhead_frac) * rate_u
