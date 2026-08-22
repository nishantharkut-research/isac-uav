"""
channel_model.py — Rician fading channels with SCEE.

Implements Eq. (7)-(9) from:
  Saikia et al., "Hybrid-RIS Empowered UAV-Assisted ISAC Systems:
  Transfer Learning-Based DRL," IEEE Trans. Commun., 2025.

Key channels (all complex numpy arrays):
  G_br  ∈ ℂ^{L × M}    BS → HRIS  (rows = per-element vectors)
  H_ru  ∈ ℂ^{N × L}    HRIS → UE/TE  (columns = per-element vectors)
  H_b   ∈ ℂ^{N × M}    Direct BS → UE/TE
"""
import numpy as np
from config import cfg


def _normal(rng, shape):
    if rng is None:
        return np.random.randn(*shape)
    return rng.standard_normal(shape)


def _complex_normal(rng, shape):
    return (_normal(rng, shape) + 1j * _normal(rng, shape)) / np.sqrt(2)


# ─── UPA / ULA Steering Vectors ──────────────────────────────────────────────

def upa_steering(Lx: int, Ly: int,
                 phi_bar1: float, phi_bar2: float) -> np.ndarray:
    """
    Uniform Planar Array response  a_R(φ_R, η_R)  (Eq. 9).

    a = a_x ⊗ a_y,  where
    [a_x]_n = exp(-j 2π n d̄_x φ̄_{AoA,R1})
    [a_y]_m = exp(-j 2π m d̄_y φ̄_{AoA,R2})

    Returns: (Lx*Ly,) complex steering vector, ‖a‖² = Lx*Ly
    """
    dx, dy = cfg.d_spacing, cfg.d_spacing
    ns = np.arange(Lx)
    ms = np.arange(Ly)
    ax = np.exp(-1j * 2 * np.pi * ns * dx * phi_bar1)   # (Lx,)
    ay = np.exp(-1j * 2 * np.pi * ms * dy * phi_bar2)   # (Ly,)
    return np.kron(ax, ay)                               # (Lx*Ly,)


def ula_steering(M: int, phi_rad: float) -> np.ndarray:
    """ULA response at BS (transmit array a_{b,h}).  ‖a‖² = M."""
    return np.exp(1j * 2 * np.pi * np.arange(M) * cfg.d_spacing * np.sin(phi_rad))


def hris_partition_indices(total: int, active: int):
    """Return disjoint active/passive indices on one physical HRIS UPA."""
    total = int(total)
    active = int(active)
    if total <= 0 or active < 0 or active > total:
        raise ValueError("HRIS partition requires 0 <= active <= total.")
    if cfg.hris_partition_mode == "paper_rectangular_subarrays_v1":
        active_indices = np.arange(active, dtype=int)
    elif cfg.hris_partition_mode == "uniformly_interleaved_full_upa_v1":
        if active == 0:
            active_indices = np.empty(0, dtype=int)
        else:
            active_indices = np.floor(
                (np.arange(active, dtype=float) + 0.5) * total / active
            ).astype(int)
    else:
        raise ValueError(f"Unknown HRIS partition mode: {cfg.hris_partition_mode}")
    if np.unique(active_indices).size != active:
        raise AssertionError("Interleaved HRIS partition produced duplicate indices.")
    mask = np.ones(total, dtype=bool)
    mask[active_indices] = False
    return active_indices, np.flatnonzero(mask)


def hris_element_coordinates() -> np.ndarray:
    """Return physical element coordinates in the partition-index order."""
    if cfg.hris_partition_mode == "uniformly_interleaved_full_upa_v1":
        if cfg.Lx * cfg.Ly != cfg.L:
            raise ValueError("Interleaved HRIS UPA dimensions do not match L.")
        return np.array(
            [(x, y) for x in range(cfg.Lx) for y in range(cfg.Ly)],
            dtype=float,
        )
    if cfg.hris_partition_mode != "paper_rectangular_subarrays_v1":
        raise ValueError(f"Unknown HRIS partition mode: {cfg.hris_partition_mode}")
    if (
        (cfg.La and cfg.La != cfg.Lax * cfg.Lay)
        or (cfg.Lp and cfg.Lp != cfg.Lpx * cfg.Lpy)
    ):
        raise ValueError("HRIS subarray dimensions do not match La/Lp.")
    active = np.array(
        [(x, y) for x in range(cfg.Lax) for y in range(cfg.Lay)],
        dtype=float,
    ) if cfg.La else np.empty((0, 2), dtype=float)
    passive_x_offset = cfg.Lax if cfg.La else 0
    passive = np.array(
        [
            (passive_x_offset + x, y)
            for x in range(cfg.Lpx)
            for y in range(cfg.Lpy)
        ],
        dtype=float,
    ) if cfg.Lp else np.empty((0, 2), dtype=float)
    coordinates = np.concatenate([active, passive], axis=0)
    if coordinates.shape != (cfg.L, 2):
        raise ValueError("Physical HRIS coordinates do not match L.")
    return coordinates


def active_hris_steering(phi_bar1: float, phi_bar2: float) -> np.ndarray:
    """Steering response at active elements on the one full HRIS aperture."""
    if cfg.hris_partition_mode == "paper_rectangular_subarrays_v1":
        if cfg.La and cfg.La != cfg.Lax * cfg.Lay:
            raise ValueError("Active HRIS dimensions do not match La.")
        return upa_steering(
            cfg.Lax, cfg.Lay, float(phi_bar1), float(phi_bar2)
        ) if cfg.La else np.empty(0, dtype=complex)
    if cfg.Lx * cfg.Ly != cfg.L:
        raise ValueError("Full HRIS UPA dimensions do not match L.")
    active_indices, _passive_indices = hris_partition_indices(cfg.L, cfg.La)
    full_response = upa_steering(
        cfg.Lx, cfg.Ly, float(phi_bar1), float(phi_bar2)
    )
    return full_response[active_indices]


# ─── Path Loss / LoS ─────────────────────────────────────────────────────────

def plos(elev_deg: float) -> float:
    """Return LoS probability under the locked printed-form interpretation.

    The paper prints ``1/(1+b*exp(-a*(phi-b)))`` but does not tabulate
    ``a`` or ``b``. The common urban model prints the same functional form
    with the symbol roles reversed. Both readings are named hypotheses so a
    reproduction never silently swaps constants to improve a curve.
    """
    from ambiguity_hypotheses import get_los_probability_hypothesis

    a, b = cfg.a_env, cfg.b_env
    hypothesis = get_los_probability_hypothesis(
        cfg.los_probability_hypothesis_id
    )
    if hypothesis.identifier == "paper_unnumbered_los_literal_roles":
        return 1.0 / (1.0 + b * np.exp(-a * (elev_deg - b)))
    if hypothesis.identifier == "al_hourani_standard_mapped_roles":
        return 1.0 / (1.0 + a * np.exp(-b * (elev_deg - a)))
    raise AssertionError("Unreachable LoS probability hypothesis")


def path_loss_exponent(los_probability: float) -> float:
    """Return the Eq. (5) exponent using explicit LoS/NLoS endpoints.

    The journal prints ``zeta=P_LoS*zeta_e+zeta_0`` but does not report either
    constant.  The executable profile stores the more interpretable endpoints
    ``zeta_los`` and ``zeta_nlos``.  Therefore the printed constants map to
    ``zeta_e=zeta_los-zeta_nlos`` and ``zeta_0=zeta_nlos``.
    """
    probability = float(np.clip(los_probability, 0.0, 1.0))
    zeta_e = float(cfg.zeta_los - cfg.zeta_nlos)
    zeta_0 = float(cfg.zeta_nlos)
    return probability * zeta_e + zeta_0


def channel_gain(pos_tx: np.ndarray, pos_rx: np.ndarray) -> float:
    """
    ω = ω₀ · d^{−ζ(φ)}
    ζ(φ) = P_LoS(φ)·ζ_e + (1−P_LoS(φ))·ζ_nlos   (Eq. 5 / Sec. II-A)
    pos_tx, pos_rx: 3-D positions in metres.
    """
    horiz = np.linalg.norm(pos_tx[:2] - pos_rx[:2])
    d_3d  = np.linalg.norm(pos_tx - pos_rx)
    d_3d  = max(d_3d, 1.0)
    elev_deg = np.degrees(np.arctan2(abs(pos_tx[2] - pos_rx[2]),
                                     max(horiz, 1e-3)))
    p     = plos(elev_deg)
    zeta  = path_loss_exponent(p)
    return cfg.omega_0 * (d_3d ** (-zeta))


def target_doppler_frequency(pos_uav: np.ndarray,
                             pos_target: np.ndarray,
                             target_velocity: np.ndarray) -> float:
    """Return Eq. (8)'s target Doppler frequency in hertz.

    The paper gives target speed times the arrival-direction cosines.  A target
    velocity vector is available in the executable scene, so its radial
    projection onto the target-to-UAV line of sight is the coordinate-free
    equivalent.  Positive velocity toward the UAV gives positive Doppler.
    """
    pos_uav = np.asarray(pos_uav, dtype=float)
    pos_target = np.asarray(pos_target, dtype=float)
    velocity = np.asarray(target_velocity, dtype=float)
    if pos_uav.shape != (3,) or pos_target.shape != (3,) or velocity.shape != (3,):
        raise ValueError("Doppler geometry and velocity must be three-vectors.")
    if not np.all(np.isfinite(np.concatenate([pos_uav, pos_target, velocity]))):
        raise ValueError("Doppler geometry and velocity must be finite.")
    displacement = pos_uav - pos_target
    distance = float(np.linalg.norm(displacement))
    if distance <= 1e-12:
        return 0.0
    radial_speed = float(np.dot(velocity, displacement / distance))
    return radial_speed / float(cfg.wavelength)


# ─── Individual Channel Generators ───────────────────────────────────────────

def gen_bs_hris(pos_uav: np.ndarray,
                n_elem: int, Lx: int, Ly: int,
                rng=None, *, element_coordinates=None) -> np.ndarray:
    """
    BS → HRIS channel  G_br ∈ ℂ^{n_elem × M}  (Eq. 7).

    g_bri = √ω_g · e^{−j2πd/λ} ·
            [√(ρ/(ρ+1)) · a_R a^H_{b,h}  +  √(1/(ρ+1)) · G̃]_row_i

    Returns: (n_elem, M) complex matrix.
    """
    M = cfg.M
    if n_elem == 0:
        return np.zeros((0, M), dtype=complex)
    rho = cfg.rho_br
    omega_g = channel_gain(pos_uav, cfg.BS_pos)
    d_bu    = np.linalg.norm(pos_uav - cfg.BS_pos)
    phase   = np.exp(-1j * 2 * np.pi * d_bu / cfg.wavelength)

    # HRIS receive UPA response  a_R ∈ ℂ^{n_elem}
    p_norm   = pos_uav / (np.linalg.norm(pos_uav) + 1e-9)
    phi_bar1, phi_bar2 = p_norm[0], p_norm[1]
    if element_coordinates is None:
        a_R = upa_steering(Lx, Ly, phi_bar1, phi_bar2)
    else:
        coordinates = np.asarray(element_coordinates, dtype=float)
        if coordinates.shape != (n_elem, 2):
            raise ValueError("element_coordinates must have shape (n_elem, 2).")
        a_R = np.exp(
            -1j * 2 * np.pi * cfg.d_spacing
            * (coordinates[:, 0] * phi_bar1 + coordinates[:, 1] * phi_bar2)
        )

    # BS ULA transmit response  a_{b,h} ∈ ℂ^M
    phi_bs   = np.arctan2(pos_uav[1] - cfg.BS_pos[1],
                          pos_uav[0] - cfg.BS_pos[0])
    a_bh     = ula_steering(M, phi_bs)                    # (M,)

    G_LoS  = np.outer(a_R, a_bh.conj())                  # (n_elem, M) rank-1
    G_NLoS = _complex_normal(rng, (n_elem, M))

    return (np.sqrt(omega_g) * phase *
            (np.sqrt(rho / (rho + 1)) * G_LoS +
             np.sqrt(1.0  / (rho + 1)) * G_NLoS))


def gen_hris_ground(pos_uav: np.ndarray,
                    pos_gnd: np.ndarray,
                    n_elem: int,
                    rng=None,
                    *,
                    element_coordinates=None,
                    target_velocity=None,
                    time_s: float = 0.0) -> np.ndarray:
    """
    HRIS → UE/TE channel  H_r ∈ ℂ^{N × n_elem}  (Eq. 8 adapted).

    Rician model; columns h_{r,j} are per-element channel vectors.  For a
    moving target, Eq. (8)'s Doppler phase is applied to the LoS term.  UE
    calls omit ``target_velocity`` and therefore have no target Doppler.
    Returns: (N, n_elem) complex matrix.
    """
    N   = cfg.N
    if n_elem == 0:
        return np.zeros((N, 0), dtype=complex)
    rho = cfg.rho_rt
    omega_h = channel_gain(pos_uav, pos_gnd)
    d_uu    = np.linalg.norm(pos_uav - pos_gnd)
    phase   = np.exp(-1j * 2 * np.pi * d_uu / cfg.wavelength)

    # LoS component: outer product of UE and HRIS array responses
    phi_gnd = np.arctan2(pos_uav[1] - pos_gnd[1],
                         pos_uav[0] - pos_gnd[0])
    a_T = np.exp(1j * 2 * np.pi * np.arange(N) * cfg.d_spacing * np.sin(phi_gnd))

    # HRIS transmit steering toward the ground node (user/target-specific direction).
    # Must use direction FROM HRIS (pos_uav) TO ground node (pos_gnd), not the
    # origin-to-UAV direction that was here before (which was identical for all users).
    dir_to_gnd = np.array(pos_gnd, dtype=float) - np.array(pos_uav, dtype=float)
    dir_norm   = dir_to_gnd / (np.linalg.norm(dir_to_gnd) + 1e-9)
    if element_coordinates is None:
        from config import _factor_upa
        Lx_h, Ly_h = _factor_upa(n_elem)
        a_R_gnd = upa_steering(
            Lx_h, Ly_h, dir_norm[0], dir_norm[1]
        )
    else:
        coordinates = np.asarray(element_coordinates, dtype=float)
        if coordinates.shape != (n_elem, 2):
            raise ValueError("element_coordinates must have shape (n_elem, 2).")
        a_R_gnd = np.exp(
            -1j * 2 * np.pi * cfg.d_spacing
            * (coordinates[:, 0] * dir_norm[0]
               + coordinates[:, 1] * dir_norm[1])
        )

    H_LoS  = np.outer(a_T, a_R_gnd.conj())              # (N, n_elem)
    H_NLoS = _complex_normal(rng, (N, n_elem))

    time_s = float(time_s)
    if not np.isfinite(time_s) or time_s < 0.0:
        raise ValueError("time_s must be a finite nonnegative value.")
    doppler_phase = 1.0 + 0.0j
    if target_velocity is not None and time_s > 0.0:
        frequency = target_doppler_frequency(
            pos_uav, pos_gnd, np.asarray(target_velocity, dtype=float)
        )
        doppler_phase = np.exp(1j * 2.0 * np.pi * frequency * time_s)

    return (np.sqrt(omega_h) * phase *
            (np.sqrt(rho / (rho + 1)) * doppler_phase * H_LoS +
             np.sqrt(1.0  / (rho + 1)) * H_NLoS))


def gen_direct(pos_tx: np.ndarray,
               pos_rx: np.ndarray,
               rng=None) -> np.ndarray:
    """
    Direct BS → UE / TE channel  H_b ∈ ℂ^{N × M}.

    The paper includes the direct links but does not specify an additional
    blockage multiplier, so no curve-calibration attenuation is applied.
    """
    N, M = cfg.N, cfg.M
    omega = channel_gain(pos_tx, pos_rx)
    H_NLoS = _complex_normal(rng, (N, M))
    return np.sqrt(omega) * H_NLoS


# ─── SCEE Perturbation ────────────────────────────────────────────────────────

def add_scee(H_hat: np.ndarray, rho_nmse: float, rng=None) -> np.ndarray:
    """
    Gaussian-Kronecker SCEE model (Sec. II-B):
        H = Ĥ + ΔH,   ΔH ~ CN(0, σ²_eff · I)

    The argument is the estimated channel. The returned true channel is formed
    by adding the paper's independent zero-mean error. Its variance is rho times
    estimated-channel power; rho is never treated as watts.
    """
    if rho_nmse <= 0.0:
        return H_hat.copy()
    rho = float(np.clip(rho_nmse, 0.0, 1.0))
    estimate_power = float(np.mean(np.abs(H_hat) ** 2))
    # Page 10 defines rho = E|H-H_hat|^2 / E|H_hat|^2.
    error = _complex_normal(rng, H_hat.shape) * np.sqrt(rho * estimate_power)
    return H_hat + error


# ─── Full Channel Set ─────────────────────────────────────────────────────────

class ChannelSet:
    """
    All channels for one time slot, with SCEE applied.

    Attributes (estimated channels, denoted _hat):
        G_brj_hat  ∈ ℂ^{Lp × M}   BS  → passive HRIS  (estimated)
        G_bri_hat  ∈ ℂ^{La × M}   BS  → active  HRIS  (estimated)
        H_bu_hat   list[U] of ℂ^{N × M}   BS → UE      (estimated)
        H_ru_hat   list[U] of ℂ^{N × Lp}  passive HRIS → UE
        H_bk_hat   list[K] of ℂ^{N × M}   BS → TE      (estimated)
        H_rt_hat   list[K] of ℂ^{N × La}  active  HRIS → TE
    """

    def __init__(self,
                 pos_uav: np.ndarray,
                 pos_ues: list,
                 pos_tes: list,
                 sigma2: float = None,
                 rng=None,
                 seed: int = None,
                 slot_time_s: float = 0.0,
                 target_velocities=None):
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both.")
        def stream(component, index=0):
            if seed is not None:
                sequence = np.random.SeedSequence(
                    [int(seed), int(component), int(index)]
                )
                return np.random.default_rng(sequence)
            return rng

        slot_time_s = float(slot_time_s)
        if not np.isfinite(slot_time_s) or slot_time_s < 0.0:
            raise ValueError("slot_time_s must be a finite nonnegative value.")
        if target_velocities is None:
            velocities = np.zeros((len(pos_tes), 3), dtype=float)
        else:
            velocities = np.asarray(target_velocities, dtype=float)
            if velocities.shape != (len(pos_tes), 3) or not np.all(np.isfinite(velocities)):
                raise ValueError("target_velocities must be a finite K-by-3 array.")

        s2 = sigma2 if sigma2 is not None else cfg.sigma2_brj
        self.scee_nmse = float(np.clip(s2, 0.0, 1.0))
        self.U, self.K = len(pos_ues), len(pos_tes)
        self.slot_time_s = slot_time_s
        self.target_velocities = velocities.copy()

        # ── Estimated-channel generation ─────────────────────────────────────
        # BS → one physical HRIS UPA, then disjoint active/passive slices.
        coordinates = hris_element_coordinates()
        active_indices, passive_indices = hris_partition_indices(cfg.L, cfg.La)
        self.active_element_indices = active_indices.copy()
        self.passive_element_indices = passive_indices.copy()
        G_br_hat = gen_bs_hris(
            pos_uav, cfg.L, cfg.Lx, cfg.Ly, stream(1),
            element_coordinates=coordinates,
        )
        G_brj_hat = G_br_hat[passive_indices]
        G_bri_hat = G_br_hat[active_indices]

        # Direct BS → UE / TE
        H_bu_hat = [
            gen_direct(cfg.BS_pos, p, stream(3, u))
            for u, p in enumerate(pos_ues)
        ]
        H_bk_hat = [
            gen_direct(cfg.BS_pos, p, stream(4, k))
            for k, p in enumerate(pos_tes)
        ]

        # Generate the same full physical aperture for every ground link and
        # retain the subset used by that signal path.
        H_ru_hat = [
            gen_hris_ground(
                pos_uav, p, cfg.L, stream(5, u),
                element_coordinates=coordinates,
            )[
                :, passive_indices
            ]
            for u, p in enumerate(pos_ues)
        ]
        H_rt_hat = [
            gen_hris_ground(
                pos_uav, p, cfg.L, stream(6, k),
                element_coordinates=coordinates,
                target_velocity=velocities[k], time_s=slot_time_s,
            )[:, active_indices]
            for k, p in enumerate(pos_tes)
        ]

        # ── SCEE covariance (Sec. II-B, Gaussian-Kronecker model) ─────────────
        # Eqs. (21)-(25) analytically marginalize SCEE. Keep the estimated
        # channels fixed across NMSE values and carry errors in covariance.
        self.G_brj_hat = G_brj_hat
        self.G_bri_hat = G_bri_hat
        self.H_bu_hat = H_bu_hat
        self.H_bk_hat = H_bk_hat
        self.H_ru_hat = H_ru_hat
        self.H_rt_hat = H_rt_hat

        # ── Store per-channel effective error variances  σ²_eff = ρ · E[|h|²] ─
        def _eff_gain(pos_tx, pos_rx, element_count):
            if int(element_count) == 0 or self.scee_nmse <= 0.0:
                return 0.0
            return self.scee_nmse * channel_gain(pos_tx, pos_rx)

        # Each normalized Rician/NLoS coefficient has theoretical second
        # moment equal to its geometry-dependent channel gain. This preserves
        # the paper's independent CSCG error instead of scaling it by one
        # realized estimate matrix.
        self.sv_bu = [
            _eff_gain(cfg.BS_pos, position, cfg.M * cfg.N)
            for position in pos_ues
        ]
        self.sv_bk = [
            _eff_gain(cfg.BS_pos, position, cfg.M * cfg.N)
            for position in pos_tes
        ]
        self.sv_brj = _eff_gain(cfg.BS_pos, pos_uav, cfg.Lp * cfg.M)
        self.sv_bri = _eff_gain(cfg.BS_pos, pos_uav, cfg.La * cfg.M)
        self.sv_ruj = [
            _eff_gain(pos_uav, position, cfg.Lp * cfg.N)
            for position in pos_ues
        ]
        self.sv_rti = [
            _eff_gain(pos_uav, position, cfg.La * cfg.N)
            for position in pos_tes
        ]

    # ── Convenience: composite channels Eq. after (10) ───────────────────────
    def composite_ue(self, theta: np.ndarray) -> list:
        """
        H̄_u = H_{b,u} + H_{ru} diag(θ) G_{brj}   (passive path)
        theta : (Lp,) complex unit-modulus phase shifts
        Returns list[U] of (N, M) matrices.
        """
        Theta = np.diag(theta)                  # (Lp, Lp)
        return [self.H_bu_hat[u] +
                self.H_ru_hat[u] @ Theta @ self.G_brj_hat
                for u in range(self.U)]

    def composite_te(self, phi: np.ndarray) -> list:
        """
        H̄_k = H_{b,k} + H_{rt} diag(ϕ) G_{bri}   (active path)
        phi : (La,) complex phase-shift × amplification
        Returns list[K] of (N, M) matrices.
        """
        Phi = np.diag(phi)                      # (La, La)
        return [self.H_bk_hat[k] +
                self.H_rt_hat[k] @ Phi @ self.G_bri_hat
                for k in range(self.K)]
