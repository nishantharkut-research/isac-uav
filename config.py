"""
config.py — System, channel, and DRL hyper-parameters.
Matches Tables III & IV of:
  Saikia et al., "Hybrid-RIS Empowered UAV-Assisted ISAC Systems:
  Transfer Learning-Based DRL," IEEE Trans. Commun., 2025.
"""
from contextlib import contextmanager

import numpy as np
from physical_hypotheses import (
    dbm_to_watts,
    dbw_to_watts,  # re-exported for compatibility with existing scripts
    get_power_hypothesis,
    watts_to_dbm,
    watts_to_dbw,
)
from ambiguity_hypotheses import (
    PRIMARY_NOISE_HYPOTHESIS_ID,
    PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID,
    PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID,
    PRIMARY_WFB_HYPOTHESIS_ID,
    PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID,
    PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID,
    PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID,
    PRIMARY_CSI_OBSERVATION_HYPOTHESIS_ID,
    PRIMARY_RATE_MODEL_HYPOTHESIS_ID,
    PRIMARY_CROSS_FUNCTION_INTERFERENCE_HYPOTHESIS_ID,
    get_noise_hypothesis,
    get_wfb_hypothesis,
)

__all__ = [
    "Config", "cfg", "dbm_to_watts", "dbw_to_watts", "watts_to_dbm",
    "watts_to_dbw", "temporary_system_configuration",
]


class Config:
    # The paper does not publish random seeds. This fixed seed is a documented
    # reproduction assumption used by NumPy, Python, PyTorch, and figure scenes.
    random_seed = 2025

    # Table III: simulation parameters. Keep these as the paper baseline.
    M   = 4          # ISAC BS transmit antennas
    N   = 2          # UE / TE receive antennas per node
    U   = 3          # Number of communication users
    K   = 3          # Number of sensing targets
    L   = 40         # Total HRIS reflecting elements  (La + Lp)
    La  = 20         # Active elements  (with signal amplification)
    Lp  = 20         # Passive elements  (unit-modulus phase shifts)

    # UPA sub-array dimensions  (La = Lax*Lay,  Lp = Lpx*Lpy)
    Lax, Lay = 4, 5
    Lpx, Lpy = 4, 5
    # Eq. (9) defines each HRIS response on its own rectangular subarray.
    # Place the disjoint active/passive rectangles next to one another on one
    # physical aperture. The prior interleaved reading remains available as a
    # named sensitivity in channel_model.py.
    Lx, Ly = Lax + Lpx, max(Lay, Lpy)
    hris_partition_mode = "paper_rectangular_subarrays_v1"
    csi_state_representation_id = (
        "finite_horizon_geometry_scale_and_estimated_primitive_channels_v3"
    )
    transition_timing_id = "score_observed_channel_then_move_v1"
    direct_channel_model_id = "shared_pathloss_rayleigh_direct_link_v1"
    capon_covariance_model_id = "expected_covariance_mvdr_proxy_v1"
    active_power_waveform_id = "complete_communication_and_sensing_waveform_v1"

    # Noise powers:  dBm → Watts
    # Table III prints sigma_k^2 but describes it as active-element noise. Keep
    # target-receiver and active-amplifier noise separate so Eqs. (12) and (14)
    # use the intended physical term.
    noise_hypothesis_id = PRIMARY_NOISE_HYPOTHESIS_ID
    _noise_hypothesis = get_noise_hypothesis(noise_hypothesis_id)
    sigma2_u_dBm = _noise_hypothesis.ue_receiver_dbm
    sigma2_k_dBm = _noise_hypothesis.target_receiver_dbm
    sigma2_a_dBm = _noise_hypothesis.active_amplifier_dbm
    sigma2_u = dbm_to_watts(sigma2_u_dBm)
    sigma2_k = dbm_to_watts(sigma2_k_dBm)
    sigma2_a = dbm_to_watts(sigma2_a_dBm)

    # The paper labels P_T only as dB. dBm is the primary hypothesis; dBW is
    # retained as a separately identified sensitivity run.
    power_hypothesis_id = "paper_db_as_dbm"
    _power_hypothesis = get_power_hypothesis(power_hypothesis_id)
    PT_dB = 10.0
    PT_reference = "1 mW"
    PT_display_unit = _power_hypothesis.display_unit
    PT = _power_hypothesis.to_watts(PT_dB)
    PT_dBm = watts_to_dbm(PT)
    PT_dBW = watts_to_dbw(PT)
    power_hypothesis_metadata = _power_hypothesis.as_dict(PT_dB)

    @property
    def sigma2_qa(self):
        """Deprecated compatibility name for active-amplifier noise only."""
        return self.sigma2_a

    @sigma2_qa.setter
    def sigma2_qa(self, value):
        self.sigma2_a = float(value)

    @property
    def sigma2_qa_dBm(self):
        """Deprecated compatibility name for sigma2_a_dBm."""
        return self.sigma2_a_dBm

    @sigma2_qa_dBm.setter
    def sigma2_qa_dBm(self, value):
        self.sigma2_a_dBm = float(value)

    # Active element constraints
    alpha_max_dB = 10        # maximum amplification (dB)
    alpha_max    = 10**(alpha_max_dB / 20)   # linear amplitude

    # Passive-element power threshold  Ψ_max  (Eq. 16c)
    Psi_max = 1.0

    # Minimum sensing rate threshold  R_min  (Eq. 16e)
    R_min   = 0.5            # bits/s/Hz  per sensing target
    # The extension PNG adds a separate UE floor but does not provide its
    # numeric value.  Equal 0.5 floors are the disclosed primary assumption.
    R_u_min = 0.5            # bits/s/Hz per UE, extension only
    # Aggregate active-HRIS transmit-power bound P_ai <= P_a_max (Eq. 16d).
    # The paper states the constraint but does not tabulate a numeric value.
    P_a_max = 10.0            # W; explicit omitted-parameter assumption
    I_a_min = 0.0            # deprecated compatibility alias; not used by Eq. 16d

    # ── Channel model ─────────────────────────────────────────────────────────
    # The paper does not report these four physical constants. They are
    # explicit reproduction assumptions, not values from Tables III/IV.
    fc          = 3e9        # carrier frequency assumption (Hz)
    wavelength  = 3e8 / fc  # λ (m)
    d_spacing   = 0.5        # standard half-wavelength spacing assumption
    omega_0     = 1e-3       # standard -30 dB power gain at d0=1 m assumption

    # Communication / sensing power split.
    comm_power_frac = 0.8     # unreported initialization; feasibility search refines it

    rho_br = 10.0            # Rician K-factor  BS → HRIS
    rho_rt = 5.0             # Rician K-factor  HRIS → target / user

    # Urban LoS probability  P_LoS(φ) = 1/(1+a·exp(−b(φ−a)))  ITU-R urban
    # a=9.61, b=0.16 for dense urban (Al-Hourani et al., GLOBECOM 2014)
    a_env  = 9.61
    b_env  = 0.16
    # The paper's unnumbered LoS formula and the conventional urban formula
    # use opposite symbol roles.
    # The printed equation is primary; the external convention is sensitivity.
    los_probability_hypothesis_id = PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID
    # Endpoint parameterization of the paper's Eq. (5):
    #   zeta(phi) = P_LoS(phi) * zeta_e + zeta_0.
    # Here zeta_0=zeta_nlos and the paper's zeta_e=(zeta_los-zeta_nlos).
    # Storing the physical endpoints avoids treating the paper's zeta_e as a
    # positive LoS exponent, which would reverse the elevation dependence.
    zeta_los  = 2.0          # P_LoS=1 endpoint (assumption)
    zeta_nlos = 3.5          # P_LoS=0 endpoint / paper zeta_0 (assumption)

    # Statistical Channel Estimation Error (SCEE)
    # rho_nmse is the paper's normalized ratio, never an absolute noise power.
    # In channel_model.py the conditional Gaussian construction uses error
    # variance rho/(1+rho) times true-channel power. This makes the paper's
    # ratio E|H-H_hat|^2 / E|H_hat|^2 equal rho.
    rho_nmse    = 0.1        # Fig. 8 baseline-point assumption
    # The paper does not define which coefficients are available for ParCSI.
    # The reproduction therefore exposes a fixed fraction of the iCSI channel
    # features to the agent. It does not tune a second NMSE to force ordering.
    partial_csi_fraction = 0.5
    rho_partial = rho_nmse  # compatibility alias; ParCSI uses observation masking
    sigma2_brj  = rho_nmse   # kept for compatibility; used as ratio, NOT watts
    sigma2_bri  = rho_nmse
    sigma2_ruj  = rho_nmse
    sigma2_rti  = rho_nmse
    sigma2_bu   = rho_nmse
    sigma2_bk   = rho_nmse

    # Algorithm 1 samples inside the time-step loop. The next realization is
    # exposed in the next pCSI state before the corresponding action is scored.
    channel_coherence_hypothesis_id = (
        PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID
    )
    # pCSI/iCSI describe estimation quality, while statistical/instantaneous
    # describe state observability. The paper does not map one axis to the other.
    csi_observation_hypothesis_id = (
        PRIMARY_CSI_OBSERVATION_HYPOTHESIS_ID
    )
    # The paper changes from scalar pCSI rates to matrix SCEE rates between
    # sections. The primary keeps one continuous matrix functional; the
    # literal section split remains a predeclared sensitivity.
    rate_model_hypothesis_id = PRIMARY_RATE_MODEL_HYPOTHESIS_ID
    cross_function_interference_hypothesis_id = (
        PRIMARY_CROSS_FUNCTION_INTERFERENCE_HYPOTHESIS_ID
    )

    # ── Geometry ──────────────────────────────────────────────────────────────
    # Fig. 10 visibly uses a 300 m x 200 m horizontal service area. These
    # limits are figure-derived because Tables III/IV do not list them.
    x_min, x_max = 0.0, 300.0    # (m)
    y_min, y_max = 0.0, 200.0
    z_UAV        = 50.0           # fixed UAV altitude (m)
    BS_pos       = np.array([0.0, 0.0, 0.0])

    # Page 10 states that UEs are selected from a 100 m x 100 m region. Its
    # origin is not reported, so [0, 100] x [0, 100] is an explicit baseline
    # assumption kept separate from the larger Fig. 10 UAV/detection domain.
    ue_x_min, ue_x_max = 0.0, 100.0
    ue_y_min, ue_y_max = 0.0, 100.0

    # Neither target support nor initial horizontal UAV coordinates are
    # tabulated. These named defaults make the implementation choice explicit.
    target_region_hypothesis_id = "uniform_full_service_area_v1"
    target_x_min, target_x_max = x_min, x_max
    target_y_min, target_y_max = y_min, y_max
    uav_start_hypothesis_id = "service_area_center_v1"
    uav_start_x = (x_min + x_max) / 2.0
    uav_start_y = (y_min + y_max) / 2.0

    # The paper states that the sensing targets have low mobility but does not
    # publish a speed. This explicit assumption is used by the seeded target
    # motion model and must not be tuned against any figure.
    target_speed_mps = 1.0
    fig10_plot_interval = 20  # 301 slots -> 16 displayed trajectory samples

    # ── UAV mobility ──────────────────────────────────────────────────────────
    v_max   = 10.0    # m/s maximum speed
    delta_T = 0.1     # s per time slot
    N_slots = 300     # time horizon per episode (Table IV)
    D_max   = v_max * delta_T    # maximum displacement per slot

    # ── Table IV: DRL Configuration ───────────────────────────────────────────
    E_episodes  = 1000          # episodes
    # Fig. 4 visibly plots episodes 0..1400 despite Table IV reporting 1000.
    # Keep the table value for sweeps and expose the figure protocol separately.
    fig4_episodes = 1500
    fig4_plot_interval = 100
    T_steps     = 300           # time steps per episode
    B_batch     = 32            # replay mini-batch size
    lr_actor    = 1e-3          # actor learning rate
    lr_critic   = 1e-3          # critic learning rate
    gamma       = 0.9           # discount factor
    tau         = 0.01          # soft-update rate  τ (Eq. 28-29)
    replay_size = 10_000        # replay buffer size W
    checkpoint_every = 1        # completed episodes; preserves every boundary

    # The learned policy must face the same beam-allocation operator during
    # training and evaluation.  "fast" includes the deterministic SciPy
    # feasibility fallback in environment.py; it is not a weaker test-time
    # physics model.
    policy_allocation_mode = "fast"

    # Exploration noise
    noise_std    = 0.3          # unreported Gaussian exploration assumption
    noise_min    = 0.05         # unreported terminal exploration floor
    noise_decay  = 0.999        # unreported per-episode decay assumption

    # Transfer learning  (Eq. 30)
    # Eq. (30) does not publish a complete Bellman update. The executable
    # interpretation is fingerprinted so checkpoints cannot mix alternatives.
    dtl_transfer_mode = PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID
    dtl_transfer_schedule_id = PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID
    upsilon_0    = 0.7          # initial transfer weight
    upsilon_min  = 0.1          # floor for transfer weight
    upsilon_decay = 0.001       # per-episode decay

    # DTL-WFB removes Capon, but the journal omits the replacement filter.
    wfb_hypothesis_id = PRIMARY_WFB_HYPOTHESIS_ID
    wfb_beamformer_mode = get_wfb_hypothesis(
        wfb_hypothesis_id
    ).beamformer_mode
    # The standalone DDPG curve is compared to DTL-pCSI under the same CSI.
    # The omitted iCSI interpretation is an explicit sensitivity branch.
    ddpg_comparator_csi_hypothesis_id = (
        PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID
    )

    # Section IV requires a related but different source/target task. The paper
    # does not publish its scenes, so distinct deterministic scene seeds are an
    # explicit reproduction assumption.
    source_scene_seed = 2025
    target_scene_seed = 2026

    # AO baseline
    ao_max_iter  = 20
    ao_tol       = 1e-4
    sca_max_iter = 15
    fig6_ao_iter = 10
    fig6_train_episodes = 1000
def _factor_upa(n: int):
    """
    Return (Lx, Ly) such that Lx * Ly == n EXACTLY.
    Finds the most-square factorisation so the UPA is as square as possible.

    Examples
    --------
    10  → (2, 5)   20 → (4, 5)   15 → (3, 5)   30 → (5, 6)
    """
    best = (1, n)
    for lx in range(2, int(n ** 0.5) + 1):
        if n % lx == 0:
            best = (lx, n // lx)
    return best   # (smaller_factor, larger_factor)


# ── Shared singleton ─────────────────────────────────────────────────────────
# Import THIS object in every module:  from config import cfg
# Never do  cfg = Config()  in other files — that breaks L-sweep patching.
cfg = Config()


@contextmanager
def temporary_system_configuration(**overrides):
    """Temporarily apply validated paper-system dimensions and restore them."""
    allowed = {"M", "N", "U", "K", "L", "La", "Lp"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown system dimensions: {sorted(unknown)}")
    dimensions = {
        key: int(overrides.get(key, getattr(cfg, key))) for key in allowed
    }
    if any(dimensions[key] <= 0 for key in ("M", "N", "U", "K", "L")):
        raise ValueError("M, N, U, K, and L must be positive.")
    if "La" in overrides and "Lp" not in overrides:
        dimensions["Lp"] = dimensions["L"] - dimensions["La"]
    elif "Lp" in overrides and "La" not in overrides:
        dimensions["La"] = dimensions["L"] - dimensions["Lp"]
    if dimensions["La"] < 0 or dimensions["Lp"] < 0:
        raise ValueError("La and Lp must be nonnegative.")
    if dimensions["La"] + dimensions["Lp"] != dimensions["L"]:
        raise ValueError("System dimensions require La + Lp == L.")

    fields = (
        "M", "N", "U", "K", "L", "La", "Lp",
        "Lax", "Lay", "Lpx", "Lpy",
        "Lx", "Ly",
    )
    saved = {field: getattr(cfg, field) for field in fields}
    try:
        for key in allowed:
            setattr(cfg, key, dimensions[key])
        cfg.Lax, cfg.Lay = _factor_upa(cfg.La) if cfg.La else (1, 1)
        cfg.Lpx, cfg.Lpy = _factor_upa(cfg.Lp) if cfg.Lp else (1, 1)
        cfg.Lx = (
            (cfg.Lax if cfg.La else 0) + (cfg.Lpx if cfg.Lp else 0)
            if cfg.hris_partition_mode == "paper_rectangular_subarrays_v1"
            else _factor_upa(cfg.L)[0]
        )
        cfg.Ly = (
            max(cfg.Lay if cfg.La else 0, cfg.Lpy if cfg.Lp else 0)
            if cfg.hris_partition_mode == "paper_rectangular_subarrays_v1"
            else _factor_upa(cfg.L)[1]
        )
        yield cfg
    finally:
        for field, value in saved.items():
            setattr(cfg, field, value)
