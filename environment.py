"""
environment.py — MDP environment for HRIS-UAV-ISAC sum-rate maximisation.

Section III of Saikia et al., IEEE Trans. Commun., 2025.

MDP tuple  (S, A, P, R):
  S  –  state vector (with CSI mode)
  A  –  {θ_j, ϕ_i, ι}  (passive phases, active phases+amplification, UAV pos)
  R  -  raw ObjectiveSpec value: communication sum for the paper baseline,
       communication plus sensing sum for the supplied PNG objective
"""
from dataclasses import dataclass

from config import cfg
import numpy as np
from channel_model import ChannelSet
from paper_contract import (
    active_power_penalty,
    cascade_penalty_16c,
    project_uav_step,
)
from research_objectives import ObjectiveSpec, evaluate_objective
from sinr_rate import (
    active_tx_power,
    compute_rates,
    resolve_sensing_beamformer_mode,
    select_feasible_beamformers,
)


@dataclass(frozen=True)
class ControlState:
    theta_phase: np.ndarray
    phi_phase: np.ndarray
    alpha: np.ndarray
    uav_pos: np.ndarray


class InfeasibleChannelError(RuntimeError):
    """Raised when the declared solver cannot establish physical feasibility."""

    def __init__(self, diagnostics):
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "Channel has no feasible control under the declared solver"
        )


def _wrapped_phase_delta(start, end):
    return np.angle(np.exp(1j * (np.asarray(end) - np.asarray(start))))


def interpolate_control_state(previous, proposed, fraction):
    """Interpolate physical controls using shortest circular phase paths."""
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("control interpolation fraction must be in [0, 1]")
    for field in ("theta_phase", "phi_phase", "alpha", "uav_pos"):
        if np.asarray(getattr(previous, field)).shape != np.asarray(
            getattr(proposed, field)
        ).shape:
            raise ValueError(f"control-state shape mismatch for {field}")
    theta = (
        np.asarray(previous.theta_phase, dtype=float)
        + fraction * _wrapped_phase_delta(previous.theta_phase, proposed.theta_phase)
    ) % (2.0 * np.pi)
    phi = (
        np.asarray(previous.phi_phase, dtype=float)
        + fraction * _wrapped_phase_delta(previous.phi_phase, proposed.phi_phase)
    ) % (2.0 * np.pi)
    alpha = (
        np.asarray(previous.alpha, dtype=float)
        + fraction * (
            np.asarray(proposed.alpha, dtype=float)
            - np.asarray(previous.alpha, dtype=float)
        )
    )
    uav = (
        np.asarray(previous.uav_pos, dtype=float)
        + fraction * (
            np.asarray(proposed.uav_pos, dtype=float)
            - np.asarray(previous.uav_pos, dtype=float)
        )
    )
    return ControlState(theta, phi, alpha, uav)




class HRISUAVISACEnv:
    """
    Gym-style environment (no gym dependency).

    Action  a ∈ [-1, 1]^{action_dim}  (normalised, de-mapped internally):
      [0  : Lp)        → passive phase-shifts β^{pas}_j ∈ [0, 2π]
      [Lp : Lp+La)     → active phase-shifts  β^{ac}_i ∈ [0, 2π]
      [Lp+La : Lp+2La) → amplifications α_i ∈ [0, α_max]
      [Lp+2La : +2)    → UAV horizontal displacement (Δx, Δy) ∈ [-D_max, D_max]

    State  s  (with channel CSI):
      [Re(e^{jθ}), Im(e^{jθ}),          # 2 Lp  passive phase shifts
       Re(ϕ),      Im(ϕ),                # 2 La  active phase shifts
       pos_UAV / pos_max,                 # 3    UAV position (normalised)
       Re(C,T), Im(C,T),                  # precoding matrices
       Re(H_u,H_k), Im(H_u,H_k)]          # full composite channel matrices
    """

    def __init__(self, sigma2_scee: float = None, seed: int = None,
                 use_capon=None, sensing_rate_min: float = None,
                 sensing_only_sinr_min: float = 0.0,
                 csi_observation_fraction: float = 1.0,
                 observe_instantaneous_csi: bool = None,
                 objective_spec: ObjectiveSpec = None,
                 allocation_mode: str = "fast",
                 sensing_beamformer_mode=None):
        self.sigma2    = sigma2_scee
        self.seed = None if seed is None else int(seed)
        self.sensing_beamformer_mode = resolve_sensing_beamformer_mode(
            sensing_beamformer_mode, use_capon
        )
        self.use_capon = self.sensing_beamformer_mode == "capon"
        if objective_spec is None:
            threshold = cfg.R_min if sensing_rate_min is None else sensing_rate_min
            objective_spec = ObjectiveSpec.paper(target_rate_min=threshold)
        elif not isinstance(objective_spec, ObjectiveSpec):
            objective_spec = ObjectiveSpec.from_mapping(objective_spec)
        if sensing_rate_min is not None and not np.isclose(
            float(sensing_rate_min), objective_spec.target_rate_min
        ):
            raise ValueError(
                "sensing_rate_min conflicts with objective_spec.target_rate_min"
            )
        self.objective_spec = objective_spec
        self.sensing_rate_min = objective_spec.target_rate_min
        self.sensing_only_sinr_min = float(sensing_only_sinr_min)
        effective_scee = cfg.rho_nmse if sigma2_scee is None else sigma2_scee
        self.observe_instantaneous_csi = (
            bool(float(effective_scee) <= 0.0)
            if observe_instantaneous_csi is None
            else bool(observe_instantaneous_csi)
        )
        self.csi_observation_fraction = float(csi_observation_fraction)
        if allocation_mode not in {"fast", "scipy"}:
            raise ValueError("allocation_mode must be 'fast' or 'scipy'.")
        self.allocation_mode = allocation_mode
        if not 0.0 < self.csi_observation_fraction <= 1.0:
            raise ValueError("csi_observation_fraction must be in (0, 1].")
        self._rng      = np.random.default_rng(seed)
        self._channel_rng = np.random.default_rng(seed)

        self.action_dim = cfg.Lp + cfg.La + cfg.La + 2
        channel_feature_count = 2 * cfg.N * cfg.M * (cfg.U + cfg.K)
        self.state_dim = (
            2 * cfg.Lp + 2 * cfg.La + 3
            + 2 * cfg.M * (cfg.U + cfg.K)
            + (channel_feature_count if self.observe_instantaneous_csi else 0)
        )
        visible_count = int(round(
            self.csi_observation_fraction * channel_feature_count
        )) if self.observe_instantaneous_csi else 0
        self.csi_observation_mask = np.zeros(channel_feature_count, dtype=bool)
        mask_order = np.random.default_rng(17_291).permutation(channel_feature_count)
        self.csi_observation_mask[mask_order[:visible_count]] = True

        self._init_scene()
        self.uav_pos = self._uav_start.copy()
        self._episode_channel_seed = None
        self.reset()

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _init_scene(self):
        """Randomly place U users and K targets in the service area."""
        def rand_gnd(n, low, high):
            return [self._rng.uniform(
                        [low[0], low[1], 0.0],
                        [high[0], high[1], 0.0])
                    for _ in range(n)]
        self.pos_ues = rand_gnd(
            cfg.U,
            (cfg.ue_x_min, cfg.ue_y_min),
            (cfg.ue_x_max, cfg.ue_y_max),
        )
        self.pos_tes = rand_gnd(
            cfg.K,
            (cfg.x_min, cfg.y_min),
            (cfg.x_max, cfg.y_max),
        )
        self._tes_initial = [position.copy() for position in self.pos_tes]
        target_angles = self._rng.uniform(0.0, 2.0 * np.pi, cfg.K)
        self._tes_velocity = np.column_stack([
            cfg.target_speed_mps * np.cos(target_angles),
            cfg.target_speed_mps * np.sin(target_angles),
            np.zeros(cfg.K),
        ])
        self._tes_velocity_initial = self._tes_velocity.copy()
        self._uav_start = np.array([
            (cfg.x_min + cfg.x_max) / 2,
            (cfg.y_min + cfg.y_max) / 2,
            cfg.z_UAV
        ], dtype=float)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Start a new episode; return initial state.

        User/target geometry returns to the seeded initial scene. A fresh
        latent channel realization is sampled at reset; the declared coherence
        hypothesis controls whether it is reused after movement steps.
        """
        self.t       = 0
        self.uav_pos = self._uav_start.copy()
        self.pos_tes = [position.copy() for position in self._tes_initial]
        self._tes_velocity = self._tes_velocity_initial.copy()
        self._episode_channel_seed = int(
            self._channel_rng.integers(0, 2**32 - 1)
        )
        self._refresh_channel(self._episode_channel_seed)

        # Canonical initial HRIS configuration (all phases = 0, half-power active).
        self.theta_phase = np.zeros(cfg.Lp)
        self.phi_phase   = np.zeros(cfg.La)
        self.alpha       = np.full(cfg.La, cfg.alpha_max / 2.0)

        self.last_sum_rate = 0.0

        # Recompute _ch_scale for canonical phases on the FIXED channel.
        theta_c = np.exp(1j * self.theta_phase)
        phi_c   = self.alpha * np.exp(1j * self.phi_phase)
        H_bar_u = self.ch.composite_ue(theta_c)
        H_bar_k = self.ch.composite_te(phi_c)
        h_all = np.concatenate([H.ravel() for H in H_bar_u + H_bar_k])
        self._ch_scale = (float(np.linalg.norm(h_all))
                          / np.sqrt(cfg.N * cfg.M * (cfg.U + cfg.K)) + 1e-12)
        self.C, self.T, self.beam_info = select_feasible_beamformers(
            self.ch, theta_c, phi_c,
            sensing_rate_min=self.sensing_rate_min,
            sensing_only_sinr_min=self.sensing_only_sinr_min,
            sensing_beamformer_mode=self.sensing_beamformer_mode,
            objective_spec=self.objective_spec,
            allocation_mode=self.allocation_mode,
        )
        assessment = self._assess_current_solution(
            self.C, self.T, self.beam_info
        )
        # Reset has no executed action or reward. Keep its feasibility explicit;
        # the first transition may still find a feasible proposed control. No
        # infeasible control is ever returned by step().
        self.initial_physical_feasible = bool(assessment["physical_feasible"])
        self.initial_feasibility_diagnostics = self._assessment_diagnostics(
            assessment
        )
        return self._build_state()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """
        Apply action, advance channel, compute reward.

        Returns
        -------
        next_state : (state_dim,) float32
        reward     : float  - raw configured objective value; never a penalty
        done       : bool
        info       : dict   — detailed metrics
        """
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_dim,) or not np.all(np.isfinite(action)):
            raise ValueError(
                f"action must contain {self.action_dim} finite values"
            )
        proposed_action = np.clip(action, -1.0, 1.0)
        previous_controls = self._snapshot_controls()
        proposed_controls = self._controls_from_action(
            previous_controls, proposed_action
        )
        self._advance_targets()
        slot_time_s = (self.t + 1) * cfg.delta_T
        from ambiguity_hypotheses import get_channel_coherence_hypothesis

        coherence = get_channel_coherence_hypothesis(
            cfg.channel_coherence_hypothesis_id
        )
        # This realization was exposed by reset() or the preceding step. The
        # selected action must be evaluated on the channel present in its state.
        channel_seed = int(self._episode_channel_seed)
        previous_powers = self.beam_info.get("stream_powers")
        controls, assessment, projection_fraction = self._find_feasible_controls(
            previous_controls,
            proposed_controls,
            channel_seed,
            previous_powers,
            slot_time_s,
        )
        self._apply_controls(controls)
        trajectory = self._trajectory_diagnostics(controls, previous_controls)
        self.ch = assessment["channel"]
        self.C = assessment["C"]
        self.T = assessment["T"]
        self.beam_info = assessment["beam_info"]

        rate_u = assessment["rate_u"]
        rate_k = assessment["rate_k"]
        sinr_u = assessment["sinr_u"]
        sinr_k = assessment["sinr_k"]
        objective = assessment["objective"]
        beam_info = assessment["beam_info"]
        sum_r = assessment["communication_sum_rate"]
        sum_sensing = assessment["sensing_sum_rate"]
        joint_rate = sum_r + sum_sensing
        physical_constraints_feasible = assessment["physical_feasible"]
        reward = float(objective.value)
        executed_action = self._action_from_controls(
            previous_controls, controls
        )

        self.last_sum_rate = sum_r
        self.t += 1
        done    = self.t >= cfg.N_slots

        if coherence.resample_each_slot and not done:
            self._episode_channel_seed = int(
                self._channel_rng.integers(0, 2**32 - 1)
            )
            self._refresh_channel(
                self._episode_channel_seed,
                slot_time_s=self.t * cfg.delta_T,
            )

        return (self._build_state(),
                reward,
                done,
                dict(rate_u=rate_u, rate_k=rate_k,
                     sinr_u=sinr_u, sinr_k=sinr_k,
                     sum_rate=sum_r, sum_sensing=sum_sensing,
                     communication_sum_rate=sum_r,
                     sensing_sum_rate=sum_sensing,
                     joint_rate=joint_rate,
                     objective=objective.as_dict(),
                     objective_value=objective.value,
                     comm_power_frac=beam_info["comm_power_frac"],
                     beamformer_score=beam_info["beamformer_score"],
                     allocation_fallback_used=bool(
                         beam_info.get("allocation_fallback_used", False)
                     ),
                     physical_constraints_feasible=physical_constraints_feasible,
                     sensing_constraint_feasible=beam_info["sensing_constraint_feasible"],
                     sensing_rate_deficit=beam_info["sensing_rate_deficit"],
                     sensing_only_sinr_deficit=beam_info[
                         "sensing_only_sinr_deficit"
                     ],
                     cascade_penalty=assessment["cascade_penalty"],
                     active_power_penalty=assessment["active_power_penalty"],
                     active_power_ok=assessment["active_power_ok"],
                     sensing_ok=assessment["sensing_ok"],
                     constr_16c_ok=assessment["cascade_ok"],
                     cascade_pwr_16c=assessment["cascade_power"],
                     P_ai=assessment["active_power"],
                     proposed_action=proposed_action.copy(),
                     executed_action=executed_action,
                     action_projection_fraction=projection_fraction,
                     action_was_projected=bool(projection_fraction < 1.0 - 1e-12),
                     projection_source=assessment.get(
                         "projection_source", "externally_selected"
                     ),
                     projection_channel_seed=channel_seed,
                     channel_coherence_hypothesis_id=coherence.identifier,
                     channel_time_s=slot_time_s,
                     uav_bounds_ok=assessment.get(
                         "uav_bounds_ok", trajectory["uav_bounds_ok"]
                     ),
                     uav_step_ok=assessment.get(
                         "uav_step_ok", trajectory["uav_step_ok"]
                     ),
                     uav_return_reachable=assessment.get(
                         "uav_return_reachable",
                         trajectory["uav_return_reachable"],
                     ),
                     uav_closure_ok=assessment.get(
                         "uav_closure_ok", trajectory["uav_closure_ok"]
                     ),
                     uav_pos=self.uav_pos.copy()))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _refresh_channel(self, channel_seed=None, slot_time_s=None):
        """Sample a channel, optionally reusing one fixed small-scale seed."""
        if channel_seed is None:
            channel_seed = int(self._channel_rng.integers(0, 2**32 - 1))
        if slot_time_s is None:
            slot_time_s = getattr(self, "t", 0) * cfg.delta_T
        self.ch = ChannelSet(
            self.uav_pos, self.pos_ues, self.pos_tes, self.sigma2,
            seed=channel_seed, slot_time_s=slot_time_s,
            target_velocities=self._tes_velocity)
        return int(channel_seed)

    def _snapshot_controls(self):
        return ControlState(
            self.theta_phase.copy(),
            self.phi_phase.copy(),
            self.alpha.copy(),
            self.uav_pos.copy(),
        )

    def _controls_from_action(self, previous, action):
        index = 0
        theta = (action[index:index + cfg.Lp] + 1.0) * np.pi
        index += cfg.Lp
        phi = (action[index:index + cfg.La] + 1.0) * np.pi
        index += cfg.La
        alpha = np.clip(
            ((action[index:index + cfg.La] + 1.0) / 2.0) * cfg.alpha_max,
            0.0,
            cfg.alpha_max,
        )
        index += cfg.La
        delta_xy = action[index:index + 2] * cfg.D_max
        uav = previous.uav_pos.copy()
        uav[:2] = project_uav_step(
            current_xy=previous.uav_pos[:2],
            proposed_delta_xy=delta_xy,
            start_xy=self._uav_start[:2],
            slot_index=self.t,
            n_slots=cfg.N_slots,
            d_max=cfg.D_max,
            bounds=([cfg.x_min, cfg.y_min], [cfg.x_max, cfg.y_max]),
        )
        return ControlState(theta, phi, alpha, uav)

    def _apply_controls(self, controls):
        self.theta_phase = np.asarray(controls.theta_phase, dtype=float).copy()
        self.phi_phase = np.asarray(controls.phi_phase, dtype=float).copy()
        self.alpha = np.asarray(controls.alpha, dtype=float).copy()
        self.uav_pos = np.asarray(controls.uav_pos, dtype=float).copy()

    def _action_from_controls(self, previous, executed):
        theta = np.asarray(executed.theta_phase) / np.pi - 1.0
        phi = np.asarray(executed.phi_phase) / np.pi - 1.0
        alpha = 2.0 * np.asarray(executed.alpha) / cfg.alpha_max - 1.0
        delta = (
            np.asarray(executed.uav_pos[:2]) - np.asarray(previous.uav_pos[:2])
        ) / cfg.D_max
        return np.clip(np.concatenate([theta, phi, alpha, delta]), -1.0, 1.0)

    def _evaluate_controls(self, controls, channel_seed, previous_powers,
                           slot_time_s, previous_controls=None):
        self._apply_controls(controls)
        self._refresh_channel(channel_seed, slot_time_s=slot_time_s)
        theta = np.exp(1j * self.theta_phase)
        phi = self.alpha * np.exp(1j * self.phi_phase)
        C, T, beam_info = select_feasible_beamformers(
            self.ch,
            theta,
            phi,
            sensing_rate_min=self.sensing_rate_min,
            sensing_only_sinr_min=self.sensing_only_sinr_min,
            sensing_beamformer_mode=self.sensing_beamformer_mode,
            objective_spec=self.objective_spec,
            allocation_mode=self.allocation_mode,
            previous_powers=previous_powers,
        )
        assessment = self._assess_current_solution(C, T, beam_info)
        if (
            self.allocation_mode == "fast"
            and not assessment["physical_feasible"]
            and assessment["cascade_ok"]
        ):
            C, T, beam_info = select_feasible_beamformers(
                self.ch,
                theta,
                phi,
                sensing_rate_min=self.sensing_rate_min,
                sensing_only_sinr_min=self.sensing_only_sinr_min,
                sensing_beamformer_mode=self.sensing_beamformer_mode,
                objective_spec=self.objective_spec,
                allocation_mode="scipy",
                previous_powers=beam_info.get(
                    "stream_powers", previous_powers
                ),
            )
            assessment = self._assess_current_solution(C, T, beam_info)
            beam_info["allocation_fallback_used"] = True
        else:
            beam_info["allocation_fallback_used"] = False
        trajectory = self._trajectory_diagnostics(controls, previous_controls)
        assessment.update(trajectory)
        assessment["physical_feasible"] = bool(
            assessment["physical_feasible"] and trajectory["uav_feasible"]
        )
        return assessment

    def _trajectory_diagnostics(self, controls, previous_controls):
        """Check Eqs. (1)-(2) for one candidate transition."""
        position = np.asarray(controls.uav_pos, dtype=float)
        tolerance = 1e-9
        bounds_ok = bool(
            position.shape == (3,)
            and np.all(np.isfinite(position))
            and cfg.x_min - tolerance <= position[0] <= cfg.x_max + tolerance
            and cfg.y_min - tolerance <= position[1] <= cfg.y_max + tolerance
            and abs(position[2] - cfg.z_UAV) <= tolerance
        )
        if previous_controls is None:
            step_ok = True
        else:
            previous_position = np.asarray(previous_controls.uav_pos, dtype=float)
            step_ok = bool(
                previous_position.shape == (3,)
                and np.all(np.isfinite(previous_position))
                and np.linalg.norm(position[:2] - previous_position[:2])
                <= cfg.D_max + tolerance
            )

        distance_to_start = float(
            np.linalg.norm(position[:2] - self._uav_start[:2])
        )
        remaining_steps = max(0, cfg.N_slots - self.t - 1)
        return_reachable = bool(
            distance_to_start <= remaining_steps * cfg.D_max + tolerance
        )
        closure_ok = bool(
            self.t < cfg.N_slots - 1 or distance_to_start <= tolerance
        )
        return {
            "uav_bounds_ok": bounds_ok,
            "uav_step_ok": step_ok,
            "uav_return_reachable": return_reachable,
            "uav_closure_ok": closure_ok,
            "uav_feasible": bool(
                bounds_ok and step_ok and return_reachable and closure_ok
            ),
        }

    def _assess_current_solution(self, C, T, beam_info):
        """Evaluate objective and every hard constraint on the current channel."""
        theta = np.exp(1j * self.theta_phase)
        phi = self.alpha * np.exp(1j * self.phi_phase)
        rate_u, rate_k, sinr_u, sinr_k = compute_rates(
            self.ch, theta, phi, C, T
        )
        objective = evaluate_objective(self.objective_spec, rate_u, rate_k)
        cascade_penalty, cascade_ok, cascade_power = cascade_penalty_16c(
            self.ch, theta, cfg.Psi_max
        )
        active_power = active_tx_power(self.ch, T, phi)
        active_penalty, active_ok = active_power_penalty(
            active_power, cfg.P_a_max
        )
        allocation_ok = bool(beam_info.get(
            "allocation_feasible",
            beam_info.get("sensing_constraint_feasible", False),
        ))
        physical_feasible = bool(
            allocation_ok
            and objective.rate_constraints_feasible
            and beam_info.get("sensing_constraint_feasible", False)
            and cascade_ok
            and active_ok
        )
        return {
            "channel": self.ch,
            "C": C,
            "T": T,
            "beam_info": beam_info,
            "rate_u": rate_u,
            "rate_k": rate_k,
            "sinr_u": sinr_u,
            "sinr_k": sinr_k,
            "objective": objective,
            "communication_sum_rate": float(np.sum(rate_u)),
            "sensing_sum_rate": float(np.sum(rate_k)),
            "sensing_ok": bool(np.all(rate_k >= self.sensing_rate_min)),
            "cascade_penalty": float(cascade_penalty),
            "cascade_ok": bool(cascade_ok),
            "cascade_power": float(cascade_power),
            "active_power": float(active_power),
            "active_power_penalty": float(active_penalty),
            "active_power_ok": bool(active_ok),
            "physical_feasible": physical_feasible,
        }

    @staticmethod
    def _assessment_diagnostics(assessment):
        beam = assessment["beam_info"]
        return {
            "physical_feasible": assessment["physical_feasible"],
            "allocation_feasible": bool(beam.get("allocation_feasible", False)),
            "allocation_normalized_deficit": float(
                beam.get("allocation_normalized_deficit", np.inf)
            ),
            "sensing_rate_deficit": float(beam.get("sensing_rate_deficit", np.inf)),
            "sensing_only_sinr_deficit": float(
                beam.get("sensing_only_sinr_deficit", np.inf)
            ),
            "cascade_ok": assessment["cascade_ok"],
            "active_power_ok": assessment["active_power_ok"],
            "uav_bounds_ok": bool(assessment.get("uav_bounds_ok", False)),
            "uav_step_ok": bool(assessment.get("uav_step_ok", False)),
            "uav_return_reachable": bool(
                assessment.get("uav_return_reachable", False)
            ),
            "uav_closure_ok": bool(assessment.get("uav_closure_ok", False)),
        }

    def _deterministic_recovery_candidates(self, previous, proposed, channel_seed):
        """Generate bounded same-channel controls when both endpoints fail."""
        rng = np.random.default_rng(
            np.random.SeedSequence([int(channel_seed), 73_119])
        )
        phase_templates = (
            (previous.theta_phase, previous.phi_phase),
            (proposed.theta_phase, proposed.phi_phase),
            (np.zeros(cfg.Lp), np.zeros(cfg.La)),
            (
                rng.uniform(0.0, 2.0 * np.pi, cfg.Lp),
                rng.uniform(0.0, 2.0 * np.pi, cfg.La),
            ),
        )
        gain_templates = (
            np.zeros(cfg.La),
            np.full(cfg.La, cfg.alpha_max / 2.0),
            np.full(cfg.La, cfg.alpha_max),
        )
        candidates = []
        for theta_phase, phi_phase in phase_templates:
            for alpha in gain_templates:
                candidates.append(ControlState(
                    np.asarray(theta_phase, dtype=float).copy(),
                    np.asarray(phi_phase, dtype=float).copy(),
                    np.asarray(alpha, dtype=float).copy(),
                    np.asarray(proposed.uav_pos, dtype=float).copy(),
                ))
        return candidates

    @staticmethod
    def _control_distance(left, right):
        def mean_square(values):
            values = np.asarray(values, dtype=float)
            return 0.0 if values.size == 0 else float(np.mean(values ** 2))

        theta = mean_square(_wrapped_phase_delta(
            left.theta_phase, right.theta_phase
        ))
        phi = mean_square(_wrapped_phase_delta(
            left.phi_phase, right.phi_phase
        ))
        alpha = mean_square(
            np.asarray(left.alpha) - np.asarray(right.alpha)
        ) / max(cfg.alpha_max ** 2, 1e-12)
        uav = np.sum(
            (np.asarray(left.uav_pos[:2]) - np.asarray(right.uav_pos[:2])) ** 2
        ) / max(cfg.D_max ** 2, 1e-12)
        return float(theta / np.pi ** 2 + phi / np.pi ** 2 + alpha + uav)

    def _find_feasible_controls(self, previous, proposed, channel_seed,
                                previous_powers, slot_time_s):
        transition_origin = previous
        proposal = self._evaluate_controls(
            proposed, channel_seed, previous_powers, slot_time_s,
            transition_origin,
        )
        if proposal["physical_feasible"]:
            proposal["projection_source"] = "proposed"
            return proposed, proposal, 1.0

        fallback = self._evaluate_controls(
            previous, channel_seed, previous_powers, slot_time_s,
            transition_origin,
        )
        projection_source = "previous_control"
        if not fallback["physical_feasible"]:
            feasible_recoveries = []
            recovery_diagnostics = []
            for candidate in self._deterministic_recovery_candidates(
                previous, proposed, channel_seed
            ):
                assessment = self._evaluate_controls(
                    candidate, channel_seed, previous_powers, slot_time_s,
                    transition_origin,
                )
                recovery_diagnostics.append(
                    self._assessment_diagnostics(assessment)
                )
                if assessment["physical_feasible"]:
                    feasible_recoveries.append((
                        self._control_distance(candidate, proposed),
                        -float(assessment["objective"].value),
                        candidate,
                        assessment,
                    ))
            if not feasible_recoveries:
                raise InfeasibleChannelError({
                    "reason": "deterministic_projection_search_exhausted",
                    "channel_seed": int(channel_seed),
                    "proposal": self._assessment_diagnostics(proposal),
                    "previous_control": self._assessment_diagnostics(fallback),
                    "recovery_candidates": recovery_diagnostics,
                })
            _, _, previous, fallback = min(
                feasible_recoveries, key=lambda item: (item[0], item[1])
            )
            projection_source = "deterministic_recovery"

        # Rate and active-power feasibility are non-convex in HRIS phase.
        # First scan the path from the proposal toward the known-feasible
        # fallback, then refine the first sampled feasible interval. A plain
        # binary search from zero can miss a feasible island at a larger step.
        best_controls, best_assessment = previous, fallback
        lower = 0.0
        upper = 1.0
        for fraction in np.arange(7, 0, -1, dtype=float) / 8.0:
            candidate = interpolate_control_state(previous, proposed, fraction)
            assessment = self._evaluate_controls(
                candidate, channel_seed, previous_powers, slot_time_s,
                transition_origin,
            )
            if assessment["physical_feasible"]:
                lower = fraction
                best_controls, best_assessment = candidate, assessment
                upper = min(1.0, fraction + 1.0 / 8.0)
                break

        if lower == 0.0:
            best_assessment["projection_source"] = projection_source
            return best_controls, best_assessment, 0.0

        for _ in range(4):
            fraction = (lower + upper) / 2.0
            candidate = interpolate_control_state(previous, proposed, fraction)
            assessment = self._evaluate_controls(
                candidate, channel_seed, previous_powers, slot_time_s,
                transition_origin,
            )
            if assessment["physical_feasible"]:
                lower = fraction
                best_controls, best_assessment = candidate, assessment
            else:
                upper = fraction
        best_assessment["projection_source"] = projection_source
        return best_controls, best_assessment, float(lower)

    def checkpoint_state(self) -> dict:
        """Capture the episode-boundary state and both environment RNG streams."""
        return {
            "schema_version": 1,
            "episode_boundary": bool(self.t in {0, cfg.N_slots}),
            "objective_spec": self.objective_spec.as_dict(),
            "allocation_mode": self.allocation_mode,
            "sensing_beamformer_mode": self.sensing_beamformer_mode,
            "channel_coherence_hypothesis_id":
                cfg.channel_coherence_hypothesis_id,
            "cross_function_interference_hypothesis_id":
                cfg.cross_function_interference_hypothesis_id,
            "episode_channel_seed": int(self._episode_channel_seed),
            "t": int(self.t),
            "uav_pos": self.uav_pos.copy(),
            "uav_start": self._uav_start.copy(),
            "pos_ues": [position.copy() for position in self.pos_ues],
            "pos_tes": [position.copy() for position in self.pos_tes],
            "tes_initial": [position.copy() for position in self._tes_initial],
            "tes_velocity": self._tes_velocity.copy(),
            "tes_velocity_initial": self._tes_velocity_initial.copy(),
            "theta_phase": self.theta_phase.copy(),
            "phi_phase": self.phi_phase.copy(),
            "alpha": self.alpha.copy(),
            "last_sum_rate": float(self.last_sum_rate),
            "rng_state": self._rng.bit_generator.state,
            "channel_rng_state": self._channel_rng.bit_generator.state,
        }

    def restore_checkpoint_state(self, state: dict):
        """Restore state saved after an episode without consuming either RNG."""
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported environment checkpoint schema.")
        if state.get("objective_spec") != self.objective_spec.as_dict():
            raise ValueError("Environment objective mismatch in checkpoint.")
        if state.get("allocation_mode") != self.allocation_mode:
            raise ValueError("Environment allocation mode mismatch in checkpoint.")
        if state.get("sensing_beamformer_mode") != self.sensing_beamformer_mode:
            raise ValueError(
                "Environment sensing-beamformer mode mismatch in checkpoint."
            )
        if state.get("channel_coherence_hypothesis_id") != (
            cfg.channel_coherence_hypothesis_id
        ):
            raise ValueError(
                "Environment channel-coherence hypothesis mismatch in checkpoint."
            )
        if state.get("cross_function_interference_hypothesis_id") != (
            cfg.cross_function_interference_hypothesis_id
        ):
            raise ValueError(
                "Environment cross-function interference hypothesis mismatch "
                "in checkpoint."
            )
        if state.get("episode_boundary") is not True:
            raise ValueError("Environment checkpoint is not at an episode boundary.")

        t = int(state.get("t", -1))
        if t not in {0, cfg.N_slots}:
            raise ValueError("Environment checkpoint is not at an episode boundary.")

        def array(name, expected_shape):
            try:
                value = np.asarray(state[name], dtype=float).copy()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid environment checkpoint field: {name}."
                ) from exc
            if value.shape != expected_shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid environment checkpoint field: {name}.")
            return value

        def vectors(name, expected_count):
            try:
                values = [np.asarray(value, dtype=float).copy() for value in state[name]]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid environment checkpoint field: {name}."
                ) from exc
            if (
                len(values) != expected_count
                or any(value.shape != (3,) for value in values)
                or any(not np.all(np.isfinite(value)) for value in values)
            ):
                raise ValueError(f"Invalid environment checkpoint field: {name}.")
            return values

        uav_pos = array("uav_pos", (3,))
        uav_start = array("uav_start", (3,))
        pos_ues = vectors("pos_ues", cfg.U)
        pos_tes = vectors("pos_tes", cfg.K)
        tes_initial = vectors("tes_initial", cfg.K)
        tes_velocity = array("tes_velocity", (cfg.K, 3))
        tes_velocity_initial = array("tes_velocity_initial", (cfg.K, 3))
        theta_phase = array("theta_phase", (cfg.Lp,))
        phi_phase = array("phi_phase", (cfg.La,))
        alpha = array("alpha", (cfg.La,))
        if np.any(alpha < 0.0) or np.any(alpha > cfg.alpha_max):
            raise ValueError("Invalid environment checkpoint field: alpha.")
        last_sum_rate = float(state["last_sum_rate"])
        if not np.isfinite(last_sum_rate):
            raise ValueError("Invalid environment checkpoint field: last_sum_rate.")
        episode_channel_seed = state.get("episode_channel_seed")
        if (
            isinstance(episode_channel_seed, bool)
            or not isinstance(episode_channel_seed, (int, np.integer))
            or not 0 <= int(episode_channel_seed) < 2**32
        ):
            raise ValueError(
                "Invalid environment checkpoint field: episode_channel_seed."
            )

        self.t = t
        self.uav_pos = uav_pos
        self._uav_start = uav_start
        self.pos_ues = pos_ues
        self.pos_tes = pos_tes
        self._tes_initial = tes_initial
        self._tes_velocity = tes_velocity
        self._tes_velocity_initial = tes_velocity_initial
        self.theta_phase = theta_phase
        self.phi_phase = phi_phase
        self.alpha = alpha
        self.last_sum_rate = last_sum_rate
        self._episode_channel_seed = int(episode_channel_seed)
        self._rng.bit_generator.state = state["rng_state"]
        self._channel_rng.bit_generator.state = state["channel_rng_state"]

    def _advance_targets(self):
        """Advance low-mobility targets and reflect them at area boundaries."""
        low = np.array([cfg.x_min, cfg.y_min], dtype=float)
        high = np.array([cfg.x_max, cfg.y_max], dtype=float)
        for index, position in enumerate(self.pos_tes):
            next_xy = position[:2] + self._tes_velocity[index, :2] * cfg.delta_T
            for axis in range(2):
                if next_xy[axis] < low[axis]:
                    next_xy[axis] = 2.0 * low[axis] - next_xy[axis]
                    self._tes_velocity[index, axis] *= -1.0
                elif next_xy[axis] > high[axis]:
                    next_xy[axis] = 2.0 * high[axis] - next_xy[axis]
                    self._tes_velocity[index, axis] *= -1.0
            position[:2] = np.clip(next_xy, low, high)
            position[2] = 0.0

    def _build_state(self) -> np.ndarray:
        """
        State vector (Sec. III):
          s = [Re/Im of passive & active phase vectors,
               normalised UAV position,
               Re/Im of C and T,
               Re/Im of full per-user and per-target composite channels]

        Instantaneous CSI adds effective channel coefficients. Statistical CSI
        contains phases, UAV position, and precoders only. ParCSI explicitly
        opts into a deterministic channel-observation mask.
        """
        theta_c = np.exp(1j * self.theta_phase)              # (Lp,)
        phi_c   = self.alpha * np.exp(1j * self.phi_phase)   # (La,)

        pos_norm = self.uav_pos / np.array([cfg.x_max, cfg.y_max,
                                             cfg.z_UAV + 1.0])
        _a_max = cfg.alpha_max + 1e-9

        precoders = np.concatenate([self.C.ravel(), self.T.ravel()])
        precoder_scale = np.sqrt(max(cfg.PT, 1e-12))

        # Use episode-level scale (fixed at reset from random-phase h_eff).
        # As phases improve within the episode, h_eff magnitude grows relative
        # to this baseline — giving the DRL a visible within-episode reward signal.
        observed_channels = np.empty(0, dtype=float)
        if self.observe_instantaneous_csi:
            H_bar_u = self.ch.composite_ue(theta_c)
            H_bar_k = self.ch.composite_te(phi_c)
            h_eff_u = np.concatenate([H.ravel() for H in H_bar_u])
            h_eff_k = np.concatenate([H.ravel() for H in H_bar_k])
            observed_channels = np.concatenate([
                h_eff_u.real / self._ch_scale,
                h_eff_u.imag / self._ch_scale,
                h_eff_k.real / self._ch_scale,
                h_eff_k.imag / self._ch_scale,
            ])
            observed_channels = np.where(
                self.csi_observation_mask, observed_channels, 0.0
            )

        s = np.concatenate([
            theta_c.real, theta_c.imag,                          # 2 Lp  ∈ [-1, 1]
            phi_c.real / _a_max, phi_c.imag / _a_max,           # 2 La  ∈ [-1, 1]
            pos_norm,                                             # 3
            precoders.real / precoder_scale,
            precoders.imag / precoder_scale,
            observed_channels,
            # last_sum_rate removed: it created Q(s,a)≈f(LSR) degeneracy that
            # prevented the critic from learning action-dependent structure.
        ])
        return s.astype(np.float32)
