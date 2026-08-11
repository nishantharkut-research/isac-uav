import inspect
import unittest
from unittest.mock import patch

import numpy as np

from config import cfg
from environment import (
    ControlState,
    HRISUAVISACEnv,
    InfeasibleChannelError,
    interpolate_control_state,
)
from research_objectives import ObjectiveSpec, evaluate_objective
import simulation


def fake_rates(*_args, **_kwargs):
    return (
        np.full(cfg.U, 2.0),
        np.full(cfg.K, 1.0),
        np.ones(cfg.U),
        np.ones(cfg.K),
    )


def beam_result(feasible):
    return (
        np.zeros((cfg.M, cfg.U), dtype=complex),
        np.zeros((cfg.M, cfg.K), dtype=complex),
        {
            "comm_power_frac": 0.5,
            "stream_powers": np.full(cfg.U + cfg.K, cfg.PT / (cfg.U + cfg.K)),
            "beamformer_score": 1.0,
            "allocation_feasible": feasible,
            "allocation_method": "fixture",
            "allocation_solver_success": True,
            "allocation_normalized_deficit": 0.0 if feasible else 1.0,
            "sensing_constraint_feasible": feasible,
            "sensing_rate_deficit": 0.0 if feasible else 1.0,
            "sensing_only_sinr_deficit": 0.0,
            "active_power": 0.0,
            "active_power_feasible": True,
            "active_power_deficit": 0.0,
        },
    )


class EnvironmentFeasibleProjectionTests(unittest.TestCase):
    def test_phase_interpolation_uses_shortest_wrapped_difference(self):
        previous = ControlState(
            theta_phase=np.array([2.0 * np.pi - 0.1]),
            phi_phase=np.array([2.0 * np.pi - 0.1]),
            alpha=np.array([1.0]),
            uav_pos=np.array([1.0, 1.0, 1.0]),
        )
        proposed = ControlState(
            theta_phase=np.array([0.1]),
            phi_phase=np.array([0.1]),
            alpha=np.array([3.0]),
            uav_pos=np.array([3.0, 1.0, 1.0]),
        )
        middle = interpolate_control_state(previous, proposed, 0.5)
        wrapped_theta = np.angle(np.exp(1j * middle.theta_phase[0]))
        self.assertAlmostEqual(wrapped_theta, 0.0, places=10)
        self.assertAlmostEqual(middle.alpha[0], 2.0)
        self.assertTrue(np.allclose(middle.uav_pos, [2.0, 1.0, 1.0]))

    def test_control_distance_is_finite_for_passive_only_controls(self):
        left = ControlState(
            theta_phase=np.array([0.0]),
            phi_phase=np.empty(0),
            alpha=np.empty(0),
            uav_pos=np.array([1.0, 1.0, 1.0]),
        )
        right = ControlState(
            theta_phase=np.array([0.1]),
            phi_phase=np.empty(0),
            alpha=np.empty(0),
            uav_pos=np.array([1.0, 1.0, 1.0]),
        )
        self.assertTrue(np.isfinite(HRISUAVISACEnv._control_distance(left, right)))

    def _patches(self, selector):
        return (
            patch("environment.select_feasible_beamformers", side_effect=selector),
            patch("environment.compute_rates", side_effect=fake_rates),
            patch("environment.cascade_penalty_16c", return_value=(0.0, True, 0.0)),
            patch("environment.active_tx_power", return_value=0.0),
            patch("environment.active_power_penalty", return_value=(0.0, True)),
        )

    def test_infeasible_proposal_is_projected_and_executed_action_is_reported(self):
        def selector(_ch, _theta, phi, **_kwargs):
            return beam_result(np.mean(np.abs(phi)) <= 0.75 * cfg.alpha_max)

        patches = self._patches(selector)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            env = HRISUAVISACEnv(seed=10)
            proposed = np.zeros(env.action_dim)
            proposed[cfg.Lp + cfg.La:cfg.Lp + 2 * cfg.La] = 1.0
            _state, _reward, _done, info = env.step(proposed)

        self.assertTrue(info["action_was_projected"])
        self.assertGreater(info["action_projection_fraction"], 0.0)
        self.assertLess(info["action_projection_fraction"], 1.0)
        self.assertTrue(info["physical_constraints_feasible"])
        self.assertFalse(np.allclose(info["executed_action"], proposed))

    def test_feasible_proposal_executes_without_projection(self):
        patches = self._patches(lambda *_args, **_kwargs: beam_result(True))
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            env = HRISUAVISACEnv(seed=10)
            proposed = np.zeros(env.action_dim)
            _state, _reward, _done, info = env.step(proposed)
        self.assertFalse(info["action_was_projected"])
        self.assertEqual(info["action_projection_fraction"], 1.0)
        self.assertTrue(np.allclose(info["executed_action"], proposed))

    def test_reset_records_an_infeasible_initial_control_without_claiming_success(self):
        patches = self._patches(lambda *_args, **_kwargs: beam_result(False))
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            env = HRISUAVISACEnv(seed=10)
        self.assertFalse(env.initial_physical_feasible)
        self.assertFalse(
            env.initial_feasibility_diagnostics["physical_feasible"]
        )

    def test_projection_search_does_not_assume_monotone_phase_feasibility(self):
        env = object.__new__(HRISUAVISACEnv)
        previous = ControlState(
            np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(3)
        )
        proposed = ControlState(
            np.zeros(1), np.zeros(1), np.ones(1), np.zeros(3)
        )

        def assess(controls, *_args):
            fraction = float(controls.alpha[0])
            feasible = fraction <= 1e-12 or 0.70 <= fraction <= 0.85
            return {"physical_feasible": feasible, "fraction": fraction}

        env._evaluate_controls = assess
        controls, assessment, fraction = env._find_feasible_controls(
            previous, proposed, channel_seed=1, previous_powers=None,
            slot_time_s=0.1,
        )

        self.assertGreaterEqual(fraction, 0.70)
        self.assertLessEqual(fraction, 0.85)
        self.assertTrue(assessment["physical_feasible"])
        self.assertAlmostEqual(controls.alpha[0], fraction)

    def test_step_commits_selected_assessment_channel_before_building_state(self):
        patches = self._patches(lambda *_args, **_kwargs: beam_result(True))
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            env = HRISUAVISACEnv(
                seed=10,
                objective_spec=ObjectiveSpec.paper(target_rate_min=0.5),
            )

        selected_controls = env._snapshot_controls()
        selected_channel = object()
        rejected_probe_channel = object()
        env.ch = rejected_probe_channel
        rate_u = np.full(cfg.U, 2.0)
        rate_k = np.full(cfg.K, 1.0)
        assessment = {
            "channel": selected_channel,
            "C": np.zeros((cfg.M, cfg.U), dtype=complex),
            "T": np.zeros((cfg.M, cfg.K), dtype=complex),
            "beam_info": beam_result(True)[2],
            "rate_u": rate_u,
            "rate_k": rate_k,
            "sinr_u": np.ones(cfg.U),
            "sinr_k": np.ones(cfg.K),
            "objective": evaluate_objective(env.objective_spec, rate_u, rate_k),
            "communication_sum_rate": float(np.sum(rate_u)),
            "sensing_sum_rate": float(np.sum(rate_k)),
            "physical_feasible": True,
            "cascade_penalty": 0.0,
            "cascade_ok": True,
            "cascade_power": 0.0,
            "active_power": 0.0,
            "active_power_penalty": 0.0,
            "active_power_ok": True,
            "sensing_ok": True,
        }

        with patch.object(
            env,
            "_find_feasible_controls",
            return_value=(selected_controls, assessment, 0.0),
        ), patch.object(
            env,
            "_build_state",
            return_value=np.zeros(env.state_dim, dtype=np.float32),
        ), patch.object(
            cfg,
            "channel_coherence_hypothesis_id",
            "episode_block_fading",
        ):
            env.step(np.zeros(env.action_dim))

        self.assertIs(env.ch, selected_channel)

    def test_final_slot_projection_cannot_break_uav_closure(self):
        def selector(_ch, _theta, phi, **_kwargs):
            return beam_result(np.mean(np.abs(phi)) <= 0.75 * cfg.alpha_max)

        original_slots = cfg.N_slots
        cfg.N_slots = 2
        try:
            patches = self._patches(selector)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                env = HRISUAVISACEnv(seed=10)
                first = np.zeros(env.action_dim)
                first[-2:] = [1.0, 0.0]
                env.step(first)
                self.assertGreater(
                    np.linalg.norm(env.uav_pos[:2] - env._uav_start[:2]), 0.0
                )

                final = np.zeros(env.action_dim)
                final[cfg.Lp + cfg.La:cfg.Lp + 2 * cfg.La] = 1.0
                _state, _reward, done, info = env.step(final)

            self.assertTrue(done)
            self.assertTrue(np.allclose(env.uav_pos[:2], env._uav_start[:2]))
            self.assertTrue(info["uav_closure_ok"])
            self.assertEqual(info["projection_source"], "deterministic_recovery")
        finally:
            cfg.N_slots = original_slots

    def test_deterministic_recovery_can_rescue_infeasible_initial_controls(self):
        def selector(_ch, _theta, phi, **_kwargs):
            return beam_result(np.allclose(np.abs(phi), 0.0))

        patches = self._patches(selector)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            env = HRISUAVISACEnv(seed=10)
            self.assertFalse(env.initial_physical_feasible)
            proposed = np.ones(env.action_dim)
            _state, _reward, _done, info = env.step(proposed)

        self.assertTrue(info["physical_constraints_feasible"])
        self.assertEqual(info["projection_source"], "deterministic_recovery")
        self.assertTrue(np.allclose(env.alpha, 0.0))

    def test_failed_fast_allocation_retries_scipy_on_the_same_control(self):
        allocation_modes = []

        def selector(_ch, _theta, _phi, **kwargs):
            mode = kwargs["allocation_mode"]
            allocation_modes.append(mode)
            return beam_result(mode == "scipy")

        patches = self._patches(selector)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            env = HRISUAVISACEnv(seed=10, allocation_mode="fast")
            allocation_modes.clear()
            _state, _reward, _done, info = env.step(np.zeros(env.action_dim))

        self.assertEqual(allocation_modes[:2], ["fast", "scipy"])
        self.assertTrue(info["physical_constraints_feasible"])
        self.assertTrue(info["allocation_fallback_used"])

    def test_exhausted_same_channel_recovery_stops_explicitly(self):
        patches = self._patches(lambda *_args, **_kwargs: beam_result(True))
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            env = HRISUAVISACEnv(seed=10)

        patches = self._patches(lambda *_args, **_kwargs: beam_result(False))
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with self.assertRaises(InfeasibleChannelError) as captured:
                env.step(np.zeros(env.action_dim))
        self.assertEqual(
            captured.exception.diagnostics["reason"],
            "deterministic_projection_search_exhausted",
        )

    def test_training_replay_uses_actor_command_not_projected_action(self):
        for function in (simulation.train_ddpg, simulation.train_dtl):
            source = inspect.getsource(function)
            self.assertIn('info["proposed_action"]', source)


if __name__ == "__main__":
    unittest.main()
