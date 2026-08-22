import unittest

import torch

from ddpg_agent import DDPGAgent
from dtl_agent import DTLAgent, build_eq30_critic_target, next_upsilon


class DTLHypothesisTests(unittest.TestCase):
    def test_current_q_bellman_blend_matches_declared_formula(self):
        reward = torch.tensor([[2.0], [3.0]])
        done = torch.tensor([[0.0], [1.0]])
        source_current = torch.tensor([[10.0], [20.0]])
        target_current = torch.tensor([[7.0], [8.0]])
        source_next = torch.tensor([[30.0], [40.0]])
        target_next = torch.tensor([[4.0], [5.0]])
        actual = build_eq30_critic_target(
            "eq30_current_q_bellman_blend",
            0.25,
            reward,
            done,
            0.9,
            source_current,
            target_current,
            source_next,
            target_next,
        )
        bellman = reward + 0.9 * target_next * (1.0 - done)
        blended = 0.25 * source_current + 0.75 * bellman
        expected = torch.where(done.bool(), reward, blended)
        torch.testing.assert_close(actual, expected)

    def test_next_state_continuation_matches_previous_interpretation(self):
        reward = torch.tensor([[2.0], [3.0]])
        done = torch.tensor([[0.0], [1.0]])
        source_current = torch.tensor([[10.0], [20.0]])
        target_current = torch.tensor([[7.0], [8.0]])
        source_next = torch.tensor([[30.0], [40.0]])
        target_next = torch.tensor([[4.0], [5.0]])
        actual = build_eq30_critic_target(
            "eq30_next_state_continuation_blend",
            0.25,
            reward,
            done,
            0.9,
            source_current,
            target_current,
            source_next,
            target_next,
        )
        continuation = 0.25 * source_next + 0.75 * target_next
        expected = reward + 0.9 * continuation * (1.0 - done)
        torch.testing.assert_close(actual, expected)

    def test_literal_eq30_blends_only_current_source_and_target_q(self):
        reward = torch.tensor([[2.0], [300.0]])
        done = torch.tensor([[0.0], [1.0]])
        source_current = torch.tensor([[10.0], [20.0]])
        target_current = torch.tensor([[4.0], [8.0]])
        source_next = torch.tensor([[3000.0], [4000.0]])
        target_next = torch.tensor([[5000.0], [6000.0]])
        actual = build_eq30_critic_target(
            "eq30_literal_current_q_blend",
            0.25,
            reward,
            done,
            0.9,
            source_current,
            target_current,
            source_next,
            target_next,
        )
        expected = 0.25 * source_current + 0.75 * target_current
        torch.testing.assert_close(actual, expected)

    def test_constant_schedule_introduces_no_unreported_annealing(self):
        self.assertEqual(
            next_upsilon("constant_upsilon", 0.7, minimum=0.1, decay=0.001),
            0.7,
        )

    def test_linear_schedule_is_bounded_by_declared_floor(self):
        self.assertAlmostEqual(
            next_upsilon("linear_annealed_upsilon", 0.7,
                          minimum=0.1, decay=0.001),
            0.699,
        )
        self.assertEqual(
            next_upsilon("linear_annealed_upsilon", 0.1005,
                          minimum=0.1, decay=0.001),
            0.1,
        )

    def test_unknown_update_and_schedule_are_rejected(self):
        tensor = torch.ones((1, 1))
        with self.assertRaisesRegex(ValueError, "Unknown transfer update"):
            build_eq30_critic_target(
                "best_curve", 0.5, tensor, tensor * 0, 0.9,
                tensor, tensor, tensor, tensor,
            )
        with self.assertRaisesRegex(ValueError, "Unknown transfer schedule"):
            next_upsilon("best_curve", 0.5, minimum=0.1, decay=0.01)

    def test_cross_dimensional_transfer_preserves_the_common_state_mapping(self):
        torch.manual_seed(41)
        source = DDPGAgent(state_dim=3, action_dim=2)
        target = DTLAgent(source, state_dim=5, action_dim=2)

        torch.testing.assert_close(
            target.actor.l1.weight[:, :3], source.actor.l1.weight
        )
        torch.testing.assert_close(
            target.critic.s_l1.weight[:, :3], source.critic.s_l1.weight
        )
        self.assertEqual(
            torch.count_nonzero(target.actor.l1.weight[:, 3:]).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(target.critic.s_l1.weight[:, 3:]).item(), 0
        )

        common_state = torch.tensor([[0.2, -0.4, 0.7]])
        extended_state = torch.cat(
            [common_state, torch.tensor([[8.0, -9.0]])], dim=1
        )
        action = torch.tensor([[0.1, -0.2]])
        torch.testing.assert_close(
            target.actor(extended_state), source.actor(common_state)
        )
        torch.testing.assert_close(
            target.critic(extended_state, action),
            source.critic(common_state, action),
        )

    def test_source_q_uses_common_prefix_when_target_adds_csi_features(self):
        torch.manual_seed(42)
        source = DDPGAgent(state_dim=3, action_dim=2)
        target = DTLAgent(source, state_dim=5, action_dim=2)
        state = torch.tensor([[0.2, -0.4, 0.7, 8.0, -9.0]])
        next_state = torch.tensor([[0.3, -0.1, 0.5, -2.0, 4.0]])
        action = torch.tensor([[0.1, -0.2]])

        current, continuation = target._source_q_values(
            state, action, next_state
        )
        torch.testing.assert_close(
            current, source.critic(state[:, :3], action)
        )
        source_next_action = source.actor_tgt(next_state[:, :3])
        torch.testing.assert_close(
            continuation,
            source.critic_tgt(next_state[:, :3], source_next_action),
        )

    def test_transfer_rejects_unmapped_state_or_action_dimensions(self):
        source = DDPGAgent(state_dim=5, action_dim=2)
        with self.assertRaisesRegex(ValueError, "target state"):
            DTLAgent(source, state_dim=3, action_dim=2)
        with self.assertRaisesRegex(ValueError, "action dimensions"):
            DTLAgent(source, state_dim=5, action_dim=3)


if __name__ == "__main__":
    unittest.main()
