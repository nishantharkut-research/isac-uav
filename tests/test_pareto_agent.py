import copy
import unittest

import numpy as np
import torch

from ddpg_agent import DDPGAgent
from pareto_agent import (
    BAPCDTLAgent,
    componentwise_bellman_target,
    remaining_discount_scale_from_state,
    scalarize_q_tensor,
)
from pareto_protocol import ParetoAnchors
from source_integrity import policy_model_digest


def _frozen_anchors():
    anchors = ParetoAnchors()
    anchors.update_training([[1.0, 2.0], [5.0, 8.0]])
    anchors.freeze({"partition": "training"})
    return anchors


class BAPCDTLArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        np.random.seed(11)
        self.baseline = DDPGAgent(4, 2)
        self.agent = BAPCDTLAgent(
            self.baseline,
            state_dim=4,
            action_dim=2,
            replay_capacity=16,
            batch_size=2,
            anchors=_frozen_anchors(),
        )

    def test_zero_residual_exactly_reproduces_frozen_baseline(self):
        state = np.array([0.2, -0.1, 0.4, 0.7], dtype=np.float32)
        expected = self.baseline.select_action(state, explore=False)
        actual = self.agent.select_action(state, [0.5, 0.5], explore=False)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in self.agent.baseline_actor.parameters()
        ))

    def test_transfer_initialization_has_zero_preference_columns_and_output(self):
        actor = self.agent.residual_actor
        torch.testing.assert_close(
            actor.l1.weight[:, :4], self.baseline.actor.l1.weight
        )
        torch.testing.assert_close(
            actor.l1.weight[:, 4:], torch.zeros_like(actor.l1.weight[:, 4:])
        )
        torch.testing.assert_close(actor.out.weight, torch.zeros_like(actor.out.weight))
        torch.testing.assert_close(actor.out.bias, torch.zeros_like(actor.out.bias))

    def test_vector_twins_have_separate_outputs(self):
        state = torch.zeros(3, 4)
        preference = torch.tensor([[0.5, 0.5]]).repeat(3, 1)
        action = torch.zeros(3, 2)
        q1, q2 = self.agent.critic_ensembles[0](state, preference, action)
        self.assertEqual(q1.shape, (3, 2))
        self.assertEqual(q2.shape, (3, 2))
        baseline_output = self.baseline.critic.out.weight.detach()[0]
        torch.testing.assert_close(q1_module(self.agent).out.weight[0], baseline_output)
        self.assertFalse(torch.equal(q1_module(self.agent).out.weight[1], baseline_output))

    def test_policy_digest_is_deterministic_and_parameter_sensitive(self):
        first = policy_model_digest(self.agent.baseline_actor)
        second = policy_model_digest(self.agent.baseline_actor)
        self.assertEqual(first, second)
        altered = copy.deepcopy(self.agent.baseline_actor)
        with torch.no_grad():
            next(altered.parameters()).add_(1.0)
        self.assertNotEqual(first, policy_model_digest(altered))

    def test_unanchored_ablation_has_zero_reference_and_distinct_contract(self):
        unanchored = BAPCDTLAgent(
            self.baseline, 4, 2, anchors=_frozen_anchors(), replay_capacity=4,
            batch_size=2, residual_radius=1.0, transfer_trunks=False,
            anchor_enabled=False,
        )
        state = np.array([0.2, -0.1, 0.4, 0.7], dtype=np.float32)
        np.testing.assert_array_equal(
            unanchored.select_action(state, [0.5, 0.5], explore=False),
            np.zeros(2, dtype=np.float32),
        )
        self.assertFalse(unanchored.training_state_dict()["contract"]["anchor_enabled"])


def q1_module(agent):
    return agent.critic_ensembles[0].q1


class BAPCDTLUpdateTests(unittest.TestCase):
    def test_componentwise_target_and_terminal_transition(self):
        reward = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        done = torch.tensor([[0.0], [1.0]])
        q1 = torch.tensor([[10.0, 30.0], [50.0, 70.0]])
        q2 = torch.tensor([[20.0, 25.0], [40.0, 80.0]])
        target = componentwise_bellman_target(reward, done, 0.5, q1, q2)
        torch.testing.assert_close(target[0], torch.tensor([6.0, 14.5]))
        torch.testing.assert_close(target[1], reward[1])

    def test_tensor_scalarization_keeps_exact_png_argmax_and_comparable_units(self):
        q_values = torch.tensor([[2.0, 5.0], [3.0, 6.0]])
        preferences = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        values = scalarize_q_tensor(q_values, preferences, _frozen_anchors())
        self.assertAlmostEqual(float(values[0]), 0.7, places=6)
        self.assertLess(abs(float(values[0]) - float(values[1])), 2.0)

    def test_remaining_discount_scale_uses_each_sample_time_feature(self):
        states = torch.tensor([
            [0.0, 9.0],
            [0.5, 9.0],
            [1.0, 9.0],
        ])
        scales = remaining_discount_scale_from_state(
            states, time_feature_index=0, gamma=0.9, horizon=300
        )

        self.assertEqual(float(scales[-1]), 1.0)
        self.assertGreater(float(scales[0]), float(scales[1]))
        self.assertGreater(float(scales[1]), float(scales[2]))

    def test_chebyshev_converts_discounted_returns_to_anchor_rate_units(self):
        value = scalarize_q_tensor(
            torch.tensor([[50.0, 80.0]]),
            torch.tensor([[1.0, 0.0]]),
            _frozen_anchors(),
            critic_return_scale=10.0,
        )
        torch.testing.assert_close(value, torch.tensor([1.05]))

    def test_update_and_exact_training_state_round_trip(self):
        torch.manual_seed(9)
        baseline = DDPGAgent(4, 2)
        agent = BAPCDTLAgent(
            baseline, 4, 2, replay_capacity=16, batch_size=2,
            policy_delay=1, anchors=_frozen_anchors(),
        )
        for index in range(3):
            agent.replay.push(
                np.full(4, index, dtype=np.float32),
                np.full(2, 0.1 * index, dtype=np.float32),
                [1.0 + index, 2.0 + index],
                np.full(4, index + 1, dtype=np.float32),
                index == 2,
            )
        result = agent.update(
            np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
        )
        self.assertIn("critic_loss", result)
        self.assertIn("actor_loss", result)

        state = agent.training_state_dict()
        restored = BAPCDTLAgent(
            baseline, 4, 2, replay_capacity=16, batch_size=2,
            policy_delay=1, anchors=_frozen_anchors(),
        )
        restored.load_training_state_dict(state)
        self.assertEqual(restored.total_steps, agent.total_steps)
        self.assertEqual(len(restored.replay), len(agent.replay))
        for left, right in zip(
            restored.residual_actor.parameters(), agent.residual_actor.parameters()
        ):
            torch.testing.assert_close(left, right)

    def test_restore_rejects_baseline_and_algorithm_mismatch(self):
        baseline = DDPGAgent(4, 2)
        agent = BAPCDTLAgent(
            baseline, 4, 2, replay_capacity=4, batch_size=2,
            anchors=_frozen_anchors(),
        )
        state = agent.training_state_dict()
        state["algorithm_id"] = "different"
        with self.assertRaisesRegex(ValueError, "algorithm"):
            agent.load_training_state_dict(state)

        state = agent.training_state_dict()
        state["baseline_policy_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "baseline"):
            agent.load_training_state_dict(state)

    def test_restore_rejects_different_pareto_anchors(self):
        baseline = DDPGAgent(4, 2)
        source = BAPCDTLAgent(
            baseline, 4, 2, replay_capacity=4, batch_size=2,
            anchors=_frozen_anchors(),
        )
        state = source.training_state_dict()

        different = ParetoAnchors()
        different.update_training([[0.0, 1.0], [7.0, 9.0]])
        different.freeze({"partition": "different_training_data"})
        target = BAPCDTLAgent(
            baseline, 4, 2, replay_capacity=4, batch_size=2,
            anchors=different,
        )
        with self.assertRaisesRegex(ValueError, "contract|anchor"):
            target.load_training_state_dict(state)


if __name__ == "__main__":
    unittest.main()
