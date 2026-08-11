import unittest

import torch

from dtl_agent import build_eq30_critic_target, next_upsilon


class DTLHypothesisTests(unittest.TestCase):
    def test_current_q_bellman_blend_matches_declared_formula(self):
        reward = torch.tensor([[2.0], [3.0]])
        done = torch.tensor([[0.0], [1.0]])
        source_current = torch.tensor([[10.0], [20.0]])
        source_next = torch.tensor([[30.0], [40.0]])
        target_next = torch.tensor([[4.0], [5.0]])
        actual = build_eq30_critic_target(
            "eq30_current_q_bellman_blend",
            0.25,
            reward,
            done,
            0.9,
            source_current,
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
        source_next = torch.tensor([[30.0], [40.0]])
        target_next = torch.tensor([[4.0], [5.0]])
        actual = build_eq30_critic_target(
            "eq30_next_state_continuation_blend",
            0.25,
            reward,
            done,
            0.9,
            source_current,
            source_next,
            target_next,
        )
        continuation = 0.25 * source_next + 0.75 * target_next
        expected = reward + 0.9 * continuation * (1.0 - done)
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
                tensor, tensor, tensor,
            )
        with self.assertRaisesRegex(ValueError, "Unknown transfer schedule"):
            next_upsilon("best_curve", 0.5, minimum=0.1, decay=0.01)


if __name__ == "__main__":
    unittest.main()
