import copy
import unittest

import numpy as np
import torch

from vector_replay import VectorReplayBuffer


class VectorReplayBufferTests(unittest.TestCase):
    def test_round_trip_preserves_vector_reward_and_dimensions(self):
        replay = VectorReplayBuffer(8, state_dim=3, action_dim=2, reward_dim=2)
        replay.push([1, 2, 3], [0.1, -0.2], [4.0, 5.0], [2, 3, 4], False)

        restored = VectorReplayBuffer(8, 3, 2, 2)
        restored.load_state_dict(replay.state_dict())

        self.assertEqual(len(restored), 1)
        state = restored.state_dict()
        self.assertEqual(state["reward_dim"], 2)
        np.testing.assert_array_equal(state["transitions"][0][2], [4.0, 5.0])

    def test_sample_returns_expected_tensor_shapes(self):
        replay = VectorReplayBuffer(8, state_dim=3, action_dim=2, reward_dim=2)
        for index in range(4):
            replay.push(
                np.full(3, index), np.full(2, index), [index, index + 1],
                np.full(3, index + 1), index == 3,
            )

        state, action, reward, next_state, done = replay.sample(3)
        self.assertEqual(state.shape, (3, 3))
        self.assertEqual(action.shape, (3, 2))
        self.assertEqual(reward.shape, (3, 2))
        self.assertEqual(next_state.shape, (3, 3))
        self.assertEqual(done.shape, (3, 1))
        self.assertEqual(reward.dtype, torch.float32)

    def test_push_rejects_wrong_dimensions_and_nonfinite_values(self):
        replay = VectorReplayBuffer(8, 3, 2, 2)
        invalid = (
            ([1, 2], [1, 2], [1, 2], [1, 2, 3], False),
            ([1, 2, 3], [1], [1, 2], [1, 2, 3], False),
            ([1, 2, 3], [1, 2], [1], [1, 2, 3], False),
            ([1, 2, 3], [1, 2], [1, 2], [1, 2], False),
            ([1, 2, np.nan], [1, 2], [1, 2], [1, 2, 3], False),
            ([1, 2, 3], [1, np.inf], [1, 2], [1, 2, 3], False),
        )
        for transition in invalid:
            with self.subTest(transition=transition):
                with self.assertRaises(ValueError):
                    replay.push(*transition)

    def test_restore_rejects_contract_mismatch_and_oversized_state(self):
        replay = VectorReplayBuffer(2, 3, 2, 2)
        replay.push([1, 2, 3], [1, 2], [1, 2], [2, 3, 4], False)
        state = replay.state_dict()

        with self.assertRaisesRegex(ValueError, "dimensions"):
            VectorReplayBuffer(2, 4, 2, 2).load_state_dict(state)

        oversized = copy.deepcopy(state)
        oversized["capacity"] = 1
        oversized["transitions"] *= 2
        with self.assertRaisesRegex(ValueError, "capacity"):
            VectorReplayBuffer(1, 3, 2, 2).load_state_dict(oversized)

    def test_constructor_and_sample_validate_arguments(self):
        for args in ((0, 3, 2, 2), (2, 0, 2, 2), (2, 3, 0, 2), (2, 3, 2, 0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    VectorReplayBuffer(*args)
        with self.assertRaisesRegex(ValueError, "enough"):
            VectorReplayBuffer(2, 3, 2, 2).sample(1)


if __name__ == "__main__":
    unittest.main()
