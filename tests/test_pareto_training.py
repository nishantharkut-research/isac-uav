import tempfile
import unittest
from pathlib import Path
import random

import numpy as np
import torch

from ddpg_agent import DDPGAgent
from pareto_agent import BAPCDTLAgent
from pareto_protocol import ParetoAnchors
from pareto_training import (
    SobolPreferenceStream,
    evaluate_preference_grid,
    train_ba_pcdtl,
)


def _anchors():
    anchors = ParetoAnchors()
    anchors.update_training([[0.0, 0.0], [10.0, 10.0]])
    anchors.freeze({"partition": "training"})
    return anchors


class TinyEnvironment:
    state_dim = 2
    action_dim = 1

    def __init__(self, seed=0):
        self.seed = int(seed)
        self.episode = 0
        self.step_index = 0
        self.episode_boundary = True
        self.proposed_actions = []

    def reset(self):
        self.episode += 1
        self.step_index = 0
        self.episode_boundary = False
        return np.array([self.seed, self.episode], dtype=np.float32)

    def step(self, action):
        self.proposed_actions.append(np.asarray(action, dtype=np.float32).copy())
        self.step_index += 1
        done = self.step_index == 2
        self.episode_boundary = done
        communication = float(1 + self.step_index)
        sensing = float(3 + self.step_index)
        next_state = np.array([self.seed, self.episode + 0.1 * self.step_index], dtype=np.float32)
        executed = np.asarray(action, dtype=np.float32) * 0.5
        info = {
            "communication_sum_rate": communication,
            "sensing_sum_rate": sensing,
            "joint_rate": communication + sensing,
            "proposed_action": np.asarray(action, dtype=np.float32),
            "executed_action": executed,
            "physical_constraints_feasible": True,
            "rate_u": np.array([communication]),
            "rate_k": np.array([sensing]),
        }
        return next_state, -999.0, done, info

    def checkpoint_state(self):
        return {
            "seed": self.seed,
            "episode": self.episode,
            "step_index": self.step_index,
            "episode_boundary": self.episode_boundary,
        }

    def restore_checkpoint_state(self, state):
        self.seed = int(state["seed"])
        self.episode = int(state["episode"])
        self.step_index = int(state["step_index"])
        self.episode_boundary = bool(state["episode_boundary"])


def _agent(seed=7):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    baseline = DDPGAgent(2, 1)
    return BAPCDTLAgent(
        baseline, 2, 1, anchors=_anchors(), replay_capacity=32,
        batch_size=2, policy_delay=1, ensemble_size=2,
    )


class PreferenceStreamTests(unittest.TestCase):
    def test_stream_round_trip_and_balanced_update_batch(self):
        stream = SobolPreferenceStream(seed=19)
        first = stream.next(3)
        state = stream.state_dict()
        expected_next = stream.next(4)
        restored = SobolPreferenceStream.from_state_dict(state)
        np.testing.assert_array_equal(restored.next(4), expected_next)
        self.assertTrue(np.all(first >= 0.0))
        np.testing.assert_allclose(first.sum(axis=1), 1.0)

        batch = restored.update_batch(5)
        np.testing.assert_array_equal(batch[0], [0.5, 0.5])
        self.assertEqual(batch.shape, (5, 2))


class TrainerTests(unittest.TestCase):
    def test_trainer_stores_physical_vector_and_actor_command(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = TinyEnvironment()
            result = train_ba_pcdtl(
                _agent(), environment, n_episodes=1,
                identity={"run_id": "physical-vector"},
                preference_stream=SobolPreferenceStream(3),
                checkpoint_path=Path(directory) / "state.pt",
                checkpoint_every=1,
            )
            self.assertEqual(result["histories"]["communication"], [2.5])
            self.assertEqual(result["histories"]["sensing"], [4.5])
            self.assertEqual(result["histories"]["joint"], [7.0])
            transitions = result["agent"].replay.state_dict()["transitions"]
            np.testing.assert_array_equal(transitions[0][2], [2.0, 4.0])
            np.testing.assert_array_equal(
                transitions[0][1], environment.proposed_actions[0]
            )

    def test_interrupted_resume_matches_uninterrupted_training(self):
        identity = {"run_id": "resume", "episodes": 2}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pt"
            torch.manual_seed(23)
            np.random.seed(23)
            interrupted_agent = _agent(seed=23)
            first = train_ba_pcdtl(
                interrupted_agent, TinyEnvironment(), 2, identity,
                SobolPreferenceStream(13), checkpoint,
                checkpoint_every=1, stop_after_episode=1,
            )
            self.assertFalse(first["complete"])

            resumed_agent = _agent(seed=23)
            resumed = train_ba_pcdtl(
                resumed_agent, TinyEnvironment(), 2, identity,
                SobolPreferenceStream(13), checkpoint,
                checkpoint_every=1, resume=True,
            )

            torch.manual_seed(23)
            np.random.seed(23)
            continuous = train_ba_pcdtl(
                _agent(seed=23), TinyEnvironment(), 2, identity,
                SobolPreferenceStream(13), Path(directory) / "continuous.pt",
                checkpoint_every=1,
            )
            self.assertEqual(resumed["histories"], continuous["histories"])
            for left, right in zip(
                resumed["agent"].residual_actor.parameters(),
                continuous["agent"].residual_actor.parameters(),
            ):
                torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    def test_evaluator_returns_unsmoothed_raw_arrays(self):
        result = evaluate_preference_grid(
            _agent(),
            environment_factory=lambda seed: TinyEnvironment(seed),
            scene_seeds=[101, 102],
            preferences=[[0.5, 0.5], [1.0, 0.0]],
        )
        self.assertEqual(result["scene_seeds"], [101, 102])
        self.assertEqual(len(result["preferences"]), 2)
        balanced = result["preferences"][0]
        self.assertEqual(balanced["communication"], [2.5, 2.5])
        self.assertEqual(balanced["sensing"], [4.5, 4.5])
        self.assertEqual(balanced["minimum_ue_rate"], [2.0, 2.0])
        self.assertEqual(balanced["minimum_target_rate"], [4.0, 4.0])
        self.assertNotIn("smoothed", balanced)


if __name__ == "__main__":
    unittest.main()
