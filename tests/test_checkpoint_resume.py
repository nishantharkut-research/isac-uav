import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
    validate_training_histories,
)
from ddpg_agent import DDPGAgent
from dtl_agent import DTLAgent
from research_objectives import ObjectiveSpec
from config import cfg
from experiment_protocol import derive_training_seed
from simulation import (
    _configuration_fingerprint,
    _training_identity,
    set_reproducibility,
)
from source_integrity import source_model_digest, source_tree_digest


class _DummyEnvironment:
    def __init__(self):
        self.value = 7

    def checkpoint_state(self):
        return {"value": self.value}

    def restore_checkpoint_state(self, state):
        self.value = int(state["value"])


class CheckpointResumeTests(unittest.TestCase):
    def _agent(self):
        agent = DDPGAgent(3, 2)
        agent.replay.push(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.25, -0.5]),
            4.0,
            np.array([2.0, 3.0, 4.0]),
            False,
        )
        agent.noise_std = 0.123
        agent.total_steps = 19
        return agent

    def test_full_state_and_rng_round_trip(self):
        identity = {
            "run_id": "paper-seed-2025",
            "configuration_fingerprint": "abc123",
            "profile_digest": "profile123",
            "objective": {"mode": "paper_baseline"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            agent = self._agent()
            env = _DummyEnvironment()
            saved_parameter = next(agent.actor.parameters()).detach().clone()
            random.seed(77)
            np.random.seed(77)
            torch.manual_seed(77)
            save_training_checkpoint(
                path,
                agent=agent,
                environment=env,
                completed_episode=12,
                histories={"rewards": [1.0, 2.0]},
                identity=identity,
                extra={"noise_schedule": 0.123},
            )
            expected_python = random.random()
            expected_numpy = np.random.random()
            expected_torch = torch.rand(1)

            with torch.no_grad():
                for parameter in agent.actor.parameters():
                    parameter.zero_()
            agent.replay.clear()
            agent.noise_std = 9.0
            agent.total_steps = 0
            env.value = -1
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)

            restored = load_training_checkpoint(
                path,
                agent=agent,
                environment=env,
                expected_identity=identity,
            )

            self.assertEqual(restored.completed_episode, 12)
            self.assertEqual(restored.histories["rewards"], [1.0, 2.0])
            self.assertEqual(len(agent.replay), 1)
            self.assertAlmostEqual(agent.noise_std, 0.123)
            self.assertEqual(agent.total_steps, 19)
            self.assertEqual(env.value, 7)
            torch.testing.assert_close(next(agent.actor.parameters()), saved_parameter)
            self.assertAlmostEqual(random.random(), expected_python)
            self.assertAlmostEqual(np.random.random(), expected_numpy)
            torch.testing.assert_close(torch.rand(1), expected_torch)

    def test_identity_mismatch_is_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            agent = self._agent()
            save_training_checkpoint(
                path,
                agent=agent,
                environment=_DummyEnvironment(),
                completed_episode=1,
                histories={},
                identity={"run_id": "baseline"},
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_training_checkpoint(
                    path,
                    agent=agent,
                    environment=_DummyEnvironment(),
                    expected_identity={"run_id": "extension"},
                )

    def test_failed_atomic_write_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            agent = self._agent()
            env = _DummyEnvironment()
            save_training_checkpoint(
                path, agent, env, 3, {"rewards": [3.0]}, {"run_id": "x"}
            )
            with patch("checkpointing.torch.save", side_effect=OSError("power loss")):
                with self.assertRaisesRegex(OSError, "power loss"):
                    save_training_checkpoint(
                        path, agent, env, 4, {"rewards": [4.0]}, {"run_id": "x"}
                    )
            restored = load_training_checkpoint(
                path, agent, env, {"run_id": "x"}
            )
            self.assertEqual(restored.completed_episode, 3)

    def test_dtl_source_state_and_transfer_weight_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dtl.pt"
            source = DDPGAgent(3, 2)
            agent = DTLAgent(source, 3, 2)
            agent.upsilon = 0.37
            saved_source = next(source.critic.parameters()).detach().clone()
            save_training_checkpoint(
                path, agent, _DummyEnvironment(), 5, {"rewards": []},
                {"run_id": "dtl"},
            )
            agent.upsilon = 0.01
            with torch.no_grad():
                next(source.critic.parameters()).zero_()
            load_training_checkpoint(
                path, agent, _DummyEnvironment(), {"run_id": "dtl"}
            )
            self.assertAlmostEqual(agent.upsilon, 0.37)
            torch.testing.assert_close(
                next(source.critic.parameters()), saved_source
            )
            self.assertFalse(next(source.critic.parameters()).requires_grad)

    def test_mid_episode_checkpoint_is_rejected(self):
        class MidEpisodeEnvironment(_DummyEnvironment):
            def checkpoint_state(self):
                return {"value": self.value, "episode_boundary": False}

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "episode boundaries"):
                save_training_checkpoint(
                    Path(directory) / "bad.pt",
                    self._agent(),
                    MidEpisodeEnvironment(),
                    1,
                    {},
                    {"run_id": "mid"},
                )

    def test_resume_histories_must_match_completed_episode(self):
        keys = ("rewards", "comm_rates")
        valid = {"rewards": [1.0, 2.0], "comm_rates": [3.0, 4.0]}
        restored = validate_training_histories(
            valid, completed_episode=2, required_keys=keys,
            requested_episodes=5,
        )
        self.assertEqual(restored, valid)

        with self.assertRaisesRegex(ValueError, "length"):
            validate_training_histories(
                {"rewards": [1.0], "comm_rates": [3.0, 4.0]},
                completed_episode=2,
                required_keys=keys,
                requested_episodes=5,
            )
        with self.assertRaisesRegex(ValueError, "requested"):
            validate_training_histories(
                valid, completed_episode=2, required_keys=keys,
                requested_episodes=1,
            )

    def test_dtl_checkpoint_identity_declares_transfer_interpretation(self):
        environment = type("Environment", (), {
            "sigma2": 0.0,
            "use_capon": True,
            "csi_observation_fraction": 1.0,
            "allocation_mode": "fast",
            "objective_spec": ObjectiveSpec.paper(target_rate_min=0.5),
            "state_dim": 10,
            "action_dim": 5,
        })()
        source = DDPGAgent(10, 5)
        identity = _training_identity(
            environment, "dtl-test", "DTL", 1500, source_agent=source
        )
        self.assertEqual(
            identity["dtl_transfer_mode"],
            "eq30_current_q_bellman_blend",
        )
        self.assertEqual(identity["dtl_transfer_schedule_id"], "constant_upsilon")
        self.assertEqual(
            identity["noise_hypothesis_id"],
            "description_literal_target_matches_ue",
        )
        self.assertEqual(
            identity["wfb_hypothesis_id"],
            "adaptive_matched_filter_no_capon",
        )
        self.assertEqual(
            identity["ddpg_comparator_csi_hypothesis_id"],
            "fair_pcsi_comparator",
        )
        self.assertEqual(identity["requested_episode_target"], 1500)
        self.assertEqual(identity["source_model_digest"], source_model_digest(source))
        self.assertEqual(identity["power_hypothesis_id"], "paper_db_as_dbm")
        self.assertRegex(identity["source_tree_digest"], r"^[0-9a-f]{64}$")

    def test_source_model_digest_changes_when_source_weights_change(self):
        source = DDPGAgent(3, 2)
        before = source_model_digest(source)
        with torch.no_grad():
            next(source.actor.parameters()).add_(1.0)
        self.assertNotEqual(before, source_model_digest(source))

    def test_source_tree_digest_changes_with_python_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            before = source_tree_digest(root)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(before, source_tree_digest(root))

    def test_configuration_fingerprint_covers_distinct_noise_terms_and_power_id(self):
        baseline = _configuration_fingerprint()
        with patch.object(cfg, "sigma2_k", cfg.sigma2_k * 2.0):
            self.assertNotEqual(baseline, _configuration_fingerprint())
        with patch.object(cfg, "power_hypothesis_id", "paper_db_as_dbw"):
            self.assertNotEqual(baseline, _configuration_fingerprint())
        with patch.object(
            cfg, "dtl_transfer_mode", "eq30_next_state_continuation_blend"
        ):
            self.assertNotEqual(baseline, _configuration_fingerprint())
        with patch.object(
            cfg, "dtl_transfer_schedule_id", "linear_annealed_upsilon"
        ):
            self.assertNotEqual(baseline, _configuration_fingerprint())
        with patch.object(
            cfg, "wfb_hypothesis_id", "channel_independent_dft_no_capon"
        ):
            self.assertNotEqual(baseline, _configuration_fingerprint())
        with patch.object(
            cfg, "ddpg_comparator_csi_hypothesis_id", "icsi_comparator"
        ):
            self.assertNotEqual(baseline, _configuration_fingerprint())

    def test_reproducibility_enables_deterministic_torch_backends(self):
        set_reproducibility(2025)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_training_seed_derivation_is_stable_and_role_separated(self):
        self.assertEqual(
            derive_training_seed(2025, "target"),
            derive_training_seed(2025, "target"),
        )
        self.assertNotEqual(
            derive_training_seed(2025, "source"),
            derive_training_seed(2025, "target"),
        )
        self.assertNotEqual(
            derive_training_seed(2025, "target"),
            derive_training_seed(2026, "target"),
        )


if __name__ == "__main__":
    unittest.main()
