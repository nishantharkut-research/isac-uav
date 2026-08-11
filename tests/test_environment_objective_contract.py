import unittest
from unittest.mock import patch

import numpy as np

from config import cfg
from environment import HRISUAVISACEnv
from research_objectives import ObjectiveSpec


def _fake_beamformers(*_args, **_kwargs):
    return (
        np.zeros((cfg.M, cfg.U), dtype=complex),
        np.zeros((cfg.M, cfg.K), dtype=complex),
        {
            "comm_power_frac": 0.5,
            "beamformer_score": 0.0,
            "sensing_constraint_feasible": True,
            "sensing_rate_deficit": 0.0,
            "sensing_only_sinr_deficit": 0.0,
            "active_power": 0.0,
            "active_power_feasible": True,
            "active_power_deficit": 0.0,
        },
    )


def _fake_rates(*_args, **_kwargs):
    return (
        np.array([2.0, 2.0, 2.0]),
        np.array([1.0, 1.0, 1.0]),
        np.ones(3),
        np.ones(3),
    )


class EnvironmentObjectiveContractTests(unittest.TestCase):
    def _step(self, spec):
        with patch("environment.select_feasible_beamformers", _fake_beamformers), patch(
            "environment.compute_rates", _fake_rates
        ):
            env = HRISUAVISACEnv(seed=11, objective_spec=spec)
            _, reward, _, info = env.step(np.zeros(env.action_dim))
        return reward, info

    def test_paper_environment_returns_raw_communication_reward(self):
        reward, info = self._step(ObjectiveSpec.paper(target_rate_min=0.5))
        self.assertAlmostEqual(reward, 6.0)
        self.assertEqual(info["objective"]["mode"], "paper_baseline")
        self.assertNotIn("diagnostic_penalized_reward", info)
        self.assertNotIn("diagnostic_constraint_penalty", info)

    def test_png_environment_returns_raw_joint_reward(self):
        reward, info = self._step(
            ObjectiveSpec.png_joint(user_rate_min=0.5, target_rate_min=0.5)
        )
        self.assertAlmostEqual(reward, 9.0)
        self.assertAlmostEqual(info["communication_sum_rate"], 6.0)
        self.assertAlmostEqual(info["sensing_sum_rate"], 3.0)

    def test_conflicting_legacy_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sensing_rate_min"):
            HRISUAVISACEnv(
                seed=11,
                sensing_rate_min=0.9,
                objective_spec=ObjectiveSpec.paper(target_rate_min=0.5),
            )


if __name__ == "__main__":
    unittest.main()
