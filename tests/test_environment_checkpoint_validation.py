import copy
import unittest

import numpy as np

from config import cfg
from environment import HRISUAVISACEnv


class EnvironmentCheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.environment = HRISUAVISACEnv(seed=2025)
        self.state = self.environment.checkpoint_state()

    def test_restore_rejects_non_boundary_state(self):
        state = copy.deepcopy(self.state)
        state["episode_boundary"] = False
        state["t"] = 1
        with self.assertRaisesRegex(ValueError, "episode boundary"):
            self.environment.restore_checkpoint_state(state)

    def test_restore_rejects_invalid_array_shapes(self):
        corruptions = {
            "uav_pos": np.zeros(2),
            "tes_velocity": np.zeros((cfg.K, 2)),
            "theta_phase": np.zeros(cfg.Lp + 1),
            "phi_phase": np.zeros(cfg.La + 1),
            "alpha": np.zeros(cfg.La + 1),
        }
        for field, value in corruptions.items():
            with self.subTest(field=field):
                state = copy.deepcopy(self.state)
                state[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self.environment.restore_checkpoint_state(state)

    def test_restore_rejects_nonfinite_or_out_of_range_amplification(self):
        for value in (float("nan"), -1.0, cfg.alpha_max + 1.0):
            with self.subTest(value=value):
                state = copy.deepcopy(self.state)
                state["alpha"][0] = value
                with self.assertRaisesRegex(ValueError, "alpha"):
                    self.environment.restore_checkpoint_state(state)


if __name__ == "__main__":
    unittest.main()
