import unittest

import numpy as np

from config import cfg
from sinr_rate import build_beam_directions, resolve_sensing_beamformer_mode


class _ChannelFixture:
    def __init__(self, target_seed):
        ue_rng = np.random.default_rng(100)
        target_rng = np.random.default_rng(target_seed)
        self._ues = [
            ue_rng.normal(size=(cfg.N, cfg.M))
            + 1j * ue_rng.normal(size=(cfg.N, cfg.M))
            for _ in range(cfg.U)
        ]
        self._targets = [
            target_rng.normal(size=(cfg.N, cfg.M))
            + 1j * target_rng.normal(size=(cfg.N, cfg.M))
            for _ in range(cfg.K)
        ]

    def composite_ue(self, _theta):
        return self._ues

    def composite_te(self, _phi):
        return self._targets


class WFBHypothesisTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.ones(cfg.Lp, dtype=complex)
        self.phi = np.ones(cfg.La, dtype=complex)

    def test_matched_filter_is_unit_norm_and_target_channel_dependent(self):
        _, left = build_beam_directions(
            _ChannelFixture(1), self.theta, self.phi,
            sensing_beamformer_mode="matched_filter",
        )
        _, right = build_beam_directions(
            _ChannelFixture(2), self.theta, self.phi,
            sensing_beamformer_mode="matched_filter",
        )
        np.testing.assert_allclose(np.linalg.norm(left, axis=0), 1.0)
        self.assertFalse(np.allclose(left, right))

    def test_dft_ablation_is_unit_norm_and_target_channel_independent(self):
        _, left = build_beam_directions(
            _ChannelFixture(1), self.theta, self.phi,
            sensing_beamformer_mode="channel_independent_dft",
        )
        _, right = build_beam_directions(
            _ChannelFixture(2), self.theta, self.phi,
            sensing_beamformer_mode="channel_independent_dft",
        )
        np.testing.assert_allclose(np.linalg.norm(left, axis=0), 1.0)
        np.testing.assert_allclose(left, right)

    def test_unknown_spatial_filter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown sensing beamformer mode"):
            build_beam_directions(
                _ChannelFixture(1), self.theta, self.phi,
                sensing_beamformer_mode="best_reward",
            )

    def test_legacy_boolean_maps_to_explicit_modes_and_conflicts_fail(self):
        self.assertEqual(resolve_sensing_beamformer_mode(use_capon=True), "capon")
        self.assertEqual(
            resolve_sensing_beamformer_mode(use_capon=False), "matched_filter"
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_sensing_beamformer_mode(
                "channel_independent_dft", use_capon=False
            )


if __name__ == "__main__":
    unittest.main()
