import copy
from unittest.mock import patch

import numpy as np
import pytest

from ambiguity_hypotheses import (
    PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID,
    PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID,
)
from channel_model import ChannelSet, plos
from config import cfg
from environment import HRISUAVISACEnv
from sinr_rate import (
    _cascade_error_covariance,
    build_beamformers,
    compute_rates,
    sensing_only_sinr,
)


def _scene():
    uav = np.array([150.0, 100.0, cfg.z_UAV])
    users = [
        np.array([20.0, 20.0, 0.0]),
        np.array([60.0, 70.0, 0.0]),
        np.array([90.0, 40.0, 0.0]),
    ]
    targets = [
        np.array([220.0, 40.0, 0.0]),
        np.array([250.0, 100.0, 0.0]),
        np.array([200.0, 160.0, 0.0]),
    ]
    return uav, users, targets


def test_los_symbol_role_hypotheses_are_explicit_and_exact():
    saved = cfg.los_probability_hypothesis_id
    elevation = 20.0
    try:
        cfg.los_probability_hypothesis_id = "paper_eq6_literal_roles"
        literal = plos(elevation)
        expected_literal = 1.0 / (
            1.0
            + cfg.b_env
            * np.exp(-cfg.a_env * (elevation - cfg.b_env))
        )
        assert literal == pytest.approx(expected_literal)

        cfg.los_probability_hypothesis_id = (
            "al_hourani_standard_mapped_roles"
        )
        mapped = plos(elevation)
        expected_mapped = 1.0 / (
            1.0
            + cfg.a_env
            * np.exp(-cfg.b_env * (elevation - cfg.a_env))
        )
        assert mapped == pytest.approx(expected_mapped)
        assert abs(literal - mapped) > 0.1
    finally:
        cfg.los_probability_hypothesis_id = saved


def test_primary_los_branch_matches_printed_eq6_and_high_los_text():
    saved = cfg.los_probability_hypothesis_id
    try:
        cfg.los_probability_hypothesis_id = (
            PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID
        )
        assert PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID == (
            "paper_eq6_literal_roles"
        )
        assert plos(20.0) > 0.99
    finally:
        cfg.los_probability_hypothesis_id = saved


def test_scee_rate_is_continuous_at_zero_for_fixed_estimated_channel():
    channel = ChannelSet(*_scene(), sigma2=0.0, seed=2025)
    theta = np.ones(cfg.Lp, dtype=complex)
    phi = np.ones(cfg.La, dtype=complex)
    C, T = build_beamformers(channel, theta, phi)
    zero_rates = compute_rates(channel, theta, phi, C, T)[0]

    perturbed = copy.deepcopy(channel)
    epsilon = 1e-14
    perturbed.scee_nmse = epsilon
    perturbed.sv_bu = [epsilon] * cfg.U
    perturbed.sv_bk = [epsilon] * cfg.K
    perturbed.sv_brj = epsilon
    perturbed.sv_bri = epsilon
    perturbed.sv_ruj = [epsilon] * cfg.U
    perturbed.sv_rti = [epsilon] * cfg.K
    epsilon_rates = compute_rates(perturbed, theta, phi, C, T)[0]

    assert np.all(epsilon_rates <= zero_rates + 1e-12)
    assert np.allclose(epsilon_rates, zero_rates, rtol=2e-4, atol=1e-8)


def test_cross_function_interference_hypotheses_are_numerically_distinct():
    saved = cfg.cross_function_interference_hypothesis_id
    channel = ChannelSet(*_scene(), sigma2=0.0, seed=2026)
    theta = np.ones(cfg.Lp, dtype=complex)
    phi = np.ones(cfg.La, dtype=complex)
    C, T = build_beamformers(channel, theta, phi)
    try:
        cfg.cross_function_interference_hypothesis_id = (
            "paper_eq12_eq14_literal_cross_channels"
        )
        literal_scalar = compute_rates(
            channel, theta, phi, C, T, use_scee_model=False
        )
        literal_matrix = compute_rates(channel, theta, phi, C, T)

        cfg.cross_function_interference_hypothesis_id = (
            "receiver_consistent_cross_channels"
        )
        receiver_scalar = compute_rates(
            channel, theta, phi, C, T, use_scee_model=False
        )
        receiver_matrix = compute_rates(channel, theta, phi, C, T)
    finally:
        cfg.cross_function_interference_hypothesis_id = saved

    assert not np.allclose(literal_scalar[0], receiver_scalar[0])
    assert not np.allclose(literal_scalar[1], receiver_scalar[1])
    assert not np.allclose(literal_matrix[0], receiver_matrix[0])
    assert not np.allclose(literal_matrix[1], receiver_matrix[1])


def test_vectorized_cascade_covariance_matches_elementwise_equation():
    rng = np.random.default_rng(77)
    h_r = rng.normal(size=(2, 5)) + 1j * rng.normal(size=(2, 5))
    g_br = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    weights = rng.normal(size=5) + 1j * rng.normal(size=5)
    factor = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    q = factor @ factor.conj().T
    sigma_g, sigma_h = 0.03, 0.07

    expected = np.zeros((2, 2), dtype=complex)
    trace_q = float(np.trace(q).real)
    identity = np.eye(2, dtype=complex)
    for index, weight in enumerate(weights):
        h = h_r[:, index]
        g = g_br[index, :]
        g_q_g = max(float(np.real(g @ q @ g.conj())), 0.0)
        expected += abs(weight) ** 2 * (
            sigma_g * trace_q * np.outer(h, h.conj())
            + sigma_h * g_q_g * identity
            + sigma_g * sigma_h * trace_q * identity
        )

    actual = _cascade_error_covariance(
        h_r, g_br, weights, q, sigma_g, sigma_h
    )
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_vectorized_sensing_sinr_matches_elementwise_equation():
    channel = ChannelSet(*_scene(), sigma2=0.0, seed=2027)
    phi = np.exp(1j * np.linspace(0.0, 1.0, cfg.La))
    theta = np.ones(cfg.Lp, dtype=complex)
    _c, t = build_beamformers(channel, theta, phi)
    actual = sensing_only_sinr(channel, phi, t)

    expected = np.zeros(cfg.K)
    for k, h_k in enumerate(channel.composite_te(phi)):
        powers = [float(np.linalg.norm(h_k @ t[:, i]) ** 2)
                  for i in range(cfg.K)]
        active_noise = cfg.sigma2_a * sum(
            abs(phi[i]) ** 2
            * float(np.linalg.norm(channel.H_rt_hat[k][:, i]) ** 2)
            for i in range(cfg.La)
        )
        expected[k] = powers[k] / (
            sum(powers) - powers[k] + active_noise + cfg.sigma2_k + 1e-15
        )
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


class _SeedObserved(RuntimeError):
    pass


def test_action_is_evaluated_on_the_channel_seed_exposed_in_its_state():
    saved = cfg.channel_coherence_hypothesis_id
    cfg.channel_coherence_hypothesis_id = (
        PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID
    )
    try:
        environment = HRISUAVISACEnv(
            sigma2_scee=0.0, seed=404, sensing_rate_min=0.0
        )
        episode_seed = environment._episode_channel_seed

        def observe_seed(_previous, _proposed, channel_seed, *_args):
            assert channel_seed == episode_seed
            raise _SeedObserved

        with patch.object(
            environment, "_find_feasible_controls", side_effect=observe_seed
        ):
            with pytest.raises(_SeedObserved):
                environment.step(np.zeros(environment.action_dim))
    finally:
        cfg.channel_coherence_hypothesis_id = saved


def test_environment_checkpoint_binds_channel_coherence_identity():
    environment = HRISUAVISACEnv(
        sigma2_scee=0.0, seed=405, sensing_rate_min=0.0
    )
    state = environment.checkpoint_state()
    assert state["episode_channel_seed"] == environment._episode_channel_seed
    state["channel_coherence_hypothesis_id"] = "episode_block_fading"
    with pytest.raises(ValueError, match="channel-coherence"):
        environment.restore_checkpoint_state(state)


def test_statistical_csi_state_omits_instantaneous_channel_coefficients():
    perfect = HRISUAVISACEnv(
        sigma2_scee=0.0, seed=406, sensing_rate_min=0.0
    )
    statistical = HRISUAVISACEnv(
        sigma2_scee=cfg.rho_nmse, seed=406, sensing_rate_min=0.0
    )
    channel_features = 2 * cfg.N * cfg.M * (cfg.U + cfg.K)

    assert perfect.observe_instantaneous_csi is True
    assert statistical.observe_instantaneous_csi is False
    assert perfect.state_dim - statistical.state_dim == channel_features
    assert perfect.reset().shape == (perfect.state_dim,)
    assert statistical.reset().shape == (statistical.state_dim,)


def test_partial_csi_explicitly_retains_masked_channel_observation_shape():
    partial = HRISUAVISACEnv(
        sigma2_scee=cfg.rho_nmse,
        seed=407,
        sensing_rate_min=0.0,
        csi_observation_fraction=0.5,
        observe_instantaneous_csi=True,
    )
    assert partial.observe_instantaneous_csi is True
    assert partial.reset().shape == (partial.state_dim,)
    assert 0 < np.count_nonzero(partial.csi_observation_mask) < (
        partial.csi_observation_mask.size
    )
