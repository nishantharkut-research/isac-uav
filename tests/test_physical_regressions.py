import copy
from unittest.mock import patch

import numpy as np
import pytest

from ambiguity_hypotheses import (
    PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID,
    PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID,
)
from channel_model import (
    ChannelSet,
    active_hris_steering,
    channel_gain,
    gen_bs_hris,
    hris_element_coordinates,
    hris_partition_indices,
    plos,
    upa_steering,
)
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


def test_primary_active_beampattern_matches_paper_eq9_subarray():
    direction = (0.2, -0.3)
    active, _passive = hris_partition_indices(cfg.L, cfg.La)
    assert cfg.hris_partition_mode == "paper_rectangular_subarrays_v1"
    np.testing.assert_array_equal(active, np.arange(cfg.La))
    expected = upa_steering(cfg.Lax, cfg.Lay, *direction)
    np.testing.assert_array_equal(active_hris_steering(*direction), expected)


def test_los_symbol_role_hypotheses_are_explicit_and_exact():
    saved = cfg.los_probability_hypothesis_id
    elevation = 20.0
    try:
        cfg.los_probability_hypothesis_id = (
            "paper_unnumbered_los_literal_roles"
        )
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


def test_primary_los_branch_matches_printed_formula_and_high_los_text():
    saved = cfg.los_probability_hypothesis_id
    try:
        cfg.los_probability_hypothesis_id = (
            PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID
        )
        assert PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID == (
            "paper_unnumbered_los_literal_roles"
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


def test_scee_nmse_changes_covariance_without_resampling_channel_estimates():
    zero_error = ChannelSet(*_scene(), sigma2=0.0, seed=2031)
    imperfect = ChannelSet(*_scene(), sigma2=0.25, seed=2031)

    for field in ("G_brj_hat", "G_bri_hat"):
        assert np.array_equal(getattr(zero_error, field), getattr(imperfect, field))
    for field in ("H_bu_hat", "H_bk_hat", "H_ru_hat", "H_rt_hat"):
        for exact, noisy in zip(
            getattr(zero_error, field), getattr(imperfect, field)
        ):
            assert np.array_equal(exact, noisy)

    uav, users, targets = _scene()
    assert imperfect.sv_brj == pytest.approx(
        0.25 * channel_gain(cfg.BS_pos, uav)
    )
    assert imperfect.sv_bri == pytest.approx(
        0.25 * channel_gain(cfg.BS_pos, uav)
    )
    assert imperfect.sv_bu == pytest.approx([
        0.25 * channel_gain(cfg.BS_pos, position) for position in users
    ])
    assert imperfect.sv_bk == pytest.approx([
        0.25 * channel_gain(cfg.BS_pos, position) for position in targets
    ])
    assert imperfect.sv_ruj == pytest.approx([
        0.25 * channel_gain(uav, position) for position in users
    ])
    assert imperfect.sv_rti == pytest.approx([
        0.25 * channel_gain(uav, position) for position in targets
    ])

    independent_fading = ChannelSet(*_scene(), sigma2=0.25, seed=2032)
    assert independent_fading.sv_bu == pytest.approx(imperfect.sv_bu)
    assert independent_fading.sv_bk == pytest.approx(imperfect.sv_bk)
    assert independent_fading.sv_brj == pytest.approx(imperfect.sv_brj)
    assert independent_fading.sv_bri == pytest.approx(imperfect.sv_bri)
    assert independent_fading.sv_ruj == pytest.approx(imperfect.sv_ruj)
    assert independent_fading.sv_rti == pytest.approx(imperfect.sv_rti)


def test_active_and_passive_channels_are_disjoint_slices_of_one_full_upa():
    seed = 2040
    uav, users, targets = _scene()
    channel = ChannelSet(uav, users, targets, sigma2=0.0, seed=seed)
    active = np.asarray(channel.active_element_indices)
    passive = np.asarray(channel.passive_element_indices)
    assert active.size == cfg.La
    assert passive.size == cfg.Lp
    assert not set(active) & set(passive)
    assert sorted(np.concatenate([active, passive]).tolist()) == list(
        range(cfg.L)
    )

    full = gen_bs_hris(
        uav,
        cfg.L,
        cfg.Lx,
        cfg.Ly,
        np.random.default_rng(np.random.SeedSequence([seed, 1, 0])),
        element_coordinates=hris_element_coordinates(),
    )
    assert np.array_equal(channel.G_bri_hat, full[active])
    assert np.array_equal(channel.G_brj_hat, full[passive])


def test_interleaved_full_upa_remains_an_explicit_sensitivity():
    saved = cfg.hris_partition_mode
    try:
        cfg.hris_partition_mode = "uniformly_interleaved_full_upa_v1"
        active, passive = hris_partition_indices(cfg.L, cfg.La)
        expected = upa_steering(cfg.Lx, cfg.Ly, 0.2, -0.3)[active]
        np.testing.assert_array_equal(active_hris_steering(0.2, -0.3), expected)
        assert not set(active) & set(passive)
    finally:
        cfg.hris_partition_mode = saved


def test_primary_active_steering_supports_passive_only_configuration():
    saved = (cfg.L, cfg.La, cfg.Lp, cfg.Lax, cfg.Lay, cfg.Lpx, cfg.Lpy)
    try:
        cfg.L, cfg.La, cfg.Lp = 4, 0, 4
        cfg.Lax, cfg.Lay = 1, 1
        cfg.Lpx, cfg.Lpy = 2, 2
        response = active_hris_steering(0.25, -0.5)
        assert response.shape == (0,)
        assert hris_element_coordinates().shape == (4, 2)
    finally:
        (
            cfg.L, cfg.La, cfg.Lp, cfg.Lax, cfg.Lay, cfg.Lpx, cfg.Lpy
        ) = saved


def test_figure11_rejects_passive_only_configuration_as_not_applicable():
    from simulation import fig11_beampattern

    saved = cfg.La
    try:
        cfg.La = 0
        with pytest.raises(ValueError, match="active HRIS"):
            fig11_beampattern()
    finally:
        cfg.La = saved


def test_zero_error_rate_formula_follows_declared_hypothesis():
    saved = cfg.rate_model_hypothesis_id
    channel = ChannelSet(*_scene(), sigma2=0.0, seed=2028)
    theta = np.ones(cfg.Lp, dtype=complex)
    phi = np.ones(cfg.La, dtype=complex)
    C, T = build_beamformers(channel, theta, phi)
    try:
        cfg.rate_model_hypothesis_id = (
            "continuous_matrix_scee_all_csi_qualities"
        )
        primary = compute_rates(channel, theta, phi, C, T)
        matrix = compute_rates(
            channel, theta, phi, C, T, use_scee_model=True
        )

        cfg.rate_model_hypothesis_id = (
            "paper_split_scalar_pcsi_matrix_icsi"
        )
        paper_split = compute_rates(channel, theta, phi, C, T)
        scalar = compute_rates(
            channel, theta, phi, C, T, use_scee_model=False
        )
    finally:
        cfg.rate_model_hypothesis_id = saved

    for observed, expected in zip(primary, matrix):
        assert np.allclose(observed, expected, rtol=0.0, atol=0.0)
    for observed, expected in zip(paper_split, scalar):
        assert np.allclose(observed, expected, rtol=0.0, atol=0.0)
    assert not np.allclose(primary[0], paper_split[0])


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
        sigma2_scee=cfg.rho_nmse,
        seed=406,
        sensing_rate_min=0.0,
        observe_instantaneous_csi=False,
    )
    primitive_complex_coefficients = (
        cfg.L * cfg.M
        + cfg.U * cfg.N * (cfg.Lp + cfg.M)
        + cfg.K * cfg.N * (cfg.La + cfg.M)
    )
    channel_features = 2 * primitive_complex_coefficients

    assert perfect.observe_instantaneous_csi is True
    assert statistical.observe_instantaneous_csi is False
    assert perfect.state_dim - statistical.state_dim == channel_features + 1
    assert perfect.reset().shape == (perfect.state_dim,)
    assert statistical.reset().shape == (statistical.state_dim,)


def test_instantaneous_csi_suffix_contains_primitive_channels_not_current_composites():
    environment = HRISUAVISACEnv(
        sigma2_scee=0.0, seed=408, sensing_rate_min=0.0
    )
    channel_feature_count = environment.csi_observation_mask.size
    before = environment._build_state()[-channel_feature_count:].copy()
    environment.theta_phase = np.linspace(0.1, 1.1, cfg.Lp)
    environment.phi_phase = np.linspace(0.2, 1.2, cfg.La)
    after = environment._build_state()[-channel_feature_count:]

    assert np.array_equal(before, after)


def test_instantaneous_csi_state_exposes_absolute_channel_scale():
    environment = HRISUAVISACEnv(
        sigma2_scee=0.0, seed=411, sensing_rate_min=0.0
    )
    state = environment.reset()
    assert environment.csi_scale_feature_index is not None
    expected = np.log10(environment._ch_scale) / 10.0
    assert state[environment.csi_scale_feature_index] == pytest.approx(expected)


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
    complex_count = partial.csi_observation_mask.size // 2
    np.testing.assert_array_equal(
        partial.csi_observation_mask[:complex_count],
        partial.csi_observation_mask[complex_count:],
    )


def test_state_exposes_time_geometry_and_target_velocity_for_finite_horizon_mdp():
    environment = HRISUAVISACEnv(
        sigma2_scee=0.0, seed=410, sensing_rate_min=0.0
    )
    state = environment.reset()

    assert state[environment.time_feature_index] == pytest.approx(0.0)
    geometry = state[
        environment.time_feature_index + 1:
        environment.precoder_feature_index
    ]
    assert geometry.size == 2 * (cfg.U + cfg.K) + 2 * cfg.K
    assert np.all(np.isfinite(geometry))

    environment.t = cfg.N_slots // 2
    progressed = environment._build_state()
    assert progressed[environment.time_feature_index] == pytest.approx(0.5)


def test_terminal_transition_does_not_move_targets_without_refreshing_channel():
    environment = HRISUAVISACEnv(
        sigma2_scee=0.0, seed=409, sensing_rate_min=0.0
    )
    environment.t = cfg.N_slots - 1
    controls = environment._snapshot_controls()
    assessment = environment._assess_current_solution(
        environment.C, environment.T, environment.beam_info
    )
    target_positions = [position.copy() for position in environment.pos_tes]

    with patch.object(
        environment,
        "_find_feasible_controls",
        return_value=(controls, assessment, 1.0),
    ):
        _state, _reward, done, _info = environment.step(
            np.zeros(environment.action_dim)
        )

    assert done is True
    for before, after in zip(target_positions, environment.pos_tes):
        np.testing.assert_array_equal(after, before)


def test_physical_baseline_scores_reset_channel_before_first_motion():
    import simulation

    events = []

    class FakeEnvironment:
        def __init__(self, *args, **kwargs):
            self.ch = "slot-0"
            self._episode_channel_seed = 123
            self._channel_rng = np.random.default_rng(99)

        def reset(self):
            events.append("reset")

        def _advance_targets(self):
            events.append("advance")

        def _refresh_channel(self, channel_seed=None, slot_time_s=None):
            events.append(("refresh", int(channel_seed), float(slot_time_s)))
            self.ch = "slot-1"

    def allocate(channel, _theta, _phi):
        events.append(("allocate", channel))
        return np.zeros((1, 1)), np.zeros((1, 1)), {}

    def rates(channel, *_args):
        events.append(("rate", channel))
        return np.ones(1), np.zeros(1), np.ones(1), np.zeros(1)

    with patch.object(simulation, "HRISUAVISACEnv", FakeEnvironment), \
            patch.object(simulation, "select_feasible_beamformers", allocate), \
            patch.object(simulation, "compute_rates", rates):
        simulation._physical_baseline_rate(
            "no_ris", n_eval=1, n_steps=2, strict_feasibility=False
        )

    assert events[:3] == [
        "reset",
        ("allocate", "slot-0"),
        ("rate", "slot-0"),
    ]
    assert events[3] == "advance"
    assert events[4][0] == "refresh"
    assert events[4][2] == pytest.approx(cfg.delta_T)
    assert events[5:] == [
        ("allocate", "slot-1"),
        ("rate", "slot-1"),
    ]
