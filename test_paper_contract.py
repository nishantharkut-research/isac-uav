"""Fast paper contract checks for the HRIS-UAV-ISAC reproduction.

Run from this directory:
    python test_paper_contract.py

These checks avoid long DRL training and AO sweeps. They verify that the
implementation follows the visually inspected paper equations and Tables III/IV.
"""
import importlib
import sys

import numpy as np

from config import cfg


def test_paper_table_defaults_are_active():
    assert cfg.M == 4
    assert cfg.N == 2
    assert cfg.U == 3
    assert cfg.K == 3
    assert cfg.L == 40
    assert cfg.La == 20
    assert cfg.Lp == 20
    assert cfg.PT_dB == 10
    assert cfg.PT_dBm == 10
    assert np.isclose(cfg.PT, 0.01)
    assert cfg.power_hypothesis_id == "paper_db_as_dbm"
    assert np.isclose(cfg.fc, 3e9)
    assert np.isclose(cfg.d_spacing, 0.5)
    assert np.isclose(cfg.omega_0, 1e-3)
    assert not hasattr(cfg, "nlos_factor")
    assert cfg.N_slots == 300
    assert cfg.T_steps == 300
    assert cfg.B_batch == 32
    assert cfg.lr_actor == 1e-3
    assert cfg.lr_critic == 1e-3
    assert cfg.gamma == 0.9
    assert cfg.replay_size == 10_000
    assert hasattr(cfg, "P_a_max")


def test_service_area_matches_the_published_fig10_axes():
    assert cfg.x_min == 0.0 and cfg.x_max == 300.0
    assert cfg.y_min == 0.0 and cfg.y_max == 200.0
    assert cfg.z_UAV == 50.0


def test_ue_region_is_the_separate_100_m_square_reported_on_page_10():
    from environment import HRISUAVISACEnv

    assert cfg.ue_x_max - cfg.ue_x_min == 100.0
    assert cfg.ue_y_max - cfg.ue_y_min == 100.0
    env = HRISUAVISACEnv(sigma2_scee=0.0, seed=112)
    positions = np.asarray(env.pos_ues, dtype=float)
    assert np.all((positions[:, 0] >= cfg.ue_x_min)
                  & (positions[:, 0] <= cfg.ue_x_max))
    assert np.all((positions[:, 1] >= cfg.ue_y_min)
                  & (positions[:, 1] <= cfg.ue_y_max))


def test_environment_dimensions_follow_paper_defaults():
    from environment import HRISUAVISACEnv

    env = HRISUAVISACEnv(sigma2_scee=0.0, seed=11)
    expected_state_dim = (
        2 * cfg.Lp + 2 * cfg.La + 3
        + 2 * cfg.M * (cfg.U + cfg.K)
        + 2 * cfg.N * cfg.M * (cfg.U + cfg.K)
    )
    expected_action_dim = cfg.Lp + 2 * cfg.La + 2
    assert env.state_dim == expected_state_dim
    assert env.action_dim == expected_action_dim
    assert expected_state_dim == 227
    assert expected_action_dim == 62


def test_paper_objective_and_sensing_constraint_match_eq16():
    from paper_contract import joint_sum_rate, sensing_rate_floor_penalty

    rate_u = np.array([0.2, 0.7, 1.0])
    rate_k = np.array([0.1, 0.6, 0.8])
    assert joint_sum_rate(rate_u, rate_k) == float(np.sum(rate_u))

    # Published Eq. (16) has R_k >= R_min but no R_u floor. The extra R_u
    # constraint exists only in the separately supplied extension PNG.
    penalty = sensing_rate_floor_penalty(rate_k, r_k_min=0.5)
    assert np.isclose(penalty, 0.4)


def test_passive_cascade_power_uses_coherent_element_sum_from_eq16c():
    from paper_contract import cascade_power_16c

    class CancellationFixture:
        # The two element contributions are +1 and -1.  A coherent sum has
        # zero power under the norm-of-sum expression printed in Eq. (16c).
        H_ru_hat = [np.array([[1.0, 1.0]], dtype=complex)]
        G_brj_hat = np.array([[1.0], [1.0]], dtype=complex)

    power = cascade_power_16c(
        CancellationFixture(), np.array([1.0, -1.0], dtype=complex)
    )
    assert np.isclose(power, 0.0)


def test_mobile_targets_are_seeded_bounded_and_reset_to_their_initial_scene():
    from environment import HRISUAVISACEnv
    from research_objectives import ObjectiveSpec

    objective = ObjectiveSpec.paper(target_rate_min=0.0)
    env_a = HRISUAVISACEnv(sigma2_scee=0.0, seed=314, objective_spec=objective)
    env_b = HRISUAVISACEnv(sigma2_scee=0.0, seed=314, objective_spec=objective)
    initial = np.asarray(env_a.pos_tes, dtype=float).copy()
    initial_velocity = env_a._tes_velocity.copy()
    assert np.allclose(initial, np.asarray(env_b.pos_tes, dtype=float))

    action = np.zeros(env_a.action_dim, dtype=float)
    env_a.step(action)
    env_b.step(action)
    moved_a = np.asarray(env_a.pos_tes, dtype=float)
    moved_b = np.asarray(env_b.pos_tes, dtype=float)

    assert not np.allclose(moved_a[:, :2], initial[:, :2])
    assert np.allclose(moved_a, moved_b)
    assert np.all((moved_a[:, 0] >= cfg.x_min) & (moved_a[:, 0] <= cfg.x_max))
    assert np.all((moved_a[:, 1] >= cfg.y_min) & (moved_a[:, 1] <= cfg.y_max))
    assert np.allclose(moved_a[:, 2], 0.0)

    env_a._tes_velocity *= -1.0
    env_a.reset()
    assert np.allclose(np.asarray(env_a.pos_tes, dtype=float), initial)
    assert np.allclose(env_a._tes_velocity, initial_velocity)


def test_path_loss_exponent_uses_paper_eq5_with_explicit_endpoint_mapping():
    from channel_model import path_loss_exponent

    saved = (cfg.zeta_los, cfg.zeta_nlos)
    try:
        cfg.zeta_los = 2.0
        cfg.zeta_nlos = 3.5

        # Eq. (5): zeta(phi) = P_LoS*zeta_e + zeta_0.  The executable
        # endpoint parameterization maps zeta_e=(zeta_LoS-zeta_NLoS) and
        # zeta_0=zeta_NLoS, so the two forms are algebraically identical.
        assert np.isclose(path_loss_exponent(0.0), 3.5)
        assert np.isclose(path_loss_exponent(1.0), 2.0)
        assert np.isclose(path_loss_exponent(0.25), 3.125)
    finally:
        cfg.zeta_los, cfg.zeta_nlos = saved


def test_eq8_doppler_applies_to_moving_target_los_but_not_ue_channels():
    from channel_model import ChannelSet, target_doppler_frequency

    pos_uav = np.array([150.0, 100.0, cfg.z_UAV])
    pos_ues = [np.array([20.0 + 10.0 * u, 30.0, 0.0]) for u in range(cfg.U)]
    pos_tes = [np.array([80.0 + 10.0 * k, 140.0, 0.0]) for k in range(cfg.K)]
    velocities = np.tile(np.array([1.0, 0.25, 0.0]), (cfg.K, 1))
    time_s = 0.37

    expected_frequency = float(
        np.dot(
            velocities[0],
            (pos_uav - pos_tes[0]) / np.linalg.norm(pos_uav - pos_tes[0]),
        ) / cfg.wavelength
    )
    assert np.isclose(
        target_doppler_frequency(pos_uav, pos_tes[0], velocities[0]),
        expected_frequency,
    )

    static = ChannelSet(
        pos_uav, pos_ues, pos_tes, sigma2=0.0, seed=812,
        slot_time_s=time_s,
    )
    moving = ChannelSet(
        pos_uav, pos_ues, pos_tes, sigma2=0.0, seed=812,
        slot_time_s=time_s, target_velocities=velocities,
    )
    assert all(
        np.allclose(left, right)
        for left, right in zip(static.H_ru_hat, moving.H_ru_hat)
    )
    assert any(
        not np.allclose(left, right)
        for left, right in zip(static.H_rt_hat, moving.H_rt_hat)
    )


def test_downlink_sinr_matches_paper_eq12_to_eq15_for_n2_fixture():
    from sinr_rate import compute_rates

    class FixtureChannel:
        def __init__(self):
            eye_rows = []
            for stream in range(cfg.U):
                h = np.zeros((cfg.N, cfg.M), dtype=complex)
                h[0, stream] = 1.0
                eye_rows.append(h)
            self._hu = eye_rows
            self._hk = [h.copy() for h in eye_rows]
            self.H_ru_hat = [np.zeros((cfg.N, cfg.Lp), dtype=complex)
                             for _ in range(cfg.U)]
            self.H_rt_hat = [np.zeros((cfg.N, cfg.La), dtype=complex)
                             for _ in range(cfg.K)]
            self.G_brj_hat = np.zeros((cfg.Lp, cfg.M), dtype=complex)
            self.G_bri_hat = np.zeros((cfg.La, cfg.M), dtype=complex)
            self.sv_bu = [0.0] * cfg.U
            self.sv_bk = [0.0] * cfg.K
            self.sv_brj = self.sv_bri = 0.0
            self.sv_ruj = [0.0] * cfg.U
            self.sv_rti = [0.0] * cfg.K

        def composite_ue(self, theta):
            return self._hu

        def composite_te(self, phi):
            return self._hk

    ch = FixtureChannel()
    C = np.zeros((cfg.M, cfg.U), dtype=complex)
    T = np.zeros((cfg.M, cfg.K), dtype=complex)
    for idx in range(cfg.U):
        C[idx, idx] = 1.0
    for idx in range(cfg.K):
        T[idx, idx] = 1.0

    rate_u, rate_k, sinr_u, sinr_k = compute_rates(
        ch,
        np.ones(cfg.Lp, dtype=complex),
        np.zeros(cfg.La, dtype=complex),
        C,
        T,
        use_scee_model=False,
    )

    # Eq. (12): each UE has signal 1, zero inter-user interference, and the
    # three target beams contribute 1 each through their H_k channels.
    expected_sinr = 1.0 / (3.0 + cfg.sigma2_u)
    expected_rate = np.log2(1.0 + expected_sinr)
    assert np.allclose(sinr_u, expected_sinr)
    assert np.allclose(rate_u, expected_rate)

    # Eq. (14): each target has signal 1, zero inter-target interference, and
    # the three communication beams contribute 1 each through H_u.
    assert np.allclose(sinr_k, expected_sinr)
    assert np.allclose(rate_k, expected_rate)


def test_scee_rate_uses_matrix_logdet_eq21():
    from sinr_rate import compute_rates

    class FakeChannel:
        sv_bu = [0.5, 0.0, 0.0]
        sv_bk = [0.0, 0.0, 0.0]
        sv_brj = 0.0
        sv_bri = 0.0
        sv_ruj = [0.0, 0.0, 0.0]
        sv_rti = [0.0, 0.0, 0.0]
        G_brj_hat = np.zeros((cfg.Lp, cfg.M), dtype=complex)
        G_bri_hat = np.zeros((cfg.La, cfg.M), dtype=complex)
        H_ru_hat = [
            np.zeros((cfg.N, cfg.Lp), dtype=complex) for _ in range(cfg.U)
        ]
        H_rt_hat = [
            np.zeros((cfg.N, cfg.La), dtype=complex) for _ in range(cfg.K)
        ]

        def composite_ue(self, theta):
            H = np.zeros((cfg.N, cfg.M), dtype=complex)
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            return [H.copy() for _ in range(cfg.U)]

        def composite_te(self, phi):
            return [
                np.zeros((cfg.N, cfg.M), dtype=complex)
                for _ in range(cfg.K)
            ]

    C = np.zeros((cfg.M, cfg.U), dtype=complex)
    C[0, 0], C[1, 1], C[2, 2] = 1.0, 1.0, 1.0
    T = np.zeros((cfg.M, cfg.K), dtype=complex)
    theta = np.ones(cfg.Lp, dtype=complex)
    phi = np.zeros(cfg.La, dtype=complex)

    rate_u, _, _, _ = compute_rates(FakeChannel(), theta, phi, C, T)
    # J_0 = diag(1.5, 2.5): direct SCEE contributes
    # sigma_bu^2 Tr(C C^H) I = 0.5 * 3 I, while C_1 interferes on antenna 2.
    expected = np.log2(1.0 + 1.0 / (1.5 + cfg.sigma2_u))
    assert np.isclose(rate_u[0], expected)


def test_perfect_csi_default_rate_uses_zero_error_eq21_eq25_path():
    from channel_model import ChannelSet
    from sinr_rate import (
        _compute_scee_rates,
        build_beamformers,
        compute_rates,
    )

    pos_uav = np.array([150.0, 100.0, cfg.z_UAV])
    pos_ues = [np.array([30.0 + 50.0 * i, 40.0, 0.0]) for i in range(cfg.U)]
    pos_tes = [np.array([40.0 + 50.0 * i, 160.0, 0.0]) for i in range(cfg.K)]
    channel = ChannelSet(pos_uav, pos_ues, pos_tes, sigma2=0.0, seed=2112)
    theta = np.ones(cfg.Lp, dtype=complex)
    phi = np.full(cfg.La, cfg.alpha_max / 2.0, dtype=complex)
    C, T = build_beamformers(channel, theta, phi)

    default_rates = compute_rates(channel, theta, phi, C, T)
    robust_zero_error_rates = _compute_scee_rates(
        channel,
        theta,
        phi,
        C,
        T,
        channel.composite_ue(theta),
        channel.composite_te(phi),
    )
    for default, expected in zip(default_rates, robust_zero_error_rates):
        assert np.allclose(default, expected)


def test_scee_estimate_does_not_inflate_channel_power():
    from channel_model import add_scee

    rng = np.random.default_rng(77)
    true_channel = (
        rng.standard_normal(100_000) + 1j * rng.standard_normal(100_000)
    ) / np.sqrt(2.0)
    estimated = add_scee(true_channel, rho_nmse=0.3, rng=rng)
    power_ratio = np.mean(np.abs(estimated) ** 2) / np.mean(np.abs(true_channel) ** 2)
    assert np.isclose(power_ratio, 1.0 / 1.3, atol=0.02)


def test_scee_nmse_is_relative_to_estimated_channel_as_defined_on_page_10():
    from channel_model import add_scee

    rng = np.random.default_rng(731)
    true_channel = (
        rng.standard_normal(500_000) + 1j * rng.standard_normal(500_000)
    ) / np.sqrt(2.0)
    rho = 0.3
    estimated = add_scee(true_channel, rho_nmse=rho, rng=rng)
    error = true_channel - estimated
    measured_nmse = (
        np.mean(np.abs(error) ** 2) / np.mean(np.abs(estimated) ** 2)
    )
    assert np.isclose(measured_nmse, rho, atol=0.02)


def test_active_power_constraint_uses_aggregate_eq18_power():
    from paper_contract import active_power_penalty

    penalty, ok = active_power_penalty(0.7, p_a_max=0.5)
    assert np.isclose(penalty, 0.2)
    assert ok is False

    penalty, ok = active_power_penalty(0.4, p_a_max=0.5)
    assert penalty == 0.0
    assert ok is True


def test_constraint_excess_is_normalized_before_combining_with_rate_reward():
    from paper_contract import normalized_constraint_violation

    assert np.isclose(normalized_constraint_violation(2e-10, 1e-10), 2.0)
    assert np.isclose(normalized_constraint_violation(3e-13, 6e-13), 0.5)
    assert normalized_constraint_violation(0.0, 1e-10) == 0.0


def test_eq18_active_power_sums_elements_and_adds_noise_once():
    from sinr_rate import active_tx_power

    class ActivePowerFixture:
        G_bri_hat = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex)

    old_pt, old_noise, old_la = cfg.PT, cfg.sigma2_qa, cfg.La
    try:
        cfg.PT = 2.0
        cfg.sigma2_qa = 0.25
        cfg.La = 2
        T = np.eye(2, dtype=complex)
        phi = np.array([3.0, 4.0], dtype=complex)
        expected = 9.0 + 4.0 * 16.0 + 0.25
        assert np.isclose(active_tx_power(ActivePowerFixture(), T, phi), expected)
    finally:
        cfg.PT, cfg.sigma2_qa, cfg.La = old_pt, old_noise, old_la


def test_eq18_does_not_apply_transmit_power_twice_to_power_normalized_T():
    from sinr_rate import active_tx_power

    class ActivePowerFixture:
        G_bri_hat = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex)

    old_pt, old_noise, old_la = cfg.PT, cfg.sigma2_qa, cfg.La
    try:
        cfg.sigma2_qa = 0.25
        cfg.La = 2
        T = np.eye(2, dtype=complex)
        phi = np.array([3.0, 4.0], dtype=complex)

        cfg.PT = 1.0
        first = active_tx_power(ActivePowerFixture(), T, phi)
        cfg.PT = 10.0
        second = active_tx_power(ActivePowerFixture(), T, phi)

        assert np.isclose(first, second)
        assert np.isclose(first, 9.0 + 4.0 * 16.0 + 0.25)
    finally:
        cfg.PT, cfg.sigma2_qa, cfg.La = old_pt, old_noise, old_la


def test_eq18_is_zero_when_the_hris_has_no_active_elements():
    from sinr_rate import active_tx_power

    class PassiveOnlyFixture:
        G_bri_hat = np.zeros((0, cfg.M), dtype=complex)

    old_la, old_noise = cfg.La, cfg.sigma2_qa
    try:
        cfg.La = 0
        cfg.sigma2_qa = 0.25
        T = np.zeros((cfg.M, cfg.K), dtype=complex)
        assert active_tx_power(PassiveOnlyFixture(), T, np.zeros(0)) == 0.0
    finally:
        cfg.La, cfg.sigma2_qa = old_la, old_noise


def test_beamformer_selection_enforces_aggregate_active_power_constraint():
    import sinr_rate

    saved = (
        sinr_rate.build_beam_directions,
        sinr_rate.compute_rates,
        sinr_rate.sensing_only_sinr,
        sinr_rate.active_tx_power,
        cfg.P_a_max,
    )
    try:
        sinr_rate.build_beam_directions = lambda *args, **kwargs: (
            np.eye(cfg.M, cfg.U, dtype=complex),
            np.eye(cfg.M, cfg.K, dtype=complex),
        )
        sinr_rate.compute_rates = lambda ch, theta, phi, C, T: (
            np.full(cfg.U, float(np.real(C[0, 0]))),
            np.ones(cfg.K),
            np.ones(cfg.U),
            np.ones(cfg.K),
        )
        sinr_rate.sensing_only_sinr = lambda ch, phi, T: np.ones(cfg.K)
        sinr_rate.active_tx_power = lambda ch, T, phi: float(
            np.linalg.norm(T, "fro") ** 2
        )
        cfg.P_a_max = 0.5

        _, _, info = sinr_rate.select_feasible_beamformers(
            object(), np.ones(cfg.Lp), np.ones(cfg.La)
        )
        assert 0.0 < info["comm_power_frac"] < 1.0
        assert info["active_power_feasible"] is True
        assert info["active_power_deficit"] == 0.0
    finally:
        (sinr_rate.build_beam_directions,
         sinr_rate.compute_rates,
         sinr_rate.sensing_only_sinr,
         sinr_rate.active_tx_power,
         cfg.P_a_max) = saved


def test_uav_episode_closes_exactly_and_respects_step_distance():
    from environment import HRISUAVISACEnv
    from research_objectives import ObjectiveSpec

    original_slots = cfg.N_slots
    cfg.N_slots = 4
    try:
        env = HRISUAVISACEnv(
            sigma2_scee=0.0,
            seed=5,
            objective_spec=ObjectiveSpec.paper(target_rate_min=0.0),
        )
        env.reset()
        positions = [env.uav_pos[:2].copy()]
        action = np.zeros(env.action_dim)
        action[-2:] = [1.0, 1.0]
        done = False
        while not done:
            _, _, done, info = env.step(action)
            positions.append(info["uav_pos"][:2].copy())

        start = env._uav_start[:2]
        assert np.linalg.norm(positions[-1] - start) < 1e-9
        for prev, cur in zip(positions, positions[1:]):
            assert np.linalg.norm(cur - prev) <= cfg.D_max + 1e-9
    finally:
        cfg.N_slots = original_slots


def test_environment_reward_is_paper_communication_sum_rate():
    from environment import HRISUAVISACEnv
    from research_objectives import ObjectiveSpec

    env = HRISUAVISACEnv(
        sigma2_scee=0.0,
        seed=9,
        objective_spec=ObjectiveSpec.paper(target_rate_min=0.0),
    )
    _, reward, _, info = env.step(np.zeros(env.action_dim))
    assert np.isclose(reward, info["sum_rate"])
    assert np.isclose(
        info["joint_rate"], info["sum_rate"] + info["sum_sensing"]
    )
    assert "diagnostic_penalized_reward" not in info
    assert "diagnostic_constraint_penalty" not in info
    assert "P_ai" in info
    assert np.isscalar(info["P_ai"])
    assert isinstance(info["active_power_ok"], bool)


def test_ao_module_imports_without_scipy_at_import_time():
    module = importlib.import_module("ao_optimizer")
    assert hasattr(module, "run_ao")


def test_ao_phase_objective_uses_paper_comm_objective():
    ao_optimizer = importlib.import_module("ao_optimizer")

    class FakeChannel:
        H_ru_hat = [np.zeros((cfg.N, cfg.Lp), dtype=complex) for _ in range(cfg.U)]
        G_brj_hat = np.zeros((cfg.Lp, cfg.M), dtype=complex)

    old_compute_rates = ao_optimizer.compute_rates
    old_active_power = ao_optimizer.active_tx_power
    old_current_phi = getattr(ao_optimizer, "_current_phi", None)
    try:
        ao_optimizer.compute_rates = lambda ch, theta, phi, c, t: (
            np.ones(cfg.U),
            2.0 * np.ones(cfg.K),
            np.ones(cfg.U),
            np.ones(cfg.K),
        )
        ao_optimizer.active_tx_power = lambda ch, t, phi: 0.0
        ao_optimizer._current_phi = np.ones(cfg.La, dtype=complex)
        obj = ao_optimizer._scee_phase_objective(
            np.zeros(cfg.Lp),
            FakeChannel(),
            np.zeros((cfg.M, cfg.U), dtype=complex),
            np.zeros((cfg.M, cfg.K), dtype=complex),
            is_passive=True,
        )
    finally:
        ao_optimizer.compute_rates = old_compute_rates
        ao_optimizer.active_tx_power = old_active_power
        if old_current_phi is None and hasattr(ao_optimizer, "_current_phi"):
            delattr(ao_optimizer, "_current_phi")
        elif old_current_phi is not None:
            ao_optimizer._current_phi = old_current_phi

    expected_comm_objective = -float(cfg.U)
    assert np.isclose(obj, expected_comm_objective)


def test_eq18_active_power_envelope_bounds_allowed_gain_and_beamformer():
    from sinr_rate import active_tx_power, active_tx_power_envelope

    class ActivePowerFixture:
        G_bri_hat = np.array([
            [1.0, 0.5j],
            [0.25, 2.0],
        ], dtype=complex)

    old_pt, old_noise, old_la, old_alpha = (
        cfg.PT, cfg.sigma2_qa, cfg.La, cfg.alpha_max,
    )
    try:
        cfg.PT = 2.0
        cfg.sigma2_qa = 0.25
        cfg.La = 2
        cfg.alpha_max = 3.0
        rng = np.random.default_rng(991)
        T = rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))
        T *= np.sqrt(cfg.PT) / np.linalg.norm(T, "fro")
        phi = np.array([1.5 * np.exp(0.3j), 3.0 * np.exp(-0.7j)])
        assert active_tx_power(ActivePowerFixture(), T, phi) <= (
            active_tx_power_envelope(ActivePowerFixture()) + 1e-12
        )
    finally:
        cfg.PT, cfg.sigma2_qa, cfg.La, cfg.alpha_max = (
            old_pt, old_noise, old_la, old_alpha,
        )


def _run():
    tests = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - test runner reports all failures.
            failures.append((test.__name__, exc))
            print(f"[FAIL] {test.__name__}: {type(exc).__name__}: {exc}")

    if failures:
        print(f"\nResult: {len(tests) - len(failures)}/{len(tests)} tests passed")
        return 1
    print(f"\nResult: {len(tests)}/{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run())
