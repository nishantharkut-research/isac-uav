"""Short reproduction-level contract checks.

Run from this directory:
    python test_reproduction_contract.py

These tests intentionally avoid long DRL training and AO sweeps. They verify
that the implementation follows the paper reproduction contract at the
algorithm and orchestration level.
"""
import ast
import inspect
import importlib.util
import sys

import numpy as np

from config import cfg


def test_declarative_figure_contracts_cover_figures_without_result_values():
    from experiment_contract import FIGURE_CONTRACTS

    assert set(FIGURE_CONTRACTS) == set(range(4, 15))
    for number, figure in FIGURE_CONTRACTS.items():
        assert figure.number == number
        assert figure.x_values
        assert figure.curves
        assert not hasattr(figure, "y_values")
        for curve in figure.curves:
            assert curve.label
            assert curve.method in {
                "DTL", "DDPG", "AO_SURROGATE", "PASSIVE_DTL",
                "RANDOM_PHASE", "NO_RIS", "MEASURED_RUNTIME",
                "TRAJECTORY", "BEAMPATTERN",
            }
            assert not hasattr(curve, "values")


def test_transfer_and_partial_csi_assumptions_are_explicit():
    assert cfg.source_scene_seed != cfg.target_scene_seed
    assert 0.0 < cfg.partial_csi_fraction < 1.0


def test_transfer_bundle_uses_distinct_scenes_and_separate_target_ddpg():
    import simulation

    created_envs = []
    trained = []

    class FakeEnv:
        state_dim = 5
        action_dim = 3

        def __init__(self, sigma2_scee=None, seed=None, use_capon=True, **kwargs):
            self.sigma2 = sigma2_scee
            self.seed = seed
            self.use_capon = use_capon
            created_envs.append(self)

    class FakeAgent:
        def __init__(self, label):
            self.label = label

    def fake_ddpg(env, n_episodes, tag, **kwargs):
        result = {"agent": FakeAgent(tag), "elapsed_seconds": 1.0,
                  "comm_rates": [1.0], "diagnostic_rewards": [1.0]}
        trained.append(("DDPG", tag, env.seed, result))
        return result

    def fake_dtl(source, env, n_episodes, tag, **kwargs):
        result = {"agent": FakeAgent(tag), "elapsed_seconds": 0.5,
                  "comm_rates": [1.0], "diagnostic_rewards": [1.0]}
        trained.append(("DTL", tag, env.seed, result, source))
        return result

    old_env = simulation.HRISUAVISACEnv
    old_ddpg = simulation.train_ddpg
    old_dtl = simulation.train_dtl
    try:
        simulation.HRISUAVISACEnv = FakeEnv
        simulation.train_ddpg = fake_ddpg
        simulation.train_dtl = fake_dtl
        bundle = simulation.train_experiment_bundle(
            n_episodes=1,
            variants=("perfect", "imperfect"),
            include_wfb=True,
        )
    finally:
        simulation.HRISUAVISACEnv = old_env
        simulation.train_ddpg = old_ddpg
        simulation.train_dtl = old_dtl

    assert bundle["source"]["agent"] is not bundle["ddpg"]["agent"]
    assert any(kind == "DDPG" and seed == cfg.source_scene_seed
               for kind, _, seed, *_ in trained)
    assert any(kind == "DDPG" and seed == cfg.target_scene_seed
               for kind, _, seed, *_ in trained)
    assert all(
        seed == cfg.target_scene_seed
        for kind, _, seed, *_ in trained if kind == "DTL"
    )
    assert all(
        entry[-1] is bundle["source"]["agent"]
        for entry in trained if entry[0] == "DTL"
    )


def test_transfer_bundle_accepts_explicit_scee_and_sensing_task():
    import simulation

    source = inspect.getsource(simulation.train_experiment_bundle)
    assert "variant_sigmas" in source
    assert "sensing_rate_min" in source
    assert "sensing_only_sinr_min" in source
    assert "sigma2_scee=sigma2" in source
    assert "csi_observation_fraction" in source

    registry_source = inspect.getsource(simulation.TrainingRegistry.get_or_train)
    assert "kwargs" in registry_source
    assert "dict" not in registry_source.split("_configuration_fingerprint", 1)[0]


def test_fig4_plots_target_ddpg_not_transfer_source_agent():
    import simulation

    source = inspect.getsource(simulation.fig4_rewards_vs_episodes)
    assert "train_experiment_bundle" in source
    assert 'bundle["ddpg"]' in source
    assert 'bundle["source"]["comm_rates"]' not in source


def test_partial_csi_is_not_derived_inline_from_imperfect_csi():
    from pathlib import Path

    source = Path(__file__).with_name("simulation.py").read_text(encoding="utf-8-sig")
    compact = source.replace(" ", "")
    assert "rho_nmse*0.5" not in compact


def test_proposed_framework_sweeps_do_not_declare_ao_as_dtl():
    from experiment_contract import FIGURE_CONTRACTS

    for number in [7, 8, 9, 13]:
        for curve in FIGURE_CONTRACTS[number].curves:
            if curve.method == "AO_SURROGATE":
                assert curve.label.startswith("AO-")


def test_fig12_contract_is_measured_runtime_not_embedded_paper_bars():
    from experiment_contract import FIGURE_CONTRACTS

    figure = FIGURE_CONTRACTS[12]
    assert {curve.method for curve in figure.curves} == {"MEASURED_RUNTIME"}
    assert "hardware-dependent" in figure.note.lower()


def test_scipy_is_available_for_ao_solver_paths():
    assert importlib.util.find_spec("scipy") is not None


def test_channelset_accepts_seed_and_is_deterministic_without_global_rng():
    from channel_model import ChannelSet

    pos_uav = np.array([50.0, 50.0, cfg.z_UAV])
    pos_ues = [
        np.array([10.0, 20.0, 0.0]),
        np.array([35.0, 70.0, 0.0]),
        np.array([80.0, 15.0, 0.0]),
    ]
    pos_tes = [
        np.array([15.0, 80.0, 0.0]),
        np.array([55.0, 60.0, 0.0]),
        np.array([90.0, 90.0, 0.0]),
    ]

    np.random.seed(1)
    ch_a = ChannelSet(pos_uav, pos_ues, pos_tes, sigma2=cfg.rho_nmse, seed=777)
    np.random.seed(999)
    ch_b = ChannelSet(pos_uav, pos_ues, pos_tes, sigma2=cfg.rho_nmse, seed=777)

    assert np.allclose(ch_a.G_brj_hat, ch_b.G_brj_hat)
    assert np.allclose(ch_a.G_bri_hat, ch_b.G_bri_hat)
    for a, b in zip(ch_a.H_bu_hat, ch_b.H_bu_hat):
        assert np.allclose(a, b)
    for a, b in zip(ch_a.H_bk_hat, ch_b.H_bk_hat):
        assert np.allclose(a, b)
    for a, b in zip(ch_a.H_ru_hat, ch_b.H_ru_hat):
        assert np.allclose(a, b)
    for a, b in zip(ch_a.H_rt_hat, ch_b.H_rt_hat):
        assert np.allclose(a, b)


def test_seeded_direct_links_do_not_change_when_hris_partition_changes():
    from channel_model import ChannelSet
    from config import _factor_upa

    saved = cfg.L, cfg.La, cfg.Lp, cfg.Lax, cfg.Lay, cfg.Lpx, cfg.Lpy
    try:
        pos_uav = np.array([50.0, 50.0, cfg.z_UAV])
        pos_ues = [np.array([10.0 + 20 * i, 20.0, 0.0]) for i in range(cfg.U)]
        pos_tes = [np.array([15.0 + 20 * i, 80.0, 0.0]) for i in range(cfg.K)]

        cfg.L, cfg.La, cfg.Lp = 40, 20, 20
        cfg.Lax, cfg.Lay = _factor_upa(cfg.La)
        cfg.Lpx, cfg.Lpy = _factor_upa(cfg.Lp)
        first = ChannelSet(pos_uav, pos_ues, pos_tes, sigma2=0.0, seed=919)

        cfg.L, cfg.La, cfg.Lp = 30, 3, 27
        cfg.Lax, cfg.Lay = _factor_upa(cfg.La)
        cfg.Lpx, cfg.Lpy = _factor_upa(cfg.Lp)
        second = ChannelSet(pos_uav, pos_ues, pos_tes, sigma2=0.0, seed=919)

        for first_link, second_link in zip(first.H_bu_hat, second.H_bu_hat):
            assert np.array_equal(first_link, second_link)
        for first_link, second_link in zip(first.H_bk_hat, second.H_bk_hat):
            assert np.array_equal(first_link, second_link)
    finally:
        (cfg.L, cfg.La, cfg.Lp, cfg.Lax, cfg.Lay,
         cfg.Lpx, cfg.Lpy) = saved


def test_physical_baselines_use_seeded_common_monte_carlo_realizations():
    import simulation

    first_no_ris = simulation._no_ris_rate(
        cfg.rho_nmse, n_eval=2, seed_base=9_300, n_steps=2,
        strict_feasibility=False,
    )
    np.random.seed(123456)
    second_no_ris = simulation._no_ris_rate(
        cfg.rho_nmse, n_eval=2, seed_base=9_300, n_steps=2,
        strict_feasibility=False,
    )
    first_random = simulation._random_phase_rate(
        cfg.rho_nmse, n_eval=2, seed_base=9_300, n_steps=2,
        strict_feasibility=False,
    )
    np.random.seed(654321)
    second_random = simulation._random_phase_rate(
        cfg.rho_nmse, n_eval=2, seed_base=9_300, n_steps=2,
        strict_feasibility=False,
    )
    assert first_no_ris == second_no_ris
    assert first_random == second_random
    source = inspect.getsource(simulation._physical_baseline_rate)
    assert "select_feasible_beamformers" in source
    assert "HRISUAVISACEnv" in source
    assert "_advance_targets" in source
    assert "_refresh_channel" in source

    fig7_source = inspect.getsource(simulation.fig7_rate_vs_L)
    assert "_fresh_scene" not in fig7_source


def test_parameter_sweeps_hold_evaluation_realizations_constant():
    import simulation

    fig5 = inspect.getsource(simulation.fig5_ao_convergence)
    fig6 = inspect.getsource(simulation.fig6_rate_vs_power)
    fig7 = inspect.getsource(simulation.fig7_rate_vs_L)
    fig8 = inspect.getsource(simulation.fig8_rate_vs_scee)
    fig9 = inspect.getsource(simulation.fig9_active_budget)
    fig13 = inspect.getsource(simulation.fig13_rate_vs_delta)
    fig14 = inspect.getsource(simulation.fig14_rate_vs_delta_bar)

    assert "seed=int(ao_seed)" in fig5
    assert "seed = int(ao_seed_base) + mc_index" in fig6
    for source in (fig7, fig8, fig9, fig13, fig14):
        assert "seed_base=int(evaluation_seed_base)" in source


def test_ao_figures_refuse_to_plot_infeasible_solutions():
    import simulation

    assert 'res["is_feasible"]' in inspect.getsource(
        simulation.fig5_ao_convergence
    )
    assert 'res["is_feasible"]' in inspect.getsource(
        simulation.fig6_rate_vs_power
    )


def test_ao_convergence_history_contains_raw_iterates_not_best_so_far_curve_repair():
    import ao_optimizer

    source = inspect.getsource(ao_optimizer.run_ao)
    assert "history_objective.append(cur_objective)" in source
    assert "history_sum_rate.append(cur_sum_rate)" in source
    assert "history_sensing_rate.append(cur_sensing_rate)" in source
    assert "history_sum_rate.append(best_sum_rate)" not in source


def test_partial_csi_masks_only_the_agent_observation_not_physical_reward():
    from environment import HRISUAVISACEnv
    from research_objectives import ObjectiveSpec

    objective = ObjectiveSpec.paper(target_rate_min=0.0)
    full = HRISUAVISACEnv(
        sigma2_scee=cfg.rho_nmse, seed=920, csi_observation_fraction=1.0,
        observe_instantaneous_csi=True,
        objective_spec=objective,
    )
    partial = HRISUAVISACEnv(
        sigma2_scee=cfg.rho_nmse, seed=920,
        csi_observation_fraction=cfg.partial_csi_fraction,
        observe_instantaneous_csi=True,
        objective_spec=objective,
    )
    state_full = full.reset()
    state_partial = partial.reset()
    assert state_full.shape == state_partial.shape
    assert np.isclose(
        partial.csi_observation_mask.mean(), cfg.partial_csi_fraction,
        atol=1.0 / partial.csi_observation_mask.size,
    )
    channel_state = state_partial[-partial.csi_observation_mask.size:]
    assert np.all(channel_state[~partial.csi_observation_mask] == 0.0)

    action = np.zeros(full.action_dim)
    _, reward_full, _, info_full = full.step(action)
    _, reward_partial, _, info_partial = partial.step(action)
    assert np.isclose(reward_full, reward_partial)
    assert np.isclose(info_full["sum_sensing"], info_partial["sum_sensing"])


def test_build_beamformers_accepts_explicit_power_split_without_mutating_cfg():
    import sinr_rate

    sig = inspect.signature(sinr_rate.build_beamformers)
    assert "comm_power_frac" in sig.parameters
    source = inspect.getsource(sinr_rate.build_beamformers)
    assert "cfg.comm_power_frac" not in source


def test_ao_uses_scipy_power_split_search_and_seeded_channels():
    import ao_optimizer

    run_sig = inspect.signature(ao_optimizer.run_ao)
    assert "seed" in run_sig.parameters

    assert hasattr(ao_optimizer, "optimize_power_split")
    solve_source = inspect.getsource(ao_optimizer.solve_beamformer)
    run_source = inspect.getsource(ao_optimizer.run_ao)
    traj_source = inspect.getsource(ao_optimizer._traj_objective)
    assert "optimize_power_split" in solve_source
    assert "cfg.comm_power_frac" not in solve_source
    assert "ChannelSet(pos" in run_source and "seed=" in run_source
    assert "ChannelSet(pos_uav" in traj_source and "seed=" in traj_source


def test_tiny_ao_smoke_runs_with_scipy_power_split():
    from config import _factor_upa
    from ao_optimizer import run_ao

    saved = (
        cfg.L, cfg.La, cfg.Lp, cfg.Lax, cfg.Lay, cfg.Lpx, cfg.Lpy,
        cfg.sca_max_iter, cfg.ao_max_iter,
    )
    try:
        cfg.L = 4
        cfg.La = 2
        cfg.Lp = 2
        cfg.Lax, cfg.Lay = _factor_upa(cfg.La)
        cfg.Lpx, cfg.Lpy = _factor_upa(cfg.Lp)
        cfg.sca_max_iter = 1
        cfg.ao_max_iter = 1

        pos_uav = np.array([50.0, 50.0, cfg.z_UAV])
        pos_ues = [
            np.array([10.0, 20.0, 0.0]),
            np.array([35.0, 70.0, 0.0]),
            np.array([80.0, 15.0, 0.0]),
        ]
        pos_tes = [
            np.array([15.0, 80.0, 0.0]),
            np.array([55.0, 60.0, 0.0]),
            np.array([90.0, 90.0, 0.0]),
        ]

        result = run_ao(pos_uav, pos_ues, pos_tes,
                        sigma2=cfg.rho_nmse, n_iter=1,
                        fix_uav=True, seed=2026)

        assert np.isfinite(result["objective"])
        assert np.isfinite(result["sum_rate"])
        assert np.isfinite(result["sensing_rate"])
        assert 0.0 < result["comm_power_frac"] < 1.0
        assert len(result["history_objective"]) == 2
        assert isinstance(result["is_feasible"], bool)
        assert {
            "target_rate", "passive_cascade", "active_power",
            "total_transmit_power",
        } <= set(result["constraint_diagnostics"])
    finally:
        (cfg.L, cfg.La, cfg.Lp, cfg.Lax, cfg.Lay, cfg.Lpx, cfg.Lpy,
         cfg.sca_max_iter, cfg.ao_max_iter) = saved


def test_ao_trajectory_update_respects_euclidean_step_bound():
    from types import SimpleNamespace
    import ao_optimizer

    old_minimize = ao_optimizer._minimize
    try:
        ao_optimizer._minimize = lambda *args, **kwargs: SimpleNamespace(
            x=np.array([50.0 + cfg.D_max, 50.0 + cfg.D_max]),
            success=True,
        )
        start = np.array([50.0, 50.0, cfg.z_UAV])
        updated = ao_optimizer.solve_uav_trajectory(
            start,
            [np.array([10.0, 20.0, 0.0])] * cfg.U,
            [np.array([80.0, 70.0, 0.0])] * cfg.K,
            np.ones(cfg.Lp, dtype=complex),
            np.ones(cfg.La, dtype=complex),
            np.zeros((cfg.M, cfg.U), dtype=complex),
            np.zeros((cfg.M, cfg.K), dtype=complex),
            sigma2=0.0,
            channel_seed=1,
        )
    finally:
        ao_optimizer._minimize = old_minimize

    assert np.linalg.norm(updated[:2] - start[:2]) <= cfg.D_max + 1e-12


def test_passive_only_ao_smoke_supports_zero_active_elements():
    from config import _factor_upa
    from ao_optimizer import run_ao

    saved = (cfg.L, cfg.La, cfg.Lp, cfg.Lax, cfg.Lay, cfg.Lpx, cfg.Lpy)
    try:
        cfg.L, cfg.La, cfg.Lp = 4, 0, 4
        cfg.Lax, cfg.Lay = 1, 1
        cfg.Lpx, cfg.Lpy = _factor_upa(cfg.Lp)
        pos_uav = np.array([50.0, 50.0, cfg.z_UAV])
        pos_ues = [np.array([10.0 + 20 * i, 20.0, 0.0]) for i in range(cfg.U)]
        pos_tes = [np.array([15.0 + 20 * i, 80.0, 0.0]) for i in range(cfg.K)]
        result = run_ao(pos_uav, pos_ues, pos_tes, sigma2=0.0,
                        n_iter=1, fix_uav=True, seed=515)
        assert np.isfinite(result["sum_rate"])
        assert result["phi"].size == 0
    finally:
        (cfg.L, cfg.La, cfg.Lp, cfg.Lax, cfg.Lay,
         cfg.Lpx, cfg.Lpy) = saved


def test_fig13_fig14_match_published_axes_and_algorithm_families():
    import simulation

    fig13_source = inspect.getsource(simulation.fig13_rate_vs_delta)
    fig14_source = inspect.getsource(simulation.fig14_rate_vs_delta_bar)
    ao_source = inspect.getsource(__import__("ao_optimizer").run_ao)

    assert "cfg.comm_power_frac" not in fig13_source
    assert "cfg.comm_power_frac" not in fig14_source
    assert "np.arange(0, 21, 5)" in fig13_source
    assert "twinx" in fig13_source
    assert "np.log2(1.0 + threshold)" in fig13_source
    assert "sensing_rate_min=sensing_rate_min" in fig13_source
    assert "sensing_only_sinr_min=threshold" not in fig13_source
    assert "sensing_only_sinr_min" in fig14_source
    assert "sensing_rate_min=" not in fig14_source
    assert "DTL" in fig14_source and "DDPG" in fig14_source
    assert "run_ao" not in fig14_source
    assert "[30, 20]" in fig14_source
    assert "sensing_rate_min=sensing_rate_min" in ao_source
    assert "14_500" not in fig14_source


def test_fig13_exports_only_the_curves_that_are_actually_plotted():
    import simulation

    source = inspect.getsource(simulation.fig13_rate_vs_delta)
    guard = source.index("if L_val in [30, 20]:")
    export = source.index('curves[f"R_k, L={L_val}"] = rk_list')
    assert guard < export
    assert source.count('curves[f"R_k, L={L_val}"] = rk_list') == 1


def test_paper_paths_do_not_substitute_total_precoder_power_for_remark1_sinr():
    import ao_optimizer
    import sinr_rate

    assert "sensing_power_min" not in inspect.signature(
        sinr_rate.select_feasible_beamformers
    ).parameters
    assert "sensing_power_min" not in inspect.signature(ao_optimizer.run_ao).parameters
    diagnostics_source = inspect.getsource(ao_optimizer._constraint_diagnostics)
    assert '"sensing_power"' not in diagnostics_source


def test_paper_training_path_has_no_legacy_fixed_channel_warm_start():
    import simulation

    assert "warm_start" not in inspect.signature(simulation.train_ddpg).parameters
    source = inspect.getsource(simulation.train_ddpg)
    assert "warm_start_fixed_channel" not in source


def test_fig4_uses_paper_reward_and_default_episode_budget():
    import simulation

    function = simulation.fig4_rewards_vs_episodes
    source = inspect.getsource(function)
    assert inspect.signature(function).parameters["n_ep"].default is None
    assert cfg.E_episodes == 1000
    assert cfg.fig4_episodes == 1500
    assert "cfg.fig4_episodes if n_ep is None" in source
    assert "np.arange(n_ep)" in source
    assert "cfg.fig4_plot_interval" in source
    assert "comm_rates" in source
    assert 'res_dtl_p["comm_rates"]' in source
    assert "fig4_rewards_vs_episodes.png" in source
    assert "fig4_rewards_vs_episodes_penalized.png" not in source
    assert "Average Reward" in source
    assert "Diagnostic Penalized Reward" not in source
    assert "cfg.rho_nmse * 0.5" not in source


def test_fig6_uses_configured_training_and_ao_budgets():
    import simulation

    source = inspect.getsource(simulation.fig6_rate_vs_power)
    assert "cfg.fig6_train_episodes" in source
    assert "cfg.fig6_ao_iter" in source
    assert "train_ddpg(env_i6, 200" not in source
    assert "n_iter=10" not in source
    assert "np.arange(10, 31, 5)" in source
    assert 'ax.set_xlabel("P_T (dB)")' in source
    assert "TrainingRegistry" in source
    assert "get_or_train" in source
    assert "fix_uav=False" in source
    assert "temporary_system_configuration" in source
    assert "cfg.L   = 20" not in source


def test_fig6_trains_policies_at_each_transmit_power_point():
    import simulation

    tree = ast.parse(inspect.getsource(simulation.fig6_rate_vs_power))
    power_loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and any(
            isinstance(child, ast.Name) and child.id == "PT_dB_range"
            for child in ast.walk(node.iter)
        )
    ]
    assert power_loops
    assert any(
        any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get_or_train"
            for child in ast.walk(loop)
        )
        for loop in power_loops
    )


def test_ddpg_agent_is_plain_single_critic_ddpg():
    from ddpg_agent import DDPGAgent

    agent = DDPGAgent(state_dim=5, action_dim=3)
    assert hasattr(agent, "critic")
    assert hasattr(agent, "critic_tgt")
    assert not hasattr(agent, "critic2")
    assert not hasattr(agent, "critic_tgt2")
    assert not hasattr(agent, "opt_critic2")
    assert not hasattr(agent, "_actor_delay")

    update_source = inspect.getsource(agent.update)
    assert "torch.min" not in update_source
    assert "policy smoothing" not in update_source.lower()
    assert "critic2" not in update_source


def test_dtl_agent_uses_single_critic_eq30_and_counts_updates():
    from ddpg_agent import DDPGAgent
    from dtl_agent import DTLAgent

    old_batch = cfg.B_batch
    cfg.B_batch = 4
    try:
        source = DDPGAgent(state_dim=5, action_dim=3)
        agent = DTLAgent(source, state_dim=5, action_dim=3)
        assert not hasattr(agent, "critic2")
        assert not hasattr(agent, "critic_tgt2")
        assert not hasattr(agent, "opt_critic2")

        update_source = inspect.getsource(agent.update)
        assert "q_min_target" not in update_source
        assert "torch.min" not in update_source
        assert "critic_tgt2" not in update_source

        rng = np.random.default_rng(123)
        for _ in range(cfg.B_batch):
            state = rng.normal(size=5).astype(np.float32)
            action = rng.uniform(-1, 1, size=3).astype(np.float32)
            next_state = rng.normal(size=5).astype(np.float32)
            agent.replay.push(state, action, 1.0, next_state, False)

        before = agent.total_steps
        result = agent.update()
        assert agent.total_steps == before + 1
        assert {"critic_loss", "actor_loss", "upsilon"} <= set(result)
    finally:
        cfg.B_batch = old_batch


def test_train_ddpg_default_path_is_paper_ddpg_without_nm_warm_start():
    import simulation

    sig = inspect.signature(simulation.train_ddpg)
    assert "warm_start" not in sig.parameters
    assert not hasattr(simulation, "warm_start_fixed_channel")
    assert "TD3" not in (simulation.train_ddpg.__doc__ or "")
    assert "Nelder" not in (simulation.train_ddpg.__doc__ or "")


def test_training_loops_use_one_replay_update_per_environment_step():
    import simulation

    ddpg_source = inspect.getsource(simulation.train_ddpg)
    dtl_source = inspect.getsource(simulation.train_dtl)
    assert ddpg_source.count("agent.update()") == 1
    assert dtl_source.count("agent.update()") == 1
    assert "2 gradient steps" not in ddpg_source
    assert "2 gradient steps" not in dtl_source


def test_seeded_environment_replays_same_first_transition():
    from environment import HRISUAVISACEnv
    from research_objectives import ObjectiveSpec

    objective = ObjectiveSpec.paper(target_rate_min=0.0)
    env_a = HRISUAVISACEnv(
        sigma2_scee=cfg.rho_nmse, seed=222, objective_spec=objective
    )
    env_b = HRISUAVISACEnv(
        sigma2_scee=cfg.rho_nmse, seed=222, objective_spec=objective
    )

    state_a = env_a.reset()
    state_b = env_b.reset()
    action = np.zeros(env_a.action_dim, dtype=float)
    next_a, reward_a, done_a, info_a = env_a.step(action)
    next_b, reward_b, done_b, info_b = env_b.step(action)

    assert np.allclose(state_a, state_b)
    assert np.allclose(next_a, next_b)
    assert np.isclose(reward_a, reward_b)
    assert done_a == done_b
    assert np.isclose(info_a["joint_rate"], info_b["joint_rate"])
    assert np.allclose(info_a["P_ai"], info_b["P_ai"])


def test_environment_keeps_paper_action_space_without_ao_inside_drl_step():
    import environment
    from environment import HRISUAVISACEnv

    env = HRISUAVISACEnv(sigma2_scee=0.0, seed=303)
    assert env.action_dim == cfg.Lp + 2 * cfg.La + 2

    step_source = inspect.getsource(environment.HRISUAVISACEnv.step)
    assert "solve_beamformer" not in step_source
    evaluation_source = inspect.getsource(
        environment.HRISUAVISACEnv._evaluate_controls
    )
    assert "select_feasible_beamformers" in evaluation_source


def test_environment_step_reports_feasible_power_split_metadata():
    from config import _factor_upa
    from environment import HRISUAVISACEnv
    from research_objectives import ObjectiveSpec

    saved = (
        cfg.L, cfg.La, cfg.Lp, cfg.Lax, cfg.Lay, cfg.Lpx, cfg.Lpy,
        cfg.N_slots, cfg.sca_max_iter,
    )
    try:
        cfg.L = 4
        cfg.La = 2
        cfg.Lp = 2
        cfg.Lax, cfg.Lay = _factor_upa(cfg.La)
        cfg.Lpx, cfg.Lpy = _factor_upa(cfg.Lp)
        cfg.N_slots = 2
        cfg.sca_max_iter = 1

        env = HRISUAVISACEnv(
            sigma2_scee=0.0,
            seed=404,
            objective_spec=ObjectiveSpec.paper(target_rate_min=0.0),
        )
        _, _, _, info = env.step(np.zeros(env.action_dim))

        assert 0.0 < info["comm_power_frac"] < 1.0
        assert np.isfinite(info["beamformer_score"])
    finally:
        (cfg.L, cfg.La, cfg.Lp, cfg.Lax, cfg.Lay, cfg.Lpx, cfg.Lpy,
         cfg.N_slots, cfg.sca_max_iter) = saved


def test_state_contains_precoders_and_full_composite_channel_matrices():
    from environment import HRISUAVISACEnv

    env = HRISUAVISACEnv(sigma2_scee=0.0, seed=405)
    expected = (
        2 * cfg.Lp + 2 * cfg.La + 3
        + 2 * cfg.M * (cfg.U + cfg.K)
        + 2 * cfg.N * cfg.M * (cfg.U + cfg.K)
    )
    assert env.state_dim == expected
    state = env.reset()
    assert state.shape == (expected,)
    assert hasattr(env, "C") and env.C.shape == (cfg.M, cfg.U)
    assert hasattr(env, "T") and env.T.shape == (cfg.M, cfg.K)


def test_fig7_includes_all_published_hris_sizes():
    import simulation

    source = inspect.getsource(simulation.fig7_rate_vs_L)
    assert "[20, 30, 40, 50, 60]" in source
    assert "TrainingRegistry" in source
    assert "get_or_train" in source
    assert "run_ao" not in source
    assert "La=0" in source


def test_fig8_uses_trained_hris_and_passive_policies_at_each_scee():
    import simulation

    source = inspect.getsource(simulation.fig8_rate_vs_scee)
    assert "TrainingRegistry" in source
    assert "variant_sigmas" in source
    assert "run_ao" not in source
    assert "La=0" in source
    assert "np.arange(0.0, 1.01, 0.2)" in source


def test_fig9_and_fig13_use_dtl_not_ao_under_proposed_labels():
    import simulation

    fig9_source = inspect.getsource(simulation.fig9_active_budget)
    assert "TrainingRegistry" in fig9_source
    assert "get_or_train" in fig9_source
    assert "run_ao" not in fig9_source

    fig13_source = inspect.getsource(simulation.fig13_rate_vs_delta)
    assert "TrainingRegistry" in fig13_source
    assert "get_or_train" in fig13_source
    assert "_eval_agent_metrics" in fig13_source
    assert "run_ao" not in fig13_source


def test_each_sweep_returns_its_own_computed_data_without_fig10_leakage():
    import simulation

    for function in (
        simulation.fig7_rate_vs_L,
        simulation.fig8_rate_vs_scee,
        simulation.fig9_active_budget,
    ):
        source = inspect.getsource(function)
        assert '"training_registry": registry' in source

    fig10_source = inspect.getsource(simulation.fig10_uav_trajectory)
    assert "L_vals" not in fig10_source
    assert "rho_range" not in fig10_source
    assert "training_registry" not in fig10_source
    assert "return path" in fig10_source


def test_agent_metric_evaluation_reports_computed_comm_and_sensing_rates():
    import simulation

    source = inspect.getsource(simulation._eval_agent_metrics)
    assert 'info["sum_rate"]' in source
    assert 'info["sum_sensing"]' in source
    assert 'info["physical_constraints_feasible"]' in source
    assert "raise RuntimeError" in source
    assert "return" in source


def test_fig10_random_baseline_randomizes_phase_not_uav_control():
    import simulation

    source = inspect.getsource(simulation.fig10_uav_trajectory)
    assert "random_phase" in source
    assert "random_action" not in source
    assert "cfg.z_UAV * np.ones(len(ues))" in source
    assert "ax.set_xlim(cfg.x_min, cfg.x_max)" in source
    assert "ax.set_ylim(cfg.y_min, cfg.y_max)" in source
    assert "ax.set_zlim(cfg.z_UAV - 1.0, cfg.z_UAV + 1.0)" in source
    assert cfg.fig10_plot_interval == 20
    assert "cfg.fig10_plot_interval" in source
    assert "th_plot" in source and "tr_plot" in source


def test_fig10_and_fig12_use_consistent_imperfect_csi_target_agents():
    import simulation

    fig4_source = inspect.getsource(simulation.fig4_rewards_vs_episodes)
    fig12_source = inspect.getsource(simulation.fig12_transmission_time)
    assert '"dtl": res_dtl_i["agent"]' in fig4_source
    assert '"DTL": res_dtl_i["elapsed_seconds"]' in fig4_source
    assert 'variants=("imperfect",)' in fig12_source
    assert 'bundle["dtl"]["imperfect"]' in fig12_source


def test_fig11_uses_scene_target_geometry_not_nominal_angles():
    import simulation

    source = inspect.getsource(simulation.fig11_beampattern)
    assert "pos_tes" in source
    assert "np.radians(30)" not in source
    assert "np.radians(45)" not in source


def test_fig11_uses_the_published_azimuth_and_elevation_domain():
    import simulation

    source = inspect.getsource(simulation.fig11_beampattern)
    assert "np.deg2rad(-80.0)" in source
    assert "np.deg2rad(0.0)" in source
    assert "np.deg2rad(80.0)" in source
    assert "-np.pi/2, np.pi/2" not in source


def test_fig5_uses_paper_iterations_one_to_twenty_without_early_truncation():
    import simulation

    source = inspect.getsource(simulation.fig5_ao_convergence)
    assert "stop_on_convergence=False" in source
    assert "fix_uav=False" in source
    assert "_temporary_hris_configuration" in source
    assert "history_sum_rate\"][1:cfg.ao_max_iter + 1]" in source
    assert "range(1, cfg.ao_max_iter + 1)" in source


def test_fig12_and_fig14_use_valid_target_task_training_bundles():
    import simulation

    fig12_source = inspect.getsource(simulation.fig12_transmission_time)
    assert "TrainingRegistry" in fig12_source
    assert "get_or_train" in fig12_source
    assert "train_dtl(ddpg" not in fig12_source
    assert "hardware-dependent" in fig12_source

    fig14_source = inspect.getsource(simulation.fig14_rate_vs_delta_bar)
    assert "TrainingRegistry" in fig14_source
    assert "get_or_train" in fig14_source
    assert 'bundle["ddpg"]' in fig14_source
    assert 'bundle["dtl"]["imperfect"]' in fig14_source
    assert "train_dtl(ddpg" not in fig14_source


def test_every_paper_figure_exports_computed_data_and_provenance():
    import simulation

    for number in range(4, 15):
        function = getattr(simulation, {
            4: "fig4_rewards_vs_episodes",
            5: "fig5_ao_convergence",
            6: "fig6_rate_vs_power",
            7: "fig7_rate_vs_L",
            8: "fig8_rate_vs_scee",
            9: "fig9_active_budget",
            10: "fig10_uav_trajectory",
            11: "fig11_beampattern",
            12: "fig12_transmission_time",
            13: "fig13_rate_vs_delta",
            14: "fig14_rate_vs_delta_bar",
        }[number])
        assert "_save_figure_data" in inspect.getsource(function)


def test_figure_data_writer_persists_exact_arrays_without_curve_transforms():
    import json
    import tempfile
    from pathlib import Path
    import simulation

    original_out = simulation.OUT
    try:
        with tempfile.TemporaryDirectory() as directory:
            simulation.OUT = directory
            paths = simulation._save_figure_data(
                99, [0, 1], {"computed curve": [1.25, 2.5]},
                metadata={"method": "unit-test-computation"},
            )
            with np.load(paths["npz"]) as archive:
                assert np.array_equal(archive["x"], np.array([0, 1]))
                assert np.array_equal(
                    archive["curve_00"], np.array([1.25, 2.5])
                )
            manifest = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            assert manifest["curves"]["computed curve"] == [1.25, 2.5]
            assert manifest["configuration_fingerprint"]
    finally:
        simulation.OUT = original_out


def test_fig12_is_published_transmission_time_bar_chart():
    import simulation

    source = inspect.getsource(simulation.fig12_transmission_time)
    assert ".bar(" in source
    assert "Transmission time (secs)" in source
    assert "Training Samples" not in source
    assert "DTL" in source and "DDPG" in source
    assert "[40, 30]" in source
    assert '["DTL", "DDPG"]' in source


def test_ao_and_sinr_helpers_do_not_keep_stale_unreachable_objectives():
    import ao_optimizer
    import sinr_rate

    traj_source = inspect.getsource(ao_optimizer._traj_objective)
    active_power_source = inspect.getsource(sinr_rate.active_tx_power)

    assert "ch_new" not in traj_source
    assert "UE channels only" not in traj_source
    assert "return -sr" not in traj_source
    assert "_score_beamformers" in traj_source
    assert "T_bar =" not in active_power_source
    assert "return cfg.PT * power" not in active_power_source


def test_copy_has_reproduction_notes_for_verified_entry_points():
    from pathlib import Path

    note_path = Path(__file__).with_name("FULL_REPRODUCTION_NOTES.md")
    text = note_path.read_text(encoding="utf-8")
    lower_text = text.lower()

    assert "isolated full-reproduction copy" in lower_text
    assert "original folder is unchanged" in lower_text
    assert "diagnostic" in lower_text
    assert "long training" in lower_text


def test_ao_returns_separate_objective_and_comm_history_contract():
    import ao_optimizer
    import simulation

    ao_source = inspect.getsource(ao_optimizer.run_ao)
    assert "history_objective" in ao_source
    assert "history_sum_rate" in ao_source
    assert "history=history" not in ao_source

    fig5_source = inspect.getsource(simulation.fig5_ao_convergence)
    assert 'res["history_sum_rate"]' in fig5_source
    assert 'res["history"]' not in fig5_source


def test_fast_mode_does_not_emit_synthetic_training_curves():
    main_source = inspect.getsource(sys.modules.get("main")) if "main" in sys.modules else None
    if main_source is None:
        from pathlib import Path

        main_source = Path(__file__).with_name("main.py").read_text(encoding="utf-8-sig")

    assert "synthetic" not in main_source.lower()
    assert "exponential" not in main_source.lower()
    assert "fig4_rewards_vs_episodes.png  (fast mode)" not in main_source


def test_fast_mode_removes_stale_skipped_figure_outputs():
    from pathlib import Path

    main_source = Path(__file__).with_name("main.py").read_text(encoding="utf-8-sig")
    assert "_remove_stale_figure" in main_source
    for figure_name in [
        "fig4_rewards_vs_episodes.png",
        "fig6_rate_vs_power.png",
        "fig10_uav_trajectory.png",
        "fig12_transmission_time.png",
        "fig14_rate_vs_delta_bar.png",
    ]:
        assert figure_name in main_source


def test_runners_share_training_registry_and_fast_mode_skips_all_drl_sweeps():
    from pathlib import Path

    main_source = Path(__file__).with_name("main.py").read_text(encoding="utf-8-sig")
    run_all_source = Path(__file__).with_name("run_all_figs.py").read_text(
        encoding="utf-8-sig"
    )
    for source in (main_source, run_all_source):
        assert "TrainingRegistry" in source
        assert "training_registry=" in source
        assert "fig14_rate_vs_delta_bar(training_runs=" not in source

    for figure_name in (
        "fig7_rate_vs_L.png", "fig8_rate_vs_scee.png",
        "fig9_active_budget.png", "fig13_rate_vs_delta.png",
    ):
        assert figure_name in main_source
    assert "if not FAST:" in main_source


def test_fig10_trajectory_requires_real_agent_or_skips():
    import simulation

    sig = inspect.signature(simulation.fig10_uav_trajectory)
    assert "agent" in sig.parameters
    source = inspect.getsource(simulation.fig10_uav_trajectory)
    assert "agent is None" in source
    assert "return None" in source
    assert "centroid" not in source.lower()


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
