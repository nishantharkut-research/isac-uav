import argparse
import os
import sys, time
sys.path.insert(0, ".")

from parameter_profiles import (
    activate_profile,
    active_profile_metadata,
    load_calibrated_profile,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the HRIS-UAV-ISAC paper reproduction"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Run only short non-DRL figures and remove stale skipped outputs",
    )
    parser.add_argument(
        "--profile", help="Validated JSON parameter profile to activate",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore and replace existing checkpoints instead of resuming",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10, metavar="EPISODES",
        help="Atomically checkpoint every N completed episodes (default: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = _parse_args()
    if not ARGS.profile:
        raise SystemExit(
            "A validated --profile is required for any reproduction run."
        )
    if ARGS.profile:
        profile = load_calibrated_profile(ARGS.profile)
        if profile.kind != "identified":
            raise SystemExit(
                "A full figure run requires a frozen identified profile."
            )
        activate_profile(profile)
else:
    ARGS = argparse.Namespace(
        fast=False, profile=None, fresh=False, checkpoint_every=10
    )

from simulation import (
    fig4_rewards_vs_episodes,
    fig5_ao_convergence,
    fig6_rate_vs_power,
    fig7_rate_vs_L,
    fig8_rate_vs_scee,
    fig9_active_budget,
    fig10_uav_trajectory,
    fig11_beampattern,
    fig12_transmission_time,
    fig13_rate_vs_delta,
    fig14_rate_vs_delta_bar,
    fig_summary,
    TrainingRegistry,
    set_reproducibility,
    OUT,
)
from config import cfg
from cross_figure_validation import validate_result_directory
from scientific_preflight import run_scientific_preflight

FAST = ARGS.fast
N_EP = cfg.E_episodes

def banner(msg):
    print(f"\n{'═'*60}")
    print(f"  {msg}")
    print(f"{'═'*60}")

def _remove_stale_figure(filename):
    figure_prefix = filename.split("_", 1)[0]
    candidates = [filename, f"{figure_prefix}_data.npz",
                  f"{figure_prefix}_data.json"]
    for candidate in candidates:
        path = os.path.join(OUT, candidate)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed stale skipped output: {path}")

if __name__ == "__main__":
    if ARGS.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive.")
    set_reproducibility()
    training_registry = TrainingRegistry(
        resume=not ARGS.fresh,
        checkpoint_every=ARGS.checkpoint_every,
    )
    t_total = time.time()
    banner("HRIS-UAV-ISAC - Transfer Learning DRL - Saikia et al. 2025")
    print(f"  Mode   : {'FAST (AO-only)' if FAST else 'FULL (DRL + AO)'}")
    print(f"  Output : {OUT}")
    profile_meta = active_profile_metadata()
    print(f"  Profile: {profile_meta['name']}  digest={profile_meta['digest']}")
    run_scientific_preflight(os.path.join(OUT, "scientific_preflight.json"))

    # ── Fig. 5  (AO convergence — always run first, fast) ─────────────────
    fig5_ao_convergence()

    # ── Fig. 4  (DRL training curves) ─────────────────────────────────────
    if not FAST:
        trained_agents = fig4_rewards_vs_episodes(
            resume=not ARGS.fresh,
            checkpoint_every=ARGS.checkpoint_every,
        )
        training_runs = dict(trained_agents["training_runs"])
    else:
        print("  Skipping Fig. 4: requires full DRL training.")
        _remove_stale_figure("fig4_rewards_vs_episodes.png")
        trained_agents = None
        training_runs = {}

    # ── Figs. 6–9  (parameter sweeps) ─────────────────────────────────────
    if not FAST:
        training_runs.update(fig6_rate_vs_power(
            training_registry=training_registry
        ))
        fig7_rate_vs_L(training_registry=training_registry, n_ep=N_EP)
        fig8_rate_vs_scee(training_registry=training_registry, n_ep=N_EP)
        fig9_active_budget(training_registry=training_registry, n_ep=N_EP)
    else:
        print("  Skipping Figs. 6-9: they require DRL training sweeps.")
        _remove_stale_figure("fig6_rate_vs_power.png")
        _remove_stale_figure("fig7_rate_vs_L.png")
        _remove_stale_figure("fig8_rate_vs_scee.png")
        _remove_stale_figure("fig9_active_budget.png")

    # ── Fig. 10  (UAV trajectory) ──────────────────────────────────────────
    fig10_agent = training_runs.get(40, {}).get("dtl")
    fig10_path = fig10_uav_trajectory(fig10_agent)
    if fig10_path is None:
        _remove_stale_figure("fig10_uav_trajectory.png")

    # ── Fig. 11  (beampattern gain) ────────────────────────────────────────
    fig11_beampattern()

    # ── Fig. 12  (transmission time) ──────────────────────────────────────
    if not FAST and trained_agents is not None:
        training_runs = fig12_transmission_time(
            training_runs=training_runs, n_ep=N_EP,
            training_registry=training_registry,
        )
    else:
        print("  Skipping Fig. 12: requires DRL training histories.")
        _remove_stale_figure("fig12_transmission_time.png")

    # ── Figs. 13–14  (sensing–comm trade-off) ─────────────────────────────
    if not FAST:
        fig13_rate_vs_delta(training_registry=training_registry, n_ep=N_EP)
        fig14_rate_vs_delta_bar(training_registry=training_registry, n_ep=N_EP)
    else:
        print("  Skipping Figs. 13-14: they require DRL training sweeps.")
        _remove_stale_figure("fig13_rate_vs_delta.png")
        _remove_stale_figure("fig14_rate_vs_delta_bar.png")

    # ── Summary dashboard ──────────────────────────────────────────────────
    summary_path = fig_summary()

    validation_path = os.path.join(OUT, "cross_figure_validation.json")
    if not FAST:
        validation = validate_result_directory(OUT, validation_path)
        if not validation["overall_pass"]:
            raise RuntimeError(
                "Full run completed, but held-out cross-figure validation failed. "
                f"See {validation_path}."
            )
    elif os.path.exists(validation_path):
        os.remove(validation_path)
        print(f"  Removed stale full-run validation: {validation_path}")

    banner(f"All figures saved to  {OUT}")
    print(f"  Total wall time : {(time.time()-t_total)/60:.1f} min\n")
