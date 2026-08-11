"""Run the paper-baseline objective across independent seeds with checkpoints."""

import argparse
import hashlib
from pathlib import Path

from ambiguity_hypotheses import add_ambiguity_cli_arguments


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated, at least five")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return add_ambiguity_cli_arguments(parser)


def run(args):
    from ambiguity_hypotheses import (
        apply_ambiguity_arguments, apply_ambiguity_manifest_fields,
        ddpg_comparator_scee,
    )
    from config import cfg
    from environment import HRISUAVISACEnv
    from experiment_protocol import (
        atomic_write_json, ensure_run_directory, evaluation_metrics,
        final_window_mean, make_run_manifest, parse_seeds,
        physical_configuration_fingerprint,
        serializable_training_result,
        TRAINING_SEED_PROTOCOL,
        derive_training_seed,
    )
    from parameter_profiles import (
        activate_profile, load_calibrated_profile, load_profile,
    )
    from research_objectives import ObjectiveSpec
    from simulation import (
        train_ddpg,
        train_dtl,
    )
    from source_integrity import source_tree_digest

    seeds = parse_seeds(args.seeds)
    profile = (
        load_profile(args.profile)
        if args.dry_run
        else load_calibrated_profile(args.profile)
    )
    activate_profile(profile)
    apply_ambiguity_arguments(args, cfg)
    objective = ObjectiveSpec.paper(target_rate_min=cfg.R_min)
    physics = physical_configuration_fingerprint(profile.digest, cfg)
    manifest = make_run_manifest(
        "paper_baseline", profile.digest, physics, objective.as_dict(), seeds
    )
    apply_ambiguity_manifest_fields(manifest, cfg)
    manifest["power_reference"] = cfg.PT_reference
    manifest["power_hypothesis_id"] = cfg.power_hypothesis_id
    manifest["source_tree_digest"] = source_tree_digest()
    manifest["training_seed_protocol"] = TRAINING_SEED_PROTOCOL
    manifest["episodes"] = args.episodes
    manifest["evaluation_episodes"] = args.eval_episodes
    output = ensure_run_directory(args.output, manifest)
    if args.dry_run:
        return 0
    if args.episodes <= 0 or args.eval_episodes <= 0 or args.checkpoint_every <= 0:
        raise ValueError("Episodes, evaluation episodes, and checkpoint interval must be positive.")
    if args.episodes != cfg.fig4_episodes:
        raise ValueError(
            f"Paper Figure 4 requires exactly {cfg.fig4_episodes} episodes."
        )

    progress_path = output / "baseline_progress.json"
    checkpoint_dir = output / "checkpoints"
    seed_records = []
    for seed in seeds:
        source_seed = seed
        target_seed = seed + 100_000
        source_training_seed = derive_training_seed(seed, "source")
        target_training_seed = derive_training_seed(seed, "target")
        source_tag = f"baseline-source-seed-{seed}"
        source_env = HRISUAVISACEnv(
            sigma2_scee=cfg.rho_nmse, seed=source_seed,
            objective_spec=objective,
        )
        source_path = checkpoint_dir / f"{source_tag}.pt"
        source = train_ddpg(
            source_env, args.episodes, source_tag,
            checkpoint_path=source_path, resume=not args.fresh,
            checkpoint_every=args.checkpoint_every,
            training_seed=source_training_seed,
        )

        methods = {}
        method_specs = {
            "DDPG": ("ddpg", ddpg_comparator_scee(cfg), "capon"),
            "DTL-pCSI": ("dtl", 0.0, "capon"),
            "DTL-WFB": ("dtl", 0.0, cfg.wfb_beamformer_mode),
            "DTL-iCSI": ("dtl", cfg.rho_nmse, "capon"),
        }
        for method, (algorithm, sigma2, beamformer_mode) in method_specs.items():
            tag = f"baseline-{method}-seed-{seed}"
            environment = HRISUAVISACEnv(
                sigma2_scee=sigma2, seed=target_seed,
                sensing_beamformer_mode=beamformer_mode,
                objective_spec=objective,
            )
            path = checkpoint_dir / f"{tag}.pt"
            if algorithm == "ddpg":
                trained = train_ddpg(
                    environment, args.episodes, tag, checkpoint_path=path,
                    resume=not args.fresh,
                    checkpoint_every=args.checkpoint_every,
                    training_seed=target_training_seed,
                )
            else:
                trained = train_dtl(
                    source["agent"], environment, args.episodes, tag,
                    checkpoint_path=path, resume=not args.fresh,
                    checkpoint_every=args.checkpoint_every,
                    training_seed=target_training_seed,
                )

            def environment_factory(
                    index, sigma2=sigma2, mode=beamformer_mode):
                return HRISUAVISACEnv(
                    sigma2_scee=sigma2,
                    seed=target_seed + 10_000 + index,
                    sensing_beamformer_mode=mode,
                    objective_spec=objective,
                    allocation_mode="scipy",
                )

            evaluation = evaluation_metrics(
                trained["agent"], environment_factory, args.eval_episodes
            )
            methods[method] = {
                "tag": tag,
                "training_seed": target_training_seed,
                "checkpoint": str(path.relative_to(output)),
                "checkpoint_sha256": _file_sha256(path),
                "training": serializable_training_result(trained),
                "evaluation": evaluation,
            }
        seed_records.append({
            "seed": seed,
            "source_seed": source_seed,
            "target_seed": target_seed,
            "source_training_seed": source_training_seed,
            "target_training_seed": target_training_seed,
            "source_tag": source_tag,
            "source_checkpoint": str(source_path.relative_to(output)),
            "source_checkpoint_sha256": _file_sha256(source_path),
            "methods": methods,
        })
        atomic_write_json(progress_path, {"completed_seeds": seed_records})

    training_final_window = 100
    final_methods = {}
    for method in ("DTL-pCSI", "DTL-WFB", "DTL-iCSI", "DDPG"):
        final_methods[method] = {
            "seed_values": [
                final_window_mean(
                    record["methods"][method]["training"]["comm_rates"],
                    window=training_final_window,
                )
                for record in seed_records
            ]
        }
    primary = [record["methods"]["DTL-pCSI"]["evaluation"] for record in seed_records]
    all_method_evaluations = [
        method["evaluation"]
        for record in seed_records
        for method in record["methods"].values()
    ]
    summary = {
        "schema_version": 1,
        "kind": "generated_result",
        "protocol": "paper_baseline",
        "physical_configuration_fingerprint": physics,
        "profile_digest": profile.digest,
        "objective": objective.as_dict(),
        "power_reference": cfg.PT_reference,
        "power_hypothesis_id": cfg.power_hypothesis_id,
        "source_tree_digest": manifest["source_tree_digest"],
        "training_seed_protocol": TRAINING_SEED_PROTOCOL,
        "seeds": list(seeds),
        "episodes": args.episodes,
        "figure4_protocol": {
            "raw_episode_count": cfg.fig4_episodes,
            "plot_x_start": 0,
            "plot_x_end": 1400,
            "plot_interval": cfg.fig4_plot_interval,
            "acceptance_smoothing_window": 20,
        },
        "evaluation_episodes": args.eval_episodes,
        "final_reward_source": "training_communication_rate",
        "training_final_window": training_final_window,
        "feasibility_rate": min(
            item["feasibility_rate"] for item in all_method_evaluations
        ),
        "final_reward_by_method": final_methods,
        "communication_seed_values": [item["communication_mean"] for item in primary],
        "sensing_seed_values": [item["sensing_mean"] for item in primary],
        "seed_records": seed_records,
    }
    apply_ambiguity_manifest_fields(summary, cfg)
    atomic_write_json(output / "baseline_summary.json", summary)
    manifest["status"] = "complete"
    manifest["summary"] = "baseline_summary.json"
    from validate_results import result_payload_digest
    manifest["summary_payload_sha256"] = result_payload_digest(summary)
    atomic_write_json(output / "run_manifest.json", manifest)
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
