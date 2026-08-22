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
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds")
    parser.add_argument(
        "--run-mode", choices=("publication", "smoke"), default="publication"
    )
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return add_ambiguity_cli_arguments(parser)


def parse_run_seeds(value, run_mode):
    from experiment_protocol import parse_seeds

    if run_mode == "publication":
        return parse_seeds(value)
    seeds = tuple(
        int(item.strip()) for item in str(value).split(",") if item.strip()
    )
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Smoke seeds must be unique nonnegative integers.")
    return seeds


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
        write_immutable_json,
        write_training_checkpoint_identity,
        EVALUATION_SEED_PROTOCOL,
        TRAINING_SEED_PROTOCOL,
        derive_training_seed,
        evaluation_scene_seed_grid,
    )
    from parameter_profiles import activate_profile, load_calibrated_profile
    from research_objectives import ObjectiveSpec
    from simulation import (
        train_ddpg,
        train_dtl,
    )
    from scientific_preflight import run_pretraining_physical_sanity
    from source_integrity import source_tree_digest

    seeds = parse_run_seeds(args.seeds, args.run_mode)
    profile = load_calibrated_profile(args.profile)
    activate_profile(profile)
    apply_ambiguity_arguments(args, cfg)
    objective = ObjectiveSpec.paper(target_rate_min=cfg.R_min)
    physics = physical_configuration_fingerprint(profile.digest, cfg)
    protocol = (
        "paper_baseline"
        if args.run_mode == "publication"
        else "paper_baseline_smoke"
    )
    manifest = make_run_manifest(
        protocol,
        profile.digest,
        physics,
        objective.as_dict(),
        seeds,
        minimum_seed_count=(5 if args.run_mode == "publication" else 1),
    )
    manifest["run_mode"] = args.run_mode
    manifest["publication_eligible"] = args.run_mode == "publication"
    apply_ambiguity_manifest_fields(manifest, cfg)
    manifest["power_reference"] = cfg.PT_reference
    manifest["power_hypothesis_id"] = cfg.power_hypothesis_id
    manifest["source_tree_digest"] = source_tree_digest()
    manifest["training_seed_protocol"] = TRAINING_SEED_PROTOCOL
    manifest["evaluation_seed_protocol"] = EVALUATION_SEED_PROTOCOL
    manifest["episodes"] = args.episodes
    manifest["evaluation_episodes"] = args.eval_episodes
    if args.episodes <= 0 or args.eval_episodes <= 0 or args.checkpoint_every <= 0:
        raise ValueError(
            "Episodes, evaluation episodes, and checkpoint interval must be "
            "positive."
        )
    if args.run_mode == "publication" and args.episodes != cfg.fig4_episodes:
        raise ValueError(
            f"Paper Figure 4 requires exactly {cfg.fig4_episodes} episodes."
        )
    if args.run_mode == "smoke" and args.episodes not in {100, 200}:
        raise ValueError("Smoke mode requires exactly 100 or 200 episodes.")
    evaluation_seed_grid = evaluation_scene_seed_grid(
        seeds, args.eval_episodes
    )
    output = ensure_run_directory(args.output, manifest)
    if args.dry_run:
        return 0
    sanity = run_pretraining_physical_sanity()
    sanity["physical_configuration_fingerprint"] = physics
    sanity["source_tree_digest"] = manifest["source_tree_digest"]
    apply_ambiguity_manifest_fields(sanity, cfg)
    sanity_path = write_immutable_json(
        output / "pretraining_physical_sanity.json", sanity
    )
    manifest["pretraining_physical_sanity"] = sanity_path.name
    manifest["pretraining_physical_sanity_sha256"] = _file_sha256(sanity_path)
    atomic_write_json(output / "run_manifest.json", manifest)
    progress_path = output / "baseline_progress.json"
    checkpoint_dir = output / "checkpoints"
    seed_records = []
    for seed in seeds:
        source_seed = seed
        target_seed = seed + 100_000
        source_training_seed = derive_training_seed(seed, "source")
        target_training_seed = derive_training_seed(seed, "target")
        evaluation_scene_seeds = evaluation_seed_grid[seed]
        source_tag = f"baseline-source-seed-{seed}"
        source_env = HRISUAVISACEnv(
            sigma2_scee=cfg.rho_nmse, seed=source_seed,
            objective_spec=objective,
            allocation_mode=cfg.policy_allocation_mode,
        )
        source_path = checkpoint_dir / f"{source_tag}.pt"
        source = train_ddpg(
            source_env, args.episodes, source_tag,
            checkpoint_path=source_path, resume=not args.fresh,
            checkpoint_every=args.checkpoint_every,
            training_seed=source_training_seed,
        )
        source_authority = write_training_checkpoint_identity(
            output, source_path, source["checkpoint_identity"]
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
                allocation_mode=cfg.policy_allocation_mode,
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
                    index, sigma2=sigma2, mode=beamformer_mode,
                    scene_seeds=evaluation_scene_seeds):
                return HRISUAVISACEnv(
                    sigma2_scee=sigma2,
                    seed=scene_seeds[index],
                    sensing_beamformer_mode=mode,
                    objective_spec=objective,
                    allocation_mode=cfg.policy_allocation_mode,
                )

            evaluation = evaluation_metrics(
                trained["agent"], environment_factory, args.eval_episodes
            )
            checkpoint_authority = write_training_checkpoint_identity(
                output, path, trained["checkpoint_identity"]
            )
            methods[method] = {
                "tag": tag,
                "training_seed": target_training_seed,
                **checkpoint_authority,
                "training": serializable_training_result(trained),
                "evaluation": evaluation,
            }
        seed_records.append({
            "seed": seed,
            "source_seed": source_seed,
            "target_seed": target_seed,
            "source_training_seed": source_training_seed,
            "target_training_seed": target_training_seed,
            "evaluation_scene_seeds": evaluation_scene_seeds,
            "source_tag": source_tag,
            "source_checkpoint": source_authority["checkpoint"],
            "source_checkpoint_sha256": source_authority["checkpoint_sha256"],
            "source_checkpoint_identity": source_authority[
                "checkpoint_identity"
            ],
            "source_checkpoint_identity_sha256": source_authority[
                "checkpoint_identity_sha256"
            ],
            "source_identity": source_authority["identity"],
            "source_training": serializable_training_result(source),
            "methods": methods,
        })
        atomic_write_json(progress_path, {"completed_seeds": seed_records})

    training_final_window = min(100, args.episodes)
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
        "protocol": protocol,
        "run_mode": args.run_mode,
        "publication_eligible": args.run_mode == "publication",
        "physical_configuration_fingerprint": physics,
        "profile_digest": profile.digest,
        "objective": objective.as_dict(),
        "power_reference": cfg.PT_reference,
        "power_hypothesis_id": cfg.power_hypothesis_id,
        "source_tree_digest": manifest["source_tree_digest"],
        "training_seed_protocol": TRAINING_SEED_PROTOCOL,
        "evaluation_seed_protocol": EVALUATION_SEED_PROTOCOL,
        "seeds": list(seeds),
        "episodes": args.episodes,
        "figure4_protocol": (
            {
                "raw_episode_count": cfg.fig4_episodes,
                "plot_x_start": 0,
                "plot_x_end": 1400,
                "plot_interval": cfg.fig4_plot_interval,
                "acceptance_smoothing_window": 20,
            }
            if args.run_mode == "publication"
            else None
        ),
        "diagnostic_scope": (
            None
            if args.run_mode == "publication"
            else {
                "purpose": "early_scale_feasibility_projection_checkpoint_check",
                "not_evidence_for_paper_curve_acceptance": True,
            }
        ),
        "evaluation_episodes": args.eval_episodes,
        "final_reward_source": "training_communication_rate",
        "training_final_window": training_final_window,
        "pretraining_physical_sanity_sha256": manifest[
            "pretraining_physical_sanity_sha256"
        ],
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
    if args.run_mode == "smoke":
        from validate_results import validate_smoke_result

        diagnostic = validate_smoke_result(
            summary, result_directory=output, verify_files=True
        )
        atomic_write_json(output / "smoke_validation.json", diagnostic)
        return 0
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
