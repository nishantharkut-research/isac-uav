"""Run the sensing extension against a completed paired baseline."""

import argparse
import hashlib
from pathlib import Path

from ambiguity_hypotheses import add_ambiguity_cli_arguments


OBJECTIVE_CHOICES = (
    "png_joint_sum", "comm_retention_sensing", "normalized_pareto",
)


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--baseline-validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated, at least five")
    parser.add_argument("--objective", required=True, choices=OBJECTIVE_CHOICES)
    parser.add_argument("--retention", type=float)
    parser.add_argument(
        "--pareto-weight",
        type=float,
        default=0.5,
        help="legacy normalized sensing weight; communication uses 1 - weight",
    )
    parser.add_argument("--user-rate-min", type=float)
    parser.add_argument("--target-rate-min", type=float)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return add_ambiguity_cli_arguments(parser)


def _load_json(path):
    import json
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def require_accepted_baseline(validation, baseline):
    """Authorize extension work only from the exact accepted baseline payload."""
    from validate_results import result_payload_digest

    if (
        validation.get("validation") != "paper_baseline"
        or validation.get("overall_pass") is not True
    ):
        raise ValueError("The baseline has not passed paper-baseline validation.")
    if validation.get("baseline_payload_sha256") != result_payload_digest(baseline):
        raise ValueError("The validation report belongs to a different baseline.")


def require_comparable_budget(args, baseline):
    """Keep extension training and evaluation budgets paired with baseline."""
    if int(args.episodes) != int(baseline.get("episodes", -1)):
        raise ValueError("Extension training budget must match the baseline.")
    if int(args.eval_episodes) != int(baseline.get("evaluation_episodes", -1)):
        raise ValueError("Extension evaluation budget must match the baseline.")


def resolve_extension_rate_floors(args, cfg):
    """Resolve the two PNG floors without allowing a vacuous literal mode."""
    import math

    user_floor = cfg.R_u_min if args.user_rate_min is None else args.user_rate_min
    target_floor = cfg.R_min if args.target_rate_min is None else args.target_rate_min
    user_floor = float(user_floor)
    target_floor = float(target_floor)
    if not math.isfinite(user_floor) or user_floor < 0.0:
        raise ValueError("The per-UE rate floor must be finite and nonnegative.")
    if not math.isfinite(target_floor) or target_floor <= 0.0:
        raise ValueError("The sensing extension requires a positive target-rate floor.")
    if args.objective == "png_joint_sum" and user_floor <= 0.0:
        raise ValueError("The literal PNG objective requires a positive per-UE rate floor.")
    return user_floor, target_floor


def build_extension_objective_manifest(args, user_floor, target_floor):
    """Build only fields that are meaningful for the selected objective."""
    if args.objective != "comm_retention_sensing" and args.retention is not None:
        raise ValueError(
            "--retention is only valid with comm_retention_sensing."
        )
    payload = {
        "mode": args.objective,
        "user_rate_min": float(user_floor),
        "target_rate_min": float(target_floor),
    }
    if args.objective == "comm_retention_sensing":
        retention = 0.95 if args.retention is None else float(args.retention)
        if not 0.0 < retention <= 1.0:
            raise ValueError("Communication retention must lie in (0, 1].")
        payload["communication_retention_fraction"] = retention
    elif args.objective == "normalized_pareto":
        payload["pareto_weight"] = float(args.pareto_weight)
    return payload


def run(args):
    from ambiguity_hypotheses import (
        apply_ambiguity_arguments,
        apply_ambiguity_manifest_fields,
        require_matching_ambiguity_contract,
    )
    from pathlib import Path
    from config import cfg
    from environment import HRISUAVISACEnv
    from experiment_protocol import (
        atomic_write_json, ensure_run_directory, evaluation_metrics,
        final_window_mean,
        make_run_manifest, parse_seeds, physical_configuration_fingerprint,
        resolve_required_checkpoint,
        serializable_training_result,
        write_training_checkpoint_identity,
        EVALUATION_SEED_PROTOCOL, TRAINING_SEED_PROTOCOL,
        evaluation_scene_seed_grid,
        derive_training_seed,
    )
    from parameter_profiles import activate_profile, load_calibrated_profile
    from research_objectives import ObjectiveSpec
    from simulation import (
        train_ddpg,
        train_dtl,
    )
    from source_integrity import source_tree_digest

    seeds = parse_seeds(args.seeds)
    profile = load_calibrated_profile(args.profile)
    activate_profile(profile)
    apply_ambiguity_arguments(args, cfg)
    baseline_path = Path(args.baseline_summary).resolve()
    baseline = _load_json(baseline_path)
    baseline_validation_path = Path(args.baseline_validation).resolve()
    baseline_validation = _load_json(baseline_validation_path)
    require_accepted_baseline(baseline_validation, baseline)
    if baseline.get("training_seed_protocol") != TRAINING_SEED_PROTOCOL:
        raise ValueError("Baseline does not use the required training-seed protocol.")
    require_comparable_budget(args, baseline)
    if baseline.get("kind") != "generated_result" or baseline.get("protocol") != "paper_baseline":
        raise ValueError("--baseline-summary must be a completed generated baseline.")
    if tuple(baseline.get("seeds", [])) != seeds:
        raise ValueError("Extension seeds must exactly match baseline seeds.")
    physics = physical_configuration_fingerprint(profile.digest, cfg)
    if baseline.get("profile_digest") != profile.digest or baseline.get(
        "physical_configuration_fingerprint"
    ) != physics:
        raise ValueError("Baseline profile/physical configuration does not match.")
    current_source_digest = source_tree_digest()
    if baseline.get("source_tree_digest") != current_source_digest:
        raise ValueError("Baseline was generated by a different source tree.")
    if baseline.get("power_hypothesis_id") != cfg.power_hypothesis_id:
        raise ValueError("Baseline used a different power hypothesis.")
    require_matching_ambiguity_contract(baseline, cfg)
    from validate_results import (
        require_canonical_baseline_validation,
        result_payload_digest,
    )
    require_canonical_baseline_validation(
        baseline, baseline_validation, baseline_path.parent
    )
    baseline_payload_digest = result_payload_digest(baseline)
    baseline_validation_digest = result_payload_digest(baseline_validation)

    user_floor, target_floor = resolve_extension_rate_floors(args, cfg)
    objective_manifest = build_extension_objective_manifest(
        args, user_floor, target_floor
    )
    manifest = make_run_manifest(
        "sensing_extension", profile.digest, physics, objective_manifest, seeds
    )
    apply_ambiguity_manifest_fields(manifest, cfg)
    manifest["power_reference"] = cfg.PT_reference
    manifest["power_hypothesis_id"] = cfg.power_hypothesis_id
    manifest["source_tree_digest"] = current_source_digest
    manifest["training_seed_protocol"] = TRAINING_SEED_PROTOCOL
    manifest["evaluation_seed_protocol"] = EVALUATION_SEED_PROTOCOL
    manifest["baseline_payload_sha256"] = baseline_payload_digest
    manifest["baseline_validation_sha256"] = baseline_validation_digest
    manifest["episodes"] = args.episodes
    manifest["evaluation_episodes"] = args.eval_episodes
    manifest["objective_reference_protocol"] = (
        "accepted_baseline_training_tail_v1"
    )
    manifest["objective_reference_window"] = 100
    evaluation_seed_grid = evaluation_scene_seed_grid(
        seeds, args.eval_episodes
    )
    output = ensure_run_directory(args.output, manifest)
    if args.dry_run:
        return 0
    baseline_by_seed = {int(row["seed"]): row for row in baseline["seed_records"]}
    checkpoint_dir = output / "checkpoints"
    seed_records = []
    objective_reference_window = 100
    for seed in seeds:
        baseline_record = baseline_by_seed[seed]
        evaluation_scene_seeds = evaluation_seed_grid[seed]
        if (
            baseline.get("evaluation_seed_protocol")
            != EVALUATION_SEED_PROTOCOL
            or baseline_record.get("evaluation_scene_seeds")
            != evaluation_scene_seeds
        ):
            raise ValueError(
                "Baseline evaluation scenes do not match the canonical paired "
                "schedule."
            )
        baseline_training = baseline_record["methods"]["DTL-pCSI"]["training"]
        comm_reference = final_window_mean(
            baseline_training["comm_rates"], objective_reference_window
        )
        sensing_reference = final_window_mean(
            baseline_training["sensing_rates"], objective_reference_window
        )
        if args.objective == "png_joint_sum":
            objective = ObjectiveSpec.png_joint(
                user_rate_min=user_floor,
                target_rate_min=target_floor,
            )
        elif args.objective == "comm_retention_sensing":
            objective = ObjectiveSpec.communication_retention(
                communication_reference=comm_reference,
                communication_retention=objective_manifest[
                    "communication_retention_fraction"
                ],
                user_rate_min=user_floor,
                target_rate_min=target_floor,
            )
        else:
            objective = ObjectiveSpec.normalized_pareto(
                communication_reference=comm_reference,
                sensing_reference=sensing_reference,
                pareto_weight=args.pareto_weight,
                user_rate_min=user_floor,
                target_rate_min=target_floor,
            )

        source_tag = baseline_record["source_tag"]
        source_checkpoint = resolve_required_checkpoint(
            baseline_path.parent, baseline_record["source_checkpoint"]
        )
        if _file_sha256(source_checkpoint) != baseline_record.get(
            "source_checkpoint_sha256"
        ):
            raise ValueError(
                "Accepted baseline source checkpoint no longer matches its summary."
            )
        source_env = HRISUAVISACEnv(
            sigma2_scee=cfg.rho_nmse,
            seed=baseline_record["source_seed"],
            objective_spec=ObjectiveSpec.paper(target_rate_min=cfg.R_min),
            allocation_mode=cfg.policy_allocation_mode,
        )
        source = train_ddpg(
            source_env, baseline["episodes"], source_tag,
            checkpoint_path=source_checkpoint, resume=True,
            checkpoint_every=args.checkpoint_every,
            training_seed=int(baseline_record.get(
                "source_training_seed", derive_training_seed(seed, "source")
            )),
        )

        target_seed = int(baseline_record["target_seed"])
        target_training_seed = int(baseline_record.get(
            "target_training_seed", derive_training_seed(seed, "target")
        ))
        tag = f"extension-{args.objective}-seed-{seed}"
        environment = HRISUAVISACEnv(
            sigma2_scee=0.0, seed=target_seed, objective_spec=objective,
            allocation_mode=cfg.policy_allocation_mode,
        )
        checkpoint = checkpoint_dir / f"{tag}.pt"
        trained = train_dtl(
            source["agent"], environment, args.episodes, tag,
            checkpoint_path=checkpoint, resume=not args.fresh,
            checkpoint_every=args.checkpoint_every,
            training_seed=target_training_seed,
        )

        def environment_factory(index):
            return HRISUAVISACEnv(
                sigma2_scee=0.0,
                seed=evaluation_scene_seeds[index],
                objective_spec=objective,
                allocation_mode=cfg.policy_allocation_mode,
            )

        evaluation = evaluation_metrics(
            trained["agent"], environment_factory, args.eval_episodes
        )
        checkpoint_authority = write_training_checkpoint_identity(
            output, checkpoint, trained["checkpoint_identity"]
        )
        seed_records.append({
            "seed": seed,
            "objective": objective.as_dict(),
            "tag": tag,
            "training_seed": target_training_seed,
            "environment_seed": target_seed,
            "evaluation_scene_seeds": evaluation_scene_seeds,
            "objective_reference": {
                "source": "accepted_baseline_training_tail",
                "window": objective_reference_window,
                "communication": comm_reference,
                "sensing": sensing_reference,
            },
            **checkpoint_authority,
            "training": serializable_training_result(trained),
            "evaluation": evaluation,
        })
        atomic_write_json(output / "extension_progress.json", {
            "completed_seeds": seed_records
        })

    summary = {
        "schema_version": 1,
        "kind": "generated_result",
        "protocol": "sensing_extension",
        "physical_configuration_fingerprint": physics,
        "profile_digest": profile.digest,
        "objective": objective_manifest,
        "power_reference": cfg.PT_reference,
        "power_hypothesis_id": cfg.power_hypothesis_id,
        "source_tree_digest": current_source_digest,
        "training_seed_protocol": TRAINING_SEED_PROTOCOL,
        "evaluation_seed_protocol": EVALUATION_SEED_PROTOCOL,
        "baseline_payload_sha256": baseline_payload_digest,
        "baseline_validation_sha256": baseline_validation_digest,
        "seeds": list(seeds),
        "episodes": args.episodes,
        "evaluation_episodes": args.eval_episodes,
        "objective_reference_protocol": "accepted_baseline_training_tail_v1",
        "objective_reference_window": objective_reference_window,
        "feasibility_rate": min(
            row["evaluation"]["feasibility_rate"] for row in seed_records
        ),
        "communication_seed_values": [
            row["evaluation"]["communication_mean"] for row in seed_records
        ],
        "sensing_seed_values": [
            row["evaluation"]["sensing_mean"] for row in seed_records
        ],
        "joint_seed_values": [
            row["evaluation"]["communication_mean"]
            + row["evaluation"]["sensing_mean"]
            for row in seed_records
        ],
        "objective_seed_values": [
            row["evaluation"]["objective_mean"] for row in seed_records
        ],
        "minimum_ue_rate_seed_values": [
            row["evaluation"]["minimum_ue_rate"] for row in seed_records
        ],
        "minimum_target_rate_seed_values": [
            row["evaluation"]["minimum_target_rate"] for row in seed_records
        ],
        "seed_records": seed_records,
    }
    apply_ambiguity_manifest_fields(summary, cfg)
    atomic_write_json(output / "extension_summary.json", summary)
    manifest["status"] = "complete"
    manifest["summary"] = "extension_summary.json"
    manifest["summary_payload_sha256"] = result_payload_digest(summary)
    atomic_write_json(output / "run_manifest.json", manifest)
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
