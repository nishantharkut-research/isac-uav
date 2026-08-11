"""Run the paired BA-PCDTL research extension from an accepted reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from ambiguity_hypotheses import add_ambiguity_cli_arguments
from pareto_agent import ALGORITHM_ID
from pareto_protocol import hypervolume_2d, validate_preferences


DEFAULT_PREFERENCE_GRID = (
    "1:0,0.875:0.125,0.75:0.25,0.625:0.375,0.5:0.5,"
    "0.375:0.625,0.25:0.75,0.125:0.875,0:1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--baseline-validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--research-seeds", required=True)
    parser.add_argument("--validation-scene-seeds", required=True)
    parser.add_argument("--test-scene-seeds", required=True)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--preference-grid", default=DEFAULT_PREFERENCE_GRID)
    parser.add_argument("--comparison-level", choices=("primary", "full"), default="primary")
    parser.add_argument("--user-rate-min", type=float)
    parser.add_argument("--target-rate-min", type=float)
    parser.add_argument("--residual-radius", type=float, default=0.25)
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--residual-scales", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--gate-uncertainty-grid", default="0,0.5,1,2")
    parser.add_argument("--gate-margin-grid", default="0,0.01,0.05")
    parser.add_argument("--dry-run", action="store_true")
    return add_ambiguity_cli_arguments(parser)


def _parse_integer_list(values, name: str) -> list[int]:
    if isinstance(values, str):
        parsed = [int(item.strip()) for item in values.split(",") if item.strip()]
    else:
        parsed = [int(item) for item in values]
    if not parsed or len(parsed) != len(set(parsed)) or any(item < 0 for item in parsed):
        raise ValueError(f"{name} must be a nonempty list of unique nonnegative integers.")
    return parsed


def validate_seed_roles(research, validation, test) -> dict[str, list[int]]:
    roles = {
        "research": _parse_integer_list(research, "research seeds"),
        "validation": _parse_integer_list(validation, "validation scene seeds"),
        "test": _parse_integer_list(test, "test scene seeds"),
    }
    minimums = {"research": 10, "validation": 5, "test": 10}
    for role, minimum in minimums.items():
        if len(roles[role]) < minimum:
            raise ValueError(f"{role} requires at least {minimum} independent seeds.")
    for left, right in (("research", "validation"), ("research", "test"), ("validation", "test")):
        overlap = set(roles[left]) & set(roles[right])
        if overlap:
            raise ValueError(f"Seed roles {left} and {right} overlap: {sorted(overlap)}.")
    return roles


def parse_preference_grid(value) -> list[list[float]]:
    if isinstance(value, str):
        rows = []
        for item in value.split(","):
            parts = item.strip().split(":")
            if len(parts) != 2:
                raise ValueError("Each preference must use communication:sensing syntax.")
            rows.append([float(parts[0]), float(parts[1])])
    else:
        rows = value
    preferences = validate_preferences(rows)
    if len({tuple(row) for row in preferences.tolist()}) != len(preferences):
        raise ValueError("Preference grid entries must be unique.")
    if not any(
        math.isclose(row[0], 0.5, abs_tol=1e-8)
        and math.isclose(row[1], 0.5, abs_tol=1e-8)
        for row in preferences
    ):
        raise ValueError("Preference grid must contain the balanced PNG operating point.")
    return preferences.tolist()


def parse_float_grid(value, *, name: str) -> list[float]:
    if isinstance(value, str):
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        values = [float(item) for item in value]
    if (
        not values
        or len(values) != len(set(values))
        or any(not math.isfinite(item) or item < 0.0 for item in values)
    ):
        raise ValueError(f"{name} must contain unique finite nonnegative values.")
    return values


def _load_json(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def require_accepted_baseline(validation: Mapping, baseline: Mapping) -> None:
    from validate_results import result_payload_digest

    if validation.get("validation") != "paper_baseline" or validation.get("overall_pass") is not True:
        raise ValueError("The baseline has not passed paper-baseline validation.")
    if validation.get("baseline_payload_sha256") != result_payload_digest(baseline):
        raise ValueError("The validation report belongs to a different baseline payload.")


def validate_baseline_contract(
    baseline: Mapping,
    *,
    profile_digest: str,
    physics_fingerprint: str,
    source_tree_digest: str,
    training_seed_protocol: str,
    power_hypothesis_id: str,
    research_seeds,
    episodes: int,
    evaluation_episodes: int,
) -> None:
    expected = {
        "kind": "generated_result",
        "protocol": "paper_baseline",
        "profile_digest": profile_digest,
        "physical_configuration_fingerprint": physics_fingerprint,
        "source_tree_digest": source_tree_digest,
        "training_seed_protocol": training_seed_protocol,
        "power_hypothesis_id": power_hypothesis_id,
        "seeds": list(research_seeds),
        "episodes": int(episodes),
        "evaluation_episodes": int(evaluation_episodes),
    }
    mismatches = [key for key, value in expected.items() if baseline.get(key) != value]
    if mismatches:
        raise ValueError("Accepted baseline contract mismatch: " + ", ".join(mismatches))


def create_research_manifest(
    *,
    profile_digest: str,
    physics_fingerprint: str,
    source_tree_digest: str,
    baseline_payload_digest: str,
    baseline_validation_digest: str,
    baseline_checkpoint_authority,
    seed_roles: Mapping[str, list[int]],
    preference_grid,
    episodes: int,
    evaluation_episodes: int,
    comparison_level: str,
    residual_radius: float,
    ensemble_size: int,
    residual_scales,
    uncertainty_grid,
    margin_grid,
    objective: Mapping,
) -> dict:
    if comparison_level not in {"primary", "full"}:
        raise ValueError("comparison_level must be primary or full.")
    if not 0.0 < float(residual_radius) <= 1.0 or int(ensemble_size) < 2:
        raise ValueError("Residual radius and ensemble size violate BA-PCDTL contract.")
    authority = [dict(row) for row in baseline_checkpoint_authority]
    if [row.get("seed") for row in authority] != list(seed_roles["research"]):
        raise ValueError("Baseline checkpoint authority must match research seeds.")
    for row in authority:
        for field in ("source_checkpoint_sha256", "dtl_pcsi_checkpoint_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("Baseline checkpoint authority has an invalid digest.")
    return {
        "schema_version": 1,
        "kind": "run_manifest",
        "status": "planned",
        "protocol": "ba_pcdtl_research_extension",
        "algorithm_id": ALGORITHM_ID,
        "profile_digest": str(profile_digest),
        "physical_configuration_fingerprint": str(physics_fingerprint),
        "source_tree_digest": str(source_tree_digest),
        "baseline_payload_sha256": str(baseline_payload_digest),
        "baseline_validation_sha256": str(baseline_validation_digest),
        "baseline_checkpoint_authority": authority,
        "objective": dict(objective),
        "seeds": list(seed_roles["research"]),
        "research_seeds": list(seed_roles["research"]),
        "validation_scene_seeds": list(seed_roles["validation"]),
        "test_scene_seeds": list(seed_roles["test"]),
        "preference_grid": parse_preference_grid(preference_grid),
        "episodes": int(episodes),
        "evaluation_episodes": int(evaluation_episodes),
        "comparison_level": comparison_level,
        "residual_radius": float(residual_radius),
        "ensemble_size": int(ensemble_size),
        "residual_scales": [float(item) for item in residual_scales],
        "gate_selection_grid": {
            "uncertainty_coefficients": [float(item) for item in uncertainty_grid],
            "margins": [float(item) for item in margin_grid],
            "selection_partition": "validation_only",
            "tie_break_order": [
                "feasibility", "png_joint_sum", "pareto_hypervolume",
                "lower_fallback_rate", "lexicographic_parameters",
            ],
        },
        "hypervolume_reference_normalized": [-1.0, -1.0],
    }


def select_gate_configuration(candidates) -> dict:
    rows = [dict(row) for row in candidates]
    if not rows:
        raise ValueError("Gate selection requires at least one validation candidate.")
    required = (
        "uncertainty_coefficient", "margin", "feasibility", "joint",
        "hypervolume", "fallback_rate",
    )
    for row in rows:
        if any(key not in row for key in required):
            raise ValueError("Gate candidate is incomplete.")
        if any(not math.isfinite(float(row[key])) for key in required):
            raise ValueError("Gate candidate metrics must be finite.")
        if not 0.0 <= float(row["feasibility"]) <= 1.0:
            raise ValueError("Gate feasibility must lie in [0, 1].")
        if not 0.0 <= float(row["fallback_rate"]) <= 1.0:
            raise ValueError("Gate fallback rate must lie in [0, 1].")
    selected = max(
        rows,
        key=lambda row: (
            float(row["feasibility"]),
            float(row["joint"]),
            float(row["hypervolume"]),
            -float(row["fallback_rate"]),
            -float(row["uncertainty_coefficient"]),
            -float(row["margin"]),
        ),
    )
    return selected


def write_immutable_json(path, payload: Mapping) -> Path:
    from experiment_protocol import atomic_write_json

    path = Path(path)
    if path.exists():
        existing = _load_json(path)
        if existing != dict(payload):
            raise ValueError(f"Existing lock is immutable and differs: {path}")
        return path
    return atomic_write_json(path, dict(payload))


def _build_baseline_checkpoint_authority(
    baseline, baseline_root, resolve_checkpoint
) -> list[dict]:
    records = baseline.get("seed_records")
    if not isinstance(records, list) or len(records) != len(baseline.get("seeds", [])):
        raise ValueError("Accepted baseline has incomplete seed checkpoint records.")
    authority = []
    for record in records:
        try:
            source_relative = record["source_checkpoint"]
            target_record = record["methods"]["DTL-pCSI"]
            target_relative = target_record["checkpoint"]
        except (KeyError, TypeError) as error:
            raise ValueError("Accepted baseline lacks source or DTL-pCSI checkpoint.") from error
        source_path = resolve_checkpoint(baseline_root, source_relative)
        target_path = resolve_checkpoint(baseline_root, target_relative)
        source_digest = _file_sha256(source_path)
        target_digest = _file_sha256(target_path)
        if (
            source_digest != record.get("source_checkpoint_sha256")
            or target_digest != target_record.get("checkpoint_sha256")
        ):
            raise ValueError(
                "Accepted baseline checkpoint no longer matches its summary hash."
            )
        authority.append({
            "seed": int(record["seed"]),
            "source_checkpoint": str(source_relative),
            "source_checkpoint_sha256": source_digest,
            "dtl_pcsi_checkpoint": str(target_relative),
            "dtl_pcsi_checkpoint_sha256": target_digest,
        })
    return authority


def run(args) -> int:
    from ambiguity_hypotheses import (
        apply_ambiguity_arguments,
        apply_ambiguity_manifest_fields,
        require_matching_ambiguity_contract,
    )
    from config import cfg
    from experiment_protocol import (
        TRAINING_SEED_PROTOCOL,
        ensure_run_directory,
        physical_configuration_fingerprint,
        resolve_required_checkpoint,
    )
    from parameter_profiles import activate_profile, load_calibrated_profile, load_profile
    from source_integrity import source_tree_digest
    from validate_results import result_payload_digest

    roles = validate_seed_roles(
        args.research_seeds, args.validation_scene_seeds, args.test_scene_seeds
    )
    preferences = parse_preference_grid(args.preference_grid)
    residual_scales = parse_float_grid(args.residual_scales, name="residual scales")
    if 0.0 not in residual_scales or 1.0 not in residual_scales:
        raise ValueError("Residual scale grid must include zero and one.")
    uncertainty_grid = parse_float_grid(
        args.gate_uncertainty_grid, name="gate uncertainty grid"
    )
    margin_grid = parse_float_grid(args.gate_margin_grid, name="gate margin grid")
    if args.episodes <= 0 or args.eval_episodes <= 0 or args.checkpoint_every <= 0:
        raise ValueError("Training, evaluation, and checkpoint budgets must be positive.")

    profile = load_profile(args.profile) if args.dry_run else load_calibrated_profile(args.profile)
    activate_profile(profile)
    apply_ambiguity_arguments(args, cfg)
    baseline = _load_json(args.baseline_summary)
    validation = _load_json(args.baseline_validation)
    require_accepted_baseline(validation, baseline)
    require_matching_ambiguity_contract(baseline, cfg)
    physics = physical_configuration_fingerprint(profile.digest, cfg)
    tree_digest = source_tree_digest()
    validate_baseline_contract(
        baseline,
        profile_digest=profile.digest,
        physics_fingerprint=physics,
        source_tree_digest=tree_digest,
        training_seed_protocol=TRAINING_SEED_PROTOCOL,
        power_hypothesis_id=cfg.power_hypothesis_id,
        research_seeds=roles["research"],
        episodes=args.episodes,
        evaluation_episodes=args.eval_episodes,
    )
    baseline_checkpoint_authority = _build_baseline_checkpoint_authority(
        baseline, Path(args.baseline_summary).resolve().parent,
        resolve_required_checkpoint,
    )
    user_floor = cfg.R_u_min if args.user_rate_min is None else float(args.user_rate_min)
    target_floor = cfg.R_min if args.target_rate_min is None else float(args.target_rate_min)
    if not math.isfinite(user_floor) or user_floor <= 0.0:
        raise ValueError("The PNG per-UE rate floor must be finite and positive.")
    if not math.isfinite(target_floor) or target_floor <= 0.0:
        raise ValueError("The PNG per-target rate floor must be finite and positive.")
    objective = {
        "mode": "png_joint_sum",
        "user_rate_min": user_floor,
        "target_rate_min": target_floor,
    }
    manifest = create_research_manifest(
        profile_digest=profile.digest,
        physics_fingerprint=physics,
        source_tree_digest=tree_digest,
        baseline_payload_digest=result_payload_digest(baseline),
        baseline_validation_digest=result_payload_digest(validation),
        baseline_checkpoint_authority=baseline_checkpoint_authority,
        seed_roles=roles,
        preference_grid=preferences,
        episodes=args.episodes,
        evaluation_episodes=args.eval_episodes,
        comparison_level=args.comparison_level,
        residual_radius=args.residual_radius,
        ensemble_size=args.ensemble_size,
        residual_scales=residual_scales,
        uncertainty_grid=uncertainty_grid,
        margin_grid=margin_grid,
        objective=objective,
    )
    manifest["training_seed_protocol"] = TRAINING_SEED_PROTOCOL
    manifest["power_hypothesis_id"] = cfg.power_hypothesis_id
    apply_ambiguity_manifest_fields(manifest, cfg)
    output = ensure_run_directory(args.output, manifest)
    if args.dry_run:
        return 0
    return _execute_research_run(args, output, manifest, baseline, validation)


def _execute_research_run(args, output, manifest, baseline, validation) -> int:
    from ambiguity_hypotheses import apply_ambiguity_manifest_fields
    from config import cfg
    from environment import HRISUAVISACEnv
    from experiment_protocol import (
        TRAINING_SEED_PROTOCOL,
        atomic_write_json,
        derive_training_seed,
        resolve_required_checkpoint,
        serializable_training_result,
    )
    from pareto_protocol import ParetoAnchors
    from pareto_training import evaluate_preference_grid
    from research_objectives import ObjectiveSpec
    from simulation import train_ddpg, train_dtl
    from validate_results import result_payload_digest

    objective = ObjectiveSpec.png_joint(
        user_rate_min=manifest["objective"]["user_rate_min"],
        target_rate_min=manifest["objective"]["target_rate_min"],
    )
    paper_objective = ObjectiveSpec.paper(target_rate_min=cfg.R_min)
    output = Path(output)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    baseline_root = Path(args.baseline_summary).resolve().parent
    baseline_by_seed = {int(row["seed"]): row for row in baseline["seed_records"]}
    if set(baseline_by_seed) != set(manifest["research_seeds"]):
        raise ValueError("Baseline seed records do not match the immutable research seeds.")
    progress_path = output / "pareto_progress.json"
    checkpoint_authority_by_seed = {
        int(row["seed"]): row for row in manifest["baseline_checkpoint_authority"]
    }

    def environment(seed, objective_spec=objective, sigma2=0.0, allocation="fast"):
        return HRISUAVISACEnv(
            sigma2_scee=float(sigma2),
            seed=int(seed),
            objective_spec=objective_spec,
            allocation_mode=allocation,
        )

    def restore_paper_agents(seed):
        record = baseline_by_seed[int(seed)]
        source_path = resolve_required_checkpoint(
            baseline_root, record["source_checkpoint"]
        )
        authority = checkpoint_authority_by_seed[int(seed)]
        _require_checkpoint_hash(
            source_path, authority["source_checkpoint_sha256"]
        )
        source = train_ddpg(
            environment(record["source_seed"], paper_objective, cfg.rho_nmse),
            baseline["episodes"],
            record["source_tag"],
            checkpoint_path=source_path,
            resume=True,
            checkpoint_every=args.checkpoint_every,
            training_seed=int(
                record.get("source_training_seed", derive_training_seed(seed, "source"))
            ),
        )
        target_record = record["methods"]["DTL-pCSI"]
        target_path = resolve_required_checkpoint(
            baseline_root, target_record["checkpoint"]
        )
        _require_checkpoint_hash(
            target_path, authority["dtl_pcsi_checkpoint_sha256"]
        )
        target = train_dtl(
            source["agent"],
            environment(record["target_seed"], paper_objective, 0.0),
            baseline["episodes"],
            target_record["tag"],
            checkpoint_path=target_path,
            resume=True,
            checkpoint_every=args.checkpoint_every,
            training_seed=int(
                record.get("target_training_seed", derive_training_seed(seed, "target"))
            ),
        )
        return source, target

    scalar_records = []
    anchors = ParetoAnchors()
    for seed in manifest["research_seeds"]:
        record = baseline_by_seed[int(seed)]
        paper_source, _paper_target = restore_paper_agents(seed)
        source_seed = int(record["source_seed"])
        target_seed = int(record["target_seed"])
        source_training_seed = derive_training_seed(seed, "source")
        target_training_seed = derive_training_seed(seed, "target")

        joint_source_tag = f"joint-source-seed-{seed}"
        joint_source_path = checkpoint_dir / f"{joint_source_tag}.pt"
        joint_source = train_ddpg(
            environment(source_seed, objective, cfg.rho_nmse),
            args.episodes,
            joint_source_tag,
            checkpoint_path=joint_source_path,
            resume=True,
            checkpoint_every=args.checkpoint_every,
            training_seed=source_training_seed,
        )
        joint_target_tag = f"same-objective-dtl-seed-{seed}"
        joint_target_path = checkpoint_dir / f"{joint_target_tag}.pt"
        joint_target = train_dtl(
            joint_source["agent"],
            environment(target_seed, objective, 0.0),
            args.episodes,
            joint_target_tag,
            checkpoint_path=joint_target_path,
            resume=True,
            checkpoint_every=args.checkpoint_every,
            training_seed=target_training_seed,
        )

        cross_tag = f"cross-objective-dtl-seed-{seed}"
        cross_path = checkpoint_dir / f"{cross_tag}.pt"
        cross_target = train_dtl(
            paper_source["agent"],
            environment(target_seed, objective, 0.0),
            args.episodes,
            cross_tag,
            checkpoint_path=cross_path,
            resume=True,
            checkpoint_every=args.checkpoint_every,
            training_seed=target_training_seed,
        )

        scratch_tag = f"joint-ddpg-scratch-seed-{seed}"
        scratch_path = checkpoint_dir / f"{scratch_tag}.pt"
        scratch = train_ddpg(
            environment(target_seed, objective, 0.0),
            args.episodes,
            scratch_tag,
            checkpoint_path=scratch_path,
            resume=True,
            checkpoint_every=args.checkpoint_every,
            training_seed=target_training_seed,
        )

        scalar_methods = {
            "same_objective_dtl": (joint_target, joint_target_path),
            "cross_objective_dtl_diagnostic": (cross_target, cross_path),
            "ddpg_scratch": (scratch, scratch_path),
        }
        for trained, _path in scalar_methods.values():
            values = np.column_stack(
                [trained["comm_rates"], trained["sensing_rates"]]
            )
            anchors.update_training(values)
        scalar_records.append({
            "seed": int(seed),
            "joint_source_checkpoint": str(joint_source_path.relative_to(output)),
            "joint_source_checkpoint_sha256": _file_sha256(joint_source_path),
            "methods": {
                name: {
                    "checkpoint": str(path.relative_to(output)),
                    "checkpoint_sha256": _file_sha256(path),
                    "training": serializable_training_result(trained),
                }
                for name, (trained, path) in scalar_methods.items()
            },
        })
        atomic_write_json(progress_path, {
            "stage": "scalar_comparators",
            "completed_scalar_seeds": scalar_records,
        })

    anchor_provenance = {
        "partition": "training_only",
        "research_seeds": manifest["research_seeds"],
        "source_methods": [
            "same_objective_dtl", "cross_objective_dtl_diagnostic", "ddpg_scratch"
        ],
        "source_checkpoint_sha256": [
            method["checkpoint_sha256"]
            for record in scalar_records
            for method in record["methods"].values()
        ],
    }
    anchors.freeze(anchor_provenance)
    anchor_lock = {
        "schema_version": 1,
        "anchors": anchors.state_dict(),
        "normalized_hypervolume_reference": manifest[
            "hypervolume_reference_normalized"
        ],
    }
    anchor_lock_path = write_immutable_json(
        output / "pareto_anchors_lock.json", anchor_lock
    )
    anchor_lock_digest = _file_sha256(anchor_lock_path)

    vector_records = []
    gate_validation_rows = []
    for seed in manifest["research_seeds"]:
        record = baseline_by_seed[int(seed)]
        _paper_source, paper_target = restore_paper_agents(seed)
        target_seed = int(record["target_seed"])
        vector_seed = _derive_research_seed(seed, "ba_pcdtl")

        proposed, proposed_training = _train_vector_method(
            baseline_agent=paper_target["agent"],
            environment=environment(target_seed, objective, 0.0),
            output=output,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            method="ba_pcdtl",
            training_seed=vector_seed,
            anchors=anchors,
            manifest=manifest,
            episodes=args.episodes,
            checkpoint_every=args.checkpoint_every,
        )
        methods = {
            "ba_pcdtl": {
                "checkpoint": proposed_training["checkpoint"],
                "checkpoint_sha256": _file_sha256(
                    output / proposed_training["checkpoint"]
                ),
                "baseline_policy_digest": proposed.baseline_policy_digest,
                "identity": proposed_training["identity"],
                "training": proposed_training["training"],
            }
        }

        if manifest["comparison_level"] == "full":
            unanchored, unanchored_training = _train_vector_method(
                baseline_agent=paper_target["agent"],
                environment=environment(target_seed, objective, 0.0),
                output=output,
                checkpoint_dir=checkpoint_dir,
                seed=seed,
                method="unanchored_vector_td3",
                training_seed=_derive_research_seed(seed, "unanchored_vector_td3"),
                anchors=anchors,
                manifest=manifest,
                episodes=args.episodes,
                checkpoint_every=args.checkpoint_every,
                residual_radius=1.0,
                transfer_trunks=False,
                anchor_enabled=False,
            )
            random_trunks, random_training = _train_vector_method(
                baseline_agent=paper_target["agent"],
                environment=environment(target_seed, objective, 0.0),
                output=output,
                checkpoint_dir=checkpoint_dir,
                seed=seed,
                method="anchored_random_trunks",
                training_seed=_derive_research_seed(seed, "anchored_random_trunks"),
                anchors=anchors,
                manifest=manifest,
                episodes=args.episodes,
                checkpoint_every=args.checkpoint_every,
                transfer_trunks=False,
            )
            methods["unanchored_vector_td3"] = _vector_training_record(
                unanchored, unanchored_training, output
            )
            methods["anchored_random_trunks"] = _vector_training_record(
                random_trunks, random_training, output
            )

            independent = []
            for preference in manifest["preference_grid"]:
                suffix = _preference_suffix(preference)
                fixed_agent, fixed_training = _train_vector_method(
                    baseline_agent=paper_target["agent"],
                    environment=environment(target_seed, objective, 0.0),
                    output=output,
                    checkpoint_dir=checkpoint_dir,
                    seed=seed,
                    method=f"independent_{suffix}",
                    training_seed=_derive_research_seed(seed, "independent_weighted"),
                    anchors=anchors,
                    manifest=manifest,
                    episodes=args.episodes,
                    checkpoint_every=args.checkpoint_every,
                    fixed_preference=preference,
                )
                independent.append({
                    "preference": preference,
                    **_vector_training_record(fixed_agent, fixed_training, output),
                })
            methods["independent_weighted_policies"] = {
                "policies": independent,
                "reward_scaled_historical_comparator": False,
            }

        validation_candidates = []
        for coefficient in manifest["gate_selection_grid"]["uncertainty_coefficients"]:
            for margin in manifest["gate_selection_grid"]["margins"]:
                proposed.uncertainty_coefficient = float(coefficient)
                proposed.gate_margin = float(margin)
                raw = evaluate_preference_grid(
                    proposed,
                    environment_factory=lambda scene_seed: environment(
                        scene_seed, objective, 0.0, "scipy"
                    ),
                    scene_seeds=manifest["validation_scene_seeds"],
                    preferences=manifest["preference_grid"],
                    use_gate=True,
                )
                metrics = _preference_grid_metrics(
                    raw,
                    anchors,
                    manifest["hypervolume_reference_normalized"],
                )
                validation_candidates.append({
                    "research_seed": int(seed),
                    "uncertainty_coefficient": float(coefficient),
                    "margin": float(margin),
                    **metrics,
                    "raw": raw,
                })
        gate_validation_rows.extend(validation_candidates)
        vector_records.append({"seed": int(seed), "methods": methods})
        atomic_write_json(progress_path, {
            "stage": "vector_training_and_gate_validation",
            "completed_scalar_seeds": scalar_records,
            "anchors_sha256": anchor_lock_digest,
            "completed_vector_seeds": vector_records,
            "gate_validation": gate_validation_rows,
        })

    aggregated_candidates = _aggregate_gate_candidates(gate_validation_rows)
    selected_gate = select_gate_configuration(aggregated_candidates)
    gate_validation_evidence = {
        "schema_version": 1,
        "kind": "ba_pcdtl_gate_validation_evidence",
        "selection_partition": "validation_only",
        "validation_scene_seeds": manifest["validation_scene_seeds"],
        "research_seeds": manifest["research_seeds"],
        "anchors_sha256": anchor_lock_digest,
        "rows": gate_validation_rows,
    }
    gate_validation_path = write_immutable_json(
        output / "gate_validation_evidence.json", gate_validation_evidence
    )
    gate_validation_digest = _file_sha256(gate_validation_path)
    gate_lock = {
        "schema_version": 1,
        "selection_partition": "validation_only",
        "validation_scene_seeds": manifest["validation_scene_seeds"],
        "research_seeds": manifest["research_seeds"],
        "anchors_sha256": anchor_lock_digest,
        "validation_evidence_sha256": gate_validation_digest,
        "selected": selected_gate,
        "candidates": aggregated_candidates,
    }
    gate_lock_path = write_immutable_json(
        output / "gate_selection_lock.json", gate_lock
    )
    gate_lock_digest = _file_sha256(gate_lock_path)

    final_records = []
    scalar_by_seed = {row["seed"]: row for row in scalar_records}
    vector_by_seed = {row["seed"]: row for row in vector_records}
    for seed in manifest["research_seeds"]:
        record = baseline_by_seed[int(seed)]
        paper_source, paper_target = restore_paper_agents(seed)
        target_seed = int(record["target_seed"])
        scalar_training = scalar_by_seed[int(seed)]
        for method_record in scalar_training["methods"].values():
            _require_checkpoint_hash(
                output / method_record["checkpoint"],
                method_record["checkpoint_sha256"],
            )
        _require_checkpoint_hash(
            output / scalar_training["joint_source_checkpoint"],
            scalar_training["joint_source_checkpoint_sha256"],
        )
        joint_source = train_ddpg(
            environment(record["source_seed"], objective, cfg.rho_nmse),
            args.episodes,
            f"joint-source-seed-{seed}",
            checkpoint_path=output / scalar_training["joint_source_checkpoint"],
            resume=True,
            checkpoint_every=args.checkpoint_every,
            training_seed=derive_training_seed(seed, "source"),
        )
        scalar_agents = {
            "same_objective_dtl": train_dtl(
                joint_source["agent"],
                environment(target_seed, objective, 0.0),
                args.episodes,
                f"same-objective-dtl-seed-{seed}",
                checkpoint_path=output / scalar_training["methods"]["same_objective_dtl"]["checkpoint"],
                resume=True,
                checkpoint_every=args.checkpoint_every,
                training_seed=derive_training_seed(seed, "target"),
            )["agent"],
            "cross_objective_dtl_diagnostic": train_dtl(
                paper_source["agent"],
                environment(target_seed, objective, 0.0),
                args.episodes,
                f"cross-objective-dtl-seed-{seed}",
                checkpoint_path=output / scalar_training["methods"]["cross_objective_dtl_diagnostic"]["checkpoint"],
                resume=True,
                checkpoint_every=args.checkpoint_every,
                training_seed=derive_training_seed(seed, "target"),
            )["agent"],
            "ddpg_scratch": train_ddpg(
                environment(target_seed, objective, 0.0),
                args.episodes,
                f"joint-ddpg-scratch-seed-{seed}",
                checkpoint_path=output / scalar_training["methods"]["ddpg_scratch"]["checkpoint"],
                resume=True,
                checkpoint_every=args.checkpoint_every,
                training_seed=derive_training_seed(seed, "target"),
            )["agent"],
            "paper_dtl_pcsi": paper_target["agent"],
        }
        test_methods = {
            name: _evaluate_scalar_agent(
                agent,
                lambda scene_seed: environment(scene_seed, objective, 0.0, "scipy"),
                manifest["test_scene_seeds"],
            )
            for name, agent in scalar_agents.items()
        }

        proposed_record = vector_by_seed[int(seed)]["methods"]["ba_pcdtl"]
        _require_checkpoint_hash(
            output / proposed_record["checkpoint"],
            proposed_record["checkpoint_sha256"],
        )
        proposed, _training = _train_vector_method(
            baseline_agent=paper_target["agent"],
            environment=environment(target_seed, objective, 0.0),
            output=output,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            method="ba_pcdtl",
            training_seed=_derive_research_seed(seed, "ba_pcdtl"),
            anchors=anchors,
            manifest=manifest,
            episodes=args.episodes,
            checkpoint_every=args.checkpoint_every,
        )
        proposed.uncertainty_coefficient = selected_gate["uncertainty_coefficient"]
        proposed.gate_margin = selected_gate["margin"]
        proposed_test = evaluate_preference_grid(
            proposed,
            lambda scene_seed: environment(scene_seed, objective, 0.0, "scipy"),
            manifest["test_scene_seeds"],
            manifest["preference_grid"],
            use_gate=True,
        )
        proposed_test["gate_configuration"] = {
            "uncertainty_coefficient": selected_gate["uncertainty_coefficient"],
            "margin": selected_gate["margin"],
        }
        test_methods["ba_pcdtl"] = proposed_test
        test_methods["ba_pcdtl_no_gate"] = evaluate_preference_grid(
            proposed,
            lambda scene_seed: environment(scene_seed, objective, 0.0, "scipy"),
            manifest["test_scene_seeds"],
            manifest["preference_grid"],
            use_gate=False,
        )

        if manifest["comparison_level"] == "full":
            for method, options in (
                ("unanchored_vector_td3", {
                    "residual_radius": 1.0, "transfer_trunks": False,
                    "anchor_enabled": False,
                }),
                ("anchored_random_trunks", {"transfer_trunks": False}),
            ):
                method_record = vector_by_seed[int(seed)]["methods"][method]
                _require_checkpoint_hash(
                    output / method_record["checkpoint"],
                    method_record["checkpoint_sha256"],
                )
                method_agent, _ = _train_vector_method(
                    baseline_agent=paper_target["agent"],
                    environment=environment(target_seed, objective, 0.0),
                    output=output,
                    checkpoint_dir=checkpoint_dir,
                    seed=seed,
                    method=method,
                    training_seed=_derive_research_seed(seed, method),
                    anchors=anchors,
                    manifest=manifest,
                    episodes=args.episodes,
                    checkpoint_every=args.checkpoint_every,
                    **options,
                )
                method_evaluation = evaluate_preference_grid(
                    method_agent,
                    lambda scene_seed: environment(scene_seed, objective, 0.0, "scipy"),
                    manifest["test_scene_seeds"],
                    manifest["preference_grid"],
                    use_gate=False,
                )
                test_methods[method] = method_evaluation
            independent_test = []
            independent_records = vector_by_seed[int(seed)]["methods"][
                "independent_weighted_policies"
            ]["policies"]
            for preference, independent_record in zip(
                manifest["preference_grid"], independent_records
            ):
                _require_checkpoint_hash(
                    output / independent_record["checkpoint"],
                    independent_record["checkpoint_sha256"],
                )
                method = f"independent_{_preference_suffix(preference)}"
                fixed_agent, _ = _train_vector_method(
                    baseline_agent=paper_target["agent"],
                    environment=environment(target_seed, objective, 0.0),
                    output=output,
                    checkpoint_dir=checkpoint_dir,
                    seed=seed,
                    method=method,
                    training_seed=_derive_research_seed(seed, "independent_weighted"),
                    anchors=anchors,
                    manifest=manifest,
                    episodes=args.episodes,
                    checkpoint_every=args.checkpoint_every,
                    fixed_preference=preference,
                )
                evaluation = evaluate_preference_grid(
                    fixed_agent,
                    lambda scene_seed: environment(scene_seed, objective, 0.0, "scipy"),
                    manifest["test_scene_seeds"],
                    [preference],
                    use_gate=False,
                )
                independent_test.append({
                    "preference": preference,
                    "use_gate": False,
                    "evaluation": evaluation["preferences"][0],
                })
            test_methods["independent_weighted_policies"] = {
                "scene_seeds": manifest["test_scene_seeds"],
                "policies": independent_test,
            }

        final_records.append({
            "seed": int(seed),
            "training": {
                "scalar": scalar_training,
                "vector": vector_by_seed[int(seed)],
            },
            "test": test_methods,
        })
        atomic_write_json(progress_path, {
            "stage": "locked_test_evaluation",
            "gate_selection_lock_sha256": gate_lock_digest,
            "completed_test_seeds": final_records,
        })

    summary = {
        "schema_version": 1,
        "kind": "generated_result",
        "protocol": "ba_pcdtl_research_extension",
        "algorithm_id": ALGORITHM_ID,
        "comparison_level": manifest["comparison_level"],
        "profile_digest": manifest["profile_digest"],
        "physical_configuration_fingerprint": manifest[
            "physical_configuration_fingerprint"
        ],
        "source_tree_digest": manifest["source_tree_digest"],
        "training_seed_protocol": TRAINING_SEED_PROTOCOL,
        "power_hypothesis_id": manifest["power_hypothesis_id"],
        "baseline_payload_sha256": manifest["baseline_payload_sha256"],
        "baseline_validation_sha256": manifest["baseline_validation_sha256"],
        "baseline_checkpoint_authority": manifest[
            "baseline_checkpoint_authority"
        ],
        "objective": manifest["objective"],
        "research_seeds": manifest["research_seeds"],
        "validation_scene_seeds": manifest["validation_scene_seeds"],
        "test_scene_seeds": manifest["test_scene_seeds"],
        "episodes": manifest["episodes"],
        "evaluation_episodes": manifest["evaluation_episodes"],
        "preference_grid": manifest["preference_grid"],
        "hypervolume_reference_normalized": manifest[
            "hypervolume_reference_normalized"
        ],
        "anchors": anchors.state_dict(),
        "anchors_lock_sha256": anchor_lock_digest,
        "gate_selection": gate_lock,
        "gate_selection_lock_sha256": gate_lock_digest,
        "gate_validation_evidence_sha256": gate_validation_digest,
        "research_seed_records": final_records,
    }
    apply_ambiguity_manifest_fields(summary, cfg)
    atomic_write_json(output / "pareto_extension_summary.json", summary)
    completed_manifest = dict(manifest)
    completed_manifest["status"] = "complete"
    completed_manifest["summary"] = "pareto_extension_summary.json"
    completed_manifest["summary_payload_sha256"] = result_payload_digest(summary)
    completed_manifest["gate_selection_lock_sha256"] = gate_lock_digest
    atomic_write_json(output / "run_manifest.json", completed_manifest)
    return 0


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_checkpoint_hash(path, expected: str) -> None:
    actual = _file_sha256(path)
    if actual != str(expected):
        raise ValueError(
            f"Checkpoint payload hash mismatch for {Path(path).name}: "
            f"expected {expected}, got {actual}."
        )


def _derive_research_seed(seed, role: str) -> int:
    material = f"ba_pcdtl_seed_v1:{int(seed)}:{role}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _preference_suffix(preference) -> str:
    return f"c{float(preference[0]):.3f}_s{float(preference[1]):.3f}".replace(".", "p")


def _ba_identity(
    manifest,
    seed,
    method,
    training_seed,
    environment_seed,
    baseline_digest,
    fixed_preference,
    residual_radius,
    transfer_trunks,
    anchor_enabled,
):
    return {
        "run_id": f"{method}-seed-{int(seed)}",
        "algorithm": ALGORITHM_ID,
        "method": str(method),
        "source_tree_digest": manifest["source_tree_digest"],
        "profile_digest": manifest["profile_digest"],
        "physical_configuration_fingerprint": manifest[
            "physical_configuration_fingerprint"
        ],
        "objective": manifest["objective"],
        "requested_episode_target": int(manifest["episodes"]),
        "environment_seed": int(environment_seed),
        "training_seed": int(training_seed),
        "baseline_policy_digest": str(baseline_digest),
        "preference_grid": manifest["preference_grid"],
        "fixed_preference": fixed_preference,
        "residual_radius": float(residual_radius),
        "ensemble_size": manifest["ensemble_size"],
        "residual_scales": manifest["residual_scales"],
        "transfer_trunks": bool(transfer_trunks),
        "anchor_enabled": bool(anchor_enabled),
    }


def _train_vector_method(
    *,
    baseline_agent,
    environment,
    output,
    checkpoint_dir,
    seed,
    method,
    training_seed,
    anchors,
    manifest,
    episodes,
    checkpoint_every,
    residual_radius=None,
    transfer_trunks=True,
    anchor_enabled=True,
    fixed_preference=None,
):
    from pareto_agent import BAPCDTLAgent
    from pareto_training import SobolPreferenceStream, train_ba_pcdtl
    from simulation import set_reproducibility

    set_reproducibility(training_seed)
    radius = manifest["residual_radius"] if residual_radius is None else residual_radius
    agent = BAPCDTLAgent(
        baseline_agent,
        environment.state_dim,
        environment.action_dim,
        anchors=anchors,
        residual_radius=radius,
        ensemble_size=manifest["ensemble_size"],
        residual_scales=manifest["residual_scales"],
        transfer_trunks=transfer_trunks,
        anchor_enabled=anchor_enabled,
    )
    tag = f"{method}-seed-{int(seed)}"
    checkpoint = Path(checkpoint_dir) / f"{tag}.pt"
    identity = _ba_identity(
        manifest,
        seed,
        method,
        training_seed,
        environment.seed,
        agent.baseline_policy_digest,
        fixed_preference,
        radius,
        transfer_trunks,
        anchor_enabled,
    )
    trained = train_ba_pcdtl(
        agent,
        environment,
        episodes,
        identity,
        SobolPreferenceStream(_derive_research_seed(training_seed, "preferences")),
        checkpoint,
        resume=True,
        checkpoint_every=checkpoint_every,
        fixed_preference=fixed_preference,
    )
    training_record = {
        "checkpoint": str(checkpoint.relative_to(output)),
        "training_seed": int(training_seed),
        "identity": identity,
        "training": {
            key: value
            for key, value in trained.items()
            if key not in {"agent"}
        },
    }
    return agent, training_record


def _vector_training_record(agent, training, output) -> dict:
    checkpoint = Path(output) / training["checkpoint"]
    return {
        "checkpoint": training["checkpoint"],
        "checkpoint_sha256": _file_sha256(checkpoint),
        "baseline_policy_digest": agent.baseline_policy_digest,
        "identity": training["identity"],
        "training": training["training"],
    }


def _preference_grid_metrics(raw, anchors, reference) -> dict:
    points = []
    feasibility = []
    fallback = []
    balanced_joint = None
    for record in raw["preferences"]:
        point = [
            float(np.mean(record["communication"])),
            float(np.mean(record["sensing"])),
        ]
        points.append(point)
        feasibility.extend(record["feasibility"])
        fallback.extend(record["baseline_fallback_rate"])
        if np.allclose(record["preference"], [0.5, 0.5], atol=1e-8, rtol=0.0):
            balanced_joint = float(np.mean(record["joint"]))
    if balanced_joint is None:
        raise ValueError("Evaluation is missing the balanced PNG preference.")
    normalized = anchors.normalize(points)
    return {
        "feasibility": float(np.min(feasibility)),
        "joint": balanced_joint,
        "hypervolume": hypervolume_2d(normalized, reference),
        "fallback_rate": float(np.mean(fallback)),
    }


def _aggregate_gate_candidates(rows) -> list[dict]:
    groups = {}
    for row in rows:
        key = (float(row["uncertainty_coefficient"]), float(row["margin"]))
        groups.setdefault(key, []).append(row)
    result = []
    for (coefficient, margin), group in sorted(groups.items()):
        result.append({
            "uncertainty_coefficient": coefficient,
            "margin": margin,
            "feasibility": float(min(row["feasibility"] for row in group)),
            "joint": float(np.mean([row["joint"] for row in group])),
            "hypervolume": float(np.mean([row["hypervolume"] for row in group])),
            "fallback_rate": float(np.mean([row["fallback_rate"] for row in group])),
            "research_seed_values": [
                {
                    "seed": int(row["research_seed"]),
                    "feasibility": row["feasibility"],
                    "joint": row["joint"],
                    "hypervolume": row["hypervolume"],
                    "fallback_rate": row["fallback_rate"],
                }
                for row in group
            ],
        })
    return result


def _evaluate_scalar_agent(agent, environment_factory, scene_seeds) -> dict:
    metrics = {
        "scene_seeds": [int(seed) for seed in scene_seeds],
        "communication": [],
        "sensing": [],
        "joint": [],
        "objective": [],
        "minimum_ue_rate": [],
        "minimum_target_rate": [],
        "feasibility": [],
    }
    for seed in metrics["scene_seeds"]:
        environment = environment_factory(seed)
        state = environment.reset()
        per_step = {key: [] for key in (
            "communication", "sensing", "minimum_ue_rate",
            "minimum_target_rate", "feasibility",
        )}
        done = False
        while not done:
            action = agent.select_action(state, explore=False)
            state, _reward, done, info = environment.step(action)
            per_step["communication"].append(float(info["communication_sum_rate"]))
            per_step["sensing"].append(float(info["sensing_sum_rate"]))
            per_step["minimum_ue_rate"].append(float(np.min(info["rate_u"])))
            per_step["minimum_target_rate"].append(float(np.min(info["rate_k"])))
            per_step["feasibility"].append(
                float(bool(info["physical_constraints_feasible"]))
            )
        communication = float(np.mean(per_step["communication"]))
        sensing = float(np.mean(per_step["sensing"]))
        metrics["communication"].append(communication)
        metrics["sensing"].append(sensing)
        metrics["joint"].append(communication + sensing)
        metrics["objective"].append(communication + sensing)
        metrics["minimum_ue_rate"].append(float(np.min(per_step["minimum_ue_rate"])))
        metrics["minimum_target_rate"].append(
            float(np.min(per_step["minimum_target_rate"]))
        )
        metrics["feasibility"].append(float(np.mean(per_step["feasibility"])))
    return metrics


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
