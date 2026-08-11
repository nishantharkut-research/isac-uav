"""Run predeclared ambiguity branches in isolated resumable directories."""

import argparse
import json
from pathlib import Path

from ambiguity_hypotheses import (
    PRIMARY_NOISE_HYPOTHESIS_ID,
    PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID,
    PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID,
    PRIMARY_WFB_HYPOTHESIS_ID,
    PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID,
    PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID,
    PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID,
    PRIMARY_CROSS_FUNCTION_INTERFERENCE_HYPOTHESIS_ID,
)
from ambiguity_protocol import build_sensitivity_plan, build_sensitivity_report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated, at least five")
    parser.add_argument(
        "--design", choices=("one_at_a_time", "full_factorial"),
        default="one_at_a_time",
    )
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--primary-baseline")
    parser.add_argument("--primary-validation")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_immutable_plan(path, payload):
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("Sensitivity execution plan is immutable")
        return path
    from experiment_protocol import atomic_write_json
    return atomic_write_json(path, payload)


def _require_compatible_primary(primary, profile_digest,
                                current_source_digest, seeds, episodes,
                                evaluation_episodes):
    """Fail before training when an external primary is not exactly paired."""
    expected = {
        "kind": "generated_result",
        "protocol": "paper_baseline",
        "profile_digest": str(profile_digest),
        "source_tree_digest": str(current_source_digest),
        "seeds": list(seeds),
        "episodes": int(episodes),
        "evaluation_episodes": int(evaluation_episodes),
        "noise_hypothesis_id": PRIMARY_NOISE_HYPOTHESIS_ID,
        "dtl_transfer_mode": PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID,
        "dtl_transfer_schedule_id":
            PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID,
        "wfb_hypothesis_id": PRIMARY_WFB_HYPOTHESIS_ID,
        "ddpg_comparator_csi_hypothesis_id":
            PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID,
        "los_probability_hypothesis_id":
            PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID,
        "channel_coherence_hypothesis_id":
            PRIMARY_CHANNEL_COHERENCE_HYPOTHESIS_ID,
        "cross_function_interference_hypothesis_id":
            PRIMARY_CROSS_FUNCTION_INTERFERENCE_HYPOTHESIS_ID,
    }
    mismatches = [
        field for field, value in expected.items()
        if primary.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            "External primary is incompatible with the sensitivity plan: "
            + ", ".join(mismatches)
        )
    return True


def run(args):
    from experiment_protocol import atomic_write_json, parse_seeds
    from run_baseline import build_parser as baseline_parser, run as run_baseline

    seeds = parse_seeds(args.seeds)
    if bool(args.primary_baseline) != bool(args.primary_validation):
        raise ValueError(
            "--primary-baseline and --primary-validation must be supplied together"
        )
    external_primary = None
    primary_authority = None
    if args.primary_baseline:
        from run_extension import require_accepted_baseline
        baseline_path = Path(args.primary_baseline).resolve()
        validation_path = Path(args.primary_validation).resolve()
        external_primary = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        require_accepted_baseline(validation, external_primary)
        from parameter_profiles import load_calibrated_profile
        from source_integrity import source_tree_digest
        profile = load_calibrated_profile(args.profile)
        _require_compatible_primary(
            external_primary,
            profile_digest=profile.digest,
            current_source_digest=source_tree_digest(),
            seeds=seeds,
            episodes=args.episodes,
            evaluation_episodes=args.eval_episodes,
        )
        from validate_results import result_payload_digest
        primary_authority = {
            "baseline": str(baseline_path),
            "validation": str(validation_path),
            "baseline_payload_sha256": result_payload_digest(external_primary),
            "validation_payload_sha256": result_payload_digest(validation),
        }
    plan = build_sensitivity_plan(args.design)
    execution_plan = {
        **plan,
        "profile": str(Path(args.profile).resolve()),
        "seeds": list(seeds),
        "episodes": int(args.episodes),
        "evaluation_episodes": int(args.eval_episodes),
        "checkpoint_every": int(args.checkpoint_every),
        "external_primary_authority": primary_authority,
    }
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_immutable_plan(
        output_root / "sensitivity_plan.json", execution_plan
    )

    summaries = {}
    if external_primary is not None:
        summaries["primary"] = external_primary
    for scenario in plan["scenarios"]:
        if scenario["id"] == "primary" and external_primary is not None:
            continue
        scenario_dir = output_root / scenario["id"]
        summary_path = scenario_dir / "baseline_summary.json"
        if summary_path.is_file() and not args.dry_run:
            summaries[scenario["id"]] = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            continue
        hypotheses = scenario["hypotheses"]
        baseline_args = baseline_parser().parse_args([
            "--profile", args.profile,
            "--output", str(scenario_dir),
            "--seeds", ",".join(str(seed) for seed in seeds),
            "--episodes", str(args.episodes),
            "--eval-episodes", str(args.eval_episodes),
            "--checkpoint-every", str(args.checkpoint_every),
            "--noise-hypothesis", hypotheses["noise"],
            "--transfer-update-hypothesis", hypotheses["transfer_update"],
            "--transfer-schedule-hypothesis", hypotheses["transfer_schedule"],
            "--wfb-hypothesis", hypotheses["wfb"],
            "--ddpg-comparator-csi-hypothesis",
            hypotheses["ddpg_comparator_csi"],
            "--los-probability-hypothesis", hypotheses["los_probability"],
            "--channel-coherence-hypothesis",
            hypotheses["channel_coherence"],
            "--cross-function-interference-hypothesis",
            hypotheses["cross_function_interference"],
            *(["--fresh"] if args.fresh else []),
            *(["--dry-run"] if args.dry_run else []),
        ])
        run_baseline(baseline_args)
        if not args.dry_run:
            summaries[scenario["id"]] = json.loads(
                summary_path.read_text(encoding="utf-8")
            )

    if not args.dry_run:
        report = build_sensitivity_report(plan, summaries)
        atomic_write_json(
            output_root / "ambiguity_sensitivity_report.json", report
        )
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
