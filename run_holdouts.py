"""Run Figure 6 calibration evidence and all quantitative paper holdouts."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment_protocol import TRAINING_SEED_PROTOCOL, derive_training_seed
from ambiguity_hypotheses import add_ambiguity_cli_arguments


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--baseline-validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated, at least five")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return add_ambiguity_cli_arguments(parser)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _research_seed_configuration(cfg, seed):
    saved = (cfg.source_scene_seed, cfg.target_scene_seed)
    try:
        cfg.source_scene_seed = int(seed)
        cfg.target_scene_seed = int(seed) + 100_000
        yield
    finally:
        cfg.source_scene_seed, cfg.target_scene_seed = saved


def _evaluation_seed_base(figure, seed):
    return int(figure) * 10_000_000 + int(seed) * 100


def _load_figure_payload(directory, figure):
    json_path = Path(directory) / f"fig{int(figure)}_data.json"
    npz_path = Path(directory) / f"fig{int(figure)}_data.npz"
    payload = _read_json(json_path)
    curve_keys = payload.get("npz_keys")
    if not isinstance(curve_keys, dict) or set(curve_keys) != set(
        payload.get("curves", {})
    ):
        raise ValueError(f"Figure {figure} has an invalid JSON/NPZ key manifest.")
    with np.load(npz_path, allow_pickle=False) as archive:
        expected_keys = {"x", *curve_keys.values()}
        if set(archive.files) != expected_keys:
            raise ValueError(f"Figure {figure} NPZ keys do not match its JSON.")
        if not np.array_equal(
            np.asarray(payload.get("x")), np.asarray(archive["x"])
        ):
            raise ValueError(f"Figure {figure} JSON/NPZ x arrays differ.")
        for label, key in curve_keys.items():
            if not np.array_equal(
                np.asarray(payload["curves"][label]), np.asarray(archive[key])
            ):
                raise ValueError(
                    f"Figure {figure} JSON/NPZ curve differs: {label}."
                )
    feasibility = payload.get("metadata", {}).get("physical_feasibility", {})
    verified = int(feasibility.get("verified_realizations", 0))
    failed = int(feasibility.get("failed_realizations", -1))
    derived_rate = (
        float((verified - failed) / verified) if verified > 0 else -1.0
    )
    if (
        float(feasibility.get("rate", -1.0)) != 1.0
        or not np.isclose(float(feasibility.get("rate", -1.0)), derived_rate)
        or verified <= 0
        or failed != 0
        or feasibility.get("verification")
        != "fail_closed_per_generated_realization"
    ):
        raise ValueError(
            f"Figure {figure} lacks fail-closed physical-feasibility evidence."
        )
    return {
        "x": payload["x"],
        "curves": payload["curves"],
        "feasibility_rate": float(feasibility["rate"]),
        "feasibility_verification": feasibility["verification"],
        "verified_realizations": verified,
        "failed_realizations": failed,
        "figure_json_sha256": _file_sha256(json_path),
        "figure_npz_sha256": _file_sha256(npz_path),
    }


def _load_progress(path, declared_seeds, fresh):
    def empty_record(seed):
        return {
            "seed": seed,
            "source_training_seed": derive_training_seed(seed, "source"),
            "target_training_seed": derive_training_seed(seed, "target"),
            "figures": {},
        }

    if fresh or not path.is_file():
        return {seed: empty_record(seed) for seed in declared_seeds}
    payload = _read_json(path)
    records = payload.get("seed_records")
    if not isinstance(records, list):
        raise ValueError("Holdout progress has no seed records")
    by_seed = {}
    for record in records:
        seed = record.get("seed")
        figures = record.get("figures")
        if seed not in declared_seeds or seed in by_seed or not isinstance(figures, dict):
            raise ValueError("Holdout progress does not match the run seeds")
        if (
            record.get("source_training_seed")
            != derive_training_seed(seed, "source")
            or record.get("target_training_seed")
            != derive_training_seed(seed, "target")
        ):
            raise ValueError("Holdout progress uses a different training-seed protocol")
        if not set(figures) <= {"5", "6", "7", "8", "9", "13", "14"}:
            raise ValueError("Holdout progress contains an undeclared figure")
        by_seed[seed] = record
    return {
        seed: by_seed.get(seed, empty_record(seed))
        for seed in declared_seeds
    }


def _run_figure(figure, simulation, registry, episodes, evaluations, seed):
    seed_base = _evaluation_seed_base(figure, seed)
    if figure == 5:
        simulation.fig5_ao_convergence(
            scene_seed=seed_base, ao_seed=seed_base + 10
        )
    elif figure == 6:
        simulation.fig6_rate_vs_power(
            training_registry=registry,
            n_ep=episodes,
            n_eval=evaluations,
            evaluation_seed_base=seed_base,
            scene_seed=seed_base + 1,
            ao_seed_base=seed_base + 10,
        )
    elif figure == 7:
        simulation.fig7_rate_vs_L(
            training_registry=registry, n_ep=episodes, n_eval=evaluations,
            evaluation_seed_base=seed_base,
        )
    elif figure == 8:
        simulation.fig8_rate_vs_scee(
            training_registry=registry, n_ep=episodes, n_eval=evaluations,
            evaluation_seed_base=seed_base,
        )
    elif figure == 9:
        simulation.fig9_active_budget(
            training_registry=registry, n_ep=episodes, n_eval=evaluations,
            evaluation_seed_base=seed_base,
        )
    elif figure == 13:
        simulation.fig13_rate_vs_delta(
            training_registry=registry, n_ep=episodes, n_eval=evaluations,
            evaluation_seed_base=seed_base,
        )
    elif figure == 14:
        simulation.fig14_rate_vs_delta_bar(
            training_registry=registry, n_ep=episodes, n_eval=evaluations,
            evaluation_seed_base=seed_base,
        )
    else:  # pragma: no cover - caller uses a closed constant set.
        raise ValueError(f"Unsupported paper figure: {figure}")


def run(args):
    from ambiguity_hypotheses import (
        apply_ambiguity_arguments,
        apply_ambiguity_manifest_fields,
        require_matching_ambiguity_contract,
    )
    from config import cfg
    from experiment_protocol import (
        atomic_write_json,
        ensure_run_directory,
        make_run_manifest,
        parse_seeds,
        physical_configuration_fingerprint,
    )
    from holdout_protocol import (
        CALIBRATION_FIGURES,
        FEASIBILITY_VERIFICATION,
        HOLDOUT_FIGURES,
        PUBLICATION_FIGURES,
        aggregate_seed_payloads,
    )
    from parameter_profiles import activate_profile, load_calibrated_profile
    from run_extension import require_accepted_baseline
    from source_integrity import source_tree_digest
    from validate_results import (
        require_canonical_baseline_validation,
        result_payload_digest,
    )

    seeds = parse_seeds(args.seeds)
    if args.episodes != cfg.E_episodes or args.episodes != 1000:
        raise ValueError("Paper holdout sweeps require Table IV's 1000 episodes")
    if args.eval_episodes <= 0 or args.checkpoint_every <= 0:
        raise ValueError("Evaluation episodes and checkpoint interval must be positive")

    profile = load_calibrated_profile(args.profile)
    activate_profile(profile)
    apply_ambiguity_arguments(args, cfg)
    baseline_path = Path(args.baseline_summary).resolve()
    baseline = _read_json(baseline_path)
    baseline_validation = _read_json(args.baseline_validation)
    require_accepted_baseline(baseline_validation, baseline)
    require_canonical_baseline_validation(
        baseline, baseline_validation, baseline_path.parent
    )
    if baseline.get("training_seed_protocol") != TRAINING_SEED_PROTOCOL:
        raise ValueError("Baseline does not use the required training-seed protocol")
    if tuple(baseline.get("seeds", [])) != seeds:
        raise ValueError("Holdout seeds must exactly match the accepted baseline")

    physics = physical_configuration_fingerprint(profile.digest, cfg)
    current_source_digest = source_tree_digest()
    if (
        baseline.get("profile_digest") != profile.digest
        or baseline.get("physical_configuration_fingerprint") != physics
        or baseline.get("source_tree_digest") != current_source_digest
        or baseline.get("power_hypothesis_id") != cfg.power_hypothesis_id
    ):
        raise ValueError("Holdouts do not match the accepted baseline provenance")
    require_matching_ambiguity_contract(baseline, cfg)

    baseline_digest = result_payload_digest(baseline)
    validation_digest = result_payload_digest(baseline_validation)
    objective = {"mode": "paper_baseline", "reported_reward": "sum_u_R_u"}
    manifest = make_run_manifest(
        "paper_calibration_and_holdouts", profile.digest, physics, objective, seeds
    )
    manifest.update({
        "power_hypothesis_id": cfg.power_hypothesis_id,
        "source_tree_digest": current_source_digest,
        "baseline_payload_sha256": baseline_digest,
        "baseline_validation_sha256": validation_digest,
        "episodes": args.episodes,
        "evaluation_episodes": args.eval_episodes,
        "training_seed_protocol": TRAINING_SEED_PROTOCOL,
    })
    apply_ambiguity_manifest_fields(manifest, cfg)
    output = ensure_run_directory(args.output, manifest)
    if args.dry_run:
        return 0

    import simulation

    progress_path = output / "holdout_progress.json"
    records = _load_progress(progress_path, seeds, args.fresh)
    original_output = simulation.OUT
    try:
        for seed in seeds:
            simulation.set_reproducibility(seed)
            record = records[seed]
            seed_directory = output / "seed_runs" / f"seed-{seed}"
            seed_directory.mkdir(parents=True, exist_ok=True)
            simulation.OUT = str(seed_directory)
            with _research_seed_configuration(cfg, seed):
                registry = simulation.TrainingRegistry(
                    resume=not args.fresh,
                    checkpoint_every=args.checkpoint_every,
                )
                for figure in PUBLICATION_FIGURES:
                    key = str(figure)
                    if key in record["figures"]:
                        current = _load_figure_payload(seed_directory, figure)
                        if current != record["figures"][key]:
                            raise ValueError(
                                "Stored holdout progress does not match raw "
                                f"Figure {figure} files for seed {seed}."
                            )
                        continue
                    _run_figure(
                        figure, simulation, registry, args.episodes,
                        args.eval_episodes, seed,
                    )
                    record["figures"][key] = _load_figure_payload(
                        seed_directory, figure
                    )
                    atomic_write_json(progress_path, {
                        "seed_records": [records[value] for value in seeds]
                    })
    finally:
        simulation.OUT = original_output

    seed_records = [records[seed] for seed in seeds]
    aggregated_figures = aggregate_seed_payloads(seed_records)
    feasibility_rate = min(
        float(payload["feasibility_rate"])
        for payload in aggregated_figures.values()
    )
    summary = {
        "schema_version": 1,
        "kind": "generated_result",
        "protocol": "paper_calibration_and_holdouts",
        "objective": objective,
        "profile_digest": profile.digest,
        "physical_configuration_fingerprint": physics,
        "source_tree_digest": current_source_digest,
        "power_hypothesis_id": cfg.power_hypothesis_id,
        "training_seed_protocol": TRAINING_SEED_PROTOCOL,
        "baseline_payload_sha256": baseline_digest,
        "baseline_validation_sha256": validation_digest,
        "calibration_figures": list(CALIBRATION_FIGURES),
        "holdout_figures": list(HOLDOUT_FIGURES),
        "seeds": list(seeds),
        "episodes": args.episodes,
        "evaluation_episodes": args.eval_episodes,
        "feasibility_rate": feasibility_rate,
        "feasibility_provenance": (
            "All AO iterates, physical baselines, and policy rollout steps use "
            "fail-closed full physical-feasibility checks; the reported rate "
            "is recomputed from every per-seed figure record."
        ),
        "feasibility_verification": FEASIBILITY_VERIFICATION,
        "seed_records": seed_records,
        "figures": aggregated_figures,
    }
    apply_ambiguity_manifest_fields(summary, cfg)
    atomic_write_json(output / "holdout_summary.json", summary)
    manifest["status"] = "complete"
    manifest["summary"] = "holdout_summary.json"
    manifest["summary_payload_sha256"] = result_payload_digest(summary)
    atomic_write_json(output / "run_manifest.json", manifest)
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
