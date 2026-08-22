"""Shared run-manifest, seed, evaluation, and atomic JSON utilities."""

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np


RUN_MANIFEST_SCHEMA_VERSION = 1
TRAINING_SEED_PROTOCOL = "sha256_v1_common_target_seed_assignment"
EVALUATION_SEED_PROTOCOL = "sha256_v1_global_paired_scene_assignment"
REPLAY_ACTION_SEMANTICS = "executed_physical_action_v1"
IMMUTABLE_MANIFEST_FIELDS = (
    "protocol", "profile_digest", "physical_configuration_fingerprint",
    "objective", "seeds", "dtl_transfer_mode", "power_hypothesis_id",
    "source_tree_digest", "episodes", "evaluation_episodes",
    "baseline_payload_sha256", "baseline_validation_sha256",
    "objective_reference_protocol", "objective_reference_window",
    "training_seed_protocol", "evaluation_seed_protocol",
    "noise_hypothesis_id",
    "dtl_transfer_schedule_id", "wfb_hypothesis_id",
    "ddpg_comparator_csi_hypothesis_id",
    "los_probability_hypothesis_id", "channel_coherence_hypothesis_id",
    "csi_observation_hypothesis_id",
    "rate_model_hypothesis_id",
    "cross_function_interference_hypothesis_id",
    "algorithm_id", "research_seeds", "validation_scene_seeds",
    "test_scene_seeds", "preference_grid", "residual_radius",
    "residual_scales", "ensemble_size", "gate_selection_grid",
    "comparison_level",
    "hypervolume_reference_rule",
    "hypervolume_reference_normalized",
    "baseline_checkpoint_authority",
)

PROJECTION_EVIDENCE_KEYS = (
    "action_projection_rate",
    "mean_projection_fraction",
    "deterministic_recovery_rate",
    "allocation_fallback_rate",
)
PROJECTION_ATTRIBUTION_THRESHOLDS = {
    "maximum_action_projection_rate": 0.10,
    "maximum_deterministic_recovery_rate": 0.01,
    "minimum_mean_projection_fraction": 0.95,
}


def projection_attribution_passes(record) -> bool:
    """Return whether evaluated rewards remain attributable to actor actions."""
    try:
        projected = np.asarray(record["action_projection_rate"], dtype=float)
        recovery = np.asarray(
            record["deterministic_recovery_rate"], dtype=float
        )
        retained = np.asarray(record["mean_projection_fraction"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        projected.size > 0
        and projected.shape == recovery.shape == retained.shape
        and np.all(np.isfinite(projected))
        and np.all(np.isfinite(recovery))
        and np.all(np.isfinite(retained))
        and np.max(projected)
        <= PROJECTION_ATTRIBUTION_THRESHOLDS[
            "maximum_action_projection_rate"
        ] + 1e-12
        and np.max(recovery)
        <= PROJECTION_ATTRIBUTION_THRESHOLDS[
            "maximum_deterministic_recovery_rate"
        ] + 1e-12
        and np.min(retained)
        >= PROJECTION_ATTRIBUTION_THRESHOLDS[
            "minimum_mean_projection_fraction"
        ] - 1e-12
    )


def derive_training_seed(base_seed, role):
    """Derive a stable library RNG seed without Python's randomized hash."""
    if role not in {"source", "target"}:
        raise ValueError("training-seed role must be 'source' or 'target'.")
    base_seed = int(base_seed)
    if base_seed < 0:
        raise ValueError("base training seed must be nonnegative.")
    material = f"{TRAINING_SEED_PROTOCOL}:{base_seed}:{role}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def derive_evaluation_scene_seed(research_seed, episode_index):
    """Derive one stable evaluation scene from a research-seed/episode pair."""
    research_seed = int(research_seed)
    episode_index = int(episode_index)
    if research_seed < 0 or episode_index < 0:
        raise ValueError("Evaluation seed inputs must be nonnegative.")
    material = (
        f"{EVALUATION_SEED_PROTOCOL}:{research_seed}:{episode_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def evaluation_scene_seed_grid(research_seeds, evaluation_episodes):
    """Return and validate the globally distinct paired evaluation schedule."""
    seeds = tuple(int(seed) for seed in research_seeds)
    count = int(evaluation_episodes)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Research seeds must be unique and nonnegative.")
    if count <= 0:
        raise ValueError("Evaluation episode count must be positive.")
    grid = {
        seed: [derive_evaluation_scene_seed(seed, index) for index in range(count)]
        for seed in seeds
    }
    flattened = [scene for row in grid.values() for scene in row]
    forbidden = set(seeds) | {seed + 100_000 for seed in seeds}
    if len(set(flattened)) != len(flattened) or set(flattened) & forbidden:
        raise RuntimeError("Evaluation scene seed collision detected.")
    return grid


def parse_seeds(value):
    if isinstance(value, str):
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        seeds = tuple(int(item) for item in value)
    if len(seeds) < 5:
        raise ValueError("Research acceptance requires at least five seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Research seeds must be unique.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Research seeds must be nonnegative.")
    return seeds


def final_window_mean(values, window=100) -> float:
    """Return the mean of the declared tail of a training metric history."""
    array = np.asarray(values, dtype=float)
    window = int(window)
    if window <= 0:
        raise ValueError("training final window must be positive.")
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("training history must be a nonempty finite vector.")
    return float(np.mean(array[-min(window, array.size):]))


def physical_configuration_fingerprint(profile_digest, cfg) -> str:
    payload = {
        "profile_digest": profile_digest,
        "M": cfg.M, "N": cfg.N, "U": cfg.U, "K": cfg.K,
        "L": cfg.L, "La": cfg.La, "Lp": cfg.Lp,
        "Lax": cfg.Lax, "Lay": cfg.Lay,
        "Lpx": cfg.Lpx, "Lpy": cfg.Lpy,
        "Lx": cfg.Lx, "Ly": cfg.Ly,
        "hris_partition_mode": cfg.hris_partition_mode,
        "csi_state_representation_id": cfg.csi_state_representation_id,
        "transition_timing_id": cfg.transition_timing_id,
        "direct_channel_model_id": cfg.direct_channel_model_id,
        "capon_covariance_model_id": cfg.capon_covariance_model_id,
        "active_power_waveform_id": cfg.active_power_waveform_id,
        "power_hypothesis_id": cfg.power_hypothesis_id,
        "noise_hypothesis_id": cfg.noise_hypothesis_id,
        "ddpg_comparator_csi_hypothesis_id":
            cfg.ddpg_comparator_csi_hypothesis_id,
        "los_probability_hypothesis_id":
            cfg.los_probability_hypothesis_id,
        "channel_coherence_hypothesis_id":
            cfg.channel_coherence_hypothesis_id,
        "csi_observation_hypothesis_id":
            cfg.csi_observation_hypothesis_id,
        "rate_model_hypothesis_id": cfg.rate_model_hypothesis_id,
        "cross_function_interference_hypothesis_id":
            cfg.cross_function_interference_hypothesis_id,
        "PT_dB": cfg.PT_dB, "PT_W": cfg.PT,
        "PT_display_unit": cfg.PT_display_unit,
        "PT_reference": cfg.PT_reference,
        "sigma2_u": cfg.sigma2_u,
        "sigma2_k": cfg.sigma2_k,
        "sigma2_a": cfg.sigma2_a,
        "alpha_max": cfg.alpha_max,
        "Psi_max": cfg.Psi_max, "P_a_max": cfg.P_a_max,
        "fc": cfg.fc, "wavelength": cfg.wavelength,
        "d_spacing": cfg.d_spacing, "omega_0": cfg.omega_0,
        "rho_br": cfg.rho_br, "rho_rt": cfg.rho_rt,
        "a_env": cfg.a_env, "b_env": cfg.b_env,
        "zeta_los": cfg.zeta_los, "zeta_nlos": cfg.zeta_nlos,
        "rho_nmse": cfg.rho_nmse,
        "x_bounds": [cfg.x_min, cfg.x_max],
        "y_bounds": [cfg.y_min, cfg.y_max],
        "z_UAV": cfg.z_UAV,
        "BS_pos": np.asarray(cfg.BS_pos, dtype=float).tolist(),
        "ue_x_bounds": [cfg.ue_x_min, cfg.ue_x_max],
        "ue_y_bounds": [cfg.ue_y_min, cfg.ue_y_max],
        "target_region_hypothesis_id": cfg.target_region_hypothesis_id,
        "target_x_bounds": [cfg.target_x_min, cfg.target_x_max],
        "target_y_bounds": [cfg.target_y_min, cfg.target_y_max],
        "uav_start_hypothesis_id": cfg.uav_start_hypothesis_id,
        "uav_start_xy": [cfg.uav_start_x, cfg.uav_start_y],
        "target_speed_mps": cfg.target_speed_mps,
        "v_max": cfg.v_max, "delta_T": cfg.delta_T,
        "N_slots": cfg.N_slots,
        "policy_allocation_mode": cfg.policy_allocation_mode,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_run_manifest(protocol, profile_digest, physics_fingerprint,
                      objective: Mapping, seeds, *, minimum_seed_count=5):
    minimum_seed_count = int(minimum_seed_count)
    if minimum_seed_count <= 0:
        raise ValueError("minimum_seed_count must be positive.")
    parsed_seeds = tuple(int(seed) for seed in seeds)
    if (
        len(parsed_seeds) < minimum_seed_count
        or len(set(parsed_seeds)) != len(parsed_seeds)
        or any(seed < 0 for seed in parsed_seeds)
    ):
        raise ValueError(
            f"Run manifest requires at least {minimum_seed_count} unique "
            "nonnegative seeds."
        )
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "kind": "run_manifest",
        "status": "planned",
        "protocol": str(protocol),
        "profile_digest": str(profile_digest),
        "physical_configuration_fingerprint": str(physics_fingerprint),
        "objective": dict(objective),
        "seeds": list(parsed_seeds),
    }


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_training_checkpoint_identity(run_directory, checkpoint, identity):
    """Bind one completed checkpoint to its immutable training identity."""
    run_directory = Path(run_directory).resolve()
    checkpoint = Path(checkpoint).resolve()
    checkpoints_directory = (run_directory / "checkpoints").resolve()
    if (
        checkpoint.parent != checkpoints_directory
        or checkpoint.suffix != ".pt"
        or not checkpoint.is_file()
    ):
        raise ValueError(
            "Training checkpoint identity requires a .pt file directly inside "
            "the run checkpoints directory."
        )
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("Training checkpoint identity must be a nonempty mapping.")
    checkpoint_relative = checkpoint.relative_to(run_directory)
    checkpoint_digest = _sha256_file(checkpoint)
    identity_path = checkpoint.with_suffix(checkpoint.suffix + ".identity.json")
    atomic_write_json(identity_path, {
        "schema_version": 1,
        "kind": "training_checkpoint_identity",
        "checkpoint": str(checkpoint_relative),
        "checkpoint_sha256": checkpoint_digest,
        "identity": dict(identity),
    })
    return {
        "checkpoint": str(checkpoint_relative),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_identity": str(identity_path.relative_to(run_directory)),
        "checkpoint_identity_sha256": _sha256_file(identity_path),
        "identity": dict(identity),
    }


def write_immutable_json(path, payload):
    """Create a JSON sidecar once or verify its exact structured content."""
    path = Path(path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != payload:
            raise ValueError(f"Immutable JSON differs from existing {path.name}")
        return path
    return atomic_write_json(path, payload)


def ensure_run_directory(output_dir, expected_manifest):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        mismatches = [
            field for field in IMMUTABLE_MANIFEST_FIELDS
            if existing.get(field) != expected_manifest.get(field)
        ]
        if mismatches:
            raise ValueError(
                "Run manifest mismatch for existing output directory: "
                + ", ".join(mismatches)
            )
        if existing.get("status") == "complete":
            raise ValueError(
                "A completed run directory is immutable; choose a new output path."
            )
    else:
        existing_entries = list(output_dir.iterdir())
        if existing_entries:
            raise ValueError(
                "Refusing to use a nonempty directory without a matching run manifest."
            )
        atomic_write_json(manifest_path, expected_manifest)
    return output_dir


def resolve_required_checkpoint(run_directory, relative_path) -> Path:
    """Resolve an existing checkpoint without allowing escape from its run."""
    run_directory = Path(run_directory).resolve()
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Checkpoint source requires a completed run manifest.")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != "run_manifest"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Checkpoint source requires a completed run manifest.")
    checkpoints_directory = (run_directory / "checkpoints").resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("Checkpoint must be inside the run checkpoints directory.")
    candidate = (run_directory / candidate).resolve()
    if candidate.parent != checkpoints_directory or candidate.suffix != ".pt":
        raise ValueError("Checkpoint must be inside the run checkpoints directory.")
    if not candidate.is_file():
        raise FileNotFoundError(f"Required baseline checkpoint not found: {candidate}")
    return candidate


def evaluation_metrics(agent, environment_factory, evaluation_episodes):
    """Evaluate a frozen policy and return raw per-episode physical metrics."""
    communication, sensing, objective_values, feasibility = [], [], [], []
    minimum_ue_rates, minimum_target_rates = [], []
    projection_evidence = {key: [] for key in PROJECTION_EVIDENCE_KEYS}
    for episode_index in range(int(evaluation_episodes)):
        environment = environment_factory(episode_index)
        state = environment.reset()
        episode_comm = episode_sensing = episode_objective = 0.0
        episode_feasible = 0
        episode_minimum_ue = float("inf")
        episode_minimum_target = float("inf")
        episode_projection = {key: 0.0 for key in PROJECTION_EVIDENCE_KEYS}
        done = False
        steps = 0
        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, done, info = environment.step(action)
            episode_comm += float(info["communication_sum_rate"])
            episode_sensing += float(info["sensing_sum_rate"])
            episode_objective += float(reward)
            episode_feasible += int(info["physical_constraints_feasible"])
            episode_minimum_ue = min(
                episode_minimum_ue, float(np.min(info["rate_u"]))
            )
            episode_minimum_target = min(
                episode_minimum_target, float(np.min(info["rate_k"]))
            )
            for key, value in step_projection_evidence(info).items():
                episode_projection[key] += value
            steps += 1
        communication.append(episode_comm / steps)
        sensing.append(episode_sensing / steps)
        objective_values.append(episode_objective / steps)
        feasibility.append(episode_feasible / steps)
        minimum_ue_rates.append(episode_minimum_ue)
        minimum_target_rates.append(episode_minimum_target)
        for key in PROJECTION_EVIDENCE_KEYS:
            projection_evidence[key].append(episode_projection[key] / steps)
    result = {
        "communication": communication,
        "sensing": sensing,
        "objective": objective_values,
        "feasibility": feasibility,
        "minimum_ue_rates": minimum_ue_rates,
        "minimum_target_rates": minimum_target_rates,
        "communication_mean": float(np.mean(communication)),
        "sensing_mean": float(np.mean(sensing)),
        "objective_mean": float(np.mean(objective_values)),
        "feasibility_rate": float(np.mean(feasibility)),
        "minimum_ue_rate": float(np.min(minimum_ue_rates)),
        "minimum_target_rate": float(np.min(minimum_target_rates)),
    }
    result.update(projection_evidence)
    result.update({
        f"{key}_mean": float(np.mean(values))
        for key, values in projection_evidence.items()
    })
    return result


def step_projection_evidence(info: Mapping) -> dict[str, float]:
    """Validate and normalize environment repair diagnostics for one step."""
    required = (
        "action_was_projected",
        "action_projection_fraction",
        "projection_source",
        "allocation_fallback_used",
    )
    missing = [key for key in required if key not in info]
    if missing:
        raise ValueError(
            f"Step diagnostics are missing projection evidence: {missing}."
        )
    boolean_fields = ("action_was_projected", "allocation_fallback_used")
    malformed_booleans = [
        key for key in boolean_fields
        if not isinstance(info[key], (bool, np.bool_))
    ]
    if malformed_booleans:
        raise ValueError(
            "Boolean projection diagnostics have invalid types: "
            f"{malformed_booleans}."
        )
    projected = bool(info["action_was_projected"])
    fraction = float(info["action_projection_fraction"])
    source = info["projection_source"]
    fallback = bool(info["allocation_fallback_used"])
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("Action projection fraction must lie in [0, 1].")
    if not isinstance(source, str) or not source:
        raise ValueError("Projection source must be a nonempty string.")
    expected_projected = fraction < 1.0 - 1e-12
    if projected != expected_projected:
        raise ValueError("Projection flag and projection fraction disagree.")
    deterministic = source == "deterministic_recovery"
    if deterministic and not projected:
        raise ValueError("Deterministic recovery must be reported as a projection.")
    return {
        "action_projection_rate": float(projected),
        "mean_projection_fraction": fraction,
        "deterministic_recovery_rate": float(deterministic),
        "allocation_fallback_rate": float(fallback),
    }


def serializable_training_result(result):
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    excluded = {"agent", "checkpoint_path", "checkpoint_identity"}
    return convert({
        key: value for key, value in result.items() if key not in excluded
    })
