"""Shared run-manifest, seed, evaluation, and atomic JSON utilities."""

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np


RUN_MANIFEST_SCHEMA_VERSION = 1
TRAINING_SEED_PROTOCOL = "sha256_v1_common_target_seed_assignment"
IMMUTABLE_MANIFEST_FIELDS = (
    "protocol", "profile_digest", "physical_configuration_fingerprint",
    "objective", "seeds", "dtl_transfer_mode", "power_hypothesis_id",
    "source_tree_digest", "episodes", "evaluation_episodes",
    "baseline_payload_sha256", "baseline_validation_sha256",
    "training_seed_protocol", "noise_hypothesis_id",
    "dtl_transfer_schedule_id", "wfb_hypothesis_id",
    "ddpg_comparator_csi_hypothesis_id",
    "algorithm_id", "research_seeds", "validation_scene_seeds",
    "test_scene_seeds", "preference_grid", "residual_radius",
    "residual_scales", "ensemble_size", "gate_selection_grid",
    "comparison_level",
    "hypervolume_reference_normalized",
    "baseline_checkpoint_authority",
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
        "power_hypothesis_id": cfg.power_hypothesis_id,
        "noise_hypothesis_id": cfg.noise_hypothesis_id,
        "ddpg_comparator_csi_hypothesis_id":
            cfg.ddpg_comparator_csi_hypothesis_id,
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
        "target_speed_mps": cfg.target_speed_mps,
        "v_max": cfg.v_max, "delta_T": cfg.delta_T,
        "N_slots": cfg.N_slots,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_run_manifest(protocol, profile_digest, physics_fingerprint,
                      objective: Mapping, seeds):
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "kind": "run_manifest",
        "status": "planned",
        "protocol": str(protocol),
        "profile_digest": str(profile_digest),
        "physical_configuration_fingerprint": str(physics_fingerprint),
        "objective": dict(objective),
        "seeds": list(parse_seeds(seeds)),
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
    for episode_index in range(int(evaluation_episodes)):
        environment = environment_factory(episode_index)
        state = environment.reset()
        episode_comm = episode_sensing = episode_objective = 0.0
        episode_feasible = 0
        episode_minimum_ue = float("inf")
        episode_minimum_target = float("inf")
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
            steps += 1
        communication.append(episode_comm / steps)
        sensing.append(episode_sensing / steps)
        objective_values.append(episode_objective / steps)
        feasibility.append(episode_feasible / steps)
        minimum_ue_rates.append(episode_minimum_ue)
        minimum_target_rates.append(episode_minimum_target)
    return {
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

    return convert({key: value for key, value in result.items() if key != "agent"})
