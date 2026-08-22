"""Fail-closed scientific validation for BA-PCDTL research results."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from ambiguity_hypotheses import AMBIGUITY_IDENTITY_REGISTRIES
from checkpointing import (
    inspect_training_checkpoint,
    validate_checkpoint_authority,
)
from pareto_protocol import (
    ParetoAnchors,
    holm_adjust,
    hypervolume_2d,
    normalized_physical_zero_reference,
    paired_confidence_interval,
    validate_preferences,
)
from experiment_protocol import (
    PROJECTION_ATTRIBUTION_THRESHOLDS,
    PROJECTION_EVIDENCE_KEYS,
    REPLAY_ACTION_SEMANTICS,
    derive_training_seed,
    projection_attribution_passes,
)
from run_pareto_extension import (
    _derive_research_seed,
    _preference_suffix,
    select_gate_configuration,
    validate_seed_roles,
)
from validate_results import (
    require_canonical_baseline_validation,
    result_payload_digest,
)
from source_integrity import source_model_state_digest


def _check(passed, **details) -> dict:
    return {"pass": bool(passed), **details}


def _recompute_training_anchors(summary: Mapping) -> ParetoAnchors:
    """Rebuild locked anchors solely from serialized scalar training tails."""
    provenance = summary.get("anchors", {}).get("provenance", {})
    required_methods = [
        "same_objective_dtl",
        "cross_objective_dtl_diagnostic",
        "ddpg_scratch",
    ]
    if (
        provenance.get("partition") != "training_only"
        or provenance.get("estimator")
        != "per-method converged-tail quantiles"
        or provenance.get("source_methods") != required_methods
        or provenance.get("research_seeds") != summary.get("research_seeds")
    ):
        raise ValueError("Pareto anchor provenance is incomplete or inconsistent.")
    tail_fraction = float(provenance["tail_fraction"])
    tail_minimum = int(provenance["tail_minimum_episodes"])
    lower = float(provenance["lower_quantile"])
    upper = float(provenance["upper_quantile"])
    if not 0.0 < tail_fraction <= 1.0 or tail_minimum <= 0:
        raise ValueError("Pareto anchor tail definition is invalid.")

    recomputed = ParetoAnchors()
    records = summary.get("research_seed_records", [])
    if [record.get("seed") for record in records] != summary.get("research_seeds"):
        raise ValueError("Pareto anchor seed records are incomplete or reordered.")
    for record in records:
        methods = record.get("training", {}).get("scalar", {}).get("methods", {})
        for method in required_methods:
            training = methods.get(method, {}).get("training", {})
            communication = np.asarray(training.get("comm_rates"), dtype=np.float64)
            sensing = np.asarray(training.get("sensing_rates"), dtype=np.float64)
            if (
                communication.ndim != 1
                or sensing.shape != communication.shape
                or communication.size != int(summary.get("episodes", -1))
                or not np.all(np.isfinite(communication))
                or not np.all(np.isfinite(sensing))
            ):
                raise ValueError(f"{method} anchor training history is invalid.")
            values = np.column_stack([communication, sensing])
            tail_count = min(
                values.shape[0],
                max(tail_minimum, int(np.ceil(tail_fraction * values.shape[0]))),
            )
            recomputed.update_training(
                values[-tail_count:],
                lower_quantile=lower,
                upper_quantile=upper,
            )
    recomputed.freeze(provenance)
    return recomputed


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _contains_forbidden_envelope(value) -> bool:
    if isinstance(value, Mapping):
        return any(
            "envelope" in str(key).lower() or _contains_forbidden_envelope(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_envelope(item) for item in value)
    return False


def _finite_array(record, key, expected_length) -> np.ndarray:
    array = np.asarray(record.get(key), dtype=np.float64)
    if array.shape != (expected_length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{key} must be a finite raw array of length {expected_length}.")
    return array


def _validate_rate_record(
        record, expected_length, *, require_policy_evidence=False,
        scene_seeds=None, evaluation_episodes=None,
) -> dict[str, np.ndarray]:
    keys = (
        "communication", "sensing", "joint", "objective", "minimum_ue_rate",
        "minimum_target_rate", "feasibility", *PROJECTION_EVIDENCE_KEYS,
    )
    arrays = {key: _finite_array(record, key, expected_length) for key in keys}
    if require_policy_evidence:
        for key in ("mean_residual_scale", "baseline_fallback_rate"):
            arrays[key] = _finite_array(record, key, expected_length)
    if not np.allclose(
        arrays["joint"], arrays["communication"] + arrays["sensing"],
        rtol=0.0, atol=1e-8,
    ):
        raise ValueError("Raw joint values do not equal communication plus sensing.")
    if not np.allclose(arrays["objective"], arrays["joint"], rtol=0.0, atol=1e-8):
        raise ValueError("PNG objective values do not equal the raw joint rate.")
    for key in ("feasibility", *PROJECTION_EVIDENCE_KEYS):
        if np.any(arrays[key] < 0.0) or np.any(arrays[key] > 1.0):
            raise ValueError(f"{key} values must lie in [0, 1].")
    if np.any(
        arrays["deterministic_recovery_rate"]
        > arrays["action_projection_rate"] + 1e-12
    ):
        raise ValueError(
            "Deterministic-recovery rate cannot exceed action-projection rate."
        )
    if np.any(
        1.0 - arrays["mean_projection_fraction"]
        > arrays["action_projection_rate"] + 1e-12
    ):
        raise ValueError(
            "Projection fraction is inconsistent with action-projection rate."
        )
    for key in ("mean_residual_scale", "baseline_fallback_rate"):
        if key in arrays and (
            np.any(arrays[key] < 0.0) or np.any(arrays[key] > 1.0)
        ):
            raise ValueError(f"{key} values must lie in [0, 1].")
    if not projection_attribution_passes(record):
        raise ValueError("Actor-action attribution thresholds are not met.")
    if scene_seeds is not None or evaluation_episodes is not None:
        if scene_seeds is None or evaluation_episodes is None:
            raise ValueError("Episode evidence requires seeds and episode count.")
        scene_seeds = [int(seed) for seed in scene_seeds]
        evaluation_episodes = int(evaluation_episodes)
        evidence = record.get("episode_evidence")
        if (
            evaluation_episodes <= 0
            or not isinstance(evidence, list)
            or len(evidence) != len(scene_seeds)
        ):
            raise ValueError("Per-episode evaluation evidence is incomplete.")
        reconstructed = {key: [] for key in arrays}
        for seed, episode in zip(scene_seeds, evidence):
            if episode.get("scene_seed") != seed:
                raise ValueError("Per-episode scene identity changed.")
            episode_arrays = _validate_rate_record(
                episode,
                evaluation_episodes,
                require_policy_evidence=require_policy_evidence,
            )
            for key in arrays:
                reducer = (
                    np.min
                    if key in {"minimum_ue_rate", "minimum_target_rate"}
                    else np.mean
                )
                reconstructed[key].append(float(reducer(episode_arrays[key])))
        for key, expected in reconstructed.items():
            if not np.allclose(
                arrays[key], np.asarray(expected, dtype=np.float64),
                rtol=0.0, atol=1e-10,
            ):
                raise ValueError(
                    f"{key} does not reconstruct from per-episode evidence."
                )
    return arrays


def _grid_records(
        evaluation, preferences, scene_seeds, evaluation_episodes
) -> list[tuple[list, dict, dict]]:
    if evaluation.get("scene_seeds") != list(scene_seeds):
        raise ValueError("Evaluation scene seeds do not match the locked test partition.")
    if evaluation.get("evaluation_episodes") != int(evaluation_episodes):
        raise ValueError("Evaluation episode count does not match the manifest.")
    rows = evaluation.get("preferences")
    if not isinstance(rows, list) or len(rows) != len(preferences):
        raise ValueError("Evaluation does not contain the complete preference grid.")
    result = []
    for expected, row in zip(preferences, rows):
        actual = validate_preferences([row.get("preference")])[0].tolist()
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-8):
            raise ValueError("Evaluation preference order or identity changed.")
        result.append((
            expected,
            row,
            _validate_rate_record(
                row,
                len(scene_seeds),
                require_policy_evidence=True,
                scene_seeds=scene_seeds,
                evaluation_episodes=evaluation_episodes,
            ),
        ))
    return result


def _balanced_record(rows):
    for preference, row, arrays in rows:
        if np.allclose(preference, [0.5, 0.5], rtol=0.0, atol=1e-8):
            return row, arrays
    raise ValueError("The balanced PNG operating point is missing.")


def _interval_dict(interval) -> dict:
    payload = asdict(interval)
    for key, value in list(payload.items()):
        if isinstance(value, float) and not math.isfinite(value):
            payload[key] = None
    return payload


def _verify_checkpoint_record(root: Path, record: Mapping) -> bool:
    try:
        relative = Path(str(record.get("checkpoint", "")))
        if relative.is_absolute():
            return False
        checkpoints = (root / "checkpoints").resolve()
        path = (root / relative).resolve()
        expected = record.get("checkpoint_sha256")
        return (
            path.parent == checkpoints
            and path.suffix == ".pt"
            and path.is_file()
            and _is_sha256(expected)
            and _sha256_file(path) == expected
        )
    except (OSError, TypeError, ValueError):
        return False


def _validate_run_manifest_payload(summary, manifest) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "run_manifest"
        or manifest.get("status") != "complete"
        or manifest.get("summary") != "pareto_extension_summary.json"
        or manifest.get("summary_payload_sha256")
        != result_payload_digest(summary)
    ):
        raise ValueError("Completed run manifest authority fields mismatch.")
    shared_fields = (
        "protocol",
        "algorithm_id",
        "profile_digest",
        "physical_configuration_fingerprint",
        "source_tree_digest",
        "training_seed_protocol",
        "power_hypothesis_id",
        "baseline_payload_sha256",
        "baseline_validation_sha256",
        "baseline_checkpoint_authority",
        "objective",
        "research_seeds",
        "validation_scene_seeds",
        "test_scene_seeds",
        "episodes",
        "evaluation_episodes",
        "ensemble_size",
        "residual_radius",
        "residual_scales",
        "preference_grid",
        "comparison_level",
        "gate_selection_grid",
        "hypervolume_reference_rule",
        "hypervolume_reference_normalized",
        "gate_selection_lock_sha256",
        *AMBIGUITY_IDENTITY_REGISTRIES.keys(),
        "ambiguity_contract",
    )
    mismatches = [
        field for field in shared_fields
        if manifest.get(field) != summary.get(field)
    ]
    if manifest.get("seeds") != summary.get("research_seeds"):
        mismatches.append("seeds")
    if mismatches:
        raise ValueError(
            "Run manifest and summary contract mismatch: " + ", ".join(mismatches)
        )


def _verify_checkpoint_identity_envelope(
        root: Path, record: Mapping, identity: Mapping
) -> bool:
    try:
        checkpoint_relative = Path(str(record["checkpoint"]))
        identity_relative = Path(str(record["checkpoint_identity"]))
        expected_identity_relative = checkpoint_relative.with_suffix(
            checkpoint_relative.suffix + ".identity.json"
        )
        checkpoints = (root / "checkpoints").resolve()
        identity_path = (root / identity_relative).resolve()
        if (
            identity_relative != expected_identity_relative
            or identity_path.parent != checkpoints
            or not identity_path.is_file()
            or not _is_sha256(record.get("checkpoint_identity_sha256"))
            or _sha256_file(identity_path)
            != record["checkpoint_identity_sha256"]
        ):
            return False
        with identity_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return payload == {
            "schema_version": 1,
            "kind": "training_checkpoint_identity",
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "identity": dict(identity),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _verify_internal_checkpoint_authority(
        root: Path, record: Mapping, identity: Mapping,
) -> bool:
    try:
        checkpoint = (root / Path(str(record["checkpoint"]))).resolve()
        validate_checkpoint_authority(
            checkpoint,
            identity,
            record["training"],
        )
        return True
    except Exception:
        # Treat malformed or inconsistent torch payloads as invalid evidence.
        return False


def _verify_vector_checkpoint_identity(
        root: Path, summary: Mapping, record: Mapping, *, seed: int,
        method: str, fixed_preference=None,
) -> bool:
    identity = record.get("identity", {})
    if not isinstance(identity, Mapping):
        return False
    if method == "ba_pcdtl":
        seed_role = method
        residual_radius = summary.get("residual_radius")
        transfer_trunks, anchor_enabled = True, True
    elif method == "unanchored_vector_td3":
        seed_role = method
        residual_radius = 1.0
        transfer_trunks, anchor_enabled = False, False
    elif method == "anchored_random_trunks":
        seed_role = method
        residual_radius = summary.get("residual_radius")
        transfer_trunks, anchor_enabled = False, True
    elif fixed_preference is not None and method == (
        f"independent_{_preference_suffix(fixed_preference)}"
    ):
        seed_role = "independent_weighted"
        residual_radius = summary.get("residual_radius")
        transfer_trunks, anchor_enabled = True, True
    else:
        return False
    try:
        expected = {
            "run_id": f"{method}-seed-{int(seed)}",
            "algorithm": summary["algorithm_id"],
            "method": method,
            "source_tree_digest": summary["source_tree_digest"],
            "profile_digest": summary["profile_digest"],
            "physical_configuration_fingerprint": summary[
                "physical_configuration_fingerprint"
            ],
            "objective": summary["objective"],
            "requested_episode_target": int(summary["episodes"]),
            "environment_seed": int(record["environment_seed"]),
            "training_seed": _derive_research_seed(seed, seed_role),
            "baseline_policy_digest": record["baseline_policy_digest"],
            "preference_grid": summary["preference_grid"],
            "fixed_preference": fixed_preference,
            "residual_radius": float(residual_radius),
            "ensemble_size": summary["ensemble_size"],
            "residual_scales": summary["residual_scales"],
            "transfer_trunks": transfer_trunks,
            "anchor_enabled": anchor_enabled,
            "anchor_digest": ParetoAnchors.from_state_dict(
                summary["anchors"]
            ).digest(),
            "replay_action_semantics": REPLAY_ACTION_SEMANTICS,
            "training_seed_protocol": summary["training_seed_protocol"],
            "power_hypothesis_id": summary["power_hypothesis_id"],
            "baseline_payload_sha256": summary["baseline_payload_sha256"],
            "baseline_validation_sha256": summary[
                "baseline_validation_sha256"
            ],
            "ambiguity_contract": summary["ambiguity_contract"],
            **{
                field: summary[field]
                for field in AMBIGUITY_IDENTITY_REGISTRIES
            },
        }
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        _verify_checkpoint_record(root, record)
        and record.get("training_seed") == expected["training_seed"]
        and _is_sha256(record.get("baseline_policy_digest"))
        and dict(identity) == expected
        and _verify_checkpoint_identity_envelope(root, record, identity)
        and _verify_internal_checkpoint_authority(root, record, identity)
    )


def _verify_scalar_checkpoint_identity(
        root: Path, summary: Mapping, record: Mapping, *, seed: int,
        method: str,
) -> bool:
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        return False
    specifications = {
        "joint_source": (
            f"joint-source-seed-{seed}", "DDPG", seed,
            derive_training_seed(seed, "source"),
        ),
        "same_objective_dtl": (
            f"same-objective-dtl-seed-{seed}", "DTL", seed + 100_000,
            derive_training_seed(seed, "target"),
        ),
        "cross_objective_dtl_diagnostic": (
            f"cross-objective-dtl-seed-{seed}", "DTL", seed + 100_000,
            derive_training_seed(seed, "target"),
        ),
        "ddpg_scratch": (
            f"joint-ddpg-scratch-seed-{seed}", "DDPG", seed + 100_000,
            derive_training_seed(seed, "target"),
        ),
    }
    if method not in specifications:
        return False
    run_id, algorithm, environment_seed, training_seed = specifications[method]
    configuration_fingerprint = identity.get("configuration_fingerprint")
    source_model_digest = identity.get("source_model_digest")
    state_dim = identity.get("state_dim")
    action_dim = identity.get("action_dim")
    if (
        not isinstance(configuration_fingerprint, str)
        or len(configuration_fingerprint) != 20
        or any(
            character not in "0123456789abcdef"
            for character in configuration_fingerprint
        )
        or isinstance(state_dim, bool)
        or isinstance(action_dim, bool)
        or not isinstance(state_dim, int)
        or not isinstance(action_dim, int)
        or state_dim <= 0
        or action_dim <= 0
        or (
            algorithm == "DDPG"
            and source_model_digest != "not_applicable"
        )
        or (
            algorithm == "DTL"
            and not _is_sha256(source_model_digest)
        )
    ):
        return False
    expected = {
        "run_id": run_id,
        "algorithm": algorithm,
        "configuration_fingerprint": configuration_fingerprint,
        "parameter_profile_digest": summary.get("profile_digest"),
        "objective": summary.get("objective"),
        "requested_episode_target": int(summary.get("episodes", -1)),
        "power_hypothesis_id": summary.get("power_hypothesis_id"),
        "source_tree_digest": summary.get("source_tree_digest"),
        "source_model_digest": source_model_digest,
        "environment_seed": environment_seed,
        "training_seed": training_seed,
        "training_seed_protocol": summary.get("training_seed_protocol"),
        "replay_action_semantics": REPLAY_ACTION_SEMANTICS,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "dtl_transfer_mode": (
            summary.get("dtl_transfer_mode")
            if algorithm == "DTL" else "not_applicable"
        ),
        "dtl_transfer_schedule_id": (
            summary.get("dtl_transfer_schedule_id")
            if algorithm == "DTL" else "not_applicable"
        ),
        **{
            field: summary.get(field)
            for field in AMBIGUITY_IDENTITY_REGISTRIES
            if field not in {
                "dtl_transfer_mode", "dtl_transfer_schedule_id"
            }
        },
    }
    return bool(
        _verify_checkpoint_record(root, record)
        and dict(identity) == expected
        and _verify_checkpoint_identity_envelope(root, record, identity)
        and _verify_internal_checkpoint_authority(root, record, identity)
    )


def _validate_gate_evidence_payload(summary, payload) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "ba_pcdtl_gate_validation_evidence"
        or payload.get("selection_partition") != "validation_only"
        or payload.get("validation_scene_seeds")
        != summary.get("validation_scene_seeds")
        or payload.get("research_seeds") != summary.get("research_seeds")
        or payload.get("anchors_sha256") != summary.get("anchors_lock_sha256")
        or payload.get("gate_selection_grid")
        != summary.get("gate_selection_grid")
        or payload.get("hypervolume_reference_normalized")
        != summary.get("hypervolume_reference_normalized")
    ):
        raise ValueError("Gate-validation evidence authority fields mismatch.")
    anchors = ParetoAnchors.from_state_dict(summary["anchors"])
    preferences = validate_preferences(summary["preference_grid"]).tolist()
    reference = summary["hypervolume_reference_normalized"]
    validation_scenes = summary["validation_scene_seeds"]
    rows = payload.get("rows")
    candidates = summary.get("gate_selection", {}).get("candidates", [])
    gate_grid = summary.get("gate_selection_grid", {})
    coefficients = gate_grid.get("uncertainty_coefficients", [])
    margins = gate_grid.get("margins", [])
    expected_pairs = {
        (float(coefficient), float(margin))
        for coefficient in coefficients
        for margin in margins
    }
    candidate_pairs = {
        (
            float(candidate["uncertainty_coefficient"]),
            float(candidate["margin"]),
        )
        for candidate in candidates
    }
    if (
        not coefficients
        or not margins
        or len(expected_pairs) != len(coefficients) * len(margins)
        or candidate_pairs != expected_pairs
        or len(candidates) != len(expected_pairs)
    ):
        raise ValueError("Locked gate candidates do not cover the preregistered grid.")
    expected_count = len(summary["research_seeds"]) * len(expected_pairs)
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("Gate-validation raw row count is incomplete.")
    recomputed = {}
    seen = set()
    for row in rows:
        key = (
            int(row["research_seed"]),
            float(row["uncertainty_coefficient"]),
            float(row["margin"]),
        )
        if key in seen or key[0] not in summary["research_seeds"]:
            raise ValueError("Gate-validation row identity is duplicate or unknown.")
        seen.add(key)
        expected_gate_configuration = {
            "uncertainty_coefficient": key[1],
            "margin": key[2],
        }
        if (
            row.get("raw", {}).get("use_gate") is not True
            or row.get("raw", {}).get("gate_configuration")
            != expected_gate_configuration
        ):
            raise ValueError(
                "Gate-validation raw evaluation used a different gate "
                "configuration."
            )
        grid = _grid_records(
            row["raw"], preferences, validation_scenes,
            summary["evaluation_episodes"],
        )
        points = []
        feasibility = []
        fallback = []
        balanced_joint = None
        for preference, raw_record, arrays in grid:
            points.append([
                float(np.mean(arrays["communication"])),
                float(np.mean(arrays["sensing"])),
            ])
            feasibility.extend(arrays["feasibility"])
            fallback.extend(
                _finite_array(
                    raw_record, "baseline_fallback_rate", len(validation_scenes)
                )
            )
            if np.allclose(preference, [0.5, 0.5], rtol=0.0, atol=1e-8):
                balanced_joint = float(np.mean(arrays["joint"]))
        metrics = {
            "feasibility": float(np.min(feasibility)),
            "joint": balanced_joint,
            "hypervolume": hypervolume_2d(anchors.normalize(points), reference),
            "fallback_rate": float(np.mean(fallback)),
        }
        if balanced_joint is None or any(
            not np.isclose(float(row[name]), value, rtol=0.0, atol=1e-10)
            for name, value in metrics.items()
        ):
            raise ValueError("Gate-validation row metrics do not match raw arrays.")
        recomputed.setdefault(key[1:], []).append((key[0], metrics))

    if len(recomputed) != len(candidates):
        raise ValueError("Gate-validation candidate grid is incomplete.")
    for candidate in candidates:
        key = (
            float(candidate["uncertainty_coefficient"]),
            float(candidate["margin"]),
        )
        group = recomputed.get(key)
        if group is None or len(group) != len(summary["research_seeds"]):
            raise ValueError("Gate candidate lacks every research seed.")
        aggregate = {
            "feasibility": min(metrics["feasibility"] for _seed, metrics in group),
            "joint": float(np.mean([metrics["joint"] for _seed, metrics in group])),
            "hypervolume": float(
                np.mean([metrics["hypervolume"] for _seed, metrics in group])
            ),
            "fallback_rate": float(
                np.mean([metrics["fallback_rate"] for _seed, metrics in group])
            ),
        }
        if any(
            not np.isclose(float(candidate[name]), value, rtol=0.0, atol=1e-10)
            for name, value in aggregate.items()
        ):
            raise ValueError("Locked gate aggregate does not match raw validation rows.")


def _validate_file_provenance(summary, root: Path) -> tuple[bool, str | None]:
    anchors_path = root / "pareto_anchors_lock.json"
    gate_path = root / "gate_selection_lock.json"
    gate_validation_path = root / "gate_validation_evidence.json"
    manifest_path = root / "run_manifest.json"
    if (
        not anchors_path.is_file()
        or not gate_path.is_file()
        or not gate_validation_path.is_file()
        or not manifest_path.is_file()
    ):
        return False, (
            "Anchor, gate, gate-validation evidence, or run manifest file is missing."
        )
    if _sha256_file(anchors_path) != summary.get("anchors_lock_sha256"):
        return False, "Anchor lock hash mismatch."
    with anchors_path.open("r", encoding="utf-8") as stream:
        anchor_lock = json.load(stream)
    if (
        anchor_lock.get("schema_version") != 1
        or anchor_lock.get("anchors") != summary.get("anchors")
        or anchor_lock.get("anchor_digest")
        != ParetoAnchors.from_state_dict(summary["anchors"]).digest()
        or anchor_lock.get("hypervolume_reference_rule")
        != summary.get("hypervolume_reference_rule")
        or anchor_lock.get("normalized_hypervolume_reference")
        != summary.get("hypervolume_reference_normalized")
    ):
        return False, "Anchor lock payload mismatch."
    if _sha256_file(gate_path) != summary.get("gate_selection_lock_sha256"):
        return False, "Gate lock hash mismatch."
    gate_validation_digest = _sha256_file(gate_validation_path)
    if (
        gate_validation_digest != summary.get("gate_validation_evidence_sha256")
        or gate_validation_digest
        != summary.get("gate_selection", {}).get("validation_evidence_sha256")
    ):
        return False, "Gate-validation evidence hash mismatch."
    with gate_path.open("r", encoding="utf-8") as stream:
        if json.load(stream) != summary.get("gate_selection"):
            return False, "Gate lock payload mismatch."
    with gate_validation_path.open("r", encoding="utf-8") as stream:
        _validate_gate_evidence_payload(summary, json.load(stream))
    with manifest_path.open("r", encoding="utf-8") as stream:
        _validate_run_manifest_payload(summary, json.load(stream))
    scalar_checkpoint_hashes = []
    for seed_record in summary.get("research_seed_records", []):
        scalar = seed_record.get("training", {}).get("scalar", {}).get("methods", {})
        vector = seed_record.get("training", {}).get("vector", {}).get("methods", {})
        scalar_training = seed_record.get("training", {}).get("scalar", {})
        source_record = {
            "checkpoint": scalar_training.get("joint_source_checkpoint"),
            "checkpoint_sha256": scalar_training.get(
                "joint_source_checkpoint_sha256"
            ),
            "checkpoint_identity": scalar_training.get(
                "joint_source_checkpoint_identity"
            ),
            "checkpoint_identity_sha256": scalar_training.get(
                "joint_source_checkpoint_identity_sha256"
            ),
            "identity": scalar_training.get("joint_source_identity"),
            "training": scalar_training.get("joint_source_training"),
        }
        if not _verify_scalar_checkpoint_identity(
            root,
            summary,
            source_record,
            seed=int(seed_record["seed"]),
            method="joint_source",
        ):
            return False, "Joint-objective source checkpoint identity mismatch."
        try:
            joint_source_digest = source_model_state_digest(
                inspect_training_checkpoint(
                    root / source_record["checkpoint"]
                )["agent"]
            )
        except Exception:
            return False, "Joint-objective source model payload mismatch."
        baseline_authority = {
            int(row["seed"]): row
            for row in summary.get("baseline_checkpoint_authority", [])
        }
        paper_source_digest = baseline_authority.get(
            int(seed_record["seed"]), {}
        ).get("source_model_digest")
        scalar_order = (
            "same_objective_dtl",
            "cross_objective_dtl_diagnostic",
            "ddpg_scratch",
        )
        if set(scalar) != set(scalar_order):
            return False, "Scalar comparator checkpoint set is incomplete."
        for method_name in scalar_order:
            method_record = scalar[method_name]
            if not _verify_scalar_checkpoint_identity(
                root,
                summary,
                method_record,
                seed=int(seed_record["seed"]),
                method=method_name,
            ):
                return False, f"{method_name} checkpoint identity mismatch."
            expected_source_digest = (
                joint_source_digest
                if method_name == "same_objective_dtl"
                else paper_source_digest
                if method_name == "cross_objective_dtl_diagnostic"
                else "not_applicable"
            )
            if method_record["identity"].get(
                "source_model_digest"
            ) != expected_source_digest:
                return False, f"{method_name} source-model linkage mismatch."
            if method_record["identity"]["action_dim"] != (
                source_record["identity"]["action_dim"]
            ):
                return False, "Scalar source/target action dimensions mismatch."
            scalar_checkpoint_hashes.append(method_record["checkpoint_sha256"])

        expected_vector = {"ba_pcdtl"}
        if summary.get("comparison_level") == "full":
            expected_vector.update({
                "unanchored_vector_td3",
                "anchored_random_trunks",
                "independent_weighted_policies",
            })
        if set(vector) != expected_vector:
            return False, "Vector-method checkpoint set is incomplete."
        for method_name in (
            "ba_pcdtl",
            "unanchored_vector_td3",
            "anchored_random_trunks",
        ):
            if method_name not in vector:
                continue
            if not _verify_vector_checkpoint_identity(
                root,
                summary,
                vector[method_name],
                seed=int(seed_record["seed"]),
                method=method_name,
            ):
                return False, f"{method_name} checkpoint authority mismatch."
        if summary.get("comparison_level") == "full":
            policies = vector.get("independent_weighted_policies", {}).get("policies", [])
            if len(policies) != len(summary.get("preference_grid", [])):
                return False, "Independent-policy checkpoint grid is incomplete."
            for preference, policy in zip(summary["preference_grid"], policies):
                if (
                    policy.get("preference") != preference
                    or policy.get("identity", {}).get("fixed_preference")
                    != preference
                    or not _verify_vector_checkpoint_identity(
                        root,
                        summary,
                        policy,
                        seed=int(seed_record["seed"]),
                        method=f"independent_{_preference_suffix(preference)}",
                        fixed_preference=preference,
                    )
                ):
                    return False, "Independent-policy checkpoint authority mismatch."
    anchor_provenance = summary.get("anchors", {}).get("provenance", {})
    if anchor_provenance.get("source_checkpoint_sha256") != scalar_checkpoint_hashes:
        return False, "Anchor provenance checkpoint digest list mismatch."
    return True, None


def validate_pareto_result(
    summary: Mapping,
    *,
    level: str = "preliminary",
    result_directory=None,
    verify_files: bool = True,
    baseline: Mapping | None = None,
    baseline_validation: Mapping | None = None,
    baseline_directory=None,
) -> dict:
    """Validate raw paired results; malformed input is reported, never accepted."""
    if level not in {"preliminary", "publication"}:
        raise ValueError("Validation level must be preliminary or publication.")
    checks = {}
    statistics = {}

    try:
        payload_digest = result_payload_digest(summary)
        checks["json_finite_payload"] = _check(True)
    except (TypeError, ValueError) as error:
        payload_digest = None
        checks["json_finite_payload"] = _check(False, error=str(error))

    checks["generated_protocol"] = _check(
        summary.get("kind") == "generated_result"
        and summary.get("protocol") == "ba_pcdtl_research_extension"
        and summary.get("algorithm_id") == "ba_pcdtl_v1"
    )
    objective = summary.get("objective", {})
    checks["exact_png_objective"] = _check(
        objective.get("mode") == "png_joint_sum"
        and "communication_retention_fraction" not in objective
        and float(objective.get("user_rate_min", 0.0)) > 0.0
        and float(objective.get("target_rate_min", 0.0)) > 0.0
    )
    checks["no_pointwise_envelope"] = _check(
        not _contains_forbidden_envelope(summary)
    )

    effective_training_seeds = None
    if baseline is None or baseline_validation is None:
        checks["accepted_baseline_authority"] = _check(
            False, error="Accepted baseline summary and validation report are required."
        )
        checks["accepted_baseline_checkpoints"] = _check(
            False, error="Accepted baseline checkpoint authority is unavailable."
        )
    else:
        try:
            baseline_digest = result_payload_digest(baseline)
            validation_digest = result_payload_digest(baseline_validation)
            baseline_authority = (
                baseline_validation.get("validation") == "paper_baseline"
                and baseline_validation.get("overall_pass") is True
                and baseline_validation.get("baseline_payload_sha256")
                == baseline_digest
                and summary.get("baseline_payload_sha256") == baseline_digest
                and summary.get("baseline_validation_sha256") == validation_digest
                and summary.get("profile_digest") == baseline.get("profile_digest")
                and summary.get("physical_configuration_fingerprint")
                == baseline.get("physical_configuration_fingerprint")
                and summary.get("source_tree_digest")
                == baseline.get("source_tree_digest")
            )
            checks["accepted_baseline_authority"] = _check(baseline_authority)
            authority = summary.get("baseline_checkpoint_authority", [])
            baseline_records = baseline.get("seed_records", [])
            effective_training_seeds = set()
            for record in baseline_records:
                seed = int(record["seed"])
                effective_training_seeds.update({
                    int(record["source_seed"]),
                    int(record["target_seed"]),
                    int(record.get(
                        "source_training_seed",
                        derive_training_seed(seed, "source"),
                    )),
                    int(record.get(
                        "target_training_seed",
                        derive_training_seed(seed, "target"),
                    )),
                })
                effective_training_seeds.update(
                    _derive_research_seed(seed, role)
                    for role in (
                        "ba_pcdtl", "unanchored_vector_td3",
                        "anchored_random_trunks", "independent_weighted",
                    )
                )
            checkpoint_authority_valid = (
                [row.get("seed") for row in authority]
                == summary.get("research_seeds")
                and len(authority) == len(baseline_records)
            )
            for authority_row, baseline_row in zip(authority, baseline_records):
                checkpoint_authority_valid = checkpoint_authority_valid and (
                    authority_row.get("seed") == baseline_row.get("seed")
                    and authority_row.get("source_checkpoint")
                    == baseline_row.get("source_checkpoint")
                    and authority_row.get("dtl_pcsi_checkpoint")
                    == baseline_row.get("methods", {}).get("DTL-pCSI", {}).get(
                        "checkpoint"
                    )
                    and _is_sha256(authority_row.get("source_checkpoint_sha256"))
                    and authority_row.get("source_checkpoint_sha256")
                    == baseline_row.get("source_checkpoint_sha256")
                    and _is_sha256(
                        authority_row.get("dtl_pcsi_checkpoint_sha256")
                    )
                    and authority_row.get("dtl_pcsi_checkpoint_sha256")
                    == baseline_row.get("methods", {}).get("DTL-pCSI", {}).get(
                        "checkpoint_sha256"
                    )
                    and _is_sha256(authority_row.get("source_model_digest"))
                    and authority_row.get("source_model_digest")
                    == baseline_row.get("methods", {}).get(
                        "DTL-pCSI", {}
                    ).get("identity", {}).get("source_model_digest")
                )
                if baseline_directory is not None:
                    root = Path(baseline_directory).resolve()
                    checkpoints = (root / "checkpoints").resolve()
                    for path_field, hash_field in (
                        ("source_checkpoint", "source_checkpoint_sha256"),
                        ("dtl_pcsi_checkpoint", "dtl_pcsi_checkpoint_sha256"),
                    ):
                        path = (root / authority_row[path_field]).resolve()
                        checkpoint_authority_valid = checkpoint_authority_valid and (
                            path.parent == checkpoints
                            and path.is_file()
                            and _sha256_file(path) == authority_row[hash_field]
                        )
            checks["accepted_baseline_checkpoints"] = _check(
                checkpoint_authority_valid
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            checks["accepted_baseline_authority"] = _check(
                False, error=str(error)
            )
            checks["accepted_baseline_checkpoints"] = _check(
                False, error=str(error)
            )

    try:
        roles = validate_seed_roles(
            summary.get("research_seeds", []),
            summary.get("validation_scene_seeds", []),
            summary.get("test_scene_seeds", []),
            effective_training_seeds=(
                None
                if effective_training_seeds is None
                else sorted(effective_training_seeds)
            ),
        )
        seed_records = summary.get("research_seed_records", [])
        record_seeds = [int(row.get("seed")) for row in seed_records]
        seeds_valid = record_seeds == roles["research"]
        checks["independent_seed_roles"] = _check(seeds_valid)
    except (TypeError, ValueError) as error:
        roles = None
        seed_records = []
        checks["independent_seed_roles"] = _check(False, error=str(error))

    try:
        preferences = validate_preferences(summary.get("preference_grid", [])).tolist()
        preference_valid = any(
            np.allclose(row, [0.5, 0.5], rtol=0.0, atol=1e-8)
            for row in preferences
        )
        checks["preference_grid"] = _check(preference_valid)
    except (TypeError, ValueError) as error:
        preferences = []
        checks["preference_grid"] = _check(False, error=str(error))

    try:
        anchors = ParetoAnchors.from_state_dict(summary.get("anchors", {}))
        anchors_valid = (
            anchors.frozen
            and anchors.provenance.get("partition") == "training_only"
        )
        reference_rule = summary.get("hypervolume_reference_rule", {})
        if (
            reference_rule.get("physical_point_bits_per_second_per_hz")
            != [0.0, 0.0]
            or reference_rule.get("lock_partition")
            != "training_only_before_validation"
        ):
            raise ValueError("Hypervolume reference rule is invalid.")
        reference = np.asarray(
            summary.get("hypervolume_reference_normalized"), dtype=np.float64
        )
        if reference.shape != (2,) or not np.all(np.isfinite(reference)):
            raise ValueError("Normalized hypervolume reference is invalid.")
        expected_reference = normalized_physical_zero_reference(
            anchors,
            margin=reference_rule.get("normalized_margin"),
        )
        reference_valid = np.allclose(
            reference, expected_reference, rtol=0.0, atol=1e-12
        )
        recomputed_anchors = _recompute_training_anchors(summary)
        training_values_match = bool(
            np.allclose(
                recomputed_anchors.ideal, anchors.ideal, rtol=0.0, atol=1e-12
            )
            and np.allclose(
                recomputed_anchors.nadir, anchors.nadir, rtol=0.0, atol=1e-12
            )
        )
        checks["training_only_anchors"] = _check(
            anchors_valid and reference_valid and training_values_match
        )
    except (TypeError, ValueError) as error:
        anchors = None
        reference = None
        checks["training_only_anchors"] = _check(False, error=str(error))

    gate_lock = summary.get("gate_selection", {})
    selected_gate = gate_lock.get("selected", {})
    gate_candidates = gate_lock.get("candidates", [])
    try:
        gate_selection_matches_rule = (
            select_gate_configuration(gate_candidates) == selected_gate
        )
        gate_grid = summary.get("gate_selection_grid", {})
        coefficients = [
            float(value) for value in gate_grid["uncertainty_coefficients"]
        ]
        margins = [float(value) for value in gate_grid["margins"]]
        expected_gate_pairs = {
            (coefficient, margin)
            for coefficient in coefficients
            for margin in margins
        }
        candidate_gate_pairs = {
            (
                float(candidate["uncertainty_coefficient"]),
                float(candidate["margin"]),
            )
            for candidate in gate_candidates
        }
        gate_grid_valid = bool(
            coefficients
            and margins
            and len(expected_gate_pairs) == len(coefficients) * len(margins)
            and len(gate_candidates) == len(expected_gate_pairs)
            and candidate_gate_pairs == expected_gate_pairs
        )
    except (KeyError, TypeError, ValueError):
        gate_selection_matches_rule = False
        gate_grid_valid = False
    gate_valid = (
        gate_lock.get("selection_partition") == "validation_only"
        and gate_lock.get("validation_scene_seeds")
        == summary.get("validation_scene_seeds")
        and gate_lock.get("research_seeds") == summary.get("research_seeds")
        and gate_lock.get("gate_selection_grid")
        == summary.get("gate_selection_grid")
        and gate_lock.get("hypervolume_reference_normalized")
        == summary.get("hypervolume_reference_normalized")
        and _is_sha256(gate_lock.get("validation_evidence_sha256"))
        and gate_lock.get("validation_evidence_sha256")
        == summary.get("gate_validation_evidence_sha256")
        and math.isfinite(float(selected_gate.get("uncertainty_coefficient", math.nan)))
        and math.isfinite(float(selected_gate.get("margin", math.nan)))
        and isinstance(gate_candidates, list)
        and len(gate_candidates) > 0
        and selected_gate in gate_candidates
        and gate_selection_matches_rule
        and gate_grid_valid
    )
    checks["validation_locked_gate"] = _check(gate_valid)

    if verify_files:
        if result_directory is None:
            checks["file_and_checkpoint_provenance"] = _check(
                False, error="result_directory is required for file verification."
            )
        else:
            try:
                passed, error = _validate_file_provenance(
                    summary, Path(result_directory).resolve()
                )
                checks["file_and_checkpoint_provenance"] = _check(
                    passed, error=error
                )
            except (OSError, TypeError, ValueError) as error:
                checks["file_and_checkpoint_provenance"] = _check(
                    False, error=str(error)
                )
    else:
        checks["file_and_checkpoint_provenance"] = _check(
            True, verification="disabled_by_explicit_test_api"
        )

    primary_ready = all(
        checks[name]["pass"]
        for name in (
            "generated_protocol", "exact_png_objective", "no_pointwise_envelope",
            "accepted_baseline_authority", "accepted_baseline_checkpoints",
            "independent_seed_roles", "preference_grid", "training_only_anchors",
            "validation_locked_gate", "file_and_checkpoint_provenance",
        )
    )
    proposed_joint = []
    baseline_joint = []
    proposed_communication = []
    baseline_communication = []
    proposed_sensing = []
    baseline_sensing = []
    proposed_hv = []
    independent_hv = []
    scalar_comparator_names = (
        "same_objective_dtl",
        "cross_objective_dtl_diagnostic",
        "ddpg_scratch",
        "paper_dtl_pcsi",
    )
    comparator_joint = {name: [] for name in scalar_comparator_names}
    raw_valid = primary_ready
    physical_valid = primary_ready
    gate_identity_valid = primary_ready
    ablation_grid_valid = primary_ready
    scalar_grid_valid = primary_ready
    publication_grid_valid = primary_ready and summary.get("comparison_level") == "full"

    if primary_ready:
        try:
            scene_seeds = roles["test"]
            for seed_record in seed_records:
                tests = seed_record.get("test", {})
                proposed_eval = tests["ba_pcdtl"]
                rows = _grid_records(
                    proposed_eval, preferences, scene_seeds,
                    summary["evaluation_episodes"],
                )
                balanced_row, balanced_arrays = _balanced_record(rows)
                gate_identity_valid = gate_identity_valid and (
                    proposed_eval.get("use_gate") is True
                    and proposed_eval.get("gate_configuration") == {
                        "uncertainty_coefficient": selected_gate[
                            "uncertainty_coefficient"
                        ],
                        "margin": selected_gate["margin"],
                    }
                )
                for _preference, _row, arrays in rows:
                    physical_valid = physical_valid and (
                        np.allclose(arrays["feasibility"], 1.0, rtol=0.0, atol=1e-12)
                        and np.all(arrays["minimum_ue_rate"] >= objective["user_rate_min"])
                        and np.all(
                            arrays["minimum_target_rate"] >= objective["target_rate_min"]
                        )
                    )
                proposed_no_gate = tests["ba_pcdtl_no_gate"]
                no_gate_rows = _grid_records(
                    proposed_no_gate, preferences, scene_seeds,
                    summary["evaluation_episodes"],
                )
                ablation_grid_valid = ablation_grid_valid and (
                    proposed_no_gate.get("use_gate") is False
                    and "gate_configuration" not in proposed_no_gate
                )
                for _preference, _row, arrays in no_gate_rows:
                    physical_valid = physical_valid and (
                        np.allclose(
                            arrays["feasibility"], 1.0, rtol=0.0, atol=1e-12
                        )
                        and np.all(
                            arrays["minimum_ue_rate"] >= objective["user_rate_min"]
                        )
                        and np.all(
                            arrays["minimum_target_rate"]
                            >= objective["target_rate_min"]
                        )
                    )
                scalar_arrays_by_name = {}
                for comparator_name in scalar_comparator_names:
                    scalar = tests[comparator_name]
                    if scalar.get("scene_seeds") != scene_seeds:
                        raise ValueError(
                            f"{comparator_name} test seeds changed."
                        )
                    scalar_arrays = _validate_rate_record(
                        scalar,
                        len(scene_seeds),
                        scene_seeds=scene_seeds,
                        evaluation_episodes=summary["evaluation_episodes"],
                    )
                    scalar_arrays_by_name[comparator_name] = scalar_arrays
                    comparator_joint[comparator_name].append(
                        float(np.mean(scalar_arrays["joint"]))
                    )
                    physical_valid = physical_valid and (
                        np.allclose(
                            scalar_arrays["feasibility"], 1.0,
                            rtol=0.0, atol=1e-12,
                        )
                        and np.all(
                            scalar_arrays["minimum_ue_rate"]
                            >= objective["user_rate_min"]
                        )
                        and np.all(
                            scalar_arrays["minimum_target_rate"]
                            >= objective["target_rate_min"]
                        )
                    )
                scalar_arrays = scalar_arrays_by_name["same_objective_dtl"]
                proposed_joint.append(float(np.mean(balanced_arrays["joint"])))
                baseline_joint.append(float(np.mean(scalar_arrays["joint"])))
                proposed_communication.append(
                    float(np.mean(balanced_arrays["communication"]))
                )
                baseline_communication.append(
                    float(np.mean(scalar_arrays["communication"]))
                )
                proposed_sensing.append(float(np.mean(balanced_arrays["sensing"])))
                baseline_sensing.append(float(np.mean(scalar_arrays["sensing"])))

                if level == "publication":
                    for method_name in (
                        "unanchored_vector_td3",
                        "anchored_random_trunks",
                    ):
                        ablation = tests[method_name]
                        ablation_rows = _grid_records(
                            ablation, preferences, scene_seeds,
                            summary["evaluation_episodes"],
                        )
                        ablation_grid_valid = ablation_grid_valid and (
                            ablation.get("use_gate") is False
                            and "gate_configuration" not in ablation
                        )
                        for _preference, _row, arrays in ablation_rows:
                            physical_valid = physical_valid and (
                                np.allclose(
                                    arrays["feasibility"], 1.0,
                                    rtol=0.0, atol=1e-12,
                                )
                                and np.all(
                                    arrays["minimum_ue_rate"]
                                    >= objective["user_rate_min"]
                                )
                                and np.all(
                                    arrays["minimum_target_rate"]
                                    >= objective["target_rate_min"]
                                )
                            )
                    independent = tests.get("independent_weighted_policies", {})
                    policies = independent.get("policies", [])
                    if (
                        independent.get("scene_seeds") != scene_seeds
                        or len(policies) != len(preferences)
                    ):
                        publication_grid_valid = False
                        continue
                    proposed_points = []
                    independent_points = []
                    for (preference, _row, arrays), policy in zip(
                            no_gate_rows, policies):
                        if not np.allclose(
                            policy.get("preference"), preference, rtol=0.0, atol=1e-8
                        ):
                            publication_grid_valid = False
                            continue
                        if (
                            policy.get("use_gate") is not False
                            or "gate_configuration" in policy
                        ):
                            publication_grid_valid = False
                            continue
                        policy_arrays = _validate_rate_record(
                            policy.get("evaluation", {}),
                            len(scene_seeds),
                            require_policy_evidence=True,
                            scene_seeds=scene_seeds,
                            evaluation_episodes=summary["evaluation_episodes"],
                        )
                        physical_valid = physical_valid and (
                            np.allclose(
                                policy_arrays["feasibility"], 1.0,
                                rtol=0.0, atol=1e-12,
                            )
                            and np.all(
                                policy_arrays["minimum_ue_rate"]
                                >= objective["user_rate_min"]
                            )
                            and np.all(
                                policy_arrays["minimum_target_rate"]
                                >= objective["target_rate_min"]
                            )
                        )
                        proposed_points.append([
                            float(np.mean(arrays["communication"])),
                            float(np.mean(arrays["sensing"])),
                        ])
                        independent_points.append([
                            float(np.mean(policy_arrays["communication"])),
                            float(np.mean(policy_arrays["sensing"])),
                        ])
                    if len(proposed_points) == len(preferences):
                        proposed_hv.append(
                            hypervolume_2d(anchors.normalize(proposed_points), reference)
                        )
                        independent_hv.append(
                            hypervolume_2d(anchors.normalize(independent_points), reference)
                        )
            raw_valid = True
        except (KeyError, TypeError, ValueError) as error:
            raw_valid = False
            scalar_grid_valid = False
            checks["raw_paired_evidence"] = _check(False, error=str(error))

    checks.setdefault("raw_paired_evidence", _check(raw_valid))
    checks["actor_action_attribution"] = _check(
        raw_valid, thresholds=PROJECTION_ATTRIBUTION_THRESHOLDS
    )
    checks["physical_feasibility_and_rate_floors"] = _check(physical_valid)
    checks["locked_gate_identity"] = _check(gate_identity_valid)
    checks["complete_ungated_ablation_grid"] = _check(ablation_grid_valid)
    scalar_grid_valid = scalar_grid_valid and all(
        len(values) == len(summary.get("research_seeds", []))
        for values in comparator_joint.values()
    )
    checks["complete_scalar_comparator_grid"] = _check(scalar_grid_valid)

    joint_pass = False
    paper_joint_pass = False
    if raw_valid and len(proposed_joint) == len(summary.get("research_seeds", [])):
        try:
            joint_interval = paired_confidence_interval(
                proposed_joint, baseline_joint, confidence=0.95
            )
            statistics["joint_improvement"] = _interval_dict(joint_interval)
            joint_pass = joint_interval.lower > 0.0

            comm_interval = paired_confidence_interval(
                proposed_communication, baseline_communication, confidence=0.95
            )
            sensing_interval = paired_confidence_interval(
                proposed_sensing, baseline_sensing, confidence=0.95
            )
            adjusted = holm_adjust([comm_interval.p_value, sensing_interval.p_value])
            statistics["secondary_communication"] = {
                **_interval_dict(comm_interval), "holm_p_value": float(adjusted[0])
            }
            statistics["secondary_sensing"] = {
                **_interval_dict(sensing_interval), "holm_p_value": float(adjusted[1])
            }
            paper_interval = paired_confidence_interval(
                proposed_joint,
                comparator_joint["paper_dtl_pcsi"],
                confidence=0.95,
            )
            statistics["joint_improvement_vs_paper_dtl"] = _interval_dict(
                paper_interval
            )
            paper_joint_pass = paper_interval.lower > 0.0
            for comparator_name in (
                "cross_objective_dtl_diagnostic",
                "ddpg_scratch",
            ):
                interval = paired_confidence_interval(
                    proposed_joint,
                    comparator_joint[comparator_name],
                    confidence=0.95,
                )
                statistics[
                    f"joint_improvement_vs_{comparator_name}"
                ] = _interval_dict(interval)
        except ValueError as error:
            statistics["joint_error"] = str(error)
    checks["positive_paired_joint_ci"] = _check(joint_pass)
    checks["positive_paired_joint_ci_vs_paper_baseline"] = _check(
        paper_joint_pass
    )

    if level == "publication":
        hv_pass = False
        publication_grid_valid = publication_grid_valid and (
            len(proposed_hv) == len(summary.get("research_seeds", []))
            and len(independent_hv) == len(proposed_hv)
        )
        if publication_grid_valid:
            try:
                hv_interval = paired_confidence_interval(
                    proposed_hv, independent_hv, confidence=0.95
                )
                statistics["hypervolume_improvement"] = _interval_dict(hv_interval)
                hv_pass = hv_interval.lower > 0.0
            except ValueError as error:
                statistics["hypervolume_error"] = str(error)
        checks["complete_fair_independent_policy_grid"] = _check(
            publication_grid_valid
        )
        checks["positive_paired_hypervolume_ci"] = _check(hv_pass)

    return {
        "schema_version": 1,
        "validation": "ba_pcdtl_research_extension",
        "validation_level": level,
        "extension_payload_sha256": payload_digest,
        "overall_pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
        "statistics": statistics,
    }


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--baseline-validation", required=True)
    parser.add_argument("--level", choices=("preliminary", "publication"), default="preliminary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary_path = Path(args.summary).resolve()
    baseline_path = Path(args.baseline_summary).resolve()
    baseline = _read_json(baseline_path)
    baseline_validation = _read_json(args.baseline_validation)
    require_canonical_baseline_validation(
        baseline,
        baseline_validation,
        baseline_path.parent,
    )
    report = validate_pareto_result(
        _read_json(summary_path),
        level=args.level,
        result_directory=summary_path.parent,
        verify_files=True,
        baseline=baseline,
        baseline_validation=baseline_validation,
        baseline_directory=baseline_path.parent,
    )
    from experiment_protocol import atomic_write_json

    atomic_write_json(args.output, report)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
