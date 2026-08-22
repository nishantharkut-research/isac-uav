"""Quantitative acceptance checks for generated baseline and extension runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from checkpointing import (
    inspect_training_checkpoint,
    validate_checkpoint_authority,
)
from experiment_protocol import (
    EVALUATION_SEED_PROTOCOL,
    PROJECTION_EVIDENCE_KEYS,
    PROJECTION_ATTRIBUTION_THRESHOLDS,
    TRAINING_SEED_PROTOCOL,
    REPLAY_ACTION_SEMANTICS,
    derive_training_seed,
    evaluation_scene_seed_grid,
    projection_attribution_passes,
)
from ambiguity_hypotheses import (
    AMBIGUITY_IDENTITY_REGISTRIES,
    valid_ambiguity_manifest_fields,
)
from paper_reference import (
    load_paper_references,
    require_canonical_paper_references,
)
from research_objectives import ObjectiveMode, ObjectiveSpec
from source_integrity import source_model_state_digest


METHOD_ORDER = ("DTL-pCSI", "DTL-WFB", "DTL-iCSI", "DDPG")
AMBIGUITY_RESULT_FIELDS = AMBIGUITY_IDENTITY_REGISTRIES


def _valid_ambiguity_contract(payload):
    return valid_ambiguity_manifest_fields(payload)


def _valid_evaluation_seed_schedule(payload):
    """Verify the globally unique deterministic evaluation-scene assignment."""
    try:
        seeds = payload["seeds"]
        episodes = int(payload["evaluation_episodes"])
        records = payload["seed_records"]
        expected = evaluation_scene_seed_grid(seeds, episodes)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    if (
        payload.get("evaluation_seed_protocol") != EVALUATION_SEED_PROTOCOL
        or not isinstance(records, list)
        or len(records) != len(seeds)
    ):
        return False
    records_by_seed = {record.get("seed"): record for record in records}
    return bool(
        len(records_by_seed) == len(seeds)
        and set(records_by_seed) == set(seeds)
        and all(
            records_by_seed[seed].get("evaluation_scene_seeds")
            == expected[seed]
            for seed in seeds
        )
    )


def _valid_paired_evaluation_seed_schedule(extension, baseline):
    if not (
        _valid_evaluation_seed_schedule(extension)
        and _valid_evaluation_seed_schedule(baseline)
    ):
        return False
    baseline_by_seed = {
        record.get("seed"): record for record in baseline["seed_records"]
    }
    return all(
        record.get("evaluation_scene_seeds")
        == baseline_by_seed.get(record.get("seed"), {}).get(
            "evaluation_scene_seeds"
        )
        for record in extension["seed_records"]
    )


def result_payload_digest(payload: dict) -> str:
    """Hash a parsed result payload independently of JSON formatting."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_canonical_baseline_validation(
        baseline, supplied_validation, result_directory
):
    """Recompute baseline acceptance; reject self-attested validation files."""
    recomputed = validate_baseline_result(
        baseline,
        references=None,
        result_directory=result_directory,
        verify_files=True,
    )
    if recomputed.get("overall_pass") is not True:
        raise ValueError(
            "Baseline fails canonical paper-reference/checkpoint validation."
        )
    if supplied_validation != recomputed:
        raise ValueError(
            "Supplied baseline validation is not the canonical recomputed report."
        )
    return recomputed


def _seed_values(values, label):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 5 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} requires at least five finite seed values.")
    return array


def _summary(values):
    values = _seed_values(values, "confidence interval")
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    half_width = float(t.ppf(0.975, values.size - 1) * standard_error)
    return {
        "n": int(values.size),
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def paired_normalized_baseline_objective(
        baseline_communication,
        baseline_sensing,
        communication_references,
        sensing_references,
        pareto_weight,
):
    """Evaluate paired baseline samples under the extension objective anchors."""
    communication = np.asarray(baseline_communication, dtype=float)
    sensing = np.asarray(baseline_sensing, dtype=float)
    communication_reference = np.asarray(
        communication_references, dtype=float
    )
    sensing_reference = np.asarray(sensing_references, dtype=float)
    arrays = (
        communication,
        sensing,
        communication_reference,
        sensing_reference,
    )
    weight = float(pareto_weight)
    if (
        communication.ndim != 1
        or communication.size == 0
        or any(array.shape != communication.shape for array in arrays[1:])
        or any(not np.all(np.isfinite(array)) for array in arrays)
        or not np.isfinite(weight)
        or not 0.0 <= weight <= 1.0
        or np.any(communication_reference <= 0.0)
        or np.any(sensing_reference <= 0.0)
    ):
        raise ValueError(
            "Normalized Pareto comparison requires aligned finite samples, "
            "positive per-seed references, and a weight in [0, 1]."
        )
    return (
        (1.0 - weight) * communication / communication_reference
        + weight * sensing / sensing_reference
    )


def _check(passed, **details):
    return {"pass": bool(passed), **details}


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_run_checkpoint(root: Path, relative_path, expected_digest) -> bool:
    try:
        relative = Path(str(relative_path or ""))
        if relative.is_absolute():
            return False
        checkpoints = (root / "checkpoints").resolve()
        path = (root / relative).resolve()
        return (
            path.parent == checkpoints
            and path.suffix == ".pt"
            and path.is_file()
            and _is_sha256(expected_digest)
            and _sha256_file(path) == expected_digest
        )
    except (OSError, TypeError, ValueError):
        return False


def _valid_hex_digest(value, length) -> bool:
    return isinstance(value, str) and len(value) == int(length) and all(
        character in "0123456789abcdef" for character in value
    )


def _verify_training_identity_envelope(root: Path, record: dict) -> bool:
    try:
        checkpoint_relative = Path(str(record["checkpoint"]))
        identity_relative = Path(str(record["checkpoint_identity"]))
        expected_identity_relative = checkpoint_relative.with_suffix(
            checkpoint_relative.suffix + ".identity.json"
        )
        identity_path = (root / identity_relative).resolve()
        if (
            identity_relative != expected_identity_relative
            or identity_path.parent != (root / "checkpoints").resolve()
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
            "identity": record["identity"],
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _verify_internal_checkpoint_authority(root: Path, record: dict) -> bool:
    """Fail closed unless the checkpoint itself matches the serialized run row."""
    try:
        checkpoint = (root / Path(str(record["checkpoint"]))).resolve()
        validate_checkpoint_authority(
            checkpoint,
            record["identity"],
            record["training"],
        )
        return True
    except Exception:
        # Result validation is a trust boundary: malformed torch payloads fail.
        return False


def _checkpoint_agent_model_digest(root: Path, record: dict) -> str:
    checkpoint = (root / Path(str(record["checkpoint"]))).resolve()
    inspected = inspect_training_checkpoint(checkpoint)
    return source_model_state_digest(inspected["agent"])


def _valid_baseline_training_identity(
        result, seed_record, checkpoint_record, method_name
) -> bool:
    identity = checkpoint_record.get("identity")
    if not isinstance(identity, dict):
        return False
    is_source = method_name == "source"
    algorithm = "DDPG" if is_source or method_name == "DDPG" else "DTL"
    run_id = (
        seed_record.get("source_tag") if is_source
        else checkpoint_record.get("tag")
    )
    environment_seed = seed_record.get(
        "source_seed" if is_source else "target_seed"
    )
    training_seed = seed_record.get(
        "source_training_seed" if is_source else "target_training_seed"
    )
    configuration_fingerprint = identity.get("configuration_fingerprint")
    source_model_digest = identity.get("source_model_digest")
    state_dim = identity.get("state_dim")
    action_dim = identity.get("action_dim")
    if (
        not _valid_hex_digest(configuration_fingerprint, 20)
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
        "parameter_profile_digest": result.get("profile_digest"),
        "objective": result.get("objective"),
        "requested_episode_target": int(result.get("episodes", -1)),
        "power_hypothesis_id": result.get("power_hypothesis_id"),
        "source_tree_digest": result.get("source_tree_digest"),
        "source_model_digest": source_model_digest,
        "environment_seed": environment_seed,
        "training_seed": training_seed,
        "training_seed_protocol": result.get("training_seed_protocol"),
        "replay_action_semantics": REPLAY_ACTION_SEMANTICS,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "dtl_transfer_mode": (
            result.get("dtl_transfer_mode")
            if algorithm == "DTL" else "not_applicable"
        ),
        "dtl_transfer_schedule_id": (
            result.get("dtl_transfer_schedule_id")
            if algorithm == "DTL" else "not_applicable"
        ),
        **{
            field: result.get(field)
            for field in AMBIGUITY_RESULT_FIELDS
            if field not in {
                "dtl_transfer_mode", "dtl_transfer_schedule_id"
            }
        },
    }
    return dict(identity) == expected


def _valid_extension_training_identity(extension, record) -> bool:
    identity = record.get("identity")
    if not isinstance(identity, dict):
        return False
    configuration_fingerprint = identity.get("configuration_fingerprint")
    source_model_digest = identity.get("source_model_digest")
    state_dim = identity.get("state_dim")
    action_dim = identity.get("action_dim")
    try:
        seed = int(record["seed"])
        environment_seed = int(record["environment_seed"])
        training_seed = int(record["training_seed"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        environment_seed != seed + 100_000
        or training_seed != derive_training_seed(seed, "target")
        or not _valid_hex_digest(configuration_fingerprint, 20)
        or not _is_sha256(source_model_digest)
        or isinstance(state_dim, bool)
        or isinstance(action_dim, bool)
        or not isinstance(state_dim, int)
        or not isinstance(action_dim, int)
        or state_dim <= 0
        or action_dim <= 0
    ):
        return False
    expected = {
        "run_id": record.get("tag"),
        "algorithm": "DTL",
        "configuration_fingerprint": configuration_fingerprint,
        "parameter_profile_digest": extension.get("profile_digest"),
        "objective": record.get("objective"),
        "requested_episode_target": int(extension.get("episodes", -1)),
        "power_hypothesis_id": extension.get("power_hypothesis_id"),
        "source_tree_digest": extension.get("source_tree_digest"),
        "source_model_digest": source_model_digest,
        "environment_seed": environment_seed,
        "training_seed": training_seed,
        "training_seed_protocol": extension.get("training_seed_protocol"),
        "replay_action_semantics": REPLAY_ACTION_SEMANTICS,
        "state_dim": state_dim,
        "action_dim": action_dim,
        **{
            field: extension.get(field)
            for field in AMBIGUITY_RESULT_FIELDS
        },
    }
    return dict(identity) == expected


def _validate_baseline_manifest_payload(result, manifest) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "run_manifest"
        or manifest.get("status") != "complete"
        or manifest.get("summary") != "baseline_summary.json"
        or manifest.get("summary_payload_sha256")
        != result_payload_digest(result)
    ):
        raise ValueError("Completed baseline run manifest authority mismatch.")
    fields = (
        "protocol",
        "profile_digest",
        "physical_configuration_fingerprint",
        "source_tree_digest",
        "training_seed_protocol",
        "evaluation_seed_protocol",
        "power_hypothesis_id",
        "objective",
        "objective_reference_protocol",
        "objective_reference_window",
        "seeds",
        "episodes",
        "evaluation_episodes",
        *AMBIGUITY_RESULT_FIELDS,
    )
    mismatches = [
        field for field in fields if manifest.get(field) != result.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Baseline manifest and summary mismatch: " + ", ".join(mismatches)
        )


def _validate_baseline_file_provenance(result, root: Path):
    root = Path(root).resolve()
    records = result.get("seed_records", [])
    if not isinstance(records, list) or len(records) != len(result.get("seeds", [])):
        return False, "Baseline seed checkpoint grid is incomplete."
    for record in records:
        source_record = {
            "checkpoint": record.get("source_checkpoint"),
            "checkpoint_sha256": record.get("source_checkpoint_sha256"),
            "checkpoint_identity": record.get("source_checkpoint_identity"),
            "checkpoint_identity_sha256": record.get(
                "source_checkpoint_identity_sha256"
            ),
            "identity": record.get("source_identity"),
            "training": record.get("source_training"),
        }
        if not _verify_run_checkpoint(
            root, source_record["checkpoint"], source_record["checkpoint_sha256"]
        ):
            return False, "Baseline source checkpoint authority mismatch."
        if (
            not _valid_baseline_training_identity(
                result, record, source_record, "source"
            )
            or not _verify_training_identity_envelope(root, source_record)
            or not _verify_internal_checkpoint_authority(root, source_record)
        ):
            return False, "Baseline source checkpoint identity mismatch."
        try:
            source_model_digest = _checkpoint_agent_model_digest(
                root, source_record
            )
        except Exception:
            return False, "Baseline source checkpoint model payload mismatch."
        methods = record.get("methods", {})
        if set(methods) != set(METHOD_ORDER):
            return False, "Baseline method checkpoint grid is incomplete."
        source_identity = source_record["identity"]
        for method_name in METHOD_ORDER:
            method = methods[method_name]
            if not _verify_run_checkpoint(
                root, method.get("checkpoint"), method.get("checkpoint_sha256")
            ):
                return False, f"Baseline {method_name} checkpoint authority mismatch."
            if (
                not _valid_baseline_training_identity(
                    result, record, method, method_name
                )
                or not _verify_training_identity_envelope(root, method)
                or not _verify_internal_checkpoint_authority(root, method)
            ):
                return False, f"Baseline {method_name} checkpoint identity mismatch."
            method_identity = method["identity"]
            if method_identity["action_dim"] != source_identity["action_dim"]:
                return False, "Baseline source/target action dimensions mismatch."
            if method_name != "DDPG" and (
                method_identity["source_model_digest"]
                != source_model_digest
                or method_identity["state_dim"] < source_identity["state_dim"]
            ):
                return False, "Baseline DTL source-model authority mismatch."
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        return False, "Completed baseline run manifest is missing."
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    _validate_baseline_manifest_payload(result, manifest)
    sanity_path = root / "pretraining_physical_sanity.json"
    sanity_digest = result.get("pretraining_physical_sanity_sha256")
    if (
        manifest.get("pretraining_physical_sanity") != sanity_path.name
        or manifest.get("pretraining_physical_sanity_sha256") != sanity_digest
        or not sanity_path.is_file()
        or not _is_sha256(sanity_digest)
        or _sha256_file(sanity_path) != sanity_digest
    ):
        return False, "Pre-training physical-sanity authority mismatch."
    try:
        with sanity_path.open("r", encoding="utf-8") as stream:
            sanity = json.load(stream)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False, "Pre-training physical-sanity report is invalid."
    if (
        sanity.get("schema_version") != 1
        or sanity.get("kind") != "pretraining_physical_sanity"
        or sanity.get("advisory_only") is not True
        or not isinstance(sanity.get("passed"), bool)
        or sanity.get("profile_digest") != result.get("profile_digest")
        or sanity.get("physical_configuration_fingerprint")
        != result.get("physical_configuration_fingerprint")
        or sanity.get("source_tree_digest") != result.get("source_tree_digest")
        or not _valid_ambiguity_contract(sanity)
    ):
        return False, "Pre-training physical-sanity report is unauthorized."
    return True, None


def _validate_extension_file_provenance(extension, root: Path, baseline):
    root = Path(root).resolve()
    records = extension.get("seed_records", [])
    if not isinstance(records, list) or len(records) != len(
        extension.get("seeds", [])
    ):
        return False, "Extension checkpoint grid is incomplete."
    baseline_by_seed = {
        record.get("seed"): record for record in baseline.get("seed_records", [])
    }
    for record in records:
        if not _verify_run_checkpoint(
            root, record.get("checkpoint"), record.get("checkpoint_sha256")
        ):
            return False, "Extension checkpoint authority mismatch."
        if (
            not _valid_extension_training_identity(extension, record)
            or not _verify_training_identity_envelope(root, record)
            or not _verify_internal_checkpoint_authority(root, record)
        ):
            return False, "Extension checkpoint identity mismatch."
        baseline_record = baseline_by_seed.get(record.get("seed"), {})
        baseline_source_digest = baseline_record.get("methods", {}).get(
            "DTL-pCSI", {}
        ).get("identity", {}).get("source_model_digest")
        if record["identity"].get("source_model_digest") != baseline_source_digest:
            return False, "Extension source-model linkage mismatch."
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        return False, "Completed extension run manifest is missing."
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "run_manifest"
        or manifest.get("status") != "complete"
        or manifest.get("summary") != "extension_summary.json"
        or manifest.get("summary_payload_sha256")
        != result_payload_digest(extension)
    ):
        return False, "Completed extension run manifest authority mismatch."
    fields = (
        "protocol",
        "profile_digest",
        "physical_configuration_fingerprint",
        "source_tree_digest",
        "training_seed_protocol",
        "evaluation_seed_protocol",
        "power_hypothesis_id",
        "baseline_payload_sha256",
        "baseline_validation_sha256",
        "objective",
        "seeds",
        "episodes",
        "evaluation_episodes",
        *AMBIGUITY_RESULT_FIELDS,
    )
    mismatches = [
        field for field in fields if manifest.get(field) != extension.get(field)
    ]
    if mismatches:
        return False, (
            "Extension manifest and summary mismatch: " + ", ".join(mismatches)
        )
    return True, None


def _validate_raw_fig4_histories(result, methods):
    seeds = result.get("seeds", [])
    records = result.get("seed_records", [])
    episodes = int(result.get("episodes", 0))
    window = int(result.get("training_final_window", 0))
    if (
        not isinstance(seeds, list)
        or len(seeds) < 5
        or len(set(seeds)) != len(seeds)
        or not isinstance(records, list)
        or len(records) != len(seeds)
        or episodes != 1500
        or window != 100
    ):
        return False
    records_by_seed = {record.get("seed"): record for record in records}
    if set(records_by_seed) != set(seeds):
        return False
    for method in METHOD_ORDER:
        try:
            declared = _seed_values(methods[method]["seed_values"], method)
        except (KeyError, TypeError, ValueError):
            return False
        calculated = []
        for seed in seeds:
            try:
                history = np.asarray(
                    records_by_seed[seed]["methods"][method]["training"]["comm_rates"],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError):
                return False
            if (
                history.shape != (episodes,)
                or not np.all(np.isfinite(history))
            ):
                return False
            calculated.append(float(np.mean(history[-window:])))
        if not np.allclose(declared, calculated, rtol=1e-12, atol=1e-12):
            return False
    return True


def _valid_unpenalized_baseline_rewards(result):
    """Require the page-7 reward to equal communication sum rate per episode."""
    episodes = int(result.get("episodes", 0))
    records = result.get("seed_records", [])
    if episodes <= 0 or not isinstance(records, list) or not records:
        return False
    for record in records:
        methods = record.get("methods", {})
        if set(methods) != set(METHOD_ORDER):
            return False
        for method in METHOD_ORDER:
            try:
                training = methods[method]["training"]
                rewards = np.asarray(training["rewards"], dtype=float)
                communication = np.asarray(training["comm_rates"], dtype=float)
            except (KeyError, TypeError, ValueError):
                return False
            if (
                rewards.shape != (episodes,)
                or communication.shape != (episodes,)
                or not np.all(np.isfinite(rewards))
                or not np.all(np.isfinite(communication))
                or not np.allclose(
                    rewards, communication, rtol=0.0, atol=1e-12
                )
            ):
                return False
    return True


def _validate_training_seed_records(result):
    """Require order-independent deterministic seeds across target methods."""
    seeds = result.get("seeds", [])
    records = result.get("seed_records", [])
    if result.get("training_seed_protocol") != TRAINING_SEED_PROTOCOL:
        return False
    if not isinstance(records, list) or len(records) != len(seeds):
        return False
    records_by_seed = {record.get("seed"): record for record in records}
    if set(records_by_seed) != set(seeds):
        return False
    for seed in seeds:
        record = records_by_seed[seed]
        source_seed = derive_training_seed(seed, "source")
        target_seed = derive_training_seed(seed, "target")
        if (
            record.get("source_training_seed") != source_seed
            or record.get("target_training_seed") != target_seed
        ):
            return False
        methods = record.get("methods", {})
        if set(methods) != set(METHOD_ORDER):
            return False
        if any(method.get("training_seed") != target_seed for method in methods.values()):
            return False
    return True


def _valid_projection_evaluation(evaluation, expected_episodes):
    """Require raw repair-dependence arrays and exact aggregate means."""
    if not isinstance(evaluation, dict) or int(expected_episodes) <= 0:
        return False
    arrays = {}
    for key in PROJECTION_EVIDENCE_KEYS:
        try:
            values = np.asarray(evaluation[key], dtype=float)
            declared_mean = float(evaluation[f"{key}_mean"])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            values.shape != (int(expected_episodes),)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
            or not np.isclose(
                declared_mean, float(np.mean(values)), rtol=0.0, atol=1e-12
            )
        ):
            return False
        arrays[key] = values
    return bool(
        np.all(
            arrays["deterministic_recovery_rate"]
            <= arrays["action_projection_rate"] + 1e-12
        )
        and np.all(
            1.0 - arrays["mean_projection_fraction"]
            <= arrays["action_projection_rate"] + 1e-12
        )
    )


def _valid_raw_evaluation(evaluation, expected_episodes, objective):
    """Reconstruct every evaluation aggregate from raw episode evidence."""
    if not _valid_projection_evaluation(evaluation, expected_episodes):
        return False
    expected = int(expected_episodes)
    try:
        arrays = {
            key: np.asarray(evaluation[key], dtype=float)
            for key in (
                "communication", "sensing", "objective", "feasibility",
                "minimum_ue_rates", "minimum_target_rates",
            )
        }
        specification = ObjectiveSpec.from_mapping(objective)
    except (KeyError, TypeError, ValueError):
        return False
    if any(
        values.shape != (expected,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        for values in arrays.values()
    ):
        return False
    if np.any(arrays["feasibility"] > 1.0):
        return False
    declared = {
        "communication": evaluation.get("communication_mean"),
        "sensing": evaluation.get("sensing_mean"),
        "objective": evaluation.get("objective_mean"),
        "feasibility": evaluation.get("feasibility_rate"),
        "minimum_ue_rates": evaluation.get("minimum_ue_rate"),
        "minimum_target_rates": evaluation.get("minimum_target_rate"),
    }
    for key, value in declared.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        reducer = np.min if key.startswith("minimum_") else np.mean
        if not np.isclose(
            value, float(reducer(arrays[key])), rtol=0.0, atol=1e-12
        ):
            return False

    if specification.mode == ObjectiveMode.PAPER_BASELINE:
        expected_objective = arrays["communication"]
    elif specification.mode == ObjectiveMode.PNG_JOINT_SUM:
        expected_objective = arrays["communication"] + arrays["sensing"]
    elif specification.mode == ObjectiveMode.COMM_RETENTION_SENSING:
        expected_objective = arrays["sensing"]
    elif specification.mode == ObjectiveMode.NORMALIZED_PARETO:
        expected_objective = (
            (1.0 - specification.pareto_weight)
            * arrays["communication"] / specification.communication_reference
            + specification.pareto_weight
            * arrays["sensing"] / specification.sensing_reference
        )
    else:  # pragma: no cover - ObjectiveMode construction is exhaustive.
        return False
    return bool(np.allclose(
        arrays["objective"], expected_objective, rtol=0.0, atol=1e-12
    ))


def _valid_baseline_raw_evaluations(result):
    seeds = result.get("seeds", [])
    records = result.get("seed_records", [])
    expected = int(result.get("evaluation_episodes", 0))
    if (
        not isinstance(seeds, list)
        or not isinstance(records, list)
        or [record.get("seed") for record in records] != seeds
    ):
        return False
    all_evaluations = []
    primary = []
    for record in records:
        methods = record.get("methods", {})
        if set(methods) != set(METHOD_ORDER):
            return False
        for method in METHOD_ORDER:
            evaluation = methods[method].get("evaluation")
            if not _valid_raw_evaluation(
                evaluation, expected, result.get("objective", {})
            ):
                return False
            all_evaluations.append(evaluation)
        primary.append(methods["DTL-pCSI"]["evaluation"])
    try:
        communication = np.asarray(
            result["communication_seed_values"], dtype=float
        )
        sensing = np.asarray(result["sensing_seed_values"], dtype=float)
        reconstructed_communication = np.asarray([
            evaluation["communication_mean"] for evaluation in primary
        ], dtype=float)
        reconstructed_sensing = np.asarray([
            evaluation["sensing_mean"] for evaluation in primary
        ], dtype=float)
        reconstructed_feasibility = min(
            float(evaluation["feasibility_rate"])
            for evaluation in all_evaluations
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        communication.shape == (len(seeds),)
        and sensing.shape == (len(seeds),)
        and np.allclose(
            communication, reconstructed_communication,
            rtol=0.0, atol=1e-12,
        )
        and np.allclose(
            sensing, reconstructed_sensing, rtol=0.0, atol=1e-12
        )
        and np.isclose(
            float(result.get("feasibility_rate", -1.0)),
            reconstructed_feasibility,
            rtol=0.0,
            atol=1e-12,
        )
    )


def _valid_extension_raw_evaluations(extension):
    seeds = extension.get("seeds", [])
    records = extension.get("seed_records", [])
    expected = int(extension.get("evaluation_episodes", 0))
    if (
        not isinstance(seeds, list)
        or not isinstance(records, list)
        or [record.get("seed") for record in records] != seeds
    ):
        return False
    summary_objective = extension.get("objective", {})
    evaluations = []
    for record in records:
        record_objective = record.get("objective", {})
        if any(
            record_objective.get(field) != value
            for field, value in summary_objective.items()
        ):
            return False
        evaluation = record.get("evaluation")
        if not _valid_raw_evaluation(evaluation, expected, record_objective):
            return False
        evaluations.append(evaluation)
    try:
        reconstructed = {
            "communication_seed_values": np.asarray([
                row["communication_mean"] for row in evaluations
            ], dtype=float),
            "sensing_seed_values": np.asarray([
                row["sensing_mean"] for row in evaluations
            ], dtype=float),
            "objective_seed_values": np.asarray([
                row["objective_mean"] for row in evaluations
            ], dtype=float),
            "minimum_ue_rate_seed_values": np.asarray([
                row["minimum_ue_rate"] for row in evaluations
            ], dtype=float),
            "minimum_target_rate_seed_values": np.asarray([
                row["minimum_target_rate"] for row in evaluations
            ], dtype=float),
        }
        reconstructed["joint_seed_values"] = (
            reconstructed["communication_seed_values"]
            + reconstructed["sensing_seed_values"]
        )
        summaries_match = all(
            np.allclose(
                np.asarray(extension[field], dtype=float),
                values,
                rtol=0.0,
                atol=1e-12,
            )
            for field, values in reconstructed.items()
        )
        feasibility = min(
            float(evaluation["feasibility_rate"])
            for evaluation in evaluations
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        summaries_match
        and np.isclose(
            float(extension.get("feasibility_rate", -1.0)),
            feasibility,
            rtol=0.0,
            atol=1e-12,
        )
    )


def _valid_extension_objective_references(extension, baseline):
    """Require extension normalization/floors to come from training evidence."""
    try:
        window = int(extension["objective_reference_window"])
        if (
            extension.get("objective_reference_protocol")
            != "accepted_baseline_training_tail_v1"
            or window != 100
        ):
            return False
        extension_records = extension["seed_records"]
        baseline_records = baseline["seed_records"]
        if (
            len(extension_records) != len(baseline_records)
            or len(extension_records) < 5
        ):
            return False
        for extension_record, baseline_record in zip(
                extension_records, baseline_records):
            if extension_record.get("seed") != baseline_record.get("seed"):
                return False
            training = baseline_record["methods"]["DTL-pCSI"]["training"]
            communication = np.asarray(training["comm_rates"], dtype=float)
            sensing = np.asarray(training["sensing_rates"], dtype=float)
            if (
                communication.ndim != 1
                or sensing.ndim != 1
                or communication.size < window
                or sensing.size < window
                or not np.all(np.isfinite(communication))
                or not np.all(np.isfinite(sensing))
            ):
                return False
            comm_reference = float(np.mean(communication[-window:]))
            sensing_reference = float(np.mean(sensing[-window:]))
            declared_reference = extension_record.get("objective_reference", {})
            if (
                declared_reference.get("source")
                != "accepted_baseline_training_tail"
                or declared_reference.get("window") != window
                or not np.isclose(
                    float(declared_reference.get("communication", np.nan)),
                    comm_reference,
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.isclose(
                    float(declared_reference.get("sensing", np.nan)),
                    sensing_reference,
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                return False
            objective = extension_record.get("objective", {})
            mode = objective.get("mode")
            if mode == "comm_retention_sensing":
                if not np.isclose(
                    float(objective.get("communication_reference", np.nan)),
                    comm_reference,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    return False
            elif mode == "normalized_pareto":
                if not (
                    np.isclose(
                        float(objective.get("communication_reference", np.nan)),
                        comm_reference,
                        rtol=0.0,
                        atol=1e-12,
                    )
                    and np.isclose(
                        float(objective.get("sensing_reference", np.nan)),
                        sensing_reference,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ):
                    return False
            elif mode != "png_joint_sum":
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _valid_baseline_projection_evidence(result):
    records = result.get("seed_records", [])
    expected = int(result.get("evaluation_episodes", 0))
    return bool(records) and all(
        set(record.get("methods", {})) == set(METHOD_ORDER)
        and all(
            _valid_projection_evaluation(
                record["methods"][method].get("evaluation"), expected
            )
            for method in METHOD_ORDER
        )
        for record in records
    )


def _valid_baseline_projection_attribution(result):
    return bool(result.get("seed_records")) and all(
        projection_attribution_passes(record["methods"][method]["evaluation"])
        for record in result["seed_records"]
        for method in METHOD_ORDER
    )


def _valid_extension_projection_evidence(extension):
    records = extension.get("seed_records", [])
    expected = int(extension.get("evaluation_episodes", 0))
    return bool(records) and all(
        _valid_projection_evaluation(record.get("evaluation"), expected)
        for record in records
    )


def _valid_extension_projection_attribution(extension):
    return bool(extension.get("seed_records")) and all(
        projection_attribution_passes(record["evaluation"])
        for record in extension["seed_records"]
    )


def _causal_smooth(values, window):
    values = np.asarray(values, dtype=float)
    cumulative = np.cumsum(values, dtype=float)
    result = cumulative.copy()
    result[window:] -= cumulative[:-window]
    divisors = np.minimum(np.arange(1, values.size + 1), window)
    return result / divisors


def _validate_fig4_full_curves(result, reference, relative_tolerance):
    """Recompute every published Figure 4 marker from raw seed histories."""
    protocol = result.get("figure4_protocol", {})
    interval = int(protocol.get("plot_interval", 0))
    window = int(protocol.get("acceptance_smoothing_window", 0))
    episodes = int(result.get("episodes", 0))
    expected_x = np.arange(0, episodes, interval) if interval > 0 else np.array([])
    reference_x = np.asarray(reference.get("x", []), dtype=float)
    if (
        window <= 0
        or expected_x.shape != reference_x.shape
        or not np.allclose(expected_x, reference_x, rtol=0.0, atol=1e-12)
    ):
        return _check(False, reason="reference/protocol x-grid mismatch")
    records = {record.get("seed"): record for record in result.get("seed_records", [])}
    seeds = result.get("seeds", [])
    uncertainty = float(reference["uncertainty"]["absolute_y"])
    details = {}
    passed = True
    for method in METHOD_ORDER:
        try:
            raw = np.stack([
                np.asarray(
                    records[seed]["methods"][method]["training"]["comm_rates"],
                    dtype=float,
                )
                for seed in seeds
            ])
            if raw.shape != (len(seeds), episodes) or not np.all(np.isfinite(raw)):
                raise ValueError("raw Figure 4 histories have invalid shape or values")
            generated = np.mean(
                np.stack([_causal_smooth(history, window) for history in raw]),
                axis=0,
            )[expected_x]
            target = np.asarray(reference["curves"][method], dtype=float)
        except (IndexError, KeyError, TypeError, ValueError):
            passed = False
            details[method] = {"pass": False, "reason": "missing raw/reference curve"}
            continue
        if target.shape != generated.shape or not np.all(np.isfinite(target)):
            passed = False
            details[method] = {"pass": False, "reason": "curve shape mismatch"}
            continue
        tolerance = np.maximum(
            float(relative_tolerance) * np.abs(target), 3.0 * uncertainty
        )
        errors = np.abs(generated - target)
        curve_pass = bool(np.all(errors <= tolerance))
        passed = passed and curve_pass
        details[method] = {
            "pass": curve_pass,
            "max_absolute_error": float(np.max(errors)),
            "max_allowed_error": float(np.max(tolerance)),
        }
    return _check(
        passed,
        smoothing="causal_moving_average",
        smoothing_window=window,
        curves=details,
    )


def validate_baseline_integrity(result, *, result_directory) -> dict:
    """Validate cached baseline structure and file authority without curve fit."""
    checks = {
        "generated_baseline": _check(
            result.get("kind") == "generated_result"
            and result.get("protocol") == "paper_baseline"
            and result.get("objective", {}).get("mode") == "paper_baseline"
        ),
        "ambiguity_contract": _check(_valid_ambiguity_contract(result)),
        "training_randomness_protocol": _check(
            _validate_training_seed_records(result)
        ),
        "evaluation_scene_protocol": _check(
            _valid_evaluation_seed_schedule(result)
        ),
        "unpenalized_reward": _check(
            _valid_unpenalized_baseline_rewards(result)
        ),
        "raw_training_histories": _check(
            _validate_raw_fig4_histories(
                result, result.get("final_reward_by_method", {})
            )
        ),
        "raw_evaluations": _check(_valid_baseline_raw_evaluations(result)),
        "projection_evidence": _check(
            _valid_baseline_projection_evidence(result)
        ),
        "actor_action_attribution": _check(
            _valid_baseline_projection_attribution(result),
            thresholds=PROJECTION_ATTRIBUTION_THRESHOLDS,
        ),
    }
    try:
        passed, error = _validate_baseline_file_provenance(
            result, Path(result_directory)
        )
        checks["file_and_checkpoint_provenance"] = _check(
            passed, error=error
        )
    except (OSError, TypeError, ValueError) as error:
        checks["file_and_checkpoint_provenance"] = _check(
            False, error=str(error)
        )
    return {
        "overall_pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
    }


def _valid_smoke_training_histories(result) -> bool:
    """Reconstruct smoke tails without applying Figure 4 acceptance rules."""
    try:
        episodes = int(result.get("episodes", 0))
        window = int(result.get("training_final_window", 0))
        seeds = result.get("seeds")
        records = result.get("seed_records")
        final = result.get("final_reward_by_method")
    except (TypeError, ValueError):
        return False
    if (
        episodes not in {100, 200}
        or window != min(100, episodes)
        or not isinstance(seeds, list)
        or not seeds
        or len(seeds) != len(set(seeds))
        or not isinstance(records, list)
        or [record.get("seed") for record in records] != seeds
        or not isinstance(final, dict)
        or set(final) != set(METHOD_ORDER)
    ):
        return False
    reconstructed = {method: [] for method in METHOD_ORDER}
    for record in records:
        methods = record.get("methods", {})
        if set(methods) != set(METHOD_ORDER):
            return False
        for method in METHOD_ORDER:
            try:
                training = methods[method]["training"]
                arrays = {
                    key: np.asarray(training[key], dtype=float)
                    for key in (
                        "rewards", "comm_rates", "sensing_rates",
                        "joint_rates", "feasibility_rates",
                    )
                }
            except (KeyError, TypeError, ValueError):
                return False
            if any(
                values.shape != (episodes,)
                or not np.all(np.isfinite(values))
                for values in arrays.values()
            ):
                return False
            if (
                not np.allclose(
                    arrays["rewards"], arrays["comm_rates"],
                    rtol=0.0, atol=1e-12,
                )
                or not np.allclose(
                    arrays["joint_rates"],
                    arrays["comm_rates"] + arrays["sensing_rates"],
                    rtol=0.0, atol=1e-12,
                )
                or not np.allclose(
                    arrays["feasibility_rates"], 1.0,
                    rtol=0.0, atol=1e-12,
                )
            ):
                return False
            reconstructed[method].append(
                float(np.mean(arrays["comm_rates"][-window:]))
            )
    for method, values in reconstructed.items():
        try:
            declared = np.asarray(final[method]["seed_values"], dtype=float)
        except (KeyError, TypeError, ValueError):
            return False
        if (
            declared.shape != (len(seeds),)
            or not np.all(np.isfinite(declared))
            or not np.allclose(
                declared, np.asarray(values), rtol=1e-12, atol=1e-12
            )
        ):
            return False
    return True


def validate_smoke_result(
    result: dict,
    *,
    result_directory=None,
    verify_files: bool = True,
) -> dict:
    """Validate early-run integrity without making a paper-curve claim."""
    diagnostic_scope = result.get("diagnostic_scope")
    checks = {
        "smoke_protocol": _check(
            result.get("kind") == "generated_result"
            and result.get("protocol") == "paper_baseline_smoke"
            and result.get("run_mode") == "smoke"
            and result.get("publication_eligible") is False
            and result.get("figure4_protocol") is None
            and int(result.get("episodes", 0)) in {100, 200}
        ),
        "diagnostic_only_scope": _check(
            diagnostic_scope == {
                "purpose": "early_scale_feasibility_projection_checkpoint_check",
                "not_evidence_for_paper_curve_acceptance": True,
            }
        ),
        "paper_objective": _check(
            result.get("objective", {}).get("mode") == "paper_baseline"
        ),
        "source_provenance": _check(
            _is_sha256(result.get("source_tree_digest"))
            and _is_sha256(result.get("profile_digest"))
            and _is_sha256(result.get("physical_configuration_fingerprint"))
            and result.get("power_hypothesis_id")
            in {"paper_db_as_dbm", "paper_db_as_dbw"}
        ),
        "ambiguity_contract": _check(_valid_ambiguity_contract(result)),
        "training_randomness_protocol": _check(
            _validate_training_seed_records(result)
        ),
        "evaluation_scene_protocol": _check(
            _valid_evaluation_seed_schedule(result)
        ),
        "raw_training_histories": _check(
            _valid_smoke_training_histories(result)
        ),
        "paper_reward_equals_communication_rate": _check(
            _valid_unpenalized_baseline_rewards(result)
        ),
        "physical_feasibility": _check(
            np.isclose(float(result.get("feasibility_rate", -1.0)), 1.0)
        ),
        "raw_evaluation_consistency": _check(
            _valid_baseline_raw_evaluations(result)
        ),
        "projection_dependence_evidence": _check(
            _valid_baseline_projection_evidence(result)
        ),
        "actor_action_attribution": _check(
            _valid_baseline_projection_attribution(result),
            thresholds=PROJECTION_ATTRIBUTION_THRESHOLDS,
        ),
    }
    if verify_files:
        if result_directory is None:
            checks["file_and_checkpoint_provenance"] = _check(
                False, error="result_directory is required for file verification."
            )
        else:
            try:
                passed, error = _validate_baseline_file_provenance(
                    result, Path(result_directory)
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
    return {
        "schema_version": 1,
        "validation": "paper_baseline_smoke_diagnostic",
        "baseline_payload_sha256": result_payload_digest(result),
        "advisory_only": True,
        "publication_eligible": False,
        "paper_curve_acceptance_performed": False,
        "overall_pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
    }


def validate_baseline_result(
    result: dict,
    references: dict,
    relative_tolerance: float = 0.10,
    *,
    result_directory=None,
    verify_files: bool = True,
) -> dict:
    """Validate generated Fig. 4 endpoints without changing any arrays."""
    references = load_paper_references(references)
    reference_digest = result_payload_digest(references)
    checks = {
        "generated_provenance": _check(result.get("kind") == "generated_result"),
        "paper_objective": _check(
            result.get("objective", {}).get("mode") == "paper_baseline"
        ),
        "physical_feasibility": _check(
            np.isclose(float(result.get("feasibility_rate", -1.0)), 1.0)
        ),
        "training_reward_endpoint": _check(
            result.get("final_reward_source") == "training_communication_rate"
            and int(result.get("training_final_window", 0)) > 0
        ),
        "publication_seeds": _check(
            isinstance(result.get("seeds"), list)
            and len(result.get("seeds")) >= 5
            and len(set(result.get("seeds"))) == len(result.get("seeds"))
        ),
        "source_provenance": _check(
            _is_sha256(result.get("source_tree_digest"))
            and result.get("power_hypothesis_id")
            in {"paper_db_as_dbm", "paper_db_as_dbw"}
        ),
        "ambiguity_contract": _check(_valid_ambiguity_contract(result)),
        "training_randomness_protocol": _check(
            _validate_training_seed_records(result)
        ),
        "evaluation_scene_protocol": _check(
            _valid_evaluation_seed_schedule(result)
        ),
        "projection_dependence_evidence": _check(
            _valid_baseline_projection_evidence(result)
        ),
        "actor_action_attribution": _check(
            _valid_baseline_projection_attribution(result),
            thresholds=PROJECTION_ATTRIBUTION_THRESHOLDS,
        ),
        "raw_evaluation_consistency": _check(
            _valid_baseline_raw_evaluations(result)
        ),
        "paper_reward_equals_communication_rate": _check(
            _valid_unpenalized_baseline_rewards(result)
        ),
    }
    if verify_files:
        if result_directory is None:
            checks["file_and_checkpoint_provenance"] = _check(
                False, error="result_directory is required for file verification."
            )
        else:
            try:
                passed, error = _validate_baseline_file_provenance(
                    result, Path(result_directory)
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
    protocol = result.get("figure4_protocol", {})
    checks["figure4_protocol"] = _check(
        int(result.get("episodes", 0)) == 1500
        and protocol == {
            "raw_episode_count": 1500,
            "plot_x_start": 0,
            "plot_x_end": 1400,
            "plot_interval": 100,
            "acceptance_smoothing_window": 20,
        }
    )
    reference = references["figures"]["4"]
    uncertainty = float(reference["uncertainty"]["absolute_y"])
    methods = result.get("final_reward_by_method", {})
    checks["raw_history_consistency"] = _check(
        _validate_raw_fig4_histories(result, methods)
    )
    checks["paper_full_curve_tolerance"] = _validate_fig4_full_curves(
        result, reference, relative_tolerance
    )
    statistics = {}
    closeness = True
    for method in METHOD_ORDER:
        if method not in methods or method not in reference["curves"]:
            closeness = False
            continue
        stats = _summary(methods[method]["seed_values"])
        target = float(reference["curves"][method][-1])
        tolerance = max(relative_tolerance * abs(target), 3.0 * uncertainty)
        stats.update({
            "paper_reference": target,
            "absolute_tolerance": tolerance,
            "within_tolerance": abs(stats["mean"] - target) <= tolerance,
        })
        statistics[method] = stats
        closeness = closeness and stats["within_tolerance"]
    checks["paper_endpoint_tolerance"] = _check(closeness)

    means = [statistics.get(method, {}).get("mean", -np.inf)
             for method in METHOD_ORDER]
    paired_ordering = {}
    ordering = True
    for upper, lower in zip(METHOD_ORDER, METHOD_ORDER[1:]):
        try:
            difference = _seed_values(
                methods[upper]["seed_values"], upper
            ) - _seed_values(methods[lower]["seed_values"], lower)
            stats = _summary(difference)
            passed = stats["ci95_low"] > 0.0
            paired_ordering[f"{upper}>{lower}"] = {**stats, "pass": passed}
            ordering = ordering and passed
        except (KeyError, TypeError, ValueError):
            ordering = False
    checks["method_ordering"] = _check(
        ordering,
        expected=list(METHOD_ORDER),
        means=means,
        paired_confidence_intervals=paired_ordering,
    )
    return {
        "schema_version": 1,
        "validation": "paper_baseline",
        "baseline_payload_sha256": result_payload_digest(result),
        "paper_reference_sha256": reference_digest,
        "overall_pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
        "statistics": statistics,
    }


def validate_extension_result(
    extension: dict,
    baseline: dict,
    *,
    baseline_validation: dict | None = None,
    result_directory=None,
    verify_files: bool = True,
) -> dict:
    """Require paired communication retention and positive sensing gain."""
    try:
        baseline_digest = result_payload_digest(baseline)
        validation_digest = result_payload_digest(baseline_validation)
        accepted_baseline = (
            baseline_validation.get("validation") == "paper_baseline"
            and baseline_validation.get("overall_pass") is True
            and baseline_validation.get("baseline_payload_sha256")
            == baseline_digest
            and extension.get("baseline_payload_sha256") == baseline_digest
            and extension.get("baseline_validation_sha256")
            == validation_digest
        )
        baseline_authority_error = None
    except (AttributeError, TypeError, ValueError) as error:
        baseline_digest = result_payload_digest(baseline)
        validation_digest = None
        accepted_baseline = False
        baseline_authority_error = str(error)
    checks = {
        "accepted_baseline_authority": _check(
            accepted_baseline, error=baseline_authority_error
        ),
        "generated_provenance": _check(
            extension.get("kind") == "generated_result"
            and baseline.get("kind") == "generated_result"
        ),
        "same_physical_configuration": _check(
            extension.get("physical_configuration_fingerprint")
            == baseline.get("physical_configuration_fingerprint")
        ),
        "same_profile": _check(
            extension.get("profile_digest") == baseline.get("profile_digest")
        ),
        "same_source_code": _check(
            _is_sha256(extension.get("source_tree_digest"))
            and extension.get("source_tree_digest")
            == baseline.get("source_tree_digest")
        ),
        "same_power_hypothesis": _check(
            extension.get("power_hypothesis_id")
            == baseline.get("power_hypothesis_id")
            and extension.get("power_hypothesis_id")
            in {"paper_db_as_dbm", "paper_db_as_dbw"}
        ),
        "same_ambiguity_contract": _check(
            _valid_ambiguity_contract(baseline)
            and _valid_ambiguity_contract(extension)
            and all(
                extension.get(field) == baseline.get(field)
                for field in AMBIGUITY_RESULT_FIELDS
            )
        ),
        "paired_seeds": _check(
            extension.get("seeds") == baseline.get("seeds")
            and isinstance(extension.get("seeds"), list)
            and len(extension.get("seeds")) >= 5
        ),
        "same_training_randomness": _check(
            extension.get("training_seed_protocol") == TRAINING_SEED_PROTOCOL
            and baseline.get("training_seed_protocol") == TRAINING_SEED_PROTOCOL
            and all(
                extension_record.get("seed") == baseline_record.get("seed")
                and extension_record.get("training_seed")
                == baseline_record.get("target_training_seed")
                for extension_record, baseline_record in zip(
                    extension.get("seed_records", []),
                    baseline.get("seed_records", []),
                )
            )
            and len(extension.get("seed_records", []))
            == len(baseline.get("seed_records", []))
            >= 5
        ),
        "paired_evaluation_scenes": _check(
            _valid_paired_evaluation_seed_schedule(extension, baseline)
        ),
        "physical_feasibility": _check(
            np.isclose(float(extension.get("feasibility_rate", -1.0)), 1.0)
        ),
        "projection_dependence_evidence": _check(
            _valid_extension_projection_evidence(extension)
        ),
        "actor_action_attribution": _check(
            _valid_extension_projection_attribution(extension),
            thresholds=PROJECTION_ATTRIBUTION_THRESHOLDS,
        ),
        "raw_evaluation_consistency": _check(
            _valid_extension_raw_evaluations(extension)
        ),
        "objective_reference_has_no_evaluation_leakage": _check(
            _valid_extension_objective_references(extension, baseline)
        ),
    }
    if verify_files:
        if result_directory is None:
            checks["file_and_checkpoint_provenance"] = _check(
                False, error="result_directory is required for file verification."
            )
        else:
            try:
                passed, error = _validate_extension_file_provenance(
                    extension, Path(result_directory), baseline
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
    objective = extension.get("objective", {})
    mode = objective.get("mode")
    checks["extension_objective"] = _check(
        mode in {"comm_retention_sensing", "png_joint_sum", "normalized_pareto"}
        and (
            mode == "comm_retention_sensing"
            or "communication_retention_fraction" not in objective
        )
    )

    baseline_comm = _seed_values(
        baseline.get("communication_seed_values", []), "baseline communication"
    )
    extension_comm = _seed_values(
        extension.get("communication_seed_values", []), "extension communication"
    )
    baseline_sensing = _seed_values(
        baseline.get("sensing_seed_values", []), "baseline sensing"
    )
    extension_sensing = _seed_values(
        extension.get("sensing_seed_values", []), "extension sensing"
    )
    lengths = {
        baseline_comm.size, extension_comm.size,
        baseline_sensing.size, extension_sensing.size,
    }
    if len(lengths) != 1:
        raise ValueError("Baseline and extension require paired seed arrays.")

    statistics = {}
    if mode == "comm_retention_sensing":
        retention = float(objective.get("communication_retention_fraction", 0.0))
        if not 0.0 < retention <= 1.0:
            raise ValueError("Extension acceptance has an invalid retention fraction.")
        retention_difference = extension_comm - retention * baseline_comm
        sensing_gain = extension_sensing - baseline_sensing
        retention_stats = _summary(retention_difference)
        sensing_stats = _summary(sensing_gain)
        checks["communication_retention"] = _check(
            retention_stats["ci95_low"] >= 0.0,
            required_fraction=retention,
        )
        checks["positive_sensing_gain"] = _check(
            sensing_stats["ci95_low"] > 0.0
        )
        statistics.update({
            "communication_retention_margin": retention_stats,
            "sensing_gain": sensing_stats,
        })
    elif mode == "png_joint_sum":
        joint = _seed_values(
            extension.get("joint_seed_values", []), "extension joint rate"
        )
        minimum_ue = _seed_values(
            extension.get("minimum_ue_rate_seed_values", []),
            "extension minimum UE rate",
        )
        minimum_target = _seed_values(
            extension.get("minimum_target_rate_seed_values", []),
            "extension minimum target rate",
        )
        if not (
            len(joint) == len(extension_comm) == len(minimum_ue)
            == len(minimum_target)
        ):
            raise ValueError("PNG extension arrays must use the same paired seeds.")
        checks["exact_png_joint_rate"] = _check(
            np.allclose(joint, extension_comm + extension_sensing,
                        rtol=0.0, atol=1e-10)
        )
        user_floor = float(objective.get("user_rate_min", -1.0))
        target_floor = float(objective.get("target_rate_min", -1.0))
        checks["png_rate_floors"] = _check(
            user_floor > 0.0
            and target_floor > 0.0
            and np.all(minimum_ue >= user_floor)
            and np.all(minimum_target >= target_floor),
            user_rate_min=user_floor,
            target_rate_min=target_floor,
        )
        joint_gain = joint - (baseline_comm + baseline_sensing)
        joint_gain_stats = _summary(joint_gain)
        checks["positive_paired_joint_gain"] = _check(
            joint_gain_stats["ci95_low"] > 0.0
        )
        statistics["joint_rate"] = _summary(joint)
        statistics["paired_joint_gain"] = joint_gain_stats
    elif mode == "normalized_pareto":
        objective_values = _seed_values(
            extension.get("objective_seed_values", []),
            "normalized Pareto objective",
        )
        weight = float(objective.get("pareto_weight", np.nan))
        references = extension.get("seed_records", [])
        communication_references = np.asarray([
            row.get("objective_reference", {}).get("communication", np.nan)
            for row in references
        ], dtype=float)
        sensing_references = np.asarray([
            row.get("objective_reference", {}).get("sensing", np.nan)
            for row in references
        ], dtype=float)
        reference_error = None
        try:
            paired_baseline_objective = paired_normalized_baseline_objective(
                baseline_comm,
                baseline_sensing,
                communication_references,
                sensing_references,
                weight,
            )
        except ValueError as error:
            reference_valid = False
            reference_error = str(error)
            paired_baseline_objective = None
        else:
            reference_valid = True
        checks["normalized_pareto_metrics"] = _check(
            len(objective_values) == len(extension_comm)
            and reference_valid,
            reference_error=reference_error,
        )
        if reference_valid:
            objective_gain = objective_values - paired_baseline_objective
            objective_gain_stats = _summary(objective_gain)
            checks["positive_normalized_pareto_gain"] = _check(
                objective_gain_stats["ci95_low"] > 0.0
            )
            statistics["paired_baseline_normalized_pareto_objective"] = (
                _summary(paired_baseline_objective)
            )
            statistics["normalized_pareto_gain_over_baseline"] = (
                objective_gain_stats
            )
        else:
            checks["positive_normalized_pareto_gain"] = _check(
                False, reference_error=reference_error
            )
            statistics["paired_baseline_normalized_pareto_objective"] = {
                "error": reference_error
            }
            statistics["normalized_pareto_gain_over_baseline"] = {
                "error": reference_error
            }
        statistics["normalized_pareto_objective"] = _summary(objective_values)
    else:
        checks["recognized_extension_mode"] = _check(False)
    return {
        "schema_version": 1,
        "validation": "sensing_extension",
        "baseline_payload_sha256": baseline_digest,
        "baseline_validation_sha256": validation_digest,
        "extension_payload_sha256": result_payload_digest(extension),
        "overall_pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
        "statistics": statistics,
    }


def _read(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--extension")
    parser.add_argument("--baseline-validation")
    parser.add_argument(
        "--mode", choices=("publication", "smoke"), default="publication"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    baseline = _read(args.baseline)
    if args.extension:
        if args.mode != "publication":
            parser.error("--mode smoke cannot validate an extension")
        if not args.baseline_validation:
            parser.error("--baseline-validation is required with --extension")
        baseline_validation = _read(args.baseline_validation)
        baseline_directory = Path(args.baseline).resolve().parent
        require_canonical_baseline_validation(
            baseline,
            baseline_validation,
            baseline_directory,
        )
        report = validate_extension_result(
            _read(args.extension),
            baseline,
            baseline_validation=baseline_validation,
            result_directory=Path(args.extension).resolve().parent,
            verify_files=True,
        )
    elif args.mode == "smoke":
        report = validate_smoke_result(
            baseline,
            result_directory=Path(args.baseline).resolve().parent,
            verify_files=True,
        )
    else:
        report = validate_baseline_result(
            baseline,
            require_canonical_paper_references(args.reference),
            result_directory=Path(args.baseline).resolve().parent,
            verify_files=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if args.mode == "smoke":
        return 0
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
