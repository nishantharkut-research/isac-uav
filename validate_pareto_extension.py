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

from pareto_protocol import (
    ParetoAnchors,
    holm_adjust,
    hypervolume_2d,
    paired_confidence_interval,
    validate_preferences,
)
from run_pareto_extension import select_gate_configuration, validate_seed_roles
from validate_results import result_payload_digest


def _check(passed, **details) -> dict:
    return {"pass": bool(passed), **details}


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


def _validate_rate_record(record, expected_length) -> dict[str, np.ndarray]:
    keys = (
        "communication", "sensing", "joint", "objective", "minimum_ue_rate",
        "minimum_target_rate", "feasibility",
    )
    arrays = {key: _finite_array(record, key, expected_length) for key in keys}
    if not np.allclose(
        arrays["joint"], arrays["communication"] + arrays["sensing"],
        rtol=0.0, atol=1e-8,
    ):
        raise ValueError("Raw joint values do not equal communication plus sensing.")
    if not np.allclose(arrays["objective"], arrays["joint"], rtol=0.0, atol=1e-8):
        raise ValueError("PNG objective values do not equal the raw joint rate.")
    return arrays


def _grid_records(evaluation, preferences, scene_seeds) -> list[tuple[list, dict, dict]]:
    if evaluation.get("scene_seeds") != list(scene_seeds):
        raise ValueError("Evaluation scene seeds do not match the locked test partition.")
    rows = evaluation.get("preferences")
    if not isinstance(rows, list) or len(rows) != len(preferences):
        raise ValueError("Evaluation does not contain the complete preference grid.")
    result = []
    for expected, row in zip(preferences, rows):
        actual = validate_preferences([row.get("preference")])[0].tolist()
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-8):
            raise ValueError("Evaluation preference order or identity changed.")
        result.append((expected, row, _validate_rate_record(row, len(scene_seeds))))
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
        "preference_grid",
        "comparison_level",
        "hypervolume_reference_normalized",
        "gate_selection_lock_sha256",
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


def _verify_vector_checkpoint_identity(
    root: Path, summary: Mapping, record: Mapping
) -> bool:
    identity = record.get("identity", {})
    return (
        _verify_checkpoint_record(root, record)
        and identity.get("baseline_policy_digest")
        == record.get("baseline_policy_digest")
        and identity.get("source_tree_digest") == summary.get("source_tree_digest")
        and identity.get("objective") == summary.get("objective")
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
    ):
        raise ValueError("Gate-validation evidence authority fields mismatch.")
    anchors = ParetoAnchors.from_state_dict(summary["anchors"])
    preferences = validate_preferences(summary["preference_grid"]).tolist()
    reference = summary["hypervolume_reference_normalized"]
    validation_scenes = summary["validation_scene_seeds"]
    rows = payload.get("rows")
    candidates = summary.get("gate_selection", {}).get("candidates", [])
    expected_count = len(summary["research_seeds"]) * len(candidates)
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
        grid = _grid_records(row["raw"], preferences, validation_scenes)
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
        }
        if not _verify_checkpoint_record(root, source_record):
            return False, "Joint-objective source checkpoint hash mismatch."
        scalar_order = (
            "same_objective_dtl",
            "cross_objective_dtl_diagnostic",
            "ddpg_scratch",
        )
        if set(scalar) != set(scalar_order):
            return False, "Scalar comparator checkpoint set is incomplete."
        for method_name in scalar_order:
            method_record = scalar[method_name]
            if not _verify_checkpoint_record(root, method_record):
                return False, f"{method_name} checkpoint hash mismatch."
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
                root, summary, vector[method_name]
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
                        root, summary, policy
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

    if baseline is not None and baseline_validation is not None:
        baseline_authority = (
            baseline_validation.get("validation") == "paper_baseline"
            and baseline_validation.get("overall_pass") is True
            and baseline_validation.get("baseline_payload_sha256")
            == result_payload_digest(baseline)
            and summary.get("baseline_payload_sha256")
            == result_payload_digest(baseline)
            and summary.get("baseline_validation_sha256")
            == result_payload_digest(baseline_validation)
            and summary.get("profile_digest") == baseline.get("profile_digest")
            and summary.get("physical_configuration_fingerprint")
            == baseline.get("physical_configuration_fingerprint")
            and summary.get("source_tree_digest") == baseline.get("source_tree_digest")
        )
        checks["accepted_baseline_authority"] = _check(baseline_authority)
        checkpoint_authority_valid = False
        try:
            authority = summary.get("baseline_checkpoint_authority", [])
            baseline_records = baseline.get("seed_records", [])
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
            checks["accepted_baseline_checkpoints"] = _check(
                False, error=str(error)
            )

    try:
        roles = validate_seed_roles(
            summary.get("research_seeds", []),
            summary.get("validation_scene_seeds", []),
            summary.get("test_scene_seeds", []),
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
        reference = np.asarray(
            summary.get("hypervolume_reference_normalized"), dtype=np.float64
        )
        if reference.shape != (2,) or not np.all(np.isfinite(reference)):
            raise ValueError("Normalized hypervolume reference is invalid.")
        checks["training_only_anchors"] = _check(anchors_valid)
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
    except (TypeError, ValueError):
        gate_selection_matches_rule = False
    gate_valid = (
        gate_lock.get("selection_partition") == "validation_only"
        and gate_lock.get("validation_scene_seeds")
        == summary.get("validation_scene_seeds")
        and gate_lock.get("research_seeds") == summary.get("research_seeds")
        and _is_sha256(gate_lock.get("validation_evidence_sha256"))
        and gate_lock.get("validation_evidence_sha256")
        == summary.get("gate_validation_evidence_sha256")
        and math.isfinite(float(selected_gate.get("uncertainty_coefficient", math.nan)))
        and math.isfinite(float(selected_gate.get("margin", math.nan)))
        and isinstance(gate_candidates, list)
        and len(gate_candidates) > 0
        and selected_gate in gate_candidates
        and gate_selection_matches_rule
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
                rows = _grid_records(proposed_eval, preferences, scene_seeds)
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
                    proposed_no_gate, preferences, scene_seeds
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
                        scalar, len(scene_seeds)
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
                            ablation, preferences, scene_seeds
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
                    for (preference, _row, arrays), policy in zip(rows, policies):
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
                            policy.get("evaluation", {}), len(scene_seeds)
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
    report = validate_pareto_result(
        _read_json(summary_path),
        level=args.level,
        result_directory=summary_path.parent,
        verify_files=True,
        baseline=_read_json(args.baseline_summary),
        baseline_validation=_read_json(args.baseline_validation),
        baseline_directory=Path(args.baseline_summary).resolve().parent,
    )
    from experiment_protocol import atomic_write_json

    atomic_write_json(args.output, report)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
