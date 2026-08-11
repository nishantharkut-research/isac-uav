"""Read-only validation of PNG figures and machine-readable result evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ambiguity_hypotheses import valid_ambiguity_manifest_fields


BUNDLE_PROVENANCE_FIELDS = (
    "source_tree_digest",
    "profile_digest",
    "physical_configuration_fingerprint",
    "power_hypothesis_id",
    "noise_hypothesis_id",
    "dtl_transfer_mode",
    "dtl_transfer_schedule_id",
    "wfb_hypothesis_id",
    "ddpg_comparator_csi_hypothesis_id",
    "seeds",
    "paper_reference_sha256",
    "baseline_summary_sha256",
    "holdout_summary_sha256",
    "baseline_payload_sha256",
    "baseline_validation_sha256",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root, relative, label):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Artifact has no {label} path")
    path = (root / relative).resolve()
    if path.parent != root:
        raise ValueError(f"Artifact {label} must be directly inside the result directory")
    return path


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _optional_validation_authority_failures(root, manifest):
    fields = ("validation", "validation_sha256")
    if not any(field in manifest for field in fields):
        return []
    if not all(field in manifest for field in fields):
        return ["result manifest has incomplete validation authority"]
    try:
        validation_path = _inside(root, manifest["validation"], "validation")
    except ValueError as error:
        return [str(error)]
    if not validation_path.is_file():
        return [f"missing validation: {validation_path.name}"]
    if (
        not _valid_sha256(manifest["validation_sha256"])
        or _sha256(validation_path) != manifest["validation_sha256"]
    ):
        return [f"validation hash mismatch: {validation_path.name}"]
    return []


def _validate_evidence(path):
    failures = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return [f"invalid evidence {path.name}: {type(error).__name__}: {error}"]
    if payload.get("schema_version") != 1:
        failures.append(f"{path.name}: unsupported schema")
    if payload.get("kind") != "generated_result_evidence":
        failures.append(f"{path.name}: invalid evidence kind")
    if not isinstance(payload.get("figure"), int) or payload["figure"] <= 0:
        failures.append(f"{path.name}: invalid figure number")
    if not _valid_sha256(payload.get("source_tree_digest")):
        failures.append(f"{path.name}: invalid source-tree digest")
    for field in ("profile_digest", "physical_configuration_fingerprint"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            failures.append(f"{path.name}: missing {field}")
    if payload.get("power_hypothesis_id") not in {
        "paper_db_as_dbm", "paper_db_as_dbw"
    }:
        failures.append(f"{path.name}: invalid power hypothesis")
    if not valid_ambiguity_manifest_fields(payload):
        failures.append(f"{path.name}: invalid ambiguity hypothesis identities")
    seeds = payload.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 5
        or len(set(seeds)) != len(seeds)
    ):
        failures.append(f"{path.name}: at least five unique seeds are required")
    if not np.isclose(float(payload.get("feasibility_rate", -1.0)), 1.0):
        failures.append(f"{path.name}: feasibility rate is not 1.0")
    units = payload.get("units")
    if not isinstance(units, dict) or not units or any(
        not isinstance(value, str) or not value for value in units.values()
    ):
        failures.append(f"{path.name}: missing array units")
    arrays = payload.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        failures.append(f"{path.name}: missing plotted arrays")
    else:
        for label, values in arrays.items():
            try:
                array = np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                failures.append(f"{path.name}: array {label} is not numeric")
                continue
            if array.ndim not in {1, 2} or array.size == 0 or not np.all(
                np.isfinite(array)
            ):
                failures.append(f"{path.name}: array {label} is invalid")
    return failures


def validate_result_evidence_bundle(directory):
    """Reject every result PNG that lacks a manifest and raw evidence sidecar."""
    root = Path(directory).resolve()
    images = sorted(root.glob("*.png"), key=lambda path: path.name)
    if not images:
        return {
            "schema_version": 1,
            "overall_pass": True,
            "images": [],
            "failures": [],
        }
    failures = []
    manifest_path = root / "result_manifest.json"
    if not manifest_path.is_file():
        return {
            "schema_version": 1,
            "overall_pass": False,
            "images": [path.name for path in images],
            "failures": ["missing result_manifest.json for generated PNG files"],
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        return {
            "schema_version": 1,
            "overall_pass": False,
            "images": [path.name for path in images],
            "failures": [f"invalid result_manifest.json: {error}"],
        }
    if manifest.get("schema_version") != 1 or manifest.get(
        "kind"
    ) != "generated_result_manifest":
        failures.append("result_manifest.json has an invalid schema or kind")
    failures.extend(_optional_validation_authority_failures(root, manifest))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
        failures.append("result_manifest.json artifacts must be a list")
    covered = []
    bundle_provenance = None
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            failures.append(f"manifest artifact {index} is not an object")
            continue
        try:
            image_path = _inside(root, artifact.get("image"), "image")
            evidence_path = _inside(root, artifact.get("evidence"), "evidence")
        except ValueError as error:
            failures.append(str(error))
            continue
        covered.append(image_path.name)
        if not image_path.is_file():
            failures.append(f"missing image: {image_path.name}")
        elif artifact.get("image_sha256") != _sha256(image_path):
            failures.append(f"image hash mismatch: {image_path.name}")
        if not evidence_path.is_file():
            failures.append(f"missing evidence: {evidence_path.name}")
        else:
            if artifact.get("evidence_sha256") != _sha256(evidence_path):
                failures.append(f"evidence hash mismatch: {evidence_path.name}")
            failures.extend(_validate_evidence(evidence_path))
            try:
                evidence_payload = json.loads(
                    evidence_path.read_text(encoding="utf-8")
                )
            except Exception:
                evidence_payload = None
            if isinstance(evidence_payload, dict):
                provenance = {
                    field: evidence_payload.get(field)
                    for field in BUNDLE_PROVENANCE_FIELDS
                }
                if bundle_provenance is None:
                    bundle_provenance = provenance
                elif provenance != bundle_provenance:
                    failures.append(
                        f"mixed provenance across evidence bundle: "
                        f"{evidence_path.name}"
                    )
    image_names = [path.name for path in images]
    if sorted(covered) != image_names:
        failures.append("manifest does not cover every PNG exactly once")
    return {
        "schema_version": 1,
        "overall_pass": not failures,
        "images": image_names,
        "failures": failures,
    }


PARETO_ARTIFACT_TYPES = {
    "training_convergence",
    "primary_paired_delta",
    "pareto_front",
}

PARETO_PROVENANCE_FIELDS = (
    "algorithm_id",
    "source_tree_digest",
    "profile_digest",
    "physical_configuration_fingerprint",
    "baseline_payload_sha256",
    "baseline_checkpoint_authority",
    "gate_selection_lock_sha256",
    "research_seeds",
    "seed_roles",
    "preference_grid",
    "anchors",
    "extension_summary_sha256",
    "baseline_validation_sha256",
    "validation_level",
)


def _pareto_evidence_failures(path):
    failures = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return None, [f"invalid evidence {path.name}: {error}"]
    if payload.get("schema_version") != 1:
        failures.append(f"{path.name}: unsupported schema")
    if payload.get("kind") != "ba_pcdtl_generated_evidence":
        failures.append(f"{path.name}: invalid evidence kind")
    if payload.get("artifact_type") not in PARETO_ARTIFACT_TYPES:
        failures.append(f"{path.name}: invalid artifact type")
    if payload.get("algorithm_id") != "ba_pcdtl_v1":
        failures.append(f"{path.name}: invalid algorithm identity")
    for field in (
        "source_tree_digest", "baseline_payload_sha256",
        "gate_selection_lock_sha256",
    ):
        if not _valid_sha256(payload.get(field)):
            failures.append(f"{path.name}: invalid {field}")
    for field in ("profile_digest", "physical_configuration_fingerprint"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            failures.append(f"{path.name}: missing {field}")
    checkpoint_authority = payload.get("baseline_checkpoint_authority")
    if (
        not isinstance(checkpoint_authority, list)
        or len(checkpoint_authority) < 10
        or any(
            not _valid_sha256(row.get("source_checkpoint_sha256"))
            or not _valid_sha256(row.get("dtl_pcsi_checkpoint_sha256"))
            for row in checkpoint_authority
            if isinstance(row, dict)
        )
        or any(not isinstance(row, dict) for row in checkpoint_authority)
    ):
        failures.append(f"{path.name}: invalid baseline checkpoint authority")
    research = payload.get("research_seeds")
    roles = payload.get("seed_roles")
    if (
        not isinstance(research, list)
        or len(research) < 10
        or len(research) != len(set(research))
        or not isinstance(roles, dict)
        or roles.get("research") != research
    ):
        failures.append(f"{path.name}: invalid research seed provenance")
    else:
        validation = roles.get("validation", [])
        test = roles.get("test", [])
        if (
            len(validation) < 5
            or len(test) < 10
            or set(research) & set(validation)
            or set(research) & set(test)
            or set(validation) & set(test)
        ):
            failures.append(f"{path.name}: seed roles overlap or are incomplete")
    try:
        preferences = np.asarray(payload.get("preference_grid"), dtype=float)
        if (
            preferences.ndim != 2
            or preferences.shape[1] != 2
            or not np.all(np.isfinite(preferences))
            or np.any(preferences < 0.0)
            or not np.allclose(preferences.sum(axis=1), 1.0, atol=1e-8, rtol=0.0)
        ):
            raise ValueError
    except (TypeError, ValueError):
        failures.append(f"{path.name}: invalid preference grid")
    anchors = payload.get("anchors")
    if not isinstance(anchors, dict) or not all(
        key in anchors for key in ("ideal", "nadir")
    ):
        failures.append(f"{path.name}: missing anchors")
    if payload.get("acceptance_uses_raw_arrays") is not True:
        failures.append(f"{path.name}: acceptance must use raw arrays")
    units = payload.get("units")
    if not isinstance(units, dict) or not units or any(
        not isinstance(value, str) or not value for value in units.values()
    ):
        failures.append(f"{path.name}: invalid units")
    arrays = payload.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        failures.append(f"{path.name}: missing raw arrays")
    else:
        for name, values in arrays.items():
            try:
                array = np.asarray(values, dtype=float)
                if array.ndim not in {1, 2, 3} or array.size == 0 or not np.all(
                    np.isfinite(array)
                ):
                    raise ValueError
            except (TypeError, ValueError):
                failures.append(f"{path.name}: invalid array {name}")
    if any("envelope" in str(key).lower() for key in payload):
        failures.append(f"{path.name}: pointwise envelopes are forbidden")
    return payload, failures


def validate_pareto_evidence_bundle(directory):
    """Validate the three hashed BA-PCDTL plots and their raw JSON sidecars."""
    root = Path(directory).resolve()
    manifest_path = root / "result_manifest.json"
    failures = []
    if not manifest_path.is_file():
        return {
            "schema_version": 1,
            "overall_pass": False,
            "failures": ["missing result_manifest.json"],
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        return {
            "schema_version": 1,
            "overall_pass": False,
            "failures": [f"invalid result manifest: {error}"],
        }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "ba_pcdtl_result_manifest"
    ):
        failures.append("result_manifest.json has an invalid schema or kind")
    failures.extend(_optional_validation_authority_failures(root, manifest))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
        failures.append("result manifest artifacts must be a list")
    observed_types = []
    covered_images = []
    common_provenance = None
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            failures.append(f"artifact {index} is not an object")
            continue
        artifact_type = artifact.get("artifact_type")
        observed_types.append(artifact_type)
        try:
            image_path = _inside(root, artifact.get("image"), "image")
            evidence_path = _inside(root, artifact.get("evidence"), "evidence")
        except ValueError as error:
            failures.append(str(error))
            continue
        covered_images.append(image_path.name)
        if not image_path.is_file():
            failures.append(f"missing image: {image_path.name}")
        elif artifact.get("image_sha256") != _sha256(image_path):
            failures.append(f"image hash mismatch: {image_path.name}")
        if not evidence_path.is_file():
            failures.append(f"missing evidence: {evidence_path.name}")
            continue
        if artifact.get("evidence_sha256") != _sha256(evidence_path):
            failures.append(f"evidence hash mismatch: {evidence_path.name}")
        payload, evidence_failures = _pareto_evidence_failures(evidence_path)
        failures.extend(evidence_failures)
        if isinstance(payload, dict):
            if payload.get("artifact_type") != artifact_type:
                failures.append(f"artifact type mismatch: {evidence_path.name}")
            provenance = {
                field: payload.get(field) for field in PARETO_PROVENANCE_FIELDS
            }
            if common_provenance is None:
                common_provenance = provenance
            elif provenance != common_provenance:
                failures.append(
                    f"mixed provenance across evidence bundle: {evidence_path.name}"
                )
    if set(observed_types) != PARETO_ARTIFACT_TYPES or len(observed_types) != 3:
        failures.append("manifest must contain each required artifact exactly once")
    images = sorted(path.name for path in root.glob("*.png"))
    if sorted(covered_images) != images:
        failures.append("manifest does not cover every PNG exactly once")
    return {
        "schema_version": 1,
        "overall_pass": not failures,
        "images": images,
        "failures": failures,
    }
