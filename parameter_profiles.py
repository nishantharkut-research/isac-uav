"""Strict, reversible parameter profiles for omitted paper values."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

from config import cfg
from paper_contract import (
    CALIBRATION_SOLVER_SCOPE,
    FIGURE_6_SYSTEM_CONFIGURATION,
    ao_surrogate_contract,
)
from physical_hypotheses import get_power_hypothesis
from ambiguity_hypotheses import (
    get_cross_function_interference_hypothesis,
    get_los_probability_hypothesis,
    get_rate_model_hypothesis,
)
from parameter_evidence import (
    PARAMETER_EVIDENCE,
    PROFILE_BOUNDS,
    PROFILE_KEYS,
    SENSITIVITY_ONLY_KEYS,
)
from paper_reference import paper_reference_digest


SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = {
    "schema_version", "name", "kind", "values", "evidence", "selection",
    "digest",
}
_ACTIVE_PROFILE = None


@dataclass(frozen=True)
class ParameterProfile:
    schema_version: int
    name: str
    kind: str
    values: Mapping[str, float]
    evidence: Mapping[str, str]
    selection: Mapping[str, Any]
    digest: str
    source_path: Optional[str] = None

    def as_dict(self, include_digest=True):
        payload = {
            "schema_version": self.schema_version,
            "name": self.name,
            "kind": self.kind,
            "values": dict(self.values),
            "evidence": dict(self.evidence),
            "selection": dict(self.selection),
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def _canonical_digest(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _numeric_values(values):
    normalized = {}
    for key, value in values.items():
        if key not in PROFILE_KEYS:
            raise ValueError(f"Unknown profile parameter: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Profile value {key} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Profile value {key} must be finite")
        low, high = PROFILE_BOUNDS[key]
        if value < low or value > high:
            raise ValueError(
                f"Profile value {key}={value} is outside [{low}, {high}]"
            )
        normalized[key] = value
    return normalized


def _merged(values, key):
    return values[key] if key in values else float(getattr(cfg, key))


def _validate_dependencies(values):
    ue_x_min = _merged(values, "ue_x_min")
    ue_x_max = _merged(values, "ue_x_max")
    ue_y_min = _merged(values, "ue_y_min")
    ue_y_max = _merged(values, "ue_y_max")
    if not math.isclose(ue_x_max - ue_x_min, 100.0, abs_tol=1e-9):
        raise ValueError("The journal requires a 100 m UE-region x width")
    if not math.isclose(ue_y_max - ue_y_min, 100.0, abs_tol=1e-9):
        raise ValueError("The journal requires a 100 m UE-region y width")
    if not (cfg.x_min <= ue_x_min < ue_x_max <= cfg.x_max):
        raise ValueError("UE x bounds must remain inside the service area")
    if not (cfg.y_min <= ue_y_min < ue_y_max <= cfg.y_max):
        raise ValueError("UE y bounds must remain inside the service area")

    zeta_los = _merged(values, "zeta_los")
    zeta_nlos = _merged(values, "zeta_nlos")
    if zeta_los >= zeta_nlos:
        raise ValueError("zeta_los must be smaller than zeta_nlos")

    upsilon_0 = _merged(values, "upsilon_0")
    upsilon_min = _merged(values, "upsilon_min")
    if upsilon_min > upsilon_0:
        raise ValueError("upsilon_min cannot exceed upsilon_0")

    noise_min = _merged(values, "noise_min")
    noise_std = _merged(values, "noise_std")
    if noise_min > noise_std:
        raise ValueError("noise_min cannot exceed noise_std")

    if "P_a_max" in values and values["P_a_max"] < cfg.sigma2_qa:
        raise ValueError("P_a_max cannot be below the active-noise floor")


def profile_from_mapping(mapping, source_path=None):
    if not isinstance(mapping, Mapping):
        raise ValueError("Profile must be a JSON object")
    unknown_top = set(mapping) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(f"Unknown profile fields: {sorted(unknown_top)}")
    if mapping.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    name = mapping.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Profile name must be a non-empty string")
    kind = mapping.get("kind")
    if kind not in {"baseline", "candidate", "identified"}:
        raise ValueError("Profile kind must be baseline, candidate, or identified")

    raw_values = mapping.get("values")
    raw_evidence = mapping.get("evidence")
    selection = mapping.get("selection", {})
    if not isinstance(raw_values, Mapping) or not raw_values:
        raise ValueError("Profile values must be a non-empty object")
    if not isinstance(raw_evidence, Mapping):
        raise ValueError("Profile evidence must be an object")
    if not isinstance(selection, Mapping):
        raise ValueError("Profile selection must be an object")

    values = _numeric_values(raw_values)
    evidence = {}
    for key in values:
        reference = raw_evidence.get(key)
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"Profile value {key} requires an evidence reference")
        if reference not in PARAMETER_EVIDENCE:
            raise ValueError(f"Unknown evidence reference for {key}: {reference}")
        if PARAMETER_EVIDENCE[reference].name != key:
            raise ValueError(f"Evidence {reference} does not apply to {key}")
        evidence[key] = reference
    extra_evidence = set(raw_evidence) - set(values)
    if extra_evidence:
        raise ValueError(
            f"Evidence supplied for absent profile values: {sorted(extra_evidence)}"
        )
    _validate_dependencies(values)

    canonical = {
        "schema_version": SCHEMA_VERSION,
        "name": name.strip(),
        "kind": kind,
        "values": values,
        "evidence": evidence,
        "selection": dict(selection),
    }
    digest = _canonical_digest(canonical)
    declared_digest = mapping.get("digest")
    if declared_digest is not None and declared_digest != digest:
        raise ValueError("Profile digest does not match its contents")
    return ParameterProfile(
        schema_version=SCHEMA_VERSION,
        name=canonical["name"],
        kind=kind,
        values=MappingProxyType(values),
        evidence=MappingProxyType(evidence),
        selection=MappingProxyType(dict(selection)),
        digest=digest,
        source_path=str(source_path) if source_path is not None else None,
    )


def load_profile(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return profile_from_mapping(payload, source_path=path)


def _load_json_object(path, label):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _resolve_locked_artifact(directory, value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Power-hypothesis lock has no {label} path")
    directory = Path(directory).resolve()
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"Power-hypothesis {label} must be relative")
    path = (directory / relative).resolve()
    try:
        path.relative_to(directory)
    except ValueError as error:
        raise ValueError(
            f"Power-hypothesis {label} escapes the calibration directory"
        ) from error
    if not path.is_file():
        raise ValueError(f"Missing power-hypothesis {label}: {path}")
    return path


def _current_source_tree_digest():
    from source_integrity import source_tree_digest
    return source_tree_digest()


def _require_source_tree_binding(label, payload, expected_digest):
    actual = payload.get("source_tree_digest")
    if actual != expected_digest:
        raise ValueError(
            f"{label} was produced by a different executable source tree"
        )


def _resolve_sibling_artifact(directory, value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Selection authority has no {label} path")
    directory = Path(directory).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = directory / path
    path = path.resolve()
    if path.parent != directory:
        raise ValueError(f"{label} must be beside the selected profile")
    return path


def load_prepared_profile(path):
    """Load only a profile backed by a successful preparation authority."""
    profile_path = Path(path).resolve()
    profile = load_profile(profile_path)
    if profile.kind != "identified":
        raise ValueError("A full run requires an identified parameter profile")
    current_source_digest = _current_source_tree_digest()
    _require_source_tree_binding(
        "Prepared profile", profile.selection, current_source_digest
    )

    directory = profile_path.parent
    status_path = directory / "preparation_status.json"
    authority_path = directory / "reproduction_selection.json"
    status = _load_json_object(status_path, "preparation status")
    authority = _load_json_object(authority_path, "selection authority")
    _require_source_tree_binding(
        "Preparation status", status, current_source_digest
    )
    _require_source_tree_binding(
        "Selection authority", authority, current_source_digest
    )

    if status.get("status") != "passed" or not status.get(
        "ready_for_full_run", False
    ):
        raise ValueError("Parameter preparation did not finish successfully")
    status_authority = _resolve_sibling_artifact(
        directory, status.get("authority"), "status authority"
    )
    if status_authority != authority_path.resolve():
        raise ValueError("Preparation status points to a different authority")
    if not authority.get("ready_for_full_run", False) or not authority.get(
        "preflight_passed", False
    ):
        raise ValueError("Selection authority is not ready for a full run")
    if authority.get("paper_curve_ordinates_used") is not False or authority.get(
        "communication_rate_magnitude_used_in_selection"
    ) is not False:
        raise ValueError("Selection authority used prohibited result fitting")

    digests = {
        profile.digest,
        status.get("selected_profile_digest"),
        authority.get("selected_profile_digest"),
    }
    if None in digests or len(digests) != 1:
        raise ValueError("Prepared-profile digests do not agree")
    selected_path = _resolve_sibling_artifact(
        directory, authority.get("selected_profile"), "selected profile"
    )
    if selected_path != profile_path:
        raise ValueError("Selection authority points to a different profile")

    artifact_paths = {
        key: _resolve_sibling_artifact(directory, authority.get(key), key)
        for key in (
            "parameter_search", "parameter_matrix", "holdout_validation",
            "stress_validation", "scientific_preflight",
        )
    }
    for label, artifact_path in artifact_paths.items():
        if not artifact_path.is_file():
            raise ValueError(f"Missing prepared-profile artifact {label}")

    search = _load_json_object(
        artifact_paths["parameter_search"], "parameter search"
    )
    holdout = _load_json_object(
        artifact_paths["holdout_validation"], "holdout validation"
    )
    stress = _load_json_object(
        artifact_paths["stress_validation"], "stress validation"
    )
    preflight = _load_json_object(
        artifact_paths["scientific_preflight"], "scientific preflight"
    )
    for label, artifact in (
        ("Parameter search", search),
        ("Holdout validation", holdout),
        ("Stress validation", stress),
        ("Scientific preflight", preflight),
    ):
        _require_source_tree_binding(label, artifact, current_source_digest)
    if search.get("selected_profile_digest") != profile.digest:
        raise ValueError("Parameter-search digest does not match the profile")
    if not holdout.get("accepted", False):
        raise ValueError("Held-out structural validation was not accepted")
    if not stress.get("accepted", False) or stress.get(
        "profile_digest"
    ) != profile.digest:
        raise ValueError("Stress validation does not authorize this profile")
    preflight_profile = preflight.get("parameter_profile", {})
    if not preflight.get("passed", False) or preflight_profile.get(
        "digest"
    ) != profile.digest:
        raise ValueError("Scientific preflight does not authorize this profile")
    return profile


def load_calibrated_profile(path):
    """Load a calibrated profile only when its sibling report authorizes it."""
    profile_path = Path(path).resolve()
    profile = load_profile(profile_path)
    if profile.kind != "identified":
        raise ValueError("A full run requires an identified calibrated profile")
    current_source_digest = _current_source_tree_digest()
    _require_source_tree_binding(
        "Calibrated profile", profile.selection, current_source_digest
    )

    expected_protocol = "bounded AO calibration on paper Fig. 6 AO curves"
    expected_targets = {"6": ["AO-pCSI", "AO-iCSI"]}
    selection = profile.selection
    if selection.get("protocol") != expected_protocol:
        raise ValueError("Profile is not authorized by the current calibration protocol")
    if selection.get("calibration_solver_scope") != CALIBRATION_SOLVER_SCOPE:
        raise ValueError("Profile does not disclose the calibration solver scope")
    if selection.get("calibration_system") != FIGURE_6_SYSTEM_CONFIGURATION:
        raise ValueError("Profile calibration system does not match Figure 6")
    if selection.get("calibration_targets") != expected_targets:
        raise ValueError("Profile calibration targets do not match the AO contract")
    if selection.get("holdout_figures_used") != []:
        raise ValueError("Profile calibration used a declared holdout figure")
    if selection.get("paper_curve_ordinates_used") is not True:
        raise ValueError("Profile does not declare paper-curve calibration")
    hypothesis_id = selection.get("power_hypothesis_id")
    if hypothesis_id not in {"paper_db_as_dbm", "paper_db_as_dbw"}:
        raise ValueError("Profile has no valid power hypothesis")
    calibration_noise_id = selection.get("calibration_noise_hypothesis_id")
    if calibration_noise_id != "description_literal_target_matches_ue":
        raise ValueError("Profile has no valid calibration noise hypothesis")
    los_probability_id = selection.get("los_probability_hypothesis_id")
    get_los_probability_hypothesis(los_probability_id)
    rate_model_id = selection.get("rate_model_hypothesis_id")
    get_rate_model_hypothesis(rate_model_id)
    cross_interference_id = selection.get(
        "cross_function_interference_hypothesis_id"
    )
    get_cross_function_interference_hypothesis(cross_interference_id)
    fixed_values = selection.get("fixed_sensitivity_values", {})
    if not isinstance(fixed_values, Mapping):
        raise ValueError("Profile fixed sensitivities must be an object")
    fixed_protocol = selection.get("fixed_sensitivity_protocol")
    if fixed_values and fixed_protocol != "predeclared_not_result_selected":
        raise ValueError("Profile fixed sensitivities were not predeclared")
    if not fixed_values and fixed_protocol is not None:
        raise ValueError("Profile declares an empty fixed-sensitivity protocol")
    for key, value in fixed_values.items():
        if key not in SENSITIVITY_ONLY_KEYS:
            raise ValueError(f"Profile fixed sensitivity is not allowed: {key}")
        if key not in profile.values or not math.isclose(
            float(value), float(profile.values[key]), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("Profile fixed sensitivities do not match its values")
    reference_digest = selection.get("paper_reference_sha256")
    if (
        not isinstance(reference_digest, str)
        or len(reference_digest) != 64
        or any(character not in "0123456789abcdef" for character in reference_digest)
    ):
        raise ValueError("Profile has no valid paper-reference digest")
    if reference_digest != paper_reference_digest(None):
        raise ValueError(
            "Profile is not bound to the canonical paper reference"
        )

    lock_candidates = (
        profile_path.parent / "power_hypothesis_lock.json",
        profile_path.parent.parent / "power_hypothesis_lock.json",
    )
    lock_path = next((item for item in lock_candidates if item.is_file()), None)
    if lock_path is None:
        raise ValueError("Missing immutable power-hypothesis lock")
    lock = _load_json_object(lock_path, "power-hypothesis lock")
    _require_source_tree_binding(
        "Power-hypothesis lock", lock, current_source_digest
    )
    if (
        lock.get("schema_version") != 1
        or lock.get("kind") != "power_hypothesis_selection_lock"
        or lock.get("locked") is not True
        or lock.get("calibration_figures_used") != [6]
        or lock.get("holdout_figures_used") != []
    ):
        raise ValueError("Invalid power-hypothesis lock")
    if lock.get("selected_hypothesis_id") != hypothesis_id:
        raise ValueError("Power-hypothesis lock selected a different model")
    if (
        lock.get("calibration_solver_scope") != CALIBRATION_SOLVER_SCOPE
        or lock.get("calibration_system") != FIGURE_6_SYSTEM_CONFIGURATION
    ):
        raise ValueError("Power-hypothesis lock has the wrong calibration system")
    if lock.get("paper_reference_sha256") != reference_digest:
        raise ValueError("Power-hypothesis lock used a different paper reference")
    expected_hypotheses = {
        "calibration_noise_hypothesis_id": calibration_noise_id,
        "los_probability_hypothesis_id": los_probability_id,
        "rate_model_hypothesis_id": rate_model_id,
        "cross_function_interference_hypothesis_id": cross_interference_id,
    }
    if lock.get("calibration_hypotheses") != expected_hypotheses:
        raise ValueError("Power-hypothesis lock has different calibration hypotheses")
    if lock.get("fixed_sensitivity_values") != dict(fixed_values):
        raise ValueError("Power-hypothesis lock has different fixed sensitivities")
    if lock.get("source_profile_digest") != selection.get("source_profile_digest"):
        raise ValueError("Power-hypothesis lock has a different source profile")
    calibration_budgets = lock.get("calibration_budgets")
    if (
        not isinstance(calibration_budgets, Mapping)
        or any(
            isinstance(calibration_budgets.get(field), bool)
            or not isinstance(calibration_budgets.get(field), int)
            or calibration_budgets[field] <= 0
            for field in ("ao_iterations", "optimizer_max_iterations")
        )
    ):
        raise ValueError("Power-hypothesis lock has invalid calibration budgets")
    selected_profile = lock.get("selected_profile")
    if not isinstance(selected_profile, str) or not selected_profile.strip():
        raise ValueError("Power-hypothesis lock has no selected profile")
    locked_profile_path = (lock_path.parent / selected_profile).resolve()
    if locked_profile_path != profile_path:
        raise ValueError("Power-hypothesis lock points to a different profile")
    candidate_artifacts = lock.get("candidate_artifacts")
    if (
        not isinstance(candidate_artifacts, Mapping)
        or set(candidate_artifacts) != {"paper_db_as_dbm", "paper_db_as_dbw"}
    ):
        raise ValueError("Power-hypothesis lock has incomplete candidate artifacts")
    resolved_artifacts = {}
    for candidate_id, artifact in candidate_artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError("Power-hypothesis candidate artifact must be an object")
        report_artifact = _resolve_locked_artifact(
            lock_path.parent,
            artifact.get("calibration_report"),
            f"{candidate_id} calibration report",
        )
        report_digest = artifact.get("calibration_report_sha256")
        profile_artifact = artifact.get("calibrated_profile")
        profile_digest = artifact.get("calibrated_profile_sha256")
        if not _is_sha256(report_digest):
            raise ValueError("Power-hypothesis candidate report has invalid hash")
        if profile_artifact is None:
            if profile_digest is not None:
                raise ValueError("Power-hypothesis candidate profile hash is orphaned")
            resolved_profile = None
        else:
            resolved_profile = _resolve_locked_artifact(
                lock_path.parent,
                profile_artifact,
                f"{candidate_id} calibrated profile",
            )
            if not _is_sha256(profile_digest):
                raise ValueError("Power-hypothesis candidate profile has invalid hash")
        resolved_artifacts[candidate_id] = {
            "report": report_artifact,
            "report_sha256": report_digest,
            "profile": resolved_profile,
            "profile_sha256": profile_digest,
        }
    selected_artifacts = resolved_artifacts[hypothesis_id]
    if selected_artifacts["profile"] != profile_path:
        raise ValueError("Power-hypothesis selected artifact points elsewhere")
    candidate_scores = lock.get("candidate_scores")
    if (
        not isinstance(candidate_scores, list)
        or {item.get("hypothesis_id") for item in candidate_scores}
        != {"paper_db_as_dbm", "paper_db_as_dbw"}
        or not any(
            item.get("hypothesis_id") == hypothesis_id
            and item.get("fully_feasible") is True
            and item.get("calibration_admissible") is True
            and item.get("ordering_violations") == 0
            and item.get("power_direction_violations") == 0
            for item in candidate_scores
        )
    ):
        raise ValueError("Power-hypothesis lock has invalid candidate scores")

    report_path = _resolve_sibling_artifact(
        profile_path.parent,
        selection.get("calibration_report"),
        "calibration report",
    )
    report = _load_json_object(report_path, "calibration report")
    if selected_artifacts["report"] != report_path:
        raise ValueError("Power-hypothesis selected report points elsewhere")
    _require_source_tree_binding(
        "Calibration report", report, current_source_digest
    )
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported calibration report schema")
    if report.get("method") != "bounded_scipy_powell":
        raise ValueError("Unexpected calibration report method")
    if (
        report.get("calibration_solver_scope") != CALIBRATION_SOLVER_SCOPE
        or report.get("calibration_system") != FIGURE_6_SYSTEM_CONFIGURATION
    ):
        raise ValueError("Calibration report has the wrong Figure 6 system")
    if report.get("ao_surrogate_contract") != ao_surrogate_contract(cfg):
        raise ValueError("Calibration report has the wrong AO surrogate contract")
    if report.get("power_hypothesis_id") != hypothesis_id:
        raise ValueError("Calibration report uses a different power hypothesis")
    if report.get("calibration_noise_hypothesis_id") != calibration_noise_id:
        raise ValueError("Calibration report uses a different noise hypothesis")
    if report.get("los_probability_hypothesis_id") != los_probability_id:
        raise ValueError(
            "Calibration report uses a different LoS probability hypothesis"
        )
    if report.get("rate_model_hypothesis_id") != rate_model_id:
        raise ValueError(
            "Calibration report uses a different rate-model hypothesis"
        )
    if report.get("cross_function_interference_hypothesis_id") != (
        cross_interference_id
    ):
        raise ValueError(
            "Calibration report uses a different cross-function interference "
            "hypothesis"
        )
    if report.get("fixed_sensitivity_values", {}) != dict(fixed_values):
        raise ValueError("Calibration report fixed sensitivities do not match")
    if report.get("fixed_sensitivity_protocol") != fixed_protocol:
        raise ValueError("Calibration report fixed-sensitivity protocol differs")
    if report.get("calibration_targets") != expected_targets:
        raise ValueError("Calibration report targets do not match the AO contract")
    if report.get("holdout_figures_used") != []:
        raise ValueError("Calibration report used a declared holdout figure")
    if report.get("optimizer_success") is not True:
        raise ValueError("Calibration report does not record optimizer success")
    if report.get("profile_digest") != profile.digest:
        raise ValueError("Calibration report digest does not match the profile")
    if report.get("paper_reference_sha256") != reference_digest:
        raise ValueError("Calibration report used a different paper reference")
    seeds = report.get("common_random_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Calibration report has invalid common-random seeds")
    if lock.get("common_random_seeds") != seeds:
        raise ValueError("Power-hypothesis lock has different common-random seeds")
    for field in ("ao_iterations", "optimizer_max_iterations"):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Calibration report has invalid {field}")
        if calibration_budgets.get(field) != value:
            raise ValueError("Power-hypothesis lock has different calibration budgets")
    prior_weight = report.get("prior_weight")
    if (
        isinstance(prior_weight, bool)
        or not isinstance(prior_weight, (int, float))
        or not math.isfinite(float(prior_weight))
        or float(prior_weight) < 0.0
    ):
        raise ValueError("Calibration report has invalid prior weight")
    if not math.isclose(
        float(selection.get("prior_weight", -1.0)),
        float(prior_weight),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Calibration prior weight does not match the profile")

    selected_values = report.get("selected_values")
    if not isinstance(selected_values, Mapping) or not selected_values:
        raise ValueError("Calibration report has no selected values")
    for key, value in selected_values.items():
        if key not in profile.values or not math.isclose(
            float(value), float(profile.values[key]), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("Calibration report values do not match the profile")
    selected_loss = report.get("selected_loss")
    if isinstance(selected_loss, bool) or not isinstance(selected_loss, (int, float)):
        raise ValueError("Calibration report has an invalid selected loss")
    if not math.isfinite(float(selected_loss)):
        raise ValueError("Calibration report has a non-finite selected loss")

    records = report.get("records")
    if not isinstance(records, list) or not records or records[-1].get("valid") is not True:
        raise ValueError("Calibration report has no valid selected candidate")
    selected_record_values = records[-1].get("values")
    if not isinstance(selected_record_values, Mapping) or any(
        key not in selected_record_values
        or not math.isclose(
            float(selected_record_values[key]),
            float(value),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        for key, value in selected_values.items()
    ):
        raise ValueError("Calibration report selected candidate does not match")
    if (
        _sha256_file(report_path) != selected_artifacts["report_sha256"]
        or _sha256_file(profile_path) != selected_artifacts["profile_sha256"]
    ):
        raise ValueError("Power-hypothesis artifact hash mismatch")
    for candidate_id, artifacts in resolved_artifacts.items():
        if candidate_id == hypothesis_id:
            continue
        if _sha256_file(artifacts["report"]) != artifacts["report_sha256"]:
            raise ValueError("Power-hypothesis artifact hash mismatch")
        if (
            artifacts["profile"] is not None
            and _sha256_file(artifacts["profile"])
            != artifacts["profile_sha256"]
        ):
            raise ValueError("Power-hypothesis artifact hash mismatch")
    return profile


def write_profile(profile, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(profile.as_dict(), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    return path


def _expanded_values(profile, cfg_obj=cfg):
    values = dict(profile.values)
    if "PT_dBm" in values:
        raise ValueError(
            "Profiles must use PT_dB; PT_dBm is derived display metadata"
        )
    if "PT_dB" in values:
        raise ValueError(
            "Profiles cannot reinterpret PT_dB; use a selected power hypothesis"
        )
    hypothesis_id = profile.selection.get("power_hypothesis_id")
    if hypothesis_id is not None:
        hypothesis = get_power_hypothesis(hypothesis_id)
        metadata = hypothesis.as_dict(cfg_obj.PT_dB)
        values.update({
            "power_hypothesis_id": hypothesis.identifier,
            "PT_reference": (
                "1 mW" if hypothesis.reference_power_watts == 1e-3 else "1 W"
            ),
            "PT_display_unit": hypothesis.display_unit,
            "PT": metadata["linear_power_watts"],
            "PT_dBm": metadata["power_dbm"],
            "PT_dBW": metadata["power_dbw"],
            "power_hypothesis_metadata": metadata,
        })
    los_probability_id = profile.selection.get(
        "los_probability_hypothesis_id"
    )
    if los_probability_id is not None:
        values["los_probability_hypothesis_id"] = (
            get_los_probability_hypothesis(los_probability_id).identifier
        )
    rate_model_id = profile.selection.get("rate_model_hypothesis_id")
    if rate_model_id is not None:
        values["rate_model_hypothesis_id"] = (
            get_rate_model_hypothesis(rate_model_id).identifier
        )
    cross_interference_id = profile.selection.get(
        "cross_function_interference_hypothesis_id"
    )
    if cross_interference_id is not None:
        values["cross_function_interference_hypothesis_id"] = (
            get_cross_function_interference_hypothesis(
                cross_interference_id
            ).identifier
        )
    if "fc" in values:
        values["wavelength"] = 3e8 / values["fc"]
    if "v_max" in values:
        delta_t = values.get("delta_T", cfg_obj.delta_T)
        values["D_max"] = values["v_max"] * delta_t
    if "rho_nmse" in values:
        rho = values["rho_nmse"]
        values["rho_partial"] = rho
        for key in (
            "sigma2_brj", "sigma2_bri", "sigma2_ruj", "sigma2_rti",
            "sigma2_bu", "sigma2_bk",
        ):
            values[key] = rho
    return values


def _set_values(profile, cfg_obj):
    changes = _expanded_values(profile, cfg_obj)
    for key, value in changes.items():
        setattr(cfg_obj, key, value)
    return changes


def activate_profile(profile, cfg_obj=cfg):
    """Apply a validated profile for the remainder of the current process."""
    global _ACTIVE_PROFILE
    if not isinstance(profile, ParameterProfile):
        profile = profile_from_mapping(profile)
    _set_values(profile, cfg_obj)
    _ACTIVE_PROFILE = profile
    return profile


def active_profile_metadata():
    """Return JSON-ready provenance for the profile active in this process."""
    if _ACTIVE_PROFILE is None:
        return {
            "name": "unprofiled-config-defaults",
            "kind": "baseline",
            "digest": None,
            "source_path": None,
            "values": {},
            "evidence_classes": {},
            "paper_curve_ordinates_used": False,
        }
    classes = {}
    for reference in _ACTIVE_PROFILE.evidence.values():
        evidence_class = PARAMETER_EVIDENCE[reference].evidence_class
        classes[evidence_class] = classes.get(evidence_class, 0) + 1
    return {
        "name": _ACTIVE_PROFILE.name,
        "kind": _ACTIVE_PROFILE.kind,
        "digest": _ACTIVE_PROFILE.digest,
        "source_path": _ACTIVE_PROFILE.source_path,
        "values": dict(_ACTIVE_PROFILE.values),
        "evidence_classes": classes,
        "paper_curve_ordinates_used": bool(
            _ACTIVE_PROFILE.selection.get("paper_curve_ordinates_used", False)
        ),
    }


@contextmanager
def applied_profile(profile, cfg_obj=cfg):
    global _ACTIVE_PROFILE
    if not isinstance(profile, ParameterProfile):
        profile = profile_from_mapping(profile)
    changes = _expanded_values(profile, cfg_obj)
    original = {key: getattr(cfg_obj, key) for key in changes}
    previous_profile = _ACTIVE_PROFILE
    try:
        _set_values(profile, cfg_obj)
        _ACTIVE_PROFILE = profile
        yield profile
    finally:
        for key, value in original.items():
            setattr(cfg_obj, key, value)
        _ACTIVE_PROFILE = previous_profile
