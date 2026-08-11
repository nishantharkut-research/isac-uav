"""Quantitative acceptance checks for generated baseline and extension runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from experiment_protocol import TRAINING_SEED_PROTOCOL, derive_training_seed
from ambiguity_hypotheses import (
    AMBIGUITY_IDENTITY_REGISTRIES,
    valid_ambiguity_manifest_fields,
)
from paper_reference import load_paper_references


METHOD_ORDER = ("DTL-pCSI", "DTL-WFB", "DTL-iCSI", "DDPG")
AMBIGUITY_RESULT_FIELDS = AMBIGUITY_IDENTITY_REGISTRIES


def _valid_ambiguity_contract(payload):
    return valid_ambiguity_manifest_fields(payload)


def result_payload_digest(payload: dict) -> str:
    """Hash a parsed result payload independently of JSON formatting."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        "power_hypothesis_id",
        "objective",
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
        if not _verify_run_checkpoint(
            root,
            record.get("source_checkpoint"),
            record.get("source_checkpoint_sha256"),
        ):
            return False, "Baseline source checkpoint authority mismatch."
        methods = record.get("methods", {})
        if set(methods) != set(METHOD_ORDER):
            return False, "Baseline method checkpoint grid is incomplete."
        for method_name in METHOD_ORDER:
            method = methods[method_name]
            if not _verify_run_checkpoint(
                root, method.get("checkpoint"), method.get("checkpoint_sha256")
            ):
                return False, f"Baseline {method_name} checkpoint authority mismatch."
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
        or sanity.get("passed") is not True
        or sanity.get("profile_digest") != result.get("profile_digest")
        or sanity.get("physical_configuration_fingerprint")
        != result.get("physical_configuration_fingerprint")
        or sanity.get("source_tree_digest") != result.get("source_tree_digest")
        or not _valid_ambiguity_contract(sanity)
    ):
        return False, "Pre-training physical-sanity report is unauthorized."
    return True, None


def _validate_extension_file_provenance(extension, root: Path):
    root = Path(root).resolve()
    records = extension.get("seed_records", [])
    if not isinstance(records, list) or len(records) != len(
        extension.get("seeds", [])
    ):
        return False, "Extension checkpoint grid is incomplete."
    for record in records:
        if not _verify_run_checkpoint(
            root, record.get("checkpoint"), record.get("checkpoint_sha256")
        ):
            return False, "Extension checkpoint authority mismatch."
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
    result_directory=None,
    verify_files: bool = True,
) -> dict:
    """Require paired communication retention and positive sensing gain."""
    checks = {
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
        "physical_feasibility": _check(
            np.isclose(float(extension.get("feasibility_rate", -1.0)), 1.0)
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
                    extension, Path(result_directory)
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
        statistics["joint_rate"] = _summary(joint)
    elif mode == "normalized_pareto":
        objective_values = _seed_values(
            extension.get("objective_seed_values", []),
            "normalized Pareto objective",
        )
        checks["normalized_pareto_metrics"] = _check(
            len(objective_values) == len(extension_comm)
        )
        statistics["normalized_pareto_objective"] = _summary(objective_values)
    else:
        checks["recognized_extension_mode"] = _check(False)
    return {
        "schema_version": 1,
        "validation": "sensing_extension",
        "baseline_payload_sha256": result_payload_digest(baseline),
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    baseline = _read(args.baseline)
    if args.extension:
        report = validate_extension_result(
            _read(args.extension),
            baseline,
            result_directory=Path(args.extension).resolve().parent,
            verify_files=True,
        )
    else:
        report = validate_baseline_result(
            baseline,
            _read(args.reference),
            result_directory=Path(args.baseline).resolve().parent,
            verify_files=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
