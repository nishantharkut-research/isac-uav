"""Five-seed aggregation and read-only acceptance for paper figure sweeps."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ambiguity_hypotheses import (
    AMBIGUITY_IDENTITY_REGISTRIES,
    valid_ambiguity_manifest_fields,
)
from scipy.stats import t

from cross_figure_validation import validate_figure_payload
from experiment_contract import FIGURE_CONTRACTS
from experiment_protocol import TRAINING_SEED_PROTOCOL, derive_training_seed
from paper_reference import paper_reference_digest
from validate_results import result_payload_digest


CALIBRATION_FIGURES = (6,)
HOLDOUT_FIGURES = (5, 7, 8, 9, 13, 14)
PUBLICATION_FIGURES = CALIBRATION_FIGURES + HOLDOUT_FIGURES
FEASIBILITY_VERIFICATION = "fail_closed_per_generated_realization"

REFERENCE_LABELS = {
    6: {},
    7: {},
    8: {
        "HRIS, L=40": "HRIS-L40",
        "HRIS, L=20": "HRIS-L20",
        "Passive RIS, L=40": "Passive-L40",
        "Passive RIS, L=20": "Passive-L20",
    },
    13: {
        "R_u, L=40": "Ru-L40",
        "R_u, L=30": "Ru-L30",
        "R_u, L=20": "Ru-L20",
        "R_k, L=30": "Rk-L30",
        "R_k, L=20": "Rk-L20",
    },
}

_MANIFEST_SHARED_FIELDS = (
    "protocol",
    "objective",
    "profile_digest",
    "physical_configuration_fingerprint",
    "source_tree_digest",
    "power_hypothesis_id",
    "training_seed_protocol",
    "baseline_payload_sha256",
    "baseline_validation_sha256",
    "episodes",
    "evaluation_episodes",
    *AMBIGUITY_IDENTITY_REGISTRIES.keys(),
    "ambiguity_contract",
)


def _check(passed, **details):
    return {"pass": bool(passed), **details}


def _sha256_text(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_completed_manifest(summary, result_directory):
    path = Path(result_directory) / "run_manifest.json"
    if not path.is_file():
        return False, "Completed holdout run manifest is missing."
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"Cannot read holdout run manifest: {error}"
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "run_manifest"
        or manifest.get("status") != "complete"
        or manifest.get("summary") != "holdout_summary.json"
        or manifest.get("summary_payload_sha256")
        != result_payload_digest(summary)
    ):
        return False, "Completed holdout manifest authority fields mismatch."
    mismatches = [
        field for field in _MANIFEST_SHARED_FIELDS
        if manifest.get(field) != summary.get(field)
    ]
    if manifest.get("seeds") != summary.get("seeds"):
        mismatches.append("seeds")
    if mismatches:
        return False, (
            "Holdout manifest and summary mismatch: " + ", ".join(mismatches)
        )
    root = Path(result_directory)
    try:
        for record in summary["seed_records"]:
            seed = int(record["seed"])
            for figure in PUBLICATION_FIGURES:
                payload = record["figures"][str(figure)]
                stem = root / "seed_runs" / f"seed-{seed}" / f"fig{figure}_data"
                for suffix, field in (
                    (".json", "figure_json_sha256"),
                    (".npz", "figure_npz_sha256"),
                ):
                    path = stem.with_suffix(suffix)
                    if not path.is_file() or _file_sha256(path) != payload[field]:
                        return False, (
                            f"Raw Figure {figure} {suffix} evidence mismatch "
                            f"for seed {seed}."
                        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        return False, f"Raw holdout file authority is incomplete: {error}"
    return True, None


def _finite_vector(values, label, expected_size=None):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite one-dimensional array")
    if expected_size is not None and array.size != expected_size:
        raise ValueError(f"{label} has the wrong number of points")
    return array


def _expected_x(figure):
    return _finite_vector(FIGURE_CONTRACTS[int(figure)].x_values, "paper x-grid")


def _expected_labels(figure):
    return {curve.label for curve in FIGURE_CONTRACTS[int(figure)].curves}


def _valid_training_seed_records(summary):
    records = summary.get("seed_records", [])
    seeds = summary.get("seeds", [])
    if (
        summary.get("training_seed_protocol") != TRAINING_SEED_PROTOCOL
        or not isinstance(records, list)
        or len(records) != len(seeds)
        or len(records) < 5
    ):
        return False
    try:
        return all(
            record.get("seed") == seed
            and record.get("source_training_seed")
            == derive_training_seed(seed, "source")
            and record.get("target_training_seed")
            == derive_training_seed(seed, "target")
            for record, seed in zip(records, seeds)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def aggregate_seed_payloads(seed_records):
    """Aggregate exact per-seed arrays; retain every raw array used."""
    if not isinstance(seed_records, list) or len(seed_records) < 5:
        raise ValueError("Holdout aggregation requires at least five seed records")
    seeds = [record.get("seed") for record in seed_records]
    if len(set(seeds)) != len(seeds) or any(seed is None for seed in seeds):
        raise ValueError("Holdout seed records require unique declared seeds")

    aggregated = {}
    for figure in PUBLICATION_FIGURES:
        key = str(figure)
        expected_x = _expected_x(figure)
        expected_labels = _expected_labels(figure)
        by_label = {label: [] for label in expected_labels}
        feasibility_rates = []
        verified_realizations = []
        failed_realizations = []
        json_digests = []
        npz_digests = []
        for record in seed_records:
            try:
                payload = record["figures"][key]
                x = _finite_vector(payload["x"], f"Figure {figure} x")
                curves = payload["curves"]
                feasibility_rate = float(payload["feasibility_rate"])
                feasibility_verification = payload[
                    "feasibility_verification"
                ]
                verified = int(payload["verified_realizations"])
                failed = int(payload["failed_realizations"])
                json_digest = payload["figure_json_sha256"]
                npz_digest = payload["figure_npz_sha256"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Seed {record.get('seed')} has no valid Figure {figure} "
                    "data or feasibility payload"
                ) from error
            if (
                not np.isfinite(feasibility_rate)
                or not 0.0 <= feasibility_rate <= 1.0
                or verified <= 0
                or failed != 0
                or not _is_sha256(json_digest)
                or not _is_sha256(npz_digest)
                or not np.isclose(
                    feasibility_rate, (verified - failed) / verified
                )
                or feasibility_verification != FEASIBILITY_VERIFICATION
            ):
                raise ValueError(
                    f"Figure {figure} has invalid physical-feasibility evidence"
                )
            feasibility_rates.append(feasibility_rate)
            verified_realizations.append(verified)
            failed_realizations.append(failed)
            json_digests.append(json_digest)
            npz_digests.append(npz_digest)
            if x.shape != expected_x.shape or not np.allclose(
                x, expected_x, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"Figure {figure} does not use the paper x-grid")
            if not isinstance(curves, dict) or set(curves) != expected_labels:
                raise ValueError(f"Figure {figure} method labels violate its contract")
            for label in expected_labels:
                by_label[label].append(
                    _finite_vector(
                        curves[label], f"Figure {figure} {label}", x.size
                    )
                )

        means, lows, highs, raw = {}, {}, {}, {}
        for label in sorted(expected_labels):
            values = np.stack(by_label[label])
            mean = np.mean(values, axis=0)
            standard_error = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
            half_width = t.ppf(0.975, values.shape[0] - 1) * standard_error
            means[label] = mean.tolist()
            lows[label] = (mean - half_width).tolist()
            highs[label] = (mean + half_width).tolist()
            raw[label] = values.tolist()
        aggregated[key] = {
            "x": expected_x.tolist(),
            "curves": means,
            "ci95_low": lows,
            "ci95_high": highs,
            "raw_seed_curves": raw,
            "raw_seed_feasibility_rates": feasibility_rates,
            "raw_seed_verified_realizations": verified_realizations,
            "raw_seed_failed_realizations": failed_realizations,
            "raw_seed_figure_json_sha256": json_digests,
            "raw_seed_figure_npz_sha256": npz_digests,
            "verified_realizations": int(sum(verified_realizations)),
            "failed_realizations": int(sum(failed_realizations)),
            "feasibility_rate": float(min(feasibility_rates)),
            "feasibility_verification": FEASIBILITY_VERIFICATION,
        }
    return aggregated


def _reference_check(figure, payload, references, relative_tolerance):
    reference = references.get("figures", {}).get(str(figure))
    if reference is None:
        return _check(True, applicable=False)
    if reference.get("kind") != "digitized_reference":
        return _check(False, applicable=True, reason="invalid reference provenance")
    x = _finite_vector(payload["x"], f"Figure {figure} generated x")
    reference_x = _finite_vector(reference.get("x", []), "reference x")
    if x.shape != reference_x.shape or not np.allclose(
        x, reference_x, rtol=0.0, atol=1e-12
    ):
        return _check(False, applicable=True, reason="x-grid mismatch")
    uncertainty = float(reference["uncertainty"]["absolute_y"])
    details = {}
    passed = True
    mapping = REFERENCE_LABELS.get(figure, {})
    for generated_label, generated_values in payload["curves"].items():
        reference_label = mapping.get(generated_label, generated_label)
        if reference_label not in reference["curves"]:
            passed = False
            details[generated_label] = {"pass": False, "reason": "missing reference"}
            continue
        generated = _finite_vector(generated_values, generated_label, x.size)
        target = _finite_vector(
            reference["curves"][reference_label], reference_label, x.size
        )
        tolerance = np.maximum(
            relative_tolerance * np.abs(target), 3.0 * uncertainty
        )
        errors = np.abs(generated - target)
        curve_pass = bool(np.all(errors <= tolerance))
        passed = passed and curve_pass
        details[generated_label] = {
            "pass": curve_pass,
            "reference_label": reference_label,
            "max_absolute_error": float(np.max(errors)),
            "max_allowed_error": float(np.max(tolerance)),
        }
    return _check(passed, applicable=True, curves=details)


def validate_holdout_result(
    summary,
    references,
    relative_tolerance=0.10,
    *,
    result_directory=None,
    verify_files=False,
):
    """Validate calibration/holdout outputs without changing generated arrays."""
    try:
        recomputed = aggregate_seed_payloads(summary.get("seed_records"))
        payload_consistent = _sha256_text(recomputed) == _sha256_text(
            summary.get("figures")
        )
        derived_feasibility = min(
            float(payload["feasibility_rate"])
            for payload in recomputed.values()
        )
    except (TypeError, ValueError):
        recomputed = None
        payload_consistent = False
        derived_feasibility = -1.0
    checks = {
        "generated_provenance": _check(
            summary.get("kind") == "generated_result"
            and summary.get("protocol") == "paper_calibration_and_holdouts"
        ),
        "publication_seeds": _check(
            isinstance(summary.get("seeds"), list)
            and len(summary.get("seeds")) >= 5
            and len(set(summary.get("seeds"))) == len(summary.get("seeds"))
        ),
        "training_randomness_protocol": _check(
            _valid_training_seed_records(summary)
        ),
        "table_iv_protocol": _check(
            int(summary.get("episodes", 0)) == 1000
            and int(summary.get("evaluation_episodes", 0)) > 0
        ),
        "physical_feasibility": _check(
            np.isclose(float(summary.get("feasibility_rate", -1.0)), 1.0)
            and np.isclose(derived_feasibility, 1.0)
            and np.isclose(
                float(summary.get("feasibility_rate", -1.0)),
                derived_feasibility,
            )
            and isinstance(summary.get("feasibility_provenance"), str)
            and "fail-closed" in summary.get("feasibility_provenance", "")
        ),
        "source_provenance": _check(
            _is_sha256(summary.get("source_tree_digest"))
            and summary.get("power_hypothesis_id")
            in {"paper_db_as_dbm", "paper_db_as_dbw"}
        ),
        "ambiguity_contract": _check(
            valid_ambiguity_manifest_fields(summary)
        ),
        "baseline_authority": _check(
            _is_sha256(summary.get("baseline_payload_sha256"))
            and _is_sha256(summary.get("baseline_validation_sha256"))
        ),
        "holdout_separation": _check(
            set(CALIBRATION_FIGURES).isdisjoint(HOLDOUT_FIGURES)
            and summary.get("calibration_figures") == list(CALIBRATION_FIGURES)
            and summary.get("holdout_figures") == list(HOLDOUT_FIGURES)
        ),
    }
    if verify_files:
        if result_directory is None:
            checks["file_provenance"] = _check(
                False, error="result_directory is required for file verification."
            )
        else:
            passed, error = _validate_completed_manifest(
                summary, result_directory
            )
            checks["file_provenance"] = _check(passed, error=error)
    else:
        checks["file_provenance"] = _check(
            True, verification="disabled_by_explicit_test_api"
        )

    checks["raw_payload_consistency"] = _check(payload_consistent)

    figure_checks = {}
    figures = summary.get("figures") if isinstance(summary.get("figures"), dict) else {}
    complete = set(figures) == {str(value) for value in PUBLICATION_FIGURES}
    checks["complete_figure_set"] = _check(complete)
    for figure in PUBLICATION_FIGURES:
        payload = figures.get(str(figure), {})
        qualitative = validate_figure_payload(figure, {
            "figure": figure,
            "x": payload.get("x"),
            "curves": payload.get("curves"),
        })
        try:
            reference = _reference_check(
                figure, payload, references, float(relative_tolerance)
            )
        except (KeyError, TypeError, ValueError):
            reference = _check(False, applicable=True, reason="malformed payload")
        figure_checks[str(figure)] = {
            "role": "calibration" if figure in CALIBRATION_FIGURES else "holdout",
            "qualitative_contract": _check(
                qualitative.get("passed", False),
                failures=qualitative.get("failures", []),
            ),
            "reference_tolerance": reference,
        }

    figures_pass = all(
        item["qualitative_contract"]["pass"]
        and item["reference_tolerance"]["pass"]
        for item in figure_checks.values()
    )
    return {
        "schema_version": 1,
        "validation": "paper_calibration_and_holdouts",
        "holdout_payload_sha256": result_payload_digest(summary),
        "paper_reference_sha256": paper_reference_digest(references),
        "overall_pass": (
            all(item["pass"] for item in checks.values()) and figures_pass
        ),
        "checks": checks,
        "figure_checks": figure_checks,
        "curves_modified": False,
        "figure6_reference_used_for_predeclared_calibration": True,
        "holdout_references_used_for_training_or_selection": False,
    }
