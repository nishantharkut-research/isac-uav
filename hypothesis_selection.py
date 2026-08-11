"""Predeclared Figure 6-only selection for the ambiguous power reference."""

import json
import os
from pathlib import Path

import numpy as np

from paper_reference import load_paper_references, paper_reference_digest
from physical_hypotheses import POWER_HYPOTHESES


SELECTION_SCHEMA_VERSION = 1
CALIBRATION_FIGURE = 6
CALIBRATION_METHODS = ("AO-pCSI", "AO-iCSI")
MAX_NORMALIZED_CURVE_ERROR = 0.15
MIN_POWER_RESPONSE_RATIO = 0.50


def _curve(candidate, reference, label):
    predictions = candidate.get("predictions", {})
    if set(predictions) != {str(CALIBRATION_FIGURE)}:
        raise ValueError("Power-hypothesis selection may use Figure 6 only")
    generated = np.asarray(
        predictions[str(CALIBRATION_FIGURE)].get(label, []), dtype=float
    )
    observed = np.asarray(reference["curves"].get(label, []), dtype=float)
    if generated.shape != observed.shape or generated.ndim != 1:
        raise ValueError(f"Invalid Figure 6 prediction for {label}")
    if generated.size < 2 or not np.all(np.isfinite(generated)):
        raise ValueError(f"Figure 6 prediction for {label} must be finite")
    return generated, observed


def score_power_hypothesis(candidate: dict, reference_source) -> dict:
    """Score one hypothesis without reading any holdout ordinate."""
    references = load_paper_references(reference_source)
    if set(references.get("calibration_figures", [])) != {CALIBRATION_FIGURE}:
        raise ValueError("Power-hypothesis selection requires Figure 6 alone")
    if candidate.get("holdout_figures_used") != []:
        raise ValueError("Power-hypothesis selection cannot use a holdout figure")
    if int(candidate.get("calibration_figure", -1)) != CALIBRATION_FIGURE:
        raise ValueError("Power-hypothesis selection requires calibration Figure 6")

    identifier = candidate.get("hypothesis_id")
    if identifier not in POWER_HYPOTHESES:
        raise ValueError(f"Unknown power hypothesis: {identifier}")
    seeds = candidate.get("common_random_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 5
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("Hypothesis scoring requires five unique nonnegative seeds")
    feasible = candidate.get("all_points_feasible")
    if not isinstance(feasible, bool):
        raise ValueError("Candidate must declare all_points_feasible as a boolean")
    if not feasible:
        prediction_keys = set(candidate.get("predictions", {}))
        if prediction_keys - {str(CALIBRATION_FIGURE)}:
            raise ValueError("An infeasible candidate contains non-calibration data")
        rejected_score = 1e12
        return {
            "hypothesis_id": identifier,
            "fully_feasible": False,
            "calibration_admissible": False,
            "ordering_violations": int(rejected_score),
            "power_direction_violations": int(rejected_score),
            "minimum_power_response_ratio": 0.0,
            "normalized_curve_error": rejected_score,
            "normalized_slope_error": rejected_score,
            "score_tuple": [int(rejected_score), rejected_score, rejected_score],
            "common_random_seeds": list(seeds),
        }

    reference = references["figures"][str(CALIBRATION_FIGURE)]
    uncertainty = float(reference["uncertainty"]["absolute_y"])
    generated_curves = {}
    observed_curves = {}
    curve_residuals = []
    slope_residuals = []
    power_direction_violations = 0
    power_response_ratios = []
    for label in CALIBRATION_METHODS:
        generated, observed = _curve(candidate, reference, label)
        generated_curves[label] = generated
        observed_curves[label] = observed
        generated_slopes = np.diff(generated)
        observed_response = float(observed[-1] - observed[0])
        if observed_response <= 0.0:
            raise ValueError(
                f"Figure 6 reference for {label} must increase with power"
            )
        power_direction_violations += int(
            np.sum(generated_slopes < -uncertainty)
        )
        power_response_ratios.append(
            float((generated[-1] - generated[0]) / observed_response)
        )
        curve_scale = np.maximum(np.abs(observed), uncertainty)
        curve_residuals.extend(((generated - observed) / curve_scale).tolist())
        observed_slope = np.diff(observed)
        generated_slope = np.diff(generated)
        slope_scale = np.maximum(np.abs(observed_slope), uncertainty)
        slope_residuals.extend(
            ((generated_slope - observed_slope) / slope_scale).tolist()
        )

    ordering_violations = 0
    for left_index, left in enumerate(CALIBRATION_METHODS):
        for right in CALIBRATION_METHODS[left_index + 1:]:
            expected = np.sign(observed_curves[left] - observed_curves[right])
            actual = np.sign(generated_curves[left] - generated_curves[right])
            ordering_violations += int(np.sum(actual != expected))

    curve_error = float(np.sqrt(np.mean(np.square(curve_residuals))))
    slope_error = float(np.sqrt(np.mean(np.square(slope_residuals))))
    minimum_response = float(min(power_response_ratios))
    admissible = bool(
        feasible
        and ordering_violations == 0
        and power_direction_violations == 0
        and minimum_response >= MIN_POWER_RESPONSE_RATIO
        and curve_error <= MAX_NORMALIZED_CURVE_ERROR
    )
    return {
        "hypothesis_id": identifier,
        "fully_feasible": feasible,
        "calibration_admissible": admissible,
        "ordering_violations": ordering_violations,
        "power_direction_violations": power_direction_violations,
        "minimum_power_response_ratio": minimum_response,
        "normalized_curve_error": curve_error,
        "normalized_slope_error": slope_error,
        "score_tuple": [ordering_violations, curve_error, slope_error],
        "common_random_seeds": list(seeds),
    }


def select_power_hypothesis(candidates, reference_source) -> dict:
    """Select by feasibility then the locked Figure 6 score tuple."""
    candidates = list(candidates)
    identifiers = {candidate.get("hypothesis_id") for candidate in candidates}
    if identifiers != set(POWER_HYPOTHESES) or len(candidates) != len(POWER_HYPOTHESES):
        raise ValueError("Selection requires exactly both predeclared hypotheses")
    scored = [score_power_hypothesis(item, reference_source) for item in candidates]
    seed_sets = {tuple(item["common_random_seeds"]) for item in scored}
    if len(seed_sets) != 1:
        raise ValueError("Power hypotheses must use identical common random seeds")
    feasible = [item for item in scored if item["fully_feasible"]]
    if not feasible:
        raise ValueError("No power hypothesis is fully feasible")
    admissible = [item for item in feasible if item["calibration_admissible"]]
    if not admissible:
        raise ValueError(
            "No power hypothesis satisfies the physical-ordering and Figure 6 "
            "calibration-fidelity gates"
        )
    selected = min(
        admissible,
        key=lambda item: (
            item["normalized_curve_error"],
            item["normalized_slope_error"],
            item["hypothesis_id"],
        ),
    )
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "kind": "power_hypothesis_selection_lock",
        "locked": True,
        "calibration_figures_used": [CALIBRATION_FIGURE],
        "holdout_figures_used": [],
        "selection_rule": (
            "fully feasible; zero pCSI/iCSI ordering and material power-"
            "direction violations; at least 50% of the reference power "
            "response; normalized curve error <= 0.15; then normalized curve "
            "and slope error"
        ),
        "admissibility_thresholds": {
            "maximum_normalized_curve_error": MAX_NORMALIZED_CURVE_ERROR,
            "minimum_power_response_ratio": MIN_POWER_RESPONSE_RATIO,
        },
        "common_random_seeds": list(next(iter(seed_sets))),
        "candidate_scores": sorted(scored, key=lambda item: item["hypothesis_id"]),
        "selected_hypothesis_id": selected["hypothesis_id"],
        "paper_reference_sha256": paper_reference_digest(reference_source),
    }


def write_selection_lock(path, payload):
    """Atomically create a lock; an existing different lock is immutable."""
    path = Path(path)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("Power-hypothesis selection lock is immutable")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
