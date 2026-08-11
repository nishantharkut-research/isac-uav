"""Bounded calibration of omitted parameters against declared paper curves."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from paper_reference import load_paper_references
from parameter_evidence import (
    PARAMETER_EVIDENCE,
    PROFILE_BOUNDS,
    PROFILE_KEYS,
    SENSITIVITY_ONLY_KEYS,
)


CALIBRATION_TARGETS = {
    6: ("AO-pCSI", "AO-iCSI"),
}
HOLDOUT_FIGURES = frozenset({4, 5, 7, 8, 9, 13, 14})
FAILED_CANDIDATE_LOSS = 1e6
DEFAULT_CALIBRATION_KEYS = (
    "omega_0", "rho_br", "rho_rt", "zeta_los", "zeta_nlos",
    "ue_x_min", "ue_y_min",
)
LOG_SCALED_KEYS = frozenset({"omega_0", "rho_br", "rho_rt"})
FIXED_SENSITIVITY_PROTOCOL = "predeclared_not_result_selected"


@dataclass(frozen=True)
class CalibrationResult:
    selected_values: dict
    loss: float
    records: list
    optimizer_success: bool
    optimizer_message: str
    prior_weight: float = 0.05


def parse_fixed_sensitivity_values(items: Sequence[str]) -> dict[str, float]:
    """Parse predeclared sensitivity values without using result quality."""
    values = {}
    for item in items:
        if not isinstance(item, str) or item.count("=") != 1:
            raise ValueError(
                "Each fixed sensitivity must use key=value syntax."
            )
        raw_key, raw_value = item.split("=", 1)
        key = raw_key.strip()
        value_text = raw_value.strip()
        if key not in SENSITIVITY_ONLY_KEYS:
            raise ValueError(
                f"Parameter {key or '<empty>'} is not sensitivity-only."
            )
        if key in values:
            raise ValueError(f"Duplicate fixed sensitivity parameter: {key}")
        try:
            value = float(value_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Fixed sensitivity {key} must be numeric."
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Fixed sensitivity {key} must be finite.")
        low, high = PROFILE_BOUNDS[key]
        if value < low or value > high:
            raise ValueError(
                f"Fixed sensitivity {key}={value} is outside [{low}, {high}]."
            )
        values[key] = value
    return values


def with_fixed_sensitivity_values(base_profile, values: Mapping[str, float]):
    """Derive a profile whose sensitivity values were fixed before calibration."""
    from parameter_profiles import profile_from_mapping

    normalized = parse_fixed_sensitivity_values(
        [f"{key}={value}" for key, value in values.items()]
    )
    if not normalized:
        return base_profile
    merged_values = dict(base_profile.values)
    merged_values.update(normalized)
    evidence = dict(base_profile.evidence)
    evidence.update({key: key for key in normalized})
    selection = dict(base_profile.selection)
    selection.update({
        "fixed_sensitivity_protocol": FIXED_SENSITIVITY_PROTOCOL,
        "fixed_sensitivity_values": normalized,
        "source_profile_digest": base_profile.digest,
    })
    return profile_from_mapping({
        "schema_version": 1,
        "name": f"{base_profile.name}-fixed-sensitivity",
        "kind": base_profile.kind,
        "values": merged_values,
        "evidence": evidence,
        "selection": selection,
    })


def validate_calibration_keys(keys: Sequence[str]):
    keys = tuple(keys)
    if not keys:
        raise ValueError("At least one calibration key is required.")
    for key in keys:
        entry = PARAMETER_EVIDENCE.get(key)
        if entry is None:
            raise ValueError(f"Unknown calibration parameter: {key}")
        if entry.evidence_class == "journal_reported":
            raise ValueError(f"Cannot calibrate journal-reported parameter {key}.")
        if key not in PROFILE_KEYS or PROFILE_BOUNDS.get(key) is None:
            raise ValueError(f"Parameter {key} is not profile-calibratable.")
        low, high = PROFILE_BOUNDS[key]
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"Parameter {key} has no usable calibration bounds.")
    return keys


def calibration_loss(predictions: Mapping, references: Mapping,
                     targets: Mapping = CALIBRATION_TARGETS) -> dict:
    """Return robust relative error using calibration figures only."""
    target_ids = {int(figure) for figure in targets}
    if target_ids & HOLDOUT_FIGURES:
        raise ValueError("A holdout figure cannot enter calibration loss.")
    components = {}
    for figure_number, labels in targets.items():
        key = str(figure_number)
        if key not in references["figures"]:
            raise ValueError(f"Missing reference figure {figure_number}.")
        if key not in predictions:
            raise ValueError(f"Missing prediction figure {figure_number}.")
        figure = references["figures"][key]
        uncertainty = float(figure["uncertainty"]["absolute_y"])
        for label in labels:
            if label not in figure["curves"] or label not in predictions[key]:
                raise ValueError(f"Missing calibration curve {key}/{label}.")
            observed = np.asarray(figure["curves"][label], dtype=float)
            predicted = np.asarray(predictions[key][label], dtype=float)
            if predicted.shape != observed.shape or not np.all(np.isfinite(predicted)):
                raise ValueError(f"Invalid prediction curve {key}/{label}.")
            scale = np.maximum(np.abs(observed), uncertainty)
            residual = (predicted - observed) / scale
            absolute = np.abs(residual)
            huber = np.where(absolute <= 1.0, 0.5 * residual ** 2,
                             absolute - 0.5)
            components[f"{key}/{label}"] = float(np.mean(huber))
    return {
        "total": float(np.mean(list(components.values()))),
        "components": components,
    }


def _decode(unit_values, keys):
    values = {}
    for key, unit in zip(keys, unit_values):
        low, high = PROFILE_BOUNDS[key]
        if key in LOG_SCALED_KEYS:
            value = np.exp(np.log(low) + float(unit) * (np.log(high) - np.log(low)))
        else:
            value = low + float(unit) * (high - low)
        values[key] = float(value)
    if "zeta_los" in values and "zeta_nlos" in values:
        values["zeta_nlos"] = max(
            values["zeta_nlos"], values["zeta_los"] + 1e-6
        )
    if "ue_x_min" in values:
        values["ue_x_max"] = values["ue_x_min"] + 100.0
    if "ue_y_min" in values:
        values["ue_y_max"] = values["ue_y_min"] + 100.0
    return values


def _encode(values, keys):
    encoded = []
    for key in keys:
        low, high = PROFILE_BOUNDS[key]
        value = float(values[key])
        if key in LOG_SCALED_KEYS:
            unit = (np.log(value) - np.log(low)) / (np.log(high) - np.log(low))
        else:
            unit = (value - low) / (high - low)
        encoded.append(float(np.clip(unit, 0.0, 1.0)))
    return np.asarray(encoded, dtype=float)


def run_bounded_calibration(
    predictor: Callable[[Mapping[str, float]], Mapping],
    initial_values: Mapping[str, float],
    reference_source,
    keys: Sequence[str] = DEFAULT_CALIBRATION_KEYS,
    max_iterations: int = 100,
    prior_weight: float = 0.05,
) -> CalibrationResult:
    """Use bounded SciPy Powell search and retain every evaluated candidate.

    The prior is measured in normalized bound coordinates and discourages
    movement away from the evidence-derived starting profile unless the AO
    reference error improves enough to justify it.
    """
    keys = validate_calibration_keys(keys)
    prior_weight = float(prior_weight)
    if not np.isfinite(prior_weight) or prior_weight < 0.0:
        raise ValueError("prior_weight must be finite and non-negative.")
    references = load_paper_references(reference_source)
    records = []
    initial_unit = _encode(initial_values, keys)

    def objective(unit_values):
        unit_values = np.asarray(unit_values, dtype=float)
        values = _decode(unit_values, keys)
        prior = prior_weight * float(np.mean((unit_values - initial_unit) ** 2))
        try:
            predictions = predictor(values)
            reference_loss = calibration_loss(predictions, references)
            loss = {
                "total": float(reference_loss["total"] + prior),
                "reference_total": float(reference_loss["total"]),
                "prior": float(prior),
                "components": reference_loss["components"],
            }
            record = {
                "candidate_index": len(records),
                "values": values,
                "predictions": predictions,
                "valid": True,
                "error": None,
                "loss": loss,
            }
        except Exception as exc:
            loss = {
                "total": float(FAILED_CANDIDATE_LOSS + prior),
                "reference_total": None,
                "prior": float(prior),
                "components": {},
            }
            record = {
                "candidate_index": len(records),
                "values": values,
                "predictions": None,
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "loss": loss,
            }
        records.append(record)
        return loss["total"]

    solution = minimize(
        objective,
        initial_unit,
        method="Powell",
        bounds=[(0.0, 1.0)] * len(keys),
        options={"maxiter": int(max_iterations), "xtol": 1e-4, "ftol": 1e-5},
    )
    selected = _decode(solution.x, keys)
    selected_loss = float(objective(solution.x))
    selected_valid = bool(records[-1]["valid"])
    return CalibrationResult(
        selected_values=selected,
        loss=selected_loss,
        records=records,
        optimizer_success=bool(solution.success and selected_valid),
        optimizer_message=str(solution.message),
        prior_weight=prior_weight,
    )


def write_calibration_report(result: CalibrationResult, output_path,
                             profile_digest=None, common_random_seeds=None,
                             ao_iterations=None,
                             optimizer_max_iterations=None,
                             paper_reference_sha256=None,
                             power_hypothesis_id="paper_db_as_dbm",
                             calibration_noise_hypothesis_id=
                             "description_literal_target_matches_ue",
                             los_probability_hypothesis_id=
                             "paper_eq6_literal_roles",
                             cross_function_interference_hypothesis_id=
                             "paper_eq12_eq14_literal_cross_channels",
                             fixed_sensitivity_values=None):
    seeds = [int(seed) for seed in (common_random_seeds or ())]
    if len(seeds) < 5 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Calibration report requires at least five unique seeds.")
    ao_iterations = int(ao_iterations)
    optimizer_max_iterations = int(optimizer_max_iterations)
    if ao_iterations <= 0 or optimizer_max_iterations <= 0:
        raise ValueError("Calibration iteration budgets must be positive.")
    if (
        not isinstance(paper_reference_sha256, str)
        or len(paper_reference_sha256) != 64
        or any(character not in "0123456789abcdef" for character in paper_reference_sha256)
    ):
        raise ValueError("Calibration report requires a paper-reference SHA-256.")
    payload = {
        "schema_version": 1,
        "method": "bounded_scipy_powell",
        "power_hypothesis_id": power_hypothesis_id,
        "calibration_noise_hypothesis_id":
            calibration_noise_hypothesis_id,
        "los_probability_hypothesis_id": los_probability_hypothesis_id,
        "cross_function_interference_hypothesis_id":
            cross_function_interference_hypothesis_id,
        "fixed_sensitivity_protocol": (
            FIXED_SENSITIVITY_PROTOCOL
            if fixed_sensitivity_values else None
        ),
        "fixed_sensitivity_values": dict(fixed_sensitivity_values or {}),
        "calibration_targets": {
            str(key): list(value) for key, value in CALIBRATION_TARGETS.items()
        },
        "holdout_figures_used": [],
        "selected_values": result.selected_values,
        "selected_loss": result.loss,
        "prior_weight": float(result.prior_weight),
        "common_random_seeds": seeds,
        "ao_iterations": ao_iterations,
        "optimizer_max_iterations": optimizer_max_iterations,
        "optimizer_success": result.optimizer_success,
        "optimizer_message": result.optimizer_message,
        "profile_digest": profile_digest,
        "paper_reference_sha256": paper_reference_sha256,
        "records": result.records,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def _scene(seed, cfg):
    rng = np.random.default_rng(int(seed))
    users = [
        rng.uniform(
            [cfg.ue_x_min, cfg.ue_y_min, 0.0],
            [cfg.ue_x_max, cfg.ue_y_max, 0.0],
        )
        for _ in range(cfg.U)
    ]
    targets = [
        rng.uniform(
            [cfg.x_min, cfg.y_min, 0.0],
            [cfg.x_max, cfg.y_max, 0.0],
        )
        for _ in range(cfg.K)
    ]
    uav = np.array([
        (cfg.x_min + cfg.x_max) / 2.0,
        (cfg.y_min + cfg.y_max) / 2.0,
        cfg.z_UAV,
    ])
    return uav, users, targets


def make_ao_calibration_predictor(base_profile, seeds=(21_000, 21_001, 21_002),
                                  ao_iterations=6,
                                  power_hypothesis_id="paper_db_as_dbm",
                                  los_probability_hypothesis_id=
                                  "paper_eq6_literal_roles"):
    """Build a common-random-number AO predictor for Fig. 6 AO curves."""
    from ao_optimizer import run_ao
    from config import cfg
    from parameter_profiles import applied_profile, profile_from_mapping
    from physical_hypotheses import apply_power_hypothesis, get_power_hypothesis
    from ambiguity_hypotheses import applied_ambiguity_hypotheses

    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("AO calibration requires at least one seed.")
    get_power_hypothesis(power_hypothesis_id)

    def predictor(candidate_values):
        merged = dict(base_profile.values)
        merged.update(candidate_values)
        payload = {
            "schema_version": 1,
            "name": "ao-calibration-candidate",
            "kind": "candidate",
            "values": merged,
            "evidence": {key: key for key in merged},
            "selection": {"purpose": "Fig. 6/7 calibration candidate"},
        }
        profile = profile_from_mapping(payload)
        predictions = {"6": {"AO-pCSI": [], "AO-iCSI": []}}
        with applied_profile(profile), applied_ambiguity_hypotheses(
            cfg,
            los_probability_hypothesis_id=los_probability_hypothesis_id,
        ):
            power_fields = (
                "power_hypothesis_id", "PT_dB", "PT_reference",
                "PT_display_unit", "PT", "PT_dBm", "PT_dBW",
                "power_hypothesis_metadata",
            )
            saved_power = {
                field: getattr(cfg, field) for field in power_fields
            }
            try:
                csi_conditions = (
                    ("AO-pCSI", 0.0),
                    ("AO-iCSI", float(cfg.rho_nmse)),
                )
                for label, sigma2 in csi_conditions:
                    for power_db in (10, 15, 20, 25, 30):
                        apply_power_hypothesis(
                            cfg, power_hypothesis_id, float(power_db)
                        )
                        values = []
                        for seed in seeds:
                            uav, users, targets = _scene(seed, cfg)
                            result = run_ao(
                                uav, users, targets, sigma2=sigma2,
                                n_iter=int(ao_iterations), fix_uav=False,
                                stop_on_convergence=True, seed=seed,
                            )
                            if not result.get("is_feasible", False):
                                raise RuntimeError(
                                    "infeasible AO calibration candidate for "
                                    f"{label}, P_T={power_db} dB, seed={seed}"
                                )
                            values.append(float(result["sum_rate"]))
                        predictions["6"][label].append(float(np.mean(values)))
            finally:
                for field, value in saved_power.items():
                    setattr(cfg, field, value)
        return predictions

    return predictor


def freeze_calibrated_profile(base_profile, result, output_path,
                              report_name="calibration_report.json",
                              paper_reference_sha256=None,
                              power_hypothesis_id="paper_db_as_dbm",
                              calibration_noise_hypothesis_id=
                              "description_literal_target_matches_ue",
                              los_probability_hypothesis_id=
                              "paper_eq6_literal_roles",
                              cross_function_interference_hypothesis_id=
                              "paper_eq12_eq14_literal_cross_channels",
                              fixed_sensitivity_values=None):
    """Merge selected values into a new immutable identified profile."""
    from parameter_profiles import profile_from_mapping, write_profile

    values = dict(base_profile.values)
    values.update(result.selected_values)
    if (
        not isinstance(paper_reference_sha256, str)
        or len(paper_reference_sha256) != 64
    ):
        raise ValueError("Calibrated profile requires a paper-reference SHA-256.")
    fixed_sensitivity_values = dict(fixed_sensitivity_values or {})
    if fixed_sensitivity_values:
        declared = base_profile.selection.get("fixed_sensitivity_values")
        if declared != fixed_sensitivity_values:
            raise ValueError(
                "Fixed sensitivities do not match the calibration base profile."
            )
    payload = {
        "schema_version": 1,
        "name": "paper-calibrated-identified-profile",
        "kind": "identified",
        "values": values,
        "evidence": {key: key for key in values},
        "selection": {
            "protocol": "bounded AO calibration on paper Fig. 6 AO curves",
            "power_hypothesis_id": power_hypothesis_id,
            "calibration_noise_hypothesis_id":
                calibration_noise_hypothesis_id,
            "los_probability_hypothesis_id":
                los_probability_hypothesis_id,
            "cross_function_interference_hypothesis_id":
                cross_function_interference_hypothesis_id,
            "fixed_sensitivity_protocol": (
                FIXED_SENSITIVITY_PROTOCOL if fixed_sensitivity_values else None
            ),
            "fixed_sensitivity_values": fixed_sensitivity_values,
            "fixed_sensitivity_source_profile_digest": (
                base_profile.selection.get("source_profile_digest")
                if fixed_sensitivity_values else None
            ),
            "source_profile_digest": base_profile.digest,
            "calibration_report": report_name,
            "paper_reference_sha256": paper_reference_sha256,
            "calibration_targets": {
                str(key): list(labels)
                for key, labels in CALIBRATION_TARGETS.items()
            },
            "holdout_figures_used": [],
            "selected_loss": result.loss,
            "prior_weight": float(result.prior_weight),
            "paper_curve_ordinates_used": True,
            "communication_rate_magnitude_used_in_selection": True,
        },
    }
    profile = profile_from_mapping(payload)
    write_profile(profile, output_path)
    return profile


def hypothesis_candidate_from_calibration(result, hypothesis_id, seeds):
    """Build the selector input without exposing any holdout data."""
    from physical_hypotheses import get_power_hypothesis

    get_power_hypothesis(hypothesis_id)
    records = list(result.records)
    selected = records[-1] if records else {}
    valid = bool(
        result.optimizer_success
        and selected.get("valid") is True
        and isinstance(selected.get("predictions"), Mapping)
        and set(selected["predictions"]) == {"6"}
    )
    return {
        "hypothesis_id": hypothesis_id,
        "calibration_figure": 6,
        "holdout_figures_used": [],
        "common_random_seeds": [int(seed) for seed in seeds],
        "all_points_feasible": valid,
        "predictions": selected.get("predictions") if valid else {},
    }


def main(argv=None):
    import argparse
    from hypothesis_selection import (
        select_power_hypothesis,
        write_selection_lock,
    )
    from parameter_profiles import load_prepared_profile
    from paper_reference import paper_reference_digest
    from physical_hypotheses import POWER_HYPOTHESES
    from ambiguity_hypotheses import (
        LOS_PROBABILITY_HYPOTHESES,
        PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", required=True)
    parser.add_argument(
        "--reference", default="references/paper_reference_curves.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="21000,21001,21002,21003,21004")
    parser.add_argument("--ao-iterations", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--los-probability-hypothesis",
        choices=tuple(LOS_PROBABILITY_HYPOTHESES),
        default=PRIMARY_LOS_PROBABILITY_HYPOTHESIS_ID,
    )
    parser.add_argument(
        "--fixed-sensitivity",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "predeclare an omitted sensitivity-only parameter without "
            "selecting it from paper-result quality; repeat as needed"
        ),
    )
    args = parser.parse_args(argv)
    reference_digest = paper_reference_digest(args.reference)

    prepared_base = load_prepared_profile(args.base_profile)
    fixed_sensitivity_values = parse_fixed_sensitivity_values(
        args.fixed_sensitivity
    )
    base = with_fixed_sensitivity_values(
        prepared_base, fixed_sensitivity_values
    )
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if len(seeds) < 5 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Power-hypothesis calibration requires at least five unique nonnegative seeds.")
    initial = {
        key: float(base.values.get(key, PARAMETER_EVIDENCE[key].value))
        for key in DEFAULT_CALIBRATION_KEYS
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(
            "Calibration output must be a new empty directory; prior calibration "
            "artifacts are immutable."
        )
    candidates = []
    artifacts = {}
    for hypothesis_id in POWER_HYPOTHESES:
        branch = output / hypothesis_id
        branch.mkdir(parents=True, exist_ok=True)
        report_path = branch / "calibration_report.json"
        profile_path = branch / "calibrated_profile.json"
        predictor = make_ao_calibration_predictor(
            base,
            seeds=seeds,
            ao_iterations=args.ao_iterations,
            power_hypothesis_id=hypothesis_id,
            los_probability_hypothesis_id=
                args.los_probability_hypothesis,
        )
        result = run_bounded_calibration(
            predictor,
            initial,
            args.reference,
            max_iterations=args.max_iterations,
        )
        profile = None
        if result.optimizer_success:
            profile = freeze_calibrated_profile(
                base,
                result,
                profile_path,
                paper_reference_sha256=reference_digest,
                power_hypothesis_id=hypothesis_id,
                los_probability_hypothesis_id=
                    args.los_probability_hypothesis,
                fixed_sensitivity_values=fixed_sensitivity_values,
            )
        elif profile_path.exists():
            profile_path.unlink()
        write_calibration_report(
            result,
            report_path,
            profile_digest=profile.digest if profile is not None else None,
            common_random_seeds=seeds,
            ao_iterations=args.ao_iterations,
            optimizer_max_iterations=args.max_iterations,
            paper_reference_sha256=reference_digest,
            power_hypothesis_id=hypothesis_id,
            los_probability_hypothesis_id=
                args.los_probability_hypothesis,
            fixed_sensitivity_values=fixed_sensitivity_values,
        )
        candidates.append(hypothesis_candidate_from_calibration(
            result, hypothesis_id, seeds
        ))
        artifacts[hypothesis_id] = {
            "calibration_report": str(report_path.relative_to(output)),
            "calibrated_profile": (
                str(profile_path.relative_to(output)) if profile is not None else None
            ),
        }

    lock = select_power_hypothesis(candidates, args.reference)
    selected_id = lock["selected_hypothesis_id"]
    selected_profile = artifacts[selected_id]["calibrated_profile"]
    if selected_profile is None:
        raise RuntimeError("Selected power hypothesis has no calibrated profile")
    lock["candidate_artifacts"] = artifacts
    lock["selected_profile"] = selected_profile
    lock_path = write_selection_lock(output / "power_hypothesis_lock.json", lock)
    print(f"Power-hypothesis lock: {lock_path}")
    print(f"Selected calibrated profile: {output / selected_profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
