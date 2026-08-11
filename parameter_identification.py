"""Provenance-bounded structural identification without curve fitting or DRL."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np

from channel_model import ChannelSet
from config import cfg
from paper_contract import cascade_power_16c
from parameter_evidence import (
    PARAMETER_EVIDENCE,
    STRUCTURAL_IDENTIFICATION_KEYS,
)
from parameter_profiles import (
    active_profile_metadata,
    applied_profile,
    profile_from_mapping,
    write_profile,
)
from sinr_rate import (
    active_tx_power,
    active_tx_power_envelope,
    compute_rates,
    select_feasible_beamformers,
)


CALIBRATION_SEEDS = tuple(range(21_000, 21_008))
HOLDOUT_SEEDS = tuple(range(22_000, 22_008))
STRESS_SEEDS = tuple(range(23_000, 23_032))
ACTION_PROBE_LABELS = (
    "zero_active", "half_active", "full_active",
    "phase_variant_half", "phase_variant_full",
)
SENSING_FLOOR_SAFETY = 0.5
UPPER_CONSTRAINT_HEADROOM = 2.0
MINIMUM_COMM_ACTION_RESPONSE = 0.005

_SEARCH_KEYS = (
    "omega_0", "rho_br", "rho_rt", "zeta_los", "zeta_nlos",
    "ue_x_min", "ue_x_max", "ue_y_min", "ue_y_max",
)
_LOG_KEYS = {"omega_0", "rho_br", "rho_rt", "R_min", "Psi_max", "P_a_max"}

if frozenset(_SEARCH_KEYS) != STRUCTURAL_IDENTIFICATION_KEYS:
    raise RuntimeError("Structural search keys and evidence policy disagree")


def _log_scale(unit_value, low, high):
    return 10.0 ** (math.log10(low) + unit_value * (math.log10(high) - math.log10(low)))


def generate_candidates(count, seed=cfg.random_seed):
    """Generate a deterministic scrambled Sobol prefix within evidence bounds."""
    count = int(count)
    if count <= 0:
        raise ValueError("candidate count must be positive")
    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for deterministic parameter identification"
        ) from exc

    exponent = int(math.ceil(math.log2(count)))
    samples = qmc.Sobol(d=7, scramble=True, seed=int(seed)).random_base2(
        exponent
    )[:count]
    profiles = []
    for index, row in enumerate(samples):
        omega_0 = _log_scale(row[0], 1e-5, 5e-2)
        rho_br = _log_scale(row[1], 0.1, 30.0)
        rho_rt = _log_scale(row[2], 0.1, 30.0)
        zeta_los = 1.8 + row[3] * 1.0
        exponent_gap = 0.25 + row[4] * min(2.5, 5.5 - zeta_los - 0.25)
        zeta_nlos = zeta_los + exponent_gap
        ue_x_min = row[5] * 200.0
        ue_y_min = row[6] * 100.0
        # These paper-omitted limits are explicit assumptions, not quantities
        # that can be inferred from the channel candidate being scored.
        r_min = float(PARAMETER_EVIDENCE["R_min"].value)
        psi_max = float(PARAMETER_EVIDENCE["Psi_max"].value)
        p_a_max = float(PARAMETER_EVIDENCE["P_a_max"].value)
        values = {
            "omega_0": omega_0,
            "rho_br": rho_br,
            "rho_rt": rho_rt,
            "zeta_los": zeta_los,
            "zeta_nlos": zeta_nlos,
            "ue_x_min": ue_x_min,
            "ue_x_max": ue_x_min + 100.0,
            "ue_y_min": ue_y_min,
            "ue_y_max": ue_y_min + 100.0,
            "R_min": r_min,
            "Psi_max": psi_max,
            "P_a_max": p_a_max,
        }
        payload = {
            "schema_version": 1,
            "name": f"structural-candidate-{index:04d}",
            "kind": "candidate",
            "values": values,
            "evidence": {key: key for key in values},
            "selection": {
                "generator": "deterministic scrambled SciPy Sobol prefix",
                "generator_seed": int(seed),
                "candidate_index": index,
                "paper_curve_ordinates_used": False,
            },
        }
        profiles.append(profile_from_mapping(payload))
    return profiles


def profile_prior_distance(profile):
    """Mean normalized distance from evidence defaults; no result data used."""
    distances = []
    for key, value in profile.values.items():
        entry = PARAMETER_EVIDENCE[key]
        if not isinstance(entry.value, (int, float)) or entry.bounds is None:
            continue
        low, high = entry.bounds
        prior = float(entry.value)
        if key in _LOG_KEYS and low > 0.0 and prior > 0.0:
            denominator = max(math.log10(high) - math.log10(low), 1e-12)
            distance = abs(math.log10(value) - math.log10(prior)) / denominator
        else:
            denominator = max(high - low, 1e-12)
            distance = abs(value - prior) / denominator
        distances.append(float(distance))
    return float(np.mean(distances)) if distances else 0.0


def derive_constraint_values(raw_scenes):
    """Derive conservative limits from calibration extrema, not paper curves."""
    if not raw_scenes:
        raise ValueError("raw calibration scenes are required")
    for key in ("min_sensing_rate", "P_ai_envelope", "cascade_power"):
        values = np.asarray([row[key] for row in raw_scenes], dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"cannot derive {key} from non-finite scenes")
    sensing = np.asarray(
        [row["min_sensing_rate"] for row in raw_scenes], dtype=float
    )
    active = np.asarray(
        [row["P_ai_envelope"] for row in raw_scenes], dtype=float
    )
    cascade = np.asarray(
        [row["cascade_power"] for row in raw_scenes], dtype=float
    )
    return {
        "R_min": float(np.clip(
            SENSING_FLOOR_SAFETY * np.min(sensing), 1e-6, 4.0
        )),
        "P_a_max": float(np.clip(
            UPPER_CONSTRAINT_HEADROOM * np.max(active),
            max(float(cfg.sigma2_qa) * 1.001, 1e-12),
            10.0,
        )),
        "Psi_max": float(np.clip(
            UPPER_CONSTRAINT_HEADROOM * np.max(cascade), 1e-20, 1.0
        )),
    }


def _scene(seed):
    rng = np.random.default_rng(int(seed))
    pos_ues = [
        rng.uniform(
            [cfg.ue_x_min, cfg.ue_y_min, 0.0],
            [cfg.ue_x_max, cfg.ue_y_max, 0.0],
        )
        for _ in range(cfg.U)
    ]
    pos_tes = [
        rng.uniform(
            [cfg.x_min, cfg.y_min, 0.0],
            [cfg.x_max, cfg.y_max, 0.0],
        )
        for _ in range(cfg.K)
    ]
    pos_uav = np.array([
        (cfg.x_min + cfg.x_max) / 2.0,
        (cfg.y_min + cfg.y_max) / 2.0,
        cfg.z_UAV,
    ])
    return pos_uav, pos_ues, pos_tes


def _action_probes(seed):
    """Cover active-gain boundaries plus independent phase configurations."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7]))
    theta = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, cfg.Lp))
    unit_active = np.exp(
        1j * rng.uniform(0.0, 2.0 * np.pi, cfg.La)
    )
    theta_variant = np.exp(
        1j * rng.uniform(0.0, 2.0 * np.pi, cfg.Lp)
    )
    active_variant = np.exp(
        1j * rng.uniform(0.0, 2.0 * np.pi, cfg.La)
    )
    return (
        ("zero_active", theta, np.zeros(cfg.La, dtype=complex)),
        ("half_active", theta, (cfg.alpha_max / 2.0) * unit_active),
        ("full_active", theta, cfg.alpha_max * unit_active),
        (
            "phase_variant_half", theta_variant,
            (cfg.alpha_max / 2.0) * active_variant,
        ),
        (
            "phase_variant_full", np.conj(theta_variant),
            cfg.alpha_max * np.conj(active_variant),
        ),
    )


def _coefficient_of_variation(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.inf
    return float(np.std(values) / (abs(float(np.mean(values))) + 1e-12))


def evaluate_candidate(profile, seeds=CALIBRATION_SEEDS,
                       sensing_rate_min=None):
    """Evaluate raw channels and boundary actions; never train an agent.

    Pass ``profile=None`` to assess the already activated configuration,
    including its current executable ambiguity hypotheses.
    """
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    raw_scenes = []
    errors = []
    profile_context = (
        nullcontext() if profile is None else applied_profile(profile)
    )
    profile_digest = (
        active_profile_metadata()["digest"] if profile is None else profile.digest
    )
    with profile_context:
        for seed in seeds:
            try:
                pos_uav, pos_ues, pos_tes = _scene(seed)
                channel = ChannelSet(
                    pos_uav, pos_ues, pos_tes, cfg.rho_nmse, seed=seed
                )
                threshold = (
                    float(cfg.R_min) if sensing_rate_min is None
                    else float(sensing_rate_min)
                )
                p_ai_envelope = active_tx_power_envelope(channel)
                for probe_label, theta, phi in _action_probes(seed):
                    C, T, beam_info = select_feasible_beamformers(
                        channel, theta, phi, sensing_rate_min=threshold
                    )
                    rate_u, rate_k, _, _ = compute_rates(
                        channel, theta, phi, C, T
                    )
                    comm_rate = float(np.sum(rate_u))
                    sensing_rate = float(np.sum(rate_k))
                    min_sensing_rate = float(np.min(rate_k))
                    p_ai = active_tx_power(channel, T, phi)
                    cascade_power = cascade_power_16c(channel, theta)
                    total_power = float(
                        np.linalg.norm(C, "fro") ** 2
                        + np.linalg.norm(T, "fro") ** 2
                    )
                    finite = bool(np.all(np.isfinite([
                        comm_rate, sensing_rate, min_sensing_rate, p_ai,
                        p_ai_envelope, cascade_power, total_power,
                    ])))
                    sensing_ok = min_sensing_rate >= threshold - 1e-9
                    active_ok = p_ai_envelope <= cfg.P_a_max + 1e-12
                    cascade_ok = cascade_power <= cfg.Psi_max + 1e-30
                    power_ok = total_power <= cfg.PT + 1e-12
                    raw_scenes.append({
                        "seed": seed,
                        "action_probe": probe_label,
                        "comm_rate": comm_rate,
                        "sensing_rate": sensing_rate,
                        "min_sensing_rate": min_sensing_rate,
                        "P_ai": p_ai,
                        "P_ai_envelope": p_ai_envelope,
                        "cascade_power": cascade_power,
                        "total_power": total_power,
                        "finite": finite,
                        "sensing_feasible": bool(sensing_ok),
                        "active_feasible": bool(active_ok),
                        "cascade_feasible": bool(cascade_ok),
                        "power_contract_feasible": bool(power_ok),
                        "joint_feasible": bool(
                            finite and sensing_ok and active_ok
                            and cascade_ok and power_ok
                        ),
                        "sensing_utilization": (
                            min_sensing_rate / max(threshold, 1e-30)
                        ),
                        "active_utilization": (
                            p_ai_envelope / max(cfg.P_a_max, 1e-30)
                        ),
                        "cascade_utilization": (
                            cascade_power / max(cfg.Psi_max, 1e-30)
                        ),
                        "comm_power_frac": float(beam_info["comm_power_frac"]),
                    })
            except Exception as exc:  # fail closed and retain the exact seed.
                errors.append({
                    "seed": seed,
                    "type": type(exc).__name__,
                    "message": str(exc),
                })

    count = float(len(seeds) * len(ACTION_PROBE_LABELS))
    comm_rates = [row["comm_rate"] for row in raw_scenes]
    sensing_rates = [row["sensing_rate"] for row in raw_scenes]

    def grouped_values(key):
        return [
            np.asarray(
                [row[key] for row in raw_scenes if row["seed"] == seed],
                dtype=float,
            )
            for seed in seeds
        ]

    def action_spans(key):
        spans = []
        for values in grouped_values(key):
            if values.size != len(ACTION_PROBE_LABELS):
                continue
            spans.append(float(
                np.ptp(values) / (abs(float(np.mean(values))) + 1e-12)
            ))
        return spans

    comm_groups = grouped_values("comm_rate")
    sensing_groups = grouped_values("sensing_rate")
    comm_seed_means = [float(np.mean(values)) for values in comm_groups]
    sensing_seed_means = [float(np.mean(values)) for values in sensing_groups]
    comm_action_spans = action_spans("comm_rate")
    sensing_action_spans = action_spans("sensing_rate")

    def fraction(key):
        return sum(bool(row[key]) for row in raw_scenes) / count

    def median(key):
        values = [row[key] for row in raw_scenes]
        return float(np.median(values)) if values else math.inf

    return {
        "profile_digest": profile_digest,
        "seeds": list(seeds),
        "finite_fraction": fraction("finite"),
        "power_contract_fraction": fraction("power_contract_feasible"),
        "sensing_feasible_fraction": fraction("sensing_feasible"),
        "active_feasible_fraction": fraction("active_feasible"),
        "cascade_feasible_fraction": fraction("cascade_feasible"),
        "joint_feasible_fraction": fraction("joint_feasible"),
        "sensing_utilization_median": median("sensing_utilization"),
        "active_utilization_median": median("active_utilization"),
        "cascade_utilization_median": median("cascade_utilization"),
        "comm_rate_mean": float(np.mean(comm_rates)) if comm_rates else math.inf,
        "sensing_rate_mean": (
            float(np.mean(sensing_rates)) if sensing_rates else math.inf
        ),
        "comm_rate_cv": _coefficient_of_variation(comm_seed_means),
        "sensing_rate_cv": _coefficient_of_variation(sensing_seed_means),
        "comm_action_response_median": (
            float(np.median(comm_action_spans))
            if comm_action_spans else 0.0
        ),
        "comm_action_response_minimum": (
            float(np.min(comm_action_spans))
            if comm_action_spans else 0.0
        ),
        "sensing_action_response_median": (
            float(np.median(sensing_action_spans))
            if sensing_action_spans else 0.0
        ),
        "raw_scenes": raw_scenes,
        "errors": errors,
    }


def calibrate_constraint_profile(profile, seeds=CALIBRATION_SEEDS):
    """Replace omitted limits from unconstrained calibration extrema."""
    pilot_metrics = evaluate_candidate(
        profile, seeds=seeds, sensing_rate_min=0.0
    )
    if pilot_metrics["errors"] or pilot_metrics["finite_fraction"] != 1.0:
        raise RuntimeError("constraint calibration pilot did not produce finite scenes")
    derived = derive_constraint_values(pilot_metrics["raw_scenes"])
    raw_extrema = {
        "minimum_sensing_rate": float(min(
            row["min_sensing_rate"]
            for row in pilot_metrics["raw_scenes"]
        )),
        "maximum_active_power_envelope": float(max(
            row["P_ai_envelope"]
            for row in pilot_metrics["raw_scenes"]
        )),
        "maximum_cascade_power": float(max(
            row["cascade_power"]
            for row in pilot_metrics["raw_scenes"]
        )),
    }
    values = dict(profile.values)
    values.update(derived)
    payload = {
        "schema_version": 1,
        "name": profile.name,
        "kind": "candidate",
        "values": values,
        "evidence": {key: key for key in values},
        "selection": {
            **dict(profile.selection),
            "constraint_calibration": {
                "R_min": "0.5 times minimum raw target rate",
                "P_a_max": (
                    "2 times maximum analytical full-gain Eq. (18) envelope"
                ),
                "Psi_max": "2 times maximum raw Eq. (16c) cascade power",
                "action_probes": list(ACTION_PROBE_LABELS),
                "seeds": list(seeds),
                "communication_rate_magnitude_used": False,
                "raw_extrema": raw_extrema,
                "selected_values": dict(derived),
                "evidence_bounds_enforced": {
                    key: list(PARAMETER_EVIDENCE[key].bounds)
                    for key in ("R_min", "P_a_max", "Psi_max")
                },
            },
            "provisional_profile_digest": profile.digest,
        },
    }
    return profile_from_mapping(payload)


def _finite_numeric_tree(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite_numeric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numeric_tree(item) for item in value)
    return True


def _upper_limit_utilization_penalty(value):
    if value <= 0.0 or not math.isfinite(value):
        return 100.0
    if value < 0.05:
        return 2.0 * math.log10(0.05 / value) ** 2
    if value > 1.0:
        return 8.0 * math.log10(value) ** 2
    return 0.0


def _lower_floor_utilization_penalty(value):
    if value <= 0.0 or not math.isfinite(value):
        return 100.0
    if value < 1.0:
        return 8.0 * math.log10(1.0 / value) ** 2
    if value > 20.0:
        return 2.0 * math.log10(value / 20.0) ** 2
    return 0.0


def score_candidate(metrics, prior_distance):
    """Return a lower-is-better structural score with no rate-level target."""
    reasons = []
    if not _finite_numeric_tree(metrics):
        reasons.append("non-finite raw or aggregate metric")
    if metrics.get("finite_fraction") != 1.0:
        reasons.append("not every scene produced finite values")
    if metrics.get("power_contract_fraction") != 1.0:
        reasons.append("Eq. (16f) power contract failed")
    if metrics.get("errors"):
        reasons.append("one or more scene evaluations raised an error")
    if reasons:
        return {
            "score": math.inf,
            "rejected": True,
            "reasons": reasons,
            "components": {},
        }

    feasibility = 0.0
    for key in (
        "sensing_feasible_fraction", "active_feasible_fraction",
        "cascade_feasible_fraction",
    ):
        feasibility += 12.0 * max(0.0, 1.0 - float(metrics[key])) ** 2
    feasibility += 20.0 * max(
        0.0, 1.0 - float(metrics["joint_feasible_fraction"])
    ) ** 2
    utilization = _lower_floor_utilization_penalty(float(
        metrics["sensing_utilization_median"]
    )) + sum(_upper_limit_utilization_penalty(float(metrics[key])) for key in (
        "active_utilization_median", "cascade_utilization_median",
    ))
    stability = min(float(metrics["comm_rate_cv"]), 10.0) + min(
        float(metrics["sensing_rate_cv"]), 10.0
    )
    action_response = float(metrics["comm_action_response_median"])
    action_sensitivity = 20.0 * max(
        0.0,
        (MINIMUM_COMM_ACTION_RESPONSE - action_response)
        / MINIMUM_COMM_ACTION_RESPONSE,
    ) ** 2
    prior = 0.5 * float(prior_distance)
    components = {
        "feasibility": feasibility,
        "constraint_utilization": utilization,
        "seed_stability": stability,
        "action_sensitivity": action_sensitivity,
        "evidence_prior_distance": prior,
    }
    return {
        "score": float(sum(components.values())),
        "rejected": False,
        "reasons": [],
        "components": components,
    }


def structural_acceptance(metrics):
    """Apply fixed predeclared feasibility thresholds to a seed partition."""
    reasons = []
    if metrics.get("finite_fraction") != 1.0:
        reasons.append("finite fraction must equal 1")
    if metrics.get("power_contract_fraction") != 1.0:
        reasons.append("power contract fraction must equal 1")
    for key, label in (
        ("sensing_feasible_fraction", "sensing"),
        ("active_feasible_fraction", "active power"),
        ("cascade_feasible_fraction", "passive cascade"),
    ):
        if float(metrics.get(key, 0.0)) != 1.0:
            reasons.append(f"{label} feasibility must equal 1")
    if float(metrics.get("joint_feasible_fraction", 0.0)) != 1.0:
        reasons.append("joint feasibility must equal 1")
    if metrics.get("errors"):
        reasons.append("scene evaluation errors are not accepted")
    if float(metrics.get("comm_action_response_median", 0.0)) < (
        MINIMUM_COMM_ACTION_RESPONSE
    ):
        reasons.append(
            "median communication response across policy actions must be at "
            f"least {MINIMUM_COMM_ACTION_RESPONSE:.3g}"
        )
    return not reasons, reasons


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_search(output_dir, candidate_count=32, generator_seed=cfg.random_seed,
               calibration_seeds=CALIBRATION_SEEDS,
               holdout_seeds=HOLDOUT_SEEDS,
               stress_seeds=STRESS_SEEDS):
    """Rank on calibration, then require strict holdout and stress passes."""
    from source_integrity import source_tree_digest

    source_digest = source_tree_digest()
    calibration_seeds = tuple(calibration_seeds)
    holdout_seeds = tuple(holdout_seeds)
    stress_seeds = tuple(stress_seeds)
    if not calibration_seeds or not holdout_seeds or not stress_seeds:
        raise ValueError("calibration, holdout, and stress seeds are required")
    partitions = (set(calibration_seeds), set(holdout_seeds), set(stress_seeds))
    if any(
        partitions[i] & partitions[j]
        for i in range(3) for j in range(i + 1, 3)
    ):
        raise ValueError("calibration, holdout, and stress seeds must be disjoint")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # A failed refresh must not leave an older profile looking current.
    for stale_name in (
        "identified_profile.json", "parameter_search.json",
        "parameter_search.npz", "holdout_validation.json",
        "stress_validation.json",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    provisional_profiles = generate_candidates(
        candidate_count, seed=generator_seed
    )
    profiles = provisional_profiles
    records = []
    for profile in profiles:
        metrics = evaluate_candidate(profile, seeds=calibration_seeds)
        prior_distance = profile_prior_distance(profile)
        score = score_candidate(metrics, prior_distance)
        accepted, acceptance_reasons = structural_acceptance(metrics)
        records.append({
            "profile": profile,
            "metrics": metrics,
            "prior_distance": prior_distance,
            "score": score,
            "accepted": accepted,
            "acceptance_reasons": acceptance_reasons,
        })
    eligible = [
        record for record in records
        if not record["score"]["rejected"] and record["accepted"]
    ]
    if not eligible:
        raise RuntimeError("Every structural candidate was rejected")
    selected = min(eligible, key=lambda record: record["score"]["score"])
    holdout_metrics = evaluate_candidate(
        selected["profile"], seeds=holdout_seeds
    )
    holdout_score = score_candidate(
        holdout_metrics, selected["prior_distance"]
    )
    holdout_accepted, holdout_reasons = structural_acceptance(holdout_metrics)
    holdout_report = {
        "schema_version": 1,
        "source_tree_digest": source_digest,
        "source_candidate_digest": selected["profile"].digest,
        "seeds": list(holdout_seeds),
        "action_probes": list(ACTION_PROBE_LABELS),
        "metrics": holdout_metrics,
        "score": holdout_score,
        "accepted": holdout_accepted,
        "acceptance_reasons": holdout_reasons,
        "calibration_candidate_count": len(records),
        "calibration_accepted_count": sum(
            bool(record["accepted"]) for record in records
        ),
    }
    holdout_path = output_dir / "holdout_validation.json"
    with holdout_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(holdout_report), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    if holdout_score["rejected"] or not holdout_accepted:
        raise RuntimeError(
            "Selected profile failed held-out structural validation: "
            + "; ".join(holdout_score["reasons"] + holdout_reasons)
        )

    selected_profile = selected["profile"]
    identified_payload = {
        "schema_version": 1,
        "name": "identified-structural-profile",
        "kind": "identified",
        "values": dict(selected_profile.values),
        "evidence": dict(selected_profile.evidence),
        "selection": {
            "protocol": (
                "raw structural feasibility/stability search; no paper curve "
                "ordinates or communication-rate magnitude target"
            ),
            "source_candidate_digest": selected_profile.digest,
            "generator_seed": int(generator_seed),
            "calibration_seeds": list(calibration_seeds),
            "holdout_seeds": list(holdout_seeds),
            "stress_seeds": list(stress_seeds),
            "calibration_score": selected["score"]["score"],
            "holdout_score": holdout_score["score"],
            "paper_curve_ordinates_used": False,
            "source_tree_digest": source_digest,
        },
    }
    identified = profile_from_mapping(identified_payload)
    stress_metrics = evaluate_candidate(identified, seeds=stress_seeds)
    stress_score = score_candidate(
        stress_metrics, selected["prior_distance"]
    )
    stress_accepted, stress_reasons = structural_acceptance(stress_metrics)
    stress_report = {
        "schema_version": 1,
        "source_tree_digest": source_digest,
        "profile_digest": identified.digest,
        "seeds": list(stress_seeds),
        "action_probes": list(ACTION_PROBE_LABELS),
        "metrics": stress_metrics,
        "score": stress_score,
        "accepted": stress_accepted,
        "acceptance_reasons": stress_reasons,
    }
    stress_path = output_dir / "stress_validation.json"
    with stress_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(stress_report), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    if stress_score["rejected"] or not stress_accepted:
        raise RuntimeError(
            "Selected profile failed independent stress validation: "
            + "; ".join(stress_score["reasons"] + stress_reasons)
        )
    profile_path = write_profile(identified, output_dir / "identified_profile.json")

    manifest = {
        "schema_version": 1,
        "source_tree_digest": source_digest,
        "method": "structural parameter identification",
        "selection_rule": (
            "minimum predeclared structural score among strict calibration "
            "candidates; exactly one selected candidate is evaluated on "
            "holdout and stress partitions"
        ),
        "score_components": [
            "strict feasibility", "constraint utilization",
            "seed stability", "action sensitivity", "evidence prior distance",
        ],
        "constraint_policy": (
            "R_min, P_a_max, and Psi_max remain explicit fixed assumptions; "
            "they are not re-derived from each candidate's channel gains"
        ),
        "generator_seed": int(generator_seed),
        "calibration_seeds": list(calibration_seeds),
        "holdout_seeds": list(holdout_seeds),
        "stress_seeds": list(stress_seeds),
        "action_probes": list(ACTION_PROBE_LABELS),
        "search_keys": list(_SEARCH_KEYS),
        "paper_curve_ordinates_used": False,
        "communication_rate_magnitude_used_in_score": False,
        "selected_profile_digest": identified.digest,
        "selected_source_candidate_digest": selected_profile.digest,
        "holdout_metrics": holdout_metrics,
        "holdout_score": holdout_score,
        "holdout_accepted": holdout_accepted,
        "holdout_acceptance_reasons": holdout_reasons,
        "stress_metrics": stress_metrics,
        "stress_score": stress_score,
        "stress_accepted": stress_accepted,
        "stress_acceptance_reasons": stress_reasons,
        "candidates": [
            {
                "profile_digest": record["profile"].digest,
                "values": dict(record["profile"].values),
                "prior_distance": record["prior_distance"],
                "metrics": record["metrics"],
                "score": record["score"],
                "accepted": record["accepted"],
                "acceptance_reasons": record["acceptance_reasons"],
            }
            for record in records
        ],
    }
    json_path = output_dir / "parameter_search.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(manifest), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")

    value_matrix = np.asarray([
        [record["profile"].values[key] for key in _SEARCH_KEYS]
        for record in records
    ], dtype=float)
    score_vector = np.asarray([
        record["score"]["score"] for record in records
    ], dtype=float)
    npz_path = output_dir / "parameter_search.npz"
    np.savez(
        npz_path,
        parameter_names=np.asarray(_SEARCH_KEYS),
        candidate_values=value_matrix,
        structural_scores=score_vector,
    )
    return {
        "profile": identified,
        "profile_path": profile_path,
        "json_path": json_path,
        "npz_path": npz_path,
        "holdout_path": holdout_path,
        "stress_path": stress_path,
        "manifest": manifest,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Identify omitted parameters from raw structural checks"
    )
    parser.add_argument("--output", default="result/parameter_identification")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_search(
        args.output, candidate_count=args.candidates, generator_seed=args.seed
    )
    print(f"Identified profile: {result['profile_path']}")
    print(f"Profile SHA-256: {result['profile'].digest}")
