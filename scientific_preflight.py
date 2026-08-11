"""Bounded, no-training feasibility preflight for an activated profile."""
import json
import math
from pathlib import Path

import numpy as np

from config import cfg
from environment import HRISUAVISACEnv
from parameter_profiles import active_profile_metadata
from sinr_rate import active_tx_power_envelope


PREFLIGHT_SEEDS = (25_001, 25_002, 25_003, 25_004)
PREFLIGHT_STEPS_PER_SEED = 12
PREFLIGHT_ACTION_MODES = (
    "half_active", "zero_active", "full_active", "random_active"
)
PRETRAINING_SANITY_SEEDS = (24_501, 24_502, 24_503)


def build_preflight_action(seed, step_index):
    """Return a deterministic action covering gain boundaries and motion."""
    step_index = int(step_index)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 90_001, step_index])
    )
    action = rng.uniform(-1.0, 1.0, cfg.Lp + 2 * cfg.La + 2)
    mode = PREFLIGHT_ACTION_MODES[step_index % len(PREFLIGHT_ACTION_MODES)]
    gain_slice = slice(cfg.Lp + cfg.La, cfg.Lp + 2 * cfg.La)
    if mode == "half_active":
        action[gain_slice] = 0.0
    elif mode == "zero_active":
        action[gain_slice] = -1.0
    elif mode == "full_active":
        action[gain_slice] = 1.0
    return mode, action


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_scientific_preflight(output_path, seeds=PREFLIGHT_SEEDS,
                             steps_per_seed=PREFLIGHT_STEPS_PER_SEED):
    """Exercise moving environments and fail unless every checked step passes."""
    seeds = tuple(int(seed) for seed in seeds)
    steps_per_seed = int(steps_per_seed)
    if not seeds:
        raise ValueError("preflight requires at least one seed")
    if steps_per_seed < len(PREFLIGHT_ACTION_MODES):
        raise ValueError("preflight must cover every declared action mode")

    records = []
    for seed in seeds:
        env = HRISUAVISACEnv(sigma2_scee=cfg.rho_nmse, seed=seed)
        env.reset()
        for step_index in range(steps_per_seed):
            action_mode, action = build_preflight_action(seed, step_index)
            _, reward, _, info = env.step(action)
            total_power = float(
                np.linalg.norm(env.C, "fro") ** 2
                + np.linalg.norm(env.T, "fro") ** 2
            )
            envelope = active_tx_power_envelope(env.ch)
            finite = bool(np.all(np.isfinite([
                reward, info["sum_rate"], info["sum_sensing"],
                info["P_ai"], info["cascade_pwr_16c"], total_power,
                envelope,
            ])))
            record = {
                "seed": seed,
                "step": step_index + 1,
                "action_mode": action_mode,
                "finite": finite,
                "sensing_feasible": bool(info["sensing_constraint_feasible"]),
                "active_power_feasible": bool(info["active_power_ok"]),
                "active_envelope_feasible": bool(
                    envelope <= cfg.P_a_max + 1e-12
                ),
                "cascade_feasible": bool(info["constr_16c_ok"]),
                "power_contract_feasible": bool(
                    total_power <= cfg.PT + 1e-12
                ),
                "min_sensing_rate": float(np.min(info["rate_k"])),
                "R_min": float(cfg.R_min),
                "P_ai": float(info["P_ai"]),
                "P_ai_envelope": float(envelope),
                "P_a_max": float(cfg.P_a_max),
                "cascade_power": float(info["cascade_pwr_16c"]),
                "Psi_max": float(cfg.Psi_max),
                "total_bs_power": total_power,
            }
            record["joint_feasible"] = bool(
                record["finite"]
                and record["sensing_feasible"]
                and record["active_power_feasible"]
                and record["active_envelope_feasible"]
                and record["cascade_feasible"]
                and record["power_contract_feasible"]
            )
            records.append(record)

    count = len(records)

    def fraction(key):
        return sum(bool(record[key]) for record in records) / count

    fractions = {
        key: fraction(key)
        for key in (
            "finite", "sensing_feasible", "active_power_feasible",
            "active_envelope_feasible", "cascade_feasible",
            "power_contract_feasible", "joint_feasible",
        )
    }
    passed = all(math.isclose(value, 1.0) for value in fractions.values())
    report = {
        "schema_version": 1,
        "method": "bounded dynamic feasibility preflight; no training",
        "parameter_profile": active_profile_metadata(),
        "seeds": list(seeds),
        "steps_per_seed": steps_per_seed,
        "action_modes": list(PREFLIGHT_ACTION_MODES),
        "fractions": fractions,
        "passed": passed,
        "records": records,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    if not passed:
        raise RuntimeError(
            "Scientific preflight failed before training. See "
            f"{output_path}."
        )
    return report


def run_pretraining_physical_sanity(seeds=PRETRAINING_SANITY_SEEDS):
    """Fail before DRL training if the activated action space is ineffective."""
    from parameter_identification import (
        MINIMUM_COMM_ACTION_RESPONSE,
        evaluate_candidate,
        structural_acceptance,
    )

    metrics = evaluate_candidate(None, seeds=seeds)
    passed, reasons = structural_acceptance(metrics)
    summary_fields = (
        "profile_digest", "seeds", "finite_fraction",
        "power_contract_fraction", "sensing_feasible_fraction",
        "active_feasible_fraction", "cascade_feasible_fraction",
        "joint_feasible_fraction", "comm_rate_mean", "sensing_rate_mean",
        "comm_rate_cv", "sensing_rate_cv",
        "comm_action_response_median", "comm_action_response_minimum",
        "sensing_action_response_median", "errors",
    )
    report = {
        "schema_version": 1,
        "kind": "pretraining_physical_sanity",
        "method": "deterministic no-training action-response gate",
        "minimum_comm_action_response": MINIMUM_COMM_ACTION_RESPONSE,
        "passed": bool(passed),
        "reasons": list(reasons),
        **{field: _json_safe(metrics[field]) for field in summary_fields},
    }
    if not passed:
        raise RuntimeError(
            "Pre-training physical sanity failed: " + "; ".join(reasons)
        )
    return report
