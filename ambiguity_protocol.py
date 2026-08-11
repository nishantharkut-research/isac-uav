"""Predeclared ambiguity-sensitivity designs with no result selection."""

from itertools import product

import numpy as np

from ambiguity_hypotheses import (
    DDPG_COMPARATOR_CSI_HYPOTHESES,
    NOISE_HYPOTHESES,
    PRIMARY_NOISE_HYPOTHESIS_ID,
    PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID,
    PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID,
    PRIMARY_WFB_HYPOTHESIS_ID,
    PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID,
    TRANSFER_SCHEDULE_HYPOTHESES,
    TRANSFER_UPDATE_HYPOTHESES,
    WFB_HYPOTHESES,
)


SENSITIVITY_SCHEMA_VERSION = 1


def _hypotheses(noise, update, schedule, wfb, ddpg_comparator_csi):
    return {
        "noise": noise,
        "transfer_update": update,
        "transfer_schedule": schedule,
        "wfb": wfb,
        "ddpg_comparator_csi": ddpg_comparator_csi,
    }


def _primary():
    return _hypotheses(
        PRIMARY_NOISE_HYPOTHESIS_ID,
        PRIMARY_TRANSFER_UPDATE_HYPOTHESIS_ID,
        PRIMARY_TRANSFER_SCHEDULE_HYPOTHESIS_ID,
        PRIMARY_WFB_HYPOTHESIS_ID,
        PRIMARY_DDPG_COMPARATOR_CSI_HYPOTHESIS_ID,
    )


def build_sensitivity_plan(design="one_at_a_time"):
    """Build a result-independent sensitivity matrix."""
    if design == "one_at_a_time":
        primary = _primary()
        scenarios = [
            {"id": "primary", "hypotheses": dict(primary)},
            {
                "id": "noise_dual_minus70",
                "hypotheses": {
                    **primary, "noise": "dual_minus70_conservative"
                },
            },
            {
                "id": "transfer_next_state",
                "hypotheses": {
                    **primary,
                    "transfer_update":
                        "eq30_next_state_continuation_blend",
                },
            },
            {
                "id": "schedule_linear_annealed",
                "hypotheses": {
                    **primary,
                    "transfer_schedule": "linear_annealed_upsilon",
                },
            },
            {
                "id": "wfb_channel_independent_dft",
                "hypotheses": {
                    **primary,
                    "wfb": "channel_independent_dft_no_capon",
                },
            },
            {
                "id": "ddpg_icsi_comparator",
                "hypotheses": {
                    **primary,
                    "ddpg_comparator_csi": "icsi_comparator",
                },
            },
        ]
    elif design == "full_factorial":
        scenarios = []
        for index, combination in enumerate(product(
                NOISE_HYPOTHESES,
                TRANSFER_UPDATE_HYPOTHESES,
                TRANSFER_SCHEDULE_HYPOTHESES,
                WFB_HYPOTHESES,
                DDPG_COMPARATOR_CSI_HYPOTHESES)):
            hypotheses = _hypotheses(*combination)
            identifier = (
                "primary" if hypotheses == _primary()
                else f"factorial_{index:02d}"
            )
            scenarios.append({"id": identifier, "hypotheses": hypotheses})
    else:
        raise ValueError(f"Unknown sensitivity design: {design}")

    return {
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "kind": "predeclared_ambiguity_sensitivity_plan",
        "design": design,
        "calibration_figures_used": [],
        "holdout_figures_used_for_selection": [],
        "result_selection": "none_report_all",
        "scenarios": scenarios,
    }


def _finite_vector(payload, key, expected_size):
    values = np.asarray(payload.get(key, []), dtype=float)
    if (
        values.shape != (expected_size,)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"Sensitivity summary has invalid {key}")
    return values


def build_sensitivity_report(plan, summaries):
    """Build paired deltas without ranking or selecting a hypothesis."""
    scenario_ids = [scenario["id"] for scenario in plan["scenarios"]]
    if set(summaries) != set(scenario_ids):
        raise ValueError("Sensitivity summaries do not match the declared plan")
    primary = summaries.get("primary")
    if primary is None:
        raise ValueError("Sensitivity analysis requires the primary scenario")
    provenance_keys = (
        "profile_digest", "power_hypothesis_id", "source_tree_digest",
        "training_seed_protocol", "objective", "seeds", "episodes",
        "evaluation_episodes",
    )
    provenance = {key: primary.get(key) for key in provenance_keys}
    scenario_lookup = {
        scenario["id"]: scenario for scenario in plan["scenarios"]
    }
    seed_count = len(provenance["seeds"] or [])
    if seed_count < 5:
        raise ValueError("Sensitivity analysis requires at least five seeds")

    for scenario_id, summary in summaries.items():
        hypotheses = scenario_lookup[scenario_id]["hypotheses"]
        expected_identities = {
            "noise_hypothesis_id": hypotheses["noise"],
            "dtl_transfer_mode": hypotheses["transfer_update"],
            "dtl_transfer_schedule_id": hypotheses["transfer_schedule"],
            "wfb_hypothesis_id": hypotheses["wfb"],
            "ddpg_comparator_csi_hypothesis_id":
                hypotheses["ddpg_comparator_csi"],
        }
        if any(
            summary.get(key) != value
            for key, value in expected_identities.items()
        ):
            raise ValueError(
                f"Sensitivity scenario {scenario_id} has mismatched hypothesis identity"
            )
        if (
            summary.get("kind") != "generated_result"
            or summary.get("protocol") != "paper_baseline"
            or any(summary.get(key) != value
                   for key, value in provenance.items())
        ):
            raise ValueError(
                f"Sensitivity scenario {scenario_id} has mismatched paired provenance"
            )

    primary_comm = _finite_vector(
        primary, "communication_seed_values", seed_count
    )
    primary_sensing = _finite_vector(
        primary, "sensing_seed_values", seed_count
    )
    primary_rewards = primary.get("final_reward_by_method", {})
    comparisons = {}
    raw = {}
    for scenario_id in scenario_ids:
        summary = summaries[scenario_id]
        comm = _finite_vector(summary, "communication_seed_values", seed_count)
        sensing = _finite_vector(summary, "sensing_seed_values", seed_count)
        raw[scenario_id] = {
            "hypotheses": scenario_lookup[scenario_id]["hypotheses"],
            "communication_seed_values": comm.tolist(),
            "sensing_seed_values": sensing.tolist(),
            "final_reward_by_method": summary.get("final_reward_by_method", {}),
        }
        if scenario_id == "primary":
            continue
        reward_deltas = {}
        candidate_rewards = summary.get("final_reward_by_method", {})
        if set(candidate_rewards) != set(primary_rewards):
            raise ValueError(
                f"Sensitivity scenario {scenario_id} has mismatched methods"
            )
        for method, primary_payload in primary_rewards.items():
            left = _finite_vector(
                primary_payload, "seed_values", seed_count
            )
            right = _finite_vector(
                candidate_rewards[method], "seed_values", seed_count
            )
            delta = right - left
            reward_deltas[method] = {
                "seed_values": delta.tolist(),
                "mean": float(np.mean(delta)),
            }
        comm_delta = comm - primary_comm
        sensing_delta = sensing - primary_sensing
        comparisons[scenario_id] = {
            "against": "primary",
            "changed_hypotheses": scenario_lookup[scenario_id]["hypotheses"],
            "communication_delta": {
                "seed_values": comm_delta.tolist(),
                "mean": float(np.mean(comm_delta)),
            },
            "sensing_delta": {
                "seed_values": sensing_delta.tolist(),
                "mean": float(np.mean(sensing_delta)),
            },
            "final_reward_delta_by_method": reward_deltas,
        }
    return {
        "schema_version": 1,
        "kind": "ambiguity_sensitivity_report",
        "design": plan["design"],
        "result_selection": "none_report_all",
        "paired_provenance": provenance,
        "scenarios": raw,
        "comparisons": comparisons,
    }
