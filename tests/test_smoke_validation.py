from copy import deepcopy

from ambiguity_hypotheses import apply_ambiguity_manifest_fields
from config import cfg
from experiment_protocol import (
    EVALUATION_SEED_PROTOCOL,
    derive_training_seed,
    evaluation_scene_seed_grid,
)
from validate_results import METHOD_ORDER, validate_smoke_result


def _evaluation():
    return {
        "communication": [2.0, 2.1],
        "sensing": [1.0, 1.1],
        "objective": [2.0, 2.1],
        "feasibility": [1.0, 1.0],
        "minimum_ue_rates": [0.8, 0.9],
        "minimum_target_rates": [0.6, 0.7],
        "communication_mean": 2.05,
        "sensing_mean": 1.05,
        "objective_mean": 2.05,
        "feasibility_rate": 1.0,
        "minimum_ue_rate": 0.8,
        "minimum_target_rate": 0.6,
        "action_projection_rate": [0.0, 0.0],
        "mean_projection_fraction": [1.0, 1.0],
        "deterministic_recovery_rate": [0.0, 0.0],
        "allocation_fallback_rate": [0.0, 0.0],
        "action_projection_rate_mean": 0.0,
        "mean_projection_fraction_mean": 1.0,
        "deterministic_recovery_rate_mean": 0.0,
        "allocation_fallback_rate_mean": 0.0,
    }


def _smoke_result():
    episodes = 100
    evaluation_scene_seeds = evaluation_scene_seed_grid([7], 2)[7]
    source_training_seed = derive_training_seed(7, "source")
    target_training_seed = derive_training_seed(7, "target")
    methods = {}
    final = {}
    for method in METHOD_ORDER:
        communication = [1.0 + index / episodes for index in range(episodes)]
        sensing = [0.75] * episodes
        methods[method] = {
            "training_seed": target_training_seed,
            "training": {
                "rewards": list(communication),
                "comm_rates": list(communication),
                "sensing_rates": sensing,
                "joint_rates": [c + s for c, s in zip(communication, sensing)],
                "feasibility_rates": [1.0] * episodes,
            },
            "evaluation": _evaluation(),
        }
        final[method] = {"seed_values": [sum(communication) / episodes]}
    result = {
        "schema_version": 1,
        "kind": "generated_result",
        "protocol": "paper_baseline_smoke",
        "run_mode": "smoke",
        "publication_eligible": False,
        "physical_configuration_fingerprint": "b" * 64,
        "profile_digest": "a" * 64,
        "objective": {"mode": "paper_baseline", "target_rate_min": 0.5},
        "power_hypothesis_id": "paper_db_as_dbm",
        "source_tree_digest": "c" * 64,
        "training_seed_protocol": "sha256_v1_common_target_seed_assignment",
        "evaluation_seed_protocol": EVALUATION_SEED_PROTOCOL,
        "seeds": [7],
        "episodes": episodes,
        "figure4_protocol": None,
        "diagnostic_scope": {
            "purpose": "early_scale_feasibility_projection_checkpoint_check",
            "not_evidence_for_paper_curve_acceptance": True,
        },
        "evaluation_episodes": 2,
        "final_reward_source": "training_communication_rate",
        "training_final_window": 100,
        "feasibility_rate": 1.0,
        "final_reward_by_method": final,
        "communication_seed_values": [2.05],
        "sensing_seed_values": [1.05],
        "seed_records": [{
            "seed": 7,
            "source_training_seed": source_training_seed,
            "target_training_seed": target_training_seed,
            "evaluation_scene_seeds": evaluation_scene_seeds,
            "methods": methods,
        }],
    }
    return apply_ambiguity_manifest_fields(result, cfg)


def test_smoke_validation_checks_integrity_without_claiming_reproduction():
    report = validate_smoke_result(
        _smoke_result(), result_directory=None, verify_files=False
    )

    assert report["overall_pass"] is True
    assert report["advisory_only"] is True
    assert report["publication_eligible"] is False
    assert report["paper_curve_acceptance_performed"] is False


def test_smoke_validation_rejects_penalized_reward_or_projection_dependence():
    result = _smoke_result()
    result["seed_records"][0]["methods"]["DDPG"]["training"]["rewards"][0] -= 1.0
    evaluation = result["seed_records"][0]["methods"]["DDPG"]["evaluation"]
    evaluation["action_projection_rate"] = [0.5, 0.5]
    evaluation["action_projection_rate_mean"] = 0.5

    report = validate_smoke_result(
        result, result_directory=None, verify_files=False
    )

    assert report["overall_pass"] is False
    assert report["checks"]["paper_reward_equals_communication_rate"]["pass"] is False
    assert report["checks"]["actor_action_attribution"]["pass"] is False
