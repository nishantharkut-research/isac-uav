import unittest

from validate_pareto_extension import validate_pareto_result
from validate_results import result_payload_digest


PREFERENCES = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
EVALUATION_EPISODES = 2


def _preference_record(preference, communication, sensing, scenes):
    count = len(scenes)
    record = {
        "key": f"{preference}",
        "preference": list(preference),
        "communication": [communication] * count,
        "sensing": [sensing] * count,
        "joint": [communication + sensing] * count,
        "objective": [communication + sensing] * count,
        "minimum_ue_rate": [0.8] * count,
        "minimum_target_rate": [0.9] * count,
        "feasibility": [1.0] * count,
        "mean_residual_scale": [0.5] * count,
        "baseline_fallback_rate": [0.2] * count,
        "action_projection_rate": [0.01] * count,
        "mean_projection_fraction": [0.99] * count,
        "deterministic_recovery_rate": [0.005] * count,
        "allocation_fallback_rate": [0.02] * count,
    }
    record["episode_evidence"] = [
        {
            "scene_seed": seed,
            "communication": [communication] * EVALUATION_EPISODES,
            "sensing": [sensing] * EVALUATION_EPISODES,
            "joint": [communication + sensing] * EVALUATION_EPISODES,
            "objective": [communication + sensing] * EVALUATION_EPISODES,
            "minimum_ue_rate": [0.8] * EVALUATION_EPISODES,
            "minimum_target_rate": [0.9] * EVALUATION_EPISODES,
            "feasibility": [1.0] * EVALUATION_EPISODES,
            "mean_residual_scale": [0.5] * EVALUATION_EPISODES,
            "baseline_fallback_rate": [0.2] * EVALUATION_EPISODES,
            "action_projection_rate": [0.01] * EVALUATION_EPISODES,
            "mean_projection_fraction": [0.99] * EVALUATION_EPISODES,
            "deterministic_recovery_rate": [0.005] * EVALUATION_EPISODES,
            "allocation_fallback_rate": [0.02] * EVALUATION_EPISODES,
        }
        for seed in scenes
    ]
    return record


def _grid_evaluation(points, scenes, gate=True):
    result = {
        "schema_version": 1,
        "scene_seeds": list(scenes),
        "evaluation_episodes": EVALUATION_EPISODES,
        "preferences": [
            _preference_record(preference, *point, scenes)
            for preference, point in zip(PREFERENCES, points)
        ],
        "use_gate": gate,
    }
    if gate:
        result["gate_configuration"] = {
            "uncertainty_coefficient": 0.5,
            "margin": 0.0,
        }
    return result


def _scalar_evaluation(joint, scenes):
    count = len(scenes)
    record = {
        "scene_seeds": list(scenes),
        "evaluation_episodes": EVALUATION_EPISODES,
        "communication": [joint - 4.0] * count,
        "sensing": [4.0] * count,
        "joint": [joint] * count,
        "objective": [joint] * count,
        "minimum_ue_rate": [0.8] * count,
        "minimum_target_rate": [0.9] * count,
        "feasibility": [1.0] * count,
        "action_projection_rate": [0.01] * count,
        "mean_projection_fraction": [0.99] * count,
        "deterministic_recovery_rate": [0.005] * count,
        "allocation_fallback_rate": [0.02] * count,
    }
    record["episode_evidence"] = [
        {
            "scene_seed": seed,
            "communication": [joint - 4.0] * EVALUATION_EPISODES,
            "sensing": [4.0] * EVALUATION_EPISODES,
            "joint": [joint] * EVALUATION_EPISODES,
            "objective": [joint] * EVALUATION_EPISODES,
            "minimum_ue_rate": [0.8] * EVALUATION_EPISODES,
            "minimum_target_rate": [0.9] * EVALUATION_EPISODES,
            "feasibility": [1.0] * EVALUATION_EPISODES,
            "action_projection_rate": [0.01] * EVALUATION_EPISODES,
            "mean_projection_fraction": [0.99] * EVALUATION_EPISODES,
            "deterministic_recovery_rate": [0.005] * EVALUATION_EPISODES,
            "allocation_fallback_rate": [0.02] * EVALUATION_EPISODES,
        }
        for seed in scenes
    ]
    return record


def valid_summary():
    research = list(range(10))
    validation = list(range(100, 105))
    test = list(range(200, 210))
    records = []
    for seed in research:
        proposed_points = [
            [8.0 + 0.01 * seed, 3.0],
            [6.0 + 0.01 * seed, 6.0],
            [3.0, 8.0 + 0.01 * seed],
        ]
        independent_points = [
            [7.0 + 0.01 * seed, 3.0],
            [5.0 + 0.01 * seed, 5.0],
            [3.0, 7.0 + 0.01 * seed],
        ]
        independent = [
            {
                "preference": preference,
                "evaluation": _preference_record(preference, *point, test),
            }
            for preference, point in zip(PREFERENCES, independent_points)
        ]
        records.append({
            "seed": seed,
            "training": {
                "scalar": {
                    "methods": {
                        "same_objective_dtl": {
                            "checkpoint": f"checkpoints/scalar-{seed}.pt",
                            "checkpoint_sha256": "a" * 64,
                            "training": {
                                "comm_rates": [10.0] * 100,
                                "sensing_rates": [10.0] * 100,
                            },
                        },
                        "cross_objective_dtl_diagnostic": {
                            "checkpoint": f"checkpoints/cross-{seed}.pt",
                            "checkpoint_sha256": "d" * 64,
                            "training": {
                                "comm_rates": [0.0] * 100,
                                "sensing_rates": [0.0] * 100,
                            },
                        },
                        "ddpg_scratch": {
                            "checkpoint": f"checkpoints/scratch-{seed}.pt",
                            "checkpoint_sha256": "e" * 64,
                            "training": {
                                "comm_rates": [5.0] * 100,
                                "sensing_rates": [5.0] * 100,
                            },
                        },
                    }
                },
                "vector": {
                    "methods": {
                        "ba_pcdtl": {
                            "checkpoint": f"checkpoints/ba-{seed}.pt",
                            "checkpoint_sha256": "b" * 64,
                            "baseline_policy_digest": "c" * 64,
                            "identity": {
                                "baseline_policy_digest": "c" * 64,
                                "source_tree_digest": "f" * 64,
                                "objective": {
                                    "mode": "png_joint_sum",
                                    "user_rate_min": 0.5,
                                    "target_rate_min": 0.5,
                                },
                            },
                        }
                    }
                },
            },
            "test": {
                "same_objective_dtl": _scalar_evaluation(11.0 + 0.005 * seed, test),
                "cross_objective_dtl_diagnostic": _scalar_evaluation(
                    10.5 + 0.005 * seed, test
                ),
                "ddpg_scratch": _scalar_evaluation(10.0 + 0.005 * seed, test),
                "paper_dtl_pcsi": _scalar_evaluation(10.8 + 0.005 * seed, test),
                "ba_pcdtl": _grid_evaluation(proposed_points, test, gate=True),
                "ba_pcdtl_no_gate": _grid_evaluation(
                    proposed_points, test, gate=False
                ),
                "unanchored_vector_td3": _grid_evaluation(
                    independent_points, test, gate=False
                ),
                "anchored_random_trunks": _grid_evaluation(
                    independent_points, test, gate=False
                ),
                "independent_weighted_policies": {
                    "scene_seeds": test,
                    "policies": [
                        {
                            **policy,
                            "use_gate": False,
                        }
                        for policy in independent
                    ],
                },
            },
        })
    return {
        "schema_version": 1,
        "kind": "generated_result",
        "protocol": "ba_pcdtl_research_extension",
        "algorithm_id": "ba_pcdtl_v1",
        "comparison_level": "full",
        "profile_digest": "d" * 64,
        "physical_configuration_fingerprint": "e" * 64,
        "source_tree_digest": "f" * 64,
        "baseline_payload_sha256": "1" * 64,
        "baseline_validation_sha256": "2" * 64,
        "objective": {
            "mode": "png_joint_sum",
            "user_rate_min": 0.5,
            "target_rate_min": 0.5,
        },
        "research_seeds": research,
        "validation_scene_seeds": validation,
        "test_scene_seeds": test,
        "episodes": 100,
        "evaluation_episodes": EVALUATION_EPISODES,
        "preference_grid": PREFERENCES,
        "gate_selection_grid": {
            "uncertainty_coefficients": [0.5],
            "margins": [0.0],
            "selection_partition": "validation_only",
            "tie_break_order": [
                "feasibility", "png_joint_sum", "pareto_hypervolume",
                "lower_fallback_rate", "lexicographic_parameters",
            ],
        },
        "hypervolume_reference_rule": {
            "physical_point_bits_per_second_per_hz": [0.0, 0.0],
            "normalized_margin": 1.0,
            "lock_partition": "training_only_before_validation",
        },
        "hypervolume_reference_normalized": [-1.0, -1.0],
        "anchors": {
            "schema_version": 1,
            "ideal": [10.0, 10.0],
            "nadir": [0.0, 0.0],
            "frozen": True,
            "provenance": {
                "partition": "training_only",
                "estimator": "per-method converged-tail quantiles",
                "tail_fraction": 0.2,
                "tail_minimum_episodes": 100,
                "lower_quantile": 0.05,
                "upper_quantile": 0.95,
                "research_seeds": research,
                "source_methods": [
                    "same_objective_dtl",
                    "cross_objective_dtl_diagnostic",
                    "ddpg_scratch",
                ],
            },
        },
        "anchors_lock_sha256": "3" * 64,
        "gate_selection": {
            "schema_version": 1,
            "selection_partition": "validation_only",
            "validation_scene_seeds": validation,
            "research_seeds": research,
            "anchors_sha256": "3" * 64,
            "gate_selection_grid": {
                "uncertainty_coefficients": [0.5],
                "margins": [0.0],
                "selection_partition": "validation_only",
                "tie_break_order": [
                    "feasibility", "png_joint_sum", "pareto_hypervolume",
                    "lower_fallback_rate", "lexicographic_parameters",
                ],
            },
            "hypervolume_reference_normalized": [-1.0, -1.0],
            "validation_evidence_sha256": "5" * 64,
            "selected": {
                "uncertainty_coefficient": 0.5,
                "margin": 0.0,
                "feasibility": 1.0,
                "joint": 12.0,
                "hypervolume": 3.0,
                "fallback_rate": 0.2,
            },
            "candidates": [{
                "uncertainty_coefficient": 0.5,
                "margin": 0.0,
                "feasibility": 1.0,
                "joint": 12.0,
                "hypervolume": 3.0,
                "fallback_rate": 0.2,
            }],
        },
        "gate_selection_lock_sha256": "4" * 64,
        "gate_validation_evidence_sha256": "5" * 64,
        "research_seed_records": records,
    }


def bind_accepted_baseline(summary, *, overlapping_scene=None):
    records = []
    authority = []
    for seed in range(10):
        source_seed = 10_000 + seed
        if seed == 0 and overlapping_scene is not None:
            source_seed = int(overlapping_scene)
        source_checkpoint = f"checkpoints/source-{seed}.pt"
        target_checkpoint = f"checkpoints/dtl-pcsi-{seed}.pt"
        source_digest = f"{seed + 1:064x}"
        target_digest = f"{seed + 101:064x}"
        records.append({
            "seed": seed,
            "source_seed": source_seed,
            "target_seed": 20_000 + seed,
            "source_checkpoint": source_checkpoint,
            "source_checkpoint_sha256": source_digest,
            "methods": {
                "DTL-pCSI": {
                    "checkpoint": target_checkpoint,
                    "checkpoint_sha256": target_digest,
                    "identity": {"source_model_digest": "a" * 64},
                }
            },
        })
        authority.append({
            "seed": seed,
            "source_checkpoint": source_checkpoint,
            "source_checkpoint_sha256": source_digest,
            "dtl_pcsi_checkpoint": target_checkpoint,
            "dtl_pcsi_checkpoint_sha256": target_digest,
            "source_model_digest": "a" * 64,
        })
    baseline = {
        "profile_digest": summary["profile_digest"],
        "physical_configuration_fingerprint": summary[
            "physical_configuration_fingerprint"
        ],
        "source_tree_digest": summary["source_tree_digest"],
        "seeds": list(range(10)),
        "seed_records": records,
    }
    baseline_validation = {
        "validation": "paper_baseline",
        "overall_pass": True,
        "baseline_payload_sha256": result_payload_digest(baseline),
    }
    summary["baseline_payload_sha256"] = result_payload_digest(baseline)
    summary["baseline_validation_sha256"] = result_payload_digest(
        baseline_validation
    )
    summary["baseline_checkpoint_authority"] = authority
    return baseline, baseline_validation


class ParetoValidationTests(unittest.TestCase):
    def test_publication_fixture_passes_positive_paired_endpoints(self):
        summary = valid_summary()
        baseline, baseline_validation = bind_accepted_baseline(summary)
        report = validate_pareto_result(
            summary,
            level="publication",
            verify_files=False,
            baseline=baseline,
            baseline_validation=baseline_validation,
        )
        self.assertTrue(report["overall_pass"], report)
        self.assertGreater(report["statistics"]["joint_improvement"]["lower"], 0.0)
        self.assertGreater(
            report["statistics"]["joint_improvement_vs_paper_dtl"]["lower"],
            0.0,
        )
        self.assertGreater(
            report["statistics"]["hypervolume_improvement"]["lower"], 0.0
        )

    def test_publication_validation_requires_accepted_baseline_inputs(self):
        report = validate_pareto_result(
            valid_summary(), level="publication", verify_files=False
        )
        self.assertFalse(report["overall_pass"], report)
        self.assertFalse(report["checks"]["accepted_baseline_authority"]["pass"])
        self.assertFalse(report["checks"]["accepted_baseline_checkpoints"]["pass"])

    def test_effective_baseline_training_scene_overlap_is_rejected(self):
        summary = valid_summary()
        baseline, baseline_validation = bind_accepted_baseline(
            summary, overlapping_scene=summary["validation_scene_seeds"][0]
        )
        report = validate_pareto_result(
            summary,
            level="publication",
            verify_files=False,
            baseline=baseline,
            baseline_validation=baseline_validation,
        )
        self.assertFalse(report["overall_pass"], report)
        self.assertFalse(report["checks"]["independent_seed_roles"]["pass"])

    def test_fail_closed_rejections(self):
        mutations = {}
        short = valid_summary()
        short["research_seeds"] = short["research_seeds"][:9]
        mutations["insufficient_seeds"] = short

        overlap = valid_summary()
        overlap["test_scene_seeds"][0] = overlap["validation_scene_seeds"][0]
        mutations["overlap"] = overlap

        infeasible = valid_summary()
        infeasible["research_seed_records"][0]["test"]["ba_pcdtl"][
            "preferences"
        ][0]["feasibility"][0] = 0.9
        mutations["infeasible"] = infeasible

        floor = valid_summary()
        floor["research_seed_records"][0]["test"]["ba_pcdtl"][
            "preferences"
        ][0]["minimum_target_rate"][0] = 0.1
        mutations["floor"] = floor

        missing_gate_fallback = valid_summary()
        del missing_gate_fallback["research_seed_records"][0]["test"][
            "ba_pcdtl"
        ]["preferences"][0]["baseline_fallback_rate"]
        mutations["missing_gate_fallback"] = missing_gate_fallback

        impossible_projection = valid_summary()
        projection_row = impossible_projection["research_seed_records"][0][
            "test"
        ]["ba_pcdtl"]["preferences"][0]
        projection_row["action_projection_rate"][0] = 0.0
        projection_row["mean_projection_fraction"][0] = 0.0
        mutations["impossible_projection"] = impossible_projection

        no_joint_gain = valid_summary()
        for record in no_joint_gain["research_seed_records"]:
            balanced = record["test"]["ba_pcdtl"]["preferences"][1]
            balanced["joint"] = [9.0] * 10
        mutations["joint"] = no_joint_gain

        no_hv_gain = valid_summary()
        for record in no_hv_gain["research_seed_records"]:
            for proposed, independent in zip(
                record["test"]["ba_pcdtl"]["preferences"],
                record["test"]["independent_weighted_policies"]["policies"],
            ):
                proposed["communication"] = list(independent["evaluation"]["communication"])
                proposed["sensing"] = list(independent["evaluation"]["sensing"])
        mutations["hypervolume"] = no_hv_gain

        missing = valid_summary()
        missing["research_seed_records"][0]["test"]["ba_pcdtl"][
            "preferences"
        ].pop()
        mutations["missing_preference"] = missing

        nonfinite = valid_summary()
        nonfinite["research_seed_records"][0]["test"]["ba_pcdtl"][
            "preferences"
        ][0]["communication"][0] = float("nan")
        mutations["nonfinite"] = nonfinite

        missing_projection = valid_summary()
        del missing_projection["research_seed_records"][0]["test"]["ba_pcdtl"][
            "preferences"
        ][0]["action_projection_rate"]
        mutations["missing_projection_evidence"] = missing_projection

        invalid_projection = valid_summary()
        invalid_projection["research_seed_records"][0]["test"][
            "same_objective_dtl"
        ]["mean_projection_fraction"][0] = 1.1
        mutations["invalid_projection_evidence"] = invalid_projection

        gate = valid_summary()
        gate["research_seed_records"][0]["test"]["ba_pcdtl"][
            "gate_configuration"
        ]["margin"] = 0.1
        mutations["gate"] = gate

        wrong_gate_selection = valid_summary()
        wrong_gate_selection["gate_selection"]["candidates"].append({
            "uncertainty_coefficient": 1.0,
            "margin": 0.0,
            "feasibility": 1.0,
            "joint": 13.0,
            "hypervolume": 4.0,
            "fallback_rate": 0.1,
        })
        mutations["wrong_gate_selection"] = wrong_gate_selection

        tampered_anchor_history = valid_summary()
        tampered_anchor_history["research_seed_records"][0]["training"][
            "scalar"
        ]["methods"]["same_objective_dtl"]["training"]["comm_rates"] = [20.0] * 100
        mutations["tampered_anchor_history"] = tampered_anchor_history

        missing_ablation = valid_summary()
        del missing_ablation["research_seed_records"][0]["test"][
            "ba_pcdtl_no_gate"
        ]
        mutations["missing_ablation"] = missing_ablation

        missing_paper_comparator = valid_summary()
        del missing_paper_comparator["research_seed_records"][0]["test"][
            "paper_dtl_pcsi"
        ]
        mutations["missing_paper_comparator"] = missing_paper_comparator

        contaminated_ablation = valid_summary()
        contaminated_ablation["research_seed_records"][0]["test"][
            "anchored_random_trunks"
        ]["use_gate"] = True
        mutations["contaminated_ablation"] = contaminated_ablation

        envelope = valid_summary()
        envelope["pointwise_envelope"] = [1.0, 2.0]
        mutations["envelope"] = envelope

        for name, summary in mutations.items():
            with self.subTest(name=name):
                baseline, baseline_validation = bind_accepted_baseline(summary)
                report = validate_pareto_result(
                    summary,
                    level="publication",
                    verify_files=False,
                    baseline=baseline,
                    baseline_validation=baseline_validation,
                )
                self.assertFalse(report["overall_pass"], report)


if __name__ == "__main__":
    unittest.main()
