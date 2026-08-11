import unittest

from validate_pareto_extension import validate_pareto_result


PREFERENCES = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]


def _preference_record(preference, communication, sensing, scenes):
    count = len(scenes)
    return {
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
    }


def _grid_evaluation(points, scenes, gate=True):
    result = {
        "schema_version": 1,
        "scene_seeds": list(scenes),
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
    return {
        "scene_seeds": list(scenes),
        "communication": [joint - 4.0] * count,
        "sensing": [4.0] * count,
        "joint": [joint] * count,
        "objective": [joint] * count,
        "minimum_ue_rate": [0.8] * count,
        "minimum_target_rate": [0.9] * count,
        "feasibility": [1.0] * count,
    }


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
                        }
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
        "preference_grid": PREFERENCES,
        "hypervolume_reference_normalized": [-1.0, -1.0],
        "anchors": {
            "schema_version": 1,
            "ideal": [10.0, 10.0],
            "nadir": [0.0, 0.0],
            "frozen": True,
            "provenance": {"partition": "training_only"},
        },
        "anchors_lock_sha256": "3" * 64,
        "gate_selection": {
            "schema_version": 1,
            "selection_partition": "validation_only",
            "validation_scene_seeds": validation,
            "research_seeds": research,
            "anchors_sha256": "3" * 64,
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


class ParetoValidationTests(unittest.TestCase):
    def test_publication_fixture_passes_positive_paired_endpoints(self):
        report = validate_pareto_result(
            valid_summary(), level="publication", verify_files=False
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
                report = validate_pareto_result(
                    summary, level="publication", verify_files=False
                )
                self.assertFalse(report["overall_pass"], report)


if __name__ == "__main__":
    unittest.main()
