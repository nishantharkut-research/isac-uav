import copy
import unittest
from unittest.mock import patch

import numpy as np

from experiment_protocol import (
    EVALUATION_SEED_PROTOCOL,
    TRAINING_SEED_PROTOCOL,
    derive_training_seed,
    evaluation_scene_seed_grid,
)
from run_extension import require_accepted_baseline
from paper_reference import load_paper_references
from validate_results import (
    paired_normalized_baseline_objective,
    result_payload_digest,
    require_canonical_baseline_validation,
    validate_baseline_result,
    validate_extension_result,
)


def _projection_evaluation(count=5):
    return {
        "action_projection_rate": [0.01] * count,
        "mean_projection_fraction": [0.99] * count,
        "deterministic_recovery_rate": [0.005] * count,
        "allocation_fallback_rate": [0.02] * count,
        "action_projection_rate_mean": 0.01,
        "mean_projection_fraction_mean": 0.99,
        "deterministic_recovery_rate_mean": 0.005,
        "allocation_fallback_rate_mean": 0.02,
    }


def _raw_evaluation(
        communication, sensing, objective, *, count=5,
        feasibility=1.0, minimum_ue_rate=0.8, minimum_target_rate=0.9,
):
    result = _projection_evaluation(count)
    result.update({
        "communication": [communication] * count,
        "sensing": [sensing] * count,
        "objective": [objective] * count,
        "feasibility": [feasibility] * count,
        "minimum_ue_rates": [minimum_ue_rate] * count,
        "minimum_target_rates": [minimum_target_rate] * count,
        "communication_mean": communication,
        "sensing_mean": sensing,
        "objective_mean": objective,
        "feasibility_rate": feasibility,
        "minimum_ue_rate": minimum_ue_rate,
        "minimum_target_rate": minimum_target_rate,
    })
    return result


class ResultValidationTests(unittest.TestCase):
    def test_paired_normalized_baseline_uses_training_weight_convention(self):
        values = paired_normalized_baseline_objective(
            baseline_communication=np.asarray([8.0, 4.0]),
            baseline_sensing=np.asarray([2.0, 6.0]),
            communication_references=np.asarray([10.0, 8.0]),
            sensing_references=np.asarray([4.0, 12.0]),
            pareto_weight=0.25,
        )

        np.testing.assert_allclose(values, [0.725, 0.5])

    def test_none_reference_resolves_to_repository_canonical_file(self):
        references = load_paper_references(None)
        self.assertIn("4", references["figures"])

    def test_canonical_baseline_report_is_recomputed_and_must_match_exactly(self):
        canonical = {"validation": "paper_baseline", "overall_pass": True}
        with patch(
            "validate_results.validate_baseline_result",
            return_value=canonical,
        ) as mocked:
            self.assertEqual(
                require_canonical_baseline_validation({}, canonical, "run"),
                canonical,
            )
            mocked.assert_called_once_with(
                {}, references=None, result_directory="run", verify_files=True
            )
        with patch(
            "validate_results.validate_baseline_result",
            return_value=canonical,
        ):
            with self.assertRaisesRegex(ValueError, "not the canonical"):
                require_canonical_baseline_validation(
                    {}, {**canonical, "extra": True}, "run"
                )
        with patch(
            "validate_results.validate_baseline_result",
            return_value={"overall_pass": False},
        ):
            with self.assertRaisesRegex(ValueError, "fails canonical"):
                require_canonical_baseline_validation({}, {}, "run")

    def _validate_baseline(self, result):
        return validate_baseline_result(
            result, self._references(), verify_files=False
        )

    def _validate_extension(self, extension, baseline):
        validation = self._validate_baseline(baseline)
        extension = copy.deepcopy(extension)
        extension["baseline_payload_sha256"] = result_payload_digest(baseline)
        extension["baseline_validation_sha256"] = result_payload_digest(
            validation
        )
        return validate_extension_result(
            extension,
            baseline,
            baseline_validation=validation,
            verify_files=False,
        )

    def _baseline(self):
        method_values = {
            "DTL-pCSI": [9.45, 9.52, 9.49, 9.51, 9.48],
            "DTL-WFB": [8.90, 8.95, 8.92, 8.94, 8.91],
            "DTL-iCSI": [8.35, 8.42, 8.38, 8.40, 8.37],
            "DDPG": [6.35, 6.42, 6.39, 6.41, 6.38],
        }
        sensing_values = [2.0, 2.1, 2.05, 2.08, 2.02]
        seeds = [1, 2, 3, 4, 5]
        scene_grid = evaluation_scene_seed_grid(seeds, 5)
        result = {
            "schema_version": 1,
            "kind": "generated_result",
            "physical_configuration_fingerprint": "physics-1",
            "profile_digest": "profile-1",
            "source_tree_digest": "a" * 64,
            "training_seed_protocol": TRAINING_SEED_PROTOCOL,
            "evaluation_seed_protocol": EVALUATION_SEED_PROTOCOL,
            "power_hypothesis_id": "paper_db_as_dbm",
            "noise_hypothesis_id": "description_literal_target_matches_ue",
            "dtl_transfer_mode": "eq30_next_state_continuation_blend",
            "dtl_transfer_schedule_id": "constant_upsilon",
            "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
            "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
            "los_probability_hypothesis_id":
                "paper_unnumbered_los_literal_roles",
            "channel_coherence_hypothesis_id": "episode_block_fading",
            "csi_observation_hypothesis_id":
                "instantaneous_estimates_all_csi_qualities",
            "rate_model_hypothesis_id":
                "continuous_matrix_scee_all_csi_qualities",
            "cross_function_interference_hypothesis_id":
                "paper_eq12_eq14_literal_cross_channels",
            "objective": {"mode": "paper_baseline"},
            "feasibility_rate": 1.0,
            "seeds": seeds,
            "episodes": 1500,
            "evaluation_episodes": 5,
            "figure4_protocol": {
                "raw_episode_count": 1500,
                "plot_x_start": 0,
                "plot_x_end": 1400,
                "plot_interval": 100,
                "acceptance_smoothing_window": 20,
            },
            "final_reward_source": "training_communication_rate",
            "training_final_window": 100,
            "final_reward_by_method": {
                method: {"seed_values": values}
                for method, values in method_values.items()
            },
            "communication_seed_values": method_values["DTL-pCSI"],
            "sensing_seed_values": sensing_values,
        }
        reference_curves = self._references()["figures"]["4"]["curves"]
        x = np.arange(0, 1500, 100)
        result["seed_records"] = [
            {
                "seed": seed,
                "source_training_seed": derive_training_seed(seed, "source"),
                "target_training_seed": derive_training_seed(seed, "target"),
                "evaluation_scene_seeds": scene_grid[seed],
                "methods": {
                    method: {
                        "training_seed": derive_training_seed(seed, "target"),
                        "evaluation": _raw_evaluation(
                            values[index], sensing_values[index], values[index]
                        ),
                        "training": {
                            "comm_rates": (
                                np.interp(
                                    np.arange(1500), x, reference_curves[method]
                                )
                                + values[index]
                                - reference_curves[method][-1]
                            ).tolist(),
                            "rewards": (
                                np.interp(
                                    np.arange(1500), x, reference_curves[method]
                                )
                                + values[index]
                                - reference_curves[method][-1]
                            ).tolist(),
                            "sensing_rates": [sensing_values[index]] * 1500,
                        },
                    }
                    for method, values in method_values.items()
                },
            }
            for index, seed in enumerate(seeds)
        ]
        return result

    @staticmethod
    def _references():
        return {
            "schema_version": 1,
            "calibration_figures": [],
            "holdout_figures": [4],
            "figures": {
                "4": {
                    "kind": "digitized_reference",
                    "source_page": 11,
                    "role": "holdout",
                    "x_axis": {"label": "episodes", "units": "episodes"},
                    "y_axis": {"label": "reward", "units": "bits/s/Hz"},
                    "digitization_method": "synthetic unit fixture",
                    "uncertainty": {"absolute_y": 0.08},
                    "x": list(range(0, 1500, 100)),
                    "curves": {
                        "DTL-pCSI": np.linspace(8.2, 9.5, 15).tolist(),
                        "DTL-WFB": np.linspace(6.3, 8.9, 15).tolist(),
                        "DTL-iCSI": np.linspace(7.1, 8.4, 15).tolist(),
                        "DDPG": np.linspace(5.1, 6.4, 15).tolist(),
                    },
                }
            }
        }

    def test_baseline_rejects_generated_result_as_reference(self):
        references = self._references()
        references["figures"]["4"]["kind"] = "generated_result"
        with self.assertRaisesRegex(ValueError, "digitized_reference"):
            validate_baseline_result(
                self._baseline(), references, verify_files=False
            )

    def test_baseline_accepts_ordered_close_multi_seed_result(self):
        baseline = self._baseline()
        report = self._validate_baseline(baseline)
        self.assertTrue(report["overall_pass"])
        self.assertEqual(report["checks"]["method_ordering"]["pass"], True)
        self.assertEqual(
            report["baseline_payload_sha256"], result_payload_digest(baseline)
        )

    def test_baseline_rejects_tampered_evaluation_scene_schedule(self):
        baseline = self._baseline()
        baseline["seed_records"][1]["evaluation_scene_seeds"][0] = (
            baseline["seed_records"][0]["evaluation_scene_seeds"][0]
        )

        report = self._validate_baseline(baseline)

        self.assertFalse(report["checks"]["evaluation_scene_protocol"]["pass"])

    def test_extension_gate_requires_passing_validation_for_exact_baseline(self):
        baseline = self._baseline()
        validation = self._validate_baseline(baseline)
        require_accepted_baseline(validation, baseline)

        validation["overall_pass"] = False
        with self.assertRaisesRegex(ValueError, "has not passed"):
            require_accepted_baseline(validation, baseline)

        validation = self._validate_baseline(baseline)
        changed = copy.deepcopy(baseline)
        changed["communication_seed_values"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "different baseline"):
            require_accepted_baseline(validation, changed)

    def test_baseline_rejects_non_generated_or_infeasible_result(self):
        result = self._baseline()
        result["kind"] = "digitized_reference"
        result["feasibility_rate"] = 0.99
        report = self._validate_baseline(result)
        self.assertFalse(report["overall_pass"])

    def test_baseline_rejects_missing_projection_dependence_evidence(self):
        result = self._baseline()
        del result["seed_records"][0]["methods"]["DTL-pCSI"]["evaluation"][
            "action_projection_rate"
        ]
        report = self._validate_baseline(result)
        self.assertFalse(report["checks"]["projection_dependence_evidence"]["pass"])

    def test_baseline_rejects_missing_or_unknown_ambiguity_contract(self):
        result = self._baseline()
        del result["noise_hypothesis_id"]
        report = self._validate_baseline(result)
        self.assertFalse(report["checks"]["ambiguity_contract"]["pass"])

        result = self._baseline()
        result["dtl_transfer_mode"] = "best_reward"
        report = self._validate_baseline(result)
        self.assertFalse(report["checks"]["ambiguity_contract"]["pass"])

    def test_baseline_rejects_endpoint_without_training_reward_provenance(self):
        result = self._baseline()
        del result["final_reward_source"]
        report = self._validate_baseline(result)
        self.assertFalse(report["overall_pass"])
        self.assertFalse(report["checks"]["training_reward_endpoint"]["pass"])

    def test_baseline_rejects_penalized_or_shaped_training_rewards(self):
        result = self._baseline()
        training = result["seed_records"][0]["methods"]["DTL-pCSI"][
            "training"
        ]
        training["rewards"][100] = training["comm_rates"][100] - 0.25

        report = self._validate_baseline(result)

        self.assertFalse(report["overall_pass"], report)
        self.assertFalse(
            report["checks"]["paper_reward_equals_communication_rate"]["pass"]
        )

    def test_baseline_rejects_wrong_fig4_protocol_or_raw_history_mismatch(self):
        result = self._baseline()
        result["episodes"] = 1000
        report = self._validate_baseline(result)
        self.assertFalse(report["checks"]["figure4_protocol"]["pass"])

        result = self._baseline()
        result["seed_records"][0]["methods"]["DTL-pCSI"]["training"][
            "comm_rates"
        ][-1] = 0.0
        report = self._validate_baseline(result)
        self.assertFalse(report["checks"]["raw_history_consistency"]["pass"])

    def test_baseline_rejects_wrong_learning_curve_with_correct_endpoint(self):
        result = self._baseline()
        history = result["seed_records"][0]["methods"]["DTL-pCSI"]["training"][
            "comm_rates"
        ]
        history[200:900] = [2.0] * 700
        report = self._validate_baseline(result)
        self.assertFalse(report["checks"]["paper_full_curve_tolerance"]["pass"])

    def test_baseline_validation_fails_closed_on_truncated_raw_history(self):
        result = self._baseline()
        for record in result["seed_records"]:
            record["methods"]["DTL-pCSI"]["training"]["comm_rates"] = record[
                "methods"
            ]["DTL-pCSI"]["training"]["comm_rates"][:200]

        report = self._validate_baseline(result)

        self.assertFalse(report["overall_pass"])
        self.assertFalse(report["checks"]["raw_history_consistency"]["pass"])
        self.assertFalse(report["checks"]["paper_full_curve_tolerance"]["pass"])

    def test_sample_mean_ordering_without_paired_confidence_is_rejected(self):
        result = self._baseline()
        result["final_reward_by_method"]["DTL-pCSI"]["seed_values"] = [
            10.0, 10.0, 10.0, 10.0, 8.0
        ]
        result["final_reward_by_method"]["DTL-WFB"]["seed_values"] = [
            9.0, 9.0, 9.0, 9.0, 8.9
        ]
        report = self._validate_baseline(result)
        self.assertFalse(report["checks"]["method_ordering"]["pass"])

    def test_extension_requires_retention_and_significant_sensing_gain(self):
        baseline = self._baseline()
        communication = [9.20, 9.25, 9.22, 9.24, 9.21]
        sensing = [3.2, 3.3, 3.25, 3.28, 3.22]
        extension = {
            "schema_version": 1,
            "kind": "generated_result",
            "physical_configuration_fingerprint": "physics-1",
            "profile_digest": "profile-1",
            "source_tree_digest": "a" * 64,
            "training_seed_protocol": TRAINING_SEED_PROTOCOL,
            "evaluation_seed_protocol": EVALUATION_SEED_PROTOCOL,
            "power_hypothesis_id": "paper_db_as_dbm",
            "noise_hypothesis_id": "description_literal_target_matches_ue",
            "dtl_transfer_mode": "eq30_next_state_continuation_blend",
            "dtl_transfer_schedule_id": "constant_upsilon",
            "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
            "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
            "los_probability_hypothesis_id":
                "paper_unnumbered_los_literal_roles",
            "channel_coherence_hypothesis_id": "episode_block_fading",
            "csi_observation_hypothesis_id":
                "instantaneous_estimates_all_csi_qualities",
            "rate_model_hypothesis_id":
                "continuous_matrix_scee_all_csi_qualities",
            "cross_function_interference_hypothesis_id":
                "paper_eq12_eq14_literal_cross_channels",
            "seeds": [1, 2, 3, 4, 5],
            "evaluation_episodes": 5,
            "objective_reference_protocol":
                "accepted_baseline_training_tail_v1",
            "objective_reference_window": 100,
            "objective": {
                "mode": "comm_retention_sensing",
                "communication_retention_fraction": 0.95,
            },
            "feasibility_rate": 1.0,
            "communication_seed_values": communication,
            "sensing_seed_values": sensing,
            "joint_seed_values": [
                comm + sense for comm, sense in zip(communication, sensing)
            ],
            "objective_seed_values": sensing,
            "minimum_ue_rate_seed_values": [0.8] * 5,
            "minimum_target_rate_seed_values": [0.9] * 5,
            "seed_records": [
                {
                    "seed": seed,
                    "training_seed": derive_training_seed(seed, "target"),
                    "evaluation_scene_seeds": baseline["seed_records"][index][
                        "evaluation_scene_seeds"
                    ],
                    "objective_reference": {
                        "source": "accepted_baseline_training_tail",
                        "window": 100,
                        "communication": baseline[
                            "communication_seed_values"
                        ][index],
                        "sensing": baseline["sensing_seed_values"][index],
                    },
                    "objective": {
                        "mode": "comm_retention_sensing",
                        "user_rate_min": 0.0,
                        "target_rate_min": 0.0,
                        "communication_reference": baseline[
                            "communication_seed_values"
                        ][index],
                        "communication_retention_fraction": 0.95,
                    },
                    "evaluation": _raw_evaluation(
                        communication[index], sensing[index], sensing[index]
                    ),
                }
                for index, seed in enumerate([1, 2, 3, 4, 5])
            ],
        }
        report = self._validate_extension(extension, baseline)
        self.assertTrue(report["overall_pass"])
        self.assertGreater(report["statistics"]["sensing_gain"]["ci95_low"], 0.0)

        unauthorized_report = validate_extension_result(
            extension, baseline, verify_files=False
        )
        self.assertFalse(unauthorized_report["overall_pass"], unauthorized_report)
        self.assertFalse(
            unauthorized_report["checks"]["accepted_baseline_authority"]["pass"]
        )

        spoofed = copy.deepcopy(extension)
        spoofed["communication_seed_values"] = [20.0] * 5
        spoofed_report = self._validate_extension(spoofed, baseline)
        self.assertFalse(spoofed_report["overall_pass"], spoofed_report)
        self.assertFalse(
            spoofed_report["checks"]["raw_evaluation_consistency"]["pass"]
        )

        mismatch = copy.deepcopy(extension)
        mismatch["dtl_transfer_schedule_id"] = "linear_annealed_upsilon"
        mismatch_report = self._validate_extension(mismatch, baseline)
        self.assertFalse(mismatch_report["checks"]["same_ambiguity_contract"]["pass"])

        failing = copy.deepcopy(extension)
        failing["communication_seed_values"] = [7.0, 7.1, 7.2, 7.1, 7.0]
        self.assertFalse(
            self._validate_extension(failing, baseline)["overall_pass"]
        )

        missing_projection = copy.deepcopy(extension)
        del missing_projection["seed_records"][0]["evaluation"][
            "allocation_fallback_rate"
        ]
        self.assertFalse(
            self._validate_extension(missing_projection, baseline)["checks"][
                "projection_dependence_evidence"
            ]["pass"]
        )

    def test_png_extension_has_no_retention_and_checks_joint_and_rate_floors(self):
        baseline = self._baseline()
        communication = [9.5, 9.6, 9.55, 9.58, 9.52]
        sensing = [3.2, 3.3, 3.25, 3.28, 3.22]
        joint = [
            comm + sense for comm, sense in zip(communication, sensing)
        ]
        extension = {
            "schema_version": 1,
            "kind": "generated_result",
            "physical_configuration_fingerprint": "physics-1",
            "profile_digest": "profile-1",
            "source_tree_digest": "a" * 64,
            "training_seed_protocol": TRAINING_SEED_PROTOCOL,
            "evaluation_seed_protocol": EVALUATION_SEED_PROTOCOL,
            "power_hypothesis_id": "paper_db_as_dbm",
            "noise_hypothesis_id": "description_literal_target_matches_ue",
            "dtl_transfer_mode": "eq30_next_state_continuation_blend",
            "dtl_transfer_schedule_id": "constant_upsilon",
            "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
            "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
            "los_probability_hypothesis_id":
                "paper_unnumbered_los_literal_roles",
            "channel_coherence_hypothesis_id": "episode_block_fading",
            "csi_observation_hypothesis_id":
                "instantaneous_estimates_all_csi_qualities",
            "rate_model_hypothesis_id":
                "continuous_matrix_scee_all_csi_qualities",
            "cross_function_interference_hypothesis_id":
                "paper_eq12_eq14_literal_cross_channels",
            "seeds": [1, 2, 3, 4, 5],
            "evaluation_episodes": 5,
            "objective_reference_protocol":
                "accepted_baseline_training_tail_v1",
            "objective_reference_window": 100,
            "objective": {
                "mode": "png_joint_sum",
                "user_rate_min": 0.5,
                "target_rate_min": 0.5,
            },
            "feasibility_rate": 1.0,
            "communication_seed_values": communication,
            "sensing_seed_values": sensing,
            "joint_seed_values": joint,
            "objective_seed_values": joint,
            "minimum_ue_rate_seed_values": [0.8] * 5,
            "minimum_target_rate_seed_values": [0.9] * 5,
            "seed_records": [
                {
                    "seed": seed,
                    "training_seed": derive_training_seed(seed, "target"),
                    "evaluation_scene_seeds": baseline["seed_records"][index][
                        "evaluation_scene_seeds"
                    ],
                    "objective_reference": {
                        "source": "accepted_baseline_training_tail",
                        "window": 100,
                        "communication": baseline[
                            "communication_seed_values"
                        ][index],
                        "sensing": baseline["sensing_seed_values"][index],
                    },
                    "objective": {
                        "mode": "png_joint_sum",
                        "user_rate_min": 0.5,
                        "target_rate_min": 0.5,
                    },
                    "evaluation": _raw_evaluation(
                        communication[index], sensing[index], joint[index]
                    ),
                }
                for index, seed in enumerate([1, 2, 3, 4, 5])
            ],
        }
        report = self._validate_extension(extension, baseline)
        self.assertTrue(report["overall_pass"], report)
        self.assertTrue(report["checks"]["positive_paired_joint_gain"]["pass"])
        self.assertNotIn("communication_retention", report["checks"])

        failing = copy.deepcopy(extension)
        failing["minimum_target_rate_seed_values"][0] = 0.1
        self.assertFalse(
            self._validate_extension(failing, baseline)["overall_pass"]
        )

        no_gain = copy.deepcopy(extension)
        no_gain["communication_seed_values"] = [7.0] * 5
        no_gain["joint_seed_values"] = [
            communication + sensing
            for communication, sensing in zip(
                no_gain["communication_seed_values"],
                no_gain["sensing_seed_values"],
            )
        ]
        no_gain_report = self._validate_extension(no_gain, baseline)
        self.assertFalse(no_gain_report["overall_pass"])
        self.assertFalse(
            no_gain_report["checks"]["positive_paired_joint_gain"]["pass"]
        )


if __name__ == "__main__":
    unittest.main()
