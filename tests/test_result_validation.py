import copy
import unittest

import numpy as np

from experiment_protocol import TRAINING_SEED_PROTOCOL, derive_training_seed
from run_extension import require_accepted_baseline
from validate_results import (
    result_payload_digest,
    validate_baseline_result,
    validate_extension_result,
)


class ResultValidationTests(unittest.TestCase):
    def _validate_baseline(self, result):
        return validate_baseline_result(
            result, self._references(), verify_files=False
        )

    @staticmethod
    def _validate_extension(extension, baseline):
        return validate_extension_result(
            extension, baseline, verify_files=False
        )

    def _baseline(self):
        method_values = {
            "DTL-pCSI": [9.45, 9.52, 9.49, 9.51, 9.48],
            "DTL-WFB": [8.90, 8.95, 8.92, 8.94, 8.91],
            "DTL-iCSI": [8.35, 8.42, 8.38, 8.40, 8.37],
            "DDPG": [6.35, 6.42, 6.39, 6.41, 6.38],
        }
        seeds = [1, 2, 3, 4, 5]
        result = {
            "schema_version": 1,
            "kind": "generated_result",
            "physical_configuration_fingerprint": "physics-1",
            "profile_digest": "profile-1",
            "source_tree_digest": "a" * 64,
            "training_seed_protocol": TRAINING_SEED_PROTOCOL,
            "power_hypothesis_id": "paper_db_as_dbm",
            "noise_hypothesis_id": "description_literal_target_matches_ue",
            "dtl_transfer_mode": "eq30_current_q_bellman_blend",
            "dtl_transfer_schedule_id": "constant_upsilon",
            "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
            "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
            "objective": {"mode": "paper_baseline"},
            "feasibility_rate": 1.0,
            "seeds": seeds,
            "episodes": 1500,
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
            "sensing_seed_values": [2.0, 2.1, 2.05, 2.08, 2.02],
        }
        reference_curves = self._references()["figures"]["4"]["curves"]
        x = np.arange(0, 1500, 100)
        result["seed_records"] = [
            {
                "seed": seed,
                "source_training_seed": derive_training_seed(seed, "source"),
                "target_training_seed": derive_training_seed(seed, "target"),
                "methods": {
                    method: {
                        "training_seed": derive_training_seed(seed, "target"),
                        "training": {"comm_rates": (
                            np.interp(np.arange(1500), x, reference_curves[method])
                            + values[index] - reference_curves[method][-1]
                        ).tolist()}
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
        extension = {
            "schema_version": 1,
            "kind": "generated_result",
            "physical_configuration_fingerprint": "physics-1",
            "profile_digest": "profile-1",
            "source_tree_digest": "a" * 64,
            "training_seed_protocol": TRAINING_SEED_PROTOCOL,
            "power_hypothesis_id": "paper_db_as_dbm",
            "noise_hypothesis_id": "description_literal_target_matches_ue",
            "dtl_transfer_mode": "eq30_current_q_bellman_blend",
            "dtl_transfer_schedule_id": "constant_upsilon",
            "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
            "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
            "seeds": [1, 2, 3, 4, 5],
            "objective": {
                "mode": "comm_retention_sensing",
                "communication_retention_fraction": 0.95,
            },
            "feasibility_rate": 1.0,
            "communication_seed_values": [9.20, 9.25, 9.22, 9.24, 9.21],
            "sensing_seed_values": [3.2, 3.3, 3.25, 3.28, 3.22],
            "seed_records": [
                {"seed": seed, "training_seed": derive_training_seed(seed, "target")}
                for seed in [1, 2, 3, 4, 5]
            ],
        }
        report = self._validate_extension(extension, baseline)
        self.assertTrue(report["overall_pass"])
        self.assertGreater(report["statistics"]["sensing_gain"]["ci95_low"], 0.0)

        mismatch = copy.deepcopy(extension)
        mismatch["dtl_transfer_schedule_id"] = "linear_annealed_upsilon"
        mismatch_report = self._validate_extension(mismatch, baseline)
        self.assertFalse(mismatch_report["checks"]["same_ambiguity_contract"]["pass"])

        failing = copy.deepcopy(extension)
        failing["communication_seed_values"] = [7.0, 7.1, 7.2, 7.1, 7.0]
        self.assertFalse(
            self._validate_extension(failing, baseline)["overall_pass"]
        )

    def test_png_extension_has_no_retention_and_checks_joint_and_rate_floors(self):
        baseline = self._baseline()
        extension = {
            "schema_version": 1,
            "kind": "generated_result",
            "physical_configuration_fingerprint": "physics-1",
            "profile_digest": "profile-1",
            "source_tree_digest": "a" * 64,
            "training_seed_protocol": TRAINING_SEED_PROTOCOL,
            "power_hypothesis_id": "paper_db_as_dbm",
            "noise_hypothesis_id": "description_literal_target_matches_ue",
            "dtl_transfer_mode": "eq30_current_q_bellman_blend",
            "dtl_transfer_schedule_id": "constant_upsilon",
            "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
            "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
            "seeds": [1, 2, 3, 4, 5],
            "objective": {
                "mode": "png_joint_sum",
                "user_rate_min": 0.5,
                "target_rate_min": 0.5,
            },
            "feasibility_rate": 1.0,
            "communication_seed_values": [7.0, 7.1, 7.2, 7.1, 7.0],
            "sensing_seed_values": [3.2, 3.3, 3.25, 3.28, 3.22],
            "joint_seed_values": [10.2, 10.4, 10.45, 10.38, 10.22],
            "minimum_ue_rate_seed_values": [0.8] * 5,
            "minimum_target_rate_seed_values": [0.9] * 5,
            "seed_records": [
                {"seed": seed, "training_seed": derive_training_seed(seed, "target")}
                for seed in [1, 2, 3, 4, 5]
            ],
        }
        report = self._validate_extension(extension, baseline)
        self.assertTrue(report["overall_pass"], report)
        self.assertNotIn("communication_retention", report["checks"])

        failing = copy.deepcopy(extension)
        failing["minimum_target_rate_seed_values"][0] = 0.1
        self.assertFalse(
            self._validate_extension(failing, baseline)["overall_pass"]
        )


if __name__ == "__main__":
    unittest.main()
