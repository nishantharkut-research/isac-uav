import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ambiguity_hypotheses import (
    apply_ambiguity_hypotheses,
    apply_ambiguity_manifest_fields,
)
from ambiguity_protocol import build_sensitivity_plan, build_sensitivity_report
from config import cfg
from run_ambiguity_sensitivity import (
    _require_compatible_primary,
    build_parser as sensitivity_parser,
    run as run_sensitivity,
)
from validate_results import result_payload_digest
from simulation import _configuration_fingerprint


class AmbiguityProtocolTests(unittest.TestCase):
    def test_one_factor_plan_has_primary_and_each_single_axis_alternative(self):
        plan = build_sensitivity_plan("one_at_a_time")
        self.assertEqual(plan["design"], "one_at_a_time")
        self.assertEqual(len(plan["scenarios"]), 9)
        self.assertEqual(
            [scenario["id"] for scenario in plan["scenarios"]],
            [
                "primary",
                "noise_dual_minus70",
                "transfer_next_state",
                "schedule_linear_annealed",
                "wfb_channel_independent_dft",
                "ddpg_icsi_comparator",
                "los_standard_mapped_roles",
                "independent_slot_fading",
                "receiver_consistent_cross_interference",
            ],
        )
        self.assertNotIn("selected_hypothesis", plan)
        self.assertEqual(plan["result_selection"], "none_report_all")

    def test_full_factorial_plan_has_all_256_unique_combinations(self):
        plan = build_sensitivity_plan("full_factorial")
        combinations = {
            tuple(sorted(scenario["hypotheses"].items()))
            for scenario in plan["scenarios"]
        }
        self.assertEqual(len(plan["scenarios"]), 256)
        self.assertEqual(len(combinations), 256)

    def test_unknown_design_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown sensitivity design"):
            build_sensitivity_plan("choose_best")

    def test_configuration_fingerprint_changes_on_every_algorithmic_axis(self):
        primary = copy.copy(cfg)
        apply_ambiguity_hypotheses(primary)
        baseline = _configuration_fingerprint({
            "transfer_update": primary.dtl_transfer_mode,
            "transfer_schedule": primary.dtl_transfer_schedule_id,
            "wfb": primary.wfb_hypothesis_id,
            "ddpg_comparator_csi":
                primary.ddpg_comparator_csi_hypothesis_id,
            "los_probability": primary.los_probability_hypothesis_id,
            "channel_coherence": primary.channel_coherence_hypothesis_id,
            "cross_function_interference":
                primary.cross_function_interference_hypothesis_id,
        })
        alternatives = [
            {"transfer_update_hypothesis_id":
             "eq30_next_state_continuation_blend"},
            {"transfer_schedule_hypothesis_id": "linear_annealed_upsilon"},
            {"wfb_hypothesis_id": "channel_independent_dft_no_capon"},
            {"ddpg_comparator_csi_hypothesis_id": "icsi_comparator"},
            {"los_probability_hypothesis_id":
             "al_hourani_standard_mapped_roles"},
            {"channel_coherence_hypothesis_id": "independent_slot_fading"},
            {"cross_function_interference_hypothesis_id":
             "receiver_consistent_cross_channels"},
        ]
        for overrides in alternatives:
            with self.subTest(overrides=overrides):
                candidate = copy.copy(cfg)
                apply_ambiguity_hypotheses(candidate, **overrides)
                actual = _configuration_fingerprint({
                    "transfer_update": candidate.dtl_transfer_mode,
                    "transfer_schedule": candidate.dtl_transfer_schedule_id,
                    "wfb": candidate.wfb_hypothesis_id,
                    "ddpg_comparator_csi":
                        candidate.ddpg_comparator_csi_hypothesis_id,
                    "los_probability":
                        candidate.los_probability_hypothesis_id,
                    "channel_coherence":
                        candidate.channel_coherence_hypothesis_id,
                    "cross_function_interference":
                        candidate.cross_function_interference_hypothesis_id,
                })
                self.assertNotEqual(actual, baseline)

    def test_manifest_fields_lock_every_ambiguity_axis(self):
        candidate = copy.copy(cfg)
        apply_ambiguity_hypotheses(candidate)
        manifest = apply_ambiguity_manifest_fields(
            {"kind": "run_manifest"}, candidate
        )
        self.assertEqual(
            manifest["noise_hypothesis_id"],
            "description_literal_target_matches_ue",
        )
        self.assertEqual(
            manifest["dtl_transfer_mode"], "eq30_current_q_bellman_blend"
        )
        self.assertEqual(
            manifest["dtl_transfer_schedule_id"], "constant_upsilon"
        )
        self.assertEqual(
            manifest["wfb_hypothesis_id"],
            "adaptive_matched_filter_no_capon",
        )
        self.assertEqual(
            manifest["ddpg_comparator_csi_hypothesis_id"],
            "fair_pcsi_comparator",
        )
        self.assertEqual(
            manifest["los_probability_hypothesis_id"],
            "paper_eq6_literal_roles",
        )
        self.assertEqual(
            manifest["channel_coherence_hypothesis_id"],
            "episode_block_fading",
        )
        self.assertEqual(
            manifest["cross_function_interference_hypothesis_id"],
            "paper_eq12_eq14_literal_cross_channels",
        )
        self.assertNotIn("selected_hypothesis", manifest["ambiguity_contract"])

    def test_sensitivity_report_is_paired_and_never_selects_a_winner(self):
        plan = build_sensitivity_plan("one_at_a_time")
        summaries = {}
        for index, scenario in enumerate(plan["scenarios"]):
            hypotheses = scenario["hypotheses"]
            summaries[scenario["id"]] = {
                "kind": "generated_result",
                "protocol": "paper_baseline",
                "profile_digest": "profile",
                "power_hypothesis_id": "paper_db_as_dbm",
                "source_tree_digest": "a" * 64,
                "training_seed_protocol": "sha256(base_seed,method,role,v1)",
                "objective": {"mode": "paper_baseline"},
                "seeds": [1, 2, 3, 4, 5],
                "episodes": 1500,
                "evaluation_episodes": 5,
                "noise_hypothesis_id": hypotheses["noise"],
                "dtl_transfer_mode": hypotheses["transfer_update"],
                "dtl_transfer_schedule_id": hypotheses["transfer_schedule"],
                "wfb_hypothesis_id": hypotheses["wfb"],
                "ddpg_comparator_csi_hypothesis_id":
                    hypotheses["ddpg_comparator_csi"],
                "los_probability_hypothesis_id":
                    hypotheses["los_probability"],
                "channel_coherence_hypothesis_id":
                    hypotheses["channel_coherence"],
                "cross_function_interference_hypothesis_id":
                    hypotheses["cross_function_interference"],
                "final_reward_by_method": {
                    "DTL-pCSI": {
                        "seed_values": [8 + index + seed / 100 for seed in range(5)]
                    }
                },
                "communication_seed_values": [7 + index] * 5,
                "sensing_seed_values": [2 + index] * 5,
            }
        report = build_sensitivity_report(plan, summaries)
        self.assertEqual(report["result_selection"], "none_report_all")
        self.assertNotIn("selected_hypothesis", report)
        self.assertEqual(
            report["comparisons"]["noise_dual_minus70"]
                  ["final_reward_delta_by_method"]["DTL-pCSI"]["mean"],
            1.0,
        )

    def test_sensitivity_report_rejects_unpaired_seed_sets(self):
        plan = build_sensitivity_plan("one_at_a_time")
        summaries = {
            scenario["id"]: {
                "kind": "generated_result",
                "protocol": "paper_baseline",
                "profile_digest": "profile",
                "power_hypothesis_id": "paper_db_as_dbm",
                "source_tree_digest": "a" * 64,
                "training_seed_protocol": "sha256(base_seed,method,role,v1)",
                "objective": {"mode": "paper_baseline"},
                "seeds": [1, 2, 3, 4, 5],
                "episodes": 1500,
                "evaluation_episodes": 5,
                "noise_hypothesis_id": scenario["hypotheses"]["noise"],
                "dtl_transfer_mode":
                    scenario["hypotheses"]["transfer_update"],
                "dtl_transfer_schedule_id":
                    scenario["hypotheses"]["transfer_schedule"],
                "wfb_hypothesis_id": scenario["hypotheses"]["wfb"],
                "ddpg_comparator_csi_hypothesis_id":
                    scenario["hypotheses"]["ddpg_comparator_csi"],
                "los_probability_hypothesis_id":
                    scenario["hypotheses"]["los_probability"],
                "channel_coherence_hypothesis_id":
                    scenario["hypotheses"]["channel_coherence"],
                "cross_function_interference_hypothesis_id":
                    scenario["hypotheses"]["cross_function_interference"],
                "final_reward_by_method": {},
                "communication_seed_values": [1] * 5,
                "sensing_seed_values": [1] * 5,
            }
            for scenario in plan["scenarios"]
        }
        summaries["noise_dual_minus70"]["seeds"] = [6, 7, 8, 9, 10]
        with self.assertRaisesRegex(ValueError, "paired provenance"):
            build_sensitivity_report(plan, summaries)

    def test_sensitivity_report_rejects_different_source_tree(self):
        plan = build_sensitivity_plan("one_at_a_time")
        summaries = {}
        for scenario in plan["scenarios"]:
            hypotheses = scenario["hypotheses"]
            summaries[scenario["id"]] = {
                "kind": "generated_result",
                "protocol": "paper_baseline",
                "profile_digest": "profile",
                "power_hypothesis_id": "paper_db_as_dbm",
                "source_tree_digest": "a" * 64,
                "training_seed_protocol": "sha256(base_seed,method,role,v1)",
                "objective": {"mode": "paper_baseline"},
                "seeds": [1, 2, 3, 4, 5],
                "episodes": 1500,
                "evaluation_episodes": 5,
                "noise_hypothesis_id": hypotheses["noise"],
                "dtl_transfer_mode": hypotheses["transfer_update"],
                "dtl_transfer_schedule_id": hypotheses["transfer_schedule"],
                "wfb_hypothesis_id": hypotheses["wfb"],
                "ddpg_comparator_csi_hypothesis_id":
                    hypotheses["ddpg_comparator_csi"],
                "los_probability_hypothesis_id":
                    hypotheses["los_probability"],
                "channel_coherence_hypothesis_id":
                    hypotheses["channel_coherence"],
                "cross_function_interference_hypothesis_id":
                    hypotheses["cross_function_interference"],
                "final_reward_by_method": {},
                "communication_seed_values": [1] * 5,
                "sensing_seed_values": [1] * 5,
            }
        summaries["transfer_next_state"]["source_tree_digest"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "paired provenance"):
            build_sensitivity_report(plan, summaries)

    def test_sensitivity_report_rejects_mislabeled_hypothesis_output(self):
        plan = build_sensitivity_plan("one_at_a_time")
        summaries = {}
        for scenario in plan["scenarios"]:
            hypotheses = scenario["hypotheses"]
            summaries[scenario["id"]] = {
                "kind": "generated_result",
                "protocol": "paper_baseline",
                "profile_digest": "profile",
                "power_hypothesis_id": "paper_db_as_dbm",
                "source_tree_digest": "a" * 64,
                "training_seed_protocol": "sha256(base_seed,method,role,v1)",
                "objective": {"mode": "paper_baseline"},
                "seeds": [1, 2, 3, 4, 5],
                "episodes": 1500,
                "evaluation_episodes": 5,
                "noise_hypothesis_id": hypotheses["noise"],
                "dtl_transfer_mode": hypotheses["transfer_update"],
                "dtl_transfer_schedule_id": hypotheses["transfer_schedule"],
                "wfb_hypothesis_id": hypotheses["wfb"],
                "ddpg_comparator_csi_hypothesis_id":
                    hypotheses["ddpg_comparator_csi"],
                "los_probability_hypothesis_id":
                    hypotheses["los_probability"],
                "channel_coherence_hypothesis_id":
                    hypotheses["channel_coherence"],
                "cross_function_interference_hypothesis_id":
                    hypotheses["cross_function_interference"],
                "final_reward_by_method": {},
                "communication_seed_values": [1] * 5,
                "sensing_seed_values": [1] * 5,
            }
        summaries["noise_dual_minus70"]["noise_hypothesis_id"] = (
            "description_literal_target_matches_ue"
        )
        with self.assertRaisesRegex(ValueError, "hypothesis identity"):
            build_sensitivity_report(plan, summaries)

    def test_sensitivity_cli_defaults_to_practical_one_factor_design(self):
        args = sensitivity_parser().parse_args([
            "--profile", "profile.json", "--output-root", "out",
            "--seeds", "1,2,3,4,5", "--dry-run",
        ])
        self.assertEqual(args.design, "one_at_a_time")
        self.assertTrue(args.dry_run)

    def test_external_accepted_primary_avoids_duplicate_primary_training(self):
        baseline = {
            "kind": "generated_result",
            "protocol": "paper_baseline",
            "profile_digest": "profile",
            "source_tree_digest": "a" * 64,
            "seeds": [1, 2, 3, 4, 5],
            "episodes": 1500,
            "evaluation_episodes": 5,
            "noise_hypothesis_id": "description_literal_target_matches_ue",
            "dtl_transfer_mode": "eq30_current_q_bellman_blend",
            "dtl_transfer_schedule_id": "constant_upsilon",
            "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
            "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
            "los_probability_hypothesis_id": "paper_eq6_literal_roles",
            "channel_coherence_hypothesis_id": "episode_block_fading",
            "cross_function_interference_hypothesis_id":
                "paper_eq12_eq14_literal_cross_channels",
        }
        validation = {
            "validation": "paper_baseline",
            "overall_pass": True,
            "baseline_payload_sha256": result_payload_digest(baseline),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            validation_path = root / "validation.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            args = sensitivity_parser().parse_args([
                "--profile", "profile.json",
                "--output-root", str(root / "sensitivity"),
                "--seeds", "1,2,3,4,5",
                "--primary-baseline", str(baseline_path),
                "--primary-validation", str(validation_path),
                "--dry-run",
            ])
            with patch("run_baseline.run", return_value=0) as mocked, patch(
                "parameter_profiles.load_calibrated_profile",
                return_value=SimpleNamespace(digest="profile"),
            ), patch("source_integrity.source_tree_digest", return_value="a" * 64):
                run_sensitivity(args)
            self.assertEqual(mocked.call_count, 8)

    def test_external_primary_mismatch_fails_before_sensitivity_training(self):
        primary = {
            "kind": "generated_result",
            "protocol": "paper_baseline",
            "profile_digest": "profile",
            "source_tree_digest": "a" * 64,
            "seeds": [1, 2, 3, 4, 5],
            "episodes": 1000,
            "evaluation_episodes": 5,
            "noise_hypothesis_id": "description_literal_target_matches_ue",
            "dtl_transfer_mode": "eq30_current_q_bellman_blend",
            "dtl_transfer_schedule_id": "constant_upsilon",
            "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
            "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
            "los_probability_hypothesis_id": "paper_eq6_literal_roles",
            "channel_coherence_hypothesis_id": "episode_block_fading",
            "cross_function_interference_hypothesis_id":
                "paper_eq12_eq14_literal_cross_channels",
        }
        with self.assertRaisesRegex(ValueError, "episodes"):
            _require_compatible_primary(
                primary,
                profile_digest="profile",
                current_source_digest="a" * 64,
                seeds=(1, 2, 3, 4, 5),
                episodes=1500,
                evaluation_episodes=5,
            )


if __name__ == "__main__":
    unittest.main()
