import copy
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from experiment_protocol import TRAINING_SEED_PROTOCOL, derive_training_seed
from holdout_protocol import (
    CALIBRATION_FIGURES,
    HOLDOUT_FIGURES,
    aggregate_seed_payloads,
    validate_holdout_result,
)
from run_holdouts import _load_progress, build_parser
from plot_holdouts import main as plot_holdouts
from paper_reference import paper_reference_digest
from result_evidence import validate_result_evidence_bundle
from source_integrity import source_tree_digest
from validate_results import result_payload_digest
from simulation import FeasibilityEvidence


class HoldoutProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference_path = Path(__file__).parents[1] / "references" / (
            "paper_reference_curves.json"
        )
        cls.references = json.loads(reference_path.read_text(encoding="utf-8"))

    def _figure_payloads(self):
        references = self.references["figures"]
        payloads = {
            "5": {
                "x": references["5"]["x"],
                "curves": copy.deepcopy(references["5"]["curves"]),
            },
            "6": {
                "x": references["6"]["x"],
                "curves": copy.deepcopy(references["6"]["curves"]),
            },
            "7": {
                "x": references["7"]["x"],
                "curves": copy.deepcopy(references["7"]["curves"]),
            },
            "8": {
                "x": references["8"]["x"],
                "curves": {
                    "HRIS, L=40": references["8"]["curves"]["HRIS-L40"],
                    "HRIS, L=20": references["8"]["curves"]["HRIS-L20"],
                    "Passive RIS, L=40": references["8"]["curves"]["Passive-L40"],
                    "Passive RIS, L=20": references["8"]["curves"]["Passive-L20"],
                },
            },
            "9": {
                "x": references["9"]["x"],
                "curves": copy.deepcopy(references["9"]["curves"]),
            },
            "13": {
                "x": references["13"]["x"],
                "curves": {
                    "R_u, L=40": references["13"]["curves"]["Ru-L40"],
                    "R_u, L=30": references["13"]["curves"]["Ru-L30"],
                    "R_u, L=20": references["13"]["curves"]["Ru-L20"],
                    "R_k, L=30": references["13"]["curves"]["Rk-L30"],
                    "R_k, L=20": references["13"]["curves"]["Rk-L20"],
                },
            },
            "14": {
                "x": references["14"]["x"],
                "curves": copy.deepcopy(references["14"]["curves"]),
            },
        }
        for payload in payloads.values():
            payload["feasibility_rate"] = 1.0
            payload["feasibility_verification"] = (
                "fail_closed_per_generated_realization"
            )
            payload["verified_realizations"] = 100
            payload["failed_realizations"] = 0
            payload["figure_json_sha256"] = "d" * 64
            payload["figure_npz_sha256"] = "e" * 64
        return payloads

    @staticmethod
    def _materialize_raw_files(root, summary):
        for record in summary["seed_records"]:
            seed = record["seed"]
            directory = root / "seed_runs" / f"seed-{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            for figure, payload in record["figures"].items():
                json_path = directory / f"fig{figure}_data.json"
                npz_path = directory / f"fig{figure}_data.npz"
                json_path.write_bytes(f"json:{seed}:{figure}".encode("ascii"))
                npz_path.write_bytes(f"npz:{seed}:{figure}".encode("ascii"))
                payload["figure_json_sha256"] = hashlib.sha256(
                    json_path.read_bytes()
                ).hexdigest()
                payload["figure_npz_sha256"] = hashlib.sha256(
                    npz_path.read_bytes()
                ).hexdigest()
        summary["figures"] = aggregate_seed_payloads(summary["seed_records"])

    def test_feasibility_evidence_is_counted_and_cannot_be_empty(self):
        evidence = FeasibilityEvidence()
        with self.assertRaisesRegex(ValueError, "no verified realizations"):
            evidence.as_metadata()
        evidence.require(True, "first")
        evidence.require(True, "second")
        metadata = evidence.as_metadata()["physical_feasibility"]
        self.assertEqual(metadata["verified_realizations"], 2)
        self.assertEqual(metadata["failed_realizations"], 0)
        self.assertEqual(metadata["rate"], 1.0)

    def _summary(self):
        records = [
            {
                "seed": seed,
                "source_training_seed": derive_training_seed(seed, "source"),
                "target_training_seed": derive_training_seed(seed, "target"),
                "figures": copy.deepcopy(self._figure_payloads()),
            }
            for seed in (1, 2, 3, 4, 5)
        ]
        summary = {
            "schema_version": 1,
            "kind": "generated_result",
            "protocol": "paper_calibration_and_holdouts",
            "objective": {
                "mode": "paper_baseline",
                "reported_reward": "sum_u_R_u",
            },
            "profile_digest": "profile",
            "physical_configuration_fingerprint": "physics",
            "source_tree_digest": "a" * 64,
            "training_seed_protocol": TRAINING_SEED_PROTOCOL,
            "power_hypothesis_id": "paper_db_as_dbm",
            "noise_hypothesis_id": "description_literal_target_matches_ue",
            "dtl_transfer_mode": "eq30_current_q_bellman_blend",
            "dtl_transfer_schedule_id": "constant_upsilon",
            "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
            "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
            "los_probability_hypothesis_id":
                "paper_unnumbered_los_literal_roles",
            "channel_coherence_hypothesis_id": "episode_block_fading",
            "csi_observation_hypothesis_id":
                "pcsi_instantaneous_icsi_statistical",
            "rate_model_hypothesis_id":
                "continuous_matrix_scee_all_csi_qualities",
            "cross_function_interference_hypothesis_id":
                "paper_eq12_eq14_literal_cross_channels",
            "baseline_payload_sha256": "b" * 64,
            "baseline_validation_sha256": "c" * 64,
            "calibration_figures": list(CALIBRATION_FIGURES),
            "holdout_figures": list(HOLDOUT_FIGURES),
            "seeds": [1, 2, 3, 4, 5],
            "episodes": 1000,
            "evaluation_episodes": 5,
            "feasibility_rate": 1.0,
            "feasibility_provenance": "fail-closed full feasibility checks",
            "seed_records": records,
        }
        summary["figures"] = aggregate_seed_payloads(records)
        return summary

    def test_protocol_keeps_calibration_and_holdouts_disjoint(self):
        self.assertEqual(CALIBRATION_FIGURES, (6,))
        self.assertEqual(HOLDOUT_FIGURES, (5, 7, 8, 9, 13, 14))
        self.assertFalse(set(CALIBRATION_FIGURES) & set(HOLDOUT_FIGURES))

    def test_aggregation_retains_raw_seed_arrays_and_confidence_intervals(self):
        summary = self._summary()
        figure = summary["figures"]["7"]
        self.assertEqual(figure["x"], [20, 30, 40, 50, 60])
        self.assertEqual(len(figure["raw_seed_curves"]["HRIS-pCSI"]), 5)
        self.assertEqual(len(figure["ci95_low"]["HRIS-pCSI"]), 5)

    def test_matching_multi_seed_holdouts_are_accepted(self):
        report = validate_holdout_result(self._summary(), self.references)
        self.assertTrue(report["overall_pass"], report)
        self.assertTrue(report["checks"]["holdout_separation"]["pass"])
        self.assertEqual(
            report["paper_reference_sha256"],
            paper_reference_digest(self.references),
        )

    def test_static_summary_cannot_hide_infeasible_figure_evidence(self):
        summary = self._summary()
        summary["seed_records"][0]["figures"]["7"]["feasibility_rate"] = 0.99
        report = validate_holdout_result(summary, self.references)
        self.assertFalse(report["checks"]["physical_feasibility"]["pass"])

    def test_aggregation_requires_per_figure_feasibility_evidence(self):
        summary = self._summary()
        del summary["seed_records"][0]["figures"]["7"]["feasibility_rate"]
        with self.assertRaisesRegex(ValueError, "feasibility"):
            aggregate_seed_payloads(summary["seed_records"])

    def test_holdouts_reject_missing_or_unknown_ambiguity_contract(self):
        missing = self._summary()
        del missing["noise_hypothesis_id"]
        report = validate_holdout_result(missing, self.references)
        self.assertFalse(report["checks"]["ambiguity_contract"]["pass"])

        unknown = self._summary()
        unknown["wfb_hypothesis_id"] = "result_selected_wfb"
        report = validate_holdout_result(unknown, self.references)
        self.assertFalse(report["checks"]["ambiguity_contract"]["pass"])

    def test_changed_reference_holdout_is_rejected_without_curve_repair(self):
        summary = self._summary()
        for values in summary["figures"]["7"]["raw_seed_curves"].values():
            for row in values:
                row[:] = [0.1] * len(row)
        for values in summary["figures"]["7"]["curves"].values():
            values[:] = [0.1] * len(values)
        report = validate_holdout_result(summary, self.references)
        self.assertFalse(report["overall_pass"])
        self.assertFalse(report["figure_checks"]["7"]["reference_tolerance"]["pass"])

    def test_runner_requires_baseline_validation_and_five_seeds(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])
        parsed = build_parser().parse_args([
            "--profile", "profile.json",
            "--baseline-summary", "baseline.json",
            "--baseline-validation", "baseline-validation.json",
            "--output", "out",
            "--seeds", "1,2,3,4,5",
            "--dry-run",
        ])
        self.assertTrue(parsed.dry_run)

    def test_partial_per_figure_progress_is_resumeable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            path.write_text(json.dumps({
                "seed_records": [{
                    "seed": 1,
                    "source_training_seed": derive_training_seed(1, "source"),
                    "target_training_seed": derive_training_seed(1, "target"),
                    "figures": {"5": self._figure_payloads()["5"]},
                }]
            }), encoding="utf-8")
            progress = _load_progress(path, (1, 2, 3, 4, 5), fresh=False)
            self.assertIn("5", progress[1]["figures"])
            self.assertEqual(progress[2]["figures"], {})

    def test_figure6_budget_and_power_metadata_are_not_hardcoded(self):
        import simulation

        signature = inspect.signature(simulation.fig6_rate_vs_power)
        self.assertIn("n_ep", signature.parameters)
        self.assertIn("n_eval", signature.parameters)
        source = inspect.getsource(simulation.fig6_rate_vs_power)
        self.assertNotIn("interpreted as dBm and converted", source)
        self.assertIn("cfg.power_hypothesis_id", source)

    def test_plotter_writes_complete_hashed_evidence_for_every_figure(self):
        summary = self._summary()
        summary["source_tree_digest"] = source_tree_digest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._materialize_raw_files(root, summary)
            summary_path = root / "holdout_summary.json"
            output = root / "plots"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            (root / "run_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "kind": "run_manifest",
                "status": "complete",
                "summary": "holdout_summary.json",
                "summary_payload_sha256": result_payload_digest(summary),
                **{
                    field: summary[field]
                    for field in (
                        "protocol", "objective", "profile_digest",
                        "physical_configuration_fingerprint",
                        "source_tree_digest", "power_hypothesis_id",
                        "training_seed_protocol", "baseline_payload_sha256",
                        "baseline_validation_sha256", "episodes",
                        "evaluation_episodes", "noise_hypothesis_id",
                        "dtl_transfer_mode", "dtl_transfer_schedule_id",
                        "wfb_hypothesis_id",
                        "ddpg_comparator_csi_hypothesis_id",
                        "los_probability_hypothesis_id",
                        "channel_coherence_hypothesis_id",
                        "csi_observation_hypothesis_id",
                        "rate_model_hypothesis_id",
                        "cross_function_interference_hypothesis_id",
                    )
                },
                "seeds": summary["seeds"],
            }), encoding="utf-8")
            result = plot_holdouts([
                "--summary", str(summary_path),
                "--reference", str(
                    Path(__file__).parents[1]
                    / "references"
                    / "paper_reference_curves.json"
                ),
                "--output", str(output),
            ])
            self.assertEqual(result, 0)
            evidence = validate_result_evidence_bundle(output)
            self.assertTrue(evidence["overall_pass"], evidence["failures"])
            self.assertEqual(len(evidence["images"]), 7)

    def test_file_verification_rejects_manifest_summary_digest_mismatch(self):
        summary = self._summary()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "kind": "run_manifest",
                "status": "complete",
                "summary": "holdout_summary.json",
                "summary_payload_sha256": "0" * 64,
            }), encoding="utf-8")
            report = validate_holdout_result(
                summary,
                self.references,
                result_directory=root,
                verify_files=True,
            )
        self.assertFalse(report["checks"]["file_provenance"]["pass"])


if __name__ == "__main__":
    unittest.main()
