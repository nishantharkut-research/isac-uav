import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from result_evidence import validate_result_evidence_bundle


class ResultEvidenceTests(unittest.TestCase):
    @staticmethod
    def _sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_png_without_manifest_and_raw_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fig4_rewards_vs_episodes.png").write_bytes(b"png")
            report = validate_result_evidence_bundle(root)
            self.assertFalse(report["overall_pass"])
            self.assertIn("result_manifest.json", " ".join(report["failures"]))

    def test_complete_png_evidence_bundle_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "fig4_rewards_vs_episodes.png"
            evidence = root / "fig4_data.json"
            image.write_bytes(b"png")
            evidence.write_text(json.dumps({
                "schema_version": 1,
                "kind": "generated_result_evidence",
                "figure": 4,
                "source_tree_digest": "a" * 64,
                "profile_digest": "profile",
                "physical_configuration_fingerprint": "physics",
                "power_hypothesis_id": "paper_db_as_dbm",
                "noise_hypothesis_id": "description_literal_target_matches_ue",
                "dtl_transfer_mode": "eq30_current_q_bellman_blend",
                "dtl_transfer_schedule_id": "constant_upsilon",
                "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
                "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
                "seeds": [1, 2, 3, 4, 5],
                "units": {"x": "episodes", "reward": "bits/s/Hz"},
                "arrays": {"x": [0, 1], "DTL-pCSI": [1.0, 2.0]},
                "feasibility_rate": 1.0,
            }), encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "kind": "generated_result_manifest",
                "artifacts": [{
                    "image": image.name,
                    "evidence": evidence.name,
                    "image_sha256": self._sha256(image),
                    "evidence_sha256": self._sha256(evidence),
                }],
            }
            (root / "result_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            report = validate_result_evidence_bundle(root)
            self.assertTrue(report["overall_pass"], report["failures"])

    def test_missing_or_unknown_ambiguity_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "fig4.png"
            evidence = root / "fig4_data.json"
            image.write_bytes(b"png")
            evidence.write_text(json.dumps({
                "schema_version": 1,
                "kind": "generated_result_evidence",
                "figure": 4,
                "source_tree_digest": "a" * 64,
                "profile_digest": "profile",
                "physical_configuration_fingerprint": "physics",
                "power_hypothesis_id": "paper_db_as_dbm",
                "noise_hypothesis_id": "best_result",
                "dtl_transfer_mode": "eq30_current_q_bellman_blend",
                "dtl_transfer_schedule_id": "constant_upsilon",
                "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
                "seeds": [1, 2, 3, 4, 5],
                "units": {"x": "episodes"},
                "arrays": {"x": [0]},
                "feasibility_rate": 1.0,
            }), encoding="utf-8")
            (root / "result_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "kind": "generated_result_manifest",
                "artifacts": [{
                    "image": image.name,
                    "evidence": evidence.name,
                    "image_sha256": self._sha256(image),
                    "evidence_sha256": self._sha256(evidence),
                }],
            }), encoding="utf-8")
            report = validate_result_evidence_bundle(root)
            self.assertFalse(report["overall_pass"])
            self.assertIn("ambiguity", " ".join(report["failures"]))

    def test_manifest_hash_or_seed_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "fig4.png"
            evidence = root / "fig4_data.json"
            image.write_bytes(b"png")
            evidence.write_text(json.dumps({
                "schema_version": 1,
                "kind": "generated_result_evidence",
                "figure": 4,
                "source_tree_digest": "a" * 64,
                "profile_digest": "profile",
                "physical_configuration_fingerprint": "physics",
                "power_hypothesis_id": "paper_db_as_dbm",
                "noise_hypothesis_id": "description_literal_target_matches_ue",
                "dtl_transfer_mode": "eq30_current_q_bellman_blend",
                "dtl_transfer_schedule_id": "constant_upsilon",
                "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
                "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
                "seeds": [1, 2, 3],
                "units": {"x": "episodes"},
                "arrays": {"x": [0]},
                "feasibility_rate": 1.0,
            }), encoding="utf-8")
            (root / "result_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "kind": "generated_result_manifest",
                "artifacts": [{
                    "image": image.name,
                    "evidence": evidence.name,
                    "image_sha256": "0" * 64,
                    "evidence_sha256": self._sha256(evidence),
                }],
            }), encoding="utf-8")
            report = validate_result_evidence_bundle(root)
            self.assertFalse(report["overall_pass"])

    def test_bundle_rejects_mixed_valid_hypothesis_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = []
            for figure, wfb in (
                (5, "adaptive_matched_filter_no_capon"),
                (6, "channel_independent_dft_no_capon"),
            ):
                image = root / f"fig{figure}.png"
                evidence = root / f"fig{figure}_data.json"
                image.write_bytes(b"png")
                evidence.write_text(json.dumps({
                    "schema_version": 1,
                    "kind": "generated_result_evidence",
                    "figure": figure,
                    "source_tree_digest": "a" * 64,
                    "profile_digest": "profile",
                    "physical_configuration_fingerprint": "physics",
                    "power_hypothesis_id": "paper_db_as_dbm",
                    "noise_hypothesis_id":
                        "description_literal_target_matches_ue",
                    "dtl_transfer_mode": "eq30_current_q_bellman_blend",
                    "dtl_transfer_schedule_id": "constant_upsilon",
                    "wfb_hypothesis_id": wfb,
                    "ddpg_comparator_csi_hypothesis_id":
                        "fair_pcsi_comparator",
                    "seeds": [1, 2, 3, 4, 5],
                    "units": {"x": "episode"},
                    "arrays": {"x": [0]},
                    "feasibility_rate": 1.0,
                }), encoding="utf-8")
                artifacts.append({
                    "image": image.name,
                    "evidence": evidence.name,
                    "image_sha256": self._sha256(image),
                    "evidence_sha256": self._sha256(evidence),
                })
            (root / "result_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "kind": "generated_result_manifest",
                "artifacts": artifacts,
            }), encoding="utf-8")
            report = validate_result_evidence_bundle(root)
            self.assertFalse(report["overall_pass"])
            self.assertIn("mixed provenance", " ".join(report["failures"]))


if __name__ == "__main__":
    unittest.main()
