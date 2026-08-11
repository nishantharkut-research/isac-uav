import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from result_evidence import validate_pareto_evidence_bundle


class ParetoEvidenceTests(unittest.TestCase):
    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _bundle(self, root):
        artifacts = []
        for artifact_type in (
            "training_convergence", "primary_paired_delta", "pareto_front"
        ):
            image = root / f"{artifact_type}.png"
            evidence = root / f"{artifact_type}.json"
            image.write_bytes(b"png-" + artifact_type.encode("ascii"))
            evidence.write_text(json.dumps({
                "schema_version": 1,
                "kind": "ba_pcdtl_generated_evidence",
                "artifact_type": artifact_type,
                "algorithm_id": "ba_pcdtl_v1",
                "source_tree_digest": "a" * 64,
                "profile_digest": "b" * 64,
                "physical_configuration_fingerprint": "c" * 64,
                "baseline_payload_sha256": "d" * 64,
                "baseline_checkpoint_authority": [{
                    "seed": seed,
                    "source_checkpoint_sha256": "1" * 64,
                    "dtl_pcsi_checkpoint_sha256": "2" * 64,
                } for seed in range(10)],
                "gate_selection_lock_sha256": "e" * 64,
                "research_seeds": list(range(10)),
                "seed_roles": {
                    "research": list(range(10)),
                    "validation": list(range(100, 105)),
                    "test": list(range(200, 210)),
                },
                "preference_grid": [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                "anchors": {"ideal": [10.0, 10.0], "nadir": [0.0, 0.0]},
                "units": {"x": "index", "y": "bits/s/Hz"},
                "arrays": {"x": [0.0, 1.0], "y": [1.0, 2.0]},
                "acceptance_uses_raw_arrays": True,
            }), encoding="utf-8")
            artifacts.append({
                "artifact_type": artifact_type,
                "image": image.name,
                "evidence": evidence.name,
                "image_sha256": self._sha(image),
                "evidence_sha256": self._sha(evidence),
            })
        (root / "result_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "kind": "ba_pcdtl_result_manifest",
            "artifacts": artifacts,
        }), encoding="utf-8")

    def test_complete_hashed_raw_bundle_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            report = validate_pareto_evidence_bundle(root)
            self.assertTrue(report["overall_pass"], report["failures"])

    def test_tampering_mixed_provenance_and_missing_artifact_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root)
            manifest_path = root / "result_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"][0]["image_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertFalse(validate_pareto_evidence_bundle(root)["overall_pass"])

            self._bundle(root)
            evidence = root / "pareto_front.json"
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            payload["baseline_payload_sha256"] = "f" * 64
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"][2]["evidence_sha256"] = self._sha(evidence)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = validate_pareto_evidence_bundle(root)
            self.assertFalse(report["overall_pass"])
            self.assertIn("mixed provenance", " ".join(report["failures"]))


if __name__ == "__main__":
    unittest.main()
