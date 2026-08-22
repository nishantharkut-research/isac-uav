import hashlib
import json
import unittest
from pathlib import Path


class DeliveryManifestTests(unittest.TestCase):
    def test_manifest_hashes_and_source_only_exclusions(self):
        root = Path(__file__).parents[1]
        manifest_path = root / "DELIVERY_MANIFEST.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["full_training_performed_during_preparation"])
        paths = {item["path"] for item in manifest["files"]}
        self.assertIn("REPRODUCTION_GUIDE.md", paths)
        self.assertIn("E2E_CORRECTION_REPORT.md", paths)
        self.assertIn(
            "Hybrid-RIS_Empowered_UAV-Assisted_ISAC_Systems_Transfer_Learning-Based_DRL.pdf",
            paths,
        )
        self.assertIn("references/problem.png", paths)
        for required in (
            "pareto_agent.py",
            "pareto_training.py",
            "run_pareto_extension.py",
            "validate_pareto_extension.py",
            "plot_pareto_extension.py",
            "run_pareto_study.py",
            "profiles/pareto_robustness_study.example.json",
        ):
            self.assertIn(required, paths)
        requirements = (root / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("scipy", requirements.lower())
        for item in manifest["files"]:
            path = root / item["path"]
            self.assertTrue(path.is_file(), item["path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, item["sha256"])
            self.assertNotIn("result/", item["path"])
            self.assertNotIn("results/", item["path"])
            self.assertNotIn("runs/", item["path"])
            self.assertNotIn("checkpoints/", item["path"])
            self.assertFalse(item["path"].endswith((".pt", ".pth", ".pyc")))


if __name__ == "__main__":
    unittest.main()
