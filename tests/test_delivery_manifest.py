import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from build_delivery import _delivery_bytes, _sha256, build_delivery


class DeliveryManifestTests(unittest.TestCase):
    def test_archive_payloads_match_manifest_content_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "portable-source"
            root.mkdir()
            source = root / "module.py"
            source.write_bytes(b"first\r\nsecond\r\n")
            binary = root / "asset.png"
            binary.write_bytes(b"binary\r\ncontent")
            manifest_path = root / "DELIVERY_MANIFEST.json"
            archive_path = root.parent / "portable-source.zip"

            with patch.multiple(
                "build_delivery",
                ROOT=root,
                MANIFEST_PATH=manifest_path,
                ARCHIVE_PATH=archive_path,
            ):
                _, built_archive = build_delivery()

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = {item["path"]: item for item in manifest["files"]}
            with ZipFile(built_archive) as archive:
                for relative_path, item in entries.items():
                    payload = archive.read(f"portable-source/{relative_path}")
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
                    self.assertEqual(len(payload), item["bytes"])
                self.assertEqual(
                    archive.read("portable-source/module.py"),
                    b"first\nsecond\n",
                )
                self.assertEqual(
                    archive.read("portable-source/asset.png"),
                    b"binary\r\ncontent",
                )

    def test_manifest_hash_contract_is_line_ending_stable_for_text_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lf_text = root / "lf.py"
            crlf_text = root / "crlf.py"
            lf_text.write_bytes(b"first\nsecond\n")
            crlf_text.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(_sha256(lf_text), _sha256(crlf_text))

            lf_binary = root / "lf.png"
            crlf_binary = root / "crlf.png"
            lf_binary.write_bytes(b"first\nsecond\n")
            crlf_binary.write_bytes(b"first\r\nsecond\r\n")
            self.assertNotEqual(_sha256(lf_binary), _sha256(crlf_binary))

    def test_manifest_hashes_and_source_only_exclusions(self):
        root = Path(__file__).parents[1]
        manifest_path = root / "DELIVERY_MANIFEST.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(
            manifest["content_hash_contract"]["text_content"],
            "crlf_and_cr_normalized_to_lf",
        )
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
            self.assertEqual(_sha256(path), item["sha256"])
            self.assertEqual(len(_delivery_bytes(path)), item["bytes"])
            self.assertNotIn("result/", item["path"])
            self.assertNotIn("results/", item["path"])
            self.assertNotIn("runs/", item["path"])
            self.assertNotIn("checkpoints/", item["path"])
            self.assertFalse(item["path"].endswith((".pt", ".pth", ".pyc")))


if __name__ == "__main__":
    unittest.main()
