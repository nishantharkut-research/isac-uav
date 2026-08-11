import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from validate_results import (
    METHOD_ORDER,
    _validate_baseline_file_provenance,
    result_payload_digest,
)


def _checkpoint(root: Path, name: str) -> dict:
    path = root / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode("ascii"))
    return {
        "checkpoint": str(path.relative_to(root)),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _bundle(root: Path):
    source = _checkpoint(root, "source.pt")
    methods = {
        method: _checkpoint(root, f"{method}.pt") for method in METHOD_ORDER
    }
    summary = {
        "schema_version": 1,
        "kind": "generated_result",
        "protocol": "paper_baseline",
        "profile_digest": "a" * 64,
        "physical_configuration_fingerprint": "b" * 64,
        "source_tree_digest": "c" * 64,
        "training_seed_protocol": "seed-v1",
        "power_hypothesis_id": "paper_db_as_dbm",
        "objective": {"mode": "paper_baseline", "target_rate_min": 0.5},
        "seeds": [1],
        "episodes": 1500,
        "evaluation_episodes": 5,
        "seed_records": [{
            "seed": 1,
            "source_checkpoint": source["checkpoint"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "methods": methods,
        }],
    }
    manifest = {
        "schema_version": 1,
        "kind": "run_manifest",
        "status": "complete",
        "summary": "baseline_summary.json",
        "summary_payload_sha256": result_payload_digest(summary),
        **{
            field: summary[field]
            for field in (
                "protocol",
                "profile_digest",
                "physical_configuration_fingerprint",
                "source_tree_digest",
                "training_seed_protocol",
                "power_hypothesis_id",
                "objective",
                "seeds",
                "episodes",
                "evaluation_episodes",
            )
        },
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return summary


class BaselineFileProvenanceTests(unittest.TestCase):
    def test_complete_baseline_bundle_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            passed, error = _validate_baseline_file_provenance(
                _bundle(root), root
            )
            self.assertTrue(passed, error)

    def test_tampered_method_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _bundle(root)
            summary["seed_records"][0]["methods"]["DTL-pCSI"][
                "checkpoint_sha256"
            ] = "0" * 64
            passed, error = _validate_baseline_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("checkpoint", error.lower())

    def test_external_checkpoint_path_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "run"
            root.mkdir()
            summary = _bundle(root)
            external = parent / "external.pt"
            external.write_bytes(b"external")
            summary["seed_records"][0]["source_checkpoint"] = str(
                external.resolve()
            )
            summary["seed_records"][0]["source_checkpoint_sha256"] = (
                hashlib.sha256(external.read_bytes()).hexdigest()
            )
            passed, error = _validate_baseline_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("source", error.lower())


if __name__ == "__main__":
    unittest.main()
