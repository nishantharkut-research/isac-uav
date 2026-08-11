import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from validate_pareto_extension import (
    _validate_file_provenance,
    _validate_run_manifest_payload,
    _verify_checkpoint_record,
)
from validate_results import result_payload_digest


def _write_json(path: Path, payload) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_file_summary(root: Path, *, anchor_payload=None) -> dict:
    anchors = {
        "schema_version": 1,
        "ideal": [10.0, 10.0],
        "nadir": [0.0, 0.0],
        "frozen": True,
        "provenance": {"partition": "training_only"},
    }
    lock = {
        "schema_version": 1,
        "anchors": anchors if anchor_payload is None else anchor_payload,
        "normalized_hypervolume_reference": [-1.0, -1.0],
    }
    anchors_digest = _write_json(root / "pareto_anchors_lock.json", lock)
    evidence_digest = _write_json(root / "gate_validation_evidence.json", {})
    gate = {
        "schema_version": 1,
        "selection_partition": "validation_only",
        "validation_evidence_sha256": evidence_digest,
        "selected": {},
        "candidates": [],
    }
    gate_digest = _write_json(root / "gate_selection_lock.json", gate)
    return {
        "protocol": "ba_pcdtl_research_extension",
        "algorithm_id": "ba_pcdtl_v1",
        "comparison_level": "primary",
        "profile_digest": "a" * 64,
        "physical_configuration_fingerprint": "b" * 64,
        "source_tree_digest": "c" * 64,
        "objective": {
            "mode": "png_joint_sum",
            "user_rate_min": 0.5,
            "target_rate_min": 0.5,
        },
        "research_seeds": [],
        "validation_scene_seeds": [],
        "test_scene_seeds": [],
        "preference_grid": [[0.5, 0.5]],
        "hypervolume_reference_normalized": [-1.0, -1.0],
        "anchors": anchors,
        "anchors_lock_sha256": anchors_digest,
        "gate_selection": gate,
        "gate_selection_lock_sha256": gate_digest,
        "gate_validation_evidence_sha256": evidence_digest,
        "research_seed_records": [],
    }


def _checkpoint_record(root: Path, name: str) -> dict:
    path = root / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode("ascii"))
    return {
        "checkpoint": str(path.relative_to(root)),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _add_full_checkpoint_grid(root: Path, summary: dict) -> None:
    objective = summary["objective"]
    scalar = {
        name: _checkpoint_record(root, f"{name}.pt")
        for name in (
            "same_objective_dtl",
            "cross_objective_dtl_diagnostic",
            "ddpg_scratch",
        )
    }
    source = _checkpoint_record(root, "joint_source.pt")
    def vector_record(name, *, preference=None):
        record = _checkpoint_record(root, f"{name}.pt")
        record.update({
            "baseline_policy_digest": "d" * 64,
            "identity": {
                "baseline_policy_digest": "d" * 64,
                "source_tree_digest": summary["source_tree_digest"],
                "objective": objective,
                "fixed_preference": preference,
            },
        })
        if preference is not None:
            record["preference"] = preference
        return record

    ba = vector_record("ba_pcdtl")
    vector = {
        "ba_pcdtl": ba,
        "unanchored_vector_td3": vector_record("unanchored_vector_td3"),
        "anchored_random_trunks": vector_record("anchored_random_trunks"),
        "independent_weighted_policies": {
            "policies": [
                vector_record("independent_balanced", preference=[0.5, 0.5])
            ]
        },
    }
    scalar_hashes = [
        scalar[name]["checkpoint_sha256"]
        for name in (
            "same_objective_dtl",
            "cross_objective_dtl_diagnostic",
            "ddpg_scratch",
        )
    ]
    summary["anchors"]["provenance"][
        "source_checkpoint_sha256"
    ] = scalar_hashes
    anchor_lock = {
        "schema_version": 1,
        "anchors": summary["anchors"],
        "normalized_hypervolume_reference": summary[
            "hypervolume_reference_normalized"
        ],
    }
    summary["anchors_lock_sha256"] = _write_json(
        root / "pareto_anchors_lock.json", anchor_lock
    )
    summary["comparison_level"] = "full"
    summary["research_seed_records"] = [{
        "seed": 0,
        "training": {
            "scalar": {
                "joint_source_checkpoint": source["checkpoint"],
                "joint_source_checkpoint_sha256": source["checkpoint_sha256"],
                "methods": scalar,
            },
            "vector": {"methods": vector},
        },
    }]


class ParetoFileProvenanceTests(unittest.TestCase):
    def test_completed_manifest_must_bind_gate_selection_lock(self):
        summary = {
            "schema_version": 1,
            "kind": "generated_result",
            "gate_selection_lock_sha256": "a" * 64,
            "research_seeds": [],
        }
        manifest = {
            "schema_version": 1,
            "kind": "run_manifest",
            "status": "complete",
            "summary": "pareto_extension_summary.json",
            "summary_payload_sha256": result_payload_digest(summary),
            "gate_selection_lock_sha256": "b" * 64,
            "seeds": [],
        }
        with self.assertRaisesRegex(ValueError, "gate_selection_lock_sha256"):
            _validate_run_manifest_payload(summary, manifest)

    def test_checkpoint_must_be_inside_run_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "run"
            (root / "checkpoints").mkdir(parents=True)
            external = parent / "external.pt"
            external.write_bytes(b"external checkpoint")
            record = {
                "checkpoint": str(external.resolve()),
                "checkpoint_sha256": hashlib.sha256(
                    external.read_bytes()
                ).hexdigest(),
            }
            self.assertFalse(_verify_checkpoint_record(root, record))

    def test_anchor_lock_payload_must_match_reported_anchors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mismatched = {
                "schema_version": 1,
                "ideal": [20.0, 20.0],
                "nadir": [0.0, 0.0],
                "frozen": True,
                "provenance": {"partition": "training_only"},
            }
            summary = _minimal_file_summary(root, anchor_payload=mismatched)
            _write_json(root / "run_manifest.json", {
                "schema_version": 1,
                "kind": "run_manifest",
                "status": "complete",
                "protocol": summary["protocol"],
                "algorithm_id": summary["algorithm_id"],
                "summary": "pareto_extension_summary.json",
            })
            with patch(
                "validate_pareto_extension._validate_gate_evidence_payload"
            ):
                passed, error = _validate_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("Anchor lock payload", error)

    def test_completed_run_manifest_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _minimal_file_summary(root)
            with patch(
                "validate_pareto_extension._validate_gate_evidence_payload"
            ):
                passed, error = _validate_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("manifest", error.lower())

    def test_every_full_comparison_checkpoint_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _minimal_file_summary(root)
            _add_full_checkpoint_grid(root, summary)
            _write_json(root / "run_manifest.json", {})
            summary["research_seed_records"][0]["training"]["vector"][
                "methods"
            ]["anchored_random_trunks"]["checkpoint_sha256"] = "0" * 64
            with (
                patch("validate_pareto_extension._validate_gate_evidence_payload"),
                patch("validate_pareto_extension._validate_run_manifest_payload"),
            ):
                passed, error = _validate_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("checkpoint", error.lower())

    def test_joint_objective_source_checkpoint_digest_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _minimal_file_summary(root)
            _add_full_checkpoint_grid(root, summary)
            _write_json(root / "run_manifest.json", {})
            del summary["research_seed_records"][0]["training"]["scalar"][
                "joint_source_checkpoint_sha256"
            ]
            with (
                patch("validate_pareto_extension._validate_gate_evidence_payload"),
                patch("validate_pareto_extension._validate_run_manifest_payload"),
            ):
                passed, error = _validate_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("source", error.lower())

    def test_complete_full_checkpoint_grid_passes_file_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _minimal_file_summary(root)
            _add_full_checkpoint_grid(root, summary)
            _write_json(root / "run_manifest.json", {})
            with (
                patch("validate_pareto_extension._validate_gate_evidence_payload"),
                patch("validate_pareto_extension._validate_run_manifest_payload"),
            ):
                passed, error = _validate_file_provenance(summary, root)
            self.assertTrue(passed, error)


if __name__ == "__main__":
    unittest.main()
