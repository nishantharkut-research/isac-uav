import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from source_integrity import source_model_state_digest
from validate_pareto_extension import (
    _validate_file_provenance,
    _validate_run_manifest_payload,
    _verify_checkpoint_record,
)
from validate_results import result_payload_digest
from pareto_protocol import ParetoAnchors
from run_pareto_extension import _derive_research_seed, _preference_suffix
from ambiguity_hypotheses import (
    AMBIGUITY_IDENTITY_REGISTRIES,
    apply_ambiguity_manifest_fields,
)
from config import cfg
from experiment_protocol import (
    REPLAY_ACTION_SEMANTICS,
    TRAINING_SEED_PROTOCOL,
    derive_training_seed,
)


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
        "anchor_digest": ParetoAnchors.from_state_dict(
            anchors if anchor_payload is None else anchor_payload
        ).digest(),
        "hypervolume_reference_rule": {
            "physical_point_bits_per_second_per_hz": [0.0, 0.0],
            "normalized_margin": 1.0,
            "lock_partition": "training_only_before_validation",
        },
        "normalized_hypervolume_reference": [-1.0, -1.0],
    }
    anchors_digest = _write_json(root / "pareto_anchors_lock.json", lock)
    evidence_digest = _write_json(root / "gate_validation_evidence.json", {})
    gate = {
        "schema_version": 1,
        "selection_partition": "validation_only",
        "gate_selection_grid": {
            "uncertainty_coefficients": [0.0],
            "margins": [0.0],
            "selection_partition": "validation_only",
            "tie_break_order": [
                "feasibility", "png_joint_sum", "pareto_hypervolume",
                "lower_fallback_rate", "lexicographic_parameters",
            ],
        },
        "hypervolume_reference_normalized": [-1.0, -1.0],
        "validation_evidence_sha256": evidence_digest,
        "selected": {},
        "candidates": [],
    }
    gate_digest = _write_json(root / "gate_selection_lock.json", gate)
    summary = {
        "protocol": "ba_pcdtl_research_extension",
        "algorithm_id": "ba_pcdtl_v1",
        "comparison_level": "primary",
        "profile_digest": "a" * 64,
        "physical_configuration_fingerprint": "b" * 64,
        "source_tree_digest": "c" * 64,
        "training_seed_protocol": TRAINING_SEED_PROTOCOL,
        "power_hypothesis_id": "paper_db_as_dbm",
        "baseline_payload_sha256": "e" * 64,
        "baseline_validation_sha256": "f" * 64,
        "objective": {
            "mode": "png_joint_sum",
            "user_rate_min": 0.5,
            "target_rate_min": 0.5,
        },
        "episodes": 10,
        "ensemble_size": 3,
        "residual_radius": 0.25,
        "residual_scales": [0.0, 0.5, 1.0],
        "research_seeds": [],
        "validation_scene_seeds": [],
        "test_scene_seeds": [],
        "preference_grid": [[0.5, 0.5]],
        "gate_selection_grid": gate["gate_selection_grid"],
        "hypervolume_reference_rule": lock["hypervolume_reference_rule"],
        "hypervolume_reference_normalized": [-1.0, -1.0],
        "anchors": anchors,
        "anchors_lock_sha256": anchors_digest,
        "gate_selection": gate,
        "gate_selection_lock_sha256": gate_digest,
        "gate_validation_evidence_sha256": evidence_digest,
        "research_seed_records": [],
    }
    return apply_ambiguity_manifest_fields(summary, cfg)


def _checkpoint_record(root: Path, name: str) -> dict:
    path = root / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode("ascii"))
    return {
        "checkpoint": str(path.relative_to(root)),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _complete_checkpoint_record(
        root: Path, record: dict, identity: dict, histories: dict,
        *, nested_training: bool, agent_state=None,
) -> dict:
    checkpoint = root / record["checkpoint"]
    completed_episode = int(identity["requested_episode_target"])
    torch.save({
        "schema_version": 1,
        "completed_episode": completed_episode,
        "identity": identity,
        "identity_canonical": json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ),
        "agent": dict(agent_state or {}),
        "environment": {},
        "histories": histories,
        "extra": {},
        "rng": {},
    }, checkpoint)
    record["checkpoint_sha256"] = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    identity_path = checkpoint.with_suffix(".pt.identity.json")
    identity_digest = _write_json(identity_path, {
        "schema_version": 1,
        "kind": "training_checkpoint_identity",
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "identity": identity,
    })
    training = (
        {"histories": histories, "completed_episode": completed_episode}
        if nested_training else
        {**histories, "elapsed_seconds": 0.0}
    )
    record.update({
        "identity": identity,
        "checkpoint_identity": str(identity_path.relative_to(root)),
        "checkpoint_identity_sha256": identity_digest,
        "training": training,
    })
    return record


def _add_full_checkpoint_grid(root: Path, summary: dict) -> None:
    objective = summary["objective"]
    serialized_source = {
        "actor": {"weight": torch.tensor([1.0])},
        "critic": {"weight": torch.tensor([2.0])},
        "actor_tgt": {"weight": torch.tensor([3.0])},
        "critic_tgt": {"weight": torch.tensor([4.0])},
    }
    serialized_source_digest = source_model_state_digest(serialized_source)
    summary["baseline_checkpoint_authority"] = [{
        "seed": 0,
        "source_model_digest": serialized_source_digest,
    }]

    def scalar_record(name, run_id, algorithm, seed_role, state_dim):
        record = _checkpoint_record(root, f"{name}.pt")
        identity = {
            "run_id": run_id,
            "algorithm": algorithm,
            "configuration_fingerprint": "9" * 20,
            "parameter_profile_digest": summary["profile_digest"],
            "objective": objective,
            "requested_episode_target": summary["episodes"],
            "power_hypothesis_id": summary["power_hypothesis_id"],
            "source_tree_digest": summary["source_tree_digest"],
            "source_model_digest": (
                "not_applicable"
                if algorithm == "DDPG" else serialized_source_digest
            ),
            "environment_seed": 0 if seed_role == "source" else 100_000,
            "training_seed": derive_training_seed(0, seed_role),
            "training_seed_protocol": summary["training_seed_protocol"],
            "replay_action_semantics": REPLAY_ACTION_SEMANTICS,
            "state_dim": state_dim,
            "action_dim": 10,
            "dtl_transfer_mode": (
                summary["dtl_transfer_mode"]
                if algorithm == "DTL" else "not_applicable"
            ),
            "dtl_transfer_schedule_id": (
                summary["dtl_transfer_schedule_id"]
                if algorithm == "DTL" else "not_applicable"
            ),
            **{
                field: summary[field]
                for field in AMBIGUITY_IDENTITY_REGISTRIES
                if field not in {
                    "dtl_transfer_mode", "dtl_transfer_schedule_id"
                }
            },
        }
        histories = {
            key: [0.0] * summary["episodes"]
            for key in (
                "rewards",
                "comm_rates",
                "sensing_rates",
                "joint_rates",
                "feasibility_rates",
            )
        }
        return _complete_checkpoint_record(
            root,
            record,
            identity,
            histories,
            nested_training=False,
            agent_state=(
                {"source_agent": serialized_source}
                if algorithm == "DTL" else serialized_source
            ),
        )

    source = scalar_record(
        "joint_source", "joint-source-seed-0", "DDPG", "source", 20
    )
    scalar = {
        "same_objective_dtl": scalar_record(
            "same_objective_dtl", "same-objective-dtl-seed-0",
            "DTL", "target", 30,
        ),
        "cross_objective_dtl_diagnostic": scalar_record(
            "cross_objective_dtl_diagnostic",
            "cross-objective-dtl-seed-0", "DTL", "target", 30,
        ),
        "ddpg_scratch": scalar_record(
            "ddpg_scratch", "joint-ddpg-scratch-seed-0",
            "DDPG", "target", 30,
        ),
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
    anchor_digest = ParetoAnchors.from_state_dict(summary["anchors"]).digest()

    def vector_record(
            name, *, preference=None, residual_radius=None,
            transfer_trunks=True, anchor_enabled=True,
    ):
        record = _checkpoint_record(root, f"{name}.pt")
        seed = 0
        seed_role = "independent_weighted" if preference is not None else name
        training_seed = _derive_research_seed(seed, seed_role)
        environment_seed = 10_000
        radius = (
            summary["residual_radius"]
            if residual_radius is None else residual_radius
        )
        identity = {
            "run_id": f"{name}-seed-{seed}",
            "algorithm": summary["algorithm_id"],
            "method": name,
            "source_tree_digest": summary["source_tree_digest"],
            "profile_digest": summary["profile_digest"],
            "physical_configuration_fingerprint": summary[
                "physical_configuration_fingerprint"
            ],
            "objective": objective,
            "requested_episode_target": summary["episodes"],
            "environment_seed": environment_seed,
            "training_seed": training_seed,
            "baseline_policy_digest": "d" * 64,
            "preference_grid": summary["preference_grid"],
            "fixed_preference": preference,
            "residual_radius": radius,
            "ensemble_size": summary["ensemble_size"],
            "residual_scales": summary["residual_scales"],
            "transfer_trunks": transfer_trunks,
            "anchor_enabled": anchor_enabled,
            "anchor_digest": anchor_digest,
            "replay_action_semantics": REPLAY_ACTION_SEMANTICS,
            "training_seed_protocol": summary["training_seed_protocol"],
            "power_hypothesis_id": summary["power_hypothesis_id"],
            "baseline_payload_sha256": summary["baseline_payload_sha256"],
            "baseline_validation_sha256": summary[
                "baseline_validation_sha256"
            ],
            "ambiguity_contract": summary["ambiguity_contract"],
            **{
                field: summary[field]
                for field in AMBIGUITY_IDENTITY_REGISTRIES
            },
        }
        histories = {
            key: [0.0] * summary["episodes"]
            for key in (
                "communication",
                "sensing",
                "joint",
                "objective",
                "feasibility",
                "mean_residual_scale",
                "baseline_fallback_rate",
            )
        }
        _complete_checkpoint_record(
            root, record, identity, histories, nested_training=True
        )
        record.update({
            "baseline_policy_digest": "d" * 64,
            "training_seed": training_seed,
            "environment_seed": environment_seed,
        })
        if preference is not None:
            record["preference"] = preference
        return record

    ba = vector_record("ba_pcdtl")
    vector = {
        "ba_pcdtl": ba,
        "unanchored_vector_td3": vector_record(
            "unanchored_vector_td3",
            residual_radius=1.0,
            transfer_trunks=False,
            anchor_enabled=False,
        ),
        "anchored_random_trunks": vector_record(
            "anchored_random_trunks", transfer_trunks=False
        ),
        "independent_weighted_policies": {
            "policies": [
                vector_record(
                    f"independent_{_preference_suffix([0.5, 0.5])}",
                    preference=[0.5, 0.5],
                )
            ]
        },
    }
    anchor_lock = {
        "schema_version": 1,
        "anchors": summary["anchors"],
        "anchor_digest": anchor_digest,
        "hypervolume_reference_rule": summary[
            "hypervolume_reference_rule"
        ],
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
                "joint_source_checkpoint_identity": source[
                    "checkpoint_identity"
                ],
                "joint_source_checkpoint_identity_sha256": source[
                    "checkpoint_identity_sha256"
                ],
                "joint_source_identity": source["identity"],
                "joint_source_training": source["training"],
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

    def test_anchor_lock_rejects_incorrect_internal_anchor_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _minimal_file_summary(root)
            lock_path = root / "pareto_anchors_lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["anchor_digest"] = "0" * 64
            summary["anchors_lock_sha256"] = _write_json(lock_path, lock)
            _write_json(root / "run_manifest.json", {})
            with (
                patch("validate_pareto_extension._validate_gate_evidence_payload"),
                patch("validate_pareto_extension._validate_run_manifest_payload"),
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

    def test_vector_checkpoint_rejects_incomplete_or_changed_training_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _minimal_file_summary(root)
            _add_full_checkpoint_grid(root, summary)
            _write_json(root / "run_manifest.json", {})
            summary["research_seed_records"][0]["training"]["vector"][
                "methods"
            ]["anchored_random_trunks"]["identity"]["method"] = "ba_pcdtl"
            with (
                patch("validate_pareto_extension._validate_gate_evidence_payload"),
                patch("validate_pareto_extension._validate_run_manifest_payload"),
            ):
                passed, error = _validate_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("checkpoint authority", error.lower())


if __name__ == "__main__":
    unittest.main()
