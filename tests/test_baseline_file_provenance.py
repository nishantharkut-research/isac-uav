import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from source_integrity import source_model_state_digest
from validate_results import (
    METHOD_ORDER,
    _validate_baseline_file_provenance,
    result_payload_digest,
)
from experiment_protocol import REPLAY_ACTION_SEMANTICS


def _checkpoint(root: Path, name: str) -> dict:
    path = root / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode("ascii"))
    return {
        "checkpoint": str(path.relative_to(root)),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _network_state(marker: float) -> dict:
    return {"weight": torch.tensor([marker], dtype=torch.float32)}


def _serialized_source_agent() -> dict:
    return {
        "actor": _network_state(1.0),
        "critic": _network_state(2.0),
        "actor_tgt": _network_state(3.0),
        "critic_tgt": _network_state(4.0),
    }


def _attach_identity(
        root: Path, record: dict, identity: dict, *, agent_state=None,
) -> dict:
    checkpoint = root / record["checkpoint"]
    episode_count = int(identity["requested_episode_target"])
    histories = {
        key: [0.0] * episode_count
        for key in (
            "rewards",
            "comm_rates",
            "sensing_rates",
            "joint_rates",
            "feasibility_rates",
        )
    }
    torch.save({
        "schema_version": 1,
        "completed_episode": episode_count,
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
    identity_path = checkpoint.with_suffix(checkpoint.suffix + ".identity.json")
    payload = {
        "schema_version": 1,
        "kind": "training_checkpoint_identity",
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "identity": identity,
    }
    identity_path.write_text(json.dumps(payload), encoding="utf-8")
    record.update({
        "checkpoint_identity": str(identity_path.relative_to(root)),
        "checkpoint_identity_sha256": hashlib.sha256(
            identity_path.read_bytes()
        ).hexdigest(),
        "identity": identity,
        "training": {**histories, "elapsed_seconds": 0.0},
    })
    return record


def _bundle(root: Path):
    source = _checkpoint(root, "source.pt")
    serialized_source = _serialized_source_agent()
    serialized_source_digest = source_model_state_digest(serialized_source)
    ambiguity = {
        "noise_hypothesis_id": "description_literal_target_matches_ue",
        "dtl_transfer_mode": "eq30_current_q_bellman_blend",
        "dtl_transfer_schedule_id": "constant_upsilon",
        "wfb_hypothesis_id": "adaptive_matched_filter_no_capon",
        "ddpg_comparator_csi_hypothesis_id": "fair_pcsi_comparator",
        "los_probability_hypothesis_id": "paper_unnumbered_los_literal_roles",
        "channel_coherence_hypothesis_id": "episode_block_fading",
        "csi_observation_hypothesis_id":
            "pcsi_instantaneous_icsi_statistical",
        "rate_model_hypothesis_id":
            "continuous_matrix_scee_all_csi_qualities",
        "cross_function_interference_hypothesis_id":
            "paper_eq12_eq14_literal_cross_channels",
    }
    objective = {"mode": "paper_baseline", "target_rate_min": 0.5}

    def training_identity(run_id, algorithm, environment_seed, training_seed,
                          state_dim):
        return {
            "run_id": run_id,
            "algorithm": algorithm,
            "configuration_fingerprint": "f" * 20,
            "parameter_profile_digest": "a" * 64,
            "objective": objective,
            "requested_episode_target": 1500,
            "power_hypothesis_id": "paper_db_as_dbm",
            "source_tree_digest": "c" * 64,
            "source_model_digest": (
                "not_applicable"
                if algorithm == "DDPG" else serialized_source_digest
            ),
            "environment_seed": environment_seed,
            "training_seed": training_seed,
            "training_seed_protocol": "seed-v1",
            "replay_action_semantics": REPLAY_ACTION_SEMANTICS,
            "state_dim": state_dim,
            "action_dim": 10,
            "dtl_transfer_mode": (
                ambiguity["dtl_transfer_mode"]
                if algorithm == "DTL" else "not_applicable"
            ),
            "dtl_transfer_schedule_id": (
                ambiguity["dtl_transfer_schedule_id"]
                if algorithm == "DTL" else "not_applicable"
            ),
            **{
                field: value for field, value in ambiguity.items()
                if field not in {
                    "dtl_transfer_mode", "dtl_transfer_schedule_id"
                }
            },
        }

    source_tag = "baseline-source-seed-1"
    source = _attach_identity(
        root,
        source,
        training_identity(source_tag, "DDPG", 1, 101, 20),
        agent_state=serialized_source,
    )
    methods = {}
    for method in METHOD_ORDER:
        tag = f"baseline-{method}-seed-1"
        algorithm = "DDPG" if method == "DDPG" else "DTL"
        state_dim = 20 if method == "DTL-iCSI" else 30
        methods[method] = {
            "tag": tag,
            **_attach_identity(
                root,
                _checkpoint(root, f"{method}.pt"),
                training_identity(tag, algorithm, 100_001, 202, state_dim),
                agent_state=(
                    {"source_agent": serialized_source}
                    if algorithm == "DTL" else serialized_source
                ),
            ),
        }
    sanity = {
        "schema_version": 1,
        "kind": "pretraining_physical_sanity",
        "advisory_only": True,
        "passed": True,
        "profile_digest": "a" * 64,
        "physical_configuration_fingerprint": "b" * 64,
        "source_tree_digest": "c" * 64,
        **ambiguity,
    }
    sanity_path = root / "pretraining_physical_sanity.json"
    sanity_path.write_text(json.dumps(sanity), encoding="utf-8")
    sanity_digest = hashlib.sha256(sanity_path.read_bytes()).hexdigest()
    summary = {
        "schema_version": 1,
        "kind": "generated_result",
        "protocol": "paper_baseline",
        "profile_digest": "a" * 64,
        "physical_configuration_fingerprint": "b" * 64,
        "source_tree_digest": "c" * 64,
        "training_seed_protocol": "seed-v1",
        "power_hypothesis_id": "paper_db_as_dbm",
        "objective": objective,
        "seeds": [1],
        "episodes": 1500,
        "evaluation_episodes": 5,
        "pretraining_physical_sanity_sha256": sanity_digest,
        **ambiguity,
        "seed_records": [{
            "seed": 1,
            "source_seed": 1,
            "target_seed": 100_001,
            "source_training_seed": 101,
            "target_training_seed": 202,
            "source_tag": source_tag,
            "source_checkpoint": source["checkpoint"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_checkpoint_identity": source["checkpoint_identity"],
            "source_checkpoint_identity_sha256": source[
                "checkpoint_identity_sha256"
            ],
            "source_identity": source["identity"],
            "source_training": source["training"],
            "methods": methods,
        }],
    }
    manifest = {
        "schema_version": 1,
        "kind": "run_manifest",
        "status": "complete",
        "summary": "baseline_summary.json",
        "summary_payload_sha256": result_payload_digest(summary),
        "pretraining_physical_sanity": sanity_path.name,
        "pretraining_physical_sanity_sha256": sanity_digest,
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
                *ambiguity,
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

    def test_checkpoint_history_must_match_serialized_training_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _bundle(root)
            method = summary["seed_records"][0]["methods"]["DTL-pCSI"]
            checkpoint = root / method["checkpoint"]
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            payload["histories"]["comm_rates"][0] = 1.0
            torch.save(payload, checkpoint)

            method["checkpoint_sha256"] = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            identity_path = root / method["checkpoint_identity"]
            identity_envelope = {
                "schema_version": 1,
                "kind": "training_checkpoint_identity",
                "checkpoint": method["checkpoint"],
                "checkpoint_sha256": method["checkpoint_sha256"],
                "identity": method["identity"],
            }
            identity_path.write_text(
                json.dumps(identity_envelope), encoding="utf-8"
            )
            method["checkpoint_identity_sha256"] = hashlib.sha256(
                identity_path.read_bytes()
            ).hexdigest()

            passed, error = _validate_baseline_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("identity", error.lower())

    def test_dtl_checkpoint_must_embed_the_accepted_source_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _bundle(root)
            method = summary["seed_records"][0]["methods"]["DTL-pCSI"]
            checkpoint = root / method["checkpoint"]
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            alternate_source = {
                "actor": _network_state(11.0),
                "critic": _network_state(12.0),
                "actor_tgt": _network_state(13.0),
                "critic_tgt": _network_state(14.0),
            }
            alternate_digest = source_model_state_digest(alternate_source)
            payload["agent"]["source_agent"] = alternate_source
            payload["identity"]["source_model_digest"] = alternate_digest
            payload["identity_canonical"] = json.dumps(
                payload["identity"], sort_keys=True, separators=(",", ":")
            )
            torch.save(payload, checkpoint)

            method["identity"] = payload["identity"]
            method["checkpoint_sha256"] = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            identity_path = root / method["checkpoint_identity"]
            identity_path.write_text(json.dumps({
                "schema_version": 1,
                "kind": "training_checkpoint_identity",
                "checkpoint": method["checkpoint"],
                "checkpoint_sha256": method["checkpoint_sha256"],
                "identity": method["identity"],
            }), encoding="utf-8")
            method["checkpoint_identity_sha256"] = hashlib.sha256(
                identity_path.read_bytes()
            ).hexdigest()

            passed, error = _validate_baseline_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("source-model", error.lower())

    def test_checkpoint_without_training_identity_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _bundle(root)
            summary["seed_records"][0]["methods"]["DTL-pCSI"].update({
                "checkpoint_identity": "checkpoints/DTL-pCSI.pt.identity.json",
                "checkpoint_identity_sha256": "0" * 64,
                "identity": {"algorithm": "DTL"},
            })
            manifest_path = root / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["summary_payload_sha256"] = result_payload_digest(summary)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            passed, error = _validate_baseline_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("identity", error.lower())

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

    def test_tampered_pretraining_sanity_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _bundle(root)
            path = root / "pretraining_physical_sanity.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["passed"] = False
            path.write_text(json.dumps(report), encoding="utf-8")
            passed, error = _validate_baseline_file_provenance(summary, root)
            self.assertFalse(passed)
            self.assertIn("physical-sanity", error.lower())


if __name__ == "__main__":
    unittest.main()
