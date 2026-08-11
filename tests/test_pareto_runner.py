import json
import tempfile
import unittest
from pathlib import Path

from run_pareto_extension import (
    _build_baseline_checkpoint_authority,
    create_research_manifest,
    parse_float_grid,
    parse_preference_grid,
    require_accepted_baseline,
    select_gate_configuration,
    validate_baseline_contract,
    validate_seed_roles,
    write_immutable_json,
)
from validate_results import result_payload_digest


def _seeds(start, count):
    return list(range(start, start + count))


class RunnerProtocolTests(unittest.TestCase):
    def test_checkpoint_authority_binds_source_and_dtl_pcsi_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            target = root / "target.pt"
            source.write_bytes(b"source checkpoint")
            target.write_bytes(b"target checkpoint")

            import hashlib

            source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target_digest = hashlib.sha256(target.read_bytes()).hexdigest()
            baseline = {
                "seeds": [7],
                "seed_records": [{
                    "seed": 7,
                    "source_checkpoint": source.name,
                    "source_checkpoint_sha256": source_digest,
                    "methods": {"DTL-pCSI": {
                        "checkpoint": target.name,
                        "checkpoint_sha256": target_digest,
                    }},
                }],
            }

            authority = _build_baseline_checkpoint_authority(
                baseline, root, lambda base, relative: base / relative
            )

        self.assertEqual(authority, [{
            "seed": 7,
            "source_checkpoint": source.name,
            "source_checkpoint_sha256": source_digest,
            "dtl_pcsi_checkpoint": target.name,
            "dtl_pcsi_checkpoint_sha256": target_digest,
        }])

    def test_seed_roles_are_disjoint_and_sufficient(self):
        roles = validate_seed_roles(_seeds(0, 10), _seeds(100, 5), _seeds(200, 10))
        self.assertEqual(len(roles["research"]), 10)
        for bad_roles in (
            (_seeds(0, 9), _seeds(100, 5), _seeds(200, 10)),
            (_seeds(0, 10), _seeds(100, 4), _seeds(200, 10)),
            (_seeds(0, 10), _seeds(100, 5), _seeds(200, 9)),
            (_seeds(0, 10), _seeds(5, 5), _seeds(200, 10)),
        ):
            with self.subTest(bad_roles=bad_roles):
                with self.assertRaises(ValueError):
                    validate_seed_roles(*bad_roles)

    def test_preference_and_gate_grids_are_strict(self):
        preferences = parse_preference_grid("1:0,0.5:0.5,0:1")
        self.assertEqual(preferences, [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        self.assertEqual(parse_float_grid("0,0.5,1", name="beta"), [0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "balanced"):
            parse_preference_grid("1:0,0:1")
        with self.assertRaises(ValueError):
            parse_float_grid("0,-1", name="margin")

    def test_accepted_baseline_is_bound_to_exact_payload_and_contract(self):
        baseline = {
            "kind": "generated_result",
            "protocol": "paper_baseline",
            "profile_digest": "a" * 64,
            "physical_configuration_fingerprint": "b" * 64,
            "source_tree_digest": "c" * 64,
            "training_seed_protocol": "seed-v1",
            "power_hypothesis_id": "power-v1",
            "seeds": _seeds(0, 10),
            "episodes": 1500,
            "evaluation_episodes": 5,
        }
        validation = {
            "validation": "paper_baseline",
            "overall_pass": True,
            "baseline_payload_sha256": result_payload_digest(baseline),
        }
        require_accepted_baseline(validation, baseline)
        validate_baseline_contract(
            baseline,
            profile_digest="a" * 64,
            physics_fingerprint="b" * 64,
            source_tree_digest="c" * 64,
            training_seed_protocol="seed-v1",
            power_hypothesis_id="power-v1",
            research_seeds=_seeds(0, 10),
            episodes=1500,
            evaluation_episodes=5,
        )
        validation["baseline_payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different baseline"):
            require_accepted_baseline(validation, baseline)

    def test_manifest_contains_all_immutable_research_fields(self):
        manifest = create_research_manifest(
            profile_digest="a" * 64,
            physics_fingerprint="b" * 64,
            source_tree_digest="c" * 64,
            baseline_payload_digest="d" * 64,
            baseline_validation_digest="e" * 64,
            baseline_checkpoint_authority=[{
                "seed": seed,
                "source_checkpoint": f"checkpoints/source-{seed}.pt",
                "source_checkpoint_sha256": "1" * 64,
                "dtl_pcsi_checkpoint": f"checkpoints/dtl-{seed}.pt",
                "dtl_pcsi_checkpoint_sha256": "2" * 64,
            } for seed in _seeds(0, 10)],
            seed_roles=validate_seed_roles(
                _seeds(0, 10), _seeds(100, 5), _seeds(200, 10)
            ),
            preference_grid=parse_preference_grid("1:0,0.5:0.5,0:1"),
            episodes=1500,
            evaluation_episodes=5,
            comparison_level="full",
            residual_radius=0.25,
            ensemble_size=2,
            residual_scales=[0.0, 0.5, 1.0],
            uncertainty_grid=[0.0, 1.0],
            margin_grid=[0.0, 0.05],
            objective={"mode": "png_joint_sum", "user_rate_min": 0.1,
                       "target_rate_min": 0.5},
        )
        self.assertEqual(manifest["algorithm_id"], "ba_pcdtl_v1")
        self.assertEqual(manifest["test_scene_seeds"], _seeds(200, 10))
        self.assertEqual(len(manifest["baseline_checkpoint_authority"]), 10)
        self.assertNotIn("communication_retention_fraction", manifest["objective"])

    def test_gate_selection_order_and_immutable_lock(self):
        candidates = [
            {"uncertainty_coefficient": 1.0, "margin": 0.0,
             "feasibility": 1.0, "joint": 7.0, "hypervolume": 2.0,
             "fallback_rate": 0.4},
            {"uncertainty_coefficient": 0.5, "margin": 0.0,
             "feasibility": 1.0, "joint": 7.0, "hypervolume": 2.0,
             "fallback_rate": 0.3},
        ]
        selected = select_gate_configuration(candidates)
        self.assertEqual(selected["uncertainty_coefficient"], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate_selection_lock.json"
            write_immutable_json(path, selected)
            write_immutable_json(path, selected)
            altered = dict(selected, margin=0.1)
            with self.assertRaisesRegex(ValueError, "immutable"):
                write_immutable_json(path, altered)
            with path.open("r", encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), selected)


if __name__ == "__main__":
    unittest.main()
