import unittest
import copy
import json
import tempfile
from pathlib import Path

from config import cfg
from experiment_protocol import (
    EVALUATION_SEED_PROTOCOL,
    derive_evaluation_scene_seed,
    evaluation_scene_seed_grid,
    final_window_mean,
    physical_configuration_fingerprint,
    resolve_required_checkpoint,
    step_projection_evidence,
    write_immutable_json,
)


class ExperimentProtocolTests(unittest.TestCase):
    def test_evaluation_scene_grid_is_global_unique_and_reproducible(self):
        research_seeds = tuple(range(2025, 2035))
        first = evaluation_scene_seed_grid(research_seeds, 5)
        second = evaluation_scene_seed_grid(research_seeds, 5)
        flattened = [seed for row in first.values() for seed in row]

        self.assertEqual(first, second)
        self.assertEqual(len(flattened), 50)
        self.assertEqual(len(set(flattened)), 50)
        self.assertEqual(
            first[2025][0], derive_evaluation_scene_seed(2025, 0)
        )
        self.assertTrue(EVALUATION_SEED_PROTOCOL.startswith("sha256_"))
        forbidden = set(research_seeds) | {
            seed + 100_000 for seed in research_seeds
        }
        self.assertFalse(set(flattened) & forbidden)

    def test_final_window_mean_uses_only_declared_training_tail(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(final_window_mean(values, window=2), 3.5)
        self.assertEqual(final_window_mean(values, window=100), 2.5)

    def test_final_window_mean_rejects_invalid_training_history(self):
        for values in ([], [1.0, float("nan")], [[1.0], [2.0]]):
            with self.assertRaises(ValueError):
                final_window_mean(values, window=2)
        with self.assertRaises(ValueError):
            final_window_mean([1.0], window=0)

    def test_physics_fingerprint_includes_power_reference(self):
        dbw_config = copy.copy(cfg)
        dbm_config = copy.copy(cfg)
        dbw_config.PT_reference = "1 W"
        dbm_config.PT_reference = "1 mW"
        dbw = physical_configuration_fingerprint("profile", dbw_config)
        dbm = physical_configuration_fingerprint("profile", dbm_config)
        self.assertNotEqual(dbw, dbm)

    def test_physics_fingerprint_covers_channel_and_geometry(self):
        baseline = physical_configuration_fingerprint("profile", copy.copy(cfg))
        for field, value in (
            ("omega_0", cfg.omega_0 * 2.0),
            ("d_spacing", cfg.d_spacing * 0.8),
            ("z_UAV", cfg.z_UAV + 1.0),
            ("alpha_max", cfg.alpha_max * 0.9),
            ("uav_start_x", cfg.uav_start_x + 1.0),
            ("transition_timing_id", "different_transition"),
        ):
            with self.subTest(field=field):
                changed = copy.copy(cfg)
                setattr(changed, field, value)
                self.assertNotEqual(
                    baseline,
                    physical_configuration_fingerprint("profile", changed),
                )

    def test_required_checkpoint_must_exist_inside_run_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            (run_directory / "run_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "kind": "run_manifest",
                "status": "complete",
            }), encoding="utf-8")
            checkpoint = run_directory / "checkpoints" / "source.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            self.assertEqual(
                resolve_required_checkpoint(
                    run_directory, "checkpoints/source.pt"
                ),
                checkpoint.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "checkpoints directory"):
                resolve_required_checkpoint(run_directory, "../outside.pt")
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
                resolve_required_checkpoint(
                    run_directory, "checkpoints/missing.pt"
                )

    def test_checkpoint_directory_requires_completed_current_run_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            checkpoint = run_directory / "checkpoints" / "source.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "completed run manifest"):
                resolve_required_checkpoint(run_directory, "checkpoints/source.pt")

            (run_directory / "run_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "kind": "run_manifest",
                "status": "planned",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed run manifest"):
                resolve_required_checkpoint(run_directory, "checkpoints/source.pt")

    def test_immutable_json_is_idempotent_and_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            payload = {"schema_version": 1, "value": [1, 2, 3]}
            write_immutable_json(path, payload)
            write_immutable_json(path, copy.deepcopy(payload))
            with self.assertRaisesRegex(ValueError, "Immutable JSON"):
                write_immutable_json(path, {**payload, "value": [3, 2, 1]})

    def test_projection_evidence_rejects_string_boolean_flags(self):
        valid = {
            "action_was_projected": False,
            "action_projection_fraction": 1.0,
            "projection_source": "actor_action",
            "allocation_fallback_used": False,
        }
        for field in ("action_was_projected", "allocation_fallback_used"):
            with self.subTest(field=field):
                malformed = {**valid, field: "False"}
                with self.assertRaisesRegex(ValueError, "Boolean"):
                    step_projection_evidence(malformed)


if __name__ == "__main__":
    unittest.main()
