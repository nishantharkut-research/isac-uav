import tempfile
import unittest
import json
from pathlib import Path

from experiment_protocol import ensure_run_directory, make_run_manifest, parse_seeds
from run_baseline import build_parser as baseline_parser
from run_extension import (
    build_extension_objective_manifest,
    build_parser as extension_parser,
    require_comparable_budget,
    resolve_extension_rate_floors,
)
from config import cfg


class EntrypointContractTests(unittest.TestCase):
    def test_seed_parser_requires_five_unique_seeds(self):
        self.assertEqual(parse_seeds("1,2,3,4,5"), (1, 2, 3, 4, 5))
        with self.assertRaisesRegex(ValueError, "at least five"):
            parse_seeds("1,2,3,4")
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_seeds("1,1,2,3,4")

    def test_output_manifest_rejects_mismatched_run(self):
        first = make_run_manifest(
            "paper_baseline", "profile-a", "physics-a",
            {"mode": "paper_baseline"}, (1, 2, 3, 4, 5),
        )
        second = make_run_manifest(
            "paper_baseline", "profile-b", "physics-a",
            {"mode": "paper_baseline"}, (1, 2, 3, 4, 5),
        )
        with tempfile.TemporaryDirectory() as directory:
            ensure_run_directory(directory, first)
            ensure_run_directory(directory, first)
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                ensure_run_directory(directory, second)

    def test_extension_manifest_locks_baseline_authority(self):
        first = make_run_manifest(
            "sensing_extension", "profile", "physics",
            {"mode": "comm_retention_sensing"}, (1, 2, 3, 4, 5),
        )
        first.update({
            "baseline_payload_sha256": "a" * 64,
            "baseline_validation_sha256": "b" * 64,
            "episodes": 1500,
            "evaluation_episodes": 5,
        })
        second = dict(first)
        second["baseline_payload_sha256"] = "c" * 64
        with tempfile.TemporaryDirectory() as directory:
            ensure_run_directory(directory, first)
            with self.assertRaisesRegex(ValueError, "baseline_payload_sha256"):
                ensure_run_directory(directory, second)

    def test_extension_budget_must_match_baseline(self):
        args = type("Args", (), {"episodes": 1500, "eval_episodes": 5})()
        require_comparable_budget(args, {"episodes": 1500, "evaluation_episodes": 5})
        args.episodes = 1000
        with self.assertRaisesRegex(ValueError, "training budget"):
            require_comparable_budget(
                args, {"episodes": 1500, "evaluation_episodes": 5}
            )

    def test_nonempty_unmanifested_and_completed_directories_are_immutable(self):
        manifest = make_run_manifest(
            "paper_baseline", "profile-a", "physics-a",
            {"mode": "paper_baseline"}, (1, 2, 3, 4, 5),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "legacy.png").write_bytes(b"legacy")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                ensure_run_directory(root, manifest)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_run_directory(root, manifest)
            payload = json.loads((root / "run_manifest.json").read_text())
            payload["status"] = "complete"
            (root / "run_manifest.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "immutable"):
                ensure_run_directory(root, manifest)

    def test_cli_parsers_require_research_inputs(self):
        with self.assertRaises(SystemExit):
            baseline_parser().parse_args([])
        baseline = baseline_parser().parse_args([
            "--profile", "profile.json", "--output", "out",
            "--seeds", "1,2,3,4,5", "--dry-run",
        ])
        self.assertTrue(baseline.dry_run)
        alternate = baseline_parser().parse_args([
            "--profile", "profile.json", "--output", "out",
            "--seeds", "1,2,3,4,5", "--dry-run",
            "--noise-hypothesis", "dual_minus70_conservative",
            "--transfer-update-hypothesis",
            "eq30_next_state_continuation_blend",
            "--transfer-schedule-hypothesis", "linear_annealed_upsilon",
            "--wfb-hypothesis", "channel_independent_dft_no_capon",
        ])
        self.assertEqual(alternate.noise_hypothesis,
                         "dual_minus70_conservative")
        self.assertEqual(alternate.transfer_schedule_hypothesis,
                         "linear_annealed_upsilon")

        with self.assertRaises(SystemExit):
            extension_parser().parse_args([])
        extension = extension_parser().parse_args([
            "--profile", "profile.json", "--baseline-summary", "baseline.json",
            "--baseline-validation", "baseline-validation.json",
            "--output", "out", "--seeds", "1,2,3,4,5",
            "--objective", "comm_retention_sensing", "--dry-run",
        ])
        self.assertEqual(extension.objective, "comm_retention_sensing")

    def test_png_extension_cannot_silently_disable_its_rate_floors(self):
        args = extension_parser().parse_args([
            "--profile", "profile.json", "--baseline-summary", "baseline.json",
            "--baseline-validation", "baseline-validation.json",
            "--output", "out", "--seeds", "1,2,3,4,5",
            "--objective", "png_joint_sum", "--dry-run",
        ])
        user_floor, target_floor = resolve_extension_rate_floors(args, cfg)
        self.assertEqual(user_floor, cfg.R_u_min)
        self.assertEqual(target_floor, cfg.R_min)
        args.user_rate_min = 0.0
        with self.assertRaisesRegex(ValueError, "positive per-UE"):
            resolve_extension_rate_floors(args, cfg)

    def test_retention_argument_is_rejected_for_literal_png_mode(self):
        args = extension_parser().parse_args([
            "--profile", "profile.json", "--baseline-summary", "baseline.json",
            "--baseline-validation", "baseline-validation.json",
            "--output", "out", "--seeds", "1,2,3,4,5",
            "--objective", "png_joint_sum", "--retention", "0.95", "--dry-run",
        ])
        with self.assertRaisesRegex(ValueError, "only valid"):
            build_extension_objective_manifest(args, 0.1, 0.5)

        args.retention = None
        manifest = build_extension_objective_manifest(args, 0.1, 0.5)
        self.assertNotIn("communication_retention_fraction", manifest)


if __name__ == "__main__":
    unittest.main()
