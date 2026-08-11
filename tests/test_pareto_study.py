import copy
import json
import tempfile
import unittest
from pathlib import Path

from run_pareto_study import (
    build_scenario_command,
    initialize_study_state,
    load_study_design,
)


def _design():
    common = {
        "research_seeds": list(range(10)),
        "validation_scene_seeds": list(range(100, 105)),
        "test_scene_seeds": list(range(200, 210)),
        "episodes": 1500,
        "evaluation_episodes": 5,
        "checkpoint_every": 10,
        "comparison_level": "primary",
    }
    return {
        "schema_version": 1,
        "kind": "ba_pcdtl_robustness_study",
        "common": common,
        "scenarios": [
            {
                "id": "pcsi",
                "profile": "profiles/pcsi.json",
                "baseline_summary": "runs/pcsi/baseline_summary.json",
                "baseline_validation": "runs/pcsi/baseline_validation.json",
                "output": "runs/pcsi/pareto",
                "extra_args": [],
            },
            {
                "id": "icsi",
                "profile": "profiles/icsi.json",
                "baseline_summary": "runs/icsi/baseline_summary.json",
                "baseline_validation": "runs/icsi/baseline_validation.json",
                "output": "runs/icsi/pareto",
                "extra_args": ["--comparison-level", "full"],
            },
        ],
    }


class ParetoStudyTests(unittest.TestCase):
    def test_comparator_csi_hypothesis_is_an_allowed_explicit_override(self):
        payload = _design()
        payload["scenarios"][0]["extra_args"] = [
            "--ddpg-comparator-csi-hypothesis",
            "icsi_comparator",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            design = load_study_design(path, require_paths=False)
            command = build_scenario_command(
                design, design["scenarios"][0], dry_run=True
            )
        self.assertIn("--ddpg-comparator-csi-hypothesis", command)
        self.assertIn("icsi_comparator", command)

    def test_design_is_immutable_and_command_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(_design()), encoding="utf-8")
            design = load_study_design(path, require_paths=False)
            state = initialize_study_state(design)
            self.assertEqual(state["scenarios"][0]["status"], "pending")
            command = build_scenario_command(design, design["scenarios"][0], dry_run=True)
            self.assertIn("--baseline-summary", command)
            self.assertIn("--dry-run", command)
            self.assertIn("0,1,2,3,4,5,6,7,8,9", command)

    def test_duplicate_ids_outputs_and_changed_resume_design_are_rejected(self):
        invalid_designs = []
        duplicate_id = _design()
        duplicate_id["scenarios"][1]["id"] = "pcsi"
        invalid_designs.append(duplicate_id)
        duplicate_output = _design()
        duplicate_output["scenarios"][1]["output"] = "runs/pcsi/pareto"
        invalid_designs.append(duplicate_output)
        missing_authority = _design()
        del missing_authority["scenarios"][0]["baseline_validation"]
        invalid_designs.append(missing_authority)

        with tempfile.TemporaryDirectory() as directory:
            for index, payload in enumerate(invalid_designs):
                with self.subTest(index=index):
                    path = Path(directory) / f"invalid-{index}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_study_design(path, require_paths=False)

            valid_path = Path(directory) / "valid.json"
            valid_path.write_text(json.dumps(_design()), encoding="utf-8")
            design = load_study_design(valid_path, require_paths=False)
            state = initialize_study_state(design)
            changed = copy.deepcopy(design)
            changed["common"]["episodes"] = 1000
            with self.assertRaisesRegex(ValueError, "design"):
                initialize_study_state(changed, existing=state)


if __name__ == "__main__":
    unittest.main()
