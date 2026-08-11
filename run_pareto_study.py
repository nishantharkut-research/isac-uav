"""Run a preregistered sequence of BA-PCDTL robustness scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiment_protocol import atomic_write_json
from run_pareto_extension import validate_seed_roles


TOP_LEVEL_FIELDS = {"schema_version", "kind", "common", "scenarios"}
COMMON_FIELDS = {
    "research_seeds", "validation_scene_seeds", "test_scene_seeds",
    "episodes", "evaluation_episodes", "checkpoint_every", "comparison_level",
}
SCENARIO_FIELDS = {
    "id", "profile", "baseline_summary", "baseline_validation", "output",
    "extra_args",
}
ALLOWED_EXTRA_FLAGS = {
    "--comparison-level",
    "--noise-hypothesis",
    "--transfer-update-hypothesis",
    "--transfer-schedule-hypothesis",
    "--wfb-hypothesis",
    "--ddpg-comparator-csi-hypothesis",
}


def _canonical_digest(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_design_digest(design) -> str:
    return _canonical_digest({
        "schema_version": design["schema_version"],
        "kind": design["kind"],
        "common": design["common"],
        "scenarios": design["scenarios"],
    })


def _validate_extra_args(values) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ValueError("Scenario extra_args must be a string list.")
    if len(values) % 2:
        raise ValueError("Every scenario override flag requires one value.")
    for index in range(0, len(values), 2):
        if values[index] not in ALLOWED_EXTRA_FLAGS:
            raise ValueError(f"Scenario override is not allowed: {values[index]}")
    return list(values)


def load_study_design(path, *, require_paths=True) -> dict:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict) or set(raw) - TOP_LEVEL_FIELDS:
        raise ValueError("Study design has unknown or invalid top-level fields.")
    if raw.get("schema_version") != 1 or raw.get("kind") != "ba_pcdtl_robustness_study":
        raise ValueError("Unsupported BA-PCDTL study schema.")
    common = raw.get("common")
    if not isinstance(common, dict) or set(common) != COMMON_FIELDS:
        raise ValueError("Study common protocol fields are incomplete or unknown.")
    validate_seed_roles(
        common["research_seeds"],
        common["validation_scene_seeds"],
        common["test_scene_seeds"],
    )
    for field in ("episodes", "evaluation_episodes", "checkpoint_every"):
        if int(common[field]) <= 0:
            raise ValueError(f"Study {field} must be positive.")
    if common["comparison_level"] not in {"primary", "full"}:
        raise ValueError("Study comparison_level must be primary or full.")
    scenarios = raw.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Study requires at least one scenario.")
    ids = []
    outputs = []
    resolved = []
    for scenario in scenarios:
        if not isinstance(scenario, dict) or set(scenario) != SCENARIO_FIELDS:
            raise ValueError("Study scenario fields are incomplete or unknown.")
        scenario_id = scenario.get("id")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("Scenario id must be a nonempty string.")
        ids.append(scenario_id)
        extra_args = _validate_extra_args(scenario["extra_args"])
        resolved_scenario = {"id": scenario_id, "extra_args": extra_args}
        for field in ("profile", "baseline_summary", "baseline_validation", "output"):
            value = scenario.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Scenario {scenario_id} has no {field} path.")
            resolved_path = (source.parent / value).resolve()
            resolved_scenario[field] = str(resolved_path)
            if require_paths and field != "output" and not resolved_path.is_file():
                raise FileNotFoundError(
                    f"Scenario {scenario_id} required path is missing: {resolved_path}"
                )
        outputs.append(resolved_scenario["output"])
        resolved.append(resolved_scenario)
    if len(ids) != len(set(ids)):
        raise ValueError("Study scenario ids must be unique.")
    if len(outputs) != len(set(outputs)):
        raise ValueError("Study scenario output paths must be unique.")
    result = {
        "schema_version": 1,
        "kind": raw["kind"],
        "common": dict(common),
        "scenarios": resolved,
        "design_path": str(source),
    }
    result["design_digest"] = _normalized_design_digest(result)
    return result


def initialize_study_state(design, existing=None) -> dict:
    digest = _normalized_design_digest(design)
    if design.get("design_digest") != digest:
        raise ValueError("Study design contents do not match its normalized digest.")
    ids = [scenario["id"] for scenario in design.get("scenarios", [])]
    if existing is not None:
        if existing.get("design_digest") != digest:
            raise ValueError("Existing state belongs to a different study design.")
        if [row.get("id") for row in existing.get("scenarios", [])] != ids:
            raise ValueError("Existing state scenario order differs from the study design.")
        return dict(existing)
    return {
        "schema_version": 1,
        "kind": "ba_pcdtl_robustness_study_state",
        "design_digest": digest,
        "design_path": design.get("design_path"),
        "scenarios": [
            {"id": scenario_id, "status": "pending", "return_code": None}
            for scenario_id in ids
        ],
    }


def _seed_argument(values) -> str:
    return ",".join(str(int(value)) for value in values)


def build_scenario_command(design, scenario, *, dry_run=False) -> list[str]:
    common = design["common"]
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_pareto_extension.py"),
        "--profile", scenario["profile"],
        "--baseline-summary", scenario["baseline_summary"],
        "--baseline-validation", scenario["baseline_validation"],
        "--output", scenario["output"],
        "--research-seeds", _seed_argument(common["research_seeds"]),
        "--validation-scene-seeds", _seed_argument(common["validation_scene_seeds"]),
        "--test-scene-seeds", _seed_argument(common["test_scene_seeds"]),
        "--episodes", str(int(common["episodes"])),
        "--eval-episodes", str(int(common["evaluation_episodes"])),
        "--checkpoint-every", str(int(common["checkpoint_every"])),
        "--comparison-level", common["comparison_level"],
    ]
    command.extend(scenario["extra_args"])
    if dry_run:
        command.append("--dry-run")
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--state")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    design = load_study_design(args.design, require_paths=True)
    if args.dry_run:
        for scenario in design["scenarios"]:
            completed = subprocess.run(
                build_scenario_command(design, scenario, dry_run=True),
                check=False,
            )
            if completed.returncode != 0:
                return int(completed.returncode)
        return 0

    state_path = Path(
        args.state or Path(args.design).resolve().with_name("pareto_study_state.json")
    ).resolve()
    existing = None
    if state_path.is_file():
        with state_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
    state = initialize_study_state(design, existing)
    for index, scenario in enumerate(design["scenarios"]):
        row = state["scenarios"][index]
        if row["status"] == "complete":
            continue
        row["status"] = "running"
        row["return_code"] = None
        atomic_write_json(state_path, state)
        completed = subprocess.run(
            build_scenario_command(design, scenario, dry_run=False),
            check=False,
        )
        row["return_code"] = int(completed.returncode)
        row["status"] = "complete" if completed.returncode == 0 else "failed"
        atomic_write_json(state_path, state)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
