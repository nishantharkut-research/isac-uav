"""Select and validate an operating profile before any reproduction run."""
import argparse
import json
from pathlib import Path

from config import cfg
from parameter_evidence import (
    SENSITIVITY_ONLY_KEYS,
    STRUCTURAL_IDENTIFICATION_KEYS,
)
from parameter_identification import run_search
from parameter_profiles import activate_profile, load_profile
from scientific_preflight import run_scientific_preflight


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def prepare_reproduction(output_dir, candidate_count=32,
                         generator_seed=cfg.random_seed):
    """Run no-training selection and preflight, then write one authority file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    authority_path = output_dir / "reproduction_selection.json"
    preflight_path = output_dir / "scientific_preflight.json"
    status_path = output_dir / "preparation_status.json"
    for stale_path in (authority_path, preflight_path):
        if stale_path.exists():
            stale_path.unlink()

    status = {
        "schema_version": 1,
        "status": "running",
        "ready_for_full_run": False,
        "candidate_count": int(candidate_count),
        "generator_seed": int(generator_seed),
    }
    _write_json(status_path, status)

    try:
        search = run_search(
            output_dir,
            candidate_count=int(candidate_count),
            generator_seed=int(generator_seed),
        )
        activate_profile(load_profile(search["profile_path"]))
        preflight = run_scientific_preflight(preflight_path)
        if not bool(preflight.get("passed", False)):
            raise RuntimeError("Scientific preflight did not report a pass")

        authority = {
            "schema_version": 1,
            "purpose": "no-training operating-profile selection authority",
            "selected_profile": Path(search["profile_path"]).name,
            "selected_profile_digest": search["profile"].digest,
            "selection_rule": (
                "minimum predeclared structural score among candidates with "
                "strict calibration feasibility, followed by one disjoint "
                "holdout, stress, and dynamic-preflight pass"
            ),
            "paper_curve_ordinates_used": False,
            "communication_rate_magnitude_used_in_selection": False,
            "candidate_count": int(candidate_count),
            "generator_seed": int(generator_seed),
            "automatically_selected_parameters": sorted(
                STRUCTURAL_IDENTIFICATION_KEYS
            ),
            "sensitivity_only_parameters": sorted(SENSITIVITY_ONLY_KEYS),
            "parameter_search": Path(search["json_path"]).name,
            "parameter_matrix": Path(search["npz_path"]).name,
            "holdout_validation": Path(search["holdout_path"]).name,
            "stress_validation": Path(search["stress_path"]).name,
            "scientific_preflight": preflight_path.name,
            "preflight_passed": True,
            "ready_for_full_run": True,
            "claim_boundary": (
                "Ready means bounded code-level feasibility only; it does not "
                "establish numerical agreement with published figures."
            ),
        }
        _write_json(authority_path, authority)
        status.update({
            "status": "passed",
            "ready_for_full_run": True,
            "selected_profile_digest": search["profile"].digest,
            "authority": authority_path.name,
        })
        _write_json(status_path, status)
    except Exception as exc:
        if authority_path.exists():
            authority_path.unlink()
        status.update({
            "status": "failed",
            "ready_for_full_run": False,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        })
        _write_json(status_path, status)
        raise

    return authority_path, authority


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select omitted physical parameters and run no-training preflight"
        )
    )
    parser.add_argument(
        "--output", default="result/parameter_identification",
        help="Directory for the selected profile and validation authority",
    )
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    path, payload = prepare_reproduction(
        args.output, candidate_count=args.candidates,
        generator_seed=args.seed,
    )
    print(f"Selection authority: {path}")
    print(f"Profile SHA-256: {payload['selected_profile_digest']}")
