import json
import tempfile
from pathlib import Path

import pytest

from parameter_profiles import (
    load_prepared_profile,
    profile_from_mapping,
    write_profile,
)
from source_integrity import source_tree_digest


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _prepared_fixture(directory):
    directory = Path(directory)
    source_digest = source_tree_digest()
    profile = profile_from_mapping({
        "schema_version": 1,
        "name": "prepared-fixture",
        "kind": "identified",
        "values": {"omega_0": 1e-3},
        "evidence": {"omega_0": "omega_0"},
        "selection": {
            "protocol": "structural preparation fixture",
            "source_tree_digest": source_digest,
        },
    })
    profile_path = write_profile(profile, directory / "identified_profile.json")
    common = {"schema_version": 1, "source_tree_digest": source_digest}
    _write_json(directory / "parameter_search.json", {
        **common, "selected_profile_digest": profile.digest,
    })
    (directory / "parameter_search.npz").write_bytes(b"fixture")
    _write_json(directory / "holdout_validation.json", {
        **common, "accepted": True,
    })
    _write_json(directory / "stress_validation.json", {
        **common, "accepted": True, "profile_digest": profile.digest,
    })
    _write_json(directory / "scientific_preflight.json", {
        **common,
        "passed": True,
        "parameter_profile": {"digest": profile.digest},
    })
    _write_json(directory / "reproduction_selection.json", {
        **common,
        "ready_for_full_run": True,
        "preflight_passed": True,
        "paper_curve_ordinates_used": False,
        "communication_rate_magnitude_used_in_selection": False,
        "selected_profile_digest": profile.digest,
        "selected_profile": profile_path.name,
        "parameter_search": "parameter_search.json",
        "parameter_matrix": "parameter_search.npz",
        "holdout_validation": "holdout_validation.json",
        "stress_validation": "stress_validation.json",
        "scientific_preflight": "scientific_preflight.json",
    })
    _write_json(directory / "preparation_status.json", {
        **common,
        "status": "passed",
        "ready_for_full_run": True,
        "selected_profile_digest": profile.digest,
        "authority": "reproduction_selection.json",
    })
    return profile_path, profile


def test_prepared_profile_accepts_only_matching_current_source_tree():
    with tempfile.TemporaryDirectory() as directory:
        profile_path, expected = _prepared_fixture(directory)
        assert load_prepared_profile(profile_path).digest == expected.digest

        authority_path = Path(directory) / "reproduction_selection.json"
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        authority["source_tree_digest"] = "0" * 64
        _write_json(authority_path, authority)
        with pytest.raises(ValueError, match="source tree"):
            load_prepared_profile(profile_path)
