from unittest.mock import patch

import pytest

from scientific_preflight import run_pretraining_physical_sanity


def _metrics(action_response=0.02, errors=None):
    return {
        "profile_digest": "profile",
        "seeds": [1, 2, 3],
        "finite_fraction": 1.0,
        "power_contract_fraction": 1.0,
        "sensing_feasible_fraction": 1.0,
        "active_feasible_fraction": 1.0,
        "cascade_feasible_fraction": 1.0,
        "joint_feasible_fraction": 1.0,
        "comm_rate_mean": 8.0,
        "sensing_rate_mean": 2.0,
        "comm_rate_cv": 0.02,
        "sensing_rate_cv": 0.03,
        "comm_action_response_median": action_response,
        "comm_action_response_minimum": action_response / 2.0,
        "sensing_action_response_median": 0.01,
        "errors": list(errors or []),
    }


def test_pretraining_sanity_accepts_feasible_action_sensitive_configuration():
    with patch(
        "parameter_identification.evaluate_candidate",
        return_value=_metrics(),
    ):
        report = run_pretraining_physical_sanity(seeds=(1, 2, 3))
    assert report["passed"] is True
    assert report["comm_action_response_median"] == pytest.approx(0.02)


def test_pretraining_sanity_reports_action_insensitivity_without_stopping():
    with patch(
        "parameter_identification.evaluate_candidate",
        return_value=_metrics(action_response=0.001),
    ):
        report = run_pretraining_physical_sanity(seeds=(1, 2, 3))
    assert report["passed"] is False
    assert report["advisory_only"] is True
    assert report["reasons"]
