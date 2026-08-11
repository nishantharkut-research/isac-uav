"""Fast contracts for evidence, parameter profiles, and identification.

Run from this directory:
    python test_parameter_identification.py

No test in this file performs DRL training or fits published curve values.
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from config import cfg


def _example_profile(values=None, evidence=None):
    selected_values = {
        "fc": 2.5e9,
        "omega_0": 2.0e-3,
        "ue_x_min": 20.0,
        "ue_x_max": 120.0,
        "ue_y_min": 30.0,
        "ue_y_max": 130.0,
    }
    if values:
        selected_values.update(values)
    selected_evidence = {
        key: key for key in selected_values
    }
    if evidence:
        selected_evidence.update(evidence)
    return {
        "schema_version": 1,
        "name": "test-profile",
        "kind": "identified",
        "values": selected_values,
        "evidence": selected_evidence,
        "selection": {"protocol": "unit-test fixture; no paper ordinates"},
    }


def test_evidence_registry_has_distinct_classes_and_source_locators():
    from parameter_evidence import EVIDENCE_CLASSES, PARAMETER_EVIDENCE

    assert EVIDENCE_CLASSES == {
        "journal_reported",
        "figure_inferred",
        "cited_reference",
        "assumption",
    }
    required = {
        "M", "N", "U", "K", "L", "La", "sigma2_u_dBm",
        "sigma2_qa_dBm", "PT_dB", "ue_region_size_m",
        "service_area_m", "fc", "d_spacing", "omega_0", "rho_br",
        "rho_rt", "R_min", "R_u_min", "Psi_max", "P_a_max", "target_speed_mps",
        "v_max", "z_UAV", "random_seed", "tau", "partial_csi_fraction",
        "upsilon_0",
    }
    assert required <= set(PARAMETER_EVIDENCE)
    for name, entry in PARAMETER_EVIDENCE.items():
        assert entry.name == name
        assert entry.evidence_class in EVIDENCE_CLASSES
        assert entry.source.strip()
        assert entry.unit.strip()
        assert entry.applicability.strip()
        assert entry.confidence in {"high", "medium", "low"}


def test_profile_allowlist_excludes_paper_reported_system_dimensions():
    from parameter_evidence import (
        PROFILE_KEYS,
        SENSITIVITY_ONLY_KEYS,
        STRUCTURAL_IDENTIFICATION_KEYS,
    )

    for reported_key in ("M", "N", "U", "K", "L", "La", "alpha_max_dB"):
        assert reported_key not in PROFILE_KEYS
    for omitted_key in (
        "fc", "omega_0", "rho_br", "rho_rt", "R_min", "R_u_min", "Psi_max",
        "P_a_max", "ue_x_min", "ue_x_max", "target_speed_mps",
    ):
        assert omitted_key in PROFILE_KEYS
    assert STRUCTURAL_IDENTIFICATION_KEYS.isdisjoint(SENSITIVITY_ONLY_KEYS)
    assert STRUCTURAL_IDENTIFICATION_KEYS | SENSITIVITY_ONLY_KEYS == PROFILE_KEYS


def test_profile_rejects_unknown_and_nonfinite_values():
    from parameter_profiles import profile_from_mapping

    invalid_key = _example_profile({"paper_curve_scale": 2.0})
    invalid_key["evidence"]["paper_curve_scale"] = "omega_0"
    try:
        profile_from_mapping(invalid_key)
    except ValueError as exc:
        assert "Unknown profile parameter" in str(exc)
    else:
        raise AssertionError("unknown profile parameter was accepted")

    nonfinite = _example_profile({"omega_0": math.inf})
    try:
        profile_from_mapping(nonfinite)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite profile value was accepted")


def test_profile_requires_valid_evidence_reference_for_every_value():
    from parameter_profiles import profile_from_mapping

    payload = _example_profile()
    del payload["evidence"]["fc"]
    try:
        profile_from_mapping(payload)
    except ValueError as exc:
        assert "evidence" in str(exc).lower()
    else:
        raise AssertionError("profile value without evidence was accepted")

    payload = _example_profile(evidence={"fc": "M"})
    try:
        profile_from_mapping(payload)
    except ValueError as exc:
        assert "does not apply" in str(exc)
    else:
        raise AssertionError("mismatched evidence reference was accepted")


def test_profile_digest_is_stable_across_json_key_order_and_load():
    from parameter_profiles import load_profile, profile_from_mapping

    first_payload = _example_profile()
    second_payload = dict(reversed(list(first_payload.items())))
    second_payload["values"] = dict(reversed(list(first_payload["values"].items())))
    second_payload["evidence"] = dict(
        reversed(list(first_payload["evidence"].items()))
    )
    first = profile_from_mapping(first_payload)
    second = profile_from_mapping(second_payload)
    assert first.digest == second.digest

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.json"
        path.write_text(json.dumps(second_payload), encoding="utf-8")
        loaded = load_profile(path)
    assert loaded.digest == first.digest


def test_full_run_profile_requires_matching_success_authority():
    from parameter_profiles import (
        load_prepared_profile,
        profile_from_mapping,
        write_profile,
    )

    def write_json(path, payload):
        Path(path).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )

    with tempfile.TemporaryDirectory() as directory_name:
        from source_integrity import source_tree_digest

        directory = Path(directory_name)
        source_digest = source_tree_digest()
        profile_payload = _example_profile()
        profile_payload["selection"]["source_tree_digest"] = source_digest
        profile = profile_from_mapping(profile_payload)
        profile_path = write_profile(
            profile, directory / "identified_profile.json"
        )
        write_json(directory / "parameter_search.json", {
            "selected_profile_digest": profile.digest,
            "source_tree_digest": source_digest,
        })
        (directory / "parameter_search.npz").write_bytes(b"fixture")
        write_json(directory / "holdout_validation.json", {
            "accepted": True,
            "source_tree_digest": source_digest,
        })
        write_json(directory / "stress_validation.json", {
            "accepted": True,
            "profile_digest": profile.digest,
            "source_tree_digest": source_digest,
        })
        write_json(directory / "scientific_preflight.json", {
            "passed": True,
            "parameter_profile": {"digest": profile.digest},
            "source_tree_digest": source_digest,
        })
        write_json(directory / "reproduction_selection.json", {
            "selected_profile": "identified_profile.json",
            "selected_profile_digest": profile.digest,
            "parameter_search": "parameter_search.json",
            "parameter_matrix": "parameter_search.npz",
            "holdout_validation": "holdout_validation.json",
            "stress_validation": "stress_validation.json",
            "scientific_preflight": "scientific_preflight.json",
            "preflight_passed": True,
            "ready_for_full_run": True,
            "paper_curve_ordinates_used": False,
            "communication_rate_magnitude_used_in_selection": False,
            "source_tree_digest": source_digest,
        })
        status_path = directory / "preparation_status.json"
        write_json(status_path, {
            "status": "passed",
            "ready_for_full_run": True,
            "selected_profile_digest": profile.digest,
            "authority": "reproduction_selection.json",
            "source_tree_digest": source_digest,
        })

        loaded = load_prepared_profile(profile_path)
        assert loaded.digest == profile.digest

        write_json(status_path, {
            "status": "failed",
            "ready_for_full_run": False,
            "selected_profile_digest": profile.digest,
            "authority": "reproduction_selection.json",
            "source_tree_digest": source_digest,
        })
        try:
            load_prepared_profile(profile_path)
        except ValueError as exc:
            assert "did not finish successfully" in str(exc)
        else:
            raise AssertionError("failed preparation authority was accepted")


def test_applied_profile_derives_dependent_values_and_restores_cfg():
    from parameter_profiles import applied_profile, profile_from_mapping

    profile = profile_from_mapping(_example_profile())
    original = {
        "fc": cfg.fc,
        "wavelength": cfg.wavelength,
        "PT": cfg.PT,
        "PT_dBm": cfg.PT_dBm,
        "PT_dB": cfg.PT_dB,
        "ue_x_min": cfg.ue_x_min,
    }
    with applied_profile(profile):
        assert cfg.fc == 2.5e9
        assert np.isclose(cfg.wavelength, 3e8 / 2.5e9)
        assert cfg.PT_dBm == original["PT_dBm"] and cfg.PT_dB == 10.0
        assert np.isclose(cfg.PT, original["PT"])
        assert cfg.ue_x_min == 20.0
    for key, value in original.items():
        assert getattr(cfg, key) == value


def test_profile_geometry_and_dependent_ranges_fail_closed():
    from parameter_profiles import profile_from_mapping

    bad_width = _example_profile({"ue_x_max": 119.0})
    try:
        profile_from_mapping(bad_width)
    except ValueError as exc:
        assert "100 m" in str(exc)
    else:
        raise AssertionError("invalid UE-region width was accepted")

    bad_exponents = _example_profile({"zeta_los": 3.0, "zeta_nlos": 3.0})
    try:
        profile_from_mapping(bad_exponents)
    except ValueError as exc:
        assert "zeta_los" in str(exc)
    else:
        raise AssertionError("invalid LoS/NLoS exponent ordering was accepted")


def test_structural_candidate_generation_is_deterministic_and_bounded():
    from parameter_identification import generate_candidates

    first = generate_candidates(6, seed=401)
    second = generate_candidates(6, seed=401)
    third = generate_candidates(6, seed=402)
    extended = generate_candidates(10, seed=401)
    assert [item.digest for item in first] == [item.digest for item in second]
    assert [item.digest for item in first] != [item.digest for item in third]
    assert [item.values for item in first] == [item.values for item in extended[:6]]
    for profile in first:
        values = profile.values
        assert np.isclose(values["ue_x_max"] - values["ue_x_min"], 100.0)
        assert np.isclose(values["ue_y_max"] - values["ue_y_min"], 100.0)
        assert values["zeta_los"] < values["zeta_nlos"]
        assert values["R_min"] == 0.5
        assert values["Psi_max"] == 1.0
        assert values["P_a_max"] == 10.0


def test_constraint_derivation_uses_conservative_extrema_not_rate_level():
    from parameter_identification import derive_constraint_values

    raw = [
        {
            "comm_rate": 1.0 + index,
            "min_sensing_rate": value,
            "P_ai": value * 1e-8,
            "P_ai_envelope": value * 2e-8,
            "cascade_power": value * 1e-12,
        }
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0))
    ]
    changed_rates = [dict(row, comm_rate=1000.0 * row["comm_rate"]) for row in raw]
    first = derive_constraint_values(raw)
    second = derive_constraint_values(changed_rates)
    assert first == second
    assert first["R_min"] == 0.5 * min([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(first["P_a_max"], 2.0 * max(
        [2e-8, 4e-8, 6e-8, 8e-8]
    ))
    assert np.isclose(first["Psi_max"], 2.0 * max(
        [1e-12, 2e-12, 3e-12, 4e-12]
    ))


def test_calibration_and_holdout_seed_sets_are_disjoint():
    from parameter_identification import (
        CALIBRATION_SEEDS,
        HOLDOUT_SEEDS,
        STRESS_SEEDS,
    )

    assert CALIBRATION_SEEDS
    assert HOLDOUT_SEEDS
    assert STRESS_SEEDS
    assert set(CALIBRATION_SEEDS).isdisjoint(HOLDOUT_SEEDS)
    assert set(CALIBRATION_SEEDS).isdisjoint(STRESS_SEEDS)
    assert set(HOLDOUT_SEEDS).isdisjoint(STRESS_SEEDS)


def test_structural_score_does_not_optimize_communication_rate_magnitude():
    from parameter_identification import score_candidate

    common = {
        "finite_fraction": 1.0,
        "power_contract_fraction": 1.0,
        "sensing_feasible_fraction": 0.875,
        "active_feasible_fraction": 0.875,
        "cascade_feasible_fraction": 0.875,
        "joint_feasible_fraction": 0.75,
        "sensing_utilization_median": 0.9,
        "active_utilization_median": 0.8,
        "cascade_utilization_median": 0.7,
        "comm_rate_cv": 0.1,
        "sensing_rate_cv": 0.1,
        "comm_action_response_median": 0.01,
        "raw_scenes": [],
        "errors": [],
    }
    low_rate = dict(common, comm_rate_mean=0.5, sensing_rate_mean=0.2)
    high_rate = dict(common, comm_rate_mean=50.0, sensing_rate_mean=20.0)
    low = score_candidate(low_rate, prior_distance=0.2)
    high = score_candidate(high_rate, prior_distance=0.2)
    assert low["score"] == high["score"]
    assert low["components"] == high["components"]


def test_structural_score_rejects_nonfinite_or_broken_power_contracts():
    from parameter_identification import score_candidate

    invalid = {
        "finite_fraction": 1.0,
        "power_contract_fraction": 1.0,
        "sensing_feasible_fraction": 1.0,
        "active_feasible_fraction": 1.0,
        "cascade_feasible_fraction": 1.0,
        "joint_feasible_fraction": 1.0,
        "sensing_utilization_median": 1.0,
        "active_utilization_median": 1.0,
        "cascade_utilization_median": 1.0,
        "comm_rate_cv": np.nan,
        "sensing_rate_cv": 0.0,
        "comm_action_response_median": 0.01,
        "comm_rate_mean": 1.0,
        "sensing_rate_mean": 1.0,
        "raw_scenes": [],
        "errors": [],
    }
    result = score_candidate(invalid, prior_distance=0.0)
    assert result["rejected"] is True
    assert math.isinf(result["score"])

    invalid["comm_rate_cv"] = 0.0
    invalid["power_contract_fraction"] = 0.75
    result = score_candidate(invalid, prior_distance=0.0)
    assert result["rejected"] is True
    assert "power contract" in " ".join(result["reasons"]).lower()


def test_structural_acceptance_requires_constraint_and_joint_feasibility():
    from parameter_identification import structural_acceptance

    passing = {
        "finite_fraction": 1.0,
        "power_contract_fraction": 1.0,
        "sensing_feasible_fraction": 1.0,
        "active_feasible_fraction": 1.0,
        "cascade_feasible_fraction": 1.0,
        "joint_feasible_fraction": 1.0,
        "comm_action_response_median": 0.01,
        "errors": [],
    }
    accepted, reasons = structural_acceptance(passing)
    assert accepted is True and reasons == []

    failing = dict(passing, joint_feasible_fraction=0.999)
    accepted, reasons = structural_acceptance(failing)
    assert accepted is False
    assert "joint" in " ".join(reasons).lower()

    insensitive = dict(passing, comm_action_response_median=0.0)
    accepted, reasons = structural_acceptance(insensitive)
    assert accepted is False
    assert "policy actions" in " ".join(reasons).lower()


def test_tiny_structural_evaluation_is_raw_deterministic_and_training_free():
    from parameter_identification import (
        ACTION_PROBE_LABELS,
        evaluate_candidate,
        generate_candidates,
    )

    profile = generate_candidates(1, seed=919)[0]
    first = evaluate_candidate(profile, seeds=(31_001, 31_002))
    second = evaluate_candidate(profile, seeds=(31_001, 31_002))
    assert first == second
    assert len(first["raw_scenes"]) == 2 * len(ACTION_PROBE_LABELS)
    assert {row["action_probe"] for row in first["raw_scenes"]} == set(
        ACTION_PROBE_LABELS
    )
    for scene in first["raw_scenes"]:
        assert set((
            "comm_rate", "min_sensing_rate", "P_ai", "P_ai_envelope",
            "cascade_power",
        )) <= set(scene)
        assert scene["P_ai"] <= scene["P_ai_envelope"] + 1e-18
        assert np.isfinite(scene["comm_rate"])
    assert "episodes" not in first
    assert "training" not in first


def _write_figure_fixture(directory, figure, x, curves):
    payload = {
        "figure": figure,
        "x": x,
        "curves": curves,
        "metadata": {},
    }
    path = Path(directory) / f"fig{figure}_data.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _passing_cross_figure_fixtures(directory):
    _write_figure_fixture(directory, 4, [0, 1, 2], {
        "DTL-pCSI": [8.0, 9.0, 9.5],
        "DTL-WFB": [6.0, 8.0, 8.9],
        "DTL-iCSI": [7.0, 8.0, 8.4],
        "DDPG": [5.0, 6.0, 6.4],
    })
    _write_figure_fixture(directory, 5, [0, 1, 2], {
        "L=30": [1.0, 2.0, 3.0],
        "L=20": [0.8, 1.6, 2.5],
        "L=10": [0.5, 1.0, 2.0],
    })
    _write_figure_fixture(directory, 6, [10, 20, 30], {
        "DTL-pCSI": [1.0, 2.0, 3.0], "DDPG": [0.5, 1.0, 1.5],
        "AO-pCSI": [0.8, 1.4, 2.0], "DTL-iCSI": [0.7, 1.3, 1.9],
        "AO-iCSI": [0.6, 1.2, 1.8], "DTL-ParCSI": [0.4, 1.0, 1.6],
    })
    _write_figure_fixture(directory, 8, [0.0, 0.5, 1.0], {
        "HRIS, L=40": [4.0, 3.0, 2.0], "HRIS, L=20": [3.5, 2.5, 1.5],
        "Passive RIS, L=40": [3.0, 2.0, 1.0],
        "Passive RIS, L=20": [2.5, 1.5, 0.5],
    })
    _write_figure_fixture(directory, 7, [20, 40, 60], {
        "HRIS-pCSI": [3.0, 4.0, 5.0], "HRIS-iCSI": [2.5, 3.5, 4.5],
        "HRIS-ParCSI": [2.0, 3.0, 4.0], "Passive RIS": [1.5, 2.0, 2.5],
        "Random Phase": [1.2, 1.5, 1.8], "No-RIS": [1.0, 1.0, 1.0],
    })
    _write_figure_fixture(directory, 9, [40, 30], {
        "nu=0.1": [4.0, 3.5], "nu=0.3": [3.0, 2.5], "nu=0.5": [2.0, 1.5],
    })
    _write_figure_fixture(directory, 13, [0, 10, 20], {
        "R_u, L=40": [5.0, 4.0, 3.0], "R_u, L=30": [4.5, 3.5, 2.5],
        "R_u, L=20": [4.0, 3.0, 2.0], "R_k, L=30": [1.0, 2.0, 3.0],
        "R_k, L=20": [0.8, 1.8, 2.8],
    })
    _write_figure_fixture(directory, 14, [0, 10, 20], {
        "DTL, L=30": [5.0, 4.0, 3.0], "DTL, L=20": [4.5, 3.5, 2.5],
        "DDPG, L=30": [4.0, 3.0, 2.0], "DDPG, L=20": [3.5, 2.5, 1.5],
    })


def test_cross_figure_validation_separates_calibration_and_holdout_checks():
    from cross_figure_validation import (
        CALIBRATION_FIGURES,
        HOLDOUT_FIGURES,
        validate_result_directory,
    )

    assert set(CALIBRATION_FIGURES).isdisjoint(HOLDOUT_FIGURES)
    with tempfile.TemporaryDirectory() as directory:
        _passing_cross_figure_fixtures(directory)
        report = validate_result_directory(directory)
    assert report["overall_pass"] is True
    assert set(report["calibration"]) == set(CALIBRATION_FIGURES)
    assert set(report["holdout"]) == set(HOLDOUT_FIGURES)


def test_cross_figure_validation_reports_heldout_failure_without_curve_changes():
    from cross_figure_validation import validate_result_directory

    with tempfile.TemporaryDirectory() as directory:
        _passing_cross_figure_fixtures(directory)
        path = Path(directory) / "fig13_data.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        bad_curve = [3.0, 4.0, 5.0]
        payload["curves"]["R_u, L=40"] = bad_curve
        path.write_text(json.dumps(payload), encoding="utf-8")
        report = validate_result_directory(directory)
        persisted = json.loads(path.read_text(encoding="utf-8"))
    assert report["overall_pass"] is False
    assert report["holdout"][13]["passed"] is False
    assert persisted["curves"]["R_u, L=40"] == bad_curve


def test_cross_figure_validation_rejects_nonfinite_generated_values():
    from cross_figure_validation import validate_figure_payload

    report = validate_figure_payload(6, {
        "figure": 6,
        "x": [10, 20],
        "curves": {
            "DTL-pCSI": [1.0, float("nan")], "DDPG": [1.0, 2.0],
            "AO-pCSI": [1.0, 2.0], "DTL-iCSI": [1.0, 2.0],
            "AO-iCSI": [1.0, 2.0], "DTL-ParCSI": [1.0, 2.0],
        },
    })
    assert report["passed"] is False
    assert "non-finite" in " ".join(report["failures"]).lower()


def test_applied_profile_exposes_and_restores_runtime_provenance():
    from parameter_profiles import (
        active_profile_metadata,
        applied_profile,
        profile_from_mapping,
    )

    profile = profile_from_mapping(_example_profile())
    before = active_profile_metadata()
    with applied_profile(profile):
        active = active_profile_metadata()
        assert active["digest"] == profile.digest
        assert active["name"] == profile.name
        assert active["kind"] == profile.kind
        assert active["paper_curve_ordinates_used"] is False
    assert active_profile_metadata() == before


def test_figure_data_manifest_records_active_profile_digest():
    import simulation
    from parameter_profiles import applied_profile, profile_from_mapping

    profile = profile_from_mapping(_example_profile())
    old_out = simulation.OUT
    try:
        with tempfile.TemporaryDirectory() as directory:
            simulation.OUT = directory
            with applied_profile(profile):
                paths = simulation._save_figure_data(
                    5,
                    [0, 1],
                    {"L=30": [1.0, 2.0], "L=20": [0.8, 1.8],
                     "L=10": [0.6, 1.6]},
                )
            manifest = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        assert manifest["parameter_profile"]["digest"] == profile.digest
        assert manifest["parameter_profile"]["source_path"] is None
    finally:
        simulation.OUT = old_out


def test_runners_activate_profile_before_importing_simulation():
    root = Path(__file__).parent
    for filename in ("main.py", "run_all_figs.py"):
        source = (root / filename).read_text(encoding="utf-8-sig")
        assert "--profile" in source
        assert "load_calibrated_profile" in source and "activate_profile" in source
        assert source.index("activate_profile") < source.index("from simulation import")


def test_full_runners_require_an_explicit_validated_profile():
    root = Path(__file__).parent
    main_source = (root / "main.py").read_text(encoding="utf-8-sig")
    run_all_source = (root / "run_all_figs.py").read_text(encoding="utf-8-sig")

    assert "if not ARGS.profile" in main_source
    assert "if not ARGS.profile" in run_all_source
    assert "A validated --profile is required" in main_source
    assert "A validated --profile is required" in run_all_source


def test_full_runners_preflight_before_figures_or_training():
    root = Path(__file__).parent
    for filename in ("main.py", "run_all_figs.py"):
        source = (root / filename).read_text(encoding="utf-8-sig")
        assert "run_scientific_preflight" in source
        assert source.index("run_scientific_preflight(") < source.rindex(
            "fig4_rewards_vs_episodes("
        )


def test_dynamic_preflight_actions_cover_gain_boundaries_deterministically():
    from scientific_preflight import (
        PREFLIGHT_ACTION_MODES,
        build_preflight_action,
    )

    first = [build_preflight_action(501, index) for index in range(4)]
    second = [build_preflight_action(501, index) for index in range(4)]
    assert [mode for mode, _ in first] == list(PREFLIGHT_ACTION_MODES)
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(first, second))
    gain_slice = slice(cfg.Lp + cfg.La, cfg.Lp + 2 * cfg.La)
    assert np.all(first[0][1][gain_slice] == 0.0)
    assert np.all(first[1][1][gain_slice] == -1.0)
    assert np.all(first[2][1][gain_slice] == 1.0)


def test_full_runners_write_cross_figure_validation_report():
    root = Path(__file__).parent
    for filename in ("main.py", "run_all_figs.py"):
        source = (root / filename).read_text(encoding="utf-8-sig")
        assert "validate_result_directory" in source
        assert "cross_figure_validation.json" in source
    main_source = (root / "main.py").read_text(encoding="utf-8-sig")
    assert "if not FAST:" in main_source


def _run():
    tests = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - report every contract failure.
            failures.append((test.__name__, exc))
            print(f"[FAIL] {test.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"\nResult: {len(tests) - len(failures)}/{len(tests)} tests passed")
        return 1
    print(f"\nResult: {len(tests)}/{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run())
