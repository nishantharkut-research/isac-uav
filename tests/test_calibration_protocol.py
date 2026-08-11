import unittest
from unittest.mock import patch

from calibrate_reproduction import (
    CALIBRATION_TARGETS,
    HOLDOUT_FIGURES,
    calibration_loss,
    hypothesis_candidate_from_calibration,
    make_ao_calibration_predictor,
    parse_fixed_sensitivity_values,
    run_bounded_calibration,
    validate_calibration_keys,
    with_fixed_sensitivity_values,
)
from parameter_profiles import profile_from_mapping
from ambiguity_hypotheses import apply_ambiguity_hypotheses
from config import cfg


class CalibrationProtocolTests(unittest.TestCase):
    def test_fixed_sensitivity_parser_accepts_only_sensitivity_parameters(self):
        values = parse_fixed_sensitivity_values([
            "fc=2.4e9",
            "d_spacing=0.5",
        ])
        self.assertEqual(values, {"fc": 2.4e9, "d_spacing": 0.5})

        for item in ("M=8", "PT_dB=20", "omega_0=0.001"):
            with self.subTest(item=item):
                with self.assertRaisesRegex(ValueError, "sensitivity-only"):
                    parse_fixed_sensitivity_values([item])

    def test_fixed_sensitivity_parser_rejects_malformed_duplicate_and_bounds(self):
        invalid = (
            ["fc"],
            ["fc=not-a-number"],
            ["fc=2.4e9", "fc=3e9"],
            ["d_spacing=2.0"],
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    parse_fixed_sensitivity_values(values)

    def test_fixed_sensitivity_values_are_predeclared_not_result_selected(self):
        base = profile_from_mapping({
            "schema_version": 1,
            "name": "prepared-base",
            "kind": "identified",
            "values": {"omega_0": 1e-3, "fc": 3e9},
            "evidence": {"omega_0": "omega_0", "fc": "fc"},
            "selection": {"protocol": "structural preparation"},
        })
        derived = with_fixed_sensitivity_values(
            base, {"fc": 2.4e9, "d_spacing": 0.5}
        )
        self.assertEqual(derived.values["fc"], 2.4e9)
        self.assertEqual(derived.values["d_spacing"], 0.5)
        self.assertEqual(
            derived.selection["fixed_sensitivity_protocol"],
            "predeclared_not_result_selected",
        )
        self.assertEqual(
            derived.selection["fixed_sensitivity_values"],
            {"fc": 2.4e9, "d_spacing": 0.5},
        )
        self.assertEqual(derived.selection["source_profile_digest"], base.digest)

    def test_calibration_result_becomes_fig6_only_hypothesis_candidate(self):
        from calibrate_reproduction import CalibrationResult

        result = CalibrationResult(
            selected_values={"omega_0": 1e-3},
            loss=0.1,
            records=[{
                "valid": True,
                "predictions": {
                    "6": {"AO-pCSI": [7.0, 8.0], "AO-iCSI": [6.0, 7.0]}
                },
            }],
            optimizer_success=True,
            optimizer_message="ok",
        )
        candidate = hypothesis_candidate_from_calibration(
            result, "paper_db_as_dbm", (1, 2, 3, 4, 5)
        )
        self.assertEqual(candidate["calibration_figure"], 6)
        self.assertEqual(candidate["holdout_figures_used"], [])
        self.assertTrue(candidate["all_points_feasible"])
        self.assertEqual(set(candidate["predictions"]), {"6"})

    def test_reported_parameters_cannot_be_calibrated(self):
        for key in ("M", "N", "U", "K", "L", "PT_dB", "sigma2_u_dBm"):
            with self.assertRaisesRegex(ValueError, "journal-reported"):
                validate_calibration_keys([key])

    def test_holdout_figures_are_not_calibration_targets(self):
        self.assertEqual(
            HOLDOUT_FIGURES,
            frozenset({4, 5, 7, 8, 9, 13, 14}),
        )
        self.assertTrue(set(CALIBRATION_TARGETS).isdisjoint(HOLDOUT_FIGURES))
        predictions = {
            "4": {"DTL-pCSI": [9.5]},
        }
        references = {
            "schema_version": 1,
            "figures": {
                "4": {
                    "uncertainty": {"absolute_y": 0.1},
                    "curves": {"DTL-pCSI": [9.5]},
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "holdout"):
            calibration_loss(predictions, references, targets={4: ("DTL-pCSI",)})

        for figure in (5, 9, 14):
            with self.subTest(figure=figure):
                with self.assertRaisesRegex(ValueError, "holdout"):
                    calibration_loss({}, {}, targets={figure: ("curve",)})

    def test_relative_loss_is_zero_for_exact_calibration_predictions(self):
        references = {
            "schema_version": 1,
            "figures": {
                "6": {
                    "uncertainty": {"absolute_y": 0.1},
                    "curves": {
                        "AO-pCSI": [7.0, 8.0],
                        "AO-iCSI": [6.5, 7.5],
                    },
                },
            },
        }
        predictions = {
            "6": {
                "AO-pCSI": [7.0, 8.0],
                "AO-iCSI": [6.5, 7.5],
            },
        }
        result = calibration_loss(predictions, references)
        self.assertEqual(result["total"], 0.0)
        self.assertEqual(set(result["components"]), {"6/AO-pCSI", "6/AO-iCSI"})

    def test_ao_predictor_uses_only_fig6_ao_curves_and_both_csi_conditions(self):
        profile = profile_from_mapping({
            "schema_version": 1,
            "name": "minimal-calibration-base",
            "kind": "identified",
            "values": {"omega_0": 1e-3},
            "evidence": {"omega_0": "omega_0"},
            "selection": {},
        })
        seen_sigma2 = []

        def fake_run_ao(*args, **kwargs):
            seen_sigma2.append(float(kwargs["sigma2"]))
            return {"sum_rate": 7.0, "is_feasible": True}

        with patch("ao_optimizer.run_ao", side_effect=fake_run_ao):
            predictor = make_ao_calibration_predictor(
                profile, seeds=(1,), ao_iterations=1
            )
            predictions = predictor({"omega_0": 1e-3})

        self.assertEqual(set(predictions), {"6"})
        self.assertEqual(set(predictions["6"]), {"AO-pCSI", "AO-iCSI"})
        self.assertEqual(len(seen_sigma2), 10)
        self.assertTrue(all(value == 0.0 for value in seen_sigma2[:5]))
        self.assertTrue(all(value > 0.0 for value in seen_sigma2[5:]))

    def test_ao_predictor_rejects_infeasible_solver_result(self):
        profile = profile_from_mapping({
            "schema_version": 1,
            "name": "minimal-calibration-base",
            "kind": "identified",
            "values": {"omega_0": 1e-3},
            "evidence": {"omega_0": "omega_0"},
            "selection": {},
        })
        with patch(
            "ao_optimizer.run_ao",
            return_value={"sum_rate": 7.0, "is_feasible": False},
        ):
            predictor = make_ao_calibration_predictor(
                profile, seeds=(1,), ao_iterations=1
            )
            with self.assertRaisesRegex(RuntimeError, "infeasible"):
                predictor({"omega_0": 1e-3})

    def test_ao_predictor_applies_the_requested_power_hypothesis(self):
        profile = profile_from_mapping({
            "schema_version": 1,
            "name": "minimal-calibration-base",
            "kind": "identified",
            "values": {"omega_0": 1e-3},
            "evidence": {"omega_0": "omega_0"},
            "selection": {},
        })
        seen_power = []

        def fake_run_ao(*args, **kwargs):
            from config import cfg
            seen_power.append(float(cfg.PT))
            return {"sum_rate": 7.0, "is_feasible": True}

        with patch("ao_optimizer.run_ao", side_effect=fake_run_ao):
            predictor = make_ao_calibration_predictor(
                profile,
                seeds=(1,),
                ao_iterations=1,
                power_hypothesis_id="paper_db_as_dbm",
            )
            predictor({"omega_0": 1e-3})

        self.assertAlmostEqual(seen_power[0], 0.01)
        self.assertAlmostEqual(seen_power[4], 1.0)

    def test_ao_calibration_locks_primary_noise_and_restores_caller_config(self):
        profile = profile_from_mapping({
            "schema_version": 1,
            "name": "noise-calibration-base",
            "kind": "identified",
            "values": {"omega_0": 1e-3},
            "evidence": {"omega_0": "omega_0"},
            "selection": {},
        })
        apply_ambiguity_hypotheses(
            cfg, noise_hypothesis_id="dual_minus70_conservative"
        )
        seen = []

        def fake_run_ao(*args, **kwargs):
            seen.append((cfg.noise_hypothesis_id, cfg.sigma2_k_dBm))
            return {"sum_rate": 7.0, "is_feasible": True}

        try:
            with patch("ao_optimizer.run_ao", side_effect=fake_run_ao):
                predictor = make_ao_calibration_predictor(
                    profile, seeds=(1,), ao_iterations=1
                )
                predictor({"omega_0": 1e-3})
            self.assertTrue(seen)
            self.assertTrue(all(item == (
                "description_literal_target_matches_ue", -94.0
            ) for item in seen))
            self.assertEqual(cfg.noise_hypothesis_id,
                             "dual_minus70_conservative")
        finally:
            apply_ambiguity_hypotheses(cfg)

    def test_failed_predictor_cannot_authorize_calibration(self):
        references = {
            "schema_version": 1,
            "calibration_figures": [6],
            "holdout_figures": [],
            "figures": {
                "6": {
                    "kind": "digitized_reference",
                    "source_page": 11,
                    "x_axis": {"label": "power", "units": "dB"},
                    "y_axis": {"label": "rate", "units": "bits/s/Hz"},
                    "digitization_method": "test fixture",
                    "uncertainty": {"absolute_y": 0.1},
                    "x": [10.0],
                    "curves": {"AO-pCSI": [7.0], "AO-iCSI": [6.5]},
                }
            },
        }

        def failed_predictor(_values):
            raise RuntimeError("infeasible AO candidate")

        result = run_bounded_calibration(
            failed_predictor,
            {"omega_0": 1e-3},
            references,
            keys=("omega_0",),
            max_iterations=1,
        )
        self.assertFalse(result.optimizer_success)
        self.assertTrue(result.records)
        self.assertTrue(all(not record["valid"] for record in result.records))

    def test_calibration_records_prior_regularization_separately(self):
        references = {
            "schema_version": 1,
            "calibration_figures": [6],
            "holdout_figures": [],
            "figures": {
                "6": {
                    "kind": "digitized_reference",
                    "source_page": 11,
                    "x_axis": {"label": "power", "units": "dB"},
                    "y_axis": {"label": "rate", "units": "bits/s/Hz"},
                    "digitization_method": "test fixture",
                    "uncertainty": {"absolute_y": 0.1},
                    "x": [10.0],
                    "curves": {"AO-pCSI": [7.0], "AO-iCSI": [6.5]},
                }
            },
        }

        def exact_predictor(_values):
            return {"6": {"AO-pCSI": [7.0], "AO-iCSI": [6.5]}}

        result = run_bounded_calibration(
            exact_predictor,
            {"omega_0": 1e-3},
            references,
            keys=("omega_0",),
            max_iterations=20,
            prior_weight=0.05,
        )
        self.assertTrue(result.optimizer_success)
        self.assertEqual(result.prior_weight, 0.05)
        self.assertTrue(all(record["valid"] for record in result.records))
        self.assertTrue(all("reference_total" in record["loss"] for record in result.records))
        self.assertTrue(all("prior" in record["loss"] for record in result.records))


if __name__ == "__main__":
    unittest.main()
