import inspect
import json
import tempfile
import unittest
from pathlib import Path

from calibrate_reproduction import (
    CalibrationResult,
    freeze_calibrated_profile,
    with_fixed_sensitivity_values,
    write_calibration_report,
)
from parameter_profiles import load_calibrated_profile, profile_from_mapping


class CalibratedProfileAuthorityTests(unittest.TestCase):
    @staticmethod
    def _base_profile():
        return profile_from_mapping({
            "schema_version": 1,
            "name": "prepared-base",
            "kind": "identified",
            "values": {"omega_0": 1e-3},
            "evidence": {"omega_0": "omega_0"},
            "selection": {"purpose": "test fixture"},
        })

    def _write_pair(self, directory, fixed_sensitivity_values=None):
        directory = Path(directory)
        reference_digest = "f" * 64
        fixed_sensitivity_values = dict(fixed_sensitivity_values or {})
        result = CalibrationResult(
            selected_values={"omega_0": 2e-3},
            loss=0.01,
            records=[{
                "candidate_index": 0,
                "values": {"omega_0": 2e-3},
                "predictions": {
                    "6": {"AO-pCSI": [7.0], "AO-iCSI": [6.5]}
                },
                "valid": True,
                "error": None,
                "loss": {
                    "total": 0.01,
                    "reference_total": 0.01,
                    "prior": 0.0,
                    "components": {
                        "6/AO-pCSI": 0.01,
                        "6/AO-iCSI": 0.01,
                    },
                },
            }],
            optimizer_success=True,
            optimizer_message="converged",
        )
        profile_path = directory / "calibrated_profile.json"
        report_path = directory / "calibration_report.json"
        base = with_fixed_sensitivity_values(
            self._base_profile(), fixed_sensitivity_values
        )
        profile = freeze_calibrated_profile(
            base, result, profile_path,
            paper_reference_sha256=reference_digest,
            fixed_sensitivity_values=fixed_sensitivity_values,
        )
        write_calibration_report(
            result,
            report_path,
            profile_digest=profile.digest,
            common_random_seeds=(21000, 21001, 21002, 21003, 21004),
            ao_iterations=6,
            optimizer_max_iterations=30,
            paper_reference_sha256=reference_digest,
            fixed_sensitivity_values=fixed_sensitivity_values,
        )
        lock = {
            "schema_version": 1,
            "kind": "power_hypothesis_selection_lock",
            "locked": True,
            "calibration_figures_used": [6],
            "holdout_figures_used": [],
            "selected_hypothesis_id": "paper_db_as_dbm",
            "selected_profile": "calibrated_profile.json",
            "paper_reference_sha256": reference_digest,
            "common_random_seeds": [21000, 21001, 21002, 21003, 21004],
            "candidate_scores": [
                {
                    "hypothesis_id": "paper_db_as_dbm",
                    "fully_feasible": True,
                    "calibration_admissible": True,
                    "ordering_violations": 0,
                    "power_direction_violations": 0,
                },
                {
                    "hypothesis_id": "paper_db_as_dbw",
                    "fully_feasible": True,
                    "calibration_admissible": False,
                    "ordering_violations": 0,
                    "power_direction_violations": 0,
                },
            ],
        }
        (directory / "power_hypothesis_lock.json").write_text(
            json.dumps(lock), encoding="utf-8"
        )
        return profile_path, report_path, profile

    def test_matching_calibrated_profile_and_report_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path, _report_path, expected = self._write_pair(directory)
            loaded = load_calibrated_profile(profile_path)
            self.assertEqual(loaded.digest, expected.digest)
            self.assertEqual(
                loaded.selection["calibration_noise_hypothesis_id"],
                "description_literal_target_matches_ue",
            )

    def test_fixed_sensitivity_authority_is_accepted_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path, report_path, expected = self._write_pair(
                directory, {"fc": 2.4e9}
            )
            loaded = load_calibrated_profile(profile_path)
            self.assertEqual(loaded.digest, expected.digest)
            self.assertEqual(
                loaded.selection["fixed_sensitivity_protocol"],
                "predeclared_not_result_selected",
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["fixed_sensitivity_values"]["fc"] = 3e9
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed sensitivities"):
                load_calibrated_profile(profile_path)

    def test_missing_or_tampered_calibration_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path, report_path, _profile = self._write_pair(directory)
            report_path.unlink()
            with self.assertRaisesRegex(ValueError, "calibration report"):
                load_calibrated_profile(profile_path)

        with tempfile.TemporaryDirectory() as directory:
            profile_path, report_path, _profile = self._write_pair(directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["profile_digest"] = "tampered"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_calibrated_profile(profile_path)

        with tempfile.TemporaryDirectory() as directory:
            profile_path, report_path, _profile = self._write_pair(directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            del report["common_random_seeds"]
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seeds"):
                load_calibrated_profile(profile_path)

        with tempfile.TemporaryDirectory() as directory:
            profile_path, report_path, _profile = self._write_pair(directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["records"][-1]["values"]["omega_0"] = 3e-3
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected candidate"):
                load_calibrated_profile(profile_path)

        with tempfile.TemporaryDirectory() as directory:
            profile_path, report_path, _profile = self._write_pair(directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["calibration_noise_hypothesis_id"] = (
                "dual_minus70_conservative"
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "noise hypothesis"):
                load_calibrated_profile(profile_path)

        with tempfile.TemporaryDirectory() as directory:
            profile_path, report_path, _profile = self._write_pair(directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["cross_function_interference_hypothesis_id"] = (
                "receiver_consistent_cross_channels"
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cross-function"):
                load_calibrated_profile(profile_path)

        with tempfile.TemporaryDirectory() as directory:
            profile_path, report_path, _profile = self._write_pair(directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["paper_reference_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "paper reference"):
                load_calibrated_profile(profile_path)

    def test_missing_power_hypothesis_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path, _report_path, _profile = self._write_pair(directory)
            (Path(directory) / "power_hypothesis_lock.json").unlink()
            with self.assertRaisesRegex(ValueError, "hypothesis lock"):
                load_calibrated_profile(profile_path)

    def test_dedicated_runners_use_calibrated_authority_loader(self):
        import calibrate_reproduction
        import run_baseline
        import run_extension

        self.assertIn(
            "load_prepared_profile", inspect.getsource(calibrate_reproduction.main)
        )
        self.assertIn("load_calibrated_profile", inspect.getsource(run_baseline.run))
        self.assertIn("load_calibrated_profile", inspect.getsource(run_extension.run))

        root = Path(__file__).parents[1]
        for filename in ("main.py", "run_all_figs.py"):
            source = (root / filename).read_text(encoding="utf-8-sig")
            self.assertIn("load_calibrated_profile", source)


if __name__ == "__main__":
    unittest.main()
