import json
import tempfile
import unittest
from pathlib import Path

from hypothesis_selection import (
    score_power_hypothesis,
    select_power_hypothesis,
    write_selection_lock,
)


def reference_fixture():
    return {
        "schema_version": 1,
        "calibration_figures": [6],
        "holdout_figures": [4],
        "figures": {
            "6": {
                "kind": "digitized_reference",
                "source_page": 11,
                "role": "calibration",
                "x_axis": {"label": "power", "units": "paper dB"},
                "y_axis": {"label": "rate", "units": "bits/s/Hz"},
                "digitization_method": "fixture",
                "uncertainty": {"absolute_y": 0.1},
                "x": [10.0, 20.0, 30.0],
                "curves": {
                    "AO-pCSI": [7.0, 8.0, 9.0],
                    "AO-iCSI": [6.0, 7.0, 8.0],
                },
            },
            "4": {
                "kind": "digitized_reference",
                "source_page": 11,
                "role": "holdout",
                "x_axis": {"label": "episode", "units": "episode"},
                "y_axis": {"label": "reward", "units": "bits/s/Hz"},
                "digitization_method": "fixture",
                "uncertainty": {"absolute_y": 0.1},
                "x": [0.0],
                "curves": {"DTL-pCSI": [9.5]},
            },
        },
    }


def candidate(identifier, offset=0.0, feasible=True):
    return {
        "hypothesis_id": identifier,
        "calibration_figure": 6,
        "holdout_figures_used": [],
        "common_random_seeds": [1, 2, 3, 4, 5],
        "all_points_feasible": feasible,
        "predictions": {
            "6": {
                "AO-pCSI": [7.0 + offset, 8.0 + offset, 9.0 + offset],
                "AO-iCSI": [6.0 + offset, 7.0 + offset, 8.0 + offset],
            }
        },
    }


class HypothesisSelectionTests(unittest.TestCase):
    def test_exact_fig6_prediction_has_zero_errors(self):
        score = score_power_hypothesis(
            candidate("paper_db_as_dbm"), reference_fixture()
        )
        self.assertEqual(score["ordering_violations"], 0)
        self.assertEqual(score["normalized_curve_error"], 0.0)
        self.assertEqual(score["normalized_slope_error"], 0.0)
        self.assertTrue(score["fully_feasible"])

    def test_holdout_usage_is_rejected(self):
        bad = candidate("paper_db_as_dbm")
        bad["holdout_figures_used"] = [4]
        with self.assertRaisesRegex(ValueError, "holdout"):
            score_power_hypothesis(bad, reference_fixture())

    def test_selection_rejects_infeasible_even_when_curve_error_is_lower(self):
        exact_but_infeasible = candidate("paper_db_as_dbw", feasible=False)
        feasible = candidate("paper_db_as_dbm", offset=0.2, feasible=True)
        lock = select_power_hypothesis(
            [exact_but_infeasible, feasible], reference_fixture()
        )
        self.assertEqual(lock["selected_hypothesis_id"], "paper_db_as_dbm")
        self.assertEqual(lock["holdout_figures_used"], [])
        self.assertRegex(lock["paper_reference_sha256"], r"^[0-9a-f]{64}$")

    def test_selection_is_not_based_on_larger_reward(self):
        close = candidate("paper_db_as_dbm", offset=0.1)
        much_larger = candidate("paper_db_as_dbw", offset=10.0)
        lock = select_power_hypothesis([much_larger, close], reference_fixture())
        self.assertEqual(lock["selected_hypothesis_id"], "paper_db_as_dbm")

    def test_both_predeclared_hypotheses_are_required(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            select_power_hypothesis(
                [candidate("paper_db_as_dbm")], reference_fixture()
            )

    def test_lock_cannot_be_overwritten_with_different_selection(self):
        lock = select_power_hypothesis(
            [
                candidate("paper_db_as_dbm"),
                candidate("paper_db_as_dbw", offset=0.2),
            ],
            reference_fixture(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "power_hypothesis_lock.json"
            write_selection_lock(path, lock)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["locked"], True
            )
            changed = dict(lock)
            changed["selected_hypothesis_id"] = "paper_db_as_dbw"
            with self.assertRaisesRegex(ValueError, "immutable"):
                write_selection_lock(path, changed)


if __name__ == "__main__":
    unittest.main()
