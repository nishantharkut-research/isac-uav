import copy
import unittest

import numpy as np

from pareto_protocol import (
    ParetoAnchors,
    holm_adjust,
    hypervolume_2d,
    nondominated_mask,
    paired_confidence_interval,
    scalarize_returns,
    validate_preferences,
)


class PreferenceTests(unittest.TestCase):
    def test_valid_preferences_and_exact_png_scalarization(self):
        preferences = validate_preferences([[0.5, 0.5], [1.0, 0.0]])
        self.assertEqual(preferences.shape, (2, 2))
        q_values = np.array([[2.0, 5.0], [7.0, 11.0]])
        np.testing.assert_array_equal(
            scalarize_returns(q_values, [0.5, 0.5], mode="png_joint_sum"),
            [7.0, 18.0],
        )

    def test_png_scalarization_rejects_nonbalanced_preference(self):
        with self.assertRaisesRegex(ValueError, "balanced"):
            scalarize_returns([[2.0, 5.0]], [0.25, 0.75], mode="png_joint_sum")

    def test_invalid_preferences_are_rejected(self):
        invalid = (
            [[0.2, 0.2]], [[-0.1, 1.1]], [[np.nan, 0.0]], [[1.0]],
            [[0.1, 0.2, 0.7]],
        )
        for preferences in invalid:
            with self.subTest(preferences=preferences):
                with self.assertRaises(ValueError):
                    validate_preferences(preferences)

    def test_chebyshev_uses_frozen_anchors_and_does_not_clip(self):
        anchors = ParetoAnchors()
        anchors.update_training([[1.0, 10.0], [3.0, 14.0]])
        anchors.freeze()
        values = np.array([[5.0, 18.0]])
        normalized = anchors.normalize(values)
        np.testing.assert_allclose(normalized, [[2.0, 2.0]])
        utility = scalarize_returns(
            values, [0.25, 0.75], mode="chebyshev", anchors=anchors
        )
        self.assertTrue(np.all(np.isfinite(utility)))


class AnchorTests(unittest.TestCase):
    def test_anchor_round_trip_and_freeze_are_immutable(self):
        anchors = ParetoAnchors()
        anchors.update_training([[1.0, 10.0], [3.0, 14.0]])
        anchors.freeze(provenance={"partition": "training", "seed_digest": "abc"})
        state = anchors.state_dict()
        restored = ParetoAnchors.from_state_dict(state)
        np.testing.assert_array_equal(restored.ideal, [3.0, 14.0])
        np.testing.assert_array_equal(restored.nadir, [1.0, 10.0])
        self.assertEqual(restored.provenance["partition"], "training")
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            restored.update_training([[0.0, 0.0]])

    def test_freeze_rejects_missing_or_zero_span(self):
        with self.assertRaises(ValueError):
            ParetoAnchors().freeze()
        anchors = ParetoAnchors()
        anchors.update_training([[1.0, 2.0], [1.0, 3.0]])
        with self.assertRaisesRegex(ValueError, "span"):
            anchors.freeze()

    def test_restore_rejects_tampered_anchor_state(self):
        anchors = ParetoAnchors()
        anchors.update_training([[1.0, 2.0], [2.0, 3.0]])
        anchors.freeze()
        state = copy.deepcopy(anchors.state_dict())
        state["nadir"] = [3.0, 4.0]
        with self.assertRaises(ValueError):
            ParetoAnchors.from_state_dict(state)

    def test_quantile_anchors_reject_outliers_and_have_stable_identity(self):
        typical = np.column_stack([
            np.linspace(1.0, 3.0, 100),
            np.linspace(10.0, 14.0, 100),
        ])
        values = np.vstack([typical, [-1_000.0, 1_000.0], [1_000.0, -1_000.0]])
        anchors = ParetoAnchors()
        anchors.update_training(values, lower_quantile=0.05, upper_quantile=0.95)
        anchors.freeze({"partition": "converged_training_tail"})

        self.assertGreater(anchors.nadir[0], -1_000.0)
        self.assertGreater(anchors.nadir[1], -1_000.0)
        self.assertLess(anchors.ideal[0], 1_000.0)
        self.assertLess(anchors.ideal[1], 1_000.0)
        restored = ParetoAnchors.from_state_dict(anchors.state_dict())
        self.assertEqual(restored.digest(), anchors.digest())

        altered = ParetoAnchors.from_state_dict(anchors.state_dict())
        altered.provenance["partition"] = "different"
        self.assertNotEqual(altered.digest(), anchors.digest())


class ParetoAndStatisticsTests(unittest.TestCase):
    def test_nondominance_and_known_hypervolume(self):
        points = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        np.testing.assert_array_equal(nondominated_mask(points), [False, True, True])
        self.assertAlmostEqual(
            hypervolume_2d([[1.0, 2.0], [2.0, 1.0]], reference=[0.0, 0.0]),
            3.0,
        )

    def test_hypervolume_rejects_invalid_or_below_reference_points(self):
        with self.assertRaises(ValueError):
            hypervolume_2d([[1.0, np.nan]], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "reference"):
            hypervolume_2d([[-1.0, 1.0]], [0.0, 0.0])

    def test_paired_interval_and_holm_adjustment(self):
        candidate = np.array([2.0, 3.0, 4.0, 5.0])
        baseline = np.array([1.0, 2.0, 3.0, 4.0])
        interval = paired_confidence_interval(candidate, baseline, confidence=0.95)
        self.assertEqual(interval.n, 4)
        self.assertAlmostEqual(interval.mean_difference, 1.0)
        self.assertAlmostEqual(interval.lower, 1.0)
        self.assertAlmostEqual(interval.upper, 1.0)
        adjusted = holm_adjust([0.01, 0.03, 0.20])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.20])

    def test_paired_interval_rejects_unpaired_or_too_few_values(self):
        with self.assertRaises(ValueError):
            paired_confidence_interval([1.0], [0.0])
        with self.assertRaisesRegex(ValueError, "paired"):
            paired_confidence_interval([1.0, 2.0], [0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
