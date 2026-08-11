import inspect
import unittest

import numpy as np

import conservative_gate
from conservative_gate import select_residual_scale


class ConservativeGateTests(unittest.TestCase):
    def test_selects_candidate_only_when_lcb_clears_baseline_and_margin(self):
        values = np.array(
            [
                [2.0, 2.2, 2.9],
                [2.0, 2.1, 2.8],
                [2.0, 2.0, 3.0],
            ]
        )
        decision = select_residual_scale(
            scales=[0.0, 0.5, 1.0],
            ensemble_values=values,
            uncertainty_coefficient=1.0,
            margin=0.1,
        )
        self.assertEqual(decision.scale, 1.0)
        self.assertEqual(decision.selected_index, 2)
        self.assertIsNone(decision.fallback_reason)
        self.assertGreater(decision.selected_lcb, decision.baseline_value + 0.1)

    def test_tie_or_insufficient_margin_returns_baseline(self):
        values = np.array([[2.0, 2.1], [2.0, 1.9]])
        decision = select_residual_scale(
            [0.0, 1.0], values, uncertainty_coefficient=0.0, margin=0.0
        )
        self.assertEqual(decision.scale, 0.0)
        self.assertEqual(decision.fallback_reason, "insufficient_conservative_advantage")

    def test_nonfinite_values_return_documented_baseline_fallback(self):
        decision = select_residual_scale(
            [0.0, 1.0], [[2.0, np.nan], [2.0, 3.0]], 1.0, 0.0
        )
        self.assertEqual(decision.scale, 0.0)
        self.assertEqual(decision.selected_index, 0)
        self.assertEqual(decision.fallback_reason, "nonfinite_ensemble_values")

    def test_contract_rejects_missing_zero_scale_and_bad_shapes(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            select_residual_scale([0.5, 1.0], [[1.0, 2.0]], 1.0, 0.0)
        with self.assertRaises(ValueError):
            select_residual_scale([0.0, 1.0], [[1.0]], 1.0, 0.0)
        with self.assertRaises(ValueError):
            select_residual_scale([0.0, 1.0], [[1.0, 2.0]], -1.0, 0.0)
        with self.assertRaises(ValueError):
            select_residual_scale([0.0, 1.0], [[1.0, 2.0]], 1.0, -0.1)

    def test_gate_is_environment_independent(self):
        source = inspect.getsource(conservative_gate).lower()
        forbidden = ("environment", "channel_model", "paper_reference", "result.zip")
        for name in forbidden:
            with self.subTest(name=name):
                self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
