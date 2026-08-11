import unittest

import numpy as np

from research_objectives import (
    ObjectiveMode,
    ObjectiveSpec,
    evaluate_objective,
)


class ResearchObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.rate_u = np.array([2.0, 3.0])
        self.rate_k = np.array([1.0, 1.5])

    def test_paper_objective_is_communication_only(self):
        result = evaluate_objective(
            ObjectiveSpec.paper(target_rate_min=0.5), self.rate_u, self.rate_k
        )
        self.assertAlmostEqual(result.value, 5.0)
        self.assertAlmostEqual(result.communication_sum_rate, 5.0)
        self.assertAlmostEqual(result.sensing_sum_rate, 2.5)
        self.assertTrue(result.rate_constraints_feasible)

    def test_png_objective_is_unweighted_joint_sum(self):
        specification = ObjectiveSpec.png_joint(
            user_rate_min=1.0, target_rate_min=0.5
        )
        result = evaluate_objective(
            specification,
            self.rate_u,
            self.rate_k,
        )
        self.assertAlmostEqual(result.value, 7.5)
        self.assertEqual(result.mode, ObjectiveMode.PNG_JOINT_SUM)
        self.assertNotIn(
            "communication_retention_fraction", specification.as_dict()
        )
        self.assertNotIn("pareto_weight", specification.as_dict())

    def test_retention_mode_maximizes_sensing_and_reports_comm_deficit(self):
        result = evaluate_objective(
            ObjectiveSpec.communication_retention(
                communication_reference=6.0,
                communication_retention=0.95,
                target_rate_min=0.5,
            ),
            self.rate_u,
            self.rate_k,
        )
        self.assertAlmostEqual(result.value, 2.5)
        self.assertAlmostEqual(result.communication_required, 5.7)
        self.assertAlmostEqual(result.communication_deficit, 0.7)
        self.assertFalse(result.rate_constraints_feasible)

    def test_normalized_pareto_has_dimensionless_weighted_value(self):
        result = evaluate_objective(
            ObjectiveSpec.normalized_pareto(
                communication_reference=10.0,
                sensing_reference=5.0,
                pareto_weight=0.25,
                target_rate_min=0.5,
            ),
            self.rate_u,
            self.rate_k,
        )
        self.assertAlmostEqual(result.value, 0.5)

    def test_invalid_references_and_rates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "communication_reference"):
            ObjectiveSpec.communication_retention(0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_objective(ObjectiveSpec.paper(), [np.nan], [1.0])

    def test_per_node_rate_floors_are_reported_separately(self):
        result = evaluate_objective(
            ObjectiveSpec.png_joint(user_rate_min=2.5, target_rate_min=1.25),
            self.rate_u,
            self.rate_k,
        )
        np.testing.assert_allclose(result.user_rate_deficits, [0.5, 0.0])
        np.testing.assert_allclose(result.target_rate_deficits, [0.25, 0.0])
        self.assertFalse(result.rate_constraints_feasible)


if __name__ == "__main__":
    unittest.main()
