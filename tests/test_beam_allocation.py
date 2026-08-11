import unittest

import numpy as np

from beam_allocation import (
    AllocationConstraints,
    optimize_power_allocation,
    select_fast_power_allocation,
)
from research_objectives import ObjectiveSpec


class BeamAllocationTests(unittest.TestCase):
    @staticmethod
    def _evaluate(powers):
        powers = np.asarray(powers, dtype=float)
        communication = powers[:2]
        sensing = powers[2:]
        sensing_interference = 0.05 * float(np.sum(sensing))
        communication_interference = 0.03 * float(np.sum(communication))
        rate_u = np.log2(1.0 + np.array([2.0, 1.5]) * communication /
                         (1.0 + sensing_interference))
        rate_k = np.log2(1.0 + np.array([1.2, 1.0]) * sensing /
                         (1.0 + communication_interference))
        return {
            "rate_u": rate_u,
            "rate_k": rate_k,
            "sensing_only_sinr": np.array([1.2, 1.0]) * sensing,
            "active_power": 0.2 * float(np.sum(sensing)),
        }

    def test_fast_allocation_is_power_bounded_and_feasible(self):
        constraints = AllocationConstraints(
            total_power=10.0,
            target_rate_min=0.5,
            user_rate_min=0.0,
            sensing_only_sinr_min=0.0,
            active_power_max=2.0,
        )
        result = select_fast_power_allocation(
            user_count=2,
            target_count=2,
            evaluate=self._evaluate,
            constraints=constraints,
            objective_spec=ObjectiveSpec.paper(target_rate_min=0.5),
        )
        self.assertTrue(result.feasible)
        self.assertTrue(np.all(result.powers >= 0.0))
        self.assertLessEqual(float(np.sum(result.powers)), 10.0 + 1e-9)
        self.assertTrue(np.all(result.rate_k >= 0.5 - 1e-7))

    def test_scipy_refinement_is_not_worse_than_fast_candidate(self):
        constraints = AllocationConstraints(
            total_power=10.0,
            target_rate_min=0.5,
            user_rate_min=0.0,
            sensing_only_sinr_min=0.0,
            active_power_max=2.0,
        )
        objective = ObjectiveSpec.png_joint(
            user_rate_min=0.0, target_rate_min=0.5
        )
        fast = select_fast_power_allocation(
            2, 2, self._evaluate, constraints, objective
        )
        refined = optimize_power_allocation(
            2, 2, self._evaluate, constraints, objective,
            initial_powers=fast.powers,
        )
        self.assertTrue(refined.feasible)
        self.assertGreaterEqual(refined.objective_value, fast.objective_value - 1e-7)

    def test_infeasible_constraints_are_reported(self):
        constraints = AllocationConstraints(
            total_power=1.0,
            target_rate_min=20.0,
            user_rate_min=20.0,
            sensing_only_sinr_min=100.0,
            active_power_max=0.01,
        )
        result = select_fast_power_allocation(
            2, 2, self._evaluate, constraints,
            ObjectiveSpec.png_joint(user_rate_min=20.0, target_rate_min=20.0),
        )
        self.assertFalse(result.feasible)
        self.assertGreater(result.total_normalized_deficit, 0.0)

    def test_fast_allocation_can_find_unequal_feasible_target_powers(self):
        def unequal_channels(powers):
            powers = np.asarray(powers, dtype=float)
            return {
                "rate_u": np.log2(1.0 + 2.0 * powers[:1]),
                "rate_k": np.log2(
                    1.0 + np.array([1.0 / 6.0, 4.0]) * powers[1:]
                ),
                "sensing_only_sinr": (
                    np.array([1.0 / 6.0, 4.0]) * powers[1:]
                ),
                "active_power": 0.1 * float(np.sum(powers[1:])),
            }

        constraints = AllocationConstraints(
            total_power=10.0,
            target_rate_min=1.0,
            active_power_max=10.0,
        )
        result = select_fast_power_allocation(
            1,
            2,
            unequal_channels,
            constraints,
            ObjectiveSpec.paper(target_rate_min=1.0),
        )
        self.assertTrue(result.feasible)
        self.assertGreater(result.powers[1], result.powers[2] * 5.0)
        self.assertTrue(np.all(result.rate_k >= 1.0 - 1e-8))

    def test_previous_split_does_not_hide_distant_feasible_allocation(self):
        def changed_channel(powers):
            powers = np.asarray(powers, dtype=float)
            return {
                "rate_u": np.array([powers[0]]),
                "rate_k": np.array([powers[1]]),
                "sensing_only_sinr": np.array([powers[1]]),
                "active_power": 0.0,
            }

        constraints = AllocationConstraints(
            total_power=1.0,
            target_rate_min=0.8,
            active_power_max=1.0,
        )
        result = select_fast_power_allocation(
            1,
            1,
            changed_channel,
            constraints,
            ObjectiveSpec.paper(target_rate_min=0.8),
            previous_powers=np.array([0.95, 0.05]),
        )
        self.assertTrue(result.feasible)
        self.assertLessEqual(result.powers[0], 0.2 + 1e-12)


if __name__ == "__main__":
    unittest.main()
