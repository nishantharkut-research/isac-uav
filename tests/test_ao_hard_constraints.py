import inspect
import unittest

import ao_optimizer


class AOHardConstraintTests(unittest.TestCase):
    def test_phase_objective_contains_no_fixed_penalty_weights(self):
        source = inspect.getsource(ao_optimizer._scee_phase_objective)
        self.assertNotIn("rate_penalty_weight", source)
        self.assertNotIn("cascade_penalty_weight", source)
        self.assertNotIn("active_penalty_weight", source)
        self.assertIn("joint_sum_rate", source)

    def test_phase_and_trajectory_solvers_use_explicit_slsqp_constraints(self):
        passive = inspect.getsource(ao_optimizer.solve_passive_phases)
        active = inspect.getsource(ao_optimizer.solve_active_phases)
        trajectory = inspect.getsource(ao_optimizer.solve_uav_trajectory)
        for source in (passive, active, trajectory):
            self.assertIn("SLSQP", source)
            self.assertIn("constraints", source)

    def test_raw_ao_score_has_no_penalty_subtraction(self):
        source = inspect.getsource(ao_optimizer._score_beamformers)
        self.assertNotIn("penalty", source)
        self.assertIn("joint_sum_rate", source)


if __name__ == "__main__":
    unittest.main()
