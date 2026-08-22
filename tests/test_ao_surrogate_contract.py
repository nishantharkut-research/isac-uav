import inspect
import unittest
from unittest.mock import patch

import numpy as np

import ao_optimizer
from paper_contract import ao_surrogate_contract
from config import cfg


class AOSurrogateContractTests(unittest.TestCase):
    @staticmethod
    def _run_with_rates(rates):
        rate_values = iter(rates)

        def compute_rates(*_args, **_kwargs):
            communication = np.array([float(next(rate_values))])
            sensing = np.array([1.0])
            return communication, sensing, None, None

        beamformer = (
            np.zeros((1, 1), dtype=complex),
            np.zeros((1, 1), dtype=complex),
            {"comm_power_frac": 0.5, "beamformer_score": 0.0},
        )
        with (
            patch.object(ao_optimizer, "ChannelSet", return_value=object()),
            patch.object(ao_optimizer, "solve_beamformer", return_value=beamformer),
            patch.object(
                ao_optimizer,
                "solve_passive_phases",
                side_effect=lambda _ch, theta, *_args, **_kwargs: np.angle(theta),
            ),
            patch.object(
                ao_optimizer,
                "solve_active_phases",
                side_effect=lambda _ch, phi, *_args, **_kwargs: phi,
            ),
            patch.object(ao_optimizer, "compute_rates", side_effect=compute_rates),
            patch.object(
                ao_optimizer,
                "_constraint_diagnostics",
                return_value=(True, {"all": {"ok": True}}),
            ),
            patch.object(
                ao_optimizer,
                "solve_uav_trajectory",
                side_effect=AssertionError(
                    "paper AO surrogate must not turn optimizer iterations into slots"
                ),
            ),
        ):
            return ao_optimizer.run_ao(
                np.array([0.0, 0.0, 50.0]),
                [np.zeros(3)],
                [np.zeros(3)],
                n_iter=len(rates) - 1,
                stop_on_convergence=False,
                seed=7,
            )

    def test_paper_surrogate_defaults_to_fixed_geometry(self):
        self.assertIs(
            inspect.signature(ao_optimizer.run_ao)
            .parameters["fix_uav"].default,
            True,
        )
        result = self._run_with_rates([1.0, 2.0])
        self.assertEqual(result["trajectory_mode"], "fixed_geometry")
        self.assertEqual(
            result["solver_identity"],
            ao_surrogate_contract(cfg)["solver_identity"],
        )

    def test_reported_result_is_terminal_not_optimistically_best_iterate(self):
        result = self._run_with_rates([1.0, 3.0, 2.0])
        self.assertEqual(result["sum_rate"], 2.0)
        self.assertEqual(result["selection"], "terminal_iterate")
        self.assertEqual(result["best_feasible_sum_rate"], 3.0)

    def test_surrogate_does_not_claim_published_sca_or_use_global_context(self):
        source = inspect.getsource(ao_optimizer)
        self.assertNotIn("global _current", source)
        self.assertNotIn("globals().get", source)
        self.assertNotIn("Algorithm 3: Alternating Optimisation", source)
        self.assertFalse(
            inspect.signature(ao_optimizer.solve_beamformer)
            .parameters["use_capon"].default
        )


if __name__ == "__main__":
    unittest.main()
