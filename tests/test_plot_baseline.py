import unittest

from plot_baseline import build_fig4_plot_data


class PlotBaselineTests(unittest.TestCase):
    def test_plot_data_comes_from_raw_seed_histories_at_declared_x_points(self):
        seeds = [1, 2, 3, 4, 5]
        methods = ("DTL-pCSI", "DTL-WFB", "DTL-iCSI", "DDPG")
        summary = {
            "seeds": seeds,
            "episodes": 1500,
            "training_final_window": 100,
            "figure4_protocol": {
                "raw_episode_count": 1500,
                "plot_x_start": 0,
                "plot_x_end": 1400,
                "plot_interval": 100,
                "acceptance_smoothing_window": 20,
            },
            "seed_records": [
                {
                    "seed": seed,
                    "methods": {
                        method: {
                            "training": {
                                "comm_rates": [float(index + method_index)] * 1500
                            }
                        }
                        for method_index, method in enumerate(methods)
                    },
                }
                for index, seed in enumerate(seeds)
            ],
        }
        data = build_fig4_plot_data(summary, smoothing_window=20)
        self.assertEqual(data["x"], list(range(0, 1500, 100)))
        self.assertEqual(set(data["curves"]), set(methods))
        self.assertEqual(data["curves"]["DTL-pCSI"], [2.0] * 15)
        self.assertEqual(len(data["raw_histories"]["DDPG"]), 5)

    def test_plot_builder_rejects_short_or_missing_histories(self):
        summary = {
            "seeds": [1, 2, 3, 4, 5],
            "episodes": 1500,
            "figure4_protocol": {
                "raw_episode_count": 1500,
                "plot_x_start": 0,
                "plot_x_end": 1400,
                "plot_interval": 100,
                "acceptance_smoothing_window": 20,
            },
            "seed_records": [],
        }
        with self.assertRaises(ValueError):
            build_fig4_plot_data(summary)


if __name__ == "__main__":
    unittest.main()
