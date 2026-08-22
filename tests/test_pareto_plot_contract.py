import unittest

from plot_pareto_extension import nondominated_indices


class ParetoPlotContractTests(unittest.TestCase):
    def test_only_unique_nondominated_points_are_called_the_front(self):
        points = [
            [4.0, 1.0],
            [3.0, 3.0],
            [2.0, 2.0],  # dominated by [3, 3]
            [1.0, 4.0],
            [3.0, 3.0],  # duplicate
        ]
        self.assertEqual(nondominated_indices(points), [3, 1, 0])

    def test_nonfinite_front_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            nondominated_indices([[1.0, float("nan")]])


if __name__ == "__main__":
    unittest.main()
