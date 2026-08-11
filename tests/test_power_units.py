import unittest

from config import cfg, dbm_to_watts, dbw_to_watts, watts_to_dbm
from physical_hypotheses import get_power_hypothesis


class PowerUnitTests(unittest.TestCase):
    def test_primary_paper_hypothesis_is_dbm(self):
        self.assertAlmostEqual(cfg.PT_dB, 10.0)
        self.assertAlmostEqual(cfg.PT, 0.01)
        self.assertAlmostEqual(cfg.PT_dBm, 10.0)
        self.assertEqual(cfg.PT_reference, "1 mW")
        self.assertEqual(cfg.power_hypothesis_id, "paper_db_as_dbm")

    def test_dbw_and_dbm_conversions_are_not_aliases(self):
        self.assertAlmostEqual(dbm_to_watts(10.0), 0.01)
        self.assertAlmostEqual(dbw_to_watts(10.0), 10.0)
        self.assertAlmostEqual(watts_to_dbm(10.0), 40.0)

    def test_both_paper_power_hypotheses_are_explicit(self):
        primary = get_power_hypothesis("paper_db_as_dbm")
        sensitivity = get_power_hypothesis("paper_db_as_dbw")
        self.assertAlmostEqual(primary.to_watts(10.0), 0.01)
        self.assertAlmostEqual(sensitivity.to_watts(10.0), 10.0)
        self.assertEqual(primary.role, "primary")
        self.assertEqual(sensitivity.role, "sensitivity")

    def test_power_conversion_rejects_nonpositive_watts(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            watts_to_dbm(0.0)


if __name__ == "__main__":
    unittest.main()
