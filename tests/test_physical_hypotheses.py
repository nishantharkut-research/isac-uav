import copy
import json
import unittest
from pathlib import Path

from config import cfg
from experiment_protocol import physical_configuration_fingerprint
from physical_hypotheses import (
    POWER_HYPOTHESES,
    apply_power_hypothesis,
    get_power_hypothesis,
)
from parameter_profiles import applied_profile, profile_from_mapping


class PhysicalHypothesisTests(unittest.TestCase):
    def test_power_hypothesis_metadata_has_an_unambiguous_reference(self):
        for identifier in ("paper_db_as_dbm", "paper_db_as_dbw"):
            with self.subTest(identifier=identifier):
                hypothesis = get_power_hypothesis(identifier)
                metadata = hypothesis.as_dict(10.0)
                self.assertEqual(metadata["id"], identifier)
                self.assertIn(metadata["display_unit"], {"dBm", "dBW"})
                self.assertGreater(metadata["reference_power_watts"], 0.0)
                self.assertGreater(metadata["linear_power_watts"], 0.0)

    def test_unknown_power_hypothesis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown power hypothesis"):
            get_power_hypothesis("largest_reward")

    def test_applying_hypothesis_updates_all_derived_power_fields(self):
        candidate = copy.copy(cfg)
        apply_power_hypothesis(candidate, "paper_db_as_dbw", 10.0)
        self.assertEqual(candidate.power_hypothesis_id, "paper_db_as_dbw")
        self.assertEqual(candidate.PT_reference, "1 W")
        self.assertAlmostEqual(candidate.PT, 10.0)
        self.assertAlmostEqual(candidate.PT_dBm, 40.0)

    def test_fingerprint_changes_with_hypothesis_even_for_same_linear_power(self):
        left = copy.copy(cfg)
        right = copy.copy(cfg)
        left.power_hypothesis_id = "paper_db_as_dbm"
        right.power_hypothesis_id = "paper_db_as_dbw"
        right.PT = left.PT
        self.assertNotEqual(
            physical_configuration_fingerprint("profile", left),
            physical_configuration_fingerprint("profile", right),
        )

    def test_receiver_and_amplifier_noises_are_distinct_parameters(self):
        self.assertTrue(hasattr(cfg, "sigma2_u"))
        self.assertTrue(hasattr(cfg, "sigma2_k"))
        self.assertTrue(hasattr(cfg, "sigma2_a"))
        changed = copy.copy(cfg)
        original = physical_configuration_fingerprint("profile", changed)
        changed.sigma2_k *= 2.0
        self.assertNotEqual(
            original, physical_configuration_fingerprint("profile", changed)
        )

    def test_reference_axis_does_not_claim_a_power_reference(self):
        path = Path(__file__).parents[1] / "references" / "paper_reference_curves.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        units = payload["figures"]["6"]["x_axis"]["units"]
        self.assertEqual(units, "paper-reported dB label (reference unspecified)")

    def test_only_predeclared_hypotheses_exist(self):
        self.assertEqual(
            set(POWER_HYPOTHESES),
            {"paper_db_as_dbm", "paper_db_as_dbw"},
        )

    def test_profile_activation_applies_its_locked_power_hypothesis(self):
        profile = profile_from_mapping({
            "schema_version": 1,
            "name": "dbw-profile",
            "kind": "identified",
            "values": {"omega_0": 1e-3},
            "evidence": {"omega_0": "omega_0"},
            "selection": {"power_hypothesis_id": "paper_db_as_dbw"},
        })
        candidate = copy.copy(cfg)
        with applied_profile(profile, cfg_obj=candidate):
            self.assertEqual(candidate.power_hypothesis_id, "paper_db_as_dbw")
            self.assertAlmostEqual(candidate.PT, 10.0)
        self.assertEqual(candidate.power_hypothesis_id, cfg.power_hypothesis_id)

    def test_figure_power_sweep_uses_active_hypothesis_and_restores_config(self):
        import simulation

        original = (cfg.PT_dB, cfg.PT, cfg.power_hypothesis_id)
        with simulation._temporary_paper_power_label(20.0):
            self.assertEqual(cfg.power_hypothesis_id, "paper_db_as_dbm")
            self.assertAlmostEqual(cfg.PT, 0.1)
        self.assertEqual((cfg.PT_dB, cfg.PT, cfg.power_hypothesis_id), original)


if __name__ == "__main__":
    unittest.main()
