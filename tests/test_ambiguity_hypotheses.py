import copy
import unittest

from ambiguity_hypotheses import (
    DDPG_COMPARATOR_CSI_HYPOTHESES,
    CHANNEL_COHERENCE_HYPOTHESES,
    CROSS_FUNCTION_INTERFERENCE_HYPOTHESES,
    LOS_PROBABILITY_HYPOTHESES,
    NOISE_HYPOTHESES,
    TRANSFER_SCHEDULE_HYPOTHESES,
    TRANSFER_UPDATE_HYPOTHESES,
    WFB_HYPOTHESES,
    ambiguity_metadata,
    apply_ambiguity_hypotheses,
    ddpg_comparator_scee,
    get_noise_hypothesis,
)
from config import cfg
from experiment_protocol import physical_configuration_fingerprint
from physical_hypotheses import dbm_to_watts


class AmbiguityHypothesisTests(unittest.TestCase):
    def test_registry_contains_only_predeclared_hypotheses(self):
        self.assertEqual(
            set(NOISE_HYPOTHESES),
            {
                "description_literal_target_matches_ue",
                "dual_minus70_conservative",
            },
        )
        self.assertEqual(
            set(TRANSFER_UPDATE_HYPOTHESES),
            {
                "eq30_current_q_bellman_blend",
                "eq30_next_state_continuation_blend",
            },
        )
        self.assertEqual(
            set(TRANSFER_SCHEDULE_HYPOTHESES),
            {"constant_upsilon", "linear_annealed_upsilon"},
        )
        self.assertEqual(
            set(WFB_HYPOTHESES),
            {
                "adaptive_matched_filter_no_capon",
                "channel_independent_dft_no_capon",
            },
        )
        self.assertEqual(
            set(DDPG_COMPARATOR_CSI_HYPOTHESES),
            {"fair_pcsi_comparator", "icsi_comparator"},
        )
        self.assertEqual(
            set(LOS_PROBABILITY_HYPOTHESES),
            {"paper_eq6_literal_roles", "al_hourani_standard_mapped_roles"},
        )
        self.assertEqual(
            set(CHANNEL_COHERENCE_HYPOTHESES),
            {"episode_block_fading", "independent_slot_fading"},
        )
        self.assertEqual(
            set(CROSS_FUNCTION_INTERFERENCE_HYPOTHESES),
            {
                "paper_eq12_eq14_literal_cross_channels",
                "receiver_consistent_cross_channels",
            },
        )

    def test_unknown_noise_hypothesis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown noise hypothesis"):
            get_noise_hypothesis("best_result")

    def test_primary_noise_mapping_follows_description_and_receiver_floor(self):
        candidate = copy.copy(cfg)
        apply_ambiguity_hypotheses(
            candidate,
            noise_hypothesis_id="description_literal_target_matches_ue",
        )
        self.assertEqual(candidate.noise_hypothesis_id,
                         "description_literal_target_matches_ue")
        self.assertAlmostEqual(candidate.sigma2_u, dbm_to_watts(-94.0))
        self.assertAlmostEqual(candidate.sigma2_k, dbm_to_watts(-94.0))
        self.assertAlmostEqual(candidate.sigma2_a, dbm_to_watts(-70.0))

    def test_noise_sensitivity_keeps_equation_terms_distinct(self):
        candidate = copy.copy(cfg)
        apply_ambiguity_hypotheses(
            candidate, noise_hypothesis_id="dual_minus70_conservative"
        )
        self.assertAlmostEqual(candidate.sigma2_k, dbm_to_watts(-70.0))
        self.assertAlmostEqual(candidate.sigma2_a, dbm_to_watts(-70.0))
        self.assertIsNot(candidate.__class__.sigma2_k,
                         candidate.__class__.sigma2_a)

    def test_metadata_records_all_axes_and_no_result_selection(self):
        metadata = ambiguity_metadata(cfg)
        self.assertEqual(metadata["selection_rule"], "predeclared_not_result_selected")
        self.assertNotIn("selected_hypothesis", metadata)
        self.assertEqual(
            set(metadata["axes"]),
            {
                "noise", "transfer_update", "transfer_schedule", "wfb",
                "ddpg_comparator_csi", "los_probability",
                "channel_coherence",
                "cross_function_interference",
            },
        )

    def test_ddpg_comparator_primary_is_fair_pcsi_and_icsi_is_sensitivity(self):
        primary = copy.copy(cfg)
        apply_ambiguity_hypotheses(primary)
        self.assertEqual(
            primary.ddpg_comparator_csi_hypothesis_id,
            "fair_pcsi_comparator",
        )
        self.assertEqual(ddpg_comparator_scee(primary), 0.0)

        sensitivity = copy.copy(cfg)
        apply_ambiguity_hypotheses(
            sensitivity,
            ddpg_comparator_csi_hypothesis_id="icsi_comparator",
        )
        self.assertEqual(ddpg_comparator_scee(sensitivity), sensitivity.rho_nmse)
        self.assertNotEqual(
            physical_configuration_fingerprint("profile", primary),
            physical_configuration_fingerprint("profile", sensitivity),
        )

    def test_physical_fingerprint_changes_with_noise_mapping(self):
        left = copy.copy(cfg)
        right = copy.copy(cfg)
        apply_ambiguity_hypotheses(
            left, noise_hypothesis_id="description_literal_target_matches_ue"
        )
        apply_ambiguity_hypotheses(
            right, noise_hypothesis_id="dual_minus70_conservative"
        )
        self.assertNotEqual(
            physical_configuration_fingerprint("profile", left),
            physical_configuration_fingerprint("profile", right),
        )


if __name__ == "__main__":
    unittest.main()
