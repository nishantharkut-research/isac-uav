import unittest
from pathlib import Path

from paper_reference import (
    load_paper_references,
    paper_reference_digest,
    require_canonical_paper_references,
)
from cross_figure_validation import CALIBRATION_FIGURES, HOLDOUT_FIGURES


class PaperReferenceTests(unittest.TestCase):
    def test_reference_schema_and_roles(self):
        path = Path(__file__).parents[1] / "references" / "paper_reference_curves.json"
        reference = load_paper_references(path)
        self.assertEqual(set(reference["calibration_figures"]), {6})
        self.assertTrue(
            {4, 5, 7, 8, 9, 13, 14}.issubset(reference["holdout_figures"])
        )
        self.assertEqual(reference["figures"]["6"]["role"], "calibration")
        self.assertEqual(reference["figures"]["7"]["role"], "holdout")
        for figure in reference["figures"].values():
            self.assertEqual(figure["kind"], "digitized_reference")
            self.assertGreater(figure["source_page"], 0)
            self.assertTrue(figure["x_axis"]["units"])
            self.assertTrue(figure["digitization_method"])
            self.assertGreater(figure["uncertainty"]["absolute_y"], 0.0)

    def test_cross_figure_validation_uses_the_same_data_split(self):
        self.assertEqual(CALIBRATION_FIGURES, (6,))
        self.assertEqual(HOLDOUT_FIGURES, (4, 5, 7, 8, 9, 13, 14))

    def test_generated_result_cannot_be_loaded_as_reference(self):
        payload = {
            "schema_version": 1,
            "calibration_figures": [6],
            "holdout_figures": [],
            "figures": {
                "6": {
                    "kind": "generated_result",
                    "source_page": 11,
                    "x_axis": {"label": "power", "units": "dB"},
                    "y_axis": {"label": "rate", "units": "bits/s/Hz"},
                    "digitization_method": "visual marker centers",
                    "uncertainty": {"absolute_y": 0.1},
                    "x": [10],
                    "curves": {"DTL-pCSI": [7.5]},
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "digitized_reference"):
            load_paper_references(payload)

    def test_reference_digest_is_canonical_and_content_sensitive(self):
        path = Path(__file__).parents[1] / "references" / "paper_reference_curves.json"
        reference = load_paper_references(path)
        self.assertEqual(paper_reference_digest(path), paper_reference_digest(reference))
        reference["figures"]["4"]["curves"]["DTL-pCSI"][0] += 0.01
        self.assertNotEqual(paper_reference_digest(path), paper_reference_digest(reference))

    def test_publication_reference_authority_rejects_modified_payload(self):
        reference = load_paper_references(None)
        reference["figures"]["4"]["curves"]["DTL-pCSI"][0] += 0.01

        with self.assertRaisesRegex(ValueError, "canonical"):
            require_canonical_paper_references(reference)


if __name__ == "__main__":
    unittest.main()
