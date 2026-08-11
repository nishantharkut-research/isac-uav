"""Validate generated Figure 6 calibration evidence and paper holdouts."""

import argparse
import json
from pathlib import Path

from experiment_protocol import atomic_write_json
from holdout_protocol import validate_holdout_result
from paper_reference import load_paper_references


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--reference", default="references/paper_reference_curves.json"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    references = load_paper_references(args.reference)
    report = validate_holdout_result(
        summary,
        references,
        result_directory=Path(args.summary).resolve().parent,
        verify_files=True,
    )
    atomic_write_json(args.output, report)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
