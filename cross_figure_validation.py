"""Qualitative validation of exact generated figure-data JSON files."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiment_contract import FIGURE_CONTRACTS
from result_evidence import validate_result_evidence_bundle


CALIBRATION_FIGURES = (6,)
HOLDOUT_FIGURES = (4, 5, 7, 8, 9, 13, 14)


def _tolerance(values, relative=0.05, absolute=1e-8):
    values = np.asarray(values, dtype=float)
    return max(absolute, relative * max(float(np.max(np.abs(values))), 1.0))


def _nondecreasing(values):
    values = np.asarray(values, dtype=float)
    return bool(np.all(np.diff(values) >= -_tolerance(values)))


def _nonincreasing(values):
    values = np.asarray(values, dtype=float)
    return bool(np.all(np.diff(values) <= _tolerance(values)))


def _ordered(upper, lower):
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)
    tolerance = _tolerance(np.concatenate([upper, lower]))
    return bool(np.all(upper + tolerance >= lower))


def _approximately_flat(values):
    values = np.asarray(values, dtype=float)
    return bool(float(np.ptp(values)) <= _tolerance(values))


def _check_trend(curves, label, direction, failures):
    checker = _nondecreasing if direction == "nondecreasing" else _nonincreasing
    if not checker(curves[label]):
        failures.append(f"{label} is not {direction}")


def validate_figure_payload(figure, payload):
    """Validate one payload without transforming or replacing any values."""
    figure = int(figure)
    failures = []
    if not isinstance(payload, dict):
        return {"passed": False, "failures": ["payload is not an object"]}
    if payload.get("figure") != figure:
        failures.append("figure number does not match file contract")
    x = payload.get("x")
    curves = payload.get("curves")
    if not isinstance(x, list) or not x:
        failures.append("x must be a non-empty list")
    if not isinstance(curves, dict):
        failures.append("curves must be an object")
        return {"passed": False, "failures": failures}

    expected_labels = {curve.label for curve in FIGURE_CONTRACTS[figure].curves}
    actual_labels = set(curves)
    if actual_labels != expected_labels:
        missing = sorted(expected_labels - actual_labels)
        extra = sorted(actual_labels - expected_labels)
        if missing:
            failures.append(f"missing method-owned curves: {missing}")
        if extra:
            failures.append(f"unexpected curves: {extra}")

    numeric_curves = {}
    for label, values in curves.items():
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            failures.append(f"{label} is not numeric")
            continue
        if array.ndim != 1:
            failures.append(f"{label} must be one-dimensional")
            continue
        if isinstance(x, list) and array.size != len(x):
            failures.append(f"{label} length does not match x")
        if not np.all(np.isfinite(array)):
            failures.append(f"{label} contains non-finite values")
        numeric_curves[label] = array

    if failures or set(numeric_curves) != expected_labels:
        return {"passed": False, "failures": failures}

    if figure == 5:
        for label in expected_labels:
            _check_trend(numeric_curves, label, "nondecreasing", failures)
        if not _ordered(numeric_curves["L=30"], numeric_curves["L=20"]):
            failures.append("L=30 must not underperform L=20")
        if not _ordered(numeric_curves["L=20"], numeric_curves["L=10"]):
            failures.append("L=20 must not underperform L=10")
    elif figure == 4:
        means = {
            label: float(np.mean(values[-max(1, len(values) // 10):]))
            for label, values in numeric_curves.items()
        }
        for upper, lower in (
            ("DTL-pCSI", "DTL-WFB"),
            ("DTL-WFB", "DTL-iCSI"),
            ("DTL-iCSI", "DDPG"),
        ):
            if means[upper] <= means[lower]:
                failures.append(f"expected Fig. 4 ordering failed: {upper} > {lower}")
    elif figure == 6:
        for label in expected_labels:
            _check_trend(numeric_curves, label, "nondecreasing", failures)
    elif figure == 8:
        for label in expected_labels:
            _check_trend(numeric_curves, label, "nonincreasing", failures)
    elif figure == 7:
        for label in expected_labels - {"No-RIS"}:
            _check_trend(numeric_curves, label, "nondecreasing", failures)
        if not _approximately_flat(numeric_curves["No-RIS"]):
            failures.append("No-RIS must be independent of HRIS element count")
        for upper, lower in (
            ("HRIS-pCSI", "HRIS-iCSI"),
            ("HRIS-iCSI", "HRIS-ParCSI"),
            ("Passive RIS", "Random Phase"),
            ("Random Phase", "No-RIS"),
        ):
            if not _ordered(numeric_curves[upper], numeric_curves[lower]):
                failures.append(f"expected ordering failed: {upper} >= {lower}")
    elif figure == 9:
        if not _ordered(numeric_curves["nu=0.1"], numeric_curves["nu=0.3"]):
            failures.append("expected Fig. 9 ordering failed: nu=0.1 >= nu=0.3")
        if not _ordered(numeric_curves["nu=0.3"], numeric_curves["nu=0.5"]):
            failures.append("expected Fig. 9 ordering failed: nu=0.3 >= nu=0.5")
    elif figure == 13:
        for label in ("R_u, L=40", "R_u, L=30", "R_u, L=20"):
            _check_trend(numeric_curves, label, "nonincreasing", failures)
        for label in ("R_k, L=30", "R_k, L=20"):
            _check_trend(numeric_curves, label, "nondecreasing", failures)
        if not _ordered(numeric_curves["R_u, L=40"], numeric_curves["R_u, L=30"]):
            failures.append("expected Fig. 13 ordering failed: Ru L=40 >= L=30")
        if not _ordered(numeric_curves["R_u, L=30"], numeric_curves["R_u, L=20"]):
            failures.append("expected Fig. 13 ordering failed: Ru L=30 >= L=20")
    elif figure == 14:
        for label in expected_labels:
            _check_trend(numeric_curves, label, "nonincreasing", failures)
        for elements in (30, 20):
            if not _ordered(
                numeric_curves[f"DTL, L={elements}"],
                numeric_curves[f"DDPG, L={elements}"],
            ):
                failures.append(f"expected Fig. 14 DTL >= DDPG for L={elements}")
        for method in ("DTL", "DDPG"):
            if not _ordered(
                numeric_curves[f"{method}, L=30"],
                numeric_curves[f"{method}, L=20"],
            ):
                failures.append(f"expected Fig. 14 L=30 >= L=20 for {method}")

    return {
        "passed": not failures,
        "failures": failures,
        "curve_count": len(numeric_curves),
        "point_count": len(x) if isinstance(x, list) else 0,
    }


def _validate_group(directory, figures):
    results = {}
    for figure in figures:
        path = directory / f"fig{figure}_data.json"
        if not path.exists():
            results[figure] = {
                "passed": False,
                "failures": [f"missing generated data file: {path.name}"],
            }
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            result = validate_figure_payload(figure, payload)
        except Exception as exc:  # malformed output fails closed.
            result = {
                "passed": False,
                "failures": [f"{type(exc).__name__}: {exc}"],
            }
        result["source"] = str(path.resolve())
        results[figure] = result
    return results


def validate_result_directory(directory, output_path=None):
    directory = Path(directory)
    calibration = _validate_group(directory, CALIBRATION_FIGURES)
    holdout = _validate_group(directory, HOLDOUT_FIGURES)
    all_results = list(calibration.values()) + list(holdout.values())
    evidence = validate_result_evidence_bundle(directory)
    report = {
        "schema_version": 1,
        "source_directory": str(directory.resolve()),
        "calibration_figures": list(CALIBRATION_FIGURES),
        "holdout_figures": list(HOLDOUT_FIGURES),
        "paper_curve_ordinates_used": False,
        "curves_modified": False,
        "result_evidence": evidence,
        "calibration": calibration,
        "holdout": holdout,
        "overall_pass": (
            all(item["passed"] for item in all_results)
            and evidence["overall_pass"]
        ),
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    return report


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Validate qualitative contracts in generated figure data"
    )
    parser.add_argument("result_directory", nargs="?", default="result")
    parser.add_argument(
        "--output", default="result/cross_figure_validation.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validation = validate_result_directory(args.result_directory, args.output)
    print(json.dumps(validation, indent=2, sort_keys=True))
    raise SystemExit(0 if validation["overall_pass"] else 1)
