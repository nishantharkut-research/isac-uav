"""Plot five-seed paper sweeps with confidence bands and evidence manifests."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment_protocol import atomic_write_json
from holdout_protocol import PUBLICATION_FIGURES, validate_holdout_result
from paper_reference import require_canonical_paper_references
from source_integrity import source_tree_digest
from validate_results import result_payload_digest


PLOT_METADATA = {
    5: ("AO Iteration", "Communication Sum Rate (bits/s/Hz)", "AO Convergence"),
    6: ("Transmitter Power (paper dB label)", "Communication Sum Rate (bits/s/Hz)", "Rate vs Transmitter Power"),
    7: ("Number of HRIS Elements", "Communication Sum Rate (bits/s/Hz)", "Rate vs HRIS Elements"),
    8: ("SCEE NMSE", "Communication Sum Rate (bits/s/Hz)", "Rate vs Channel Estimation Error"),
    9: ("Number of HRIS Elements", "Communication Sum Rate (bits/s/Hz)", "Rate vs Active-Element Fraction"),
    13: ("Minimum Target SINR (linear)", "Rate (bits/s/Hz)", "Communication and Target Rate vs SINR Floor"),
    14: ("Minimum Sensing-Only SINR (dB)", "Communication Sum Rate (bits/s/Hz)", "Rate vs Sensing-Only SINR Floor"),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _plot_one(output, figure, payload, power_hypothesis_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#1769aa", "#c43c35", "#2b8a3e", "#e07a1f", "#5b5f97", "#00838f")
    markers = ("o", "s", "^", "D", "v", "P")
    x = np.asarray(payload["x"], dtype=float)
    plot, axis = plt.subplots(figsize=(8.2, 5.2))
    x_label, y_label, title = PLOT_METADATA[figure]
    if figure == 9:
        labels = list(payload["curves"])
        width = 0.22
        positions = np.arange(x.size, dtype=float)
        for index, label in enumerate(labels):
            mean = np.asarray(payload["curves"][label], dtype=float)
            low = np.asarray(payload["ci95_low"][label], dtype=float)
            high = np.asarray(payload["ci95_high"][label], dtype=float)
            offset = (index - (len(labels) - 1) / 2.0) * width
            axis.bar(
                positions + offset,
                mean,
                width=width,
                color=colors[index % len(colors)],
                label=label,
                yerr=np.vstack([mean - low, high - mean]),
                capsize=3,
                linewidth=0.5,
                edgecolor="black",
            )
        axis.set_xticks(positions, [f"L = {int(value)}" for value in x])
        axis.set_xlabel("Number of HRIS Elements")
        axis.set_ylabel(y_label)
        axis.legend()
    elif figure == 13:
        target_axis = axis.twinx()
        for index, label in enumerate(payload["curves"]):
            mean = np.asarray(payload["curves"][label], dtype=float)
            low = np.asarray(payload["ci95_low"][label], dtype=float)
            high = np.asarray(payload["ci95_high"][label], dtype=float)
            selected_axis = target_axis if label.startswith("R_k") else axis
            color = colors[index % len(colors)]
            selected_axis.plot(
                x, mean, color=color, marker=markers[index % len(markers)],
                linewidth=1.7, markersize=4.5, label=label,
            )
            selected_axis.fill_between(
                x, low, high, color=color, alpha=0.12, linewidth=0
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Communication Sum Rate, R_u (bits/s/Hz)")
        target_axis.set_ylabel("Target Sum Rate, R_k (bits/s/Hz)")
        handles_left, labels_left = axis.get_legend_handles_labels()
        handles_right, labels_right = target_axis.get_legend_handles_labels()
        axis.legend(handles_left + handles_right, labels_left + labels_right,
                    ncol=2)
    else:
        for index, label in enumerate(payload["curves"]):
            mean = np.asarray(payload["curves"][label], dtype=float)
            low = np.asarray(payload["ci95_low"][label], dtype=float)
            high = np.asarray(payload["ci95_high"][label], dtype=float)
            color = colors[index % len(colors)]
            axis.plot(
                x, mean, color=color, marker=markers[index % len(markers)],
                linewidth=1.7, markersize=4.5, label=label,
            )
            axis.fill_between(
                x, low, high, color=color, alpha=0.12, linewidth=0
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.legend(ncol=2 if len(payload["curves"]) > 4 else 1)
    if figure == 6:
        title = f"{title} ({power_hypothesis_id})"
    axis.set_title(f"Fig. {figure} - {title}")
    axis.grid(True, alpha=0.25)
    plot.tight_layout()
    image = output / f"fig{figure}_five_seed.png"
    plot.savefig(image, dpi=220, bbox_inches="tight")
    plt.close(plot)
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--reference", default="references/paper_reference_curves.json"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    summary = _read_json(args.summary)
    references = require_canonical_paper_references(args.reference)
    validation = validate_holdout_result(
        summary,
        references,
        result_directory=Path(args.summary).resolve().parent,
        verify_files=True,
    )
    if validation.get("overall_pass") is not True:
        raise ValueError(
            "Holdout result failed validation; publication figures were not generated."
        )
    current_source_digest = source_tree_digest()
    if summary.get("source_tree_digest") != current_source_digest:
        raise ValueError("Holdout summary was generated by a different source tree")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Holdout plot directory must be empty")

    artifacts = []
    for figure in PUBLICATION_FIGURES:
        payload = summary["figures"][str(figure)]
        image = _plot_one(
            output, figure, payload, summary["power_hypothesis_id"]
        )
        arrays = {"x": payload["x"]}
        for label, values in payload["curves"].items():
            arrays[label] = values
            arrays[f"{label}_ci95_low"] = payload["ci95_low"][label]
            arrays[f"{label}_ci95_high"] = payload["ci95_high"][label]
        evidence = {
            "schema_version": 1,
            "kind": "generated_result_evidence",
            "figure": figure,
            "role": "calibration" if figure == 6 else "holdout",
            "source_tree_digest": current_source_digest,
            "profile_digest": summary["profile_digest"],
            "physical_configuration_fingerprint": summary[
                "physical_configuration_fingerprint"
            ],
            "power_hypothesis_id": summary["power_hypothesis_id"],
            "noise_hypothesis_id": summary["noise_hypothesis_id"],
            "dtl_transfer_mode": summary["dtl_transfer_mode"],
            "dtl_transfer_schedule_id": summary[
                "dtl_transfer_schedule_id"
            ],
            "wfb_hypothesis_id": summary["wfb_hypothesis_id"],
            "ddpg_comparator_csi_hypothesis_id": summary[
                "ddpg_comparator_csi_hypothesis_id"
            ],
            "los_probability_hypothesis_id": summary[
                "los_probability_hypothesis_id"
            ],
            "channel_coherence_hypothesis_id": summary[
                "channel_coherence_hypothesis_id"
            ],
            "csi_observation_hypothesis_id": summary[
                "csi_observation_hypothesis_id"
            ],
            "rate_model_hypothesis_id": summary[
                "rate_model_hypothesis_id"
            ],
            "cross_function_interference_hypothesis_id": summary[
                "cross_function_interference_hypothesis_id"
            ],
            "baseline_payload_sha256": summary["baseline_payload_sha256"],
            "baseline_validation_sha256": summary[
                "baseline_validation_sha256"
            ],
            "holdout_summary_sha256": result_payload_digest(summary),
            "paper_reference_sha256": validation[
                "paper_reference_sha256"
            ],
            "seeds": summary["seeds"],
            "units": {
                "x": PLOT_METADATA[figure][0],
                "y": "bits/s/Hz",
            },
            "arrays": arrays,
            "x": payload["x"],
            "curves": payload["curves"],
            "ci95_low": payload["ci95_low"],
            "ci95_high": payload["ci95_high"],
            "raw_seed_curves": payload["raw_seed_curves"],
            "acceptance_uses_unmodified_raw_seed_curves": True,
            "feasibility_rate": summary["feasibility_rate"],
        }
        evidence_path = atomic_write_json(
            output / f"fig{figure}_data.json", evidence
        )
        artifacts.append({
            "image": image.name,
            "evidence": evidence_path.name,
            "image_sha256": _sha256(image),
            "evidence_sha256": _sha256(evidence_path),
        })

    validation_path = atomic_write_json(
        output / "holdout_validation.json", validation
    )
    atomic_write_json(output / "result_manifest.json", {
        "schema_version": 1,
        "kind": "generated_result_manifest",
        "holdout_summary_sha256": result_payload_digest(summary),
        "paper_reference_sha256": validation["paper_reference_sha256"],
        "validation": validation_path.name,
        "validation_sha256": _sha256(validation_path),
        "artifacts": artifacts,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
