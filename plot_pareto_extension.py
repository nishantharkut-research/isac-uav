"""Plot BA-PCDTL raw convergence, paired deltas, and Pareto evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from experiment_protocol import atomic_write_json
from source_integrity import source_tree_digest
from validate_pareto_extension import validate_pareto_result
from validate_results import result_payload_digest


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mean_ci(values, axis=0):
    array = np.asarray(values, dtype=np.float64)
    if array.shape[axis] < 2 or not np.all(np.isfinite(array)):
        raise ValueError("Confidence intervals require finite replicated arrays.")
    mean = np.mean(array, axis=axis)
    standard_error = np.std(array, axis=axis, ddof=1) / np.sqrt(array.shape[axis])
    half_width = t.ppf(0.975, array.shape[axis] - 1) * standard_error
    return mean, mean - half_width, mean + half_width


def _causal_smooth(values, window):
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    for index in range(array.shape[-1]):
        result[..., index] = np.mean(
            array[..., max(0, index - window + 1):index + 1], axis=-1
        )
    return result


def build_plot_data(summary, smoothing_window=20):
    records = summary.get("research_seed_records", [])
    seeds = summary.get("research_seeds", [])
    preferences = summary.get("preference_grid", [])
    episodes = int(summary.get("episodes", 0))
    if len(records) != len(seeds) or len(seeds) < 10 or episodes <= 0:
        raise ValueError("Summary has incomplete research training records.")
    ba_histories = []
    scalar_histories = []
    proposed_joint = []
    scalar_joint = []
    proposed_points = [[] for _ in preferences]
    independent_points = [[] for _ in preferences]
    for record in records:
        vector_method = record["training"]["vector"]["methods"]["ba_pcdtl"]
        ba_history = np.asarray(
            vector_method["training"]["histories"]["joint"], dtype=np.float64
        )
        scalar_history = np.asarray(
            record["training"]["scalar"]["methods"]["same_objective_dtl"]
            ["training"]["joint_rates"],
            dtype=np.float64,
        )
        if (
            ba_history.shape != (episodes,)
            or scalar_history.shape != (episodes,)
            or not np.all(np.isfinite(ba_history))
            or not np.all(np.isfinite(scalar_history))
        ):
            raise ValueError("Training histories are incomplete or nonfinite.")
        ba_histories.append(ba_history)
        scalar_histories.append(scalar_history)

        proposed = record["test"]["ba_pcdtl"]["preferences"]
        if len(proposed) != len(preferences):
            raise ValueError("Proposed test preference grid is incomplete.")
        balanced_index = next(
            index for index, preference in enumerate(preferences)
            if np.allclose(preference, [0.5, 0.5], atol=1e-8, rtol=0.0)
        )
        proposed_joint.append(float(np.mean(proposed[balanced_index]["joint"])))
        scalar_joint.append(
            float(np.mean(record["test"]["same_objective_dtl"]["joint"]))
        )
        for index, item in enumerate(proposed):
            proposed_points[index].append([
                float(np.mean(item["communication"])),
                float(np.mean(item["sensing"])),
            ])
        independent = record["test"].get("independent_weighted_policies")
        if independent is not None:
            policies = independent.get("policies", [])
            if len(policies) != len(preferences):
                raise ValueError("Independent-policy test grid is incomplete.")
            for index, policy in enumerate(policies):
                evaluation = policy["evaluation"]
                independent_points[index].append([
                    float(np.mean(evaluation["communication"])),
                    float(np.mean(evaluation["sensing"])),
                ])

    ba_raw = np.stack(ba_histories)
    scalar_raw = np.stack(scalar_histories)
    ba_visual = _causal_smooth(ba_raw, smoothing_window)
    scalar_visual = _causal_smooth(scalar_raw, smoothing_window)
    ba_mean, ba_low, ba_high = _mean_ci(ba_visual)
    scalar_mean, scalar_low, scalar_high = _mean_ci(scalar_visual)
    deltas = np.asarray(proposed_joint) - np.asarray(scalar_joint)
    delta_mean, delta_low, delta_high = _mean_ci(deltas)

    proposed_array = np.asarray(proposed_points, dtype=np.float64)
    proposed_mean = np.mean(proposed_array, axis=1)
    data = {
        "convergence": {
            "episodes": list(range(1, episodes + 1)),
            "ba_raw": ba_raw.tolist(),
            "scalar_raw": scalar_raw.tolist(),
            "ba_visual_mean": ba_mean.tolist(),
            "ba_visual_ci_low": ba_low.tolist(),
            "ba_visual_ci_high": ba_high.tolist(),
            "scalar_visual_mean": scalar_mean.tolist(),
            "scalar_visual_ci_low": scalar_low.tolist(),
            "scalar_visual_ci_high": scalar_high.tolist(),
            "smoothing_window": int(smoothing_window),
        },
        "paired_delta": {
            "seeds": list(seeds),
            "proposed_joint": proposed_joint,
            "same_objective_joint": scalar_joint,
            "delta": deltas.tolist(),
            "mean": float(delta_mean),
            "ci_low": float(delta_low),
            "ci_high": float(delta_high),
        },
        "pareto": {
            "preferences": preferences,
            "proposed_raw": proposed_array.tolist(),
            "proposed_mean": proposed_mean.tolist(),
        },
    }
    if independent_points and all(independent_points):
        independent_array = np.asarray(independent_points, dtype=np.float64)
        data["pareto"]["independent_raw"] = independent_array.tolist()
        data["pareto"]["independent_mean"] = np.mean(
            independent_array, axis=1
        ).tolist()
    return data


def _common_evidence(summary):
    return {
        "schema_version": 1,
        "kind": "ba_pcdtl_generated_evidence",
        "algorithm_id": summary["algorithm_id"],
        "source_tree_digest": summary["source_tree_digest"],
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
        "cross_function_interference_hypothesis_id": summary[
            "cross_function_interference_hypothesis_id"
        ],
        "baseline_payload_sha256": summary["baseline_payload_sha256"],
        "baseline_checkpoint_authority": summary[
            "baseline_checkpoint_authority"
        ],
        "gate_selection_lock_sha256": summary["gate_selection_lock_sha256"],
        "research_seeds": summary["research_seeds"],
        "seed_roles": {
            "research": summary["research_seeds"],
            "validation": summary["validation_scene_seeds"],
            "test": summary["test_scene_seeds"],
        },
        "preference_grid": summary["preference_grid"],
        "anchors": summary["anchors"],
        "acceptance_uses_raw_arrays": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--baseline-validation", required=True)
    parser.add_argument("--validation-level", choices=("preliminary", "publication"), default="preliminary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoothing-window", type=int, default=20)
    args = parser.parse_args(argv)

    summary_path = Path(args.summary).resolve()
    summary = _read_json(summary_path)
    if summary.get("source_tree_digest") != source_tree_digest():
        raise ValueError("Summary was generated by a different final source tree.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Plot output directory must be empty.")
    validation = validate_pareto_result(
        summary,
        level=args.validation_level,
        result_directory=summary_path.parent,
        verify_files=True,
        baseline=_read_json(args.baseline_summary),
        baseline_validation=_read_json(args.baseline_validation),
        baseline_directory=Path(args.baseline_summary).resolve().parent,
    )
    data = build_plot_data(summary, args.smoothing_window)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = []
    common = _common_evidence(summary)
    common.update({
        "extension_summary_sha256": result_payload_digest(summary),
        "baseline_validation_sha256": summary[
            "baseline_validation_sha256"
        ],
        "validation_level": args.validation_level,
    })

    convergence = data["convergence"]
    x = np.asarray(convergence["episodes"])
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for label, prefix, color in (
        ("BA-PCDTL", "ba", "#146c94"),
        ("Same-objective DTL", "scalar", "#b33b32"),
    ):
        axis.plot(x, convergence[f"{prefix}_visual_mean"], label=label,
                  color=color, linewidth=1.8)
        axis.fill_between(
            x,
            convergence[f"{prefix}_visual_ci_low"],
            convergence[f"{prefix}_visual_ci_high"],
            color=color, alpha=0.15, linewidth=0,
        )
    axis.set_xlabel("Episode")
    axis.set_ylabel("Joint sum rate (bits/s/Hz)")
    axis.set_title("Training convergence")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    convergence_image = output / "training_convergence.png"
    figure.savefig(convergence_image, dpi=220, bbox_inches="tight")
    plt.close(figure)
    convergence_evidence = atomic_write_json(
        output / "training_convergence.json",
        {
            **common,
            "artifact_type": "training_convergence",
            "units": {"episodes": "episode", "rates": "bits/s/Hz"},
            "arrays": {
                key: value for key, value in convergence.items()
                if key != "smoothing_window"
            },
            "smoothing": {
                "purpose": "visualization_only",
                "method": "causal_moving_average",
                "window": convergence["smoothing_window"],
            },
        },
    )
    artifacts.append(_artifact("training_convergence", convergence_image, convergence_evidence))

    delta = data["paired_delta"]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.axhline(0.0, color="#303030", linewidth=1.0)
    axis.scatter(delta["seeds"], delta["delta"], color="#146c94", s=34)
    axis.axhline(delta["mean"], color="#b33b32", linewidth=1.8, label="Mean delta")
    axis.fill_between(
        [min(delta["seeds"]), max(delta["seeds"])],
        delta["ci_low"], delta["ci_high"], color="#b33b32", alpha=0.14,
        label="Paired 95% CI",
    )
    axis.set_xlabel("Research seed")
    axis.set_ylabel("Joint-rate delta (bits/s/Hz)")
    axis.set_title("BA-PCDTL minus same-objective DTL")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    delta_image = output / "primary_paired_delta.png"
    figure.savefig(delta_image, dpi=220, bbox_inches="tight")
    plt.close(figure)
    delta_evidence = atomic_write_json(
        output / "primary_paired_delta.json",
        {
            **common,
            "artifact_type": "primary_paired_delta",
            "units": {"seeds": "seed", "rates": "bits/s/Hz"},
            "arrays": {
                key: value for key, value in delta.items()
                if key in {"seeds", "proposed_joint", "same_objective_joint", "delta"}
            },
            "paired_statistics": {
                "mean": delta["mean"],
                "ci_low": delta["ci_low"],
                "ci_high": delta["ci_high"],
            },
        },
    )
    artifacts.append(_artifact("primary_paired_delta", delta_image, delta_evidence))

    pareto = data["pareto"]
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    proposed = np.asarray(pareto["proposed_mean"])
    axis.plot(proposed[:, 0], proposed[:, 1], color="#146c94", marker="o",
              linewidth=1.8, label="BA-PCDTL")
    if "independent_mean" in pareto:
        independent = np.asarray(pareto["independent_mean"])
        axis.plot(independent[:, 0], independent[:, 1], color="#b33b32",
                  marker="s", linewidth=1.5, label="Independent policies")
    axis.set_xlabel("Communication sum rate (bits/s/Hz)")
    axis.set_ylabel("Sensing sum rate (bits/s/Hz)")
    axis.set_title("Communication-sensing Pareto comparison")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    pareto_image = output / "pareto_front.png"
    figure.savefig(pareto_image, dpi=220, bbox_inches="tight")
    plt.close(figure)
    pareto_evidence = atomic_write_json(
        output / "pareto_front.json",
        {
            **common,
            "artifact_type": "pareto_front",
            "units": {"rates": "bits/s/Hz", "preferences": "simplex weight"},
            "arrays": pareto,
        },
    )
    artifacts.append(_artifact("pareto_front", pareto_image, pareto_evidence))

    validation_path = atomic_write_json(
        output / "pareto_validation.json", validation
    )
    atomic_write_json(output / "result_manifest.json", {
        "schema_version": 1,
        "kind": "ba_pcdtl_result_manifest",
        "extension_summary_sha256": result_payload_digest(summary),
        "baseline_payload_sha256": summary["baseline_payload_sha256"],
        "baseline_validation_sha256": summary[
            "baseline_validation_sha256"
        ],
        "validation_level": args.validation_level,
        "validation": validation_path.name,
        "validation_sha256": _sha256(validation_path),
        "artifacts": artifacts,
    })
    return 0 if validation["overall_pass"] else 1


def _artifact(artifact_type, image, evidence):
    return {
        "artifact_type": artifact_type,
        "image": Path(image).name,
        "evidence": Path(evidence).name,
        "image_sha256": _sha256(image),
        "evidence_sha256": _sha256(evidence),
    }


if __name__ == "__main__":
    raise SystemExit(main())
