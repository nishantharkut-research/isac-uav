"""Run legacy diagnostic figures from a publication-authorized profile."""
import argparse
import os

from parameter_profiles import activate_profile, load_calibrated_profile


def _parse_args():
    parser = argparse.ArgumentParser(description="Run all HRIS paper figures")
    parser.add_argument(
        "--profile", help="Validated JSON parameter profile to activate"
    )
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = _parse_args()
    if not ARGS.profile:
        raise SystemExit(
            "A validated --profile is required for a full reproduction run."
        )
    if ARGS.profile:
        profile = load_calibrated_profile(ARGS.profile)
        if profile.kind != "identified":
            raise SystemExit(
                "A full figure run requires a frozen identified profile."
            )
        activate_profile(profile)
else:
    ARGS = argparse.Namespace(profile=None)

from simulation import (
    fig4_rewards_vs_episodes,
    fig5_ao_convergence,
    fig6_rate_vs_power,
    fig7_rate_vs_L,
    fig8_rate_vs_scee,
    fig9_active_budget,
    fig10_uav_trajectory,
    fig11_beampattern,
    fig12_transmission_time,
    fig13_rate_vs_delta,
    fig14_rate_vs_delta_bar,
    fig_summary,
    TrainingRegistry,
    set_reproducibility,
    OUT,
)
from cross_figure_validation import validate_result_directory
from scientific_preflight import run_scientific_preflight


def main():
    set_reproducibility()
    run_scientific_preflight(os.path.join(OUT, "scientific_preflight.json"))
    training_registry = TrainingRegistry()
    fig4_result = fig4_rewards_vs_episodes()
    training_runs = dict(fig4_result["training_runs"])
    fig5_ao_convergence()
    training_runs.update(fig6_rate_vs_power(training_registry=training_registry))
    fig7_rate_vs_L(training_registry=training_registry)
    fig8_rate_vs_scee(training_registry=training_registry)
    fig9_active_budget(training_registry=training_registry)
    fig10_uav_trajectory(training_runs[40]["dtl"])
    fig11_beampattern()
    training_runs = fig12_transmission_time(
        training_runs=training_runs, training_registry=training_registry,
    )
    fig13_rate_vs_delta(training_registry=training_registry)
    fig14_rate_vs_delta_bar(training_registry=training_registry)
    fig_summary()
    validation_path = os.path.join(OUT, "cross_figure_validation.json")
    validation = validate_result_directory(OUT, validation_path)
    if not validation["overall_pass"]:
        raise RuntimeError(
            "Full run completed, but cross-figure validation failed. "
            f"See {validation_path}."
        )
    print("\nAll figures done. Check result/ folder.")


if __name__ == "__main__":
    main()
