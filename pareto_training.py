"""Deterministic preference sampling, training, resume, and raw evaluation."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Mapping
import warnings

import numpy as np
from scipy.stats import qmc

from checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
    validate_training_histories,
)
from pareto_protocol import validate_preferences


HISTORY_KEYS = (
    "communication",
    "sensing",
    "joint",
    "objective",
    "feasibility",
    "mean_residual_scale",
    "baseline_fallback_rate",
)


class SobolPreferenceStream:
    """Checkpointable scrambled Sobol stream over the two-objective simplex."""

    SCHEMA_VERSION = 1

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("Sobol preference seed must be nonnegative.")
        self.position = 0
        self._engine = qmc.Sobol(d=1, scramble=True, seed=self.seed)

    def next(self, count: int) -> np.ndarray:
        count = int(count)
        if count <= 0:
            raise ValueError("Preference count must be positive.")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The balance properties of Sobol",
                category=UserWarning,
            )
            coordinate = self._engine.random(count).reshape(-1)
        self.position += count
        preferences = np.column_stack([coordinate, 1.0 - coordinate])
        return validate_preferences(preferences)

    def update_batch(self, batch_size: int) -> np.ndarray:
        """Return a batch with exact PNG preference in a fixed first slot."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size == 1:
            return np.array([[0.5, 0.5]], dtype=np.float64)
        return np.vstack(
            [np.array([[0.5, 0.5]], dtype=np.float64), self.next(batch_size - 1)]
        )

    def state_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "seed": self.seed,
            "position": self.position,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ValueError("Unsupported Sobol-preference checkpoint schema.")
        if int(state["seed"]) != self.seed:
            raise ValueError("Sobol-preference seed mismatch.")
        position = int(state["position"])
        if position < 0:
            raise ValueError("Sobol-preference position must be nonnegative.")
        self._engine = qmc.Sobol(d=1, scramble=True, seed=self.seed)
        if position:
            self._engine.fast_forward(position)
        self.position = position

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "SobolPreferenceStream":
        result = cls(int(state["seed"]))
        result.load_state_dict(state)
        return result


def _empty_histories() -> dict[str, list]:
    return {key: [] for key in HISTORY_KEYS}


def _require_step_info(info: Mapping) -> tuple[float, float, np.ndarray]:
    required = (
        "communication_sum_rate",
        "sensing_sum_rate",
        "proposed_action",
        "physical_constraints_feasible",
    )
    missing = [key for key in required if key not in info]
    if missing:
        raise ValueError(f"Step diagnostics are missing required fields: {missing}.")
    communication = float(info["communication_sum_rate"])
    sensing = float(info["sensing_sum_rate"])
    if not np.isfinite(communication) or not np.isfinite(sensing):
        raise ValueError("Physical reward components must be finite.")
    proposed_action = np.asarray(info["proposed_action"], dtype=np.float32)
    if not np.all(np.isfinite(proposed_action)):
        raise ValueError("Proposed action must be finite.")
    return communication, sensing, proposed_action


def train_ba_pcdtl(
    agent,
    environment,
    n_episodes: int,
    identity: Mapping,
    preference_stream: SobolPreferenceStream,
    checkpoint_path,
    *,
    resume: bool = True,
    checkpoint_every: int = 10,
    stop_after_episode: int | None = None,
    fixed_preference=None,
) -> dict:
    """Train from physical reward vectors with exact episode-boundary resume."""
    n_episodes = int(n_episodes)
    checkpoint_every = int(checkpoint_every)
    if n_episodes <= 0 or checkpoint_every <= 0:
        raise ValueError("n_episodes and checkpoint_every must be positive.")
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("Training identity must be a nonempty mapping.")
    if not agent.anchors.frozen:
        raise ValueError("BA-PCDTL training requires preregistered frozen anchors.")
    fixed = None
    if fixed_preference is not None:
        fixed = validate_preferences([fixed_preference])[0]
    checkpoint = Path(checkpoint_path)
    if checkpoint.exists() and not resume:
        raise FileExistsError(
            f"Checkpoint already exists; use resume or a new output: {checkpoint}"
        )

    histories = _empty_histories()
    start_episode = 1
    previous_elapsed = 0.0
    if resume and checkpoint.is_file():
        restored = load_training_checkpoint(
            checkpoint, agent, environment, expected_identity=identity
        )
        histories.update(
            validate_training_histories(
                restored.histories,
                restored.completed_episode,
                HISTORY_KEYS,
                n_episodes,
            )
        )
        stream_state = restored.extra.get("preference_stream")
        if stream_state is None:
            raise ValueError("Checkpoint is missing the preference-stream state.")
        preference_stream.load_state_dict(stream_state)
        start_episode = restored.completed_episode + 1
        previous_elapsed = float(restored.extra.get("elapsed_seconds", 0.0))

    stop_episode = n_episodes
    if stop_after_episode is not None:
        stop_episode = min(stop_episode, int(stop_after_episode))
        if stop_episode < start_episode:
            raise ValueError("stop_after_episode precedes the next unfinished episode.")
    started = time.time() - previous_elapsed

    for episode_index in range(start_episode, stop_episode + 1):
        state = environment.reset()
        totals = {
            "communication": 0.0,
            "sensing": 0.0,
            "feasibility": 0.0,
            "residual_scale": 0.0,
            "fallback": 0.0,
        }
        step_count = 0
        done = False
        while not done:
            behavior_preference = (
                fixed if fixed is not None else preference_stream.next(1)[0]
            )
            action, action_info = agent.select_action(
                state,
                behavior_preference,
                explore=True,
                return_info=True,
            )
            next_state, _ignored_scalar_reward, done, info = environment.step(action)
            communication, sensing, proposed_action = _require_step_info(info)
            agent.replay.push(
                state,
                proposed_action,
                [communication, sensing],
                next_state,
                done,
            )
            if len(agent.replay) >= agent.batch_size:
                update_preferences = (
                    np.repeat(fixed.reshape(1, 2), agent.batch_size, axis=0)
                    if fixed is not None
                    else preference_stream.update_batch(agent.batch_size)
                )
                agent.update(update_preferences)
            state = next_state
            step_count += 1
            totals["communication"] += communication
            totals["sensing"] += sensing
            totals["feasibility"] += float(
                bool(info["physical_constraints_feasible"])
            )
            totals["residual_scale"] += float(action_info["residual_scale"])
            totals["fallback"] += float(action_info["fallback_reason"] is not None)

        if step_count <= 0:
            raise RuntimeError("An episode ended without a transition.")
        communication_mean = totals["communication"] / step_count
        sensing_mean = totals["sensing"] / step_count
        histories["communication"].append(communication_mean)
        histories["sensing"].append(sensing_mean)
        histories["joint"].append(communication_mean + sensing_mean)
        histories["objective"].append(communication_mean + sensing_mean)
        histories["feasibility"].append(totals["feasibility"] / step_count)
        histories["mean_residual_scale"].append(
            totals["residual_scale"] / step_count
        )
        histories["baseline_fallback_rate"].append(totals["fallback"] / step_count)

        if (
            episode_index % checkpoint_every == 0
            or episode_index == n_episodes
            or episode_index == stop_episode
        ):
            save_training_checkpoint(
                checkpoint,
                agent,
                environment,
                episode_index,
                histories,
                identity,
                extra={
                    "elapsed_seconds": time.time() - started,
                    "preference_stream": preference_stream.state_dict(),
                },
            )

    completed_episode = len(histories["joint"])
    return {
        "agent": agent,
        "histories": histories,
        "checkpoint_path": str(checkpoint),
        "completed_episode": completed_episode,
        "complete": completed_episode == n_episodes,
        "elapsed_seconds": time.time() - started,
        "preference_stream": preference_stream.state_dict(),
    }


def _preference_key(preference) -> str:
    return f"comm_{float(preference[0]):.6f}_sense_{float(preference[1]):.6f}"


def evaluate_preference_grid(
    agent,
    environment_factory,
    scene_seeds,
    preferences,
    *,
    use_gate: bool = True,
) -> dict:
    """Evaluate each method/preference separately and retain unsmoothed arrays."""
    seed_list = [int(seed) for seed in scene_seeds]
    if not seed_list or len(seed_list) != len(set(seed_list)):
        raise ValueError("scene_seeds must be a nonempty unique sequence.")
    preference_array = validate_preferences(preferences)
    records = []
    for preference in preference_array:
        metrics = {
            "key": _preference_key(preference),
            "preference": preference.tolist(),
            "communication": [],
            "sensing": [],
            "joint": [],
            "objective": [],
            "minimum_ue_rate": [],
            "minimum_target_rate": [],
            "feasibility": [],
            "mean_residual_scale": [],
            "baseline_fallback_rate": [],
        }
        for seed in seed_list:
            environment = environment_factory(seed)
            state = environment.reset()
            done = False
            communication_values = []
            sensing_values = []
            minimum_ue_values = []
            minimum_target_values = []
            feasibility_values = []
            residual_scales = []
            fallbacks = []
            while not done:
                action, action_info = agent.select_action(
                    state,
                    preference,
                    explore=False,
                    use_gate=use_gate,
                    return_info=True,
                )
                state, _ignored_scalar_reward, done, info = environment.step(action)
                communication, sensing, _proposed_action = _require_step_info(info)
                rate_u = np.asarray(info.get("rate_u"), dtype=np.float64)
                rate_k = np.asarray(info.get("rate_k"), dtype=np.float64)
                if (
                    rate_u.size == 0
                    or rate_k.size == 0
                    or not np.all(np.isfinite(rate_u))
                    or not np.all(np.isfinite(rate_k))
                ):
                    raise ValueError("Evaluation requires finite UE and target rate arrays.")
                communication_values.append(communication)
                sensing_values.append(sensing)
                minimum_ue_values.append(float(np.min(rate_u)))
                minimum_target_values.append(float(np.min(rate_k)))
                feasibility_values.append(
                    float(bool(info["physical_constraints_feasible"]))
                )
                residual_scales.append(float(action_info["residual_scale"]))
                fallbacks.append(float(action_info["fallback_reason"] is not None))
            communication_mean = float(np.mean(communication_values))
            sensing_mean = float(np.mean(sensing_values))
            metrics["communication"].append(communication_mean)
            metrics["sensing"].append(sensing_mean)
            metrics["joint"].append(communication_mean + sensing_mean)
            metrics["objective"].append(communication_mean + sensing_mean)
            metrics["minimum_ue_rate"].append(float(np.min(minimum_ue_values)))
            metrics["minimum_target_rate"].append(
                float(np.min(minimum_target_values))
            )
            metrics["feasibility"].append(float(np.mean(feasibility_values)))
            metrics["mean_residual_scale"].append(float(np.mean(residual_scales)))
            metrics["baseline_fallback_rate"].append(float(np.mean(fallbacks)))
        records.append(metrics)
    return {
        "schema_version": 1,
        "scene_seeds": seed_list,
        "preferences": records,
        "use_gate": bool(use_gate),
    }
