"""Atomic, full-state checkpoints for reproducible episode-boundary resume."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RestoredTrainingState:
    completed_episode: int
    histories: dict
    identity: dict
    extra: dict


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Checkpoint identity contains unsupported type {type(value)!r}.")


def _canonical_identity(identity: Mapping[str, Any]) -> str:
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("Checkpoint identity must be a non-empty mapping.")
    return json.dumps(
        dict(identity), sort_keys=True, separators=(",", ":"), default=_json_default
    )


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint requires CUDA RNG state, but CUDA is unavailable.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_training_histories(
    histories: Mapping[str, Any],
    completed_episode: int,
    required_keys,
    requested_episodes: int,
) -> dict:
    """Validate that resume histories represent every completed episode."""
    completed_episode = int(completed_episode)
    requested_episodes = int(requested_episodes)
    if completed_episode < 0:
        raise ValueError("completed episode must be non-negative.")
    if requested_episodes < completed_episode:
        raise ValueError(
            "Checkpoint completed episode exceeds the requested episode count."
        )
    if not isinstance(histories, Mapping):
        raise ValueError("Checkpoint histories must be a mapping.")

    validated = dict(histories)
    for key in tuple(required_keys):
        if key not in validated:
            raise ValueError(f"Checkpoint histories are missing {key}.")
        array = np.asarray(validated[key], dtype=float)
        if array.ndim != 1 or array.size != completed_episode:
            raise ValueError(
                f"Checkpoint history {key} length must equal completed episode."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Checkpoint history {key} contains non-finite values.")
    return validated


def save_training_checkpoint(
    path,
    agent,
    environment,
    completed_episode: int,
    histories: Mapping[str, Any],
    identity: Mapping[str, Any],
    extra: Mapping[str, Any] = None,
):
    """Atomically save a complete checkpoint after a finished episode."""
    completed_episode = int(completed_episode)
    if completed_episode < 0:
        raise ValueError("completed_episode must be non-negative.")
    canonical_identity = _canonical_identity(identity)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    environment_state = environment.checkpoint_state()
    if environment_state.get("episode_boundary") is False:
        raise ValueError("Training checkpoints may only be saved at episode boundaries.")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "completed_episode": completed_episode,
        "identity": dict(identity),
        "identity_canonical": canonical_identity,
        "agent": agent.training_state_dict(),
        "environment": environment_state,
        "histories": dict(histories),
        "extra": dict(extra or {}),
        "rng": _capture_rng_state(),
    }
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_training_checkpoint(
    path,
    agent,
    environment,
    expected_identity: Mapping[str, Any],
) -> RestoredTrainingState:
    """Validate identity first, then restore every mutable training component."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    with source.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported training checkpoint schema.")

    expected = _canonical_identity(expected_identity)
    actual = payload.get("identity_canonical")
    if actual != expected:
        raise ValueError(
            "Checkpoint identity mismatch; refusing to combine different runs. "
            f"Expected {expected}, got {actual}."
        )

    agent.load_training_state_dict(payload["agent"])
    environment.restore_checkpoint_state(payload["environment"])
    _restore_rng_state(payload["rng"])
    return RestoredTrainingState(
        completed_episode=int(payload["completed_episode"]),
        histories=dict(payload["histories"]),
        identity=dict(payload["identity"]),
        extra=dict(payload.get("extra", {})),
    )
