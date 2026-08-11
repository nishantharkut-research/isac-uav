"""Replay storage for physical communication and sensing reward vectors."""

from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np
import torch


class VectorReplayBuffer:
    """Fixed-capacity replay buffer with a validated vector-reward contract."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        reward_dim: int = 2,
    ) -> None:
        dimensions = {
            "capacity": capacity,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "reward_dim": reward_dim,
        }
        if any(int(value) <= 0 for value in dimensions.values()):
            raise ValueError("Replay capacity and dimensions must be positive.")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.reward_dim = int(reward_dim)
        self.buffer: deque[tuple[np.ndarray, ...]] = deque(maxlen=self.capacity)

    @staticmethod
    def _finite_vector(value: Iterable[float], size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array.copy()

    def push(self, state, action, reward, next_state, done) -> None:
        state_array = self._finite_vector(state, self.state_dim, "state")
        action_array = self._finite_vector(action, self.action_dim, "action")
        reward_array = self._finite_vector(reward, self.reward_dim, "reward")
        next_state_array = self._finite_vector(
            next_state, self.state_dim, "next_state"
        )
        done_value = float(done)
        if not np.isfinite(done_value) or done_value not in (0.0, 1.0):
            raise ValueError("done must be finite and equal to zero or one.")
        self.buffer.append(
            (state_array, action_array, reward_array, next_state_array, done_value)
        )

    def sample(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(self.buffer) < batch_size:
            raise ValueError("Replay does not contain enough transitions to sample.")
        state, action, reward, next_state, done = map(
            np.stack, zip(*random.sample(self.buffer, batch_size))
        )
        return (
            torch.as_tensor(state, dtype=torch.float32),
            torch.as_tensor(action, dtype=torch.float32),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(next_state, dtype=torch.float32),
            torch.as_tensor(done, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()

    def state_dict(self) -> dict:
        transitions = [
            tuple(item.copy() if isinstance(item, np.ndarray) else item for item in row)
            for row in self.buffer
        ]
        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "reward_dim": self.reward_dim,
            "transitions": transitions,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ValueError("Unsupported vector-replay checkpoint schema.")
        saved_dimensions = (
            int(state["state_dim"]),
            int(state["action_dim"]),
            int(state["reward_dim"]),
        )
        current_dimensions = (self.state_dim, self.action_dim, self.reward_dim)
        if saved_dimensions != current_dimensions:
            raise ValueError(
                "Replay dimensions mismatch: "
                f"expected {current_dimensions}, got {saved_dimensions}."
            )
        capacity = int(state["capacity"])
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        transitions = state.get("transitions", [])
        if len(transitions) > capacity:
            raise ValueError("Serialized replay exceeds its declared capacity.")

        restored = VectorReplayBuffer(
            capacity, self.state_dim, self.action_dim, self.reward_dim
        )
        for transition in transitions:
            if not isinstance(transition, (tuple, list)) or len(transition) != 5:
                raise ValueError("Each replay transition must contain five fields.")
            restored.push(*transition)
        self.capacity = capacity
        self.buffer = restored.buffer
