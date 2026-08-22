"""Stable digests for checkpoint source-code and transfer-model identity."""

import hashlib
from pathlib import Path
from typing import Mapping

import torch


ROOT = Path(__file__).resolve().parent


def _update_tensor(digest, name, tensor):
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest.update(raw)


def source_model_digest(agent) -> str:
    """Hash only source policy/value networks, independent of replay state."""
    digest = hashlib.sha256()
    for module_name in ("actor", "critic", "actor_tgt", "critic_tgt"):
        module = getattr(agent, module_name, None)
        if module is None:
            raise ValueError(f"Source agent has no {module_name} network")
        digest.update(module_name.encode("ascii"))
        for key, value in sorted(module.state_dict().items()):
            _update_tensor(digest, key, value)
    return digest.hexdigest()


def source_model_state_digest(agent_state: Mapping) -> str:
    """Hash source-network state serialized inside a training checkpoint."""
    if not isinstance(agent_state, Mapping):
        raise ValueError("Serialized source agent state must be a mapping.")
    digest = hashlib.sha256()
    for module_name in ("actor", "critic", "actor_tgt", "critic_tgt"):
        module_state = agent_state.get(module_name)
        if not isinstance(module_state, Mapping) or not module_state:
            raise ValueError(
                f"Serialized source agent has no {module_name} network state."
            )
        digest.update(module_name.encode("ascii"))
        for key, value in sorted(module_state.items()):
            if not torch.is_tensor(value):
                raise ValueError(
                    f"Serialized source network entry {key!r} is not a tensor."
                )
            _update_tensor(digest, key, value)
    return digest.hexdigest()


def policy_model_digest(module) -> str:
    """Hash one policy module independently of optimizer or replay state."""
    if not hasattr(module, "state_dict"):
        raise ValueError("Policy object must provide state_dict().")
    state = module.state_dict()
    if not state:
        raise ValueError("Policy state is empty.")
    digest = hashlib.sha256()
    digest.update(type(module).__name__.encode("utf-8"))
    for key, value in sorted(state.items()):
        if not torch.is_tensor(value):
            raise ValueError(f"Policy state entry {key!r} is not a tensor.")
        _update_tensor(digest, key, value)
    return digest.hexdigest()


def source_tree_digest(root=None) -> str:
    """Hash executable source files used by a run, excluding tests/results."""
    root = Path(root or ROOT).resolve()
    if not root.is_dir():
        raise ValueError(f"Source root is not a directory: {root}")
    paths = sorted(
        (
            path for path in root.glob("*.py")
            if not path.name.startswith("test_")
        ),
        key=lambda path: path.name,
    )
    requirements = root / "requirements.txt"
    if requirements.is_file():
        paths.append(requirements)
    if not paths:
        raise ValueError(f"Source root contains no executable source: {root}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
