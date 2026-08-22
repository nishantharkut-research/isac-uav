"""Shared wall-clock progress calculations for long training runs."""

import math


PROGRESS_INTERVAL_EPISODES = 25


def progress_snapshot(completed_episodes, total_episodes, elapsed_seconds):
    """Return rate and ETA derived only from completed episode boundaries."""
    completed = int(completed_episodes)
    total = int(total_episodes)
    elapsed = float(elapsed_seconds)
    if (
        completed <= 0
        or total <= 0
        or completed > total
        or not math.isfinite(elapsed)
        or elapsed < 0.0
    ):
        raise ValueError("Progress inputs must describe a valid completed prefix.")
    seconds_per_episode = elapsed / completed
    return {
        "completed_episodes": completed,
        "total_episodes": total,
        "elapsed_seconds": elapsed,
        "seconds_per_episode": seconds_per_episode,
        "eta_seconds": seconds_per_episode * (total - completed),
        "completion_fraction": completed / total,
    }
