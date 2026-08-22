import pytest

from training_progress import progress_snapshot


def test_progress_snapshot_computes_rate_and_eta():
    snapshot = progress_snapshot(
        completed_episodes=25,
        total_episodes=100,
        elapsed_seconds=50.0,
    )

    assert snapshot == {
        "completed_episodes": 25,
        "total_episodes": 100,
        "elapsed_seconds": 50.0,
        "seconds_per_episode": 2.0,
        "eta_seconds": 150.0,
        "completion_fraction": 0.25,
    }


@pytest.mark.parametrize(
    "completed,total,elapsed",
    [
        (0, 100, 1.0),
        (101, 100, 1.0),
        (1, 0, 1.0),
        (1, 100, -1.0),
        (1, 100, float("inf")),
    ],
)
def test_progress_snapshot_rejects_invalid_inputs(completed, total, elapsed):
    with pytest.raises(ValueError):
        progress_snapshot(completed, total, elapsed)
