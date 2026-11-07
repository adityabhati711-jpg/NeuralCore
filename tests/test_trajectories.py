
"""Tests for NeuralCore optimizer trajectory experiments."""

import numpy as np
import pytest

from experiments import optimizer_trajectories as traj
from neuralcore.landscape import flatten_parameters


def test_model_initialization_is_reproducible():
    first = traj.build_model(seed=42)
    second = traj.build_model(seed=42)

    np.testing.assert_array_equal(
        flatten_parameters(first),
        flatten_parameters(second),
    )


def test_different_seeds_change_initial_parameters():
    first = traj.build_model(seed=42)
    second = traj.build_model(seed=43)

    assert not np.array_equal(
        flatten_parameters(first),
        flatten_parameters(second),
    )


def test_model_output_shape():
    model = traj.build_model(seed=42)
    x = np.zeros((5, 2))

    assert model.forward(x).shape == (5, 1)


def test_data_preparation_is_reproducible():
    first = traj.prepare_data(seed=42)
    second = traj.prepare_data(seed=42)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)

    x, y = first

    assert x.shape == (420, 2)
    assert y.shape == (420, 1)
    assert np.all(np.isfinite(x))
    assert np.allclose(x.mean(axis=0), 0, atol=1e-12)


def test_trajectory_records_every_epoch():
    x, y = traj.prepare_data(seed=42)

    result = traj.run_optimizer(
        "Adam", x, y, epochs=4
    )

    assert result["trajectory"].shape == (5, 193)
    assert result["losses"].shape == (5,)
    assert np.all(np.isfinite(result["trajectory"]))
    assert np.all(np.isfinite(result["losses"]))


def test_optimizers_share_identical_initial_weights():
    x, y = traj.prepare_data(seed=42)

    results = {
        name: traj.run_optimizer(
            name, x, y, epochs=3
        )
        for name in traj.OPTIMIZERS
    }

    initial = [
        result["trajectory"][0]
        for result in results.values()
    ]

    for vector in initial[1:]:
        np.testing.assert_array_equal(
            initial[0], vector
        )


def test_adam_reduces_training_loss():
    x, y = traj.prepare_data(seed=42)

    result = traj.run_optimizer(
        "Adam", x, y, epochs=30
    )

    assert result["losses"][-1] < result["losses"][0]


def test_unknown_optimizer_rejected():
    x, y = traj.prepare_data(seed=42)

    with pytest.raises(ValueError):
        traj.run_optimizer(
            "Unknown", x, y, epochs=3
        )


@pytest.mark.parametrize("epochs", [0, -3])
def test_invalid_epoch_budget_rejected(epochs):
    x, y = traj.prepare_data(seed=42)

    with pytest.raises(ValueError):
        traj.run_optimizer(
            "Adam", x, y, epochs=epochs
        )


def test_main_saves_trajectory_artifact(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(traj, "OUTPUT", tmp_path)

    original_run = traj.run_optimizer

    def short_run(name, x, y, epochs=traj.EPOCHS):
        return original_run(
            name, x, y, epochs=3
        )

    monkeypatch.setattr(
        traj, "run_optimizer", short_run
    )

    traj.main()

    path = tmp_path / "optimizer_trajectories.npz"

    assert path.is_file()

    with np.load(path) as saved:
        for name in ("sgd", "momentum", "adam"):
            assert f"{name}_trajectory" in saved
            assert f"{name}_losses" in saved
            assert saved[f"{name}_trajectory"].shape == (
                4, 193
            )
            assert saved[f"{name}_losses"].shape == (4,)

        np.testing.assert_array_equal(
            saved["sgd_trajectory"][0],
            saved["adam_trajectory"][0],
        )
