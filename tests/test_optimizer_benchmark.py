
"""Tests for NeuralCore's multi-seed optimizer benchmark."""

import numpy as np
import pytest

from experiments.optimizer_benchmark import (
    prepare_data,
    classification_metrics,
    run_experiment,
    summarize,
    paired_comparisons,
)


def test_data_shapes_and_finite_values():
    data = prepare_data(42)
    assert len(data) == 6

    x_train, y_train, x_val, y_val, x_test, y_test = data

    assert len(x_train) + len(x_val) + len(x_test) == 600

    for x, y in (
        (x_train, y_train),
        (x_val, y_val),
        (x_test, y_test),
    ):
        assert x.shape[1] == 2
        assert y.shape == (len(x), 1)
        assert np.all(np.isfinite(x))


def test_metrics_perfect_predictions():
    y = np.array([[0], [1], [1], [0]])
    p = np.array([[0.1], [0.9], [0.8], [0.2]])

    result = classification_metrics(y, p)

    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["fp"] == 0
    assert result["fn"] == 0


@pytest.mark.parametrize("optimizer", ["SGD", "Momentum", "Adam"])
def test_optimizer_run(optimizer):
    result = run_experiment(
        42, optimizer, max_epochs=8, patience=4
    )

    assert 1 <= result["best_epoch"] <= 8
    assert 1 <= result["epochs_run"] <= 8
    assert np.isfinite(result["test_loss"])
    assert 0 <= result["accuracy"] <= 1
    assert len(result["history"]) == result["epochs_run"]


def test_identical_initialization():
    results = [
        run_experiment(42, name, max_epochs=2)
        for name in ("SGD", "Momentum", "Adam")
    ]

    for result in results[1:]:
        np.testing.assert_array_equal(
            results[0]["initial_weights"],
            result["initial_weights"],
        )


def test_reproducibility():
    first = run_experiment(42, "Adam", max_epochs=8)
    second = run_experiment(42, "Adam", max_epochs=8)

    assert first["test_loss"] == second["test_loss"]
    assert first["best_epoch"] == second["best_epoch"]


def test_summary_and_paired_comparisons():
    results = []

    for seed in (42, 43):
        for name in ("SGD", "Momentum", "Adam"):
            results.append({
                "seed": seed,
                "optimizer": name,
                "accuracy": 0.8 + 0.01 * (seed - 42),
                "f1": 0.8,
                "test_loss": 0.2,
                "best_val_loss": 0.2,
                "best_epoch": 10,
            })

    summary = summarize(results)
    comparisons = paired_comparisons(results)

    assert len(summary) == 3
    assert len(comparisons) == 3
    assert comparisons["SGD_minus_Adam"]["n_pairs"] == 2
    assert comparisons["SGD_minus_Adam"]["mean_difference"] == 0


@pytest.mark.parametrize("epochs,patience", [
    (0, 10),
    (10, 0),
    (-1, 5),
])
def test_invalid_training_configuration(epochs, patience):
    with pytest.raises(ValueError):
        run_experiment(
            42, "Adam",
            max_epochs=epochs,
            patience=patience,
        )
