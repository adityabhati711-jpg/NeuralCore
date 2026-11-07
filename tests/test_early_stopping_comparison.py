
"""Tests for paired early stopping versus fixed training."""

import csv
import json

import numpy as np
import pytest

from experiments import early_stopping_comparison as esc
from experiments.optimizer_trajectories import build_model
from neuralcore.layers import Dense


def test_parameter_copies_are_independent():
    model = build_model(seed=42)
    copies = esc.parameter_copies(model)

    assert len(copies) == 6

    original = copies[0].copy()

    dense = next(
        layer for layer in model.layers
        if isinstance(layer, Dense)
    )
    dense.weights[0, 0] += 1.0

    np.testing.assert_array_equal(copies[0], original)


def test_restore_parameters():
    model = build_model(seed=42)
    checkpoint = esc.parameter_copies(model)

    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.weights[...] += 3.0
            layer.biases[...] -= 2.0

    esc.restore_parameters(model, checkpoint)

    restored = esc.parameter_copies(model)

    for actual, expected in zip(restored, checkpoint):
        np.testing.assert_array_equal(actual, expected)


def test_restore_rejects_wrong_count():
    model = build_model(seed=42)

    with pytest.raises(ValueError):
        esc.restore_parameters(model, [])


def test_restore_rejects_wrong_shape():
    model = build_model(seed=42)
    checkpoint = esc.parameter_copies(model)
    checkpoint[0] = np.zeros((99, 99))

    with pytest.raises(ValueError):
        esc.restore_parameters(model, checkpoint)


def test_evaluation_returns_finite_metrics():
    model = build_model(seed=42)

    x = np.array([
        [-1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, -0.5],
    ])
    y = np.array([[0], [1], [1], [0]])

    scores = esc.evaluate(model, x, y)

    assert np.isfinite(scores["test_loss"])
    assert 0 <= scores["accuracy"] <= 1
    assert 0 <= scores["f1"] <= 1


@pytest.mark.parametrize(
    "max_epochs,patience,lr",
    [
        (0, 2, 0.01),
        (5, 0, 0.01),
        (5, 2, 0.0),
        (5, 2, -0.01),
        (5, 2, float("nan")),
    ],
)
def test_invalid_run_configuration(max_epochs, patience, lr):
    with pytest.raises(ValueError):
        esc.run_seed(
            42,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=lr,
        )


def test_run_seed_returns_paired_results():
    result = esc.run_seed(
        seed=42,
        max_epochs=12,
        patience=3,
        learning_rate=0.01,
    )

    assert len(result["history"]) == 12
    assert result["fixed"]["seed"] == 42
    assert result["early"]["seed"] == 42

    assert result["fixed"]["epochs_run"] == 12
    assert 1 <= result["early"]["epochs_run"] <= 12
    assert (
        result["early"]["checkpoint_epoch"]
        <= result["early"]["epochs_run"]
    )

    for method in ("fixed", "early"):
        assert 0 <= result[method]["accuracy"] <= 1
        assert np.isfinite(result[method]["test_loss"])


def test_run_seed_is_reproducible():
    first = esc.run_seed(
        42, max_epochs=12, patience=3
    )
    second = esc.run_seed(
        42, max_epochs=12, patience=3
    )

    for method in ("fixed", "early"):
        assert first[method] == second[method]

    assert first["history"] == second["history"]


def sample_results():
    return [
        {
            "seed": 1, "method": "Fixed",
            "accuracy": 0.8, "f1": 0.8,
            "test_loss": 0.3,
            "epochs_run": 100,
            "checkpoint_epoch": 100,
        },
        {
            "seed": 1, "method": "EarlyStopping",
            "accuracy": 0.9, "f1": 0.9,
            "test_loss": 0.2,
            "epochs_run": 60,
            "checkpoint_epoch": 40,
        },
        {
            "seed": 2, "method": "Fixed",
            "accuracy": 0.7, "f1": 0.7,
            "test_loss": 0.4,
            "epochs_run": 100,
            "checkpoint_epoch": 100,
        },
        {
            "seed": 2, "method": "EarlyStopping",
            "accuracy": 0.8, "f1": 0.8,
            "test_loss": 0.3,
            "epochs_run": 70,
            "checkpoint_epoch": 50,
        },
    ]


def test_summary_means():
    summary = esc.summarize(sample_results())

    assert summary["Fixed"]["accuracy"]["mean"] == pytest.approx(0.75)
    assert summary["EarlyStopping"]["accuracy"]["mean"] == pytest.approx(0.85)
    assert summary["EarlyStopping"]["epochs_run"]["mean"] == pytest.approx(65)


def test_paired_bootstrap_difference():
    comparison = esc.paired_bootstrap(
        sample_results(),
        metric="accuracy",
        n_resamples=1000,
        seed=42,
    )

    assert comparison["n_pairs"] == 2
    assert comparison["mean_difference"] == pytest.approx(0.1)
    assert comparison["bootstrap_ci_95"][0] <= 0.1
    assert comparison["bootstrap_ci_95"][1] >= 0.1


def test_paired_bootstrap_is_reproducible():
    first = esc.paired_bootstrap(
        sample_results(), "test_loss",
        n_resamples=1000, seed=42
    )
    second = esc.paired_bootstrap(
        sample_results(), "test_loss",
        n_resamples=1000, seed=42
    )

    assert first == second


def test_paired_bootstrap_rejects_missing_seed():
    results = sample_results()
    results.pop()

    with pytest.raises(ValueError):
        esc.paired_bootstrap(results, "accuracy")


def test_output_files_created(tmp_path, monkeypatch):
    monkeypatch.setattr(esc, "OUTPUT", tmp_path)

    results = sample_results()
    histories = {
        1: [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
            {"epoch": 2, "train_loss": 0.4, "val_loss": 0.5},
        ],
        2: [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
            {"epoch": 2, "train_loss": 0.4, "val_loss": 0.5},
        ],
    }

    summary = esc.summarize(results)
    comparisons = {
        "accuracy": esc.paired_bootstrap(
            results, "accuracy", n_resamples=100
        )
    }

    esc.save_outputs(
        results, histories, summary, comparisons
    )

    expected = [
        "runs.csv",
        "learning_curves.csv",
        "summary.json",
        "accuracy_comparison.png",
        "paired_test_loss.png",
        "early_stopping_curve.png",
    ]

    for filename in expected:
        assert (tmp_path / filename).is_file()

    with (tmp_path / "summary.json").open(
        encoding="utf-8"
    ) as file:
        report = json.load(file)

    assert "paired_comparisons" in report

    with (tmp_path / "runs.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 4


def test_early_checkpoint_is_within_training_budget():
    result = esc.run_seed(
        43, max_epochs=20, patience=4
    )

    early = result["early"]

    assert early["checkpoint_epoch"] >= 1
    assert early["checkpoint_epoch"] <= early["epochs_run"]
    assert early["epochs_run"] <= 20
