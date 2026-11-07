
"""Tests for NeuralCore's research-oriented hyperparameter search."""

import csv
import json

import numpy as np
import pytest

from experiments import hyperparameter_search as hs


# ---------------------------------------------------------
# 1. Dataset and reproducibility tests
# ---------------------------------------------------------

def test_dataset_shapes():
    data = hs.dataset(100)

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
        assert np.all(np.isfinite(y))


def test_dataset_reproducibility():
    first = hs.dataset(100)
    second = hs.dataset(100)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_training_only_standardization():
    x_train, _, _, _, _, _ = hs.dataset(100)

    np.testing.assert_allclose(
        x_train.mean(axis=0),
        np.zeros(2),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        x_train.std(axis=0),
        np.ones(2),
        atol=1e-12,
    )


def test_tuning_and_evaluation_seeds_disjoint():
    assert set(hs.TUNING_SEEDS).isdisjoint(
        hs.EVALUATION_SEEDS
    )

    assert len(hs.TUNING_SEEDS) == 10
    assert len(hs.EVALUATION_SEEDS) == 20


# ---------------------------------------------------------
# 2. Optimizer configuration tests
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected_type",
    [
        ("SGD", hs.SGD),
        ("Momentum", hs.Momentum),
        ("Adam", hs.Adam),
    ],
)
def test_optimizer_factory(name, expected_type):
    optimizer = hs.optimizer_for(name, 0.01)

    assert isinstance(optimizer, expected_type)


def test_unknown_optimizer_rejected():
    with pytest.raises(ValueError):
        hs.optimizer_for("UnknownOptimizer", 0.01)


def test_learning_rate_grids():
    assert set(hs.GRIDS) == {
        "SGD", "Momentum", "Adam"
    }

    for rates in hs.GRIDS.values():
        assert len(rates) == 5
        assert all(np.isfinite(lr) and lr > 0 for lr in rates)
        assert len(set(rates)) == len(rates)


# ---------------------------------------------------------
# 3. Training engine tests
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["SGD", "Momentum", "Adam"],
)
def test_short_tuning_run(name):
    row = hs.train(
        seed=100,
        name=name,
        lr=0.01,
        max_epochs=8,
        patience=4,
        stage="tuning",
    )

    assert row["stage"] == "tuning"
    assert row["optimizer"] == name
    assert 1 <= row["best_epoch"] <= 8
    assert 1 <= row["epochs_run"] <= 8
    assert np.isfinite(row["best_val_loss"])

    # Tuning must not produce test metrics.
    assert row["test_loss"] == ""
    assert row["accuracy"] == ""
    assert row["f1"] == ""


def test_short_evaluation_run():
    row = hs.train(
        seed=1000,
        name="Adam",
        lr=0.01,
        max_epochs=8,
        patience=4,
        stage="evaluation",
    )

    assert row["stage"] == "evaluation"
    assert np.isfinite(row["test_loss"])
    assert 0 <= row["accuracy"] <= 1
    assert 0 <= row["f1"] <= 1


def test_training_reproducibility():
    kwargs = {
        "seed": 101,
        "name": "Adam",
        "lr": 0.01,
        "max_epochs": 10,
        "patience": 5,
        "stage": "tuning",
    }

    first = hs.train(**kwargs)
    second = hs.train(**kwargs)

    assert first == second


@pytest.mark.parametrize(
    "epochs,patience",
    [
        (0, 10),
        (-1, 10),
        (10, 0),
        (10, -1),
        (True, 10),
        (10, False),
    ],
)
def test_invalid_training_configuration(epochs, patience):
    with pytest.raises(ValueError):
        hs.train(
            seed=100,
            name="Adam",
            lr=0.01,
            max_epochs=epochs,
            patience=patience,
        )


def test_invalid_training_stage():
    with pytest.raises(ValueError):
        hs.train(
            seed=100,
            name="Adam",
            lr=0.01,
            max_epochs=5,
            stage="invalid",
        )


# ---------------------------------------------------------
# 4. CSV saving and resume tests
# ---------------------------------------------------------

def test_csv_roundtrip(tmp_path):
    path = tmp_path / "runs.csv"

    row = hs.train(
        seed=100,
        name="Adam",
        lr=0.01,
        max_epochs=5,
    )

    hs.append_row(path, row)
    loaded = hs.read_rows(path)

    assert len(loaded) == 1
    assert loaded[0]["optimizer"] == "Adam"
    assert int(loaded[0]["seed"]) == 100
    assert float(loaded[0]["lr"]) == 0.01


def test_csv_append_multiple_rows(tmp_path):
    path = tmp_path / "runs.csv"

    for seed in (100, 101):
        row = hs.train(
            seed=seed,
            name="SGD",
            lr=0.01,
            max_epochs=3,
        )
        hs.append_row(path, row)

    loaded = hs.read_rows(path)

    assert len(loaded) == 2
    assert {int(r["seed"]) for r in loaded} == {
        100, 101
    }


def test_missing_csv_returns_empty_list(tmp_path):
    assert hs.read_rows(
        tmp_path / "missing.csv"
    ) == []


def test_run_keys_are_unique_across_stages():
    tuning_key = hs.key(
        "tuning", 100, "Adam", 0.01
    )
    evaluation_key = hs.key(
        "evaluation", 100, "Adam", 0.01
    )

    assert tuning_key != evaluation_key


# ---------------------------------------------------------
# 5. Validation-only hyperparameter selection
# ---------------------------------------------------------

def synthetic_tuning_rows():
    rows = []

    # Adam should select 0.001, not 0.01.
    for seed in (100, 101, 102):
        for lr, loss in (
            (0.001, 0.10),
            (0.01, 0.30),
        ):
            rows.append({
                "stage": "tuning",
                "seed": str(seed),
                "optimizer": "Adam",
                "lr": str(lr),
                "best_val_loss": str(loss),
            })

    return rows


def test_selection_uses_validation_loss():
    selected = hs.select_learning_rates(
        synthetic_tuning_rows(),
        grids={"Adam": (0.001, 0.01)},
        seeds=(100, 101, 102),
    )

    assert selected == {"Adam": 0.001}


def test_selection_rejects_incomplete_grid():
    rows = synthetic_tuning_rows()

    rows = [
        r for r in rows
        if not (
            r["seed"] == "102"
            and r["lr"] == "0.001"
        )
    ]

    with pytest.raises(ValueError):
        hs.select_learning_rates(
            rows,
            grids={"Adam": (0.001, 0.01)},
            seeds=(100, 101, 102),
        )


def test_selection_ignores_test_accuracy():
    rows = synthetic_tuning_rows()

    for row in rows:
        # Deliberately misleading test accuracy.
        row["accuracy"] = (
            "0.01"
            if row["lr"] == "0.001"
            else "1.0"
        )

    selected = hs.select_learning_rates(
        rows,
        grids={"Adam": (0.001, 0.01)},
        seeds=(100, 101, 102),
    )

    assert selected["Adam"] == 0.001


def test_selection_tie_breaks_to_smaller_lr():
    rows = synthetic_tuning_rows()

    for row in rows:
        row["best_val_loss"] = "0.2"

    selected = hs.select_learning_rates(
        rows,
        grids={"Adam": (0.001, 0.01)},
        seeds=(100, 101, 102),
    )

    assert selected["Adam"] == 0.001


# ---------------------------------------------------------
# 6. Bootstrap statistics
# ---------------------------------------------------------

def test_bootstrap_reproducibility():
    values = [0.8, 0.9, 0.85, 0.95]

    first = hs.bootstrap_mean_ci(
        values,
        np.random.default_rng(42),
        n_boot=500,
    )

    second = hs.bootstrap_mean_ci(
        values,
        np.random.default_rng(42),
        n_boot=500,
    )

    assert first == second
    assert first[0] <= first[1]


def test_bootstrap_constant_values():
    low, high = hs.bootstrap_mean_ci(
        [0.9, 0.9, 0.9, 0.9],
        np.random.default_rng(42),
        n_boot=500,
    )

    assert low == pytest.approx(0.9)
    assert high == pytest.approx(0.9)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.5],
        [0.5, np.nan],
        [0.5, np.inf],
    ],
)
def test_bootstrap_invalid_inputs(values):
    with pytest.raises(ValueError):
        hs.bootstrap_mean_ci(
            values,
            np.random.default_rng(42),
            n_boot=100,
        )


# ---------------------------------------------------------
# 7. Final evaluation statistics
# ---------------------------------------------------------

def synthetic_evaluation_rows():
    rows = []

    for seed in (1000, 1001, 1002):
        for name, accuracy in (
            ("SGD", 0.80),
            ("Momentum", 0.85),
            ("Adam", 0.95),
        ):
            rows.append({
                "stage": "evaluation",
                "seed": str(seed),
                "optimizer": name,
                "lr": "0.01",
                "accuracy": str(accuracy),
                "f1": str(accuracy),
                "test_loss": str(1 - accuracy),
            })

    return rows


def test_final_summary_and_paired_differences():
    report = hs.evaluate_summary(
        synthetic_evaluation_rows(),
        selected={
            "SGD": 0.01,
            "Momentum": 0.01,
            "Adam": 0.01,
        },
        seeds=(1000, 1001, 1002),
    )

    assert len(report["metrics"]) == 3
    assert len(report["paired_accuracy"]) == 3

    assert report["metrics"]["Adam"]["accuracy"][
        "mean"
    ] == pytest.approx(0.95)

    assert report["paired_accuracy"][
        "SGD_minus_Adam"
    ]["mean_difference"] == pytest.approx(-0.15)


def test_final_summary_rejects_missing_seed():
    rows = synthetic_evaluation_rows()

    rows = [
        r for r in rows
        if not (
            r["seed"] == "1002"
            and r["optimizer"] == "Adam"
        )
    ]

    with pytest.raises(ValueError):
        hs.evaluate_summary(
            rows,
            selected={
                "SGD": 0.01,
                "Momentum": 0.01,
                "Adam": 0.01,
            },
            seeds=(1000, 1001, 1002),
        )


# ---------------------------------------------------------
# 8. Plot generation tests
# ---------------------------------------------------------

def test_tuning_plot_created(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "OUTPUT", tmp_path)
    monkeypatch.setattr(
        hs,
        "GRIDS",
        {
            "SGD": (0.01,),
            "Momentum": (0.01,),
            "Adam": (0.01,),
        },
    )

    rows = [
        {
            "stage": "tuning",
            "optimizer": name,
            "lr": "0.01",
            "best_val_loss": "0.2",
        }
        for name in hs.GRIDS
    ]

    hs.plot_tuning(rows)

    assert (
        tmp_path / "validation_lr_search.png"
    ).is_file()


def test_final_plot_created(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "OUTPUT", tmp_path)

    report = {
        "metrics": {
            name: {
                "accuracy": {
                    "mean": 0.9,
                    "std": 0.02,
                }
            }
            for name in hs.GRIDS
        }
    }

    hs.plot_final(report)

    assert (
        tmp_path / "final_accuracy.png"
    ).is_file()
