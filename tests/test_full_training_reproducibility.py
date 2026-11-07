
"""Tests for independent full-training reproducibility."""

import json

import numpy as np
import pytest

from experiments import full_training_reproducibility as rep


def test_array_hash_is_deterministic():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert rep.array_hash(data) == rep.array_hash(data.copy())


def test_array_hash_detects_value_changes():
    first = np.array([1.0, 2.0])
    second = np.array([1.0, 3.0])
    assert rep.array_hash(first) != rep.array_hash(second)


def test_array_hash_detects_shape_changes():
    first = np.arange(6, dtype=np.float64)
    second = first.reshape(2, 3)
    assert rep.array_hash(first) != rep.array_hash(second)


def test_array_hash_detects_dtype_changes():
    first = np.array([1, 2], dtype=np.int32)
    second = np.array([1, 2], dtype=np.int64)
    assert rep.array_hash(first) != rep.array_hash(second)


@pytest.mark.parametrize(
    "epochs,lr",
    [
        (0, 0.01),
        (-1, 0.01),
        (True, 0.01),
        (5, 0.0),
        (5, -0.01),
        (5, float("nan")),
        (5, float("inf")),
    ],
)
def test_invalid_training_configuration(epochs, lr):
    with pytest.raises(ValueError):
        rep.train_once(42, epochs=epochs, learning_rate=lr)


def test_training_history_length():
    result = rep.train_once(42, epochs=8)

    assert len(result["train_history"]) == 8
    assert len(result["val_history"]) == 8
    assert np.all(np.isfinite(result["train_history"]))
    assert np.all(np.isfinite(result["val_history"]))


def test_independent_training_is_exact():
    first = rep.train_once(42, epochs=12)
    second = rep.train_once(42, epochs=12)

    comparison = rep.compare_runs(first, second)

    assert comparison["all_exact"]
    assert all(comparison["checks"].values())
    assert len(comparison["checks"]) == 7


def test_different_seeds_change_training():
    first = rep.train_once(42, epochs=12)
    second = rep.train_once(43, epochs=12)

    assert not np.array_equal(
        first["train_history"],
        second["train_history"],
    )


def test_compare_rejects_different_seeds():
    first = rep.train_once(42, epochs=3)
    second = rep.train_once(43, epochs=3)

    with pytest.raises(ValueError):
        rep.compare_runs(first, second)


def test_compare_rejects_different_epoch_budgets():
    first = rep.train_once(42, epochs=3)
    second = rep.train_once(42, epochs=4)

    with pytest.raises(ValueError):
        rep.compare_runs(first, second)


def test_compare_detects_modified_parameters():
    first = rep.train_once(42, epochs=5)
    second = rep.train_once(42, epochs=5)

    second["parameters"][0][0, 0] += 1.0

    comparison = rep.compare_runs(first, second)

    assert not comparison["all_exact"]
    assert comparison["checks"]["parameters_exact"] is False


def test_compare_detects_modified_history():
    first = rep.train_once(42, epochs=5)
    second = rep.train_once(42, epochs=5)

    second["train_history"][0] += 0.01

    comparison = rep.compare_runs(first, second)

    assert not comparison["all_exact"]
    assert comparison["checks"]["train_history_exact"] is False


def test_compare_detects_modified_logits():
    first = rep.train_once(42, epochs=5)
    second = rep.train_once(42, epochs=5)

    second["test_logits"][0, 0] += 0.01

    comparison = rep.compare_runs(first, second)

    assert not comparison["all_exact"]
    assert comparison["checks"]["test_logits_exact"] is False


def test_manifest_contains_expected_hashes():
    result = rep.train_once(42, epochs=5)
    manifest = rep.numerical_manifest(result)

    assert len(manifest["train_history_sha256"]) == 64
    assert len(manifest["validation_history_sha256"]) == 64
    assert len(manifest["parameter_sha256"]) == 6
    assert len(manifest["test_logits_sha256"]) == 64
    assert manifest["epochs"] == 5


def test_manifest_is_deterministic():
    first = rep.numerical_manifest(
        rep.train_once(42, epochs=5)
    )
    second = rep.numerical_manifest(
        rep.train_once(42, epochs=5)
    )

    assert first == second


def test_main_generates_report(tmp_path, monkeypatch):
    monkeypatch.setattr(rep, "OUTPUT", tmp_path)
    monkeypatch.setattr(rep, "SEEDS", (42, 43))
    monkeypatch.setattr(rep, "EPOCHS", 5)

    original_train = rep.train_once

    def short_train(seed):
        return original_train(seed, epochs=5)

    monkeypatch.setattr(rep, "train_once", short_train)

    rep.main()

    report_path = tmp_path / "full_training_report.json"

    assert report_path.is_file()

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert report["summary"]["seed_pairs"] == 2
    assert report["summary"]["exact_pairs"] == 2
    assert report["summary"]["all_exact"] is True
    assert (
        report["summary"]["different_seed_sensitivity_passed"]
        is True
    )
