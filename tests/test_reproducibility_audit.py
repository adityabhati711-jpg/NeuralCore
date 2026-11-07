
"""Automated tests for NeuralCore's reproducibility audit."""

import csv
import hashlib
import json

import numpy as np
import pytest

from experiments import reproducibility_audit as audit


def test_digest_is_deterministic():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert audit.digest(data) == audit.digest(data.copy())


def test_digest_changes_when_data_changes():
    first = np.array([1.0, 2.0, 3.0])
    second = np.array([1.0, 2.0, 4.0])

    assert audit.digest(first) != audit.digest(second)


def test_digest_accounts_for_shape():
    first = np.arange(6, dtype=np.float64)
    second = first.reshape(2, 3)

    assert audit.digest(first) != audit.digest(second)


def test_digest_accounts_for_dtype():
    first = np.array([1, 2, 3], dtype=np.int32)
    second = np.array([1, 2, 3], dtype=np.int64)

    assert audit.digest(first) != audit.digest(second)


def test_digest_matches_sha256_format():
    result = audit.digest(np.array([1, 2, 3]))

    assert len(result) == 64
    int(result, 16)


def test_record_tracks_pass_and_failure():
    audit.CHECKS.clear()

    audit.record("Passing check", True, "OK")
    audit.record("Failing check", False, "Problem")

    assert len(audit.CHECKS) == 2
    assert audit.CHECKS[0]["passed"] is True
    assert audit.CHECKS[1]["passed"] is False

    audit.CHECKS.clear()


def test_json_loader(tmp_path):
    path = tmp_path / "sample.json"
    expected = {"seed": 42, "accuracy": 0.97}

    path.write_text(
        json.dumps(expected), encoding="utf-8"
    )

    assert audit.load_json(path) == expected


def test_csv_loader(tmp_path):
    path = tmp_path / "sample.csv"

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["seed", "accuracy"]
        )
        writer.writeheader()
        writer.writerow({
            "seed": 42,
            "accuracy": 0.97,
        })

    rows = audit.load_csv(path)

    assert len(rows) == 1
    assert rows[0]["seed"] == "42"
    assert float(rows[0]["accuracy"]) == 0.97


def test_model_parameters_reproducible():
    first = audit.model_parameters(42)
    second = audit.model_parameters(42)

    assert len(first) == len(second)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_model_parameters_different_seed():
    first = audit.model_parameters(42)
    second = audit.model_parameters(43)

    assert any(
        not np.array_equal(a, b)
        for a, b in zip(first, second)
    )


def test_model_parameter_copies_are_independent():
    first = audit.model_parameters(42)
    original = first[0].copy()

    first[0][0, 0] += 10

    second = audit.model_parameters(42)

    np.testing.assert_array_equal(second[0], original)


def test_dataset_audit_passes():
    audit.CHECKS.clear()

    result = audit.audit_dataset()

    assert result["partition_sizes"] == {
        "train": 420,
        "validation": 90,
        "test": 90,
    }

    assert all(item["passed"] for item in audit.CHECKS)

    audit.CHECKS.clear()


def test_initialization_audit_passes():
    audit.CHECKS.clear()

    result = audit.audit_initialization()

    assert len(result["seed_42_parameter_hashes"]) > 0
    assert all(item["passed"] for item in audit.CHECKS)

    audit.CHECKS.clear()


def test_benchmark_detects_wrong_mean(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        audit, "ARTIFACT_ROOT", tmp_path
    )

    folder = tmp_path / "optimizer_benchmark"
    folder.mkdir()

    report = {
        "summary": {
            name: {
                "accuracy": {"mean": 0.99}
            }
            for name in ("SGD", "Momentum", "Adam")
        }
    }

    (folder / "summary.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    with (folder / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed", "optimizer", "accuracy"
            ],
        )
        writer.writeheader()

        for name in ("SGD", "Momentum", "Adam"):
            writer.writerow({
                "seed": 42,
                "optimizer": name,
                "accuracy": 0.5,
            })

    audit.CHECKS.clear()
    audit.audit_benchmark()

    assert audit.CHECKS[-1]["passed"] is False
    audit.CHECKS.clear()


def test_benchmark_missing_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        audit, "ARTIFACT_ROOT", tmp_path
    )

    audit.CHECKS.clear()
    audit.audit_benchmark()

    assert audit.CHECKS[-1]["passed"] is False
    audit.CHECKS.clear()


def test_early_stopping_detects_unpaired_seeds(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        audit, "ARTIFACT_ROOT", tmp_path
    )

    folder = tmp_path / "early_stopping_comparison"
    folder.mkdir()

    (folder / "summary.json").write_text(
        json.dumps({"paired_comparisons": {}}),
        encoding="utf-8",
    )

    with (folder / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["seed", "method", "accuracy"],
        )
        writer.writeheader()
        writer.writerow({
            "seed": 1,
            "method": "Fixed",
            "accuracy": 0.9,
        })

    audit.CHECKS.clear()
    audit.audit_early_stopping()

    assert audit.CHECKS[-1]["passed"] is False
    audit.CHECKS.clear()


def test_figures_detect_missing_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        audit, "ARTIFACT_ROOT", tmp_path
    )

    audit.CHECKS.clear()
    result = audit.audit_figures()

    assert result["missing"]
    assert audit.CHECKS[-1]["passed"] is False

    audit.CHECKS.clear()


def test_main_writes_report(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit, "OUTPUT", tmp_path)

    monkeypatch.setattr(
        audit,
        "audit_dataset",
        lambda: {"dataset_hash": "abc"},
    )
    monkeypatch.setattr(
        audit,
        "audit_initialization",
        lambda: {"seed": 42},
    )
    monkeypatch.setattr(
        audit,
        "audit_benchmark",
        lambda: {},
    )
    monkeypatch.setattr(
        audit,
        "audit_hyperparameter_search",
        lambda: {},
    )
    monkeypatch.setattr(
        audit,
        "audit_early_stopping",
        lambda: {},
    )
    monkeypatch.setattr(
        audit,
        "audit_figures",
        lambda: {},
    )

    audit.CHECKS.clear()
    audit.main()

    report_path = tmp_path / "audit_report.json"

    assert report_path.is_file()

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert report["summary"]["failed"] == 0
    assert "limitations" in report


def test_main_exits_on_failed_check(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit, "OUTPUT", tmp_path)

    def failed_dataset():
        audit.record("Intentional failure", False)
        return {}

    monkeypatch.setattr(
        audit, "audit_dataset", failed_dataset
    )

    for name in (
        "audit_initialization",
        "audit_benchmark",
        "audit_hyperparameter_search",
        "audit_early_stopping",
        "audit_figures",
    ):
        monkeypatch.setattr(
            audit, name, lambda: {}
        )

    with pytest.raises(SystemExit) as error:
        audit.main()

    assert error.value.code == 1

    audit.CHECKS.clear()
