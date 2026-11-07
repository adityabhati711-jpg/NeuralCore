
"""Reproducibility and research-integrity audit for NeuralCore.

Checks:
- Dataset and split determinism
- Train/validation/test partition isolation
- Training-only feature scaling
- Seed-controlled model initialization
- Saved benchmark and experiment artifact consistency
- Paired comparison arithmetic
- Generated figure integrity

This audit does not retrain models or prove complete reproducibility.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from neuralcore.datasets import (
    make_moons,
    train_val_test_split,
    StandardScaler,
)
from experiments.optimizer_trajectories import build_model
from neuralcore.layers import Dense


OUTPUT = Path("outputs/reproducibility_audit")
ARTIFACT_ROOT = Path("outputs")

CHECKS = []


def record(name, passed, details=""):
    CHECKS.append({
        "name": name,
        "passed": bool(passed),
        "details": str(details),
    })
    print(f"{'PASS' if passed else 'FAIL'} | {name}")


def digest(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(
        str(array.shape).encode()
        + str(array.dtype).encode()
        + array.tobytes()
    ).hexdigest()


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):
    with Path(path).open(
        newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def model_parameters(seed):
    model = build_model(seed=seed)

    return [
        parameter.copy()
        for layer in model.layers
        if isinstance(layer, Dense)
        for parameter in (layer.weights, layer.biases)
    ]


def audit_dataset():
    x1, y1 = make_moons(
        n_samples=600, noise=0.12, seed=42
    )
    x2, y2 = make_moons(
        n_samples=600, noise=0.12, seed=42
    )

    record(
        "Dataset deterministic for identical seed",
        np.array_equal(x1, x2)
        and np.array_equal(y1, y2),
    )

    x3, y3 = make_moons(
        n_samples=600, noise=0.12, seed=43
    )

    record(
        "Different seeds produce different samples",
        not np.array_equal(x1, x3),
    )

    split1 = train_val_test_split(
        x1, y1, seed=42
    )
    split2 = train_val_test_split(
        x1, y1, seed=42
    )

    record(
        "Dataset split deterministic",
        all(
            np.array_equal(a, b)
            for a, b in zip(split1, split2)
        ),
    )

    x_train, y_train, x_val, y_val, x_test, y_test = (
        split1
    )

    sizes_correct = (
        len(x_train) == 420
        and len(x_val) == 90
        and len(x_test) == 90
    )

    record(
        "Train/validation/test partition sizes",
        sizes_correct,
        f"{len(x_train)}/{len(x_val)}/{len(x_test)}",
    )

    # Synthetic feature vectors are continuous, so exact duplicate
    # samples are unlikely. This checks sample-value overlap,
    # not original source-index provenance.
    def row_hashes(x):
        return {
            digest(row)
            for row in x
        }

    train_rows = row_hashes(x_train)
    val_rows = row_hashes(x_val)
    test_rows = row_hashes(x_test)

    no_overlap = (
        train_rows.isdisjoint(val_rows)
        and train_rows.isdisjoint(test_rows)
        and val_rows.isdisjoint(test_rows)
    )

    record(
        "No identical feature rows across partitions",
        no_overlap,
        "Value-based check, not index-provenance proof.",
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    val_scaled = scaler.transform(x_val)
    test_scaled = scaler.transform(x_test)

    expected_mean = x_train.mean(axis=0, keepdims=True)
    expected_std = x_train.std(axis=0, keepdims=True)

    record(
        "Scaler fitted using training statistics",
        np.allclose(scaler.mean_, expected_mean)
        and np.allclose(
            scaler.scale_,
            np.where(expected_std > 0, expected_std, 1),
        ),
    )

    record(
        "Scaled features are finite",
        all(
            np.all(np.isfinite(x))
            for x in (
                train_scaled, val_scaled, test_scaled
            )
        ),
    )

    return {
        "dataset_hash": digest(x1),
        "labels_hash": digest(y1),
        "split_hashes": {
            "train": digest(x_train),
            "validation": digest(x_val),
            "test": digest(x_test),
        },
        "partition_sizes": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
    }


def audit_initialization():
    first = model_parameters(seed=42)
    second = model_parameters(seed=42)
    third = model_parameters(seed=43)

    record(
        "Model initialization reproducible",
        all(
            np.array_equal(a, b)
            for a, b in zip(first, second)
        ),
    )

    record(
        "Different initialization seeds change parameters",
        any(
            not np.array_equal(a, b)
            for a, b in zip(first, third)
        ),
    )

    return {
        "seed_42_parameter_hashes": [
            digest(p) for p in first
        ],
    }


def audit_benchmark():
    folder = ARTIFACT_ROOT / "optimizer_benchmark"
    summary_path = folder / "summary.json"
    runs_path = folder / "runs.csv"

    if not summary_path.is_file() or not runs_path.is_file():
        record(
            "Optimizer benchmark artifacts available",
            False,
            "Missing summary.json or runs.csv.",
        )
        return

    report = load_json(summary_path)
    runs = load_csv(runs_path)

    valid = True
    differences = {}

    for optimizer in ("SGD", "Momentum", "Adam"):
        group = [
            row for row in runs
            if row.get("optimizer", row.get("method"))
            == optimizer
        ]

        if not group:
            valid = False
            continue

        observed = np.array(
            [float(row["accuracy"]) for row in group]
        )

        expected = report["summary"][optimizer]["accuracy"]

        if not np.isclose(
            observed.mean(), expected["mean"]
        ):
            valid = False

        differences[optimizer] = {
            "observed_mean_accuracy": float(
                observed.mean()
            ),
            "reported_mean_accuracy": expected["mean"],
            "runs": len(group),
        }

    record(
        "Optimizer benchmark means match CSV",
        valid,
    )

    return differences


def audit_hyperparameter_search():
    folder = ARTIFACT_ROOT / "hyperparameter_search"
    report_path = folder / "final_report.json"
    selection_path = folder / "selected_learning_rates.json"

    if not report_path.is_file():
        record(
            "Hyperparameter final report available",
            False,
        )
        return

    report = load_json(report_path)
    rates = report["selected_learning_rates"]

    record(
        "Selected learning rates are positive",
        all(
            np.isfinite(float(rate)) and float(rate) > 0
            for rate in rates.values()
        ),
    )

    if selection_path.is_file():
        selected = load_json(selection_path)

        # Support either a flat mapping or a nested mapping.
        candidate = selected.get(
            "selected_learning_rates", selected
        )

        record(
            "Locked learning rates match final report",
            all(
                np.isclose(
                    float(candidate[name]),
                    float(rate),
                )
                for name, rate in rates.items()
            ),
        )

    return rates


def audit_early_stopping():
    folder = ARTIFACT_ROOT / "early_stopping_comparison"
    report_path = folder / "summary.json"
    runs_path = folder / "runs.csv"

    if not report_path.is_file() or not runs_path.is_file():
        record(
            "Early stopping comparison artifacts available",
            False,
        )
        return

    report = load_json(report_path)
    runs = load_csv(runs_path)

    fixed = {
        int(row["seed"]): row
        for row in runs if row["method"] == "Fixed"
    }
    early = {
        int(row["seed"]): row
        for row in runs
        if row["method"] == "EarlyStopping"
    }

    seeds_match = (
        len(fixed) > 0
        and set(fixed) == set(early)
        and len(runs) == 2 * len(fixed)
    )

    record(
        "Early stopping paired seeds match",
        seeds_match,
    )

    if not seeds_match:
        return

    comparisons_correct = True

    for metric, reported in report[
        "paired_comparisons"
    ].items():
        differences = [
            float(early[s][metric])
            - float(fixed[s][metric])
            for s in sorted(fixed)
        ]

        if not np.isclose(
            np.mean(differences),
            reported["mean_difference"],
        ):
            comparisons_correct = False

    record(
        "Paired differences match saved run data",
        comparisons_correct,
    )

    checkpoint_valid = all(
        1 <= int(row["checkpoint_epoch"])
        <= int(row["epochs_run"])
        <= report["configuration"]["max_epochs"]
        for row in runs
    )

    record(
        "Checkpoint epochs respect training budget",
        checkpoint_valid,
    )

    summary_correct = True

    for method in ("Fixed", "EarlyStopping"):
        group = [
            row for row in runs if row["method"] == method
        ]

        for metric, stats in report["summary"][method].items():
            observed = np.mean([
                float(row[metric]) for row in group
            ])

            if not np.isclose(observed, stats["mean"]):
                summary_correct = False

    record(
        "Early stopping summary means match CSV",
        summary_correct,
    )

    return {
        "paired_seeds": len(fixed),
        "reported_accuracy_difference": report[
            "paired_comparisons"
        ]["accuracy"]["mean_difference"],
    }


def audit_figures():
    expected = [
        "moons/decision_boundary.png",
        "moons/learning_curves.png",
        "landscape/loss_contours.png",
        "landscape/loss_landscape_3d.png",
        "optimizer_benchmark/accuracy_comparison.png",
        "optimizer_benchmark/seed_stability.png",
        "hyperparameter_search/final_accuracy.png",
        "gradient_diagnostics/gradient_flow.png",
        "gradient_diagnostics/update_ratios.png",
        "early_stopping_comparison/accuracy_comparison.png",
        "early_stopping_comparison/paired_test_loss.png",
        "early_stopping_comparison/early_stopping_curve.png",
    ]

    missing = []

    for relative in expected:
        path = ARTIFACT_ROOT / relative

        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)

    record(
        "Expected research figures exist and are nonempty",
        len(missing) == 0,
        f"Missing: {missing}" if missing else
        f"{len(expected)} figures checked.",
    )

    return {
        "figures_checked": len(expected),
        "missing": missing,
    }


def main():
    CHECKS.clear()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset": audit_dataset(),
        "initialization": audit_initialization(),
        "benchmark": audit_benchmark(),
        "hyperparameter_search": audit_hyperparameter_search(),
        "early_stopping": audit_early_stopping(),
        "figures": audit_figures(),
    }

    passed = sum(item["passed"] for item in CHECKS)
    failed = len(CHECKS) - passed

    report = {
        "audit_type": "artifact-and-determinism",
        "checks": CHECKS,
        "results": results,
        "summary": {
            "total": len(CHECKS),
            "passed": passed,
            "failed": failed,
        },
        "limitations": [
            "This audit does not rerun full model training.",
            "Dataset isolation is checked by feature values, "
            "not original sample indices.",
            "Artifact consistency does not independently "
            "verify the experiment design.",
            "A passing audit does not establish external "
            "dataset generalization.",
        ],
    }

    (OUTPUT / "audit_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nAUDIT COMPLETE | "
        f"{passed}/{len(CHECKS)} passed | "
        f"{failed} failed"
    )
    print(f"Saved report: {OUTPUT / 'audit_report.json'}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
