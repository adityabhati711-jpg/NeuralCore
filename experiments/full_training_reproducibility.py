
"""Independent full-training reproducibility verification.

Runs two independent training sessions with identical settings and
compares loss histories, final parameters, logits, and metrics.

Also checks that changing the random seed changes the experiment.

Run:
    python -m experiments.full_training_reproducibility
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from experiments.optimizer_benchmark import (
    prepare_data,
    classification_metrics,
)
from experiments.optimizer_trajectories import build_model
from neuralcore.activations import sigmoid
from neuralcore.layers import Dense
from neuralcore.losses import BinaryCrossEntropyWithLogits
from neuralcore.optimizers import Adam


OUTPUT = Path("outputs/full_training_reproducibility")

SEEDS = (42, 43, 44)
EPOCHS = 300
LEARNING_RATE = 0.03


def array_hash(array):
    """Hash shape, dtype and raw array bytes."""
    array = np.ascontiguousarray(array)

    payload = (
        str(array.shape).encode("utf-8")
        + str(array.dtype).encode("utf-8")
        + array.tobytes()
    )

    return hashlib.sha256(payload).hexdigest()


def parameter_snapshot(model):
    """Return independent copies of trainable parameters."""
    return [
        parameter.copy()
        for layer in model.layers
        if isinstance(layer, Dense)
        for parameter in (layer.weights, layer.biases)
    ]


def train_once(seed, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    """Train a newly initialized model using a fresh optimizer."""

    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
    ):
        raise ValueError("epochs must be a positive integer.")

    if (
        isinstance(learning_rate, bool)
        or not np.isscalar(learning_rate)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            "learning_rate must be positive and finite."
        )

    x_train, y_train, x_val, y_val, x_test, y_test = (
        prepare_data(seed)
    )

    model = build_model(seed=seed)
    optimizer = Adam(lr=learning_rate)
    criterion = BinaryCrossEntropyWithLogits()

    train_history = []
    val_history = []

    for epoch in range(1, epochs + 1):
        logits = model.forward(x_train)
        criterion.forward(logits, y_train)

        model.backward(criterion.backward())
        optimizer.step(model.parameters_and_gradients())

        train_loss = float(
            criterion.forward(
                model.forward(x_train), y_train
            )
        )
        val_loss = float(
            criterion.forward(
                model.forward(x_val), y_val
            )
        )

        if not np.isfinite(train_loss + val_loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}."
            )

        train_history.append(train_loss)
        val_history.append(val_loss)

    test_logits = model.forward(x_test).copy()
    test_probabilities = sigmoid(test_logits)
    test_loss = float(
        criterion.forward(test_logits, y_test)
    )

    metrics = classification_metrics(
        y_test, test_probabilities
    )

    return {
        "seed": int(seed),
        "epochs": epochs,
        "learning_rate": float(learning_rate),
        "train_history": np.asarray(
            train_history, dtype=np.float64
        ),
        "val_history": np.asarray(
            val_history, dtype=np.float64
        ),
        "parameters": parameter_snapshot(model),
        "test_logits": test_logits,
        "test_labels": y_test.copy(),
        "test_loss": test_loss,
        "metrics": metrics,
    }


def compare_runs(first, second):
    """Compare independently produced numerical results exactly."""

    if first["seed"] != second["seed"]:
        raise ValueError(
            "Exact comparison requires matching seeds."
        )

    if (
        first["epochs"] != second["epochs"]
        or first["learning_rate"] != second["learning_rate"]
    ):
        raise ValueError(
            "Training configurations must match."
        )

    parameter_match = (
        len(first["parameters"]) == len(second["parameters"])
        and all(
            np.array_equal(a, b)
            for a, b in zip(
                first["parameters"],
                second["parameters"],
            )
        )
    )

    checks = {
        "train_history_exact": np.array_equal(
            first["train_history"],
            second["train_history"],
        ),
        "validation_history_exact": np.array_equal(
            first["val_history"],
            second["val_history"],
        ),
        "parameters_exact": parameter_match,
        "test_logits_exact": np.array_equal(
            first["test_logits"],
            second["test_logits"],
        ),
        "test_labels_exact": np.array_equal(
            first["test_labels"],
            second["test_labels"],
        ),
        "test_loss_exact": (
            first["test_loss"] == second["test_loss"]
        ),
        "test_metrics_exact": (
            first["metrics"] == second["metrics"]
        ),
    }

    return {
        "seed": first["seed"],
        "checks": {
            name: bool(passed)
            for name, passed in checks.items()
        },
        "all_exact": all(checks.values()),
    }


def numerical_manifest(run):
    """Record hashes and final numerical measurements."""

    return {
        "seed": run["seed"],
        "epochs": run["epochs"],
        "learning_rate": run["learning_rate"],
        "train_history_sha256": array_hash(
            run["train_history"]
        ),
        "validation_history_sha256": array_hash(
            run["val_history"]
        ),
        "parameter_sha256": [
            array_hash(parameter)
            for parameter in run["parameters"]
        ],
        "test_logits_sha256": array_hash(
            run["test_logits"]
        ),
        "test_labels_sha256": array_hash(
            run["test_labels"]
        ),
        "final_train_loss": float(
            run["train_history"][-1]
        ),
        "final_validation_loss": float(
            run["val_history"][-1]
        ),
        "test_loss": run["test_loss"],
        "test_metrics": {
            name: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for name, value in run["metrics"].items()
        },
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = []
    manifests = []

    for seed in SEEDS:
        print(f"\nSeed {seed}: training independent run A")
        first = train_once(seed)

        print(f"Seed {seed}: training independent run B")
        second = train_once(seed)

        comparison = compare_runs(first, second)
        results.append(comparison)
        manifests.append(numerical_manifest(first))

        status = (
            "PASS" if comparison["all_exact"] else "FAIL"
        )

        print(
            f"{status} | Seed {seed} | "
            f"{sum(comparison['checks'].values())}/"
            f"{len(comparison['checks'])} exact checks"
        )

        for name, passed in comparison["checks"].items():
            if not passed:
                print(f"  FAILED: {name}")

    # Different seeds should not produce identical full
    # numerical manifests. This is a sensitivity check,
    # not an exact reproducibility requirement.
    distinct_seeds = (
        len({
            manifest["train_history_sha256"]
            for manifest in manifests
        }) == len(manifests)
    )

    exact_reproducibility = all(
        result["all_exact"] for result in results
    )

    report = {
        "experiment": "Independent full-training reproducibility",
        "configuration": {
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "independent_runs_per_seed": 2,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "comparisons": results,
        "different_seed_sensitivity": distinct_seeds,
        "numerical_manifests": manifests,
        "summary": {
            "seed_pairs": len(results),
            "exact_pairs": sum(
                result["all_exact"]
                for result in results
            ),
            "all_exact": exact_reproducibility,
            "different_seed_sensitivity_passed": (
                distinct_seeds
            ),
        },
        "limitations": [
            "Verification is within one execution environment.",
            "Exact cross-platform agreement is not established.",
            "The test uses one architecture and optimizer.",
            "Seed variation is a sensitivity check, not "
            "a statistical generalization assessment.",
        ],
    }

    output_path = OUTPUT / "full_training_report.json"

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL REPRODUCIBILITY SUMMARY")
    print(
        f"Exact seed pairs: "
        f"{report['summary']['exact_pairs']}/"
        f"{report['summary']['seed_pairs']}"
    )
    print(
        f"Different-seed sensitivity: "
        f"{'PASS' if distinct_seeds else 'FAIL'}"
    )
    print(f"Report saved to: {output_path}")

    if not exact_reproducibility or not distinct_seeds:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
