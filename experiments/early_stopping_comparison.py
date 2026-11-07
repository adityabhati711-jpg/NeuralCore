
"""Paired multi-seed comparison: fixed training vs early stopping.

Both approaches share the same training trajectory per seed.
Test data is evaluated only after checkpoint selection.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.optimizer_benchmark import (
    prepare_data,
    classification_metrics,
)
from experiments.optimizer_trajectories import build_model
from neuralcore.activations import sigmoid
from neuralcore.early_stopping import EarlyStopping
from neuralcore.layers import Dense
from neuralcore.losses import BinaryCrossEntropyWithLogits
from neuralcore.optimizers import Adam


OUTPUT = Path("outputs/early_stopping_comparison")
SEEDS = tuple(range(1000, 1020))
MAX_EPOCHS = 500
PATIENCE = 50
MIN_DELTA = 0.0
LEARNING_RATE = 0.03
BOOTSTRAP_RESAMPLES = 10000


def parameter_copies(model):
    """Capture independent copies of all Dense parameters."""
    return [
        parameter.copy()
        for layer in model.layers
        if isinstance(layer, Dense)
        for parameter in (layer.weights, layer.biases)
    ]


def restore_parameters(model, checkpoint):
    """Restore a checkpoint without replacing parameter arrays."""
    parameters = [
        parameter
        for layer in model.layers
        if isinstance(layer, Dense)
        for parameter in (layer.weights, layer.biases)
    ]

    if len(parameters) != len(checkpoint):
        raise ValueError("Checkpoint parameter count mismatch.")

    if any(
        parameter.shape != saved.shape
        for parameter, saved in zip(parameters, checkpoint)
    ):
        raise ValueError("Checkpoint parameter shape mismatch.")

    for parameter, saved in zip(parameters, checkpoint):
        parameter[...] = saved


def evaluate(model, x_test, y_test):
    """Evaluate a frozen model on the held-out test partition."""
    logits = model.forward(x_test)
    criterion = BinaryCrossEntropyWithLogits()

    test_loss = float(criterion.forward(logits, y_test))
    probabilities = sigmoid(logits)

    return {
        "test_loss": test_loss,
        **classification_metrics(y_test, probabilities),
    }


def run_seed(
    seed,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    learning_rate=LEARNING_RATE,
):
    """Train once and compare two checkpoints fairly."""

    if (
        isinstance(max_epochs, bool)
        or not isinstance(max_epochs, int)
        or max_epochs < 1
    ):
        raise ValueError("max_epochs must be a positive integer.")

    if (
        isinstance(patience, bool)
        or not isinstance(patience, int)
        or patience < 1
    ):
        raise ValueError("patience must be a positive integer.")

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")

    x_train, y_train, x_val, y_val, x_test, y_test = (
        prepare_data(seed)
    )

    model = build_model(seed=seed)
    optimizer = Adam(lr=learning_rate)
    criterion = BinaryCrossEntropyWithLogits()

    stopper = EarlyStopping(
        patience=patience,
        min_delta=MIN_DELTA,
    )

    history = []
    stop_epoch = None
    first_patience_exhaustion = None

    for epoch in range(1, max_epochs + 1):
        logits = model.forward(x_train)
        criterion.forward(logits, y_train)

        model.backward(criterion.backward())
        optimizer.step(model.parameters_and_gradients())

        train_loss = float(
            criterion.forward(model.forward(x_train), y_train)
        )
        val_loss = float(
            criterion.forward(model.forward(x_val), y_val)
        )

        if not np.isfinite(train_loss + val_loss):
            raise FloatingPointError(
                "Training produced a non-finite loss."
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # Observe the first point where patience is exhausted.
        # Continue training for the fixed-epoch baseline.
        if stop_epoch is None:
            if stopper.update(model, val_loss, epoch):
                stop_epoch = epoch
                first_patience_exhaustion = epoch

    fixed_checkpoint = parameter_copies(model)
    fixed_scores = evaluate(model, x_test, y_test)

    # Restore the best checkpoint observed before stopping.
    stopper.restore_best(model)
    early_scores = evaluate(model, x_test, y_test)

    early_epochs = (
        stop_epoch if stop_epoch is not None else max_epochs
    )

    fixed_result = {
        "seed": int(seed),
        "method": "Fixed",
        "epochs_run": max_epochs,
        "checkpoint_epoch": max_epochs,
        "best_val_loss": float(
            min(row["val_loss"] for row in history)
        ),
        **fixed_scores,
    }

    early_result = {
        "seed": int(seed),
        "method": "EarlyStopping",
        "epochs_run": early_epochs,
        "checkpoint_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_validation_loss,
        **early_scores,
    }

    # Verify that the fixed checkpoint was genuinely different
    # whenever the restored best epoch was earlier.
    if stopper.best_epoch < max_epochs:
        restored = parameter_copies(model)
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(restored, fixed_checkpoint)
        )

    return {
        "fixed": fixed_result,
        "early": early_result,
        "history": history,
        "first_patience_exhaustion": first_patience_exhaustion,
    }


def summarize(results):
    """Mean and sample standard deviation by method."""
    summary = {}

    for method in ("Fixed", "EarlyStopping"):
        group = [r for r in results if r["method"] == method]

        summary[method] = {}

        for metric in (
            "accuracy",
            "f1",
            "test_loss",
            "epochs_run",
            "checkpoint_epoch",
        ):
            values = np.array(
                [r[metric] for r in group], dtype=float
            )

            summary[method][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1))
                if len(values) > 1 else 0.0,
            }

    return summary


def paired_bootstrap(
    results,
    metric,
    n_resamples=BOOTSTRAP_RESAMPLES,
    seed=2026,
):
    """Bootstrap paired EarlyStopping-minus-Fixed differences."""

    fixed = {
        r["seed"]: r[metric]
        for r in results
        if r["method"] == "Fixed"
    }
    early = {
        r["seed"]: r[metric]
        for r in results
        if r["method"] == "EarlyStopping"
    }

    if not fixed or set(fixed) != set(early):
        raise ValueError(
            "Both methods must have identical nonempty seed sets."
        )

    differences = np.array([
        early[s] - fixed[s]
        for s in sorted(fixed)
    ], dtype=float)

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        differences,
        size=(n_resamples, len(differences)),
        replace=True,
    ).mean(axis=1)

    low, high = np.percentile(samples, [2.5, 97.5])

    return {
        "comparison": "EarlyStopping_minus_Fixed",
        "metric": metric,
        "mean_difference": float(differences.mean()),
        "bootstrap_ci_95": [float(low), float(high)],
        "n_pairs": len(differences),
    }


def save_outputs(results, histories, summary, comparisons):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with (OUTPUT / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    with (OUTPUT / "learning_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed", "epoch", "train_loss", "val_loss"
            ],
        )
        writer.writeheader()

        for seed, history in histories.items():
            for row in history:
                writer.writerow({"seed": seed, **row})

    report = {
        "configuration": {
            "seeds": list(SEEDS),
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "comparison": (
                "Same training trajectory per seed; "
                "fixed final checkpoint versus validation-selected "
                "checkpoint before first patience exhaustion."
            ),
        },
        "summary": summary,
        "paired_comparisons": comparisons,
    }

    (OUTPUT / "summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    methods = ["Fixed", "EarlyStopping"]
    means = [
        summary[m]["accuracy"]["mean"]
        for m in methods
    ]
    errors = [
        summary[m]["accuracy"]["std"]
        for m in methods
    ]

    ax.bar(methods, means, yerr=errors, capsize=6)
    ax.set(
        title="NeuralCore | Fixed vs Early Stopping",
        xlabel="Training Strategy",
        ylabel="Held-out Test Accuracy",
    )
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "accuracy_comparison.png", dpi=220
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))

    seeds = sorted(histories)
    fixed = {
        r["seed"]: r
        for r in results if r["method"] == "Fixed"
    }
    early = {
        r["seed"]: r
        for r in results if r["method"] == "EarlyStopping"
    }

    differences = [
        early[s]["test_loss"] - fixed[s]["test_loss"]
        for s in seeds
    ]

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.plot(seeds, differences, marker="o")
    ax.set(
        title="NeuralCore | Paired Test-Loss Difference",
        xlabel="Random Seed",
        ylabel="Early Stopping - Fixed Test BCE",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "paired_test_loss.png", dpi=220
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))

    example_seed = seeds[0]
    history = histories[example_seed]

    epochs = [r["epoch"] for r in history]

    ax.plot(
        epochs,
        [r["train_loss"] for r in history],
        label="Training BCE",
    )
    ax.plot(
        epochs,
        [r["val_loss"] for r in history],
        label="Validation BCE",
    )

    best_epoch = early[example_seed]["checkpoint_epoch"]
    stop_epoch = early[example_seed]["epochs_run"]

    ax.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best checkpoint: {best_epoch}",
    )
    ax.axvline(
        stop_epoch,
        linestyle=":",
        label=f"Early stopping: {stop_epoch}",
    )

    ax.set(
        title=f"NeuralCore | Early Stopping (Seed {example_seed})",
        xlabel="Epoch",
        ylabel="Binary Cross-Entropy",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "early_stopping_curve.png", dpi=220
    )
    plt.close(fig)


def main():
    results = []
    histories = {}

    for seed in SEEDS:
        run = run_seed(seed)

        results.extend([
            run["fixed"],
            run["early"],
        ])
        histories[seed] = run["history"]

        print(
            f"Seed {seed} | "
            f"Fixed acc={run['fixed']['accuracy']:.4f} | "
            f"Early acc={run['early']['accuracy']:.4f} | "
            f"Best epoch={run['early']['checkpoint_epoch']} | "
            f"Stop epoch={run['early']['epochs_run']}"
        )

    summary = summarize(results)

    comparisons = {
        metric: paired_bootstrap(results, metric)
        for metric in (
            "accuracy",
            "f1",
            "test_loss",
            "epochs_run",
        )
    }

    save_outputs(
        results, histories, summary, comparisons
    )

    print("\nFINAL SUMMARY")

    for method in ("Fixed", "EarlyStopping"):
        print(
            f"{method:13s} | "
            f"Accuracy {summary[method]['accuracy']['mean']:.4f} | "
            f"Test BCE {summary[method]['test_loss']['mean']:.5f} | "
            f"Epochs {summary[method]['epochs_run']['mean']:.1f}"
        )

    print(f"\nSaved results to: {OUTPUT}")


if __name__ == "__main__":
    main()
