
"""Reproducible multi-seed optimizer benchmark for NeuralCore."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.optimizer_trajectories import (
    OPTIMIZERS,
    build_model,
)
from neuralcore.activations import sigmoid
from neuralcore.datasets import (
    make_moons,
    train_val_test_split,
    StandardScaler,
)
from neuralcore.landscape import flatten_parameters
from neuralcore.layers import Dense
from neuralcore.losses import BinaryCrossEntropyWithLogits


OUTPUT = Path("outputs/optimizer_benchmark")
SEEDS = tuple(range(42, 52))
MAX_EPOCHS = 600
PATIENCE = 100
MIN_DELTA = 1e-8


def prepare_data(seed):
    """Create a stratified split with training-only preprocessing."""
    x, y = make_moons(
        n_samples=600, noise=0.18, seed=seed
    )

    x_train, y_train, x_val, y_val, x_test, y_test = (
        train_val_test_split(x, y, seed=seed)
    )

    scaler = StandardScaler().fit(x_train)

    return (
        scaler.transform(x_train), y_train,
        scaler.transform(x_val), y_val,
        scaler.transform(x_test), y_test,
    )


def parameters(model):
    result = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            result.extend([layer.weights, layer.biases])
    return result


def classification_metrics(y_true, probabilities):
    actual = np.asarray(y_true).reshape(-1).astype(int)
    predicted = (
        np.asarray(probabilities).reshape(-1) >= 0.5
    ).astype(int)

    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "accuracy": float(np.mean(actual == predicted)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_experiment(seed, optimizer_name, max_epochs=MAX_EPOCHS,
                   patience=PATIENCE):
    """Train using validation-only checkpoint selection."""
    if optimizer_name not in OPTIMIZERS:
        raise ValueError("Unknown optimizer.")
    if not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError("max_epochs must be positive.")
    if not isinstance(patience, int) or patience < 1:
        raise ValueError("patience must be positive.")

    x_train, y_train, x_val, y_val, x_test, y_test = (
        prepare_data(seed)
    )

    model = build_model(seed=seed)
    initial_weights = flatten_parameters(model).copy()

    optimizer = OPTIMIZERS[optimizer_name]()
    loss_fn = BinaryCrossEntropyWithLogits()

    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    waiting = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        logits = model.forward(x_train)
        loss_fn.forward(logits, y_train)

        model.backward(loss_fn.backward())
        optimizer.step(model.parameters_and_gradients())

        train_loss = float(
            loss_fn.forward(model.forward(x_train), y_train)
        )
        val_loss = float(
            loss_fn.forward(model.forward(x_val), y_val)
        )

        if not np.isfinite(train_loss + val_loss):
            raise FloatingPointError("Non-finite loss.")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = [p.copy() for p in parameters(model)]
            waiting = 0
        else:
            waiting += 1

        if waiting >= patience:
            break

    for parameter, saved in zip(parameters(model), best_weights):
        parameter[...] = saved

    test_logits = model.forward(x_test)
    test_loss = float(loss_fn.forward(test_logits, y_test))
    probabilities = sigmoid(test_logits)
    scores = classification_metrics(y_test, probabilities)

    return {
        "seed": int(seed),
        "optimizer": optimizer_name,
        "initial_weights": initial_weights,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_loss": float(best_loss),
        "test_loss": test_loss,
        **scores,
        "history": history,
    }


def summarize(results):
    summary = {}
    for name in OPTIMIZERS:
        group = [r for r in results if r["optimizer"] == name]
        summary[name] = {}

        for metric in (
            "accuracy", "f1", "test_loss",
            "best_val_loss", "best_epoch"
        ):
            values = np.array(
                [r[metric] for r in group], dtype=float
            )
            summary[name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1))
                if len(values) > 1 else 0.0,
            }

    return summary


def paired_comparisons(results, metric="accuracy"):
    """Paired bootstrap CIs over seed-level differences."""
    names = list(OPTIMIZERS)
    comparisons = {}
    rng = np.random.default_rng(2026)

    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a = {
                r["seed"]: r[metric] for r in results
                if r["optimizer"] == first
            }
            b = {
                r["seed"]: r[metric] for r in results
                if r["optimizer"] == second
            }

            if set(a) != set(b):
                raise ValueError("Seed sets must match.")

            seeds = sorted(a)
            differences = np.array(
                [a[s] - b[s] for s in seeds], dtype=float
            )

            samples = rng.choice(
                differences,
                size=(10000, len(differences)),
                replace=True,
            ).mean(axis=1)

            low, high = np.percentile(samples, [2.5, 97.5])

            comparisons[f"{first}_minus_{second}"] = {
                "mean_difference": float(differences.mean()),
                "ci_95": [float(low), float(high)],
                "n_pairs": len(differences),
            }

    return comparisons


def save_outputs(results, summary, comparisons):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    clean_results = [
        {
            k: v for k, v in result.items()
            if k not in ("history", "initial_weights")
        }
        for result in results
    ]

    with (OUTPUT / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(clean_results[0])
        )
        writer.writeheader()
        writer.writerows(clean_results)

    report = {
        "configuration": {
            "seeds": list(SEEDS),
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "optimizer_learning_rates": {
                "SGD": 0.05,
                "Momentum": 0.01,
                "Adam": 0.01,
            },
            "comparison_type": "fixed-hyperparameter",
            "bootstrap_resamples": 10000,
        },
        "summary": summary,
        "paired_accuracy_comparisons": comparisons,
    }

    (OUTPUT / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    names = list(OPTIMIZERS)
    means = [summary[n]["accuracy"]["mean"] for n in names]
    errors = [summary[n]["accuracy"]["std"] for n in names]

    ax.bar(names, means, yerr=errors, capsize=6)
    ax.set(
        title="NeuralCore | Optimizer Accuracy Across 10 Seeds",
        ylabel="Held-out Test Accuracy",
        xlabel="Optimizer",
    )
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "accuracy_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in names:
        group = sorted(
            (r for r in results if r["optimizer"] == name),
            key=lambda r: r["seed"],
        )
        ax.plot(
            [r["seed"] for r in group],
            [r["test_loss"] for r in group],
            marker="o", label=name,
        )

    ax.set(
        title="NeuralCore | Test Loss Across Random Seeds",
        xlabel="Random Seed",
        ylabel="Held-out Test BCE Loss",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "seed_stability.png", dpi=220)
    plt.close(fig)


def main():
    results = []

    for seed in SEEDS:
        reference = None

        for name in OPTIMIZERS:
            result = run_experiment(seed, name)

            if reference is None:
                reference = result["initial_weights"]
            else:
                np.testing.assert_array_equal(
                    reference, result["initial_weights"]
                )

            results.append(result)

            print(
                f"Seed {seed} | {name:8s} | "
                f"Accuracy {result['accuracy']:.4f} | "
                f"F1 {result['f1']:.4f} | "
                f"Epoch {result['best_epoch']}"
            )

    summary = summarize(results)
    comparisons = paired_comparisons(results)
    save_outputs(results, summary, comparisons)

    print("\nBENCHMARK SUMMARY")
    for name, scores in summary.items():
        accuracy = scores["accuracy"]
        print(
            f"{name:8s} | "
            f"Accuracy {accuracy['mean']:.4f} "
            f"+/- {accuracy['std']:.4f}"
        )

    print(f"\nSaved results to: {OUTPUT}")


if __name__ == "__main__":
    main()
