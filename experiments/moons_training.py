
"""NeuralCore: leakage-safe Two-Moons classification experiment."""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuralcore.activations import sigmoid
from neuralcore.datasets import (
    make_moons, train_val_test_split, StandardScaler
)
from neuralcore.layers import Dense
from neuralcore.losses import BinaryCrossEntropyWithLogits
from neuralcore.network import Activation, Sequential
from neuralcore.optimizers import Adam


SEED = 42
EPOCHS = 2000
PATIENCE = 150
OUTPUT = Path("outputs/moons")


def build_model(seed):
    rng = np.random.default_rng(seed)
    return Sequential([
        Dense(2, 32, rng=rng, init="xavier"),
        Activation("tanh"),
        Dense(32, 16, rng=rng, init="xavier"),
        Activation("tanh"),
        Dense(16, 1, rng=rng, init="xavier"),
    ])


def get_parameters(model):
    params = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            params.extend([layer.weights, layer.biases])
    return params


def metrics(y_true, probabilities):
    actual = y_true.reshape(-1).astype(int)
    predicted = (probabilities.reshape(-1) >= 0.5).astype(int)

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


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Dataset is split before any preprocessing.
    x, y = make_moons(n_samples=600, noise=0.18, seed=SEED)

    x_train, y_train, x_val, y_val, x_test, y_test = (
        train_val_test_split(x, y, seed=SEED)
    )

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    model = build_model(SEED)
    optimizer = Adam(lr=0.01)
    loss_fn = BinaryCrossEntropyWithLogits()

    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    waiting = 0

    for epoch in range(1, EPOCHS + 1):
        train_logits = model.forward(x_train)
        train_loss = loss_fn.forward(train_logits, y_train)

        model.backward(loss_fn.backward())
        optimizer.step(model.parameters_and_gradients())

        # Evaluate after the optimizer update.
        train_loss = loss_fn.forward(
            model.forward(x_train), y_train
        )
        val_loss = loss_fn.forward(
            model.forward(x_val), y_val
        )

        history.append((epoch, train_loss, val_loss))

        if not np.isfinite(train_loss + val_loss):
            raise FloatingPointError("Non-finite training loss.")

        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = [
                parameter.copy()
                for parameter in get_parameters(model)
            ]
            waiting = 0
        else:
            waiting += 1

        if epoch % 200 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Train: {train_loss:.5f} | "
                f"Validation: {val_loss:.5f}"
            )

        if waiting >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best validation checkpoint.
    for parameter, saved in zip(
        get_parameters(model), best_weights
    ):
        parameter[...] = saved

    # Test data is evaluated only after model selection.
    test_logits = model.forward(x_test)
    test_loss = loss_fn.forward(test_logits, y_test)
    test_probs = sigmoid(test_logits)
    results = metrics(y_test, test_probs)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Test loss: {test_loss:.6f}")
    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    with (OUTPUT / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(history)

    report = {
        "seed": SEED,
        "samples": {
            "train": len(x_train),
            "validation": len(x_val),
            "test": len(x_test),
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test_loss": test_loss,
        "test_metrics": results,
    }

    (OUTPUT / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Training history.
    history_array = np.asarray(history)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        history_array[:, 0], history_array[:, 1],
        label="Training", linewidth=2
    )
    ax.plot(
        history_array[:, 0], history_array[:, 2],
        label="Validation", linewidth=2
    )
    ax.axvline(
        best_epoch, color="gray", linestyle="--",
        label=f"Best epoch: {best_epoch}"
    )
    ax.set(
        title="NeuralCore | Two-Moons Learning Curves",
        xlabel="Epoch",
        ylabel="Binary Cross-Entropy Loss"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "learning_curves.png", dpi=220)
    plt.close(fig)

    # Decision boundary in standardized feature space.
    margin = 0.7
    x_min = min(x_train[:, 0].min(), x_test[:, 0].min()) - margin
    x_max = max(x_train[:, 0].max(), x_test[:, 0].max()) + margin
    y_min = min(x_train[:, 1].min(), x_test[:, 1].min()) - margin
    y_max = max(x_train[:, 1].max(), x_test[:, 1].max()) + margin

    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, 220),
        np.linspace(y_min, y_max, 220)
    )
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    probability_grid = sigmoid(model.forward(grid))
    probability_grid = probability_grid.reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(
        gx, gy, probability_grid,
        levels=np.linspace(0, 1, 21),
        cmap="RdBu_r", alpha=0.8
    )
    ax.contour(
        gx, gy, probability_grid,
        levels=[0.5], colors="black", linewidths=1.8
    )
    ax.scatter(
        x_test[:, 0], x_test[:, 1],
        c=y_test.ravel(), cmap="bwr",
        edgecolors="black", linewidths=0.5,
        s=35, label="Held-out test samples"
    )
    fig.colorbar(contour, ax=ax, label="Predicted P(class 1)")
    ax.set(
        title="NeuralCore | Two-Moons Decision Boundary",
        xlabel="Standardized Feature 1",
        ylabel="Standardized Feature 2"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT / "decision_boundary.png", dpi=220)
    plt.close(fig)

    print(f"\nSaved experiment outputs to: {OUTPUT}")


if __name__ == "__main__":
    main()
