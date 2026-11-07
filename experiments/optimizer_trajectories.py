
"""Reproducible optimizer trajectory experiments for NeuralCore."""

from pathlib import Path

import numpy as np

from neuralcore.datasets import (
    make_moons,
    train_val_test_split,
    StandardScaler,
)
from neuralcore.layers import Dense
from neuralcore.network import Activation, Sequential
from neuralcore.losses import BinaryCrossEntropyWithLogits
from neuralcore.optimizers import SGD, Momentum, Adam
from neuralcore.landscape import flatten_parameters


SEED = 42
EPOCHS = 350
OUTPUT = Path("outputs/optimizer_comparison")

OPTIMIZERS = {
    "SGD": lambda: SGD(lr=0.05),
    "Momentum": lambda: Momentum(lr=0.01, beta=0.9),
    "Adam": lambda: Adam(lr=0.01),
}


def build_model(seed=SEED):
    rng = np.random.default_rng(seed)

    return Sequential([
        Dense(2, 16, rng=rng, init="xavier"),
        Activation("tanh"),
        Dense(16, 8, rng=rng, init="xavier"),
        Activation("tanh"),
        Dense(8, 1, rng=rng, init="xavier"),
    ])


def prepare_data(seed=SEED):
    x, y = make_moons(
        n_samples=600,
        noise=0.18,
        seed=seed,
    )

    x_train, y_train, _, _, _, _ = (
        train_val_test_split(x, y, seed=seed)
    )

    scaler = StandardScaler().fit(x_train)

    return scaler.transform(x_train), y_train


def run_optimizer(name, x_train, y_train, epochs=EPOCHS):
    """Train from identical initial weights and record every step."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")

    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")

    model = build_model()
    loss_fn = BinaryCrossEntropyWithLogits()
    optimizer = OPTIMIZERS[name]()

    trajectory = [flatten_parameters(model)]
    losses = [
        float(loss_fn.forward(model.forward(x_train), y_train))
    ]

    for _ in range(epochs):
        logits = model.forward(x_train)
        loss_fn.forward(logits, y_train)

        model.backward(loss_fn.backward())
        optimizer.step(model.parameters_and_gradients())

        current_loss = float(
            loss_fn.forward(model.forward(x_train), y_train)
        )

        if not np.isfinite(current_loss):
            raise FloatingPointError(
                f"{name} produced non-finite loss."
            )

        trajectory.append(flatten_parameters(model))
        losses.append(current_loss)

    return {
        "trajectory": np.stack(trajectory),
        "losses": np.asarray(losses),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x_train, y_train = prepare_data()

    results = {}

    for name in OPTIMIZERS:
        result = run_optimizer(name, x_train, y_train)
        results[name] = result

        print(
            f"{name:10s} | "
            f"Initial: {result['losses'][0]:.6f} | "
            f"Final: {result['losses'][-1]:.6f}"
        )

    initial = [
        result["trajectory"][0]
        for result in results.values()
    ]

    for vector in initial[1:]:
        np.testing.assert_array_equal(initial[0], vector)

    np.savez_compressed(
        OUTPUT / "optimizer_trajectories.npz",
        **{
            f"{name.lower()}_{key}": value
            for name, result in results.items()
            for key, value in result.items()
        },
    )

    print(f"\nSaved trajectories to: {OUTPUT}")


if __name__ == "__main__":
    main()
