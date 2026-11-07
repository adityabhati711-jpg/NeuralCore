
"""NeuralCore: reproducible neural network loss landscape."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuralcore.datasets import (
    make_moons,
    train_val_test_split,
    StandardScaler,
)
from neuralcore.layers import Dense
from neuralcore.network import Activation, Sequential
from neuralcore.losses import BinaryCrossEntropyWithLogits
from neuralcore.optimizers import Adam
from neuralcore.landscape import (
    flatten_parameters,
    orthonormal_directions,
    compute_loss_surface,
)


SEED = 42
OUTPUT = Path("outputs/landscape")


def build_model():
    rng = np.random.default_rng(SEED)

    return Sequential([
        Dense(2, 16, rng=rng, init="xavier"),
        Activation("tanh"),
        Dense(16, 8, rng=rng, init="xavier"),
        Activation("tanh"),
        Dense(8, 1, rng=rng, init="xavier"),
    ])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    x, y = make_moons(
        n_samples=600, noise=0.18, seed=SEED
    )

    x_train, y_train, _, _, _, _ = (
        train_val_test_split(x, y, seed=SEED)
    )

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)

    model = build_model()
    loss_fn = BinaryCrossEntropyWithLogits()
    optimizer = Adam(lr=0.01)

    initial_weights = flatten_parameters(model)
    initial_loss = loss_fn.forward(
        model.forward(x_train), y_train
    )

    for _ in range(250):
        logits = model.forward(x_train)
        loss_fn.forward(logits, y_train)
        model.backward(loss_fn.backward())
        optimizer.step(model.parameters_and_gradients())

    trained_weights = flatten_parameters(model)
    trained_loss = loss_fn.forward(
        model.forward(x_train), y_train
    )

    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Trained loss: {trained_loss:.6f}")
    print(f"Parameters: {trained_weights.size}")

    direction_a, direction_b = orthonormal_directions(
        trained_weights.size, seed=SEED
    )

    # The surface is centered on the trained model.
    coordinates = np.linspace(-3.0, 3.0, 51)

    aa, bb, surface = compute_loss_surface(
        model=model,
        loss_fn=loss_fn,
        x=x_train,
        y=y_train,
        center=trained_weights,
        direction_a=direction_a,
        direction_b=direction_b,
        coordinates=coordinates,
    )

    np.savez_compressed(
        OUTPUT / "landscape_data.npz",
        axis_a=aa,
        axis_b=bb,
        loss=surface,
        center=trained_weights,
        direction_a=direction_a,
        direction_b=direction_b,
    )

    # 3D loss surface
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        aa, bb, surface,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )

    ax.scatter(
        [0], [0], [trained_loss],
        color="red", s=65, label="Trained model"
    )

    ax.set(
        title="NeuralCore | 3D Neural Network Loss Landscape",
        xlabel="Parameter Direction 1",
        ylabel="Parameter Direction 2",
        zlabel="Binary Cross-Entropy Loss",
    )
    ax.view_init(elev=30, azim=225)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "loss_landscape_3d.png", dpi=220
    )
    plt.close(fig)

    # 2D contour map
    fig, ax = plt.subplots(figsize=(9, 7))

    contour = ax.contourf(
        aa, bb, surface, levels=35, cmap="viridis"
    )

    ax.contour(
        aa, bb, surface,
        levels=15, colors="white",
        linewidths=0.35, alpha=0.5
    )

    ax.scatter(
        [0], [0], color="red",
        s=100, edgecolors="white",
        label="Trained model", zorder=5
    )

    fig.colorbar(
        contour, ax=ax,
        label="Binary Cross-Entropy Loss"
    )

    ax.set(
        title="NeuralCore | Loss Landscape Contours",
        xlabel="Parameter Direction 1",
        ylabel="Parameter Direction 2",
    )
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()

    fig.savefig(
        OUTPUT / "loss_contours.png", dpi=220
    )
    plt.close(fig)

    print(f"\nSaved visualizations to: {OUTPUT}")


if __name__ == "__main__":
    main()
