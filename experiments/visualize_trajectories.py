
"""Scientific visualization of NeuralCore optimizer trajectories."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuralcore.landscape import (
    flatten_parameters,
    compute_loss_surface,
    project_trajectory,
)
from neuralcore.losses import BinaryCrossEntropyWithLogits
from experiments.optimizer_trajectories import (
    build_model,
    prepare_data,
)

OUTPUT = Path("outputs/optimizer_comparison")
NAMES = ("sgd", "momentum", "adam")
COLORS = {
    "sgd": "#2563eb",
    "momentum": "#f59e0b",
    "adam": "#ef4444",
}


def load_results():
    path = OUTPUT / "optimizer_trajectories.npz"
    with np.load(path) as data:
        results = {
            name: {
                "trajectory": data[f"{name}_trajectory"].copy(),
                "losses": data[f"{name}_losses"].copy(),
            }
            for name in NAMES
        }

    initial = results[NAMES[0]]["trajectory"][0]
    for result in results.values():
        trajectory = result["trajectory"]
        losses = result["losses"]

        if trajectory.ndim != 2 or losses.shape != (
            trajectory.shape[0],
        ):
            raise ValueError("Invalid experiment data shape.")

        if not np.all(np.isfinite(trajectory)) or not np.all(
            np.isfinite(losses)
        ):
            raise ValueError("Non-finite experiment data.")

        np.testing.assert_array_equal(
            trajectory[0], initial
        )

    return results


def compute_projection(results):
    """Fit a common 2D PCA plane to recorded weight displacements."""
    initial = results["sgd"]["trajectory"][0]

    displacement = np.concatenate([
        result["trajectory"] - initial
        for result in results.values()
    ])

    # PCA is fitted to the actual optimizer trajectories.
    mean_displacement = displacement.mean(axis=0)
    centered = displacement - mean_displacement

    _, singular_values, vt = np.linalg.svd(
        centered, full_matrices=False
    )

    directions = vt[:2]

    # Fix arbitrary SVD sign choices for reproducible plots.
    for direction in directions:
        pivot = np.argmax(np.abs(direction))
        if direction[pivot] < 0:
            direction *= -1

    # Center the visualization on the PCA mean.
    center = initial + mean_displacement

    total_variance = np.sum(singular_values**2)
    explained = (
        np.sum(singular_values[:2]**2) / total_variance
        if total_variance > 0 else 0.0
    )

    return center, directions[0], directions[1], explained


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = load_results()

    center, direction_a, direction_b, explained = (
        compute_projection(results)
    )

    projected = {
        name: project_trajectory(
            result["trajectory"],
            center,
            direction_a,
            direction_b,
        )
        for name, result in results.items()
    }

    all_points = np.concatenate(list(projected.values()))
    extent = max(0.5, np.max(np.abs(all_points)) * 1.15)
    coordinates = np.linspace(-extent, extent, 65)

    x_train, y_train = prepare_data()
    model = build_model()
    loss_fn = BinaryCrossEntropyWithLogits()

    aa, bb, surface = compute_loss_surface(
        model,
        loss_fn,
        x_train,
        y_train,
        center,
        direction_a,
        direction_b,
        coordinates,
    )

    # Measure how much of each trajectory lies outside the plane.
    diagnostics = {}

    for name, result in results.items():
        displacement = result["trajectory"] - center
        reconstructed = (
            projected[name][:, :1] * direction_a
            + projected[name][:, 1:] * direction_b
        )

        residual = displacement - reconstructed

        rms_error = float(np.sqrt(
            np.mean(np.sum(residual**2, axis=1))
        ))

        diagnostics[name] = rms_error

    np.savez_compressed(
        OUTPUT / "projected_landscape.npz",
        axis_a=aa,
        axis_b=bb,
        surface=surface,
        center=center,
        direction_a=direction_a,
        direction_b=direction_b,
        explained_variance=explained,
        **{
            f"{name}_projected": projected[name]
            for name in NAMES
        },
    )

    # 1. Contour trajectories
    fig, ax = plt.subplots(figsize=(11, 8))

    contour = ax.contourf(
        aa, bb, surface, levels=40, cmap="viridis"
    )

    for name in NAMES:
        points = projected[name]
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=COLORS[name],
            linewidth=2.3,
            label=name.upper(),
        )
        ax.scatter(
            points[-1, 0],
            points[-1, 1],
            color=COLORS[name],
            s=65,
            edgecolors="white",
            zorder=5,
        )

    ax.scatter(
        projected["sgd"][0, 0],
        projected["sgd"][0, 1],
        marker="*",
        color="white",
        edgecolors="black",
        s=180,
        label="Shared initialization",
        zorder=6,
    )

    fig.colorbar(contour, ax=ax, label="Loss on PCA plane")
    ax.set(
        title="NeuralCore | Optimizer Paths on Loss Landscape",
        xlabel="PCA Direction 1",
        ylabel="PCA Direction 2",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "optimizer_contours.png", dpi=220)
    plt.close(fig)

    # 2. 3D projected trajectories
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        aa, bb, surface,
        cmap="viridis",
        alpha=0.65,
        linewidth=0,
    )

    # These heights are actual full-dimensional training losses,
    # NOT necessarily the loss on the projected PCA plane.
    for name in NAMES:
        points = projected[name]
        losses = results[name]["losses"]

        ax.plot(
            points[:, 0],
            points[:, 1],
            losses,
            color=COLORS[name],
            linewidth=2.4,
            label=f"{name.upper()} (actual loss)",
        )

    ax.set(
        title="NeuralCore | Projected Paths with Actual Loss",
        xlabel="PCA Direction 1",
        ylabel="PCA Direction 2",
        zlabel="Binary Cross-Entropy Loss",
    )
    ax.view_init(elev=30, azim=225)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "optimizer_paths_3d.png", dpi=220)
    plt.close(fig)

    # 3. Convergence curves
    fig, ax = plt.subplots(figsize=(11, 7))

    for name in NAMES:
        ax.plot(
            results[name]["losses"],
            label=name.upper(),
            color=COLORS[name],
            linewidth=2,
        )

    ax.set(
        title="NeuralCore | Optimizer Convergence",
        xlabel="Training update",
        ylabel="Training BCE loss",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "optimizer_convergence.png", dpi=220)
    plt.close(fig)

    print(f"Top-2 PCA explained variance: {explained:.2%}")

    for name, error in diagnostics.items():
        print(f"{name.upper():10s} projection RMS error: {error:.6f}")

    print(f"\nSaved visualizations to: {OUTPUT}")


if __name__ == "__main__":
    main()
