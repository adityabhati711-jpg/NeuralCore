
"""Visualize gradient flow and parameter updates during training."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.optimizer_trajectories import build_model
from neuralcore.datasets import (
    make_moons, train_val_test_split, StandardScaler
)
from neuralcore.losses import BinaryCrossEntropyWithLogits
from neuralcore.optimizers import Adam
from neuralcore.diagnostics import (
    collect_gradient_diagnostics,
    collect_update_diagnostics,
    snapshot_parameters,
    gradient_flow_summary,
    stability_warnings,
    diagnostics_to_dict,
)


OUTPUT = Path("outputs/gradient_diagnostics")
SEED = 42
EPOCHS = 500
LEARNING_RATE = 0.01


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    x, y = make_moons(
        n_samples=600, noise=0.18, seed=SEED
    )

    x_train, y_train, x_val, y_val, _, _ = (
        train_val_test_split(x, y, seed=SEED)
    )

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    model = build_model(seed=SEED)
    optimizer = Adam(lr=LEARNING_RATE)
    criterion = BinaryCrossEntropyWithLogits()

    history = []
    warning_counts = {}

    for epoch in range(1, EPOCHS + 1):
        logits = model.forward(x_train)
        criterion.forward(logits, y_train)
        model.backward(criterion.backward())

        gradients = collect_gradient_diagnostics(model)
        before = snapshot_parameters(model)

        optimizer.step(model.parameters_and_gradients())

        updates = collect_update_diagnostics(model, before)

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
            raise FloatingPointError("Non-finite training loss.")

        warnings = stability_warnings(
            gradients,
            updates,
        )

        for warning in warnings:
            key = (
                warning["layer"],
                warning["type"],
            )
            warning_counts[key] = (
                warning_counts.get(key, 0) + 1
            )

        for grad, update in zip(gradients, updates):
            history.append({
                "epoch": epoch,
                "layer": grad.name,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "gradient_norm": grad.gradient_norm,
                "gradient_mean_abs": grad.gradient_mean_abs,
                "gradient_max_abs": grad.gradient_max_abs,
                "gradient_zero_fraction": grad.gradient_zero_fraction,
                "weight_norm": grad.weight_norm,
                "update_norm": update.update_norm,
                "update_ratio": update.update_to_parameter_ratio,
            })

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train BCE: {train_loss:.5f} | "
                f"Val BCE: {val_loss:.5f}"
            )

    with (OUTPUT / "gradient_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(history[0])
        )
        writer.writeheader()
        writer.writerows(history)

    layers = sorted(set(row["layer"] for row in history))

    fig, ax = plt.subplots(figsize=(10, 6))

    for layer in layers:
        rows = [r for r in history if r["layer"] == layer]
        ax.plot(
            [r["epoch"] for r in rows],
            [max(r["gradient_norm"], 1e-16) for r in rows],
            label=layer,
            linewidth=1.7,
        )

    ax.set_yscale("log")
    ax.set(
        title="NeuralCore | Layer-wise Gradient Flow",
        xlabel="Training Epoch",
        ylabel="Gradient L2 Norm (log scale)",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "gradient_flow.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))

    for layer in layers:
        rows = [r for r in history if r["layer"] == layer]
        ax.plot(
            [r["epoch"] for r in rows],
            [max(r["update_ratio"], 1e-16) for r in rows],
            label=layer,
            linewidth=1.7,
        )

    ax.set_yscale("log")
    ax.set(
        title="NeuralCore | Update-to-Parameter Ratios",
        xlabel="Training Epoch",
        ylabel="Relative Parameter Update (log scale)",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "update_ratios.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = sorted(set(row["epoch"] for row in history))
    train = [
        next(r["train_loss"] for r in history
             if r["epoch"] == epoch)
        for epoch in epochs
    ]
    validation = [
        next(r["val_loss"] for r in history
             if r["epoch"] == epoch)
        for epoch in epochs
    ]

    ax.plot(epochs, train, label="Training BCE")
    ax.plot(epochs, validation, label="Validation BCE")
    ax.set(
        title="NeuralCore | Diagnostic Training Loss",
        xlabel="Training Epoch",
        ylabel="Binary Cross-Entropy",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "diagnostic_losses.png", dpi=220)
    plt.close(fig)

    report = {
        "seed": SEED,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "final_train_loss": train[-1],
        "final_validation_loss": validation[-1],
        "final_gradient_flow": gradient_flow_summary(gradients),
        "final_layer_diagnostics": diagnostics_to_dict(gradients),
        "final_update_diagnostics": diagnostics_to_dict(updates),
        "warning_counts": [
            {
                "layer": layer,
                "type": warning_type,
                "epochs_flagged": count,
            }
            for (layer, warning_type), count
            in sorted(warning_counts.items())
        ],
        "note": (
            "Gradient thresholds are heuristics. "
            "This experiment is diagnostic, not a "
            "held-out model performance evaluation."
        ),
    }

    (OUTPUT / "diagnostics_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved diagnostics to: {OUTPUT}")


if __name__ == "__main__":
    main()
