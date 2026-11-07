
"""Research-oriented, resumable hyperparameter search for NeuralCore."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.optimizer_trajectories import build_model
from neuralcore.activations import sigmoid
from neuralcore.datasets import (
    make_moons,
    train_val_test_split,
    StandardScaler,
)
from neuralcore.layers import Dense
from neuralcore.losses import BinaryCrossEntropyWithLogits
from neuralcore.optimizers import SGD, Momentum, Adam


OUTPUT = Path("outputs/hyperparameter_search")

TUNING_SEEDS = tuple(range(100, 110))
EVALUATION_SEEDS = tuple(range(1000, 1020))

MAX_EPOCHS = 1000
PATIENCE = 150
MIN_DELTA = 1e-8

GRIDS = {
    "SGD": (0.005, 0.02, 0.05, 0.1, 0.2),
    "Momentum": (0.001, 0.003, 0.01, 0.03, 0.1),
    "Adam": (0.0003, 0.001, 0.003, 0.01, 0.03),
}

FIELDS = (
    "stage", "seed", "optimizer", "lr",
    "best_epoch", "epochs_run", "best_val_loss",
    "test_loss", "accuracy", "f1",
)


def dataset(seed):
    x, y = make_moons(
        n_samples=600, noise=0.18, seed=seed
    )

    xtr, ytr, xv, yv, xt, yt = (
        train_val_test_split(x, y, seed=seed)
    )

    scaler = StandardScaler().fit(xtr)

    return (
        scaler.transform(xtr), ytr,
        scaler.transform(xv), yv,
        scaler.transform(xt), yt,
    )


def optimizer_for(name, lr):
    if name == "SGD":
        return SGD(lr=lr)
    if name == "Momentum":
        return Momentum(lr=lr, beta=0.9)
    if name == "Adam":
        return Adam(lr=lr)

    raise ValueError(f"Unknown optimizer: {name}")


def train(
    seed, name, lr,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    stage="tuning",
):
    """Train and select checkpoints using validation loss."""

    if stage not in ("tuning", "evaluation"):
        raise ValueError("Invalid stage.")

    if (
        not isinstance(max_epochs, int)
        or isinstance(max_epochs, bool)
        or max_epochs < 1
    ):
        raise ValueError("max_epochs must be positive.")

    if (
        not isinstance(patience, int)
        or isinstance(patience, bool)
        or patience < 1
    ):
        raise ValueError("patience must be positive.")

    xtr, ytr, xv, yv, xt, yt = dataset(seed)

    model = build_model(seed=seed)
    opt = optimizer_for(name, lr)
    criterion = BinaryCrossEntropyWithLogits()

    params = [
        p
        for layer in model.layers
        if isinstance(layer, Dense)
        for p in (layer.weights, layer.biases)
    ]

    best = math.inf
    checkpoint = None
    best_epoch = 0
    waiting = 0
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        logits = model.forward(xtr)
        criterion.forward(logits, ytr)

        model.backward(criterion.backward())
        opt.step(model.parameters_and_gradients())

        val_loss = float(
            criterion.forward(model.forward(xv), yv)
        )

        if not math.isfinite(val_loss):
            raise FloatingPointError(
                f"Non-finite validation loss: "
                f"{name}, seed={seed}, lr={lr}"
            )

        if val_loss < best - MIN_DELTA:
            best = val_loss
            best_epoch = epoch
            checkpoint = [p.copy() for p in params]
            waiting = 0
        else:
            waiting += 1

        if waiting >= patience:
            break

    for p, saved in zip(params, checkpoint):
        p[...] = saved

    row = {
        "stage": stage,
        "seed": int(seed),
        "optimizer": name,
        "lr": float(lr),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_val_loss": float(best),
        "test_loss": "",
        "accuracy": "",
        "f1": "",
    }

    if stage == "evaluation":
        logits = model.forward(xt)

        row["test_loss"] = float(
            criterion.forward(logits, yt)
        )

        pred = (
            sigmoid(logits).ravel() >= 0.5
        ).astype(int)

        actual = yt.ravel().astype(int)

        tp = int(np.sum(
            (pred == 1) & (actual == 1)
        ))
        fp = int(np.sum(
            (pred == 1) & (actual == 0)
        ))
        fn = int(np.sum(
            (pred == 0) & (actual == 1)
        ))

        row["accuracy"] = float(
            np.mean(pred == actual)
        )

        denominator = 2 * tp + fp + fn
        row["f1"] = (
            float(2 * tp / denominator)
            if denominator else 0.0
        )

    return row


def read_rows(path):
    if not path.exists():
        return []

    with path.open(
        newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def append_row(path, row):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    new = (
        not path.exists()
        or path.stat().st_size == 0
    )

    with path.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )

        if new:
            writer.writeheader()

        writer.writerow(row)
        file.flush()


def key(stage, seed, name, lr):
    return (
        stage, int(seed),
        name, float(lr),
    )


def select_learning_rates(
    rows, grids=GRIDS, seeds=TUNING_SEEDS
):
    """Select LR by mean validation BCE only."""

    selection = {}
    expected = set(seeds)

    for name, rates in grids.items():
        options = []

        for lr in rates:
            matches = [
                r for r in rows
                if r["stage"] == "tuning"
                and r["optimizer"] == name
                and float(r["lr"]) == float(lr)
                and int(r["seed"]) in expected
            ]

            observed = {
                int(r["seed"]) for r in matches
            }

            if (
                observed != expected
                or len(matches) != len(expected)
            ):
                raise ValueError(
                    f"Incomplete tuning grid: "
                    f"{name}, lr={lr}"
                )

            vals = np.asarray([
                float(r["best_val_loss"])
                for r in matches
            ])

            if not np.all(np.isfinite(vals)):
                raise ValueError(
                    "Non-finite tuning losses."
                )

            options.append((
                float(np.mean(vals)),
                float(lr),
            ))

        # Lowest validation loss wins.
        # Ties are resolved using the smaller LR.
        selection[name] = min(options)[1]

    return selection


def bootstrap_mean_ci(
    values, rng, n_boot=10000
):
    values = np.asarray(
        values, dtype=float
    )

    if (
        values.ndim != 1
        or len(values) < 2
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "Need at least two finite values."
        )

    means = np.empty(n_boot)

    for i in range(n_boot):
        means[i] = rng.choice(
            values,
            size=len(values),
            replace=True,
        ).mean()

    return [
        float(v)
        for v in np.percentile(
            means, (2.5, 97.5)
        )
    ]


def evaluate_summary(
    rows, selected, seeds=EVALUATION_SEEDS
):
    """Evaluate selected optimizers on unseen seeds."""

    rng = np.random.default_rng(2026)
    expected = set(seeds)
    by_name = {}

    report = {
        "selected_learning_rates": selected,
        "metrics": {},
        "paired_accuracy": {},
    }

    for name in GRIDS:
        group = [
            r for r in rows
            if r["stage"] == "evaluation"
            and r["optimizer"] == name
            and float(r["lr"]) == selected[name]
            and int(r["seed"]) in expected
        ]

        observed = {
            int(r["seed"]) for r in group
        }

        if (
            len(group) != len(expected)
            or observed != expected
        ):
            raise ValueError(
                f"Incomplete evaluation: {name}"
            )

        by_name[name] = {
            int(r["seed"]): r for r in group
        }

        report["metrics"][name] = {}

        for metric in (
            "accuracy", "f1", "test_loss"
        ):
            values = np.array([
                float(by_name[name][s][metric])
                for s in seeds
            ])

            report["metrics"][name][metric] = {
                "mean": float(values.mean()),
                "std": float(
                    values.std(ddof=1)
                ),
                "bootstrap_ci_95": (
                    bootstrap_mean_ci(values, rng)
                ),
            }

    names = list(GRIDS)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = np.array([
                float(
                    by_name[a][s]["accuracy"]
                )
                - float(
                    by_name[b][s]["accuracy"]
                )
                for s in seeds
            ])

            report["paired_accuracy"][
                f"{a}_minus_{b}"
            ] = {
                "mean_difference": float(
                    diff.mean()
                ),
                "bootstrap_ci_95": (
                    bootstrap_mean_ci(diff, rng)
                ),
                "n_pairs": len(diff),
            }

    return report


def plot_tuning(rows):
    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    for name, rates in GRIDS.items():
        means = [
            np.mean([
                float(r["best_val_loss"])
                for r in rows
                if r["stage"] == "tuning"
                and r["optimizer"] == name
                and float(r["lr"]) == lr
            ])
            for lr in rates
        ]

        ax.plot(
            rates, means,
            marker="o",
            label=name,
        )

    ax.set_xscale("log")

    ax.set(
        title=(
            "NeuralCore | Learning-rate search "
            "(validation only)"
        ),
        xlabel="Learning rate (log scale)",
        ylabel="Mean best validation BCE",
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "validation_lr_search.png",
        dpi=220,
    )
    plt.close(fig)


def plot_final(report):
    names = list(GRIDS)

    means = [
        report["metrics"][n]["accuracy"]["mean"]
        * 100
        for n in names
    ]

    stds = [
        report["metrics"][n]["accuracy"]["std"]
        * 100
        for n in names
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        names, means,
        yerr=stds,
        capsize=5,
    )

    ax.set(
        title=(
            "NeuralCore | Final evaluation "
            "on unseen seeds"
        ),
        ylabel=(
            "Mean test accuracy (%) "
            "+/- seed SD"
        ),
        ylim=(0, 105),
    )

    ax.grid(
        axis="y", alpha=0.2
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "final_accuracy.png",
        dpi=220,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--stage",
        choices=(
            "tuning", "evaluation", "all"
        ),
        default="tuning",
    )

    args = parser.parse_args()

    OUTPUT.mkdir(
        parents=True, exist_ok=True
    )

    config = {
        "tuning_seeds": TUNING_SEEDS,
        "evaluation_seeds": EVALUATION_SEEDS,
        "grids": GRIDS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "selection_metric": (
            "mean best validation BCE"
        ),
        "evaluation_policy": (
            "No test evaluation during tuning; "
            "final seeds untouched until lock."
        ),
    }

    config_path = OUTPUT / "config.json"
    config_text = json.dumps(
        config, indent=2
    )

    if (
        config_path.exists()
        and config_path.read_text(
            encoding="utf-8"
        ) != config_text
    ):
        raise RuntimeError(
            "Existing configuration differs. "
            "Refusing to mix runs."
        )

    config_path.write_text(
        config_text, encoding="utf-8"
    )

    path = OUTPUT / "runs.csv"
    rows = read_rows(path)

    seen = {
        key(
            r["stage"],
            r["seed"],
            r["optimizer"],
            r["lr"],
        )
        for r in rows
    }

    if len(seen) != len(rows):
        raise ValueError(
            "Duplicate run keys in CSV."
        )

    if args.stage in (
        "tuning", "all"
    ):
        total = (
            len(TUNING_SEEDS)
            * sum(
                len(rates)
                for rates in GRIDS.values()
            )
        )

        count = sum(
            r["stage"] == "tuning"
            for r in rows
        )

        for seed in TUNING_SEEDS:
            for name, rates in GRIDS.items():
                for lr in rates:
                    k = key(
                        "tuning",
                        seed, name, lr,
                    )

                    if k in seen:
                        continue

                    row = train(
                        seed, name, lr,
                        stage="tuning",
                    )

                    append_row(path, row)

                    rows.append({
                        k: str(v)
                        for k, v in row.items()
                    })

                    seen.add(k)
                    count += 1

                    print(
                        f"[{count}/{total}] "
                        f"TUNE seed={seed} "
                        f"{name:8} "
                        f"lr={lr:.4g} "
                        f"val={row['best_val_loss']:.5f}",
                        flush=True,
                    )

        selected = select_learning_rates(
            rows
        )

        selection_path = (
            OUTPUT / "selected_learning_rates.json"
        )

        if selection_path.exists():
            previous = json.loads(
                selection_path.read_text(
                    encoding="utf-8"
                )
            )

            if previous != selected:
                raise RuntimeError(
                    "Locked learning rates differ. "
                    "Stop and investigate."
                )

        selection_path.write_text(
            json.dumps(
                selected, indent=2
            ),
            encoding="utf-8",
        )

        plot_tuning(rows)

        print(
            "\nTuning complete. "
            "Locked learning rates:",
            selected,
        )

        print(
            "Review tuning outputs before "
            "running --stage evaluation."
        )

    if args.stage in (
        "evaluation", "all"
    ):
        selection_path = (
            OUTPUT / "selected_learning_rates.json"
        )

        if not selection_path.exists():
            raise RuntimeError(
                "Run tuning first to lock "
                "hyperparameters."
            )

        selected = json.loads(
            selection_path.read_text(
                encoding="utf-8"
            )
        )

        if selected != select_learning_rates(
            rows
        ):
            raise RuntimeError(
                "Locked selection does not "
                "match tuning data."
            )

        total = (
            len(EVALUATION_SEEDS)
            * len(GRIDS)
        )

        count = sum(
            r["stage"] == "evaluation"
            for r in rows
        )

        for seed in EVALUATION_SEEDS:
            for name in GRIDS:
                lr = selected[name]

                k = key(
                    "evaluation",
                    seed, name, lr,
                )

                if k in seen:
                    continue

                row = train(
                    seed, name, lr,
                    stage="evaluation",
                )

                append_row(path, row)

                rows.append({
                    k: str(v)
                    for k, v in row.items()
                })

                seen.add(k)
                count += 1

                print(
                    f"[{count}/{total}] "
                    f"EVAL seed={seed} "
                    f"{name:8} "
                    f"acc={row['accuracy']:.4f}",
                    flush=True,
                )

        report = evaluate_summary(
            rows, selected
        )

        (
            OUTPUT / "final_report.json"
        ).write_text(
            json.dumps(
                report, indent=2
            ),
            encoding="utf-8",
        )

        plot_final(report)

        print(
            "\nFinal evaluation complete:",
            OUTPUT / "final_report.json",
        )


if __name__ == "__main__":
    main()
