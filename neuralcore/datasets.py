
"""Reproducible synthetic datasets and leakage-safe splitting."""

import numpy as np


def make_moons(n_samples=600, noise=0.12, seed=42):
    """Generate two interleaving half-moons using NumPy only."""

    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(
        n_samples, (int, np.integer)
    ) or n_samples < 4:
        raise ValueError("n_samples must be an integer >= 4.")

    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative.")

    rng = np.random.default_rng(seed)

    n_first = n_samples // 2
    n_second = n_samples - n_first

    theta_first = rng.uniform(0, np.pi, n_first)
    theta_second = rng.uniform(0, np.pi, n_second)

    first = np.column_stack([
        np.cos(theta_first),
        np.sin(theta_first),
    ])

    second = np.column_stack([
        1.0 - np.cos(theta_second),
        0.5 - np.sin(theta_second),
    ])

    x = np.vstack([first, second])
    y = np.concatenate([
        np.zeros(n_first, dtype=int),
        np.ones(n_second, dtype=int),
    ]).reshape(-1, 1)

    x += rng.normal(0.0, noise, size=x.shape)

    order = rng.permutation(n_samples)

    return x[order], y[order]


def train_val_test_split(
    x, y, train_ratio=0.7, val_ratio=0.15, seed=42
):
    """Split samples without overlap, preserving class proportions."""

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 2 or y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("Expected X=(n,features), y=(n,1).")

    if len(x) != len(y) or len(x) < 6:
        raise ValueError("Invalid dataset size.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Dataset must contain finite values.")

    if not (
        0 < train_ratio < 1
        and 0 < val_ratio < 1
        and train_ratio + val_ratio < 1
    ):
        raise ValueError("Invalid split ratios.")

    classes = np.unique(y)
    if len(classes) != 2 or not np.array_equal(
        classes, np.array([0, 1])
    ):
        raise ValueError("Expected binary labels 0 and 1.")

    rng = np.random.default_rng(seed)

    partitions = [[], [], []]

    for cls in classes:
        indices = np.flatnonzero(y[:, 0] == cls)
        indices = rng.permutation(indices)

        if len(indices) < 3:
            raise ValueError(
                "At least three samples per class are required."
            )

        n_train = int(np.floor(len(indices) * train_ratio))
        n_val = int(np.floor(len(indices) * val_ratio))

        n_test = len(indices) - n_train - n_val

        if min(n_train, n_val, n_test) < 1:
            raise ValueError(
                "Each class needs samples in every partition."
            )

        partitions[0].extend(indices[:n_train])
        partitions[1].extend(
            indices[n_train:n_train + n_val]
        )
        partitions[2].extend(
            indices[n_train + n_val:]
        )

    result = []

    for indices in partitions:
        indices = rng.permutation(indices)
        result.extend([x[indices].copy(), y[indices].copy()])

    return tuple(result)


class StandardScaler:
    """Standardize using training statistics only."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 2 or len(x) == 0:
            raise ValueError("Expected nonempty 2D training data.")

        if not np.all(np.isfinite(x)):
            raise ValueError("Training data must be finite.")

        self.mean_ = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)

        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, x):
        if self.mean_ is None:
            raise RuntimeError("Fit the scaler first.")

        x = np.asarray(x, dtype=np.float64)

        if (
            x.ndim != 2
            or x.shape[1] != self.mean_.shape[1]
            or not np.all(np.isfinite(x))
        ):
            raise ValueError("Invalid transform data.")

        return (x - self.mean_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)
