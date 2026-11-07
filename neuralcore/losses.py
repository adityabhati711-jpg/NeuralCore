
"""Numerically stable loss functions built with NumPy."""

import numpy as np

from neuralcore.activations import sigmoid


def _validate_inputs(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if predictions.ndim != 2 or predictions.size == 0:
        raise ValueError("Predictions must be a nonempty 2D array.")

    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must match exactly.")

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Predictions must be finite.")

    if not np.all(np.isfinite(targets)):
        raise ValueError("Targets must be finite.")

    return predictions, targets


class MeanSquaredError:
    """Mean squared error with analytical gradient."""

    def __init__(self):
        self._difference = None

    def forward(self, predictions, targets):
        predictions, targets = _validate_inputs(
            predictions, targets
        )

        self._difference = predictions - targets
        return float(np.mean(self._difference ** 2))

    def backward(self):
        if self._difference is None:
            raise RuntimeError("Call forward() first.")

        return 2.0 * self._difference / self._difference.size


class BinaryCrossEntropyWithLogits:
    """Stable binary cross-entropy computed from raw logits."""

    def __init__(self):
        self._logits = None
        self._targets = None

    def forward(self, logits, targets):
        logits, targets = _validate_inputs(logits, targets)

        if np.any((targets < 0.0) | (targets > 1.0)):
            raise ValueError("Binary targets must be in [0, 1].")

        self._logits = logits.copy()
        self._targets = targets.copy()

        # Stable softplus formulation:
        # BCE(z,y) = log(1 + exp(z)) - y*z
        losses = np.logaddexp(0.0, logits) - targets * logits

        return float(np.mean(losses))

    def backward(self):
        if self._logits is None:
            raise RuntimeError("Call forward() first.")

        probabilities = sigmoid(self._logits)

        return (
            (probabilities - self._targets)
            / self._logits.size
        )
