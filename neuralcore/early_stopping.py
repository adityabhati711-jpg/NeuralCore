
"""Validation-based early stopping with best-checkpoint restoration.

The monitor never examines test data. It tracks validation loss,
stores independent parameter copies, and restores the best model.

Patience counts consecutive epochs without sufficient improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from neuralcore.layers import Dense


@dataclass(frozen=True)
class EarlyStoppingResult:
    best_epoch: int
    best_validation_loss: float
    epochs_observed: int
    stopped_early: bool


class EarlyStopping:
    """Track validation loss and restore the best parameters."""

    def __init__(self, patience=50, min_delta=0.0):
        if (
            isinstance(patience, (bool, np.bool_))
            or not isinstance(patience, (int, np.integer))
            or patience < 1
        ):
            raise ValueError("patience must be a positive integer.")

        if (
            isinstance(min_delta, (bool, np.bool_))
            or not isinstance(
                min_delta, (int, float, np.integer, np.floating)
            )
            or not math.isfinite(float(min_delta))
            or min_delta < 0
        ):
            raise ValueError(
                "min_delta must be finite and nonnegative."
            )

        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.best_epoch = 0
        self.best_validation_loss = math.inf
        self.epochs_observed = 0
        self.wait = 0
        self.should_stop = False

        self._checkpoint = None
        self._parameter_shapes = None

    @staticmethod
    def _parameters(model):
        if not hasattr(model, "layers"):
            raise TypeError("Expected a Sequential-like model.")

        parameters = []

        for layer in model.layers:
            if isinstance(layer, Dense):
                parameters.extend(
                    (layer.weights, layer.biases)
                )

        if not parameters:
            raise ValueError("Model has no Dense parameters.")

        return parameters

    def update(self, model, validation_loss, epoch):
        """Observe a validation loss after an optimizer update.

        Returns True when patience is exhausted.
        """

        if self.should_stop:
            raise RuntimeError(
                "Early stopping has already been triggered."
            )

        if (
            isinstance(epoch, (bool, np.bool_))
            or not isinstance(epoch, (int, np.integer))
            or epoch != self.epochs_observed + 1
        ):
            raise ValueError(
                "Epochs must be consecutive, starting at 1."
            )

        if (
            isinstance(validation_loss, (bool, np.bool_))
            or not isinstance(
                validation_loss,
                (int, float, np.integer, np.floating),
            )
            or not math.isfinite(float(validation_loss))
        ):
            raise ValueError(
                "Validation loss must be a finite number."
            )

        parameters = self._parameters(model)

        shapes = tuple(p.shape for p in parameters)

        if (
            self._parameter_shapes is not None
            and shapes != self._parameter_shapes
        ):
            raise ValueError(
                "Model parameter shapes changed during training."
            )

        for parameter in parameters:
            if not np.all(np.isfinite(parameter)):
                raise ValueError(
                    "Model parameters must be finite."
                )

        loss = float(validation_loss)

        improved = (
            self.best_epoch == 0
            or loss < self.best_validation_loss - self.min_delta
        )

        if improved:
            self.best_epoch = int(epoch)
            self.best_validation_loss = loss
            self.wait = 0
            self._checkpoint = [
                parameter.copy()
                for parameter in parameters
            ]
        else:
            self.wait += 1

        self._parameter_shapes = shapes
        self.epochs_observed = int(epoch)

        if self.wait >= self.patience:
            self.should_stop = True

        return self.should_stop

    def restore_best(self, model):
        """Restore the best observed validation checkpoint."""

        if self._checkpoint is None:
            raise RuntimeError(
                "No checkpoint available. Call update() first."
            )

        parameters = self._parameters(model)

        if len(parameters) != len(self._checkpoint):
            raise ValueError("Parameter count mismatch.")

        for parameter, saved in zip(
            parameters, self._checkpoint
        ):
            if parameter.shape != saved.shape:
                raise ValueError(
                    "Checkpoint parameter shape mismatch."
                )

        for parameter, saved in zip(
            parameters, self._checkpoint
        ):
            parameter[...] = saved

        return self.best_epoch

    def result(self):
        if self.best_epoch == 0:
            raise RuntimeError(
                "No validation observations available."
            )

        return EarlyStoppingResult(
            best_epoch=self.best_epoch,
            best_validation_loss=self.best_validation_loss,
            epochs_observed=self.epochs_observed,
            stopped_early=self.should_stop,
        )
